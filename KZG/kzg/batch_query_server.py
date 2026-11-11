from __future__ import annotations

import atexit
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from kzg.batch_kzg import (
    AUTO_EDGE_BATCH_SIZE,
    GEOHASH_BATCH_PROOF_MODE,
    aggregate_polynomials,
    aggregate_values,
    batch_weights,
    geohash_batch_challenge,
    partition_edge_ids,
    resolve_edge_batch_size,
)
from kzg.canonical import cell_value
from kzg.cell_merkle import SparseMerkleTree, query_cell_payloads
from kzg.field import FR_MODULUS, fr_to_bytes
from kzg.geohash_index import CellRect, GeohashCellIndex
from kzg.kzg import KZG, SRS, g1_to_bytes
from kzg.ntt import domain_points, evaluations_to_coefficients
from kzg.sparse_slots import SPARSE_SLOT_ENCODING
from kzg.storage import DailyStore
from kzg.time_slots import complete_slots_for_interval, day_start_timestamp, partition_slots
from kzg.trajectory_id import format_trajectory_id
from kzg.vector import base_evaluations


DEFAULT_BATCH_PROOF_WORKERS = max(1, min(8, os.cpu_count() or 4))
SMALL_QUERY_PROOF_WORKERS = max(1, min(4, os.cpu_count() or 4))
LARGE_QUERY_WORK_UNITS = 10_000
PARALLEL_BATCH_THRESHOLD = 2
_BATCH_PROOF_EXECUTOR = None
_BATCH_PROOF_EXECUTOR_KEY = None
_BATCH_WORKER_KZG = None
_BATCH_WORKER_DOMAIN_POINTS = None
_BATCH_WORKER_BASE_EVALUATIONS = None

DENSE_POLYNOMIAL_STORAGE = "dense-coefficients-v1"
SPARSE_POLYNOMIAL_STORAGE = "sparse-evaluations-v1"


def _require_equal(name: str, left, right) -> None:
    if left != right:
        raise ValueError(f"{name} mismatch: {left!r} != {right!r}")


def _initialize_batch_proof_worker(srs_path: str, day: str) -> None:
    global _BATCH_WORKER_KZG, _BATCH_WORKER_DOMAIN_POINTS, _BATCH_WORKER_BASE_EVALUATIONS
    worker_srs = SRS.load(srs_path)
    _BATCH_WORKER_KZG = KZG(worker_srs)
    _BATCH_WORKER_DOMAIN_POINTS = domain_points(worker_srs.domain_size)
    _BATCH_WORKER_BASE_EVALUATIONS = base_evaluations(day, worker_srs.domain_size)


def _batch_proof_worker_ready(value: int) -> int:
    return value


def _aggregate_sparse_coefficients(
    base_values: list[int],
    sparse_values_by_edge: list[dict[int, int]],
    weights: list[int],
) -> list[int]:
    """在线性组合后的求值向量上只做一次逆 NTT。

    这与先为每条道路做逆 NTT、再聚合系数完全等价，因为逆 NTT 是
    线性变换。数据库只需保存偏离公共空槽向量的非空槽值。
    """
    if not sparse_values_by_edge:
        raise ValueError("at least one sparse edge is required")
    if len(sparse_values_by_edge) != len(weights):
        raise ValueError("sparse edges and weights must have the same length")

    normalized_weights = [int(weight) % FR_MODULUS for weight in weights]
    weight_sum = sum(normalized_weights) % FR_MODULUS
    aggregate_evaluations = [
        int(base_value) * weight_sum % FR_MODULUS for base_value in base_values
    ]
    for slot_values, weight in zip(sparse_values_by_edge, normalized_weights):
        for slot, value in slot_values.items():
            aggregate_evaluations[slot] = (
                aggregate_evaluations[slot]
                + weight * (int(value) - int(base_values[slot]))
            ) % FR_MODULUS
    return evaluations_to_coefficients(aggregate_evaluations)


def _aggregate_sparse_claimed_values(
    base_values: list[int],
    sparse_values_by_edge: list[dict[int, int]],
    weights: list[int],
    slots: list[int],
) -> list[int]:
    """Aggregate only the queried evaluations without materializing an E x T matrix."""
    normalized_weights = [int(weight) % FR_MODULUS for weight in weights]
    weight_sum = sum(normalized_weights) % FR_MODULUS
    result = [int(base_values[slot]) * weight_sum % FR_MODULUS for slot in slots]
    for slot_values,weight in zip(sparse_values_by_edge,normalized_weights):
        for result_index,slot in enumerate(slots):
            value = slot_values.get(slot)
            if value is not None:
                result[result_index] = (
                    result[result_index]
                    + weight * (int(value) - int(base_values[slot]))
                ) % FR_MODULUS
    return result


def _compute_proof_opening(
    kzg: KZG,
    domain: list[int],
    base_values: list[int],
    polynomial_storage: str,
    polynomial_inputs: list,
    task,
):
    batch_index,edge_batch_index,weights,slots,values_by_edge = task
    aggregate_started = time.perf_counter()
    if polynomial_storage == SPARSE_POLYNOMIAL_STORAGE:
        aggregate_coefficients = _aggregate_sparse_coefficients(
            base_values,polynomial_inputs,weights
        )
        aggregate_claimed_values = _aggregate_sparse_claimed_values(
            base_values,polynomial_inputs,weights,slots
        )
    else:
        aggregate_coefficients = aggregate_polynomials(polynomial_inputs,weights)
        aggregate_claimed_values = aggregate_values(values_by_edge,weights)
    aggregate_seconds = time.perf_counter() - aggregate_started

    opening_started = time.perf_counter()
    points = [domain[slot] for slot in slots]
    opening = kzg.open_multi_fast(
        aggregate_coefficients,points,aggregate_claimed_values
    )
    opening_seconds = time.perf_counter() - opening_started
    return (
        batch_index,
        edge_batch_index,
        g1_to_bytes(opening.proof),
        aggregate_seconds,
        opening_seconds,
    )


def _generate_proof_bundle(bundle):
    """Process several time chunks after receiving one spatial batch once."""
    if (
        _BATCH_WORKER_KZG is None
        or _BATCH_WORKER_DOMAIN_POINTS is None
        or _BATCH_WORKER_BASE_EVALUATIONS is None
    ):
        raise RuntimeError("batch proof worker is not initialized")
    polynomial_storage,polynomial_inputs,tasks = bundle
    return [
        _compute_proof_opening(
            _BATCH_WORKER_KZG,
            _BATCH_WORKER_DOMAIN_POINTS,
            _BATCH_WORKER_BASE_EVALUATIONS,
            polynomial_storage,
            polynomial_inputs,
            task,
        )
        for task in tasks
    ]


def _generate_proof_bundle_local(
    kzg: KZG,domain: list[int],base_values: list[int],bundle
):
    polynomial_storage,polynomial_inputs,tasks = bundle
    return [
        _compute_proof_opening(
            kzg,
            domain,
            base_values,
            polynomial_storage,
            polynomial_inputs,
            task,
        )
        for task in tasks
    ]


def _time_shards_by_edge_group(
    edge_group_count: int,time_chunk_count: int,requested_workers: int
) -> tuple[int,list[int]]:
    """Allocate workers to mandatory proofs, preferring the time dimension.

    Every edge group gets at least one bundle.  Extra workers split its already
    required time chunks, so parallelism does not increase proof count.
    """
    group_count = int(edge_group_count)
    chunk_count = int(time_chunk_count)
    worker_limit = max(1,int(requested_workers))
    if group_count <= 0 or chunk_count <= 0:
        return 1,[]
    proof_count = group_count * chunk_count
    worker_count = min(worker_limit,proof_count)
    if worker_count < group_count:
        return worker_count,[1] * group_count
    base_shards,remainder = divmod(worker_count,group_count)
    shards = [
        min(chunk_count,base_shards + (1 if index < remainder else 0))
        for index in range(group_count)
    ]
    return min(worker_count,sum(shards)),shards


def _automatic_proof_worker_limit(edge_count: int,time_chunk_count: int) -> int:
    """Use extra processes only when one spatial batch has enough work to repay IPC."""
    work_units = max(0,int(edge_count)) * max(0,int(time_chunk_count))
    if work_units >= LARGE_QUERY_WORK_UNITS:
        return DEFAULT_BATCH_PROOF_WORKERS
    return SMALL_QUERY_PROOF_WORKERS


def _get_batch_proof_executor(
    srs_path: str | Path, worker_count: int, day: str
) -> ProcessPoolExecutor:
    global _BATCH_PROOF_EXECUTOR, _BATCH_PROOF_EXECUTOR_KEY
    resolved_srs_path = str(Path(srs_path).resolve())
    executor_key = (resolved_srs_path, int(worker_count), str(day))
    if _BATCH_PROOF_EXECUTOR is not None and _BATCH_PROOF_EXECUTOR_KEY == executor_key:
        return _BATCH_PROOF_EXECUTOR
    if _BATCH_PROOF_EXECUTOR is not None:
        _BATCH_PROOF_EXECUTOR.shutdown(wait=True, cancel_futures=True)
    _BATCH_PROOF_EXECUTOR = ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_batch_proof_worker,
        initargs=(resolved_srs_path, str(day)),
    )
    _BATCH_PROOF_EXECUTOR_KEY = executor_key
    list(_BATCH_PROOF_EXECUTOR.map(_batch_proof_worker_ready, range(worker_count), chunksize=1))
    return _BATCH_PROOF_EXECUTOR


def _shutdown_batch_proof_executor() -> None:
    global _BATCH_PROOF_EXECUTOR, _BATCH_PROOF_EXECUTOR_KEY
    if _BATCH_PROOF_EXECUTOR is not None:
        _BATCH_PROOF_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        _BATCH_PROOF_EXECUTOR = None
        _BATCH_PROOF_EXECUTOR_KEY = None


atexit.register(_shutdown_batch_proof_executor)


@dataclass
class QueryContext:
    """Long-lived server state.

    Disk/SRS/index loading happens once in ``prepare_query_context``.  Query-specific
    rows can then be prefetched with ``preload_query_data``.  The timed online path
    ``generate_batch_query_package_from_memory`` performs no SQLite/File I/O.
    """

    cell_index: GeohashCellIndex
    srs: SRS
    kzg: KZG
    anchor: dict
    day: str
    period_start_ts: int
    period_end_ts: int
    slot_seconds: int
    slot_count: int
    tree: SparseMerkleTree
    daily_leaf_hashes: dict[str, bytes]
    commitment_by_edge: dict[int, bytes]
    query_domain_points: list[int]
    base_evaluation_values: list[int]
    daily_root_bytes: bytes
    srs_fingerprint_bytes: bytes
    srs_path: str
    store: DailyStore
    polynomial_storage: str = DENSE_POLYNOMIAL_STORAGE
    polynomial_cache: dict[int, list[int]] = field(default_factory=dict)
    sparse_slot_value_cache: dict[int, dict[int, int]] = field(default_factory=dict)
    slot_records_by_edge_cache: dict[int, dict[int, tuple[list, int]]] = field(
        default_factory=dict
    )
    preloaded_slot_queries: set[tuple[tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=set
    )
    proof_worker_count: int = DEFAULT_BATCH_PROOF_WORKERS

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "QueryContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def prepare_query_context(
    cell_index_dir: str | Path,
    database_path: str | Path,
    anchor_path: str | Path,
    srs_path: str | Path,
) -> QueryContext:
    """Load persistent server state once.  This initialization is not query time."""
    cell_index = GeohashCellIndex.load(cell_index_dir)
    srs = SRS.load(srs_path)
    kzg = KZG(srs)
    anchor = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    actual_srs_fingerprint = SRS.fingerprint(srs_path)
    store = DailyStore(database_path, readonly=True)

    try:
        day = store.get_metadata("date")
        _require_equal("date", day, anchor["date"])
        _require_equal("daily root", store.get_metadata("daily_root"), anchor["daily_root"])
        _require_equal("SRS fingerprint", store.get_metadata("srs_fingerprint"), anchor["srs_fingerprint"])
        _require_equal("actual SRS fingerprint", actual_srs_fingerprint, anchor["srs_fingerprint"])
        _require_equal("domain size", store.get_metadata("domain_size"), srs.domain_size)
        _require_equal("maximum multi points", store.get_metadata("max_multi_points"), srs.max_multi_points)
        _require_equal("edge count", store.get_metadata("edge_count"), cell_index.edge_count)
        _require_equal("cell bits", store.get_metadata("cell_bits"), cell_index.bits)
        _require_equal(
            "static cell index fingerprint",
            store.get_metadata("static_cell_index_fingerprint"),
            cell_index.fingerprint(),
        )

        default_start_ts = day_start_timestamp(day)
        slot_seconds = int(store.get_metadata_or("slot_seconds", 60))
        slot_count = int(store.get_metadata_or("slot_count", 1440))
        period_start_ts = int(store.get_metadata_or("period_start_ts", default_start_ts))
        period_end_ts = int(
            store.get_metadata_or(
                "period_end_ts", period_start_ts + slot_seconds * slot_count
            )
        )
        if slot_seconds <= 0 or slot_count <= 0 or slot_count > srs.domain_size:
            raise ValueError("invalid authenticated time-slot configuration")
        if period_end_ts != period_start_ts + slot_seconds * slot_count:
            raise ValueError("period length does not match slot configuration")
        for key,value in (
            ("slot_seconds",slot_seconds),
            ("slot_count",slot_count),
            ("period_start_ts",period_start_ts),
            ("period_end_ts",period_end_ts),
        ):
            _require_equal(key,anchor.get(key,value),value)

        daily_leaf_hashes = store.all_daily_cell_leaves()
        tree = SparseMerkleTree(cell_index.bits, daily_leaf_hashes)
        _require_equal("reconstructed daily root", tree.root_hash.hex(), anchor["daily_root"])

        commitment_items = store.all_edge_commitments()
        commitment_by_edge = {edge_id: commitment for edge_id, commitment in commitment_items}
        _require_equal("commitment count", len(commitment_by_edge), cell_index.edge_count)

        polynomial_storage = store.get_metadata_or(
            "polynomial_storage", DENSE_POLYNOMIAL_STORAGE
        )
        if polynomial_storage not in {
            DENSE_POLYNOMIAL_STORAGE,
            SPARSE_POLYNOMIAL_STORAGE,
        }:
            raise ValueError(f"unsupported polynomial storage: {polynomial_storage!r}")

        return QueryContext(
            cell_index=cell_index,
            srs=srs,
            kzg=kzg,
            anchor=anchor,
            day=day,
            period_start_ts=period_start_ts,
            period_end_ts=period_end_ts,
            slot_seconds=slot_seconds,
            slot_count=slot_count,
            tree=tree,
            daily_leaf_hashes=daily_leaf_hashes,
            commitment_by_edge=commitment_by_edge,
            query_domain_points=domain_points(srs.domain_size),
            base_evaluation_values=base_evaluations(day, srs.domain_size, slot_count),
            daily_root_bytes=bytes.fromhex(anchor["daily_root"]),
            srs_fingerprint_bytes=bytes.fromhex(anchor["srs_fingerprint"]),
            srs_path=str(Path(srs_path).resolve()),
            store=store,
            polynomial_storage=polynomial_storage,
        )
    except Exception:
        store.close()
        raise


def _query_shape(
    context: QueryContext,
    cell_rect: CellRect,
    start_ts: int,
    end_ts: int,
    edge_batch_size: int,
) -> tuple[CellRect, list[str], list[int], list[list[int]], list[int], list[list[int]], int]:
    normalized_rect = cell_rect.normalized()
    queried_cell_ids = context.cell_index.cell_ids_for_rect(normalized_rect)
    spatial_edge_ids = context.cell_index.edge_ids_for_rect(normalized_rect)
    effective_edge_batch_size = resolve_edge_batch_size(
        len(spatial_edge_ids),edge_batch_size
    )
    edge_groups = partition_edge_ids(spatial_edge_ids,effective_edge_batch_size)
    query_slots = complete_slots_for_interval(
        context.period_start_ts,
        int(start_ts),
        int(end_ts),
        context.slot_seconds,
        context.slot_count,
    )
    query_slot_chunks = partition_slots(query_slots, context.srs.max_multi_points)
    return (
        normalized_rect,
        queried_cell_ids,
        spatial_edge_ids,
        edge_groups,
        query_slots,
        query_slot_chunks,
        effective_edge_batch_size,
    )


def preload_query_data(
    context: QueryContext,
    cell_rect: CellRect,
    start_ts: int,
    end_ts: int,
    edge_batch_size: int = AUTO_EDGE_BATCH_SIZE,
    proof_workers: int | None = None,
) -> dict:
    """Populate query-relevant SQLite data and proof workers before timing.

    This function intentionally performs the storage I/O excluded by the user's
    in-memory benchmark model.  After it returns, the corresponding online query
    uses only in-memory dictionaries plus cryptographic computation.
    """
    started = time.perf_counter()
    (
        normalized_rect,
        queried_cell_ids,
        spatial_edge_ids,
        edge_groups,
        query_slots,
        query_slot_chunks,
        effective_edge_batch_size,
    ) = _query_shape(context, cell_rect, start_ts, end_ts, edge_batch_size)

    if context.polynomial_storage == SPARSE_POLYNOMIAL_STORAGE:
        missing_sparse_edges = [
            edge_id
            for edge_id in spatial_edge_ids
            if edge_id not in context.sparse_slot_value_cache
        ]
        if missing_sparse_edges:
            context.sparse_slot_value_cache.update(
                context.store.get_edge_slot_values(missing_sparse_edges)
            )
    else:
        missing_polynomial_edges = [
            edge_id
            for edge_id in spatial_edge_ids
            if edge_id not in context.polynomial_cache
        ]
        if missing_polynomial_edges:
            context.polynomial_cache.update(
                context.store.get_edge_polynomials(missing_polynomial_edges)
            )

    slot_query_key = (tuple(spatial_edge_ids),tuple(query_slots))
    if slot_query_key not in context.preloaded_slot_queries:
        loaded_slots = context.store.get_slots(spatial_edge_ids, query_slots)
        for (edge_id,slot),(records,value) in loaded_slots.items():
            if cell_value(context.day,edge_id,slot,records) != value:
                raise ValueError(f"stored value mismatch for edge {edge_id}, slot {slot}")
            context.slot_records_by_edge_cache.setdefault(edge_id,{})[slot] = (
                records,value
            )
        context.preloaded_slot_queries.add(slot_query_key)

    expected_proof_task_count = len(edge_groups) * len(query_slot_chunks)
    requested_workers = (
        _automatic_proof_worker_limit(
            len(spatial_edge_ids),len(query_slot_chunks)
        )
        if proof_workers is None
        else max(1,int(proof_workers))
    )
    # Each mandatory KZG proof (edge group x time chunk) is independent.  Time
    # chunks are the preferred parallel dimension because parallel execution
    # does not create any additional proofs.
    worker_count,_time_shards = _time_shards_by_edge_group(
        len(edge_groups),len(query_slot_chunks),requested_workers
    )
    context.proof_worker_count = worker_count
    worker_initialization_seconds = 0.0
    if worker_count >= PARALLEL_BATCH_THRESHOLD:
        worker_started = time.perf_counter()
        _get_batch_proof_executor(context.srs_path, worker_count, context.day)
        worker_initialization_seconds = time.perf_counter() - worker_started

    return {
        "preload_seconds": time.perf_counter() - started,
        "worker_initialization_seconds": worker_initialization_seconds,
        "worker_count": max(1, worker_count),
        "queried_cell_count": len(queried_cell_ids),
        "queried_edge_count": len(spatial_edge_ids),
        "query_slot_count": len(query_slots),
        "batch_count": expected_proof_task_count,
        "edge_batch_count": len(edge_groups),
        "time_chunk_count": len(query_slot_chunks),
        "time_shards_by_edge_batch": _time_shards,
        "edge_batch_size": effective_edge_batch_size,
        "cell_rect": normalized_rect.to_dict(),
    }


def generate_batch_query_package_from_memory(
    context: QueryContext,
    cell_rect: CellRect,
    start_ts: int,
    end_ts: int,
    edge_batch_size: int = AUTO_EDGE_BATCH_SIZE,
) -> dict:
    """Generate a proof package with all persistent/query data already in memory."""
    online_started = time.perf_counter()

    localization_started = time.perf_counter()
    (
        normalized_rect,
        queried_cell_ids,
        spatial_edge_ids,
        edge_groups,
        query_slots,
        query_slot_chunks,
        effective_edge_batch_size,
    ) = _query_shape(context, cell_rect, start_ts, end_ts, edge_batch_size)
    cell_rect_values = [
        normalized_rect.lon_index_min,
        normalized_rect.lon_index_max,
        normalized_rect.lat_index_min,
        normalized_rect.lat_index_max,
    ]
    cell_localization_seconds = time.perf_counter() - localization_started

    if context.polynomial_storage == SPARSE_POLYNOMIAL_STORAGE:
        missing_sparse_edges = [
            edge_id
            for edge_id in spatial_edge_ids
            if edge_id not in context.sparse_slot_value_cache
        ]
        if missing_sparse_edges:
            raise RuntimeError(
                f"query data is not preloaded: missing sparse values for edge {missing_sparse_edges[0]}; "
                "call preload_query_data() before the timed query"
            )
    else:
        missing_polynomials = [
            edge_id
            for edge_id in spatial_edge_ids
            if edge_id not in context.polynomial_cache
        ]
        if missing_polynomials:
            raise RuntimeError(
                f"query data is not preloaded: missing polynomial for edge {missing_polynomials[0]}; "
                "call preload_query_data() before the timed query"
            )
    slot_query_key = (tuple(spatial_edge_ids),tuple(query_slots))
    if slot_query_key not in context.preloaded_slot_queries:
        raise RuntimeError(
            "query slot records are not preloaded; "
            "call preload_query_data() before the timed query"
        )

    spatial_proof_started = time.perf_counter()
    cell_packages, queried_leaf_hashes = query_cell_payloads(
        context.day, context.cell_index, queried_cell_ids, context.commitment_by_edge
    )
    for cell_id, leaf_hash in queried_leaf_hashes.items():
        stored_leaf = context.daily_leaf_hashes.get(
            cell_id, context.tree.default_hash_for_prefix(context.cell_index.bits)
        )
        if leaf_hash != stored_leaf:
            raise ValueError(f"daily cell payload mismatch for {cell_id}")
    spatial_multiproof = context.tree.multiproof(queried_cell_ids).to_dict()
    spatial_proof_generation_seconds = time.perf_counter() - spatial_proof_started

    trajectory_started = time.perf_counter()
    result_trajectory_ids: set[int] = set()
    prepared_by_edge: dict[int, dict] = {}
    edge_packages: list[dict] = []
    query_slot_set = set(query_slots)

    for edge_id in spatial_edge_ids:
        if edge_id not in context.commitment_by_edge:
            raise ValueError(f"queried edge {edge_id} has no KZG commitment")
        cell_id = context.cell_index.edge_to_cell[edge_id]
        records_by_slot = []
        query_slot_values: dict[int,int] = {}
        cached_edge_slots = context.slot_records_by_edge_cache.get(edge_id,{})
        for slot in sorted(cached_edge_slots):
            if slot not in query_slot_set:
                continue
            records,value = cached_edge_slots[slot]
            # The query is an exact union of complete authenticated slots.
            # Every record in a selected slot belongs to the result; applying
            # a finer timestamp filter would reintroduce partial-slot queries.
            result_trajectory_ids.update(record.traj_id for record in records)
            records_by_slot.append({
                "slot": slot,
                "records": [record.as_list() for record in records],
                "value": fr_to_bytes(value).hex(),
            })
            query_slot_values[slot] = value

        prepared_edge = {
            "edge_id": edge_id,
            "cell_id": cell_id,
            "commitment": context.commitment_by_edge[edge_id],
            "query_slot_values": query_slot_values,
        }
        if context.polynomial_storage == SPARSE_POLYNOMIAL_STORAGE:
            prepared_edge["sparse_slot_values"] = context.sparse_slot_value_cache[edge_id]
        else:
            prepared_edge["coefficients"] = context.polynomial_cache[edge_id]
        prepared_by_edge[edge_id] = prepared_edge
        edge_packages.append({
            "edge_id": edge_id,
            "cell_id": cell_id,
            "slots": records_by_slot,
        })
    trajectory_processing_seconds = time.perf_counter() - trajectory_started

    kzg_started = time.perf_counter()
    preparation_started = time.perf_counter()
    batch_packages: list[dict] = []
    polynomial_inputs_by_edge_batch: list[list] = []
    for edge_ids in edge_groups:
        if context.polynomial_storage == SPARSE_POLYNOMIAL_STORAGE:
            polynomial_inputs = [
                prepared_by_edge[edge_id]["sparse_slot_values"] for edge_id in edge_ids
            ]
        else:
            polynomial_inputs = [
                prepared_by_edge[edge_id]["coefficients"] for edge_id in edge_ids
            ]
        polynomial_inputs_by_edge_batch.append(polynomial_inputs)

    proof_tasks_by_edge_batch: list[list[tuple]] = [
        [] for _ in edge_groups
    ]
    batch_index = 0
    # Challenge/transcript construction stays in the main process. Each
    # mandatory edge-group x time-chunk proof becomes one independent task.
    for time_chunk_index, slot_chunk in enumerate(query_slot_chunks):
        for edge_batch_index, edge_ids in enumerate(edge_groups):
            batch_edges = [prepared_by_edge[edge_id] for edge_id in edge_ids]
            commitments = [item["commitment"] for item in batch_edges]
            cell_prefixes = [item["cell_id"] for item in batch_edges]
            values_by_edge = [
                [
                    item["query_slot_values"].get(
                        slot,context.base_evaluation_values[slot]
                    )
                    for slot in slot_chunk
                ]
                for item in batch_edges
            ]
            challenge = geohash_batch_challenge(
                context.day,
                context.daily_root_bytes,
                context.srs_fingerprint_bytes,
                context.srs.domain_size,
                cell_rect_values,
                int(start_ts),
                int(end_ts),
                batch_index,
                slot_chunk,
                edge_ids,
                cell_prefixes,
                commitments,
                values_by_edge,
            )
            weights = batch_weights(challenge, len(edge_ids))
            proof_tasks_by_edge_batch[edge_batch_index].append(
                (
                    batch_index,
                    edge_batch_index,
                    weights,
                    slot_chunk,
                    None
                    if context.polynomial_storage == SPARSE_POLYNOMIAL_STORAGE
                    else values_by_edge,
                )
            )
            batch_packages.append({
                "batch_index": batch_index,
                "time_chunk_index": time_chunk_index,
                "edge_batch_index": edge_batch_index,
                "slots": slot_chunk,
                "edge_ids": edge_ids,
                "cell_prefixes": cell_prefixes,
                "challenge": fr_to_bytes(challenge).hex(),
            })
            batch_index += 1
    task_preparation_seconds = time.perf_counter() - preparation_started

    expected_proof_task_count = len(edge_groups) * len(query_slot_chunks)
    actual_proof_task_count = sum(len(tasks) for tasks in proof_tasks_by_edge_batch)
    if actual_proof_task_count != expected_proof_task_count:
        raise RuntimeError("batch proof task count mismatch")

    worker_count,time_shards_by_edge_batch = _time_shards_by_edge_group(
        len(edge_groups),len(query_slot_chunks),context.proof_worker_count
    )
    proof_bundles = []
    for edge_batch_index,tasks in enumerate(proof_tasks_by_edge_batch):
        shard_count = time_shards_by_edge_batch[edge_batch_index]
        for shard_index in range(shard_count):
            shard_tasks = tasks[shard_index::shard_count]
            if shard_tasks:
                proof_bundles.append(
                    (
                        context.polynomial_storage,
                        polynomial_inputs_by_edge_batch[edge_batch_index],
                        shard_tasks,
                    )
                )
    parallel_started = time.perf_counter()
    if proof_bundles:
        if worker_count >= PARALLEL_BATCH_THRESHOLD:
            proof_executor = _get_batch_proof_executor(
                context.srs_path, worker_count, context.day
            )
            proof_result_bundles = list(
                proof_executor.map(_generate_proof_bundle,proof_bundles,chunksize=1)
            )
        else:
            proof_result_bundles = [
                _generate_proof_bundle_local(
                    context.kzg,
                    context.query_domain_points,
                    context.base_evaluation_values,
                    bundle,
                )
                for bundle in proof_bundles
            ]
        proof_results = [
            result for result_bundle in proof_result_bundles for result in result_bundle
        ]
    else:
        proof_results = []
    parallel_wall_seconds = time.perf_counter() - parallel_started

    proof_by_batch_index = {}
    worker_aggregation_seconds_sum = 0.0
    worker_opening_seconds_sum = 0.0
    for (
        proof_batch_index,
        _edge_batch_index,
        proof_bytes,
        aggregate_seconds,
        opening_seconds,
    ) in proof_results:
        proof_by_batch_index[proof_batch_index] = proof_bytes
        worker_aggregation_seconds_sum += aggregate_seconds
        worker_opening_seconds_sum += opening_seconds

    if len(proof_by_batch_index) != expected_proof_task_count:
        raise RuntimeError("batch proof result count mismatch")
    for batch_package in batch_packages:
        batch_package["proof"] = proof_by_batch_index[batch_package["batch_index"]].hex()

    kzg_proof_generation_seconds = time.perf_counter() - kzg_started
    sorted_ids = sorted(result_trajectory_ids)
    online_query_seconds = time.perf_counter() - online_started
    return {
        "scheme_version": context.anchor["scheme_version"],
        "proof_mode": GEOHASH_BATCH_PROOF_MODE,
        "slot_encoding": SPARSE_SLOT_ENCODING,
        "edge_batch_size": int(effective_edge_batch_size),
        "date": context.day,
        "timezone": "UTC",
        "timestamp_semantics": "unix-seconds-utc",
        "period_start_ts": context.period_start_ts,
        "period_end_ts": context.period_end_ts,
        "slot_seconds": context.slot_seconds,
        "slot_count": context.slot_count,
        "trajectory_id_scheme": context.anchor["trajectory_id_scheme"],
        "domain_size": context.srs.domain_size,
        "max_multi_points": context.srs.max_multi_points,
        "daily_root": context.anchor["daily_root"],
        "daily_root_type": context.anchor["daily_root_type"],
        "srs_fingerprint": context.anchor["srs_fingerprint"],
        "query": {
            "cell_rect": normalized_rect.to_dict(),
            "covered_geo_rectangle": context.cell_index.covered_geo_rectangle(normalized_rect),
            "cell_ids": queried_cell_ids,
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "slots": query_slots,
            "slot_chunks": query_slot_chunks,
        },
        "cells": cell_packages,
        "spatial_multiproof": spatial_multiproof,
        "result_trajectory_ids": sorted_ids,
        "result_trajectory_labels": [
            format_trajectory_id(value, context.anchor["trajectory_id_scheme"])
            for value in sorted_ids
        ],
        "edges": edge_packages,
        "returned_nonempty_slot_count": sum(len(item["slots"]) for item in edge_packages),
        "batches": batch_packages,
        # New benchmark fields: no File/SQLite I/O is included.
        "online_query_seconds": online_query_seconds,
        "cell_localization_seconds": cell_localization_seconds,
        "spatial_proof_generation_seconds": spatial_proof_generation_seconds,
        "trajectory_processing_seconds": trajectory_processing_seconds,
        "kzg_proof_generation_seconds": kzg_proof_generation_seconds,
        "kzg_task_preparation_seconds": task_preparation_seconds,
        "kzg_parallel_wall_seconds": parallel_wall_seconds,
        "kzg_worker_aggregation_seconds_sum": worker_aggregation_seconds_sum,
        "kzg_worker_opening_seconds_sum": worker_opening_seconds_sum,
        "kzg_aggregation_parallel": worker_count >= PARALLEL_BATCH_THRESHOLD,
        "kzg_fast_quotient_opening": True,
        "proof_worker_initialization_seconds": 0.0,
        "proof_worker_count": max(1, worker_count),
        "edge_batch_count": len(edge_groups),
        "time_chunk_count": len(query_slot_chunks),
        "proof_bundle_count": len(proof_bundles),
        # Backward-compatible alias used by older scripts.
        "server_proof_generation_seconds": online_query_seconds,
    }


def generate_batch_query_package(
    cell_index_dir: str | Path,
    database_path: str | Path,
    anchor_path: str | Path,
    srs_path: str | Path,
    cell_rect: CellRect,
    start_ts: int,
    end_ts: int,
    edge_batch_size: int = AUTO_EDGE_BATCH_SIZE,
) -> dict:
    """Backward-compatible one-shot API.

    Initialization/preload still happen, but the timing fields inside the returned
    package measure only the in-memory online computation.
    """
    with prepare_query_context(cell_index_dir, database_path, anchor_path, srs_path) as context:
        preload = preload_query_data(context, cell_rect, start_ts, end_ts, edge_batch_size)
        package = generate_batch_query_package_from_memory(
            context, cell_rect, start_ts, end_ts, edge_batch_size
        )
        package["data_preload_seconds"] = preload["preload_seconds"]
        package["proof_worker_initialization_seconds"] = preload["worker_initialization_seconds"]
        return package
