from __future__ import annotations

import atexit
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass,field
from pathlib import Path

from kzg.batch_kzg import (
    GEOHASH_BATCH_PROOF_MODE,
    aggregate_commitments,
    batch_weights,
    geohash_batch_challenge_packed,
    partition_edge_ids,
)
from kzg.cell_merkle import SparseMerkleTree, daily_cell_leaf_hash
from kzg.config import DEFAULT_VERIFICATION_WORKERS, SLOT_SECONDS, SLOTS_PER_DAY
from kzg.field import FR_MODULUS,fr_from_bytes
from kzg.geohash_index import CellEdgeRecord, CellRect, GeohashGrid
from kzg.kzg import KZG, SRS, g1_from_bytes, prepare_g1_points
from kzg.ntt import domain_points
from kzg.sparse_slots import (
    SPARSE_SLOT_ENCODING,
    prepare_sparse_slot_reconstruction,
    reconstruct_sparse_slots_prepared,
)
from kzg.time_slots import complete_slots_for_interval, day_start_timestamp, partition_slots
from kzg.trajectory_id import format_trajectory_id
from kzg.vector import base_evaluations


_VERIFICATION_EXECUTOR = None
_VERIFICATION_EXECUTOR_KEY = None
_VERIFICATION_WORKER_KZG = None
_VERIFICATION_WORKER_COMMITMENT_CACHE = {}


def _initialize_verification_worker(srs_path: str) -> None:
    global _VERIFICATION_WORKER_KZG,_VERIFICATION_WORKER_COMMITMENT_CACHE
    _VERIFICATION_WORKER_KZG = KZG(SRS.load(srs_path))
    _VERIFICATION_WORKER_COMMITMENT_CACHE = {}


def _verification_worker_ready(value: int) -> int:
    return value


def _prepared_commitments_from_blob(blob: bytes,cache: dict):
    cached = cache.get(blob)
    if cached is not None:
        return cached
    if not blob or len(blob) % 48 != 0:
        raise ValueError("commitment batch must contain compressed G1 points")
    points = [
        g1_from_bytes(blob[offset:offset + 48])
        for offset in range(0,len(blob),48)
    ]
    cached = (points,prepare_g1_points(points))
    if len(cache) >= 8:
        cache.pop(next(iter(cache)))
    cache[blob] = cached
    return cached


def _verify_kzg_task(task):
    if _VERIFICATION_WORKER_KZG is None:
        raise RuntimeError("verification worker is not initialized")
    batch_index,commitment_blob,weights,points,values,proof_bytes = task
    aggregation_started = time.perf_counter()
    commitments,prepared_commitments = _prepared_commitments_from_blob(
        commitment_blob,_VERIFICATION_WORKER_COMMITMENT_CACHE
    )
    aggregate_commitment = aggregate_commitments(
        commitments,weights,prepared_commitments
    )
    aggregation_seconds = time.perf_counter() - aggregation_started
    pairing_started = time.perf_counter()
    valid = _VERIFICATION_WORKER_KZG.verify_multi(
        aggregate_commitment,points,values,g1_from_bytes(proof_bytes)
    )
    pairing_seconds = time.perf_counter() - pairing_started
    return batch_index,bool(valid),aggregation_seconds,pairing_seconds


def _verify_kzg_task_local(kzg: KZG,task,cache: dict):
    batch_index,commitment_blob,weights,points,values,proof_bytes = task
    aggregation_started = time.perf_counter()
    commitments,prepared_commitments = _prepared_commitments_from_blob(
        commitment_blob,cache
    )
    aggregate_commitment = aggregate_commitments(
        commitments,weights,prepared_commitments
    )
    aggregation_seconds = time.perf_counter() - aggregation_started
    pairing_started = time.perf_counter()
    valid = kzg.verify_multi(
        aggregate_commitment,points,values,g1_from_bytes(proof_bytes)
    )
    pairing_seconds = time.perf_counter() - pairing_started
    return batch_index,bool(valid),aggregation_seconds,pairing_seconds


def _get_verification_executor(
    srs_path: str | Path,worker_count: int
) -> ProcessPoolExecutor:
    global _VERIFICATION_EXECUTOR,_VERIFICATION_EXECUTOR_KEY
    resolved_srs_path = str(Path(srs_path).resolve())
    executor_key = (resolved_srs_path,int(worker_count))
    if _VERIFICATION_EXECUTOR is not None and _VERIFICATION_EXECUTOR_KEY == executor_key:
        return _VERIFICATION_EXECUTOR
    if _VERIFICATION_EXECUTOR is not None:
        _VERIFICATION_EXECUTOR.shutdown(wait=True,cancel_futures=True)
    _VERIFICATION_EXECUTOR = ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_verification_worker,
        initargs=(resolved_srs_path,),
    )
    _VERIFICATION_EXECUTOR_KEY = executor_key
    list(
        _VERIFICATION_EXECUTOR.map(
            _verification_worker_ready,range(worker_count),chunksize=1
        )
    )
    return _VERIFICATION_EXECUTOR


def _shutdown_verification_executor() -> None:
    global _VERIFICATION_EXECUTOR,_VERIFICATION_EXECUTOR_KEY
    if _VERIFICATION_EXECUTOR is not None:
        _VERIFICATION_EXECUTOR.shutdown(wait=True,cancel_futures=True)
        _VERIFICATION_EXECUTOR = None
        _VERIFICATION_EXECUTOR_KEY = None


atexit.register(_shutdown_verification_executor)


def _require_equal(name: str, left, right) -> None:
    if left != right:
        raise ValueError(f"{name} mismatch: {left!r} != {right!r}")


def _decode_hex(value: object, expected_length: int, name: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a hexadecimal string")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not valid hexadecimal data") from exc
    if len(decoded) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} bytes")
    return decoded


def _aggregate_sparse_claimed_values(
    base_values: list[int],
    slots: list[int],
    changed_values_by_edge: list[list[tuple[int,int,int,bytes]]],
    weights: list[int],
) -> list[int]:
    """Aggregate a shared base row plus its authenticated sparse changes."""
    if len(changed_values_by_edge) != len(weights):
        raise ValueError("sparse value rows and weights must have the same length")
    normalized_weights = [int(weight) % FR_MODULUS for weight in weights]
    weight_sum = sum(normalized_weights) % FR_MODULUS
    result = [int(base_values[slot]) * weight_sum for slot in slots]
    for changed_values,weight in zip(changed_values_by_edge,normalized_weights):
        for result_index,slot,value,_encoded_value in changed_values:
            result[result_index] += weight * (
                int(value) - int(base_values[slot])
            )
    return [value % FR_MODULUS for value in result]


def _expected_cell_ids(grid: GeohashGrid, rect: CellRect) -> list[str]:
    normalized = rect.normalized()
    result: list[str] = []
    for lat_index in range(normalized.lat_index_min, normalized.lat_index_max + 1):
        for lon_index in range(normalized.lon_index_min, normalized.lon_index_max + 1):
            result.append(grid.prefix_from_indices(lon_index, lat_index))
    return result


@dataclass(frozen=True)
class VerificationContext:
    """Long-lived verifier state loaded before the online verification timer."""

    anchor: dict
    srs: SRS
    kzg: KZG
    actual_srs_fingerprint: str
    expected_root: bytes
    srs_fingerprint_bytes: bytes
    day_start_ts: int
    day_end_ts: int
    slot_seconds: int
    slot_count: int
    all_domain_points: list[int]
    base_evaluation_values: list[int]
    base_evaluation_bytes: tuple[bytes,...]
    grid: GeohashGrid
    srs_path: str
    verification_worker_count: int
    prepared_commitment_cache: dict[bytes, tuple[list, object]] = field(
        default_factory=dict,compare=False,repr=False
    )


def prepare_verification_context(
    anchor_path: str | Path,
    srs_path: str | Path,
    verification_workers: int | None = None,
) -> VerificationContext:
    anchor = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    srs = SRS.load(srs_path)
    actual_srs_fingerprint = SRS.fingerprint(srs_path)
    default_start_ts = day_start_timestamp(anchor["date"])
    slot_seconds = int(anchor.get("slot_seconds", SLOT_SECONDS))
    slot_count = int(anchor.get("slot_count", SLOTS_PER_DAY))
    day_start_ts = int(anchor.get("period_start_ts", default_start_ts))
    day_end_ts = int(
        anchor.get("period_end_ts", day_start_ts + slot_seconds * slot_count)
    )
    if slot_seconds <= 0 or slot_count <= 0 or slot_count > srs.domain_size:
        raise ValueError("invalid authenticated time-slot configuration")
    if day_end_ts != day_start_ts + slot_seconds * slot_count:
        raise ValueError("period length does not match slot configuration")
    requested_workers = (
        DEFAULT_VERIFICATION_WORKERS
        if verification_workers is None
        else max(1,int(verification_workers))
    )
    worker_count = min(requested_workers,max(1,os.cpu_count() or 1))
    if worker_count >= 2:
        _get_verification_executor(srs_path,worker_count)
    base_evaluation_values = base_evaluations(
        anchor["date"],srs.domain_size,slot_count
    )
    return VerificationContext(
        anchor=anchor,
        srs=srs,
        kzg=KZG(srs),
        actual_srs_fingerprint=actual_srs_fingerprint,
        expected_root=_decode_hex(anchor["daily_root"], 32, "daily root"),
        srs_fingerprint_bytes=_decode_hex(anchor["srs_fingerprint"], 32, "SRS fingerprint"),
        day_start_ts=day_start_ts,
        day_end_ts=day_end_ts,
        slot_seconds=slot_seconds,
        slot_count=slot_count,
        all_domain_points=domain_points(srs.domain_size),
        base_evaluation_values=base_evaluation_values,
        base_evaluation_bytes=tuple(
            value.to_bytes(32,"big") for value in base_evaluation_values
        ),
        grid=GeohashGrid(int(anchor["cell_bits"]), int(anchor["coordinate_scale"])),
        srs_path=str(Path(srs_path).resolve()),
        verification_worker_count=worker_count,
    )


def _verify_batch_or_raise_from_memory(
    context: VerificationContext,
    cell_rect: CellRect,
    start_ts: int,
    end_ts: int,
    claimed_result_trajectory_ids: list[int],
    package: dict,
    edge_batch_size: int | None = None,
) -> dict:
    """Verify one package without loading any File/SRS data inside the timer."""
    if edge_batch_size is None:
        edge_batch_size = int(package["edge_batch_size"])
    anchor = context.anchor
    srs = context.srs
    kzg = context.kzg
    normalized_rect = cell_rect.normalized()
    rect_values = [
        normalized_rect.lon_index_min,
        normalized_rect.lon_index_max,
        normalized_rect.lat_index_min,
        normalized_rect.lat_index_max,
    ]
    expected_cell_ids = _expected_cell_ids(context.grid, normalized_rect)
    expected_cell_set = set(expected_cell_ids)
    client_started = time.perf_counter()

    invariant_started = time.perf_counter()
    _require_equal("proof mode", package["proof_mode"], GEOHASH_BATCH_PROOF_MODE)
    _require_equal("slot encoding", package["slot_encoding"], SPARSE_SLOT_ENCODING)
    _require_equal("edge batch size", int(package["edge_batch_size"]), int(edge_batch_size))
    _require_equal("scheme version", package["scheme_version"], anchor["scheme_version"])
    _require_equal("date", package["date"], anchor["date"])
    _require_equal("timezone", package["timezone"], "UTC")
    _require_equal(
        "period start timestamp",
        int(package.get("period_start_ts", context.day_start_ts)),
        context.day_start_ts,
    )
    _require_equal(
        "period end timestamp",
        int(package.get("period_end_ts", context.day_end_ts)),
        context.day_end_ts,
    )
    _require_equal(
        "slot seconds",
        int(package.get("slot_seconds", context.slot_seconds)),
        context.slot_seconds,
    )
    _require_equal(
        "slot count",
        int(package.get("slot_count", context.slot_count)),
        context.slot_count,
    )
    _require_equal("domain size", package["domain_size"], anchor["domain_size"])
    _require_equal("maximum multi points", package["max_multi_points"], anchor["max_multi_points"])
    _require_equal("daily root", package["daily_root"], anchor["daily_root"])
    _require_equal("daily root type", package["daily_root_type"], anchor["daily_root_type"])
    _require_equal("SRS fingerprint", package["srs_fingerprint"], anchor["srs_fingerprint"])
    _require_equal("actual SRS fingerprint", context.actual_srs_fingerprint, anchor["srs_fingerprint"])
    _require_equal("SRS domain size", srs.domain_size, anchor["domain_size"])
    _require_equal("SRS maximum multi points", srs.max_multi_points, anchor["max_multi_points"])

    package_query = package["query"]
    expected_rect_dict = normalized_rect.to_dict()
    _require_equal("query cell rectangle", package_query["cell_rect"], expected_rect_dict)
    _require_equal("query cell IDs", package_query["cell_ids"], expected_cell_ids)
    _require_equal("query start timestamp", int(package_query["start_ts"]), int(start_ts))
    _require_equal("query end timestamp", int(package_query["end_ts"]), int(end_ts))
    expected_slots = complete_slots_for_interval(
        context.day_start_ts,
        int(start_ts),
        int(end_ts),
        context.slot_seconds,
        context.slot_count,
    )
    expected_slot_chunks = partition_slots(expected_slots, srs.max_multi_points)
    _require_equal("query slots", [int(v) for v in package_query["slots"]], expected_slots)
    _require_equal(
        "query slot chunks",
        [[int(v) for v in chunk] for chunk in package_query["slot_chunks"]],
        expected_slot_chunks,
    )

    claimed_result = [int(value) for value in claimed_result_trajectory_ids]
    if claimed_result != sorted(set(claimed_result)):
        raise ValueError("claimed result trajectory IDs must be sorted and unique")
    _require_equal(
        "package claimed result",
        [int(value) for value in package["result_trajectory_ids"]],
        claimed_result,
    )
    invariant_validation_seconds = time.perf_counter() - invariant_started

    # Stage 1: authenticate complete queried cells, edge lists and edge KZG commitments.
    spatial_started = time.perf_counter()
    raw_cells = package.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != len(expected_cell_ids):
        raise ValueError("package does not contain exactly the requested cells")

    queried_leaf_hashes: dict[str, bytes] = {}
    authenticated_commitments: dict[int, bytes] = {}
    authenticated_cell_by_edge: dict[int, str] = {}
    seen_cells: set[str] = set()

    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict):
            raise ValueError("malformed cell package")
        cell_id = str(raw_cell.get("cell_id", ""))
        if cell_id not in expected_cell_set or cell_id in seen_cells:
            raise ValueError(f"unexpected or duplicate cell {cell_id}")
        seen_cells.add(cell_id)
        expected_lon, expected_lat = context.grid.indices_from_prefix(cell_id)
        _require_equal(f"longitude index for cell {cell_id}", int(raw_cell["lon_index"]), expected_lon)
        _require_equal(f"latitude index for cell {cell_id}", int(raw_cell["lat_index"]), expected_lat)
        raw_edges = raw_cell.get("edges")
        if not isinstance(raw_edges, list):
            raise ValueError(f"cell {cell_id} has malformed edge list")
        if int(raw_cell.get("edge_count", -1)) != len(raw_edges):
            raise ValueError(f"cell {cell_id} edge count mismatch")

        records: list[CellEdgeRecord] = []
        commitment_by_edge: dict[int, bytes] = {}
        last_edge_id = -1
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                raise ValueError(f"cell {cell_id} contains malformed edge record")
            record = CellEdgeRecord.from_mapping(raw_edge)
            if record.edge_id <= last_edge_id:
                raise ValueError(f"cell {cell_id} edge IDs must be strictly increasing")
            last_edge_id = record.edge_id
            if record.edge_id in authenticated_commitments:
                raise ValueError(f"edge {record.edge_id} appears in more than one queried cell")
            commitment = _decode_hex(
                raw_edge.get("kzg_commitment"), 48, f"KZG commitment for edge {record.edge_id}"
            )
            records.append(record)
            commitment_by_edge[record.edge_id] = commitment
            authenticated_commitments[record.edge_id] = commitment
            authenticated_cell_by_edge[record.edge_id] = cell_id

        queried_leaf_hashes[cell_id] = daily_cell_leaf_hash(
            anchor["date"], cell_id, records, commitment_by_edge, int(anchor["cell_bits"])
        )

    if seen_cells != expected_cell_set:
        raise ValueError("one or more requested cells are missing")

    verifier_tree = SparseMerkleTree(int(anchor["cell_bits"]), {})
    if not verifier_tree.verify_multiproof(
        queried_leaf_hashes, package.get("spatial_multiproof", {}), context.expected_root
    ):
        raise ValueError("spatial Geohash cell Merkle multiproof failed")
    spatial_verification_seconds = time.perf_counter() - spatial_started

    # Stage 2A: reconstruct the requested time-slot values/trajectory IDs from proof data.
    reconstruction_started = time.perf_counter()
    expected_edge_ids = sorted(authenticated_commitments)
    edge_packages = package.get("edges")
    if not isinstance(edge_packages, list):
        raise ValueError("edge packages must be a list")
    actual_edge_ids = [int(item["edge_id"]) for item in edge_packages]
    if actual_edge_ids != expected_edge_ids:
        raise ValueError("edge packages do not exactly match authenticated cell edges")

    returned_nonempty_slot_count = sum(len(item.get("slots", [])) for item in edge_packages)
    _require_equal(
        "returned nonempty slot count",
        int(package["returned_nonempty_slot_count"]),
        returned_nonempty_slot_count,
    )

    edge_contexts: dict[int, dict] = {}
    derived_result: set[int] = set()
    sparse_slot_context = prepare_sparse_slot_reconstruction(
        expected_slots,
        anchor["date"],
        context.base_evaluation_values,
        context.day_start_ts,
        context.day_end_ts,
        int(start_ts),
        int(end_ts),
        context.slot_seconds,
        context.slot_count,
    )
    slot_chunk_positions = {
        slot:(chunk_index,result_index)
        for chunk_index,chunk_slots in enumerate(expected_slot_chunks)
        for result_index,slot in enumerate(chunk_slots)
    }
    for edge_package in edge_packages:
        edge_id = int(edge_package["edge_id"])
        expected_cell_id = authenticated_cell_by_edge[edge_id]
        _require_equal(f"cell membership for edge {edge_id}", edge_package["cell_id"], expected_cell_id)
        claimed_nonempty_values, edge_trajectory_ids = reconstruct_sparse_slots_prepared(
            edge_package.get("slots", []),
            edge_id,
            sparse_slot_context,
        )
        derived_result.update(edge_trajectory_ids)
        claimed_values_by_chunk = [
            [] for _ in expected_slot_chunks
        ]
        for slot,value in claimed_nonempty_values.items():
            chunk_index,result_index = slot_chunk_positions[slot]
            claimed_values_by_chunk[chunk_index].append(
                (result_index,slot,value,value.to_bytes(32,"big"))
            )
        commitment_bytes = authenticated_commitments[edge_id]
        edge_contexts[edge_id] = {
            "cell_id": expected_cell_id,
            "commitment_bytes": commitment_bytes,
            "claimed_values_by_chunk": claimed_values_by_chunk,
        }
    trajectory_reconstruction_seconds = time.perf_counter() - reconstruction_started

    batch_packages = package.get("batches")
    if not isinstance(batch_packages, list):
        raise ValueError("batch proof packages must be a list")
    expected_groups = partition_edge_ids(expected_edge_ids, edge_batch_size)
    expected_batch_count = len(expected_groups) * len(expected_slot_chunks)
    if len(batch_packages) != expected_batch_count:
        raise ValueError(f"batch count mismatch: {len(batch_packages)} != {expected_batch_count}")

    verification_worker_count = min(
        context.verification_worker_count,max(1,expected_batch_count)
    )
    verification_executor = None
    if verification_worker_count >= 2:
        verification_executor = _get_verification_executor(
            context.srs_path,context.verification_worker_count
        )

    # Stage 2B: cross-cell Batch KZG verification.
    kzg_started = time.perf_counter()
    kzg_task_preparation_started = time.perf_counter()
    edge_group_contexts = [
        [edge_contexts[edge_id] for edge_id in edge_ids]
        for edge_ids in expected_groups
    ]
    edge_group_commitment_bytes = [
        [item["commitment_bytes"] for item in contexts]
        for contexts in edge_group_contexts
    ]
    edge_group_commitment_blobs = [
        b"".join(commitment_bytes)
        for commitment_bytes in edge_group_commitment_bytes
    ]
    edge_group_prefixes = [
        [item["cell_id"] for item in contexts]
        for contexts in edge_group_contexts
    ]
    base_packed_rows = [
        b"".join(context.base_evaluation_bytes[slot] for slot in chunk_slots)
        for chunk_slots in expected_slot_chunks
    ]
    batch_index = 0
    verification_tasks = []
    verification_futures = []
    first_worker_submission_time = None
    transcript_reconstruction_seconds = 0.0
    value_aggregation_seconds = 0.0
    for time_chunk_index, expected_chunk_slots in enumerate(expected_slot_chunks):
        points = [context.all_domain_points[slot] for slot in expected_chunk_slots]
        for edge_batch_index, expected_batch_edge_ids in enumerate(expected_groups):
            batch_package = batch_packages[batch_index]
            _require_equal("batch index", int(batch_package["batch_index"]), batch_index)
            _require_equal(
                f"time chunk index for batch {batch_index}",
                int(batch_package["time_chunk_index"]),
                time_chunk_index,
            )
            _require_equal(
                f"edge batch index for batch {batch_index}",
                int(batch_package["edge_batch_index"]),
                edge_batch_index,
            )
            batch_slots = [int(value) for value in batch_package["slots"]]
            batch_edge_ids = [int(value) for value in batch_package["edge_ids"]]
            _require_equal(f"slots for batch {batch_index}", batch_slots, expected_chunk_slots)
            _require_equal(f"edge IDs for batch {batch_index}", batch_edge_ids, expected_batch_edge_ids)
            contexts = edge_group_contexts[edge_batch_index]
            expected_prefixes = edge_group_prefixes[edge_batch_index]
            _require_equal(
                f"cell prefixes for batch {batch_index}",
                [str(value) for value in batch_package["cell_prefixes"]],
                expected_prefixes,
            )
            commitment_bytes = edge_group_commitment_bytes[edge_batch_index]
            transcript_started = time.perf_counter()
            changed_values_by_edge = [
                item["claimed_values_by_chunk"][time_chunk_index]
                for item in contexts
            ]
            base_packed_row = base_packed_rows[time_chunk_index]
            packed_values_by_edge = []
            for changed_values in changed_values_by_edge:
                if not changed_values:
                    packed_values_by_edge.append(base_packed_row)
                    continue
                packed_row = bytearray(base_packed_row)
                for result_index,_slot,_value,encoded_value in changed_values:
                    start_offset = result_index * 32
                    packed_row[start_offset:start_offset + 32] = encoded_value
                packed_values_by_edge.append(bytes(packed_row))
            challenge = geohash_batch_challenge_packed(
                anchor["date"],
                context.expected_root,
                context.srs_fingerprint_bytes,
                srs.domain_size,
                rect_values,
                int(start_ts),
                int(end_ts),
                batch_index,
                expected_chunk_slots,
                batch_edge_ids,
                expected_prefixes,
                commitment_bytes,
                packed_values_by_edge,
            )
            claimed_challenge = fr_from_bytes(
                _decode_hex(batch_package["challenge"], 32, f"challenge for batch {batch_index}")
            )
            if challenge != claimed_challenge:
                raise ValueError(f"batch challenge mismatch for batch {batch_index}")
            transcript_reconstruction_seconds += (
                time.perf_counter() - transcript_started
            )
            weights = batch_weights(challenge, len(batch_edge_ids))
            value_aggregation_started = time.perf_counter()
            aggregate_claimed_values = _aggregate_sparse_claimed_values(
                context.base_evaluation_values,
                expected_chunk_slots,
                changed_values_by_edge,
                weights,
            )
            value_aggregation_seconds += (
                time.perf_counter() - value_aggregation_started
            )
            proof_bytes = _decode_hex(
                batch_package["proof"],48,f"KZG proof for batch {batch_index}"
            )
            verification_task = (
                batch_index,
                edge_group_commitment_blobs[edge_batch_index],
                weights,
                points,
                aggregate_claimed_values,
                proof_bytes,
            )
            if verification_executor is None:
                verification_tasks.append(verification_task)
            else:
                if first_worker_submission_time is None:
                    first_worker_submission_time = time.perf_counter()
                verification_futures.append(
                    verification_executor.submit(_verify_kzg_task,verification_task)
                )
            batch_index += 1
    kzg_task_preparation_seconds = (
        time.perf_counter() - kzg_task_preparation_started
    )

    kzg_post_preparation_started = time.perf_counter()
    if verification_executor is not None:
        verification_results = [future.result() for future in verification_futures]
        kzg_worker_wall_seconds = (
            time.perf_counter() - first_worker_submission_time
        )
    else:
        local_worker_started = time.perf_counter()
        verification_results = [
            _verify_kzg_task_local(
                kzg,task,context.prepared_commitment_cache
            )
            for task in verification_tasks
        ]
        kzg_worker_wall_seconds = time.perf_counter() - local_worker_started
    commitment_aggregation_seconds = 0.0
    worker_pairing_seconds_sum = 0.0
    for (
        verified_batch_index,
        valid,
        aggregation_seconds,
        pairing_seconds,
    ) in verification_results:
        commitment_aggregation_seconds += aggregation_seconds
        worker_pairing_seconds_sum += pairing_seconds
        if not valid:
            raise ValueError(f"batch KZG proof failed for batch {verified_batch_index}")
    kzg_post_preparation_seconds = (
        time.perf_counter() - kzg_post_preparation_started
    )
    kzg_verification_seconds = time.perf_counter() - kzg_started

    result_check_started = time.perf_counter()
    derived_result_list = sorted(derived_result)
    if derived_result_list != claimed_result:
        raise ValueError(
            f"claimed result {claimed_result} does not equal verified result {derived_result_list}"
        )
    result_check_seconds = time.perf_counter() - result_check_started
    client_verification_seconds = time.perf_counter() - client_started

    return {
        "claimed_result_trajectory_ids": claimed_result,
        "claimed_result_trajectory_labels": [
            format_trajectory_id(v, anchor["trajectory_id_scheme"])
            for v in claimed_result
        ],
        "derived_result_trajectory_ids": derived_result_list,
        "derived_result_trajectory_labels": [
            format_trajectory_id(v, anchor["trajectory_id_scheme"])
            for v in derived_result_list
        ],
        "verified_cell_count": len(expected_cell_ids),
        "verified_edge_count": len(expected_edge_ids),
        "spatial_multiproof_node_count": len(package["spatial_multiproof"].get("nodes", [])),
        "expected_slots": expected_slots,
        "expected_slot_chunks": expected_slot_chunks,
        "returned_nonempty_slot_count": returned_nonempty_slot_count,
        "verified_batch_count": len(batch_packages),
        "verification_worker_count": verification_worker_count,
        "time_chunk_count": len(expected_slot_chunks),
        "invariant_validation_seconds": invariant_validation_seconds,
        "spatial_verification_seconds": spatial_verification_seconds,
        "trajectory_reconstruction_seconds": trajectory_reconstruction_seconds,
        "kzg_verification_seconds": kzg_verification_seconds,
        "kzg_task_preparation_seconds": kzg_task_preparation_seconds,
        # Backward-compatible alias.  With pipelining this is the time left
        # after task preparation, not the complete worker/pairing wall time.
        "kzg_pairing_wall_seconds": kzg_post_preparation_seconds,
        "kzg_post_preparation_seconds": kzg_post_preparation_seconds,
        "kzg_worker_wall_seconds": kzg_worker_wall_seconds,
        "worker_pairing_seconds_sum": worker_pairing_seconds_sum,
        "transcript_reconstruction_seconds": transcript_reconstruction_seconds,
        "commitment_aggregation_seconds": commitment_aggregation_seconds,
        "value_aggregation_seconds": value_aggregation_seconds,
        "result_check_seconds": result_check_seconds,
        "client_verification_seconds": client_verification_seconds,
        "online_verification_seconds": client_verification_seconds,
    }


def verify_batch_query_package_from_memory(
    context: VerificationContext,
    cell_rect: CellRect,
    start_ts: int,
    end_ts: int,
    claimed_result_trajectory_ids: list[int],
    package: dict,
    edge_batch_size: int | None = None,
) -> dict:
    try:
        details = _verify_batch_or_raise_from_memory(
            context,
            cell_rect,
            start_ts,
            end_ts,
            claimed_result_trajectory_ids,
            package,
            edge_batch_size,
        )
        return {"valid": True, "errors": [], **details}
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}


def verify_batch_query_package(
    anchor_path: str | Path,
    srs_path: str | Path,
    cell_rect: CellRect,
    start_ts: int,
    end_ts: int,
    claimed_result_trajectory_ids: list[int],
    package: dict,
    edge_batch_size: int | None = None,
) -> dict:
    """Backward-compatible one-shot verifier.

    The returned timing fields still exclude anchor/SRS loading because the
    in-memory context is prepared before the verification timer starts.
    """
    verification_workers = min(
        DEFAULT_VERIFICATION_WORKERS,max(1,len(package.get("batches",[])))
    )
    context = prepare_verification_context(
        anchor_path,srs_path,verification_workers
    )
    return verify_batch_query_package_from_memory(
        context,
        cell_rect,
        start_ts,
        end_ts,
        claimed_result_trajectory_ids,
        package,
        edge_batch_size,
    )
