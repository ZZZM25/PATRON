from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from kzg.canonical import cell_value, normalize_records
from kzg.cell_merkle import SparseMerkleTree, build_daily_leaf_hashes
from kzg.config import (
    DEFAULT_BUILD_WORKERS,
    PARALLEL_BUILD_EDGE_THRESHOLD,
    SCHEME_VERSION,
    SLOTS_PER_DAY,
)
from kzg.geohash_index import GeohashCellIndex
from kzg.kzg import (
    KZG,
    SRS,
    commit_sparse_evaluations,
    g1_from_bytes,
    g1_to_bytes,
    lagrange_g1_powers,
)
from kzg.ntt import evaluations_to_coefficients
from kzg.road_network import RoadNetwork
from kzg.storage import DailyStore
from kzg.trajectory_parser import parse_trajectory_segments
from kzg.vector import base_evaluations


_BUILD_WORKER_BASE_EVALUATIONS = None
_BUILD_WORKER_EMPTY_COMMITMENT = None
_BUILD_WORKER_LAGRANGE_POWERS = None
_BUILD_WORKER_DAY = None


def _initialize_build_worker(
    srs_path: str,
    day: str,
    empty_commitment: bytes,
    slot_count: int = SLOTS_PER_DAY,
) -> None:
    global _BUILD_WORKER_BASE_EVALUATIONS
    global _BUILD_WORKER_EMPTY_COMMITMENT
    global _BUILD_WORKER_LAGRANGE_POWERS
    global _BUILD_WORKER_DAY
    srs = SRS.load(srs_path)
    _BUILD_WORKER_BASE_EVALUATIONS = base_evaluations(
        day, srs.domain_size, slot_count
    )
    _BUILD_WORKER_EMPTY_COMMITMENT = g1_from_bytes(empty_commitment)
    _BUILD_WORKER_LAGRANGE_POWERS = lagrange_g1_powers(srs)
    _BUILD_WORKER_DAY = day


def _build_edge_artifacts(
    edge_id: int,
    slots,
    day: str,
    base_values: list[int],
    empty_commitment,
    lagrange_powers,
) -> tuple[int, dict[int, int], bytes]:
    slot_values: dict[int, int] = {}
    for slot, records in sorted(slots.items()):
        value = cell_value(day, edge_id, slot, records)
        slot_values[int(slot)] = value
    commitment = commit_sparse_evaluations(
        empty_commitment,
        base_values,
        slot_values,
        lagrange_powers,
    )
    return (
        int(edge_id),
        slot_values,
        g1_to_bytes(commitment),
    )


def _build_edge_worker(task) -> tuple[int, dict[int, int], bytes]:
    if (
        _BUILD_WORKER_BASE_EVALUATIONS is None
        or _BUILD_WORKER_EMPTY_COMMITMENT is None
        or _BUILD_WORKER_LAGRANGE_POWERS is None
        or _BUILD_WORKER_DAY is None
    ):
        raise RuntimeError("build worker is not initialized")
    edge_id, slots = task
    return _build_edge_artifacts(
        edge_id,
        slots,
        _BUILD_WORKER_DAY,
        _BUILD_WORKER_BASE_EVALUATIONS,
        _BUILD_WORKER_EMPTY_COMMITMENT,
        _BUILD_WORKER_LAGRANGE_POWERS,
    )


def _normalized_worker_count(requested_worker_count: int | None, active_edge_count: int) -> int:
    requested = DEFAULT_BUILD_WORKERS if requested_worker_count is None else int(requested_worker_count)
    if requested <= 0:
        raise ValueError("build worker count must be positive")
    if active_edge_count < PARALLEL_BUILD_EDGE_THRESHOLD:
        return 1
    return max(1, min(requested, active_edge_count))


def build_day(
    nodes_path: str | Path,
    edges_path: str | Path,
    trajectories_path: str | Path,
    cell_index_dir: str | Path,
    srs_path: str | Path,
    output_database: str | Path,
    anchor_path: str | Path,
    day: str | None = None,
    parallel_policy: str = "min",
    worker_count: int | None = None,
) -> dict:
    """Build one UTC-day dataset using Geohash-cell spatial authentication.

    Each road edge keeps its own KZG commitment.  The daily trusted root is no
    longer an ordinary edge Merkle tree: every non-empty Geohash cell leaf
    binds the complete road-edge records in that cell plus each edge's daily
    KZG commitment, and those leaves are aggregated by the fixed-depth prefix
    sparse Merkle tree.
    """
    started = time.perf_counter()
    database_path = Path(output_database)
    anchor_file = Path(anchor_path)
    if database_path.exists():
        raise FileExistsError(f"output database already exists: {database_path}")
    if anchor_file.exists():
        raise FileExistsError(f"anchor file already exists: {anchor_file}")

    road = RoadNetwork.from_files(nodes_path, edges_path)
    cell_index = GeohashCellIndex.load(cell_index_dir)
    if set(road.edges) != set(cell_index.edge_to_cell):
        missing = sorted(set(road.edges) - set(cell_index.edge_to_cell))
        extra = sorted(set(cell_index.edge_to_cell) - set(road.edges))
        raise ValueError(
            f"processed road and cell index do not contain the same edges; missing={missing[:3]}, extra={extra[:3]}"
        )

    srs = SRS.load(srs_path)
    if srs.domain_size < SLOTS_PER_DAY:
        raise ValueError("SRS domain size is smaller than 1440 minute slots")
    kzg = KZG(srs)
    srs_fingerprint = SRS.fingerprint(srs_path)

    parse_started = time.perf_counter()
    detected_day, raw_edge_slots, segment_count, trajectory_count = parse_trajectory_segments(
        trajectories_path, road, day, parallel_policy
    )
    parse_seconds = time.perf_counter() - parse_started
    edge_slots = {
        edge_id: {
            slot: normalize_records(records)
            for slot, records in slots.items()
        }
        for edge_id, slots in raw_edge_slots.items()
    }

    pure_compute_started = time.perf_counter()
    empty_compute_started = pure_compute_started
    empty_evaluations = base_evaluations(detected_day, srs.domain_size)
    empty_coefficients = evaluations_to_coefficients(empty_evaluations)
    empty_commitment_point = kzg.commit_g1(empty_coefficients)
    empty_commitment = g1_to_bytes(empty_commitment_point)
    empty_commitment_seconds = time.perf_counter() - empty_compute_started

    all_edge_ids = sorted(road.edges)
    active_edge_ids = sorted(edge_slots)
    active_total = len(active_edge_ids)
    build_worker_count = _normalized_worker_count(worker_count, active_total)
    commitments: dict[int, bytes] = {
        edge_id: empty_commitment for edge_id in all_edge_ids
    }
    active_compute_started = time.perf_counter()
    tasks = ((edge_id, edge_slots[edge_id]) for edge_id in active_edge_ids)
    if build_worker_count > 1:
        chunksize = max(1, min(16, active_total // (build_worker_count * 16)))
        with ProcessPoolExecutor(
            max_workers=build_worker_count,
            initializer=_initialize_build_worker,
            initargs=(
                str(Path(srs_path).resolve()),
                detected_day,
                empty_commitment,
                SLOTS_PER_DAY,
            ),
        ) as executor:
            active_results = list(
                executor.map(_build_edge_worker, tasks, chunksize=chunksize)
            )
    else:
        lagrange_powers = lagrange_g1_powers(srs)
        active_results = [
            _build_edge_artifacts(
                edge_id,
                slots,
                detected_day,
                empty_evaluations,
                empty_commitment_point,
                lagrange_powers,
            )
            for edge_id, slots in tasks
        ]
    for edge_id, _slot_values, commitment in active_results:
        commitments[edge_id] = commitment
    active_commitment_seconds = time.perf_counter() - active_compute_started

    spatial_compute_started = time.perf_counter()
    daily_cell_leaves = build_daily_leaf_hashes(detected_day, cell_index, commitments)
    cell_tree = SparseMerkleTree(cell_index.bits, daily_cell_leaves)
    daily_root = cell_tree.root_hash
    spatial_root_seconds = time.perf_counter() - spatial_compute_started
    pure_compute_seconds = time.perf_counter() - pure_compute_started

    storage_started = time.perf_counter()
    with DailyStore(database_path) as store:
        store.set_metadata("scheme_version", SCHEME_VERSION)
        store.set_metadata("date", detected_day)
        store.set_metadata("timezone", "UTC")
        store.set_metadata("timestamp_semantics", "unix-seconds-utc")
        store.set_metadata("trajectory_id_scheme", "YYYYMMDD*1000000+daily_ordinal_1based")
        store.set_metadata("domain_size", srs.domain_size)
        store.set_metadata("slot_count", SLOTS_PER_DAY)
        store.set_metadata("max_multi_points", srs.max_multi_points)
        store.set_metadata("srs_fingerprint", srs_fingerprint)
        store.set_metadata("edge_count", len(all_edge_ids))
        store.set_metadata("trajectory_count", trajectory_count)
        store.set_metadata("segment_count", segment_count)
        store.set_metadata("active_edge_count", active_total)
        store.set_metadata("build_worker_count", build_worker_count)
        store.set_metadata("cell_bits", cell_index.bits)
        store.set_metadata("coordinate_scale", cell_index.coordinate_scale)
        store.set_metadata("static_cell_index_fingerprint", cell_index.fingerprint())
        store.set_metadata("static_cell_root", cell_index.static_root_hash)
        store.set_metadata("polynomial_storage", "sparse-evaluations-v1")
        store.set_metadata(
            "proof_polynomial_reconstruction",
            "weighted-evaluations-then-inverse-ntt-v1",
        )

        for edge_id in all_edge_ids:
            if edge_id not in edge_slots:
                store.put_edge_commitment(edge_id, empty_commitment, True, 0)

        active_done = 0
        for edge_id, slot_values, commitment in active_results:
            slots = edge_slots[edge_id]
            for slot, records in sorted(slots.items()):
                store.put_slot(edge_id, slot, records, slot_values[slot])
            store.put_edge_commitment(edge_id, commitment, False, len(slots))
            active_done += 1
            if active_done == 1 or active_done % 50 == 0 or active_done == active_total:
                print(f"已完成活跃道路承诺：{active_done}/{active_total}，edge_id={edge_id}")

        # Spatial authentication root: Cell(prefix, road edges, edge KZG commitments)
        for cell_id, leaf_hash in daily_cell_leaves.items():
            store.put_daily_cell_leaf(cell_id, leaf_hash)

        store.set_metadata("daily_root", daily_root.hex())
        store.set_metadata("daily_root_type", "geohash-cell-sparse-merkle-v1")
        store.set_metadata("nonempty_spatial_cell_count", len(daily_cell_leaves))
        store.set_metadata("nonempty_time_slot_count", sum(len(slots) for slots in edge_slots.values()))
        store.commit()

    storage_seconds = time.perf_counter() - storage_started
    commitment_seconds = pure_compute_seconds + storage_seconds
    anchor = {
        "scheme_version": SCHEME_VERSION,
        "date": detected_day,
        "timezone": "UTC",
        "timestamp_semantics": "unix-seconds-utc",
        "trajectory_id_scheme": "YYYYMMDD*1000000+daily_ordinal_1based",
        "domain_size": srs.domain_size,
        "max_multi_points": srs.max_multi_points,
        "edge_count": len(all_edge_ids),
        "cell_bits": cell_index.bits,
        "coordinate_scale": cell_index.coordinate_scale,
        "nonempty_spatial_cell_count": len(cell_index.cells),
        "daily_root_type": "geohash-cell-sparse-merkle-v1",
        "daily_root": daily_root.hex(),
        "static_cell_index_fingerprint": cell_index.fingerprint(),
        "static_cell_root": cell_index.static_root_hash,
        "srs_fingerprint": srs_fingerprint,
    }
    anchor_file.parent.mkdir(parents=True, exist_ok=True)
    anchor_file.write_text(
        json.dumps(anchor, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    elapsed_seconds = time.perf_counter() - started
    return {
        **anchor,
        "active_edges": active_total,
        "nonempty_time_slots": sum(len(slots) for slots in edge_slots.values()),
        "segments": segment_count,
        "trajectories": trajectory_count,
        "build_worker_count": build_worker_count,
        "parse_seconds": parse_seconds,
        "pure_compute_seconds": pure_compute_seconds,
        "empty_commitment_seconds": empty_commitment_seconds,
        "active_commitment_seconds": active_commitment_seconds,
        "spatial_root_seconds": spatial_root_seconds,
        "storage_seconds": storage_seconds,
        "commitment_seconds": commitment_seconds,
        "elapsed_seconds": elapsed_seconds,
        "database": str(database_path),
        "anchor": str(anchor_file),
    }
