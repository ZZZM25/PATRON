from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from kzg.builder import (
    _build_edge_artifacts,
    _build_edge_worker,
    _initialize_build_worker,
    _normalized_worker_count,
)
from kzg.canonical import normalize_records
from kzg.cell_merkle import SparseMerkleTree, build_daily_leaf_hashes
from kzg.config import SCHEME_VERSION
from kzg.geohash_index import GeohashCellIndex
from kzg.kzg import KZG, SRS, g1_to_bytes, lagrange_g1_powers
from kzg.models import SegmentRecord
from kzg.ntt import evaluations_to_coefficients
from kzg.road_network import RoadNetwork
from kzg.storage import DailyStore
from kzg.vector import base_evaluations


PERIOD_SCHEME_VERSION = SCHEME_VERSION + "-PERIOD-V1"


def build_period(
    nodes_path: str | Path,
    edges_path: str | Path,
    cell_index_dir: str | Path,
    srs_path: str | Path,
    output_database: str | Path,
    anchor_path: str | Path,
    period_start_day: str,
    period_start_ts: int,
    period_end_ts: int,
    slot_seconds: int,
    edge_slots: dict[int, dict[int, list[SegmentRecord]]],
    trajectory_count: int,
    segment_count: int,
    worker_count: int | None = None,
) -> dict:
    """Build one authenticated multi-day period from canonical edge-slot records."""
    started = time.perf_counter()
    database_path = Path(output_database)
    anchor_file = Path(anchor_path)
    if database_path.exists():
        raise FileExistsError(f"output database already exists: {database_path}")
    if anchor_file.exists():
        raise FileExistsError(f"anchor file already exists: {anchor_file}")
    if slot_seconds <= 0 or period_end_ts <= period_start_ts:
        raise ValueError("invalid period boundaries or slot size")
    duration = int(period_end_ts) - int(period_start_ts)
    if duration % int(slot_seconds):
        raise ValueError("period duration must be divisible by slot size")
    slot_count = duration // int(slot_seconds)

    road = RoadNetwork.from_files(nodes_path, edges_path)
    cell_index = GeohashCellIndex.load(cell_index_dir)
    if set(road.edges) != set(cell_index.edge_to_cell):
        raise ValueError("processed road and cell index do not contain the same edges")

    srs = SRS.load(srs_path)
    if slot_count > srs.domain_size:
        raise ValueError(
            f"period needs {slot_count} slots but SRS domain has {srs.domain_size}"
        )
    kzg = KZG(srs)
    srs_fingerprint = SRS.fingerprint(srs_path)

    normalized_edge_slots: dict[int, dict[int, list[SegmentRecord]]] = {}
    for edge_id, slots in edge_slots.items():
        if edge_id not in road.edges:
            raise ValueError(f"trajectory references unknown split edge {edge_id}")
        normalized_slots = {}
        for slot, records in slots.items():
            normalized_slot = int(slot)
            if not 0 <= normalized_slot < slot_count:
                raise ValueError(f"invalid period slot {normalized_slot}")
            normalized_slots[normalized_slot] = normalize_records(records)
        if normalized_slots:
            normalized_edge_slots[int(edge_id)] = normalized_slots

    pure_compute_started = time.perf_counter()
    empty_compute_started = pure_compute_started
    empty_evaluations = base_evaluations(
        period_start_day, srs.domain_size, slot_count
    )
    empty_coefficients = evaluations_to_coefficients(empty_evaluations)
    empty_commitment_point = kzg.commit_g1(empty_coefficients)
    empty_commitment = g1_to_bytes(empty_commitment_point)
    empty_commitment_seconds = time.perf_counter() - empty_compute_started
    all_edge_ids = sorted(road.edges)
    active_edge_ids = sorted(normalized_edge_slots)
    build_worker_count = _normalized_worker_count(worker_count, len(active_edge_ids))
    commitments: dict[int, bytes] = {
        edge_id: empty_commitment for edge_id in all_edge_ids
    }
    active_compute_started = time.perf_counter()
    tasks = (
        (edge_id, normalized_edge_slots[edge_id]) for edge_id in active_edge_ids
    )
    if build_worker_count > 1:
        chunksize = max(
            1, min(16, len(active_edge_ids) // (build_worker_count * 16))
        )
        with ProcessPoolExecutor(
            max_workers=build_worker_count,
            initializer=_initialize_build_worker,
            initargs=(
                str(Path(srs_path).resolve()),
                period_start_day,
                empty_commitment,
                slot_count,
            ),
        ) as executor:
            results = list(
                executor.map(_build_edge_worker, tasks, chunksize=chunksize)
            )
    else:
        lagrange_powers = lagrange_g1_powers(srs)
        results = [
            _build_edge_artifacts(
                edge_id,
                slots,
                period_start_day,
                empty_evaluations,
                empty_commitment_point,
                lagrange_powers,
            )
            for edge_id, slots in tasks
        ]
    for edge_id, _slot_values, commitment in results:
        commitments[edge_id] = commitment
    active_commitment_seconds = time.perf_counter() - active_compute_started

    spatial_compute_started = time.perf_counter()
    cell_leaves = build_daily_leaf_hashes(
        period_start_day, cell_index, commitments
    )
    tree = SparseMerkleTree(cell_index.bits, cell_leaves)
    spatial_root_seconds = time.perf_counter() - spatial_compute_started
    pure_compute_seconds = time.perf_counter() - pure_compute_started

    storage_started = time.perf_counter()
    with DailyStore(database_path) as store:
        metadata = {
            "scheme_version": PERIOD_SCHEME_VERSION,
            "date": period_start_day,
            "period_start_ts": int(period_start_ts),
            "period_end_ts": int(period_end_ts),
            "slot_seconds": int(slot_seconds),
            "slot_count": int(slot_count),
            "timezone": "UTC",
            "timestamp_semantics": "unix-seconds-utc",
            "trajectory_id_scheme": "global-source-row-ordinal-1based",
            "domain_size": srs.domain_size,
            "max_multi_points": srs.max_multi_points,
            "srs_fingerprint": srs_fingerprint,
            "edge_count": len(all_edge_ids),
            "trajectory_count": int(trajectory_count),
            "segment_count": int(segment_count),
            "active_edge_count": len(active_edge_ids),
            "build_worker_count": build_worker_count,
            "cell_bits": cell_index.bits,
            "coordinate_scale": cell_index.coordinate_scale,
            "static_cell_index_fingerprint": cell_index.fingerprint(),
            "static_cell_root": cell_index.static_root_hash,
            "polynomial_storage": "sparse-evaluations-v1",
            "proof_polynomial_reconstruction": (
                "weighted-evaluations-then-inverse-ntt-v1"
            ),
        }
        for key, value in metadata.items():
            store.set_metadata(key, value)

        for edge_id in all_edge_ids:
            if edge_id not in normalized_edge_slots:
                store.put_edge_commitment(edge_id, empty_commitment, True, 0)

        for edge_id, slot_values, commitment in results:
            slots = normalized_edge_slots[edge_id]
            for slot, records in sorted(slots.items()):
                store.put_slot(edge_id, slot, records, slot_values[slot])
            store.put_edge_commitment(
                edge_id, commitment, False, len(slots)
            )

        for cell_id, leaf_hash in cell_leaves.items():
            store.put_daily_cell_leaf(cell_id, leaf_hash)
        store.set_metadata("daily_root", tree.root_hash.hex())
        store.set_metadata(
            "daily_root_type", "geohash-cell-period-sparse-merkle-v1"
        )
        store.set_metadata("nonempty_spatial_cell_count", len(cell_leaves))
        store.set_metadata(
            "nonempty_time_slot_count",
            sum(len(slots) for slots in normalized_edge_slots.values()),
        )
        store.commit()

    storage_seconds = time.perf_counter() - storage_started
    commitment_seconds = pure_compute_seconds + storage_seconds
    anchor = {
        "scheme_version": PERIOD_SCHEME_VERSION,
        "date": period_start_day,
        "period_start_ts": int(period_start_ts),
        "period_end_ts": int(period_end_ts),
        "slot_seconds": int(slot_seconds),
        "slot_count": int(slot_count),
        "timezone": "UTC",
        "timestamp_semantics": "unix-seconds-utc",
        "trajectory_id_scheme": "global-source-row-ordinal-1based",
        "domain_size": srs.domain_size,
        "max_multi_points": srs.max_multi_points,
        "edge_count": len(all_edge_ids),
        "cell_bits": cell_index.bits,
        "coordinate_scale": cell_index.coordinate_scale,
        "nonempty_spatial_cell_count": len(cell_index.cells),
        "daily_root_type": "geohash-cell-period-sparse-merkle-v1",
        "daily_root": tree.root_hash.hex(),
        "static_cell_index_fingerprint": cell_index.fingerprint(),
        "static_cell_root": cell_index.static_root_hash,
        "srs_fingerprint": srs_fingerprint,
    }
    anchor_file.parent.mkdir(parents=True, exist_ok=True)
    anchor_file.write_text(
        json.dumps(anchor, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        **anchor,
        "active_edges": len(active_edge_ids),
        "nonempty_time_slots": sum(
            len(slots) for slots in normalized_edge_slots.values()
        ),
        "segments": int(segment_count),
        "trajectories": int(trajectory_count),
        "build_worker_count": build_worker_count,
        "pure_compute_seconds": pure_compute_seconds,
        "empty_commitment_seconds": empty_commitment_seconds,
        "active_commitment_seconds": active_commitment_seconds,
        "spatial_root_seconds": spatial_root_seconds,
        "storage_seconds": storage_seconds,
        "commitment_seconds": commitment_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "database": str(database_path),
        "anchor": str(anchor_file),
    }
