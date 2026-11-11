"""Trajectory ingestion for the independent 50 m cell-partitioned variant."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from time import perf_counter

from .cell_partition import expand_trajectory_segment
from .entry_store import PackedEdgeEntryStore
from .models import ParsedTrajectory
from .trajectory_catalog import PackedTrajectoryCatalog
from .trajectory_id import compute_trajectory_id
from .trajectory_parser import (
    collect_files,
    load_relaxed_json,
    parse_cell_partitioned_trajectory,
)


def _insert_parsed(store, catalog, parsed, seen: set[bytes]) -> float:
    if parsed.trajectory_id in seen:
        raise ValueError("duplicate trajectory_id: " + parsed.trajectory_id.hex())
    seen.add(parsed.trajectory_id)
    catalog.add(parsed.trajectory_id, parsed.segments)
    started = perf_counter()
    for eid, segment in zip(parsed.eid_path, parsed.segments):
        store.insert_by_start(
            eid,
            parsed.trajectory_id,
            segment.start_time,
            segment.end_time,
        )
    return perf_counter() - started


def _build_json_store_and_catalog(
    source_road,
    partition,
    trajectory_globs: list[str],
    config_dir: Path,
):
    files = collect_files(trajectory_globs, config_dir)
    store = PackedEdgeEntryStore(partition.road.max_eid)
    catalog = PackedTrajectoryCatalog()
    seen: set[bytes] = set()
    total_trajectories = 0
    total_source_segments = 0
    total_partitioned_segments = 0
    ordered_insert_time_s = 0.0

    for file_index, path in enumerate(files, 1):
        print(f"[{file_index}/{len(files)}] parse {path}", flush=True)
        data = load_relaxed_json(path)
        if not isinstance(data, list):
            raise ValueError(f"top-level JSON must be list: {path}")
        for trajectory in data:
            parsed = parse_cell_partitioned_trajectory(
                trajectory, source_road, partition
            )
            ordered_insert_time_s += _insert_parsed(
                store, catalog, parsed, seen
            )
            total_trajectories += 1
            total_source_segments += len(trajectory[1])
            total_partitioned_segments += len(parsed.segments)

    store.validate_start_order()
    return (
        store,
        catalog,
        files,
        total_trajectories,
        total_source_segments,
        total_partitioned_segments,
        ordered_insert_time_s,
    )


def _set_large_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _load_beijing_points(text: str, row_no: int) -> list:
    try:
        points = json.loads(text)
    except Exception as exc:
        raise ValueError(f"row {row_no}: points_data JSON parse failed") from exc
    if not isinstance(points, list):
        raise ValueError(f"row {row_no}: points_data must be a JSON list")
    return points


def parse_cell_partitioned_beijing_points(
    points: list,
    row_no: int,
    source_road,
    partition,
) -> tuple[ParsedTrajectory | None, int, int]:
    """Parse one Beijing ``points_data`` value using existing CSV semantics.

    Returns ``(parsed, source_movement_segments, stationary_steps)``.  A row
    containing no movement returns ``parsed=None``.
    """
    if len(points) < 2:
        return None, 0, 0

    normalized: list[tuple[int, int]] = []
    previous_time: int | None = None
    for point_index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) < 4:
            raise ValueError(
                f"row {row_no} point {point_index}: "
                "expected [node_id,lon,lat,timestamp]"
            )
        node_id = int(point[0])
        timestamp = int(point[3])
        if previous_time is not None and timestamp < previous_time:
            raise ValueError(
                f"row {row_no}: timestamp goes backwards: "
                f"{previous_time}->{timestamp}"
            )
        normalized.append((node_id, timestamp))
        previous_time = timestamp

    eid_path: list[int] = []
    segments = []
    source_segment_count = 0
    stationary_steps = 0
    for (u, start), (v, end) in zip(normalized, normalized[1:]):
        if u == v:
            stationary_steps += 1
            continue
        if end < start:
            raise ValueError(f"row {row_no}: end < start")
        try:
            source_eid = source_road.resolve_edge(u, v)
        except KeyError as exc:
            raise KeyError(f"row {row_no}: road has no edge {u}->{v}") from exc
        expanded_eids, expanded_segments = expand_trajectory_segment(
            partition, source_eid, u, start, v, end
        )
        eid_path.extend(expanded_eids)
        segments.extend(expanded_segments)
        source_segment_count += 1

    if not eid_path:
        return None, 0, stationary_steps
    trajectory_id = compute_trajectory_id(
        eid_path, segments[0].start_time, segments[-1].end_time
    )
    return (
        ParsedTrajectory(trajectory_id, eid_path, segments),
        source_segment_count,
        stationary_steps,
    )


def _build_beijing_store_and_catalog(source_road, partition, path: Path):
    """Apply Beijing's established CSV semantics to the partitioned road."""
    _set_large_csv_field_limit()
    store = PackedEdgeEntryStore(partition.road.max_eid)
    catalog = PackedTrajectoryCatalog()
    seen: set[bytes] = set()
    total_raw = 0
    total_trajectories = 0
    total_source_segments = 0
    total_partitioned_segments = 0
    stationary_steps = 0
    skipped_short = 0
    skipped_no_movement = 0
    ordered_insert_time_s = 0.0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if "points_data" not in fields:
            raise ValueError("Beijing CSV missing points_data")
        if "source_traj_id" not in fields and "traj_id" not in fields:
            raise ValueError("Beijing CSV missing source_traj_id/traj_id")

        for row_no, row in enumerate(reader, 2):
            total_raw += 1
            points = _load_beijing_points(row["points_data"], row_no)
            if len(points) < 2:
                skipped_short += 1
                continue
            parsed, source_segment_count, row_stationary = (
                parse_cell_partitioned_beijing_points(
                    points, row_no, source_road, partition
                )
            )
            stationary_steps += row_stationary
            if parsed is None:
                skipped_no_movement += 1
                continue
            trajectory_id = parsed.trajectory_id
            if trajectory_id in seen:
                raise ValueError(
                    f"row {row_no}: duplicate trajectory_id {trajectory_id.hex()}"
                )
            seen.add(trajectory_id)
            catalog.add(trajectory_id, parsed.segments)

            started = perf_counter()
            for eid, segment in zip(parsed.eid_path, parsed.segments):
                store.insert_by_start(
                    eid, trajectory_id, segment.start_time, segment.end_time
                )
            ordered_insert_time_s += perf_counter() - started
            total_trajectories += 1
            total_source_segments += source_segment_count
            total_partitioned_segments += len(parsed.segments)

            if total_raw % 10_000 == 0:
                print(
                    f"read {total_raw} Beijing rows; kept {total_trajectories}; "
                    f"partitioned entries {total_partitioned_segments}",
                    flush=True,
                )

    store.validate_start_order()
    print(
        "Beijing cell-partitioned CSV parsed: "
        f"raw={total_raw}, kept={total_trajectories}, "
        f"source_segments={total_source_segments}, "
        f"partitioned_segments={total_partitioned_segments}, "
        f"stationary={stationary_steps}, short_skipped={skipped_short}, "
        f"no_movement_skipped={skipped_no_movement}",
        flush=True,
    )
    return (
        store,
        catalog,
        [path],
        total_trajectories,
        total_source_segments,
        total_partitioned_segments,
        ordered_insert_time_s,
    )


def build_cell_store_and_catalog(
    city: str,
    source_road,
    partition,
    trajectory_globs: list[str],
    config_dir: Path,
):
    """Build the new edge store while leaving old ingestion paths untouched."""
    if city.lower() == "beijing":
        files = collect_files(trajectory_globs, config_dir)
        if len(files) != 1:
            raise ValueError(f"expected one Beijing CSV, got {len(files)}")
        return _build_beijing_store_and_catalog(source_road, partition, files[0])
    return _build_json_store_and_catalog(
        source_road, partition, trajectory_globs, config_dir
    )
