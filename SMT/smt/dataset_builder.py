from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .entry_store import PackedEdgeEntryStore
from .trajectory_catalog import PackedTrajectoryCatalog
from .trajectory_parser import collect_files, load_relaxed_json, parse_trajectory


def build_store_and_catalog(road, trajectory_globs: list[str], config_dir: Path):
    files = collect_files(trajectory_globs, config_dir)
    store = PackedEdgeEntryStore(road.max_eid)
    catalog = PackedTrajectoryCatalog()
    seen: set[bytes] = set()
    total_trajectories = 0
    total_segments = 0
    ordered_insert_time_s = 0.0

    for file_index, path in enumerate(files, 1):
        print(f"[{file_index}/{len(files)}] parse {path}")
        data = load_relaxed_json(path)
        if not isinstance(data, list):
            raise ValueError(f"top-level JSON must be list: {path}")
        for trajectory in data:
            parsed = parse_trajectory(trajectory, road)
            if parsed.trajectory_id in seen:
                raise ValueError("duplicate trajectory_id: " + parsed.trajectory_id.hex())
            seen.add(parsed.trajectory_id)
            catalog.add(parsed.trajectory_id, parsed.segments)

            # Only the ordered insertion itself is timed as index construction.
            # JSON loading, trajectory parsing and catalog preparation remain
            # outside Index Construction Time, matching the existing benchmark
            # accounting policy.
            t_insert = perf_counter()
            for eid, seg in zip(parsed.eid_path, parsed.segments):
                store.insert_by_start(
                    eid,
                    parsed.trajectory_id,
                    seg.start_time,
                    seg.end_time,
                )
            ordered_insert_time_s += perf_counter() - t_insert

            total_trajectories += 1
            total_segments += len(parsed.segments)

    store.validate_start_order()

    return (
        store,
        catalog,
        files,
        total_trajectories,
        total_segments,
        ordered_insert_time_s,
    )
