from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import map_traj_to_new_network_undirected as mapper
import prepare_rsg_dataset as prepare


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "beijing" / "source"
DEFAULT_OUTPUT = ROOT / "rsg_50m_beijing"
UTC = timezone.utc


@dataclass(frozen=True)
class TimeUnit:
    name: str
    first_month: str
    second_month: str
    start_timestamp: int
    end_timestamp: int

    @property
    def slot_count(self) -> int:
        return (self.end_timestamp - self.start_timestamp) // 60


def month_start(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=UTC).timestamp())


def next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def build_units() -> list[TimeUnit]:
    """Pair every month from 2007-04 through 2012-07, inclusive."""
    months: list[tuple[int, int]] = []
    year, month = 2007, 4
    while (year, month) <= (2012, 7):
        months.append((year, month))
        year, month = next_month(year, month)
    if len(months) % 2:
        raise AssertionError("Beijing month range must contain an even number of months")

    units = []
    for index in range(0, len(months), 2):
        y1, m1 = months[index]
        y2, m2 = months[index + 1]
        end_y, end_m = next_month(y2, m2)
        first = f"{y1:04d}-{m1:02d}"
        second = f"{y2:04d}-{m2:02d}"
        units.append(
            TimeUnit(
                name=f"{first}_{second}",
                first_month=first,
                second_month=second,
                start_timestamp=month_start(y1, m1),
                end_timestamp=month_start(end_y, end_m),
            )
        )
    return units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Beijing 50 m road network and consecutive two-month trajectory units"
    )
    parser.add_argument("--nodes", type=Path, default=DEFAULT_INPUT / "beijing_nodes.txt")
    parser.add_argument("--edges", type=Path, default=DEFAULT_INPUT / "beijing_edges.txt")
    parser.add_argument(
        "--trajectories", type=Path, default=DEFAULT_INPUT / "beijing_trajectories.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--road-only", action="store_true")
    parser.add_argument("--force-map", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def load_pair_lookup(edges_path: Path, grouped_path: Path) -> dict[tuple[int, int], tuple[int, int, list]]:
    original_edges = mapper.read_original_edges(edges_path, False)
    edge_by_id = {
        str(edge_id): (int(start_id), int(end_id))
        for edge_id, start_id, end_id in original_edges
    }
    lookup: dict[tuple[int, int], tuple[int, int, list]] = {}
    with grouped_path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            old_edge_id = str(row["old_edge_id"])
            start_id, end_id = edge_by_id[old_edge_id]
            key = (min(start_id, end_id), max(start_id, end_id))
            if key in lookup:
                raise ValueError(f"duplicate undirected road pair: {key}")
            subedges = json.loads(row["subedges"])
            lookup[key] = (start_id, end_id, subedges)
    return lookup


def locate_unit(units: list[TimeUnit], timestamp: int) -> int | None:
    # There are only 32 units; a tiny linear scan is faster than date parsing here.
    for index, unit in enumerate(units):
        if unit.start_timestamp <= timestamp < unit.end_timestamp:
            return index
    return None


def map_all_trajectories(
    trajectories_path: Path,
    cleaned_edges_path: Path,
    output_dir: Path,
    progress_every: int,
) -> dict:
    units = build_units()
    units_dir = output_dir / "bimonthly_units"
    units_dir.mkdir(parents=True, exist_ok=True)
    summary_path = units_dir / "bimonthly_mapping_summary.json"
    if summary_path.exists():
        print("[北京轨迹映射] 已存在，跳过重复映射。")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    lookup = load_pair_lookup(cleaned_edges_path, output_dir / "subedges_grouped.csv")
    print(f"[北京轨迹映射] loaded road-pair mappings={len(lookup):,}")

    max_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_field_size)
            break
        except OverflowError:
            max_field_size //= 10

    stats = [Counter() for _ in units]
    global_stats: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    started = time.perf_counter()

    with ExitStack() as stack:
        temp_paths = [units_dir / f"traj-{unit.name}.csv.tmp" for unit in units]
        final_paths = [units_dir / f"traj-{unit.name}.csv" for unit in units]
        handles = [
            stack.enter_context(path.open("w", encoding="utf-8-sig", newline=""))
            for path in temp_paths
        ]
        writers = [csv.writer(handle) for handle in handles]
        for writer in writers:
            writer.writerow(["trajectory_id", "trajectory_data"])

        mapping_tmp = units_dir / "source_trajectory_ids.csv.tmp"
        mapping_final = units_dir / "source_trajectory_ids.csv"
        mapping_handle = stack.enter_context(
            mapping_tmp.open("w", encoding="utf-8-sig", newline="")
        )
        mapping_writer = csv.writer(mapping_handle)
        mapping_writer.writerow(["trajectory_id", "source_traj_id"])

        with trajectories_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            for trajectory_id, row in enumerate(reader):
                global_stats["source_trajectory_count"] += 1
                mapping_writer.writerow([trajectory_id, row["source_traj_id"]])
                points = json.loads(row["points_data"])
                unit_segments: dict[int, list] = defaultdict(list)

                if len(points) < 2:
                    failures["trajectory_has_fewer_than_two_points"] += 1
                    continue

                for left, right in zip(points, points[1:]):
                    start_node = int(left[0])
                    end_node = int(right[0])
                    start_ts = int(left[3])
                    end_ts = int(right[3])
                    global_stats["source_edge_interval_count"] += 1
                    if end_ts < start_ts:
                        failures["timestamp_decreased"] += 1
                        continue
                    key = (min(start_node, end_node), max(start_node, end_node))
                    road = lookup.get(key)
                    if road is None:
                        failures["road_pair_not_in_nondegenerate_network"] += 1
                        continue
                    road_start, road_end, stored_subedges = road
                    if (start_node, end_node) == (road_start, road_end):
                        subedges = stored_subedges
                        global_stats["forward_source_edge_count"] += 1
                    elif (start_node, end_node) == (road_end, road_start):
                        subedges = [
                            [int(item[0]), int(item[2]), int(item[1]), float(item[3])]
                            for item in reversed(stored_subedges)
                        ]
                        global_stats["reverse_source_edge_count"] += 1
                    else:
                        failures["road_direction_mismatch"] += 1
                        continue

                    boundaries = mapper.make_time_boundaries(
                        start_ts, end_ts, [float(item[3]) for item in subedges]
                    )
                    for sub_index, subedge in enumerate(subedges):
                        sub_start = int(boundaries[sub_index])
                        sub_end = int(boundaries[sub_index + 1])
                        first_unit = locate_unit(units, sub_start)
                        last_unit = locate_unit(units, sub_end)
                        target_units = set()
                        if first_unit is not None:
                            target_units.add(first_unit)
                        if last_unit is not None:
                            target_units.add(last_unit)
                        for unit_index in sorted(target_units):
                            unit = units[unit_index]
                            clipped_start = max(sub_start, unit.start_timestamp)
                            clipped_end = min(sub_end, unit.end_timestamp - 1)
                            if clipped_end < clipped_start:
                                continue
                            unit_segments[unit_index].append(
                                [
                                    int(subedge[0]),
                                    int(subedge[1]),
                                    clipped_start,
                                    int(subedge[2]),
                                    clipped_end,
                                ]
                            )
                            stats[unit_index]["mapped_subedge_interval_count"] += 1

                for unit_index, segments in unit_segments.items():
                    writers[unit_index].writerow(
                        [
                            trajectory_id,
                            json.dumps(segments, ensure_ascii=False, separators=(",", ":")),
                        ]
                    )
                    stats[unit_index]["trajectory_count"] += 1

                if progress_every and (trajectory_id + 1) % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    mapped = sum(item["mapped_subedge_interval_count"] for item in stats)
                    print(
                        f"[北京轨迹映射] trajectories={trajectory_id + 1:,}, "
                        f"mapped_subedges={mapped:,}, elapsed={elapsed:.1f}s"
                    )

        for handle in handles:
            handle.flush()
        mapping_handle.flush()

    for temp_path, final_path in zip(temp_paths, final_paths):
        temp_path.replace(final_path)
    mapping_tmp.replace(mapping_final)

    unit_reports = []
    for unit, path, counter in zip(units, final_paths, stats):
        unit_reports.append(
            {
                "name": unit.name,
                "first_month": unit.first_month,
                "second_month": unit.second_month,
                "start_timestamp": unit.start_timestamp,
                "end_timestamp_exclusive": unit.end_timestamp,
                "time_unit_seconds": 60,
                "time_slot_count": unit.slot_count,
                "binary_depth": (unit.slot_count - 1).bit_length(),
                "trajectory_csv": str(path.resolve()),
                **dict(counter),
            }
        )

    summary = {
        "dataset_city": "beijing",
        "source_trajectory_file": str(trajectories_path.resolve()),
        "excluded_months": {
            "2000-01": "one anomalous trajectory; outside the continuous 2007-04..2012-07 series"
        },
        "time_unit_seconds": 60,
        "unit_count": len(units),
        "global_counts": dict(global_stats),
        "failure_counts": dict(failures),
        "units": unit_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    nodes_path = args.nodes.resolve()
    edges_path = args.edges.resolve()
    trajectories_path = args.trajectories.resolve()
    output_dir = args.output_dir.resolve()
    for path in (nodes_path, edges_path, trajectories_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_map:
        summary_path = output_dir / "bimonthly_units" / "bimonthly_mapping_summary.json"
        if summary_path.exists():
            summary_path.unlink()

    started = time.perf_counter()
    cleaned_edges_path, edge_audit = prepare.prepare_non_degenerate_edges(
        nodes_path, edges_path, output_dir
    )
    prepare.build_exact_50m_network(nodes_path, cleaned_edges_path, output_dir)
    grid = prepare.write_spatial_index(nodes_path, output_dir)
    mapping = None
    if not args.road_only:
        mapping = map_all_trajectories(
            trajectories_path,
            cleaned_edges_path,
            output_dir,
            args.progress_every,
        )

    manifest = {
        "dataset_city": "beijing",
        "timezone": "UTC",
        "cell_size_m": 50.0,
        "time_unit_seconds": 60,
        "source_nodes_file": str(nodes_path),
        "source_edges_file": str(edges_path),
        "source_trajectory_file": str(trajectories_path),
        "source_edge_audit": edge_audit,
        "grid": grid,
        "mapping_summary": None
        if mapping is None
        else str((output_dir / "bimonthly_units" / "bimonthly_mapping_summary.json").resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
