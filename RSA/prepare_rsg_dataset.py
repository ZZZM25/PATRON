from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from collections import Counter
from datetime import date
from pathlib import Path

import grid_network_splitter_pycharm_grouped_subedges as splitter
import map_traj_to_new_network_undirected as mapper




def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024):
    """Incrementally decode the top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        pos = 0
        eof = False

        def refill() -> bool:
            nonlocal buffer, pos, eof
            if pos:
                buffer = buffer[pos:]
                pos = 0
            chunk = f.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        refill()
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer):
                break
            if not refill():
                raise ValueError(f"JSON 文件为空: {path}")
        if buffer[pos] != "[":
            raise ValueError(f"JSON 顶层必须是数组: {path}")
        pos += 1

        while True:
            while True:
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                    pos += 1
                if pos < len(buffer):
                    break
                if not refill():
                    raise ValueError("JSON 数组未正常结束")

            if buffer[pos] == "]":
                return

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    yield value
                    break
                except json.JSONDecodeError:
                    if eof or not refill():
                        raise


def iter_trajectory_items(path: Path):
    """Yield trajectory records from the JSON or NumPy-backed pickle sources."""
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        # The Chengdu daily source files are Python pickles containing NumPy
        # scalar values.  Loading requires the data-preparation Python runtime,
        # but the records have the same [nodes, edge_pairs, point_groups]
        # structure as the JSON files used by the mapper.
        with path.open("rb") as source:
            records = pickle.load(source)
        yield from records
        return
    yield from iter_json_array(path)


def prepare_non_degenerate_edges(nodes_path: Path, edges_path: Path, output_dir: Path) -> tuple[Path, dict]:
    cleaned_path = output_dir / "source_edges_nondegenerate.txt"
    audit_path = output_dir / "source_edge_audit.json"
    if cleaned_path.exists() and audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as f:
            return cleaned_path, json.load(f)

    nodes = splitter.read_nodes(nodes_path, False)
    edges = splitter.read_edges(edges_path, False)
    lons = [coord[0] for coord in nodes.values()]
    lats = [coord[1] for coord in nodes.values()]
    projector = splitter.Projector(sum(lons) / len(lons), sum(lats) / len(lats))
    node_xy = {
        node_id: projector.lonlat_to_xy(lon, lat)
        for node_id, (lon, lat) in nodes.items()
    }
    kept = []
    excluded = []
    for edge_id, start_id, end_id in edges:
        x0, y0 = node_xy[start_id]
        x1, y1 = node_xy[end_id]
        length_m = splitter.segment_length(x0, y0, x1, y1)
        if length_m <= 1e-8:
            excluded.append(
                {
                    "edge_id": edge_id,
                    "start_node_id": start_id,
                    "end_node_id": end_id,
                    "length_m": length_m,
                    "reason": "zero_length_edge",
                }
            )
        else:
            kept.append((edge_id, start_id, end_id))

    with cleaned_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=" ", lineterminator="\n")
        writer.writerows(kept)
    audit = {
        "input_edge_count": len(edges),
        "kept_edge_count": len(kept),
        "excluded_edge_count": len(excluded),
        "excluded_edges": excluded,
    }
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"[边审计] kept={len(kept):,}, excluded_zero_length={len(excluded):,}")
    return cleaned_path, audit


def build_exact_50m_network(nodes_path: Path, edges_path: Path, output_dir: Path) -> None:
    required = [
        output_dir / "new_nodes.csv",
        output_dir / "new_edges.csv",
        output_dir / "subedges_grouped.csv",
        output_dir / "summary.json",
    ]
    if all(path.exists() for path in required):
        print("[路网] 已存在，跳过重复切分。")
        return

    splitter.NODES_FILE = nodes_path
    splitter.EDGES_FILE = edges_path
    splitter.OUTPUT_DIR = output_dir
    splitter.CELL_SIZE_M = 50.0
    splitter.HAS_HEADER = False
    splitter.PRINT_PROGRESS_EVERY = 500
    splitter.main()


def write_spatial_index(nodes_path: Path, output_dir: Path) -> dict:
    index_path = output_dir / "edge_cells.csv"
    grid_path = output_dir / "grid_metadata.json"
    if index_path.exists() and grid_path.exists():
        print("[空间索引] 已存在，跳过重复生成。")
        with grid_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    original_nodes = splitter.read_nodes(nodes_path, False)
    lons = [coord[0] for coord in original_nodes.values()]
    lats = [coord[1] for coord in original_nodes.values()]
    projector = splitter.Projector(sum(lons) / len(lons), sum(lats) / len(lats))
    original_xy = [projector.lonlat_to_xy(lon, lat) for lon, lat in original_nodes.values()]
    grid = splitter.build_grid(original_xy, 50.0)

    new_nodes: dict[int, tuple[float, float]] = {}
    with (output_dir / "new_nodes.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            new_nodes[int(row["node_id"])] = projector.lonlat_to_xy(
                float(row["lon"]), float(row["lat"])
            )

    cell_counts: Counter[str] = Counter()
    edge_count = 0
    with (output_dir / "new_edges.csv").open("r", encoding="utf-8-sig", newline="") as src, index_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.writer(dst)
        writer.writerow(["edge_id", "cell_id", "cell_col", "cell_row"])
        for row in reader:
            edge_id = int(row["edge_id"])
            x0, y0 = new_nodes[int(row["start_node_id"])]
            x1, y1 = new_nodes[int(row["end_node_id"])]
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            col = splitter.point_to_index(mx, grid["origin_x"], 50.0, grid["nx"] - 1)
            row_idx = splitter.point_to_index(my, grid["origin_y"], 50.0, grid["ny"] - 1)
            cell_id = f"{col}:{row_idx}"
            writer.writerow([edge_id, cell_id, col, row_idx])
            cell_counts[cell_id] += 1
            edge_count += 1

    metadata = {
        "projection": "local_equirectangular",
        "earth_radius_m": splitter.EARTH_RADIUS_M,
        "lon0": projector.lon0,
        "lat0": projector.lat0,
        "origin_x": grid["origin_x"],
        "origin_y": grid["origin_y"],
        "cell_size_m": 50.0,
        "grid_cols": grid["nx"],
        "grid_rows": grid["ny"],
        "indexed_edges": edge_count,
        "nonempty_road_cells": len(cell_counts),
    }
    with grid_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[空间索引] edge_cells={edge_count:,}, road_cells={len(cell_counts):,}")
    return metadata


def map_trajectories_streaming(
    edges_path: Path,
    traj_path: Path,
    output_dir: Path,
    output_name: str = "traj-10-10.csv",
    failed_name: str = "mapped_trajectories_failed.csv",
    summary_name: str = "trajectory_mapping_summary.json",
) -> dict:
    for name, label in (
        (output_name, "output_name"),
        (failed_name, "failed_name"),
        (summary_name, "summary_name"),
    ):
        if Path(name).name != name:
            raise ValueError(f"{label} 必须是文件名，不能包含目录")
    output_path = output_dir / output_name
    failed_path = output_dir / failed_name
    summary_path = output_dir / summary_name
    if output_path.exists() and summary_path.exists():
        print("[轨迹映射] 已存在，跳过重复映射。")
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    original_edges = mapper.read_original_edges(edges_path, False)
    pair_to_edges, old_edge_direction = mapper.build_undirected_pair_to_old_edges(original_edges)
    old_edge_to_subedges = mapper.read_grouped_subedges(output_dir / "subedges_grouped.csv")
    new_edges = mapper.read_new_edges(output_dir / "new_edges.csv")
    new_nodes = mapper.read_new_nodes(output_dir / "new_nodes.csv")

    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_failed = failed_path.with_suffix(failed_path.suffix + ".tmp")
    direction_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    failed_counter: Counter[str] = Counter()
    trajectory_count = 0
    mapped_segment_count = 0
    failed_count = 0
    t0 = time.perf_counter()

    with tmp_output.open("w", encoding="utf-8-sig", newline="") as out_f, tmp_failed.open(
        "w", encoding="utf-8-sig", newline=""
    ) as failed_f:
        out_writer = csv.writer(out_f)
        fail_writer = csv.writer(failed_f)
        out_writer.writerow(["trajectory_id", "trajectory_data"])
        fail_writer.writerow(
            ["trajectory_id", "edge_index", "old_start_node_id", "old_end_node_id", "reason"]
        )

        for trajectory_id, traj_item in enumerate(iter_trajectory_items(traj_path)):
            trajectory_count += 1
            mapped_list = []

            if not isinstance(traj_item, (list, tuple)) or len(traj_item) < 3:
                reason = "bad_trajectory_format"
                fail_writer.writerow([trajectory_id, "", "", "", reason])
                failed_counter[reason] += 1
                failed_count += 1
                out_writer.writerow([trajectory_id, "[]"])
                continue

            edge_pairs = traj_item[1]
            edge_point_groups = traj_item[2]
            if len(edge_pairs) != len(edge_point_groups):
                reason = (
                    f"edge_pairs_count_{len(edge_pairs)}_not_equal_"
                    f"point_groups_count_{len(edge_point_groups)}"
                )
                fail_writer.writerow([trajectory_id, "", "", "", reason])
                failed_counter[mapper.reason_prefix(reason)] += 1
                failed_count += 1
                out_writer.writerow([trajectory_id, "[]"])
                continue

            for edge_index, pair in enumerate(edge_pairs):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    reason = "bad_edge_pair_format"
                    fail_writer.writerow([trajectory_id, edge_index, "", "", reason])
                    failed_counter[reason] += 1
                    failed_count += 1
                    continue

                traj_start = str(pair[0])
                traj_end = str(pair[1])
                points_on_edge = edge_point_groups[edge_index]
                if not points_on_edge:
                    reason = "empty_points_on_edge"
                    fail_writer.writerow([trajectory_id, edge_index, traj_start, traj_end, reason])
                    failed_counter[reason] += 1
                    failed_count += 1
                    continue

                try:
                    start_ts = mapper.get_edge_start_time(points_on_edge)
                    end_ts = mapper.get_edge_end_time(edge_point_groups, edge_index)
                    old_edge_id, issue = mapper.choose_old_edge_undirected(
                        pair_to_edges, old_edge_to_subedges, traj_start, traj_end
                    )
                    if old_edge_id is None:
                        raise LookupError(issue)
                    if issue:
                        issue_counter[mapper.reason_prefix(issue)] += 1

                    reverse_needed = mapper.need_reverse_subedges(
                        old_edge_id, old_edge_direction, traj_start, traj_end
                    )
                    if reverse_needed:
                        direction_counter["reverse"] += 1
                        subedges = mapper.reverse_subedges(old_edge_to_subedges[old_edge_id])
                    else:
                        direction_counter["forward"] += 1
                        subedges = old_edge_to_subedges[old_edge_id]

                    boundaries = mapper.make_time_boundaries(
                        start_ts, end_ts, [subedge[3] for subedge in subedges]
                    )
                    for k, subedge in enumerate(subedges):
                        new_edge_id = int(subedge[0])
                        new_start_node_id = int(subedge[1])
                        new_end_node_id = int(subedge[2])
                        mapper.validate_new_edge_undirected(
                            new_edge_id,
                            new_start_node_id,
                            new_end_node_id,
                            new_edges,
                            new_nodes,
                        )
                        mapped_list.append(
                            [
                                new_edge_id,
                                new_start_node_id,
                                int(boundaries[k]),
                                new_end_node_id,
                                int(boundaries[k + 1]),
                            ]
                        )
                        mapped_segment_count += 1
                except Exception as exc:
                    reason = f"mapping_error:{type(exc).__name__}:{exc}"
                    fail_writer.writerow([trajectory_id, edge_index, traj_start, traj_end, reason])
                    failed_counter[mapper.reason_prefix(reason)] += 1
                    failed_count += 1

            out_writer.writerow(
                [trajectory_id, json.dumps(mapped_list, ensure_ascii=False, separators=(",", ":"))]
            )
            if trajectory_count % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[轨迹映射] trajectories={trajectory_count:,}, "
                    f"segments={mapped_segment_count:,}, elapsed={elapsed:.1f}s"
                )

    tmp_output.replace(output_path)
    tmp_failed.replace(failed_path)
    summary = {
        "source_trajectory_file": str(traj_path),
        "trajectory_count": trajectory_count,
        "mapped_segment_count": mapped_segment_count,
        "failed_edge_count": failed_count,
        "direction_counts": dict(direction_counter),
        "mapping_issue_counts": dict(issue_counter),
        "failure_counts": dict(failed_counter),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"[轨迹映射完成] trajectories={trajectory_count:,}, "
        f"segments={mapped_segment_count:,}, failures={failed_count:,}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a 50 m RSG road network and mapped trajectory dataset"
    )
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--dataset-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    nodes_path = args.nodes.resolve()
    edges_path = args.edges.resolve()
    traj_path = args.trajectories.resolve()
    for path in (nodes_path, edges_path, traj_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cleaned_edges_path, edge_audit = prepare_non_degenerate_edges(
        nodes_path, edges_path, output_dir
    )
    build_exact_50m_network(nodes_path, cleaned_edges_path, output_dir)
    grid_metadata = write_spatial_index(nodes_path, output_dir)
    mapping_summary = map_trajectories_streaming(
        cleaned_edges_path, traj_path, output_dir
    )
    manifest = {
        "dataset_date": date.fromisoformat(args.dataset_date).isoformat(),
        "timezone": "UTC",
        "cell_size_m": 50.0,
        "source_nodes": str(nodes_path),
        "source_edges": str(edges_path),
        "source_trajectories": str(traj_path),
        "output_dir": str(output_dir),
        "source_edge_audit": edge_audit,
        "grid": grid_metadata,
        "mapping": mapping_summary,
        "total_elapsed_seconds": time.perf_counter() - t0,
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[数据准备完成] {output_dir}")


if __name__ == "__main__":
    main()
