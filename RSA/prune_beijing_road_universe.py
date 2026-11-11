from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "rsg_50m_beijing"


def collect_used_edges(units_dir: Path) -> tuple[set[int], dict[str, int]]:
    max_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_field_size)
            break
        except OverflowError:
            max_field_size //= 10
    all_edges: set[int] = set()
    per_unit: dict[str, int] = {}
    for path in sorted(units_dir.glob("traj-*.csv")):
        unit_edges: set[int] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                for segment in json.loads(row["trajectory_data"]):
                    unit_edges.add(int(segment[0]))
        all_edges.update(unit_edges)
        per_unit[path.stem.removeprefix("traj-")] = len(unit_edges)
        print(f"[边宇宙] {path.stem}: used_edges={len(unit_edges):,}")
    return all_edges, per_unit


def filter_csv(
    source_path: Path,
    target_path: Path,
    predicate,
) -> int:
    count = 0
    temporary = target_path.with_suffix(target_path.suffix + ".tmp")
    with source_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if predicate(row):
                writer.writerow(row)
                count += 1
    temporary.replace(target_path)
    return count


def main() -> None:
    dataset = DEFAULT_DATASET
    units_dir = dataset / "bimonthly_units"
    started = time.perf_counter()
    used_edges, per_unit = collect_used_edges(units_dir)

    edge_nodes: set[int] = set()
    new_edges_used = dataset / "new_edges_used.csv"
    temporary = new_edges_used.with_suffix(".csv.tmp")
    kept_edges = 0
    with (dataset / "new_edges.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source, temporary.open("w", encoding="utf-8-sig", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["edge_id"]) not in used_edges:
                continue
            writer.writerow(row)
            edge_nodes.add(int(row["start_node_id"]))
            edge_nodes.add(int(row["end_node_id"]))
            kept_edges += 1
    temporary.replace(new_edges_used)

    kept_nodes = filter_csv(
        dataset / "new_nodes.csv",
        dataset / "new_nodes_used.csv",
        lambda row: int(row["node_id"]) in edge_nodes,
    )
    kept_edge_cells = filter_csv(
        dataset / "edge_cells.csv",
        dataset / "edge_cells_used.csv",
        lambda row: int(row["edge_id"]) in used_edges,
    )
    kept_grouped = filter_csv(
        dataset / "subedges_grouped.csv",
        dataset / "subedges_grouped_used.csv",
        lambda row: any(int(item[0]) in used_edges for item in json.loads(row["subedges"])),
    )

    cell_ids: set[str] = set()
    with (dataset / "edge_cells_used.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        for row in csv.DictReader(source):
            cell_ids.add(row["cell_id"])

    used_path = dataset / "used_edge_ids.txt"
    used_path.write_text("".join(f"{edge_id}\n" for edge_id in sorted(used_edges)), encoding="ascii")

    query_road = dataset / "query_road"
    query_road.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset / "new_edges_used.csv", query_road / "new_edges.csv")
    shutil.copyfile(dataset / "new_nodes_used.csv", query_road / "new_nodes.csv")
    shutil.copyfile(dataset / "edge_cells_used.csv", query_road / "edge_cells.csv")
    grid_metadata = json.loads((dataset / "grid_metadata.json").read_text(encoding="utf-8"))
    grid_metadata["indexed_edges"] = len(used_edges)
    grid_metadata["nonempty_road_cells"] = len(cell_ids)
    grid_metadata["road_universe_policy"] = "five-year traversed subedge union"
    (query_road / "grid_metadata.json").write_text(
        json.dumps(grid_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "policy": "union of every 50 m subedge traversed during 2007-04..2012-07",
        "security_note": (
            "Every bimonthly ADS must include all edges in this fixed universe, including "
            "edges empty in that unit, so an untrusted server cannot hide an active edge."
        ),
        "full_edge_count": sum(1 for _ in (dataset / "new_edges.csv").open("rb")) - 1,
        "used_edge_count": len(used_edges),
        "removed_never_used_edge_count": (
            sum(1 for _ in (dataset / "new_edges.csv").open("rb")) - 1 - len(used_edges)
        ),
        "used_node_count": kept_nodes,
        "used_edge_cell_row_count": kept_edge_cells,
        "used_road_cell_count": len(cell_ids),
        "used_original_edge_count": kept_grouped,
        "per_unit_active_edge_count": per_unit,
        "files": {
            "edge_ids": str(used_path.resolve()),
            "edges": str(new_edges_used.resolve()),
            "nodes": str((dataset / "new_nodes_used.csv").resolve()),
            "edge_cells": str((dataset / "edge_cells_used.csv").resolve()),
            "grouped_subedges": str((dataset / "subedges_grouped_used.csv").resolve()),
            "query_road_directory": str(query_road.resolve()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (dataset / "road_universe_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
