"""Isolated Cell50 complete-index construction scaling benchmark.

Each city/percentage runs in a fresh process and selects the first N valid
trajectories in natural source-file/source-record order.  The final physical
index is complete but sharded into three files: manifest, edge records and
materialized per-edge SMT nodes.  Temporary trajectory/catalog files are not
counted as index bytes.

The primary construction timer follows the original benchmark policy: it
includes index ingestion, ordering, authenticated time chains, full EdgeSMT
construction and spatial authentication, while excluding source parsing,
Cell50 transformation, validation and serialization/persistence.  A separate
physical wall-clock metric is retained for reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

from baseline.cell_dataset_builder import (
    _load_beijing_points,
    _set_large_csv_field_limit,
    parse_cell_partitioned_beijing_points,
)
from baseline.cell_partition import grid_from_road, split_road_network
from baseline.config import load_config, resolve_path
from baseline.disk_smt import (
    ShardedEntrySpool,
    build_disk_smt_baseline,
    spool_cell_trajectories,
)
from baseline.persistence import save_index
from baseline.road_network import RoadNetwork
from baseline.trajectory_parser import collect_files, load_relaxed_json


ROOT = Path(__file__).resolve().parent
DEFAULT_JOBS = (
    ("chengdu", ROOT / "configs/chengdu_cell50_edge_smt.json"),
    ("xian", ROOT / "configs/xian_cell50_edge_smt.json"),
    ("beijing", ROOT / "configs/beijing_cell50_edge_smt.json"),
)
DEFAULT_PERCENTAGES = (20, 40, 60, 80, 100)
FORMAT_VERSION = "cell50-complete-disk-smt-v1"


def _required_path(cfg: dict, key: str) -> Path:
    if key not in cfg:
        raise ValueError(f"Cell50 config missing {key}")
    return resolve_path(cfg, cfg[key])


def _build_partition(cfg: dict):
    source_road = RoadNetwork()
    source_road.load(
        str(_required_path(cfg, "source_node_file")),
        str(_required_path(cfg, "source_edge_file")),
    )
    cell_cfg = cfg.get("cell_partition")
    if not isinstance(cell_cfg, dict):
        raise ValueError("cell_partition object is required")
    cell_size = float(cell_cfg.get("size_m", 50.0))
    grid = grid_from_road(
        source_road,
        cell_width_m=float(cell_cfg.get("width_m", cell_size)),
        cell_height_m=float(cell_cfg.get("height_m", cell_size)),
        reference_lon=(
            float(cell_cfg["reference_lon"])
            if "reference_lon" in cell_cfg else None
        ),
        reference_lat=(
            float(cell_cfg["reference_lat"])
            if "reference_lat" in cell_cfg else None
        ),
        origin_x_m=float(cell_cfg.get("origin_x_m", 0.0)),
        origin_y_m=float(cell_cfg.get("origin_y_m", 0.0)),
    )
    return source_road, split_road_network(source_road, grid), grid


def _metrics_total(cfg: dict) -> int | None:
    raw_path = cfg.get("build_metrics_file")
    if not raw_path:
        return None
    path = resolve_path(cfg, raw_path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("trajectory_count")
    return int(value) if value is not None and int(value) > 0 else None


def _count_source_trajectories(city: str, cfg: dict, source_road, partition) -> int:
    """Fallback when an existing full-build metrics file is unavailable."""
    files = collect_files(cfg["trajectory_globs"], Path(cfg["_config_dir"]))
    if city == "beijing":
        if len(files) != 1:
            raise ValueError(f"expected one Beijing CSV, got {len(files)}")
        _set_large_csv_field_limit()
        total = 0
        with files[0].open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row_no, row in enumerate(reader, 2):
                points = _load_beijing_points(row["points_data"], row_no)
                parsed, _, _ = parse_cell_partitioned_beijing_points(
                    points, row_no, source_road, partition
                )
                total += parsed is not None
        return total
    total = 0
    for path in files:
        data = load_relaxed_json(path)
        if not isinstance(data, list):
            raise ValueError(f"top-level JSON must be list: {path}")
        total += len(data)
    return total


def build_worker(
    city: str,
    config_path: Path,
    percentage: int,
    cell_output: Path,
    repeat: int,
) -> None:
    cfg = load_config(str(config_path))
    source_road, partition, grid = _build_partition(cfg)
    total = _metrics_total(cfg)
    total_source = "existing full-build metrics"
    if total is None:
        total = _count_source_trajectories(city, cfg, source_road, partition)
        total_source = "source scan"
    target = total if percentage == 100 else max(
        1, math.floor(total * percentage / 100)
    )

    storage_cfg = cfg.get("disk_storage", {})
    shard_count = int(storage_cfg.get("construction_shards", 512))
    block_cache_bytes = int(
        float(storage_cfg.get("edge_block_cache_mib", 512)) * 1024 * 1024
    )
    smt_cache_bytes = int(
        float(storage_cfg.get("edge_smt_cache_mib", 1024)) * 1024 * 1024
    )
    theta = int(cfg.get("theta", 64))

    cell_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{city}_{percentage:03d}_", dir=cell_output.parent
    ) as temporary_name:
        work = Path(temporary_name)
        spool = ShardedEntrySpool(work / "spool", shard_count=shard_count)
        spool_wall_started = perf_counter()
        spool_stats = spool_cell_trajectories(
            city=city,
            source_road=source_road,
            partition=partition,
            trajectory_globs=list(cfg["trajectory_globs"]),
            config_dir=Path(cfg["_config_dir"]),
            spool=spool,
            catalog_file=work / "trajectory_catalog.dat",
            expected_file_count=cfg.get("expected_trajectory_file_count"),
            max_trajectories=target,
        )
        spool_wall_seconds = perf_counter() - spool_wall_started
        selected = int(spool_stats["trajectory_count"])
        if selected != target:
            raise RuntimeError(f"selected {selected}, expected {target}")

        physical_started = perf_counter()
        edge_data_file = work / "edge_data.bin"
        smt_data_file = work / "edge_smt_nodes.bin"
        index, build_stats = build_disk_smt_baseline(
            road=partition.road,
            spool=spool,
            edge_data_file=edge_data_file,
            smt_data_file=smt_data_file,
            theta=theta,
            trajectory_files=spool_stats["files"],
            block_cache_bytes=block_cache_bytes,
            smt_cache_bytes=smt_cache_bytes,
        )
        manifest_file = work / "index_manifest.dat"
        manifest_save_started = perf_counter()
        manifest_size = save_index(index, manifest_file)
        manifest_save_seconds = perf_counter() - manifest_save_started
        physical_index_build_seconds = (
            perf_counter() - physical_started + spool_wall_seconds
        )

        edge_data_size = int(build_stats["edge_data_size_bytes"])
        smt_data_size = int(build_stats["smt_data_size_bytes"])
        index_size = int(manifest_size) + edge_data_size + smt_data_size
        if index_size != sum(
            path.stat().st_size
            for path in (manifest_file, edge_data_file, smt_data_file)
        ):
            raise RuntimeError("complete index size accounting mismatch")

        # Loading one persisted tree proves that the recorded offsets point to
        # complete, root-consistent SMT objects.  This validation is untimed.
        populated = next(
            (eid for eid, n in enumerate(index.store._unique_counts) if n), None
        )
        if populated is not None:
            loaded_smt = index._edge_smt(populated)
            if loaded_smt.root_hash != index.edge_smt_roots[populated]:
                raise RuntimeError("persisted SMT validation failed")

        smt_persistence_seconds = float(build_stats["smt_persistence_seconds"])
        authenticated_edge_build_seconds = max(
            0.0, float(build_stats["edge_build_seconds"]) - smt_persistence_seconds
        )
        ingestion_seconds = float(spool_stats["spool_append_seconds"])
        spatial_seconds = float(build_stats["spatial_tree_seconds"])
        construction_seconds = (
            ingestion_seconds + authenticated_edge_build_seconds + spatial_seconds
        )
        result = {
            "success": True,
            "format_version": FORMAT_VERSION,
            "city": city,
            "percentage": percentage,
            "repeat": repeat,
            "selection_policy": "first N valid trajectories in natural source/record order",
            "total_trajectory_count": total,
            "total_trajectory_count_source": total_source,
            "selected_trajectory_count": selected,
            "source_trajectory_segment_count": spool_stats["source_segment_count"],
            "partitioned_trajectory_segment_count": spool_stats["partitioned_segment_count"],
            "edge_record_count": build_stats["record_count"],
            "smt_leaf_count": sum(index.store._unique_counts),
            "populated_edge_count": build_stats["populated_edge_count"],
            "source_edge_count": len(source_road.valid_edge_ids()),
            "partitioned_edge_count": len(partition.road.valid_edge_ids()),
            "theta": theta,
            "grid": grid.as_dict(),
            "index_ingestion_seconds": ingestion_seconds,
            "authenticated_edge_build_seconds": authenticated_edge_build_seconds,
            "spatial_tree_build_seconds": spatial_seconds,
            "index_construction_seconds": construction_seconds,
            "trajectory_transform_and_spool_wall_seconds": spool_wall_seconds,
            "smt_node_persistence_seconds": smt_persistence_seconds,
            "manifest_save_seconds": manifest_save_seconds,
            "physical_index_build_wall_seconds": physical_index_build_seconds,
            "index_manifest_size_bytes": manifest_size,
            "edge_data_size_bytes": edge_data_size,
            "smt_data_size_bytes": smt_data_size,
            "index_size_bytes": index_size,
            "index_size_mib": index_size / 1024 / 1024,
            "trusted_root_hex": index.root_hash.hex(),
            "construction_shards": shard_count,
            "complete_materialized_edge_smt_nodes_persisted": True,
            "excluded_from_primary_construction_time": [
                "road-network loading and Cell50 road partitioning",
                "trajectory-file loading and parsing",
                "trajectory Cell50 transformation and catalog writing",
                "source counting and subset selection",
                "validation",
                "SMT-node serialization and persistence",
                "manifest serialization and saving",
                "result writing",
            ],
        }

    cell_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del index
    gc.collect()


def aggregate(output_dir: Path, raw_rows: list[dict], repeats: int) -> None:
    timing_fields = (
        "index_ingestion_seconds",
        "authenticated_edge_build_seconds",
        "spatial_tree_build_seconds",
        "index_construction_seconds",
        "physical_index_build_wall_seconds",
    )
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in raw_rows:
        groups.setdefault((row["city"], int(row["percentage"])), []).append(row)
    rows = []
    for (_city, _percentage), runs in sorted(groups.items()):
        runs.sort(key=lambda row: int(row["repeat"]))
        if len(runs) != repeats:
            continue
        row = dict(runs[0])
        row["repeats"] = repeats
        row["primary_statistic"] = "arithmetic_mean"
        for field in timing_fields:
            samples = [float(run[field]) for run in runs]
            row[field] = statistics.fmean(samples)
            row[field + "_runs"] = samples
        rows.append(row)
    fields = [
        "city", "percentage", "total_trajectory_count",
        "selected_trajectory_count", "partitioned_trajectory_segment_count",
        "smt_leaf_count", "populated_edge_count", "index_ingestion_seconds",
        "authenticated_edge_build_seconds", "spatial_tree_build_seconds",
        "index_construction_seconds", "physical_index_build_wall_seconds",
        "index_manifest_size_bytes", "edge_data_size_bytes", "smt_data_size_bytes",
        "index_size_bytes", "index_size_mib", "trusted_root_hex",
    ]
    with (output_dir / "index_scaling_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    by_key = {(row["city"], int(row["percentage"])): row for row in rows}
    for filename, metric in (
        ("construction_time_matrix_s.csv", "index_construction_seconds"),
        ("index_size_matrix_mib.csv", "index_size_mib"),
    ):
        with (output_dir / filename).open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["city", *[f"{p}%" for p in DEFAULT_PERCENTAGES]])
            for city, _ in DEFAULT_JOBS:
                writer.writerow([
                    city,
                    *[
                        by_key[(city, p)][metric]
                        if (city, p) in by_key else ""
                        for p in DEFAULT_PERCENTAGES
                    ],
                ])
    (output_dir / "index_scaling_results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repeats": repeats,
                "primary_statistic": "arithmetic_mean",
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _parse_csv_set(raw: str, allowed: set[str], label: str) -> set[str]:
    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = values - allowed
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(sorted(unknown))}")
    return values


def run_parent(
    output_dir: Path,
    force: bool,
    selected_cities: set[str],
    selected_percentages: set[int],
    repeats: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for city, config_path in DEFAULT_JOBS:
        if city not in selected_cities:
            continue
        for percentage in DEFAULT_PERCENTAGES:
            if percentage not in selected_percentages:
                continue
            for repeat in range(1, repeats + 1):
                cell = output_dir / "cells" / f"{city}_{percentage:03d}_r{repeat}.json"
                if cell.is_file() and not force:
                    result = json.loads(cell.read_text(encoding="utf-8"))
                    if (
                        result.get("success")
                        and result.get("format_version") == FORMAT_VERSION
                        and result.get("city") == city
                        and int(result.get("percentage", -1)) == percentage
                        and int(result.get("repeat", -1)) == repeat
                    ):
                        print(f"RESUME {city} {percentage}% r{repeat}", flush=True)
                        rows.append(result)
                        continue
                print(f"START {city} {percentage}% r{repeat}/{repeats}", flush=True)
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--city", city, "--config", str(config_path),
                    "--percentage", str(percentage), "--cell-output", str(cell),
                    "--repeat", str(repeat),
                ]
                completed = subprocess.run(command, cwd=ROOT)
                if completed.returncode != 0:
                    raise SystemExit(completed.returncode)
                rows.append(json.loads(cell.read_text(encoding="utf-8")))
                aggregate(output_dir, rows, repeats)
                print(f"DONE {city} {percentage}% r{repeat}", flush=True)
    aggregate(output_dir, rows, repeats)
    print(f"ALL DONE: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "build_scaling_experiment",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cities", default="chengdu,xian,beijing")
    parser.add_argument("--percentages", default="20,40,60,80,100")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--city", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--percentage", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cell-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not all((args.city, args.config, args.percentage, args.cell_output, args.repeat)):
            parser.error("worker arguments are incomplete")
        if args.percentage not in DEFAULT_PERCENTAGES:
            parser.error("percentage must be one of 20,40,60,80,100")
        build_worker(
            args.city.lower(), args.config.resolve(), args.percentage,
            args.cell_output.resolve(), args.repeat,
        )
        return

    cities = _parse_csv_set(
        args.cities, {city for city, _ in DEFAULT_JOBS}, "city"
    )
    percentage_strings = _parse_csv_set(
        args.percentages, {str(p) for p in DEFAULT_PERCENTAGES}, "percentage"
    )
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    run_parent(
        args.output_dir.resolve(), args.force, cities,
        {int(value) for value in percentage_strings}, args.repeats,
    )


if __name__ == "__main__":
    main()
