from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import multiprocessing as mp
import re
import shutil
import statistics
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from build_rsg_index import validate_binary


ROOT = Path(__file__).resolve().parent
BUILDER_MODULE = "rsa_index_builder"
PERCENTAGES = (20, 40, 60, 80, 100)
FIRST_TS = re.compile(r"^\s*\[\[\s*-?\d+\s*,\s*-?\d+\s*,\s*(-?\d+)")
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure pure RSA ADS construction time and index size at 20%-100% input volume"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of formal builds per measurement (default: 3)",
    )
    parser.add_argument(
        "--city",
        action="append",
        choices=("chengdu", "xian", "beijing"),
        help="repeatable; default runs all three datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "rsa_build_scaling_experiment",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--keep-indexes", action="store_true")
    return parser.parse_args()


def first_timestamp(payload: str) -> int:
    match = FIRST_TS.match(payload)
    if match is None:
        value = json.loads(payload)
        if not value or not value[0] or len(value[0]) < 3:
            raise ValueError("trajectory_data does not contain a first timestamp")
        return int(value[0][2])
    return int(match.group(1))


def scan_csv(path: Path) -> dict:
    count = 0
    minimum = None
    maximum = None
    day_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "trajectory_data" not in reader.fieldnames:
            raise ValueError(f"unexpected trajectory CSV header: {path}")
        for row in reader:
            ts = first_timestamp(row["trajectory_data"])
            count += 1
            minimum = ts if minimum is None else min(minimum, ts)
            maximum = ts if maximum is None else max(maximum, ts)
            day = datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
            day_counts[day] = day_counts.get(day, 0) + 1
    return {
        "path": str(path.resolve()),
        "row_count": count,
        "min_timestamp": minimum,
        "max_timestamp": maximum,
        "day_counts": day_counts,
    }


def discover_datasets() -> dict[str, dict]:
    chengdu_dir = ROOT / "rsg_50m_2018-10-10"
    xian_dir = ROOT / "rsg_50m_xian_2018-10-01-roadtxt"
    beijing_dir = ROOT / "rsg_50m_beijing"
    mapping = json.loads(
        (beijing_dir / "bimonthly_units" / "bimonthly_mapping_summary.json").read_text(
            encoding="utf-8"
        )
    )

    def daily(city: str, display: str, dataset: Path) -> dict:
        units = []
        for path in sorted(dataset.glob("traj-??-??.csv")):
            units.append({"name": path.stem.removeprefix("traj-"), "csv": path, "mode": "daily"})
        return {
            "city": city,
            "display": display,
            "dataset": dataset,
            "edge_csv": dataset / "new_edges.csv",
            "units": units,
        }

    beijing_units = []
    for item in mapping["units"]:
        beijing_units.append(
            {
                "name": item["name"],
                "csv": Path(item["trajectory_csv"]),
                "mode": "bimonthly",
                "first_month": item["first_month"],
                "start_timestamp": int(item["start_timestamp"]),
                "end_timestamp_exclusive": int(item["end_timestamp_exclusive"]),
                "time_unit_seconds": int(item["time_unit_seconds"]),
                "time_slot_count": int(item["time_slot_count"]),
            }
        )

    return {
        "chengdu": daily("chengdu", "成都", chengdu_dir),
        "xian": daily("xian", "西安", xian_dir),
        "beijing": {
            "city": "beijing",
            "display": "北京",
            "dataset": beijing_dir,
            "edge_csv": beijing_dir / "new_edges_used.csv",
            "units": beijing_units,
        },
    }


def select_prefix(units: list[dict], target: int) -> list[dict]:
    remaining = target
    selected = []
    for unit in units:
        if remaining <= 0:
            break
        take = min(remaining, int(unit["scan"]["row_count"]))
        selected.append({"unit": unit, "selected_rows": take})
        remaining -= take
    if remaining:
        raise ValueError(f"prefix target exceeds dataset by {remaining} rows")
    return selected


def boundary_details(specs: list[dict]) -> dict:
    final = specs[-1]
    unit = final["unit"]
    selected_rows = int(final["selected_rows"])
    rows = []
    with Path(unit["csv"]).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for sequence, row in enumerate(reader):
            rows.append((first_timestamp(row["trajectory_data"]), sequence))
    rows.sort()
    cutoff_ts = rows[selected_rows - 1][0]
    cutoff_day = datetime.fromtimestamp(cutoff_ts, timezone.utc).date().isoformat()
    included_on_day = sum(
        1
        for ts, _ in rows[:selected_rows]
        if datetime.fromtimestamp(ts, timezone.utc).date().isoformat() == cutoff_day
    )
    total_on_day = int(unit["scan"]["day_counts"][cutoff_day])
    return {
        "cutoff_unit": unit["name"],
        "cutoff_timestamp": cutoff_ts,
        "cutoff_day_utc": cutoff_day,
        "cutoff_day_selected_trajectories": included_on_day,
        "cutoff_day_total_trajectories": total_on_day,
        "partial_cutoff_day": included_on_day < total_on_day,
    }


def write_partial_csv(source: Path, target: Path, selected_rows: int) -> None:
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(source)
        for sequence, row in enumerate(reader):
            rows.append((first_timestamp(row["trajectory_data"]), sequence, row))
    rows.sort(key=lambda item: (item[0], item[1]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, _, row in rows[:selected_rows]:
            writer.writerow(row)


def find_token_key(config: dict) -> bytes:
    dataset = Path(config["dataset"])
    if config["city"] == "beijing":
        path = dataset / "beijing.owner-token-key.bin"
    else:
        matches = sorted(dataset.glob("*.owner-token-key.bin"))
        if not matches:
            raise FileNotFoundError(f"owner token key under {dataset}")
        path = matches[0]
    value = path.read_bytes()
    if len(value) < 32:
        raise ValueError(f"owner token key is too short: {path}")
    return value


def configure_common(builder, config: dict, input_csv: Path, output_bin: Path, workers: int) -> None:
    builder.TRAJ_CSV_FILES = [str(input_csv.resolve())]
    builder.OUTPUT_BIN = str(output_bin.resolve())
    builder.TIMEZONE = "UTC"
    builder.TZ_OFFSET_SECONDS = 0
    builder.BASE = 2
    builder.NUM_WORKERS = max(1, int(workers))
    builder.CHUNK_SIZE = 32
    builder.INCLUDE_EMPTY_EDGES = True
    builder.EDGE_CSV = str(Path(config["edge_csv"]).resolve())
    builder.PRINT_TIMING = False
    builder.PRINT_PROGRESS_EVERY_TRAJ = 1_000_000_000
    builder.FUSE_PREPARE_AND_ACCUMULATOR = True
    builder.PRECOMPUTE_MEMBER_WITNESSES = True
    builder.BINARY_VERSION = 3
    builder.OPAQUE_TOKEN_BYTES = 16
    builder.OPAQUE_TOKEN_KEY = find_token_key(config)
    builder.CHECK_AFTER_WRITE = False


def configure_unit(
    builder,
    original_local_day,
    config: dict,
    unit: dict,
    input_csv: Path,
    output_bin: Path,
    workers: int,
) -> None:
    configure_common(builder, config, input_csv, output_bin, workers)
    if unit["mode"] == "daily":
        builder.DAY_SECONDS = 86400
        builder.REAL_MAX_SECOND = 86399
        builder.DEPTH = builder.calc_depth(builder.DAY_SECONDS, builder.BASE)
        builder.local_day_and_second = original_local_day
        builder.FUSED_CHUNK_SIZE = 1
        return

    origin = int(unit["start_timestamp"])
    end = int(unit["end_timestamp_exclusive"])
    slot_seconds = int(unit["time_unit_seconds"])
    slot_count = int(unit["time_slot_count"])
    bucket_day_idx = origin // 86400

    def continuous_unit_and_slot(timestamp: int, _cache: dict):
        timestamp = int(timestamp)
        if not origin <= timestamp < end:
            raise ValueError(f"timestamp {timestamp} outside [{origin}, {end}) for {unit['name']}")
        slot = (timestamp - origin) // slot_seconds
        return bucket_day_idx, str(unit["first_month"]) + "-01", slot

    builder.DAY_SECONDS = slot_count
    builder.REAL_MAX_SECOND = slot_count - 1
    builder.DEPTH = builder.calc_depth(slot_count, builder.BASE)
    builder.local_day_and_second = continuous_unit_and_slot
    builder.FUSED_CHUNK_SIZE = 8


def cache_key(city: str, unit: dict, selected_rows: int, repeat: int = 1) -> str:
    return f"{city}|{unit['name']}|{selected_rows}|repeat={repeat}"


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_build(
    builder,
    original_local_day,
    config: dict,
    unit: dict,
    selected_rows: int,
    output_dir: Path,
    workers: int,
    keep_indexes: bool,
) -> dict:
    total_rows = int(unit["scan"]["row_count"])
    artifact = f"{config['city']}-{unit['name']}-{selected_rows}of{total_rows}"
    temp_dir = output_dir / "temporary"
    logs_dir = output_dir / "logs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    source_csv = Path(unit["csv"])
    input_csv = source_csv
    partial_csv = None
    if selected_rows != total_rows:
        partial_csv = temp_dir / f"{artifact}.csv"
        write_partial_csv(source_csv, partial_csv, selected_rows)
        input_csv = partial_csv
    output_bin = temp_dir / f"{artifact}.bin"
    log_path = logs_dir / f"{artifact}.log"
    configure_unit(
        builder,
        original_local_day,
        config,
        unit,
        input_csv,
        output_bin,
        workers,
    )
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        timings = builder.main()
    validation = validate_binary(output_bin, builder)
    result = {
        "city": config["city"],
        "unit": unit["name"],
        "selected_rows": selected_rows,
        "unit_rows": total_rows,
        "is_full_unit": selected_rows == total_rows,
        "pure_build_seconds": float(timings["accumulator_build_time_all_in"]),
        "index_size_bytes": int(output_bin.stat().st_size),
        "trajectory_load_seconds_excluded": float(timings["trajectory_process_time"]),
        "write_seconds_excluded": float(timings["write_bin_time"]),
        "validation": validation,
        "log": str(log_path.resolve()),
    }
    if partial_csv is not None:
        partial_csv.unlink(missing_ok=True)
    if not keep_indexes:
        output_bin.unlink(missing_ok=True)
    return result


def write_results(output_dir: Path, report: dict) -> None:
    save_json(output_dir / "build_scaling_results.json", report)
    with (output_dir / "build_scaling_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "dataset",
            "percentage",
            "selected_trajectories",
            "total_trajectories",
            "cutoff_day_utc",
            "cutoff_day_selected_trajectories",
            "cutoff_day_total_trajectories",
            "partial_cutoff_day",
            "pure_build_seconds",
            "index_size_bytes",
            "index_size_mb",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.get("results", []):
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = discover_datasets()
    selected_cities = args.city or ["chengdu", "xian", "beijing"]

    inventory_path = output_dir / "dataset_inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        inventory = {}

    plans: dict[str, dict] = {}
    for city in selected_cities:
        config = datasets[city]
        for unit in config["units"]:
            path_key = str(Path(unit["csv"]).resolve())
            source_stat = Path(unit["csv"]).stat()
            cached = inventory.get(path_key)
            if (
                cached is None
                or int(cached.get("source_size", -1)) != source_stat.st_size
                or int(cached.get("source_mtime_ns", -1)) != source_stat.st_mtime_ns
            ):
                cached = scan_csv(Path(unit["csv"]))
                cached["source_size"] = source_stat.st_size
                cached["source_mtime_ns"] = source_stat.st_mtime_ns
                inventory[path_key] = cached
                save_json(inventory_path, inventory)
            unit["scan"] = cached
        config["units"].sort(key=lambda item: int(item["scan"]["min_timestamp"]))
        total = sum(int(unit["scan"]["row_count"]) for unit in config["units"])
        percentages = {}
        required: dict[str, dict] = {}
        for percentage in PERCENTAGES:
            target = total if percentage == 100 else math.floor(total * percentage / 100)
            target = max(1, target)
            specs = select_prefix(config["units"], target)
            details = boundary_details(specs)
            percentages[str(percentage)] = {
                "target_rows": target,
                "specs": [
                    {"unit": spec["unit"]["name"], "selected_rows": spec["selected_rows"]}
                    for spec in specs
                ],
                **details,
            }
            for spec in specs:
                key = f"{city}|{spec['unit']['name']}|{int(spec['selected_rows'])}"
                required[key] = spec
        plans[city] = {
            "total_rows": total,
            "percentages": percentages,
            "required": required,
        }

    plan_export = {
        city: {
            "total_trajectories": plan["total_rows"],
            "percentages": plan["percentages"],
            "unique_build_count": len(plan["required"]),
        }
        for city, plan in plans.items()
    }
    save_json(output_dir / "build_scaling_plan.json", plan_export)
    if args.plan_only:
        print(json.dumps(plan_export, ensure_ascii=False, indent=2))
        return

    metrics_path = output_dir / "unit_build_metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    builder = importlib.import_module(BUILDER_MODULE)
    original_local_day = builder.local_day_and_second

    for city in selected_cities:
        config = datasets[city]
        required_items = list(plans[city]["required"].items())
        for position, (_plan_key, spec) in enumerate(required_items, start=1):
            unit = spec["unit"]
            selected_rows = int(spec["selected_rows"])
            for repeat in range(1, args.repeats + 1):
                key = cache_key(city, unit, selected_rows, repeat)
                if key in metrics:
                    continue
                print(
                    f"[{config['display']}] build {position}/{len(required_items)} "
                    f"repeat {repeat}/{args.repeats}: {unit['name']} "
                    f"rows={selected_rows}/{unit['scan']['row_count']}",
                    flush=True,
                )
                metrics[key] = run_build(
                    builder,
                    original_local_day,
                    config,
                    unit,
                    selected_rows,
                    output_dir,
                    args.workers,
                    args.keep_indexes,
                )
                save_json(metrics_path, metrics)
                print(
                    f"  pure={metrics[key]['pure_build_seconds']:.3f}s "
                    f"size={metrics[key]['index_size_bytes'] / (1024 * 1024):.3f}MiB",
                    flush=True,
                )

    results = []
    for city in selected_cities:
        config = datasets[city]
        plan = plans[city]
        unit_by_name = {unit["name"]: unit for unit in config["units"]}
        for percentage in PERCENTAGES:
            pct_plan = plan["percentages"][str(percentage)]
            repeated_metrics = []
            for repeat in range(1, args.repeats + 1):
                selected_metrics = []
                for spec in pct_plan["specs"]:
                    unit = unit_by_name[spec["unit"]]
                    key = cache_key(city, unit, int(spec["selected_rows"]), repeat)
                    selected_metrics.append(metrics[key])
                repeated_metrics.append(selected_metrics)
            size_bytes = sum(
                int(item["index_size_bytes"]) for item in repeated_metrics[0]
            )
            build_seconds_runs = [
                sum(float(item["pure_build_seconds"]) for item in run)
                for run in repeated_metrics
            ]
            results.append(
                {
                    "dataset": config["display"],
                    "city": city,
                    "percentage": percentage,
                    "selected_trajectories": int(pct_plan["target_rows"]),
                    "total_trajectories": int(plan["total_rows"]),
                    "cutoff_unit": pct_plan["cutoff_unit"],
                    "cutoff_timestamp": pct_plan["cutoff_timestamp"],
                    "cutoff_day_utc": pct_plan["cutoff_day_utc"],
                    "cutoff_day_selected_trajectories": pct_plan[
                        "cutoff_day_selected_trajectories"
                    ],
                    "cutoff_day_total_trajectories": pct_plan[
                        "cutoff_day_total_trajectories"
                    ],
                    "partial_cutoff_day": pct_plan["partial_cutoff_day"],
                    "pure_build_seconds": statistics.fmean(build_seconds_runs),
                    "pure_build_seconds_runs": build_seconds_runs,
                    "index_size_bytes": size_bytes,
                    "index_size_mb": size_bytes / (1024 * 1024),
                }
            )

    report = {
        "metric_definition": (
            "accumulator_build_time_all_in: starts after trajectory CSV loading and ends "
            "before binary serialization; includes prime generation, edge accumulators, "
            "global roots, and offline witnesses"
        ),
        "percentage_basis": "chronological trajectory rows",
        "cutoff_rule": (
            "all earlier UTC days are included; if the target ends inside a day, only the "
            "earliest required trajectory rows from that day are included"
        ),
        "workers": int(args.workers),
        "repeats": int(args.repeats),
        "primary_statistic": "arithmetic_mean",
        "size_unit": "MiB = 1024^2 bytes",
        "results": results,
    }
    write_results(output_dir, report)
    temporary = output_dir / "temporary"
    if temporary.exists() and not args.keep_indexes:
        shutil.rmtree(temporary)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
