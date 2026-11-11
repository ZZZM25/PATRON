from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import run_rsg_50m_experiment_matrix as matrix


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "rsa_query_experiment"
SELECTION_JSON = OUTPUT_DIR / "selected_hotspots.json"
RESULT_JSON = OUTPUT_DIR / "hotspot_full_timing_results.json"
RESULT_CSV = OUTPUT_DIR / "hotspot_full_timing_results.csv"
SPAN_DEGREES = 0.0075
REPEATS = 3
TARGET_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_configs() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    daily_specs = [
        ("chengdu", "成都", ROOT / "rsg_50m_2018-10-10", "2018"),
        ("xian", "西安", ROOT / "rsg_50m_xian_2018-10-01-roadtxt", "2018"),
    ]
    for key, display, dataset_dir, year in daily_specs:
        units = []
        for trajectory_csv in sorted(dataset_dir.glob("traj-??-??.csv")):
            month_day = trajectory_csv.stem.removeprefix("traj-")
            unit_date = f"{year}-{month_day}"
            index_file = dataset_dir / f"{unit_date}.rsg-50m-private-token.bin"
            trusted_manifest = (
                dataset_dir / f"{unit_date}.rsg-50m-private-token.trusted-manifest.json"
            )
            if index_file.exists() and trusted_manifest.exists():
                units.append(
                    {
                        "name": unit_date,
                        "trajectory_csv": trajectory_csv,
                        "index_file": index_file,
                        "trusted_manifest": trusted_manifest,
                        "claim_date": unit_date,
                    }
                )
        configs[key] = {
            "key": key,
            "city": display,
            "dataset_dir": dataset_dir,
            "units": units,
        }

    beijing_root = ROOT / "rsg_50m_beijing"
    unit_dir = beijing_root / "bimonthly_units"
    summary = read_json(unit_dir / "bimonthly_mapping_summary.json")
    units = []
    for item in summary["units"]:
        name = item["name"]
        trajectory_csv = Path(item["trajectory_csv"])
        index_file = unit_dir / f"{name}.rsg-50m-private-token.bin"
        trusted_manifest = unit_dir / f"{name}.rsg-50m-private-token.trusted-manifest.json"
        if trajectory_csv.exists() and index_file.exists() and trusted_manifest.exists():
            manifest = read_json(trusted_manifest)
            units.append(
                {
                    "name": name,
                    "trajectory_csv": trajectory_csv,
                    "index_file": index_file,
                    "trusted_manifest": trusted_manifest,
                    "claim_date": manifest["date"],
                }
            )
    configs["beijing"] = {
        "key": "beijing",
        "city": "北京",
        "dataset_dir": beijing_root / "query_road",
        "units": units,
    }
    return configs


def cell_center_lonlat(metadata: dict, col: int, row: int) -> tuple[float, float]:
    radius = float(metadata["earth_radius_m"])
    lon0 = float(metadata["lon0"])
    lat0 = float(metadata["lat0"])
    cell_size = float(metadata["cell_size_m"])
    x = float(metadata["origin_x"]) + (col + 0.5) * cell_size
    y = float(metadata["origin_y"]) + (row + 0.5) * cell_size
    lon = lon0 + math.degrees(x / (radius * math.cos(math.radians(lat0))))
    lat = lat0 + math.degrees(y / radius)
    return lon, lat


def spatial_window(metadata: dict, edge_cells: list, lon: float, lat: float) -> dict:
    half = SPAN_DEGREES / 2.0
    result = matrix.resolve_spatial_range_from_index(
        metadata,
        edge_cells,
        lon - half,
        lat - half,
        lon + half,
        lat + half,
    )
    result["center_lon"] = lon
    result["center_lat"] = lat
    return result


def scan_full_trajectory_cell_bits(config: dict, edge_to_cell: dict[int, tuple[int, int]]):
    cell_bits: dict[tuple[int, int], int] = {}
    trajectory_count = 0
    for unit in config["units"]:
        with Path(unit["trajectory_csv"]).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                cells = {
                    edge_to_cell[int(segment[0])]
                    for segment in json.loads(row["trajectory_data"])
                    if int(segment[0]) in edge_to_cell
                }
                bit = 1 << trajectory_count
                for cell in cells:
                    cell_bits[cell] = cell_bits.get(cell, 0) | bit
                trajectory_count += 1
    return cell_bits, trajectory_count


def mask_for_window(cell_bits: dict, window: dict) -> int:
    col0, col1 = window["cell_col_range"]
    row0, row1 = window["cell_row_range"]
    value = 0
    for col in range(int(col0), int(col1) + 1):
        for row in range(int(row0), int(row1) + 1):
            value |= cell_bits.get((col, row), 0)
    return value


def choose_density_points(config: dict) -> list[dict]:
    metadata, edge_cells = matrix.load_spatial_index(config["dataset_dir"])
    edge_to_cell = {int(edge): (int(col), int(row)) for edge, col, row in edge_cells}
    cell_bits, total_trajectories = scan_full_trajectory_cell_bits(config, edge_to_cell)
    active_cells = sorted(cell_bits)
    target_samples = 3500
    stride = max(1, len(active_cells) // target_samples)
    sampled_cells = active_cells[::stride]
    if active_cells and active_cells[-1] not in sampled_cells:
        sampled_cells.append(active_cells[-1])

    candidates = []
    for col, row in sampled_cells:
        lon, lat = cell_center_lonlat(metadata, col, row)
        window = spatial_window(metadata, edge_cells, lon, lat)
        count = mask_for_window(cell_bits, window).bit_count()
        if count:
            candidates.append(
                {
                    "center_lon": lon,
                    "center_lat": lat,
                    "anchor_cell_col": col,
                    "anchor_cell_row": row,
                    "full_dataset_trajectory_count": count,
                    "covered_cell_count": int(window["covered_cell_count"]),
                    "road_edge_count": int(window["road_edge_count"]),
                    "cell_col_range": window["cell_col_range"],
                    "cell_row_range": window["cell_row_range"],
                    "requested_bounds": window["requested_bounds"],
                    "edge_ids": [int(value) for value in window["edge_ids"]],
                }
            )
    candidates.sort(key=lambda value: value["full_dataset_trajectory_count"])
    if len(candidates) < 5:
        raise RuntimeError(f"{config['city']} has fewer than five nonempty candidates")

    selected = []
    previous_count = -1
    minimum_cell_distance = 6
    for quantile in TARGET_QUANTILES:
        target_index = round((len(candidates) - 1) * quantile)
        order = sorted(range(len(candidates)), key=lambda index: abs(index - target_index))
        chosen = None
        for require_distance in (True, False):
            for index in order:
                candidate = candidates[index]
                count = int(candidate["full_dataset_trajectory_count"])
                if count <= previous_count:
                    continue
                if require_distance and any(
                    abs(int(candidate["anchor_cell_col"]) - int(old["anchor_cell_col"]))
                    < minimum_cell_distance
                    and abs(int(candidate["anchor_cell_row"]) - int(old["anchor_cell_row"]))
                    < minimum_cell_distance
                    for old in selected
                ):
                    continue
                chosen = dict(candidate)
                break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"cannot select increasing hotspot for {config['city']}")
        chosen["density_quantile_target"] = quantile
        selected.append(chosen)
        previous_count = int(chosen["full_dataset_trajectory_count"])

    selected.sort(key=lambda value: value["full_dataset_trajectory_count"])
    for index, point in enumerate(selected, 1):
        point["point_id"] = index
        point["city"] = config["city"]
        point["dataset_total_trajectory_count"] = total_trajectories
    del cell_bits
    gc.collect()
    return selected


def unit_time_axis(unit: dict) -> dict:
    manifest = read_json(Path(unit["trusted_manifest"]))
    axis = manifest.get("time_axis")
    if axis:
        return {
            "origin": int(axis["start_timestamp"]),
            "end": int(axis["end_timestamp_exclusive"]),
            "slot_seconds": int(axis["time_unit_seconds"]),
        }
    origin = int(
        datetime.fromisoformat(unit["claim_date"])
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    return {"origin": origin, "end": origin + 86400, "slot_seconds": 1}


def query_start_range(origin: int, end: int, t0: int, t1: int) -> tuple[int, int] | None:
    a = max(origin, min(t0, t1))
    b = min(end, max(t0, t1) + 1)
    if a >= b:
        return None
    max_start_minute = (end - origin - 3600) // 60
    lower = max(0, int((a - 3600 - origin) // 60 + 1))
    upper = min(int(max_start_minute), int((b - 1 - origin) // 60))
    return (lower, upper) if lower <= upper else None


def choose_dense_hours(config: dict, points: list[dict]) -> None:
    best = [None for _ in points]
    edge_to_points: dict[int, list[int]] = defaultdict(list)
    for point_index, point in enumerate(points):
        for edge_id in point["edge_ids"]:
            edge_to_points[int(edge_id)].append(point_index)

    for unit in config["units"]:
        axis = unit_time_axis(unit)
        origin, end = axis["origin"], axis["end"]
        start_count = (end - origin - 3600) // 60 + 1
        if start_count <= 0:
            continue
        diffs = [[0] * (start_count + 1) for _ in points]
        with Path(unit["trajectory_csv"]).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                ranges_by_point = [[] for _ in points]
                for segment in json.loads(row["trajectory_data"]):
                    point_indexes = edge_to_points.get(int(segment[0]))
                    if not point_indexes:
                        continue
                    candidate_range = query_start_range(
                        origin, end, int(segment[2]), int(segment[4])
                    )
                    if candidate_range is None:
                        continue
                    for point_index in point_indexes:
                        ranges_by_point[point_index].append(candidate_range)
                for point_index, ranges in enumerate(ranges_by_point):
                    if not ranges:
                        continue
                    ranges.sort()
                    merged = []
                    for lower, upper in ranges:
                        if not merged or lower > merged[-1][1] + 1:
                            merged.append([lower, upper])
                        else:
                            merged[-1][1] = max(merged[-1][1], upper)
                    for lower, upper in merged:
                        diffs[point_index][lower] += 1
                        diffs[point_index][upper + 1] -= 1

        for point_index, diff in enumerate(diffs):
            running = 0
            local_best_count = -1
            local_best_minute = 0
            for minute in range(start_count):
                running += diff[minute]
                if running > local_best_count:
                    local_best_count = running
                    local_best_minute = minute
            if best[point_index] is None or local_best_count > best[point_index]["count"]:
                query_start = origin + local_best_minute * 60
                best[point_index] = {
                    "count": int(local_best_count),
                    "unit_name": unit["name"],
                    "claim_date": unit["claim_date"],
                    "query_start_timestamp": int(query_start),
                    "query_end_timestamp": int(query_start + 3600),
                    "slot_seconds": int(axis["slot_seconds"]),
                    "axis_origin": int(origin),
                }

    unit_by_name = {unit["name"]: unit for unit in config["units"]}
    china_tz = timezone(timedelta(hours=8))
    for point, chosen in zip(points, best):
        if chosen is None or int(chosen["count"]) <= 0:
            raise RuntimeError(f"{config['city']} P{point['point_id']} has no nonempty hour")
        point.update(chosen)
        point["estimated_query_trajectory_count"] = int(chosen.pop("count"))
        point["query_start_utc"] = datetime.fromtimestamp(
            point["query_start_timestamp"], timezone.utc
        ).isoformat()
        point["query_end_utc"] = datetime.fromtimestamp(
            point["query_end_timestamp"], timezone.utc
        ).isoformat()
        point["query_start_beijing"] = datetime.fromtimestamp(
            point["query_start_timestamp"], china_tz
        ).isoformat()
        point["query_end_beijing"] = datetime.fromtimestamp(
            point["query_end_timestamp"], china_tz
        ).isoformat()
        point["index_file"] = str(Path(unit_by_name[point["unit_name"]]["index_file"]).resolve())
        point["trusted_manifest"] = str(
            Path(unit_by_name[point["unit_name"]]["trusted_manifest"]).resolve()
        )


def selection_payload(configs: dict[str, dict], selected: dict[str, list[dict]]) -> dict:
    return {
        "spatial_window": "W2: 0.0075 x 0.0075 degrees expanded to the minimum covering 50 m cell rectangle",
        "time_window_seconds": 3600,
        "full_dataset_count_definition": "trajectory rows intersecting the W2 edge set across every constructed RSA unit",
        "selection": selected,
        "constructed_unit_counts": {
            config["city"]: len(config["units"]) for config in configs.values()
        },
    }


def write_results(results: list[dict], selected: dict[str, list[dict]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "repeats": REPEATS,
        "schema_version": 1,
        "statistic": "arithmetic mean of three formal runs",
        "query_timing": "spatial bounds-to-cell/edge resolution + temporal/result preparation + RSA member proof generation + RSA completeness proof generation",
        "verification_timing": "complete client verification through RSA correctness, completeness, edge coverage, status aggregation and VO accounting",
        "one_time_setup_excluded": "trusted-manifest hashing, index scanning and worker/index preload",
        "vo_mb_divisor": 1_000_000,
        "selection": selected,
        "results": results,
    }
    RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "city", "point_id", "center_lon", "center_lat",
        "full_dataset_trajectory_count", "dataset_total_trajectory_count",
        "unit_name", "query_start_timestamp", "query_end_timestamp",
        "query_start_utc", "query_end_utc", "query_start_beijing", "query_end_beijing",
        "result_trajectory_count", "covered_cell_count", "road_edge_count",
        "average_spatial_seconds", "average_prepare_seconds",
        "average_rsa_proof_seconds", "average_full_query_seconds",
        "average_full_verify_seconds", "average_vo_bytes", "average_vo_mb", "all_ok",
    ]
    with RESULT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def benchmark_unit_group(
    config: dict,
    points: list[dict],
    results: list[dict],
    workers: int,
) -> None:
    unit_name = points[0]["unit_name"]
    unit = next(unit for unit in config["units"] if unit["name"] == unit_name)
    query_module = __import__(matrix.QUERY_MODULE)
    trusted = matrix.verify_trusted_ads_manifest(
        Path(unit["trusted_manifest"]),
        Path(config["dataset_dir"]),
        Path(unit["index_file"]),
        query_module,
        expected_date=unit["claim_date"],
    )
    metadata, edge_cells = matrix.load_spatial_index(config["dataset_dir"])
    union_edges = sorted({int(edge) for point in points for edge in point["edge_ids"]})
    _rsa, edge_infos = query_module.scan_edges_for_date(
        str(unit["index_file"]), unit["claim_date"]
    )
    edge_info_by_id = {int(info["edge_id"]): info for info in edge_infos}
    assignments = matrix.assign_edges(union_edges, edge_info_by_id, workers)

    context = mp.get_context("spawn")
    conns = []
    processes = []
    try:
        for edge_ids in assignments:
            parent, child = context.Pipe()
            process = context.Process(
                target=matrix.matrix_worker,
                args=(child, str(unit["index_file"]), edge_ids, trusted, unit["claim_date"]),
            )
            process.start()
            child.close()
            conns.append(parent)
            processes.append(process)
        for conn in conns:
            reply = conn.recv()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error"))

        for point in points:
            runs = []
            for repeat in range(REPEATS):
                spatial_started = time.perf_counter()
                resolved = spatial_window(
                    metadata,
                    edge_cells,
                    float(point["center_lon"]),
                    float(point["center_lat"]),
                )
                spatial_seconds = time.perf_counter() - spatial_started
                qs = (
                    int(point["query_start_timestamp"]) - int(point["axis_origin"])
                ) // int(point["slot_seconds"])
                qe = (
                    int(point["query_end_timestamp"]) - 1 - int(point["axis_origin"])
                ) // int(point["slot_seconds"])
                run = matrix.execute_once(conns, resolved["edge_ids"], qs, qe)
                run["spatial_seconds"] = spatial_seconds
                run["full_query_seconds"] = (
                    spatial_seconds
                    + float(run["prepare_seconds"])
                    + float(run["member_batch_gen_seconds"])
                    + float(run["completeness_gen_seconds"])
                )
                runs.append(run)
                print(
                    f"[{config['city']} P{point['point_id']} {repeat + 1}/{REPEATS}] "
                    f"query={run['full_query_seconds']:.6f}s "
                    f"verify={run['full_verify_seconds']:.6f}s "
                    f"traj={run['result_trajectory_count']} ok={run['all_ok']}",
                    flush=True,
                )
            row = {key: value for key, value in point.items() if key != "edge_ids"}
            row.update(
                {
                    "result_trajectory_count": int(runs[0]["result_trajectory_count"]),
                    "average_spatial_seconds": statistics.fmean(run["spatial_seconds"] for run in runs),
                    "average_prepare_seconds": statistics.fmean(run["prepare_seconds"] for run in runs),
                    "average_rsa_proof_seconds": statistics.fmean(
                        run["member_batch_gen_seconds"] + run["completeness_gen_seconds"]
                        for run in runs
                    ),
                    "average_full_query_seconds": statistics.fmean(run["full_query_seconds"] for run in runs),
                    "average_full_verify_seconds": statistics.fmean(run["full_verify_seconds"] for run in runs),
                    "average_vo_bytes": statistics.fmean(run["total_vo_bytes"] for run in runs),
                    "average_vo_mb": statistics.fmean(run["total_vo_bytes"] for run in runs) / 1_000_000,
                    "all_ok": all(bool(run["all_ok"]) for run in runs),
                    "runs": runs,
                }
            )
            results.append(row)
            write_results(results, {})
    finally:
        for conn in conns:
            try:
                conn.send({"command": "close"})
            except Exception:
                pass
        for conn in conns:
            try:
                conn.recv()
            except Exception:
                pass
            conn.close()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--city", action="append", choices=("chengdu", "xian", "beijing"))
    args = parser.parse_args()
    configs = discover_configs()
    requested = args.city or ["chengdu", "xian", "beijing"]
    configs = {key: configs[key] for key in requested}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume and SELECTION_JSON.exists():
        selection_file = read_json(SELECTION_JSON)
        selected = {
            city: points
            for city, points in selection_file["selection"].items()
            if any(config["city"] == city for config in configs.values())
        }
    else:
        selected = {}
        for config in configs.values():
            print(f"[{config['city']}] selecting five increasing nonempty hotspots", flush=True)
            points = choose_density_points(config)
            choose_dense_hours(config, points)
            selected[config["city"]] = points
        SELECTION_JSON.write_text(
            json.dumps(selection_payload(configs, selected), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.select_only:
        return

    results = []
    if args.resume and RESULT_JSON.exists():
        results = list(read_json(RESULT_JSON).get("results", []))
    completed = {(row["city"], int(row["point_id"])) for row in results}
    for config in configs.values():
        pending = [
            point for point in selected[config["city"]]
            if (config["city"], int(point["point_id"])) not in completed
        ]
        groups: dict[str, list[dict]] = defaultdict(list)
        for point in pending:
            groups[point["unit_name"]].append(point)
        for points in groups.values():
            benchmark_unit_group(config, points, results, args.workers)
            write_results(results, selected)
    results.sort(key=lambda row: (row["city"], int(row["point_id"])))
    write_results(results, selected)
    print(f"completed rows={len(results)} output={RESULT_CSV}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
