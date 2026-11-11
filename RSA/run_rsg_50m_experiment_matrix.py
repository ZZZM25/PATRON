"""Run the canonical RSA/RSG query, verification, and VO-size experiment."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import statistics
import time
from datetime import date, datetime, timezone
from pathlib import Path

from gmpy2 import mpz


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / "rsg_50m_2018-10-10"
QUERY_MODULE = "rsa_query"
CENTER_LON = 104.07
CENTER_LAT = 30.67
WINDOWS = [
    ("W1", 0.005),
    ("W2", 0.0075),
    ("W3", 0.010),
    ("W4", 0.015),
    ("W5", 0.020),
]
TIME_RANGES = [
    ("1h", 60 * 60),
    ("2h", 2 * 60 * 60),
    ("4h", 4 * 60 * 60),
    ("8h", 8 * 60 * 60),
    ("12h", 12 * 60 * 60),
]
OUTPUT_STEM = "rsa_query_experiment"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_trusted_ads_manifest(
    manifest_path: Path,
    dataset_dir: Path,
    index_file: Path,
    query_module,
    expected_date: str = "2018-10-10",
) -> dict:
    """Verify files and RSA date root against the owner-pinned manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {
        "rsg-trusted-ads-manifest-v1",
        "rsg-private-token-trusted-manifest-v2",
    }:
        raise ValueError("unsupported trusted ADS manifest schema")
    if manifest.get("date") != expected_date:
        raise ValueError("trusted ADS manifest date mismatch")

    paths_by_role = {
        "rsa_index": index_file,
        "grid_metadata": dataset_dir / "grid_metadata.json",
        "edge_cells": dataset_dir / "edge_cells.csv",
    }
    records = {record["role"]: record for record in manifest.get("files", [])}
    for role, path in paths_by_role.items():
        record = records.get(role)
        if record is None:
            raise ValueError(f"trusted ADS manifest missing file role: {role}")
        if record.get("name") != path.name:
            raise ValueError(f"trusted ADS file name mismatch: {role}")
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"trusted ADS file size mismatch: {role}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"trusted ADS file hash mismatch: {role}")

    root_db = query_module.load_binary_selected_edges(
        str(index_file), manifest["date"], set()
    )
    trusted_root = root_db.get("trusted_root") or {}
    if str(root_db["rsa"]["n"]) != str(manifest["rsa"]["n"]):
        raise ValueError("trusted ADS RSA n mismatch")
    if str(root_db["rsa"]["g"]) != str(manifest["rsa"]["g"]):
        raise ValueError("trusted ADS RSA g mismatch")
    if str(trusted_root.get("data_ac")) != str(manifest["data_root"]["data_ac"]):
        raise ValueError("trusted ADS data accumulator root mismatch")
    return manifest


def load_spatial_index(dataset_dir: Path) -> tuple[dict, list[tuple[int, int, int]]]:
    with (dataset_dir / "grid_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    edge_cells = []
    with (dataset_dir / "edge_cells.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            edge_cells.append(
                (int(row["edge_id"]), int(row["cell_col"]), int(row["cell_row"]))
            )
    return metadata, edge_cells


def resolve_spatial_range_from_index(
    metadata: dict,
    edge_cells: list[tuple[int, int, int]],
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> dict:
    """Expand a lon/lat box to the minimum covering Cell rectangle."""
    min_lon = float(min_lon)
    min_lat = float(min_lat)
    max_lon = float(max_lon)
    max_lat = float(max_lat)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("非法空间查询范围")

    radius = float(metadata["earth_radius_m"])
    lon0 = float(metadata["lon0"])
    lat0 = float(metadata["lat0"])
    cos_lat0 = math.cos(math.radians(lat0))
    origin_x = float(metadata["origin_x"])
    origin_y = float(metadata["origin_y"])
    cell_size = float(metadata["cell_size_m"])

    def lonlat_to_xy(lon: float, lat: float) -> tuple[float, float]:
        return (
            radius * math.radians(lon - lon0) * cos_lat0,
            radius * math.radians(lat - lat0),
        )

    min_x, min_y = lonlat_to_xy(min_lon, min_lat)
    max_x, max_y = lonlat_to_xy(max_lon, max_lat)
    col_min = max(0, math.floor((min_x - origin_x) / cell_size))
    col_max = min(
        int(metadata["grid_cols"]) - 1,
        math.ceil((max_x - origin_x) / cell_size) - 1,
    )
    row_min = max(0, math.floor((min_y - origin_y) / cell_size))
    row_max = min(
        int(metadata["grid_rows"]) - 1,
        math.ceil((max_y - origin_y) / cell_size) - 1,
    )
    if col_min > col_max or row_min > row_max:
        raise ValueError("空间查询范围不与当前网格相交")
    edge_ids = sorted(
        edge_id
        for edge_id, col, row in edge_cells
        if col_min <= col <= col_max and row_min <= row <= row_max
    )
    return {
        "requested_bounds": [min_lon, min_lat, max_lon, max_lat],
        "cell_col_range": [col_min, col_max],
        "cell_row_range": [row_min, row_max],
        "covered_cell_count": (col_max - col_min + 1) * (row_max - row_min + 1),
        "road_edge_count": len(edge_ids),
        "edge_ids": edge_ids,
    }


def resolve_spatial_range(
    dataset_dir: Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> tuple[dict, dict]:
    metadata, edge_cells = load_spatial_index(dataset_dir)
    return (
        resolve_spatial_range_from_index(
            metadata, edge_cells, min_lon, min_lat, max_lon, max_lat
        ),
        metadata,
    )


def load_spatial_windows(
    dataset_dir: Path,
    center_lon: float = CENTER_LON,
    center_lat: float = CENTER_LAT,
) -> tuple[dict[str, dict], dict]:
    metadata, edge_cells = load_spatial_index(dataset_dir)
    windows = {}
    for label, span in WINDOWS:
        min_lon = center_lon - span / 2.0
        max_lon = center_lon + span / 2.0
        min_lat = center_lat - span / 2.0
        max_lat = center_lat + span / 2.0
        window = resolve_spatial_range_from_index(
            metadata, edge_cells, min_lon, min_lat, max_lon, max_lat
        )
        window.update({"label": label, "span_degrees": span})
        windows[label] = window
    return windows, metadata


def matrix_worker(
    conn,
    input_bin: str,
    assigned_edge_ids: list[int],
    trusted_manifest: dict,
    claim_date: str = "2018-10-10",
) -> None:
    q = importlib.import_module(QUERY_MODULE)
    q.ENABLE_MEMBER_NIPOE = False
    q.ENABLE_COMPLETENESS_NI_PROOF = True
    q.STORE_QUERY_PAIR_SEEDS_IN_VO = False
    q.USE_UNCACHED_QUERY_PAIR_PRIME = True
    # v3 completeness is bound by authenticated count + opaque-token coverage;
    # it never reconstructs U-R from a client-side ADS.
    q.STRICT_COMPLETENESS_BINDING = False
    q.CLAIM_DATE = claim_date
    contexts = []
    try:
        db = q.load_binary_selected_edges(input_bin, q.CLAIM_DATE, set(assigned_edge_ids))
        rsa = trusted_manifest["rsa"]
        # This is the verifier's complete database: owner-pinned public
        # parameters and the authenticated global root only.  It deliberately
        # contains no server edge records or trajectory universe.
        client_db = {
            "rsa": {
                "n": mpz(rsa["n"]),
                "g": mpz(rsa["g"]),
                "hprime_bits": int(rsa["hprime_bits"]),
                "depth": int(rsa["depth"]),
                "format_version": int(rsa["format_version"]),
                "token_bytes": int(rsa.get("opaque_token_bytes", 0)),
                "base": int(rsa.get("base", 2)),
                "day_seconds": int(rsa.get("day_seconds", 86400)),
            },
            "trusted_root": {"data_ac": mpz(trusted_manifest["data_root"]["data_ac"])},
            "data": {},
        }
        loaded_edge_count = sum(
            1
            for edge_id in assigned_edge_ids
            if str(edge_id) in db["data"][q.CLAIM_DATE]
        )
        conn.send(
            {
                "ok": True,
                "event": "ready",
                "assigned_edge_count": len(assigned_edge_ids),
                "loaded_nonempty_edge_count": loaded_edge_count,
            }
        )

        assigned_set = set(assigned_edge_ids)
        while True:
            message = conn.recv()
            command = message["command"]
            if command == "close":
                conn.send({"ok": True, "event": "closed"})
                return
            if command == "prepare":
                q.CLAIM_QS = int(message["qs"])
                q.CLAIM_QE = int(message["qe_inclusive"])
                requested = [
                    int(edge_id)
                    for edge_id in message["edge_ids"]
                    if int(edge_id) in assigned_set
                    and str(int(edge_id)) in db["data"][q.CLAIM_DATE]
                ]
                contexts = [q.worker_prepare_edge_context(db, edge_id) for edge_id in requested]
                result_ids = set()
                present_edges = 0
                for context in contexts:
                    result_ids.update(context["used_r"])
                    edge_obj = db["data"][q.CLAIM_DATE].get(str(context["edge_id"]))
                    if edge_obj is not None and int(edge_obj.get("traj_count", 0)) > 0:
                        present_edges += 1
                conn.send(
                    {
                        "ok": True,
                        "event": "prepared",
                        "edge_count": len(contexts),
                        "edge_ids": [int(context["edge_id"]) for context in contexts],
                        "present_edge_count": present_edges,
                        "result_ids": sorted(result_ids),
                    }
                )
                continue

            if command == "member_gen":
                results = [q.worker_member_gen(db, context) for context in contexts]
            elif command == "completeness_gen":
                results = [q.worker_completeness_gen(db, context) for context in contexts]
            elif command == "member_verify":
                results = [q.worker_member_verify(client_db, context) for context in contexts]
            elif command == "completeness_verify":
                results = [q.worker_completeness_verify(client_db, context) for context in contexts]
            elif command == "release":
                contexts = []
                gc.collect()
                results = []
            else:
                raise ValueError(f"未知命令: {command}")
            conn.send({"ok": True, "event": command, "results": results})
    except BaseException as exc:
        try:
            conn.send({"ok": False, "event": "error", "error": repr(exc)})
        except Exception:
            pass
        raise
    finally:
        conn.close()


def send_all(conns, message: dict) -> list[dict]:
    for conn in conns:
        conn.send(message)
    replies = []
    for conn in conns:
        reply = conn.recv()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error"))
        replies.append(reply)
    return replies


def run_timed_stage(conns, command: str) -> tuple[float, list[dict]]:
    started = time.perf_counter()
    replies = send_all(conns, {"command": command})
    elapsed = time.perf_counter() - started
    results = []
    for reply in replies:
        results.extend(reply.get("results", []))
    return elapsed, results


def has_exact_edge_coverage(results: list[dict], expected_edge_ids: set[int]) -> bool:
    """Require every requested edge exactly once; reject omissions/duplicates/extras."""
    try:
        returned = [int(result["edge_id"]) for result in results]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        len(returned) == len(expected_edge_ids)
        and len(set(returned)) == len(returned)
        and set(returned) == expected_edge_ids
    )


def assign_edges(edge_ids: list[int], edge_info_by_id: dict[int, dict], workers: int) -> list[list[int]]:
    workers = max(1, min(workers, len(edge_ids)))
    buckets = [(0, index, []) for index in range(workers)]
    import heapq

    heapq.heapify(buckets)
    weighted = []
    for edge_id in edge_ids:
        info = edge_info_by_id.get(edge_id)
        weight = int(info["weight"]) if info else 1
        weighted.append((weight, edge_id))
    for weight, edge_id in sorted(weighted, reverse=True):
        load, index, values = heapq.heappop(buckets)
        values.append(edge_id)
        heapq.heappush(buckets, (load + weight, index, values))
    buckets.sort(key=lambda item: item[1])
    return [values for _load, _index, values in buckets]


def assign_nested_window_edges(
    windows: dict[str, dict],
    edge_info_by_id: dict[int, dict],
    workers: int,
) -> list[list[int]]:
    """Balance every nested spatial window, not only the largest W5 set.

    W1..W5 are nested.  Balancing the whole W5 set once can leave a smaller
    prefix such as W2 badly skewed across workers.  LPT-balancing each spatial
    ring (W1, W2-W1, ..., W5-W4) independently keeps every cumulative window
    close to balanced while retaining one persistent worker pool.
    """
    largest = set(int(x) for x in windows["W5"]["edge_ids"])
    workers = max(1, min(int(workers), len(largest)))
    assignments = [[] for _ in range(workers)]
    previous: set[int] = set()

    for window_label, _span in WINDOWS:
        current = set(int(x) for x in windows[window_label]["edge_ids"])
        ring = current - previous
        ring_buckets = [(0, index) for index in range(workers)]
        import heapq

        heapq.heapify(ring_buckets)
        weighted_ring = []
        for edge_id in ring:
            info = edge_info_by_id.get(edge_id)
            weight = int(info["weight"]) if info else 1
            weighted_ring.append((weight, edge_id))
        for weight, edge_id in sorted(weighted_ring, reverse=True):
            load, index = heapq.heappop(ring_buckets)
            assignments[index].append(edge_id)
            heapq.heappush(ring_buckets, (load + weight, index))
        previous = current

    # Defensive support for a malformed/non-nested window description.
    unassigned = largest - {edge for group in assignments for edge in group}
    for offset, edge_id in enumerate(sorted(unassigned)):
        assignments[offset % workers].append(edge_id)
    return [sorted(values) for values in assignments if values]


def execute_once(conns, edge_ids: list[int], qs: int, qe_inclusive: int) -> dict:
    expected_edge_ids = {int(edge_id) for edge_id in edge_ids}
    prepare_started = time.perf_counter()
    prepare_replies = send_all(
        conns,
        {
            "command": "prepare",
            "qs": int(qs),
            "qe_inclusive": int(qe_inclusive),
            "edge_ids": edge_ids,
        },
    )
    prepare_seconds = time.perf_counter() - prepare_started
    result_ids = set()
    present_edge_count = 0
    prepared_edge_ids = []
    for reply in prepare_replies:
        result_ids.update(int(value) for value in reply["result_ids"])
        present_edge_count += int(reply["present_edge_count"])
        prepared_edge_ids.extend(int(value) for value in reply.get("edge_ids", []))

    member_gen, member_gen_results = run_timed_stage(conns, "member_gen")
    completeness_gen, completeness_gen_results = run_timed_stage(conns, "completeness_gen")
    full_verify_started = time.perf_counter()
    member_verify, member_verify_results = run_timed_stage(conns, "member_verify")
    completeness_verify, completeness_verify_results = run_timed_stage(
        conns, "completeness_verify"
    )
    prepare_coverage_ok = (
        len(prepared_edge_ids) == len(expected_edge_ids)
        and len(set(prepared_edge_ids)) == len(prepared_edge_ids)
        and set(prepared_edge_ids) == expected_edge_ids
    )
    stage_coverage_ok = all(
        has_exact_edge_coverage(results, expected_edge_ids)
        for results in (
            member_gen_results,
            completeness_gen_results,
            member_verify_results,
            completeness_verify_results,
        )
    )
    all_ok = (
        prepare_coverage_ok
        and stage_coverage_ok
        and all(result.get("soundness_ok", False) for result in member_verify_results)
        and all(
            result.get("completeness_ok", False)
            for result in completeness_verify_results
        )
    )
    statuses = {}
    for result in completeness_gen_results:
        status = result.get("completeness_status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    member_vo_bytes = sum(int(result.get("member_vo_bytes", 0)) for result in member_gen_results)
    invalid_items_vo_bytes = sum(
        int(result.get("invalid_items_vo_bytes", 0)) for result in member_gen_results
    )
    completeness_vo_bytes = sum(
        int(result.get("completeness_vo_bytes", 0))
        for result in completeness_gen_results
    )
    # End-to-end client verification also includes response/edge coverage,
    # status aggregation and VO-size accounting, not only the RSA checks.
    full_verify_seconds = time.perf_counter() - full_verify_started
    send_all(conns, {"command": "release"})
    return {
        "prepare_seconds": prepare_seconds,
        "member_batch_gen_seconds": member_gen,
        "completeness_gen_seconds": completeness_gen,
        "query_seconds": member_gen + completeness_gen,
        "member_batch_verify_seconds": member_verify,
        "completeness_verify_seconds": completeness_verify,
        "verify_seconds": member_verify + completeness_verify,
        "full_verify_seconds": full_verify_seconds,
        "result_trajectory_count": len(result_ids),
        "result_trajectory_ids": sorted(result_ids),
        "present_edge_count": present_edge_count,
        "all_ok": all_ok,
        "prepare_edge_coverage_ok": prepare_coverage_ok,
        "stage_edge_coverage_ok": stage_coverage_ok,
        "completeness_status_counts": statuses,
        "member_vo_bytes": member_vo_bytes,
        "invalid_items_vo_bytes": invalid_items_vo_bytes,
        "completeness_vo_bytes": completeness_vo_bytes,
        "total_vo_bytes": member_vo_bytes + invalid_items_vo_bytes + completeness_vo_bytes,
    }


def summarize_runs(runs: list[dict]) -> dict:
    summary = dict(runs[0])
    for field in (
        "prepare_seconds",
        "member_batch_gen_seconds",
        "completeness_gen_seconds",
        "query_seconds",
        "member_batch_verify_seconds",
        "completeness_verify_seconds",
        "verify_seconds",
    ):
        values = [float(run[field]) for run in runs]
        summary[field] = statistics.fmean(values)
        summary[field + "_min"] = min(values)
        summary[field + "_max"] = max(values)
    summary["all_ok"] = all(run["all_ok"] for run in runs)
    summary["repeat_count"] = len(runs)
    summary["runs"] = runs
    return summary


def write_outputs(output_dir: Path, payload: dict, output_stem: str = OUTPUT_STEM) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_stem}.json"
    csv_path = output_dir / f"{output_stem}.csv"
    md_path = output_dir / f"{output_stem}.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fields = [
        "query_id",
        "window",
        "time_level",
        "span_degrees",
        "duration_seconds",
        "covered_cell_count",
        "road_edge_count",
        "present_edge_count",
        "result_trajectory_count",
        "query_seconds",
        "verify_seconds",
        "prepare_seconds",
        "member_batch_gen_seconds",
        "completeness_gen_seconds",
        "member_batch_verify_seconds",
        "completeness_verify_seconds",
        "member_vo_bytes",
        "invalid_items_vo_bytes",
        "completeness_vo_bytes",
        "total_vo_bytes",
        "all_ok",
        "repeat_count",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["results"])

    lines = [
        f"# RSG 50 m query/verification matrix ({payload['dataset_date']} UTC)",
        "",
        "查询时间为左闭右开；整数秒实现使用闭区间 `[0, duration-1]`。",
        "查询与验证时间为 3 次正式运行的墙钟时间算术平均值；准备时间不计入查询时间。",
        "",
        "| ID | Window | Time | cells | road edges | indexed edges | trajectories | VO bytes | query (s) | verify (s) | all_ok |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['query_id']} | {row['window']} | {row['time_level']} | "
            f"{row['covered_cell_count']} | {row['road_edge_count']} | "
            f"{row['present_edge_count']} | {row['result_trajectory_count']} | "
            f"{row.get('total_vo_bytes', 0)} | "
            f"{row['query_seconds']:.6f} | {row['verify_seconds']:.6f} | "
            f"{row['all_ok']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"results_json={json_path}")
    print(f"results_csv={csv_path}")
    print(f"results_markdown={md_path}")


def json_safe_rsa_metadata(metadata: dict) -> dict:
    result = {}
    for key, value in metadata.items():
        if key in {"n", "g"}:
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            try:
                result[key] = int(value)
            except (TypeError, ValueError):
                result[key] = str(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RSA/RSG 50 米查询、验证与 VO 大小实验")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--date", default="2018-10-10")
    parser.add_argument("--center-lon", type=float, default=CENTER_LON)
    parser.add_argument("--center-lat", type=float, default=CENTER_LAT)
    parser.add_argument(
        "--query-start-timestamp",
        type=int,
        help="UTC 查询日起点 Unix 时间戳；默认由 --date 的 UTC 00:00:00 计算",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        help="待测索引；默认使用优化前的 v1 索引",
    )
    parser.add_argument(
        "--trusted-manifest",
        type=Path,
        help="数据拥有者通过可信信道发布并由客户端固定的 ADS manifest",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--time-ranges",
        default=",".join(f"{label}:{duration}" for label, duration in TIME_RANGES),
        help="逗号分隔的 标签:秒数，例如 1h:3600,2h:7200,4h:14400,8h:28800,12h:43200",
    )
    parser.add_argument("--only", help="只运行一个组合，例如 W1T1")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 继续")
    parser.add_argument(
        "--output-stem",
        default=OUTPUT_STEM,
        help="输出文件名前缀；用于把不同索引/优化版本的结果隔离保存",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_ranges = []
    for item in str(args.time_ranges).split(","):
        label, separator, seconds = item.strip().partition(":")
        if not separator or not label or int(seconds) <= 0:
            raise ValueError(f"非法时间范围: {item!r}")
        time_ranges.append((label, int(seconds)))
    dataset_date = date.fromisoformat(args.date).isoformat()
    query_start_timestamp = (
        int(args.query_start_timestamp)
        if args.query_start_timestamp is not None
        else int(
            datetime.fromisoformat(dataset_date)
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    )
    requested_output_stem = str(args.output_stem)
    dataset_dir = args.dataset_dir.resolve()
    input_bin = (
        args.index_file.resolve()
        if args.index_file is not None
        else dataset_dir / f"{dataset_date}.rsg-50m.bin"
    )
    if not input_bin.exists():
        raise FileNotFoundError(input_bin)
    q = importlib.import_module(QUERY_MODULE)
    manifest_path = (
        args.trusted_manifest.resolve()
        if args.trusted_manifest is not None
        else dataset_dir / f"{dataset_date}.rsg-50m.trusted-manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"缺少数据拥有者发布的可信 manifest: {manifest_path}"
        )
    trusted_manifest = verify_trusted_ads_manifest(
        manifest_path, dataset_dir, input_bin, q, expected_date=dataset_date
    )
    print(f"trusted_ads_manifest_ok={manifest_path}")

    windows, grid_metadata = load_spatial_windows(
        dataset_dir, center_lon=args.center_lon, center_lat=args.center_lat
    )
    largest_edge_ids = windows["W5"]["edge_ids"]

    scan_started = time.perf_counter()
    rsa_metadata, edge_infos = q.scan_edges_for_date(str(input_bin), dataset_date)
    scan_seconds = time.perf_counter() - scan_started
    edge_info_by_id = {int(info["edge_id"]): info for info in edge_infos}
    assignments = assign_nested_window_edges(windows, edge_info_by_id, args.workers)
    print(
        f"index_scan_seconds={scan_seconds:.3f}, indexed_edges={len(edge_infos):,}, "
        f"W5_road_edges={len(largest_edge_ids):,}, workers={len(assignments)}"
    )

    context = mp.get_context("spawn")
    conns = []
    processes = []
    load_started = time.perf_counter()
    try:
        for edge_ids in assignments:
            parent, child = context.Pipe()
            process = context.Process(
                target=matrix_worker,
                args=(child, str(input_bin), edge_ids, trusted_manifest, dataset_date),
            )
            process.start()
            child.close()
            conns.append(parent)
            processes.append(process)
        ready = []
        for conn in conns:
            reply = conn.recv()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error"))
            ready.append(reply)
        worker_load_seconds = time.perf_counter() - load_started
        print(
            f"worker_load_seconds={worker_load_seconds:.3f}, "
            f"loaded_nonempty_edges={sum(x['loaded_nonempty_edge_count'] for x in ready):,}"
        )

        checkpoint_path = (
            dataset_dir
            / "experiment_results"
            / (
                f"{requested_output_stem}.checkpoint.json"
                if not args.only
                else f"{requested_output_stem}-{args.only.upper()}-smoke.checkpoint.json"
            )
        )
        results = []
        if args.resume and checkpoint_path.exists():
            checkpoint_payload = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            results = list(checkpoint_payload.get("results", []))
            print(f"resume_completed_queries={len(results)}")
        completed_query_ids = {row["query_id"] for row in results}
        query_number = 0
        for window_label, _span in WINDOWS:
            window = windows[window_label]
            for time_label, duration in time_ranges:
                query_id = window_label + time_label
                if args.only and query_id.upper() != args.only.upper():
                    continue
                if query_id in completed_query_ids:
                    continue
                query_number += 1
                print(
                    f"[{query_id}] edges={window['road_edge_count']}, "
                    f"cells={window['covered_cell_count']}, duration={duration}s"
                )
                query_end_timestamp = query_start_timestamp + int(duration)
                time_axis = trusted_manifest.get("time_axis")
                if time_axis:
                    origin = int(time_axis["start_timestamp"])
                    axis_end = int(time_axis["end_timestamp_exclusive"])
                    slot_seconds = int(time_axis["time_unit_seconds"])
                    if query_start_timestamp < origin or query_end_timestamp > axis_end:
                        raise ValueError(
                            f"{query_id} 查询区间超出索引时间轴: "
                            f"[{query_start_timestamp},{query_end_timestamp}) not in "
                            f"[{origin},{axis_end})"
                        )
                    qs = (query_start_timestamp - origin) // slot_seconds
                    qe = (query_end_timestamp - 1 - origin) // slot_seconds
                else:
                    day_start = int(
                        datetime.fromisoformat(dataset_date)
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    )
                    if query_start_timestamp < day_start or query_end_timestamp > day_start + 86400:
                        raise ValueError(
                            f"{query_id} 查询区间超出单日索引: "
                            f"[{query_start_timestamp},{query_end_timestamp})"
                        )
                    qs = query_start_timestamp - day_start
                    qe = query_end_timestamp - day_start - 1
                for warmup_index in range(args.warmup):
                    warmup = execute_once(conns, window["edge_ids"], qs, qe)
                    print(
                        f"  warmup {warmup_index + 1}: query={warmup['query_seconds']:.6f}s, "
                        f"verify={warmup['verify_seconds']:.6f}s, all_ok={warmup['all_ok']}"
                    )
                runs = []
                for repeat_index in range(args.repeats):
                    run = execute_once(conns, window["edge_ids"], qs, qe)
                    runs.append(run)
                    print(
                        f"  run {repeat_index + 1}: trajectories={run['result_trajectory_count']}, "
                        f"query={run['query_seconds']:.6f}s, "
                        f"verify={run['verify_seconds']:.6f}s, all_ok={run['all_ok']}"
                    )
                summary = summarize_runs(runs)
                summary.update(
                    {
                        "query_id": query_id,
                        "window": window_label,
                        "time_level": time_label,
                        "span_degrees": window["span_degrees"],
                        "duration_seconds": duration,
                        "query_start_timestamp": query_start_timestamp,
                        "query_end_timestamp": query_end_timestamp,
                        "query_slot_start": qs,
                        "query_slot_end_inclusive": qe,
                        "requested_bounds": window["requested_bounds"],
                        "covered_cell_count": window["covered_cell_count"],
                        "road_edge_count": window["road_edge_count"],
                    }
                )
                results.append(summary)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("w", encoding="utf-8") as checkpoint_file:
                    json.dump(
                        {
                            "completed_query_count": len(results),
                            "expected_query_count": len(WINDOWS) * len(time_ranges) if not args.only else 1,
                            "results": results,
                        },
                        checkpoint_file,
                        ensure_ascii=False,
                        indent=2,
                    )

        payload = {
            "schema_version": 1,
            "primary_statistic": "arithmetic_mean",
            "metric_units": {
                "query_time": "seconds",
                "verification_time": "seconds",
                "vo_size": "bytes",
            },
            "dataset_date": dataset_date,
            "timezone": "UTC",
            "query_start_timestamp": query_start_timestamp,
            "query_interval_semantics": "left_closed_right_open",
            "center": [args.center_lon, args.center_lat],
            "cell_size_m": 50.0,
            "index_file": str(input_bin),
            "trusted_manifest": str(manifest_path),
            "trusted_data_ac": trusted_manifest["data_root"]["data_ac"],
            "query_module": QUERY_MODULE,
            "workers": len(assignments),
            "warmup_count": args.warmup,
            "repeat_count": args.repeats,
            "index_scan_seconds": scan_seconds,
            "worker_load_seconds": worker_load_seconds,
            "rsa_metadata": json_safe_rsa_metadata(rsa_metadata),
            "grid_metadata": grid_metadata,
            "windows": windows,
            "results": results,
        }
        output_stem = (
            requested_output_stem
            if not args.only
            else f"{requested_output_stem}-{args.only.upper()}-smoke"
        )
        write_outputs(dataset_dir / "experiment_results", payload, output_stem)
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
            try:
                conn.close()
            except Exception:
                pass
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    main()
