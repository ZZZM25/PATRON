"""Run the canonical Beijing W1-W5 RPMT-SMT experiment."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from baseline.persistence import load_index
from baseline.smt_mode import SmtBaselineServerQuery, is_per_edge_smt_index
from baseline.verifier import verify_vo
from baseline.vo_io import serialize_vo


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_RESULT_DIR = PROJECT_DIR / "outputs" / "indexes" / "beijing"
INDEX_FILE = SOURCE_RESULT_DIR / "beijing_cell50_edge_smt_index.dat"
TRUSTED_ROOT_FILE = SOURCE_RESULT_DIR / "trusted_root.txt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "beijing_query_experiment"
BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

SPATIAL_WINDOWS = (
    ("W1", (116.325000, 39.975000, 116.330000, 39.980000)),
    ("W2", (116.323750, 39.973750, 116.331250, 39.981250)),
    ("W3", (116.322500, 39.972500, 116.332500, 39.982500)),
    ("W4", (116.320000, 39.970000, 116.335000, 39.985000)),
    ("W5", (116.317500, 39.967500, 116.337500, 39.987500)),
)

# Common half-open Unix intervals selected from the original Beijing date.
DEFAULT_TIME_WINDOWS = (
    ("1h", 1185897600, 1185901200),   # 2007-08-01 00:00-01:00 UTC+8
    ("2h", 1185897600, 1185904800),   # 2007-08-01 00:00-02:00 UTC+8
    ("4h", 1185897600, 1185912000),   # 2007-08-01 00:00-04:00 UTC+8
    ("8h", 1185922800, 1185951600),   # 2007-08-01 07:00-15:00 UTC+8
    ("12h", 1185897600, 1185940800),  # 2007-08-01 00:00-12:00 UTC+8
)

ALL_TIME_WINDOWS = (
    ("1h", 1185897600, 1185901200),  # 2007-08-01 00:00-01:00 UTC+8
    ("2h", 1185897600, 1185904800),  # 2007-08-01 00:00-02:00 UTC+8
    ("4h", 1185897600, 1185912000),  # 2007-08-01 00:00-04:00 UTC+8
    ("8h", 1185922800, 1185951600),  # 2007-08-01 07:00-15:00 UTC+8
    ("12h", 1185897600, 1185940800),  # 2007-08-01 00:00-12:00 UTC+8
)


def _format_local(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, BEIJING_TZ).isoformat()


def _write_outputs(
    output_dir: Path,
    metadata: dict,
    rows: list[dict],
    time_windows: tuple[tuple[str, int, int], ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({**metadata, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not rows:
        return
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for filename, field in (
        ("query_time_matrix_s.csv", "mean_query_seconds"),
        ("verification_time_matrix_s.csv", "mean_verification_seconds"),
        ("vo_size_matrix_bytes.csv", "vo_bytes"),
    ):
        lookup = {
            (row["spatial_window"], row["time_window"]): row[field]
            for row in rows
        }
        with (output_dir / filename).open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.writer(stream)
            labels = [label for label, _start, _end in time_windows]
            writer.writerow(["spatial_window", *labels])
            for spatial_label, _ in SPATIAL_WINDOWS:
                writer.writerow(
                    [spatial_label]
                    + [lookup.get((spatial_label, label), "") for label in labels]
                )


def benchmark(
    repeats: int,
    warmups: int,
    output_dir: Path,
    index_file: Path = INDEX_FILE,
    trusted_root_file: Path = TRUSTED_ROOT_FILE,
    time_windows: tuple[tuple[str, int, int], ...] = DEFAULT_TIME_WINDOWS,
) -> None:
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    load_started = perf_counter()
    index = load_index(index_file)
    index_load_seconds = perf_counter() - load_started
    if not is_per_edge_smt_index(index):
        raise TypeError("expected the compatible per-edge SMT index")
    trusted_root = bytes.fromhex(trusted_root_file.read_text(encoding="ascii").strip())
    if index.root_hash != trusted_root:
        raise RuntimeError("index root does not match trusted root")

    metadata = {
        "schema_version": 1,
        "primary_statistic": "arithmetic_mean",
        "metric_units": {
            "query_time": "seconds",
            "verification_time": "seconds",
            "vo_size": "bytes",
            "vo_size_display": "MiB = bytes / 1024^2",
        },
        "method": "RPMT-SMT per-edge independent SMT baseline",
        "experiment": "beijing_w1_w5_1h_2h_4h_8h_12h",
        "local_date": "2007-08-01",
        "timezone": "UTC+08:00",
        "interval_semantics": (
            "Published timestamps are half-open [start,end); this implementation "
            "uses inclusive seconds internally, so query_end_internal=end-1."
        ),
        "spatial_windows": [
            {"label": label, "bounds": list(bounds)}
            for label, bounds in SPATIAL_WINDOWS
        ],
        "time_windows": [
            {
                "label": label,
                "start_ts": start,
                "end_ts": end,
                "duration_seconds": end - start,
                "start_local": _format_local(start),
                "end_local": _format_local(end),
            }
            for label, start, end in time_windows
        ],
        "repeats": repeats,
        "warmups": warmups,
        "index_load_seconds_excluded": index_load_seconds,
        "index_file": str(index_file),
        "trusted_root_hex": trusted_root.hex(),
        "timing_scope": (
            "Query includes server spatial/temporal query and independent per-edge "
            "SMT proof generation. Verification is client cryptographic VO "
            "verification. Index load, serialization, and file I/O are excluded."
        ),
    }
    rows: list[dict] = []
    total = len(SPATIAL_WINDOWS) * len(time_windows)
    current = 0
    for spatial_label, bounds in SPATIAL_WINDOWS:
        for time_label, query_start, query_end_exclusive in time_windows:
            current += 1
            # RPMT-SMT's overlap predicate is inclusive at both ends.
            query_end_internal = query_end_exclusive - 1
            query_args = (*bounds, query_start, query_end_internal)

            for _ in range(warmups):
                response = SmtBaselineServerQuery(index, *query_args).build()
                report = verify_vo(
                    response.vo,
                    index.road,
                    trusted_root,
                    *query_args,
                    response.candidate_ids,
                )
                if not report.success:
                    raise RuntimeError(
                        f"warmup verification failed: {spatial_label} x {time_label}"
                    )

            query_seconds = []
            verification_seconds = []
            last_response = None
            for repeat in range(1, repeats + 1):
                started = perf_counter()
                response = SmtBaselineServerQuery(index, *query_args).build()
                query_seconds.append(perf_counter() - started)

                started = perf_counter()
                report = verify_vo(
                    response.vo,
                    index.road,
                    trusted_root,
                    *query_args,
                    response.candidate_ids,
                )
                verification_seconds.append(perf_counter() - started)
                if not report.success:
                    raise RuntimeError(
                        f"verification failed: {spatial_label} x {time_label} "
                        f"repeat {repeat}"
                    )
                last_response = response

            assert last_response is not None
            raw_vo = serialize_vo(last_response.vo)
            proofs = last_response.vo.smt_proofs
            if not isinstance(proofs, list):
                raise TypeError("query did not return per-edge SMT proofs")
            row = {
                "method": "RPMT-SMT",
                "spatial_window": spatial_label,
                "min_lon": bounds[0],
                "min_lat": bounds[1],
                "max_lon": bounds[2],
                "max_lat": bounds[3],
                "time_window": time_label,
                "start_ts": query_start,
                "end_ts": query_end_exclusive,
                "duration_seconds": query_end_exclusive - query_start,
                "start_local": _format_local(query_start),
                "end_local": _format_local(query_end_exclusive),
                "mean_query_seconds": statistics.fmean(query_seconds),
                "mean_verification_seconds": statistics.fmean(
                    verification_seconds
                ),
                "vo_bytes": len(raw_vo),
                "vo_mib": len(raw_vo) / (1024 * 1024),
                "trajectory_count": len(last_response.candidate_ids),
                "spatial_matched_edge_count": (
                    last_response.stats.spatial_matched_edges
                ),
                "scanned_entry_count": last_response.stats.scanned_entries,
                "proof_count": len(proofs),
                "verification_success": True,
                "query_seconds_runs": query_seconds,
                "verification_seconds_runs": verification_seconds,
            }
            rows.append(row)
            _write_outputs(output_dir, metadata, rows, time_windows)
            print(
                f"[{current}/{total}] {spatial_label} x {time_label}: "
                f"query={row['mean_query_seconds']:.6f}s "
                f"verify={row['mean_verification_seconds']:.6f}s "
                f"VO={row['vo_bytes']} bytes",
                flush=True,
            )

    print(f"BENCHMARK_SUCCESS output={output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--time-windows",
        default="1h,2h,4h,8h,12h",
        help="comma-separated labels chosen from 1h,2h,4h,8h,12h",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--trusted-root", type=Path, default=TRUSTED_ROOT_FILE)
    args = parser.parse_args()
    available = {item[0]: item for item in ALL_TIME_WINDOWS}
    labels = [value.strip() for value in args.time_windows.split(",") if value.strip()]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("time-window labels must be nonempty and unique")
    unknown = [label for label in labels if label not in available]
    if unknown:
        raise ValueError(f"unknown time-window labels: {unknown}")
    selected = tuple(available[label] for label in labels)
    benchmark(
        args.repeats,
        args.warmups,
        args.output.resolve(),
        args.index.resolve(),
        args.trusted_root.resolve(),
        selected,
    )


if __name__ == "__main__":
    main()
