"""Run the canonical Beijing W1-W5 RSA/RSG experiment."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import multiprocessing as mp
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import run_rsg_50m_experiment_matrix as matrix


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "rsg_50m_beijing"
SPATIAL_DATASET_DIR = DATASET_DIR / "query_road"
UNIT_DIR = DATASET_DIR / "bimonthly_units"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "beijing_rsa_query_experiment"
BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

SPATIAL_WINDOWS = (
    ("W1", (116.325000, 39.975000, 116.330000, 39.980000)),
    ("W2", (116.323750, 39.973750, 116.331250, 39.981250)),
    ("W3", (116.322500, 39.972500, 116.332500, 39.982500)),
    ("W4", (116.320000, 39.970000, 116.335000, 39.985000)),
    ("W5", (116.317500, 39.967500, 116.337500, 39.987500)),
)

DEFAULT_TIME_WINDOWS = (
    ("1h", 1185897600, 1185901200),   # 2007-08-01 00:00-01:00 UTC+8
    ("2h", 1185897600, 1185904800),   # 2007-08-01 00:00-02:00 UTC+8
    ("4h", 1185897600, 1185912000),   # 2007-08-01 00:00-04:00 UTC+8
    ("8h", 1185922800, 1185951600),   # 2007-08-01 07:00-15:00 UTC+8
    ("12h", 1185897600, 1185940800),  # 2007-08-01 00:00-12:00 UTC+8
)

ALL_TIME_WINDOWS = (
    ("1h", 1185897600, 1185901200),   # 2007-08-01 00:00-01:00 UTC+8
    ("2h", 1185897600, 1185904800),   # 2007-08-01 00:00-02:00 UTC+8
    ("4h", 1185897600, 1185912000),   # 2007-08-01 00:00-04:00 UTC+8
    ("8h", 1185922800, 1185951600),   # 2007-08-01 07:00-15:00 UTC+8
    ("12h", 1185897600, 1185940800),  # 2007-08-01 00:00-12:00 UTC+8
)

UNIT_NAMES = ("2007-06_2007-07", "2007-08_2007-09")


def _format_local(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, BEIJING_TZ).isoformat()


class UnitRuntime:
    def __init__(self, name: str, windows: dict[str, dict], workers: int):
        self.name = name
        self.windows = windows
        self.index_file = UNIT_DIR / f"{name}.rsg-50m-private-token.bin"
        self.manifest_file = (
            UNIT_DIR / f"{name}.rsg-50m-private-token.trusted-manifest.json"
        )
        query_module = importlib.import_module(matrix.QUERY_MODULE)
        raw_manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        self.claim_date = str(raw_manifest["date"])
        self.trusted_manifest = matrix.verify_trusted_ads_manifest(
            self.manifest_file,
            SPATIAL_DATASET_DIR,
            self.index_file,
            query_module,
            expected_date=self.claim_date,
        )
        self.axis = dict(self.trusted_manifest["time_axis"])
        if int(self.axis["time_unit_seconds"]) != 60:
            raise ValueError(f"{name}: expected the existing one-minute RSA slots")

        scan_started = time.perf_counter()
        _rsa_metadata, edge_infos = query_module.scan_edges_for_date(
            str(self.index_file), self.claim_date
        )
        self.scan_seconds = time.perf_counter() - scan_started
        edge_info_by_id = {int(item["edge_id"]): item for item in edge_infos}
        assignments = matrix.assign_nested_window_edges(
            windows, edge_info_by_id, workers
        )

        self.conns = []
        self.processes = []
        context = mp.get_context("spawn")
        load_started = time.perf_counter()
        for edge_ids in assignments:
            parent, child = context.Pipe()
            process = context.Process(
                target=matrix.matrix_worker,
                args=(
                    child,
                    str(self.index_file),
                    edge_ids,
                    self.trusted_manifest,
                    self.claim_date,
                ),
            )
            process.start()
            child.close()
            self.conns.append(parent)
            self.processes.append(process)
        ready = []
        for conn in self.conns:
            reply = conn.recv()
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error"))
            ready.append(reply)
        self.worker_load_seconds = time.perf_counter() - load_started
        self.loaded_nonempty_edges = sum(
            int(item["loaded_nonempty_edge_count"]) for item in ready
        )
        print(
            f"[{name}] scan={self.scan_seconds:.3f}s "
            f"worker_load={self.worker_load_seconds:.3f}s "
            f"loaded_nonempty_edges={self.loaded_nonempty_edges}",
            flush=True,
        )

    @property
    def start_ts(self) -> int:
        return int(self.axis["start_timestamp"])

    @property
    def end_ts(self) -> int:
        return int(self.axis["end_timestamp_exclusive"])

    def overlaps(self, start_ts: int, end_ts: int) -> bool:
        return self.start_ts < end_ts and start_ts < self.end_ts

    def execute(self, spatial_label: str, start_ts: int, end_ts: int) -> dict:
        slice_start = max(int(start_ts), self.start_ts)
        slice_end = min(int(end_ts), self.end_ts)
        if slice_start >= slice_end:
            raise ValueError(f"{self.name}: empty query slice")
        slot_seconds = int(self.axis["time_unit_seconds"])
        if slice_start % 3600 or slice_end % 3600:
            raise ValueError("common Beijing query boundaries must align to clock hours")
        qs = (slice_start - self.start_ts) // slot_seconds
        qe = (slice_end - 1 - self.start_ts) // slot_seconds
        run = matrix.execute_once(
            self.conns, self.windows[spatial_label]["edge_ids"], qs, qe
        )
        run.update(
            {
                "unit": self.name,
                "slice_start_ts": slice_start,
                "slice_end_ts": slice_end,
                "query_slot_start": qs,
                "query_slot_end_inclusive": qe,
                "rsa_minute_slot_count": qe - qs + 1,
            }
        )
        return run

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send({"command": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for conn in self.conns:
            try:
                if conn.poll(5):
                    conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                conn.close()
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.conns.clear()
        self.processes.clear()
        gc.collect()


def _aggregate_slices(slices: list[dict]) -> dict:
    result_ids = {
        int(value)
        for item in slices
        for value in item.get("result_trajectory_ids", [])
    }
    return {
        "query_seconds": sum(float(item["query_seconds"]) for item in slices),
        "verify_seconds": sum(float(item["verify_seconds"]) for item in slices),
        "prepare_seconds_excluded": sum(
            float(item["prepare_seconds"]) for item in slices
        ),
        "total_vo_bytes": sum(int(item["total_vo_bytes"]) for item in slices),
        "member_vo_bytes": sum(int(item["member_vo_bytes"]) for item in slices),
        "invalid_items_vo_bytes": sum(
            int(item["invalid_items_vo_bytes"]) for item in slices
        ),
        "completeness_vo_bytes": sum(
            int(item["completeness_vo_bytes"]) for item in slices
        ),
        "result_trajectory_count": len(result_ids),
        "result_trajectory_ids": sorted(result_ids),
        "all_ok": all(bool(item["all_ok"]) for item in slices),
        "rsa_minute_slot_count": sum(
            int(item["rsa_minute_slot_count"]) for item in slices
        ),
        "period_count": len(slices),
        "periods": [item["unit"] for item in slices],
        "slices": slices,
    }


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
    csv_fields = [
        "method",
        "spatial_window",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
        "time_window",
        "start_ts",
        "end_ts",
        "duration_seconds",
        "start_local",
        "end_local",
        "kzg_hour_slot_count",
        "rsa_minute_slot_count",
        "period_count",
        "periods",
        "covered_cell_count",
        "road_edge_count",
        "trajectory_count",
        "mean_query_seconds",
        "mean_verification_seconds",
        "vo_bytes",
        "vo_mib",
        "verification_success",
    ]
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in csv_fields}
            flat["periods"] = json.dumps(flat["periods"], ensure_ascii=False)
            writer.writerow(flat)

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
                    [
                        spatial_label,
                        *[
                            lookup.get((spatial_label, time_label), "")
                            for time_label in labels
                        ],
                    ]
                )


def benchmark(
    repeats: int,
    warmups: int,
    workers: int,
    output_dir: Path,
    time_windows: tuple[tuple[str, int, int], ...] = DEFAULT_TIME_WINDOWS,
) -> None:
    if repeats < 1 or warmups < 0 or workers < 1:
        raise ValueError("invalid benchmark counts")
    windows, _grid = matrix.load_spatial_windows(
        SPATIAL_DATASET_DIR, center_lon=116.327500, center_lat=39.977500
    )
    # Pin the exact previous W1-W5 bounds rather than relying silently on the
    # center/span calculation.
    for label, expected_bounds in SPATIAL_WINDOWS:
        actual = windows[label]["requested_bounds"]
        if any(abs(float(a) - float(b)) > 1e-12 for a, b in zip(actual, expected_bounds)):
            raise ValueError(f"{label}: spatial bounds changed: {actual}")

    runtimes: list[UnitRuntime] = []
    try:
        for name in UNIT_NAMES:
            runtimes.append(UnitRuntime(name, windows, workers))

        metadata = {
            "schema_version": 1,
            "primary_statistic": "arithmetic_mean",
            "metric_units": {
                "query_time": "seconds",
                "verification_time": "seconds",
                "vo_size": "bytes",
                "vo_size_display": "MiB = bytes / 1024^2",
            },
            "method": "RSA/RSG private-token accumulator",
            "experiment": "beijing_w1_w5_" + "_".join(
                label for label, _start, _end in time_windows
            ),
            "local_date": "2007-08-01",
            "timezone": "UTC+08:00",
            "interval_semantics": "half-open [start,end)",
            "alignment": (
                "All common query boundaries are Beijing clock hours. KZG uses "
                "one-hour slots; the existing RSA index represents exactly the "
                "same half-open intervals with one-minute slots."
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
            "workers_per_period": workers,
            "timing_scope": (
                "Query is member plus completeness proof generation; verification "
                "is the corresponding client cryptographic verification. Spatial "
                "resolution, selected-edge preparation, index load, and file I/O "
                "are excluded, matching the existing RSA matrix experiment."
            ),
            "unit_loads_excluded": [
                {
                    "unit": runtime.name,
                    "scan_seconds": runtime.scan_seconds,
                    "worker_load_seconds": runtime.worker_load_seconds,
                }
                for runtime in runtimes
            ],
        }

        rows: list[dict] = []
        total = len(SPATIAL_WINDOWS) * len(time_windows)
        current = 0
        for spatial_label, bounds in SPATIAL_WINDOWS:
            window = windows[spatial_label]
            for time_label, query_start, query_end in time_windows:
                current += 1
                selected = [
                    runtime
                    for runtime in runtimes
                    if runtime.overlaps(query_start, query_end)
                ]
                if not selected:
                    raise ValueError(f"no RSA unit covers {time_label}")

                for warmup_index in range(warmups):
                    aggregate = _aggregate_slices(
                        [
                            runtime.execute(spatial_label, query_start, query_end)
                            for runtime in selected
                        ]
                    )
                    if not aggregate["all_ok"]:
                        raise RuntimeError(
                            f"warmup verification failed: {spatial_label} x {time_label}"
                        )
                    print(
                        f"  warmup {warmup_index + 1} {spatial_label} x {time_label}: "
                        f"query={aggregate['query_seconds']:.6f}s "
                        f"verify={aggregate['verify_seconds']:.6f}s",
                        flush=True,
                    )

                runs = []
                for repeat in range(1, repeats + 1):
                    aggregate = _aggregate_slices(
                        [
                            runtime.execute(spatial_label, query_start, query_end)
                            for runtime in selected
                        ]
                    )
                    if not aggregate["all_ok"]:
                        raise RuntimeError(
                            f"verification failed: {spatial_label} x {time_label} "
                            f"repeat {repeat}"
                        )
                    runs.append(aggregate)
                    print(
                        f"  run {repeat} {spatial_label} x {time_label}: "
                        f"query={aggregate['query_seconds']:.6f}s "
                        f"verify={aggregate['verify_seconds']:.6f}s "
                        f"VO={aggregate['total_vo_bytes']} bytes",
                        flush=True,
                    )

                row = {
                    "method": "RSA",
                    "spatial_window": spatial_label,
                    "min_lon": bounds[0],
                    "min_lat": bounds[1],
                    "max_lon": bounds[2],
                    "max_lat": bounds[3],
                    "time_window": time_label,
                    "start_ts": query_start,
                    "end_ts": query_end,
                    "duration_seconds": query_end - query_start,
                    "start_local": _format_local(query_start),
                    "end_local": _format_local(query_end),
                    "kzg_hour_slot_count": (query_end - query_start) // 3600,
                    "rsa_minute_slot_count": runs[-1]["rsa_minute_slot_count"],
                    "period_count": runs[-1]["period_count"],
                    "periods": runs[-1]["periods"],
                    "covered_cell_count": window["covered_cell_count"],
                    "road_edge_count": window["road_edge_count"],
                    "trajectory_count": runs[-1]["result_trajectory_count"],
                    "mean_query_seconds": statistics.fmean(
                        run["query_seconds"] for run in runs
                    ),
                    "mean_verification_seconds": statistics.fmean(
                        run["verify_seconds"] for run in runs
                    ),
                    "vo_bytes": runs[-1]["total_vo_bytes"],
                    "vo_mib": runs[-1]["total_vo_bytes"] / (1024 * 1024),
                    "verification_success": all(run["all_ok"] for run in runs),
                    "runs": runs,
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
    finally:
        for runtime in runtimes:
            runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--time-windows",
        default=",".join(label for label, _start, _end in DEFAULT_TIME_WINDOWS),
        help="comma-separated labels chosen from 1h,2h,4h,8h,12h",
    )
    args = parser.parse_args()
    window_lookup = {item[0]: item for item in ALL_TIME_WINDOWS}
    labels = [label.strip() for label in args.time_windows.split(",") if label.strip()]
    if not labels or len(labels) != len(set(labels)):
        parser.error("--time-windows must contain unique, non-empty labels")
    unknown = [label for label in labels if label not in window_lookup]
    if unknown:
        parser.error(f"unknown time-window labels: {', '.join(unknown)}")
    selected_time_windows = tuple(window_lookup[label] for label in labels)
    benchmark(
        args.repeats,
        args.warmups,
        args.workers,
        args.output.resolve(),
        selected_time_windows,
    )


if __name__ == "__main__":
    main()
