"""Run each rectangle/time benchmark cell in a fresh process.

The worker loads the authenticated index, performs one or more warmups and three
measured repetitions for exactly one spatial/temporal cell, writes its result,
and exits.  The lightweight parent then starts the next worker.  This prevents
Python allocator retention from one cell affecting later cells.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from baseline.benchmark_utils import TEMPORAL_RANGES
from baseline.cell_query import (
    CELL_COVER_QUERY_MODE,
    cell_cover_policy_from_config,
)
from baseline.config import load_config, resolve_path
from baseline.persistence import load_index
from baseline.smt_mode import (
    CellCoverSmtBaselineServerQuery,
    is_per_edge_smt_index,
)
from baseline.verifier import verify_vo
from baseline.vo_io import serialize_vo


def _execution_counts(benchmark: dict, window_label: str) -> tuple[int, int]:
    """Resolve per-window run counts while preserving the default protocol."""
    overrides = benchmark.get("window_execution_overrides", {})
    override = overrides.get(window_label, {}) if isinstance(overrides, dict) else {}
    repeats = int(override.get("repeats", benchmark.get("repeats", 3)))
    warmups = int(
        override.get("warmup_repeats", benchmark.get("warmup_repeats", 1))
    )
    if repeats < 1 or warmups < 0:
        raise ValueError(
            f"invalid execution counts for {window_label}: "
            f"repeats={repeats}, warmups={warmups}"
        )
    return repeats, warmups


def _write_matrix(path: Path, rows: list[dict], field: str) -> None:
    labels = list(dict.fromkeys(str(row["spatial_window"]) for row in rows))
    lookup = {
        (str(row["spatial_window"]), str(row["temporal_label"])): row[field]
        for row in rows
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["spatial_window", *(label for label, _ in TEMPORAL_RANGES)])
        for window_label in labels:
            writer.writerow(
                [
                    window_label,
                    *(lookup.get((window_label, label), "") for label, _ in TEMPORAL_RANGES),
                ]
            )


def _validate_index(cfg: dict) -> tuple[object, bytes, object]:
    index = load_index(resolve_path(cfg, cfg["index_file"]))
    if not is_per_edge_smt_index(index):
        raise ValueError("obsolete/global SMT index")
    if index.root_hash != index.tree.node_hash[index.tree.root_index]:
        raise ValueError("index root is not the single authenticated spatial root")
    trusted = bytes.fromhex(
        resolve_path(cfg, cfg["trusted_root_file"])
        .read_text(encoding="ascii")
        .strip()
    )
    if index.root_hash != trusted:
        raise ValueError("index root differs from published trusted root")
    policy = cell_cover_policy_from_config(cfg, index.road)
    return index, trusted, policy


def _run_worker(
    config_path: Path,
    window_index: int,
    temporal_index: int,
    result_file: Path,
) -> None:
    cfg = load_config(config_path)
    benchmark = cfg["benchmark"]
    windows = benchmark["spatial_windows"]
    window = windows[window_index]
    temporal_label, seconds = TEMPORAL_RANGES[temporal_index]
    label = str(window["label"])
    bounds = tuple(float(value) for value in window["bounds"])
    if len(bounds) != 4:
        raise ValueError(f"{label}: bounds must have four values")
    repeats, warmups = _execution_counts(benchmark, label)
    query_start = int(benchmark["base_query_start"])
    query = (*bounds, query_start, query_start + seconds)

    load_started = perf_counter()
    index, trusted, cell_policy = _validate_index(cfg)
    index_load_s = perf_counter() - load_started
    cover = cell_policy.cover_rectangle(*bounds)

    for _ in range(warmups):
        response = CellCoverSmtBaselineServerQuery(
            index, cell_policy, *query
        ).build()
        report = verify_vo(
            response.vo,
            index.road,
            trusted,
            *query,
            response.candidate_ids,
            cell_policy=cell_policy,
        )
        if not report.success:
            raise RuntimeError(
                f"warmup verification failed for {label} x {temporal_label}: "
                f"{report.reason}"
            )
    if warmups:
        del response, report
        gc.collect()

    query_times: list[float] = []
    query_wall_times: list[float] = []
    edge_load_times: list[float] = []
    smt_materialization_times: list[float] = []
    verification_times: list[float] = []
    run_rows: list[dict] = []
    last_response = None
    for run_number in range(1, repeats + 1):
        started = perf_counter()
        response = CellCoverSmtBaselineServerQuery(
            index, cell_policy, *query
        ).build()
        query_wall_time = perf_counter() - started
        edge_load_time = response.stats.edge_data_load_time_s
        smt_materialization_time = response.stats.smt_materialization_time_s
        query_time = max(
            0.0,
            query_wall_time - edge_load_time - smt_materialization_time,
        )

        started = perf_counter()
        report = verify_vo(
            response.vo,
            index.road,
            trusted,
            *query,
            response.candidate_ids,
            cell_policy=cell_policy,
        )
        verification_time = perf_counter() - started
        if not report.success:
            raise RuntimeError(
                f"verification failed for {label} x {temporal_label} run "
                f"{run_number}: {report.reason}"
            )
        query_times.append(query_time)
        query_wall_times.append(query_wall_time)
        edge_load_times.append(edge_load_time)
        smt_materialization_times.append(smt_materialization_time)
        verification_times.append(verification_time)
        run_rows.append(
            {
                "run": run_number,
                "query_time_s": query_time,
                "query_wall_time_s": query_wall_time,
                "edge_data_load_time_excluded_s": edge_load_time,
                "smt_materialization_time_excluded_s": smt_materialization_time,
                "verification_time_s": verification_time,
            }
        )
        if run_number == repeats:
            last_response = response
        else:
            # Do not retain the preceding VO while constructing the next one.
            # This keeps the five repetitions independent even for 100+ MB VOs.
            del response, report
            gc.collect()

    assert last_response is not None
    vo_bytes = len(serialize_vo(last_response.vo))
    row = {
        "spatial_window": label,
        "spatial_side_degrees": float(window["side_degrees"]),
        "query_min_lon": bounds[0],
        "query_min_lat": bounds[1],
        "query_max_lon": bounds[2],
        "query_max_lat": bounds[3],
        "temporal_label": temporal_label,
        "temporal_seconds": seconds,
        "query_start": query_start,
        "query_end": query_start + seconds,
        "spatial_query_mode": CELL_COVER_QUERY_MODE,
        "cover_min_cell_x": cover.min_cell_x,
        "cover_min_cell_y": cover.min_cell_y,
        "cover_max_cell_x": cover.max_cell_x,
        "cover_max_cell_y": cover.max_cell_y,
        "cover_cell_count": cover.cell_count,
        "effective_query_min_lon": cover.effective_bounds[0],
        "effective_query_min_lat": cover.effective_bounds[1],
        "effective_query_max_lon": cover.effective_bounds[2],
        "effective_query_max_lat": cover.effective_bounds[3],
        "repeats": repeats,
        "primary_statistic": "arithmetic_mean",
        "warmup_repeats": warmups,
        "isolation": "fresh_process_per_spatial_temporal_cell",
        "index_load_time_excluded_s": index_load_s,
        "query_time_s": statistics.mean(query_times),
        "query_time_std_s": statistics.stdev(query_times) if repeats > 1 else 0.0,
        "query_wall_time_s": statistics.mean(query_wall_times),
        "edge_data_load_time_excluded_s": statistics.mean(edge_load_times),
        "smt_materialization_time_excluded_s": statistics.mean(
            smt_materialization_times
        ),
        "verification_time_s": statistics.mean(verification_times),
        "verification_time_std_s": (
            statistics.stdev(verification_times) if repeats > 1 else 0.0
        ),
        "vo_size_bytes": vo_bytes,
        "vo_size_mib": vo_bytes / (1024 * 1024),
        "result_count": len(last_response.candidate_ids),
        "smt_proof_count": len(last_response.vo.smt_proofs),
        "smt_proven_edge_count": len(
            {proof.eid for proof in last_response.vo.smt_proofs}
        ),
        "smt_sibling_hash_count": sum(
            len(proof.siblings) for proof in last_response.vo.smt_proofs
        ),
        "verification_set_count": len(last_response.vo.verification_set),
        "verification_success": True,
        "runs": run_rows,
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")


def _write_outputs(destination: Path, rows: list[dict]) -> None:
    if not rows:
        return
    flat_rows = [{key: value for key, value in row.items() if key != "runs"} for row in rows]
    with (destination / "edge_smt_rect_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    run_rows: list[dict] = []
    for row in rows:
        for run in row["runs"]:
            run_rows.append(
                {
                    "spatial_window": row["spatial_window"],
                    "temporal_label": row["temporal_label"],
                    **run,
                }
            )
    with (destination / "edge_smt_rect_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    for filename, field in (
        ("query_time_matrix_s.csv", "query_time_s"),
        ("verification_time_matrix_s.csv", "verification_time_s"),
        ("vo_size_matrix_mib.csv", "vo_size_mib"),
        ("result_count_matrix.csv", "result_count"),
    ):
        _write_matrix(destination / filename, rows, field)


def _run_parent(config_path: Path) -> None:
    cfg = load_config(config_path)
    benchmark = cfg["benchmark"]
    windows = benchmark["spatial_windows"]
    destination = resolve_path(cfg, benchmark["output_dir"])
    cell_dir = destination / "isolated_cells"
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = len(windows) * len(TEMPORAL_RANGES)
    current = 0
    for window_index, window in enumerate(windows):
        for temporal_index, (temporal_label, _) in enumerate(TEMPORAL_RANGES):
            current += 1
            label = str(window["label"])
            expected_repeats, expected_warmups = _execution_counts(benchmark, label)
            cell_file = cell_dir / f"{window_index:02d}_{temporal_index:02d}.json"
            if cell_file.is_file():
                row = json.loads(cell_file.read_text(encoding="utf-8"))
                if (
                    row.get("spatial_window") == label
                    and row.get("temporal_label") == temporal_label
                    and row.get("repeats") == expected_repeats
                    and row.get("warmup_repeats") == expected_warmups
                    and row.get("isolation")
                    == "fresh_process_per_spatial_temporal_cell"
                    and len(row.get("runs", [])) == expected_repeats
                    and row.get("verification_success") is True
                    and row.get("spatial_query_mode") == CELL_COVER_QUERY_MODE
                ):
                    rows.append(row)
                    _write_outputs(destination, rows)
                    print(
                        f"[{current}/{total}] RESUME {label} x {temporal_label}: "
                        f"validated existing isolated cell",
                        flush=True,
                    )
                    continue
                raise ValueError(f"invalid existing isolated result: {cell_file}")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(config_path.resolve()),
                "--worker",
                "--window-index",
                str(window_index),
                "--temporal-index",
                str(temporal_index),
                "--result-file",
                str(cell_file),
            ]
            print(
                f"[{current}/{total}] START {label} x {temporal_label} "
                f"(fresh process, {expected_warmups} warmup + "
                f"{expected_repeats} runs)",
                flush=True,
            )
            subprocess.run(command, check=True)
            row = json.loads(cell_file.read_text(encoding="utf-8"))
            rows.append(row)
            _write_outputs(destination, rows)
            print(
                f"[{current}/{total}] DONE {label} x {temporal_label}: "
                f"query={row['query_time_s']:.6f}s "
                f"verify={row['verification_time_s']:.6f}s "
                f"VO={row['vo_size_mb']:.6f}MB "
                f"results={row['result_count']}",
                flush=True,
            )
    print(f"ISOLATED_BENCHMARK_SUCCESS results={destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--window-index", type=int)
    parser.add_argument("--temporal-index", type=int)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    if args.worker:
        if (
            args.window_index is None
            or args.temporal_index is None
            or args.result_file is None
        ):
            parser.error("worker mode requires window/temporal indexes and result file")
        _run_worker(
            args.config.resolve(),
            args.window_index,
            args.temporal_index,
            args.result_file.resolve(),
        )
    else:
        _run_parent(args.config.resolve())


if __name__ == "__main__":
    main()
