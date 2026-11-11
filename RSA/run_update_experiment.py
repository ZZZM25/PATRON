from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import shutil
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

from gmpy2 import mpz, powmod

import prepare_rsg_dataset as prepare
from build_rsg_index import validate_binary
from run_build_scaling_experiment import (
    configure_unit,
    find_token_key,
    first_timestamp,
    save_json,
)


ROOT = Path(__file__).resolve().parent
BUILDER_MODULE = "rsa_index_builder"
INSERT_SIZES = (1000, 2000, 5000, 8000, 10000)
BASE_FRACTION = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure pure RSA ADS update time from a chronological 60% baseline"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of formal update runs per measurement (default: 3)",
    )
    parser.add_argument(
        "--city",
        action="append",
        choices=("chengdu", "xian", "beijing"),
        help="repeatable; default runs all datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "rsa_update_experiment",
    )
    parser.add_argument(
        "--chengdu-source-dir",
        type=Path,
        default=ROOT / "data" / "chengdu" / "source",
        help="folder containing Chengdu traj-10-*.json files",
    )
    parser.add_argument(
        "--xian-source-dir",
        type=Path,
        default=ROOT / "data" / "xian" / "source",
        help="folder containing Xi'an traj_mapped_xian_xian10-*.json files",
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def count_json_records(path_text: str) -> tuple[str, int, int, int]:
    path = Path(path_text)
    count = sum(1 for _ in prepare.iter_trajectory_items(path))
    stat = path.stat()
    return str(path.resolve()), count, stat.st_size, stat.st_mtime_ns


def source_day(path: Path) -> int:
    match = re.search(r"10-(\d+)\.json$", path.name)
    if match is None:
        raise ValueError(f"cannot parse October day from {path}")
    return int(match.group(1))


def daily_source_units(city: str, source_dir: Path) -> list[dict]:
    if city == "chengdu":
        folder = source_dir
        paths = list(folder.glob("traj-10-*.json"))
    elif city == "xian":
        folder = source_dir
        paths = list(folder.glob("traj_mapped_xian_xian10-*.json"))
    else:
        raise ValueError(city)
    units = []
    for path in sorted(paths, key=source_day):
        day = source_day(path)
        units.append(
            {
                "name": f"2018-10-{day:02d}",
                "date": f"2018-10-{day:02d}",
                "source": str(path.resolve()),
                "mode": "daily",
            }
        )
    if len(units) != 31:
        raise ValueError(f"{city} expected 31 October files, found {len(units)}")
    return units


def scan_csv_count(path: Path) -> int:
    csv.field_size_limit(2_147_483_647)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def beijing_units(dataset: Path) -> list[dict]:
    summary = json.loads(
        (dataset / "bimonthly_units" / "bimonthly_mapping_summary.json").read_text(
            encoding="utf-8"
        )
    )
    units = []
    for item in summary["units"]:
        units.append(
            {
                "name": item["name"],
                "mode": "bimonthly",
                "mapped_csv": item["trajectory_csv"],
                "count": int(item["trajectory_count"]),
                "first_month": item["first_month"],
                "start_timestamp": int(item["start_timestamp"]),
                "end_timestamp_exclusive": int(item["end_timestamp_exclusive"]),
                "time_unit_seconds": int(item["time_unit_seconds"]),
                "time_slot_count": int(item["time_slot_count"]),
            }
        )
    return units


def dataset_configs(chengdu_source_dir: Path, xian_source_dir: Path) -> dict[str, dict]:
    chengdu = ROOT / "rsg_50m_2018-10-10"
    xian = ROOT / "rsg_50m_xian_2018-10-01-roadtxt"
    beijing = ROOT / "rsg_50m_beijing"
    return {
        "chengdu": {
            "city": "chengdu",
            "display": "成都",
            "dataset": chengdu,
            "edge_csv": chengdu / "new_edges.csv",
            "units": daily_source_units("chengdu", chengdu_source_dir),
        },
        "xian": {
            "city": "xian",
            "display": "西安",
            "dataset": xian,
            "edge_csv": xian / "new_edges.csv",
            "units": daily_source_units("xian", xian_source_dir),
        },
        "beijing": {
            "city": "beijing",
            "display": "北京",
            "dataset": beijing,
            "edge_csv": beijing / "new_edges_used.csv",
            "units": beijing_units(beijing),
        },
    }


def populate_counts(
    configs: dict[str, dict], selected_cities: list[str], inventory_path: Path, workers: int
) -> None:
    inventory = (
        json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory_path.exists()
        else {}
    )
    pending = []
    for city in selected_cities:
        for unit in configs[city]["units"]:
            if unit["mode"] != "daily":
                path = Path(unit["mapped_csv"])
                actual = scan_csv_count(path)
                if actual != int(unit["count"]):
                    raise ValueError(f"Beijing count mismatch for {path}: {actual} != {unit['count']}")
                continue
            path = Path(unit["source"])
            stat = path.stat()
            cached = inventory.get(str(path.resolve()))
            if (
                cached is not None
                and int(cached["source_size"]) == stat.st_size
                and int(cached["source_mtime_ns"]) == stat.st_mtime_ns
            ):
                unit["count"] = int(cached["count"])
            else:
                pending.append((city, unit, path))
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(count_json_records, str(path)) for _, _, path in pending]
            for (city, unit, path), future in zip(pending, futures):
                resolved, count, size, mtime_ns = future.result()
                unit["count"] = int(count)
                inventory[resolved] = {
                    "city": city,
                    "unit": unit["name"],
                    "count": int(count),
                    "source_size": int(size),
                    "source_mtime_ns": int(mtime_ns),
                }
                save_json(inventory_path, inventory)


def locate_baseline(config: dict) -> dict:
    total = sum(int(unit["count"]) for unit in config["units"])
    baseline_count = math.floor(total * BASE_FRACTION)
    cumulative = 0
    cutoff_index = None
    cutoff_offset = None
    for index, unit in enumerate(config["units"]):
        next_total = cumulative + int(unit["count"])
        if baseline_count <= next_total:
            cutoff_index = index
            cutoff_offset = baseline_count - cumulative
            break
        cumulative = next_total
    if cutoff_index is None or cutoff_offset is None:
        raise RuntimeError("failed to locate 60% baseline")
    remaining = total - baseline_count
    if remaining < max(INSERT_SIZES):
        raise ValueError(
            f"{config['display']} has only {remaining} records after 60%, fewer than 10000"
        )
    return {
        "total": total,
        "baseline_count": baseline_count,
        "cutoff_index": cutoff_index,
        "cutoff_unit": config["units"][cutoff_index]["name"],
        "cutoff_offset": cutoff_offset,
        "cutoff_unit_count": int(config["units"][cutoff_index]["count"]),
        "remaining": remaining,
    }


def ensure_network_files(dataset: Path, prepared: Path) -> None:
    prepared.mkdir(parents=True, exist_ok=True)
    for name in (
        "source_edges_nondegenerate.txt",
        "subedges_grouped.csv",
        "new_edges.csv",
        "new_nodes.csv",
    ):
        source = dataset / name
        target = prepared / name
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def prepare_needed_mapped_csvs(config: dict, plan: dict, prepared_root: Path) -> None:
    cutoff = int(plan["cutoff_index"])
    needed = int(plan["cutoff_offset"]) + max(INSERT_SIZES)
    index = cutoff
    while needed > 0 and index < len(config["units"]):
        unit = config["units"][index]
        if unit["mode"] == "daily":
            prepared = prepared_root / config["city"]
            ensure_network_files(Path(config["dataset"]), prepared)
            output_name = f"traj-{unit['date'][5:]}.csv"
            output_path = prepared / output_name
            if not output_path.exists():
                prepare.map_trajectories_streaming(
                    prepared / "source_edges_nondegenerate.txt",
                    Path(unit["source"]),
                    prepared,
                    output_name=output_name,
                    failed_name=f"failed-{unit['date'][5:]}.csv",
                    summary_name=f"summary-{unit['date'][5:]}.json",
                )
            unit["mapped_csv"] = str(output_path.resolve())
        mapped_count = scan_csv_count(Path(unit["mapped_csv"]))
        if mapped_count != int(unit["count"]):
            raise ValueError(
                f"mapped row count mismatch for {unit['name']}: {mapped_count} != {unit['count']}"
            )
        needed -= int(unit["count"]) if index > cutoff else int(unit["count"])
        index += 1
    if needed > 0:
        raise ValueError(f"not enough mapped records after cutoff for {config['display']}")


def sorted_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    csv.field_size_limit(2_147_483_647)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(path)
        fieldnames = list(reader.fieldnames)
        for sequence, row in enumerate(reader):
            payload = row["trajectory_data"]
            try:
                timestamp = first_timestamp(payload)
            except Exception:
                timestamp = 2**63 - 1
            rows.append((timestamp, sequence, row))
    rows.sort(key=lambda item: (item[0], item[1]))
    return fieldnames, [row for _, _, row in rows]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unit_slice_csv(
    unit: dict, start: int, count: int, target: Path, row_cache: dict[str, tuple[list[str], list[dict]]]
) -> Path:
    source = Path(unit["mapped_csv"])
    key = str(source.resolve())
    if key not in row_cache:
        row_cache[key] = sorted_csv_rows(source)
    fieldnames, rows = row_cache[key]
    selected = rows[start : start + count]
    if len(selected) != count:
        raise ValueError(f"slice {start}:{start + count} exceeds {source}")
    write_rows(target, fieldnames, selected)
    return target


def scenario_selection(config: dict, plan: dict, insert_count: int) -> list[dict]:
    remaining = insert_count
    selected = []
    cutoff_index = int(plan["cutoff_index"])
    cutoff_offset = int(plan["cutoff_offset"])
    for index in range(cutoff_index, len(config["units"])):
        unit = config["units"][index]
        start = cutoff_offset if index == cutoff_index else 0
        available = int(unit["count"]) - start
        take = min(remaining, available)
        if take:
            selected.append({"unit_index": index, "start": start, "count": take})
            remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"scenario lacks {remaining} records")
    return selected


def load_v3_state(path: Path, builder) -> dict:
    with path.open("rb") as handle:
        if handle.read(4) != b"RSGA":
            raise ValueError(path)
        version = builder.read_u16(handle)
        if version != 3:
            raise ValueError(f"expected v3 index, got {version}")
        base = builder.read_u8(handle)
        token_bytes = builder.read_u8(handle)
        day_seconds = builder.read_u32(handle)
        timezone_offset = builder.read_i32(handle)
        hprime_bits = builder.read_u16(handle)
        builder.read_u16(handle)
        n = builder.bytes_to_mpz(builder.read_varbytes(handle))
        g = builder.bytes_to_mpz(builder.read_varbytes(handle))
        modulus_bytes = (n.bit_length() + 7) // 8
        dates = {}
        for _ in range(builder.read_varint(handle)):
            day_idx = builder.read_varint(handle)
            data_xp_bytes = builder.read_varbytes(handle)
            data_ac_bytes = builder.read_varbytes(handle)
            edges = []
            for _ in range(builder.read_varint(handle)):
                edge_id = builder.read_varint(handle)
                edge_xp_bytes = builder.read_varbytes(handle)
                edge_ac_bytes = builder.read_varbytes(handle)
                root_witness = handle.read(modulus_bytes)
                identity_witness = handle.read(modulus_bytes)
                if len(root_witness) != modulus_bytes or len(identity_witness) != modulus_bytes:
                    raise EOFError("truncated edge witnesses")
                traj_rows = []
                for _ in range(builder.read_varint(handle)):
                    traj_id = builder.read_varint(handle)
                    token = handle.read(token_bytes)
                    occs = []
                    for _ in range(builder.read_varint(handle)):
                        occ = handle.read(5)
                        witness = handle.read(modulus_bytes)
                        if len(occ) != 5 or len(witness) != modulus_bytes:
                            raise EOFError("truncated occurrence")
                        occs.append((occ, witness))
                    traj_rows.append((traj_id, token, occs))
                edges.append(
                    {
                        "edge_id": edge_id,
                        "edge_xp_bytes": edge_xp_bytes,
                        "edge_ac_bytes": edge_ac_bytes,
                        "root_witness_bytes": root_witness,
                        "identity_witness_bytes": identity_witness,
                        "traj_rows": traj_rows,
                        "traj_count": len(traj_rows),
                    }
                )
            dates[day_idx] = {
                "data_xp_bytes": data_xp_bytes,
                "data_ac_bytes": data_ac_bytes,
                "edges": edges,
            }
        if handle.read(1):
            raise ValueError("trailing bytes")
    return {
        "version": version,
        "base": base,
        "token_bytes": token_bytes,
        "day_seconds": day_seconds,
        "timezone_offset": timezone_offset,
        "hprime_bits": hprime_bits,
        "n": n,
        "g": g,
        "dates": dates,
    }


def edge_traj_map(builder, edge_obj: dict) -> defaultdict[int, list[tuple[int, int]]]:
    result: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for traj_id, _token, occurrences in edge_obj["traj_rows"]:
        for occurrence, _witness in occurrences:
            result[int(traj_id)].append(builder.unpack_occurrence(occurrence))
    return result


def descriptor_prime(builder, day_idx: int, edge_obj: dict) -> int:
    edge_ac = builder.bytes_to_mpz(edge_obj["edge_ac_bytes"])
    seed = builder.edge_ac_seed(
        str(edge_obj["edge_id"]),
        edge_ac,
        day_idx=day_idx,
        traj_count=len(edge_obj["traj_rows"]),
    )
    return int(builder.hash_to_prime_local(seed, {}))


def descriptor_worker(args: tuple[int, int, bytes, int]) -> tuple[int, int]:
    day_idx, edge_id, edge_ac_bytes, traj_count = args
    builder = importlib.import_module(BUILDER_MODULE)
    builder.BINARY_VERSION = 3
    edge_ac = builder.bytes_to_mpz(edge_ac_bytes)
    seed = builder.edge_ac_seed(
        str(edge_id), edge_ac, day_idx=day_idx, traj_count=traj_count
    )
    return edge_id, int(builder.hash_to_prime_local(seed, {}))


def prepare_descriptors(
    builder, state: dict, sidecar: Path, workers: int
) -> dict[int, int]:
    if sidecar.exists():
        with sidecar.open("rb") as handle:
            return pickle.load(handle)
    if len(state["dates"]) != 1:
        raise ValueError("baseline index must contain one storage unit")
    day_idx, day_obj = next(iter(state["dates"].items()))
    tasks = [
        (day_idx, int(edge["edge_id"]), edge["edge_ac_bytes"], len(edge["traj_rows"]))
        for edge in day_obj["edges"]
    ]
    descriptors = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for edge_id, representative in executor.map(descriptor_worker, tasks, chunksize=64):
            descriptors[int(edge_id)] = int(representative)
    with sidecar.open("wb") as handle:
        pickle.dump(descriptors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return descriptors


def process_delta(builder, delta_csv: Path) -> dict:
    data = {}
    builder.process_trajectory_file(str(delta_csv), data)
    return data


def update_cutoff_unit(
    builder,
    baseline_state: dict,
    base_descriptors: dict[int, int],
    delta_csv: Path,
    output_bin: Path,
    workers: int,
) -> dict:
    delta_data = process_delta(builder, delta_csv)
    if len(baseline_state["dates"]) != 1:
        raise ValueError("baseline must contain one date bucket")
    base_day_idx, base_day = next(iter(baseline_state["dates"].items()))
    delta_bucket = delta_data.get(base_day_idx, {})
    unexpected_days = set(delta_data) - {base_day_idx}
    if unexpected_days:
        raise ValueError(f"delta crossed storage unit unexpectedly: {unexpected_days}")
    base_edges = {int(edge["edge_id"]): edge for edge in base_day["edges"]}
    combined_tasks = []
    for edge_id, delta_trajs in delta_bucket.items():
        combined = edge_traj_map(builder, base_edges[int(edge_id)])
        for traj_id, occurrences in delta_trajs.items():
            combined[int(traj_id)].extend((int(start), int(end)) for start, end in occurrences)
        combined_tasks.append((base_day_idx, int(edge_id), combined))

    p, q, n, g = builder.ensure_rsa_params()
    if n != baseline_state["n"] or g != baseline_state["g"]:
        raise ValueError("RSA parameters differ from baseline index")
    p_bytes = builder.mpz_to_bytes(p)
    q_bytes = builder.mpz_to_bytes(q)
    n_bytes = builder.mpz_to_bytes(n)
    g_bytes = builder.mpz_to_bytes(g)

    started = time.perf_counter()
    affected_results = []
    if combined_tasks:
        with mp.Pool(
            processes=max(1, workers),
            initializer=builder.worker_init,
            initargs=(
                p_bytes,
                q_bytes,
                n_bytes,
                g_bytes,
                bytes(builder.OPAQUE_TOKEN_KEY),
                builder.BINARY_VERSION,
                builder.OPAQUE_TOKEN_BYTES,
                builder.DAY_SECONDS,
            ),
        ) as pool:
            affected_results = list(
                pool.imap_unordered(
                    builder.compute_edge_fused_task,
                    combined_tasks,
                    chunksize=max(1, int(builder.FUSED_CHUNK_SIZE)),
                )
            )
    affected_by_id = {int(edge["edge_id"]): edge for edge in affected_results}
    new_descriptors = dict(base_descriptors)
    removed = []
    added = []
    for edge_id, edge_obj in affected_by_id.items():
        old = int(base_descriptors[edge_id])
        new = descriptor_prime(builder, base_day_idx, edge_obj)
        new_descriptors[edge_id] = new
        removed.append(old)
        added.append(new)
    base_xp = builder.bytes_to_mpz(base_day["data_xp_bytes"])
    if removed:
        removed_product = builder.product_tree(removed)
        if base_xp % removed_product != 0:
            raise ValueError("baseline descriptor product does not divide data_xp")
        new_xp = (base_xp // removed_product) * builder.product_tree(added)
    else:
        new_xp = base_xp
    new_ac = powmod(g, new_xp, n)

    witness_tasks = [(edge_id, new_descriptors[edge_id]) for edge_id in sorted(base_edges)]
    root_witnesses = {}
    with mp.Pool(
        processes=max(1, workers),
        initializer=builder.root_witness_worker_init,
        initargs=(
            p_bytes,
            q_bytes,
            n_bytes,
            g_bytes,
            bytes(builder.OPAQUE_TOKEN_KEY),
            builder.BINARY_VERSION,
            builder.OPAQUE_TOKEN_BYTES,
            builder.mpz_to_bytes(new_xp),
        ),
    ) as pool:
        for edge_id, witness in pool.imap_unordered(
            builder.compute_root_witness_task, witness_tasks, chunksize=32
        ):
            root_witnesses[int(edge_id)] = witness
    pure_seconds = time.perf_counter() - started

    output_edges = []
    for edge_id in sorted(base_edges):
        source = affected_by_id.get(edge_id, base_edges[edge_id])
        copied = dict(source)
        copied["root_witness_bytes"] = root_witnesses[edge_id]
        copied["traj_count"] = len(copied["traj_rows"])
        output_edges.append(copied)
    output_state = {
        base_day_idx: {
            "data_xp_bytes": builder.mpz_to_bytes(new_xp),
            "data_ac_bytes": builder.mpz_to_bytes(new_ac),
            "edges": output_edges,
        }
    }
    write_started = time.perf_counter()
    builder.write_binary(str(output_bin), n, g, output_state)
    write_seconds = time.perf_counter() - write_started
    validation = validate_binary(output_bin, builder)
    return {
        "kind": "update_existing_unit",
        "pure_update_seconds": pure_seconds,
        "write_seconds_excluded": write_seconds,
        "index_size_bytes": output_bin.stat().st_size,
        "affected_edge_count": len(affected_by_id),
        "validation": validation,
    }


def build_unit(
    builder,
    original_local_day,
    config: dict,
    unit: dict,
    input_csv: Path,
    output_bin: Path,
    workers: int,
) -> dict:
    configure_unit(
        builder,
        original_local_day,
        config,
        unit,
        input_csv,
        output_bin,
        workers,
    )
    timings = builder.main()
    validation = validate_binary(output_bin, builder)
    return {
        "kind": "build_new_unit",
        "pure_update_seconds": float(timings["accumulator_build_time_all_in"]),
        "write_seconds_excluded": float(timings["write_bin_time"]),
        "index_size_bytes": output_bin.stat().st_size,
        "affected_edge_count": int(validation["edge_count"]),
        "validation": validation,
    }


def metric_key(
    city: str,
    kind: str,
    unit_name: str,
    start: int,
    count: int,
    repeat: int,
) -> str:
    return f"{city}|{kind}|{unit_name}|{start}|{count}|repeat={repeat}"


def write_report(output_dir: Path, report: dict) -> None:
    save_json(output_dir / "incremental_update_results.json", report)
    with (output_dir / "incremental_update_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "dataset",
            "baseline_percentage",
            "baseline_trajectories",
            "total_trajectories",
            "cutoff_unit",
            "cutoff_offset",
            "inserted_trajectories",
            "touched_unit_count",
            "pure_update_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = dataset_configs(
        args.chengdu_source_dir.resolve(), args.xian_source_dir.resolve()
    )
    selected_cities = args.city or ["chengdu", "xian", "beijing"]
    populate_counts(
        configs,
        selected_cities,
        output_dir / "source_inventory.json",
        args.workers,
    )
    plans = {city: locate_baseline(configs[city]) for city in selected_cities}
    plan_export = {}
    for city in selected_cities:
        config = configs[city]
        plan = plans[city]
        plan_export[city] = {
            "dataset": config["display"],
            **plan,
            "scenarios": {
                str(size): [
                    {
                        "unit": config["units"][item["unit_index"]]["name"],
                        "start": item["start"],
                        "count": item["count"],
                    }
                    for item in scenario_selection(config, plan, size)
                ]
                for size in INSERT_SIZES
            },
        }
    save_json(output_dir / "incremental_update_plan.json", plan_export)
    if args.plan_only:
        print(json.dumps(plan_export, ensure_ascii=False, indent=2))
        return

    builder = importlib.import_module(BUILDER_MODULE)
    original_local_day = builder.local_day_and_second
    metrics_path = output_dir / "artifact_metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    row_cache: dict[str, tuple[list[str], list[dict]]] = {}
    scenario_rows = []

    for city in selected_cities:
        config = configs[city]
        plan = plans[city]
        prepare_needed_mapped_csvs(config, plan, output_dir / "prepared")
        cutoff_index = int(plan["cutoff_index"])
        cutoff_unit = config["units"][cutoff_index]
        city_dir = output_dir / city
        inputs_dir = city_dir / "inputs"
        indexes_dir = city_dir / "indexes"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        indexes_dir.mkdir(parents=True, exist_ok=True)

        baseline_csv = unit_slice_csv(
            cutoff_unit,
            0,
            int(plan["cutoff_offset"]),
            inputs_dir / f"baseline-60-{cutoff_unit['name']}.csv",
            row_cache,
        )
        baseline_bin = indexes_dir / f"baseline-60-{cutoff_unit['name']}.bin"
        configure_unit(
            builder,
            original_local_day,
            config,
            cutoff_unit,
            baseline_csv,
            baseline_bin,
            args.workers,
        )
        if not baseline_bin.exists():
            print(f"[{config['display']}] build 60% cutoff baseline {cutoff_unit['name']}", flush=True)
            builder.main()
            validate_binary(baseline_bin, builder)
        baseline_state = load_v3_state(baseline_bin, builder)
        descriptors = prepare_descriptors(
            builder,
            baseline_state,
            city_dir / "baseline_descriptor_primes_v3.pkl",
            args.workers,
        )

        for insert_size in INSERT_SIZES:
            selections = scenario_selection(config, plan, insert_size)
            repeated_metrics = []
            for repeat in range(1, args.repeats + 1):
                selected_metrics = []
                for item in selections:
                    unit = config["units"][item["unit_index"]]
                    start = int(item["start"])
                    count = int(item["count"])
                    if item["unit_index"] == cutoff_index:
                        kind = "update"
                        key = metric_key(
                            city, kind, unit["name"], start, count, repeat
                        )
                        delta_csv = unit_slice_csv(
                            unit,
                            start,
                            count,
                            inputs_dir / f"delta-{unit['name']}-{start}-{count}.csv",
                            row_cache,
                        )
                        output_bin = indexes_dir / f"updated-{unit['name']}-{start + count}.bin"
                        if key not in metrics:
                            configure_unit(
                                builder,
                                original_local_day,
                                config,
                                unit,
                                delta_csv,
                                output_bin,
                                args.workers,
                            )
                            print(
                                f"[{config['display']}] update cutoff unit +{count} "
                                f"({insert_size} scenario), repeat {repeat}/{args.repeats}",
                                flush=True,
                            )
                            metrics[key] = update_cutoff_unit(
                                builder,
                                baseline_state,
                                descriptors,
                                delta_csv,
                                output_bin,
                                args.workers,
                            )
                            save_json(metrics_path, metrics)
                        selected_metrics.append(metrics[key])
                    else:
                        kind = "new"
                        key = metric_key(
                            city, kind, unit["name"], start, count, repeat
                        )
                        input_csv = unit_slice_csv(
                            unit,
                            start,
                            count,
                            inputs_dir / f"new-{unit['name']}-{count}.csv",
                            row_cache,
                        )
                        output_bin = indexes_dir / f"new-{unit['name']}-{count}.bin"
                        if key not in metrics:
                            print(
                                f"[{config['display']}] build new unit {unit['name']} "
                                f"rows={count}, repeat {repeat}/{args.repeats}",
                                flush=True,
                            )
                            metrics[key] = build_unit(
                                builder,
                                original_local_day,
                                config,
                                unit,
                                input_csv,
                                output_bin,
                                args.workers,
                            )
                            save_json(metrics_path, metrics)
                        selected_metrics.append(metrics[key])
                repeated_metrics.append(selected_metrics)
            update_seconds_runs = [
                sum(float(metric["pure_update_seconds"]) for metric in run)
                for run in repeated_metrics
            ]
            scenario_rows.append(
                {
                    "dataset": config["display"],
                    "city": city,
                    "baseline_percentage": 60,
                    "baseline_trajectories": int(plan["baseline_count"]),
                    "total_trajectories": int(plan["total"]),
                    "cutoff_unit": cutoff_unit["name"],
                    "cutoff_offset": int(plan["cutoff_offset"]),
                    "inserted_trajectories": insert_size,
                    "touched_unit_count": len(selections),
                    "pure_update_seconds": statistics.fmean(update_seconds_runs),
                    "pure_update_seconds_runs": update_seconds_runs,
                    "artifacts": [
                        {
                            "unit": config["units"][item["unit_index"]]["name"],
                            "start": int(item["start"]),
                            "count": int(item["count"]),
                        }
                        for item in selections
                    ],
                }
            )

    report = {
        "definition": (
            "Pure update time excludes source mapping, CSV/index loading, serialization and "
            "disk writes. It includes affected-edge RSA recomputation, descriptor replacement, "
            "new global accumulator, all root-witness refreshes, and from-scratch construction "
            "of any newly crossed storage unit."
        ),
        "baseline": "chronological 60% of each complete dataset",
        "scenario_semantics": "each insertion size independently starts from the same 60% baseline",
        "storage_units": "Chengdu/Xian daily; Beijing consecutive bimonthly",
        "workers": int(args.workers),
        "repeats": int(args.repeats),
        "primary_statistic": "arithmetic_mean",
        "results": scenario_rows,
    }
    write_report(output_dir, report)
    print(json.dumps(scenario_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
