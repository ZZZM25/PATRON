"""Cell50 incremental-update benchmark for the per-edge SMT method.

Every measured point independently loads the same complete 60% disk index and
inserts the next N Cell50-transformed trajectories.  The primary update timer
contains only:

1. stable start-time ordered insertion of the new edge records;
2. incremental Patricia-path insertion into every affected edge SMT plus a
   complete recomputation of that edge's ordered time commitment;
3. authentication updates on affected spatial-tree paths to the root.

Base construction/load, trajectory parsing and Cell50 transformation, affected
edge disk reads/SMT deserialization, path mapping, validation and persistence
are measured separately or excluded, matching the pre-Cell50 experiment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import statistics
from array import array
from bisect import bisect_right
from pathlib import Path
from time import perf_counter

from baseline.authenticated_index import authenticate_spatial_tree
from baseline.cell_dataset_builder import (
    _load_beijing_points,
    _set_large_csv_field_limit,
    parse_cell_partitioned_beijing_points,
)
from baseline.config import load_config
from baseline.crypto import (
    ENTRY_LIST_TAIL,
    entry_list_prepend,
    hash_entry,
    hash_smt_edge,
)
from baseline.disk_smt import ShardedEntrySpool, build_disk_smt_baseline, spool_cell_trajectories
from baseline.edge_smt import EdgeSMT, key_for_id
from baseline.persistence import MAGIC_INDEX, VERSION, _HEADER, load_index
from baseline.trajectory_parser import (
    collect_files,
    load_relaxed_json,
    parse_cell_partitioned_trajectory,
)
from run_build_scaling_experiment import (
    DEFAULT_JOBS,
    _build_partition,
    _count_source_trajectories,
    _metrics_total,
)


ROOT = Path(__file__).resolve().parent
BASE_PERCENT = 60
INSERT_COUNTS = (1000, 2000, 5000, 8000, 10000)
BASE_FORMAT_VERSION = "cell50-update-base-v1"
DELTA_FORMAT_VERSION = "cell50-update-delta-v1"
UPDATE_ALGORITHM = "cell50-disk-incremental-patricia-path-v1"


def _build_spatial_metadata(tree, max_eid: int):
    """Build parent/depth/edge-owner maps for incremental spatial updates."""
    node_count = len(tree.node_type)
    parent = [-1] * node_count
    depth = [0] * node_count
    owner = [-1] * (max_eid + 1)
    stack = [(int(tree.root_index), 0)]
    while stack:
        node, node_depth = stack.pop()
        depth[node] = node_depth
        for eid in tree.node_edges[node]:
            owner[int(eid)] = node
        left = int(tree.left_child[node])
        right = int(tree.right_child[node])
        if left >= 0:
            parent[left] = node
            stack.append((left, node_depth + 1))
        if right >= 0:
            parent[right] = node
            stack.append((right, node_depth + 1))
    return parent, depth, owner


def _update_spatial_paths(index, affected_eids, metadata):
    """Update owner nodes and their ancestors while reusing sibling hashes."""
    from baseline.crypto import (
        EDGE_GROUP_INIT,
        edge_group_append,
        hash_internal,
        hash_leaf,
    )
    from baseline.spatial_tree import NODE_LEAF

    tree = index.tree
    parent, depth, owner = metadata
    local_dirty = {owner[eid] for eid in affected_eids}
    if -1 in local_dirty:
        raise RuntimeError("affected edge not found in spatial tree")
    dirty = set(local_dirty)
    for node in list(local_dirty):
        ancestor = parent[node]
        while ancestor >= 0:
            dirty.add(ancestor)
            ancestor = parent[ancestor]

    changed_hash = {}
    for node in sorted(dirty, key=lambda value: depth[value], reverse=True):
        if node in local_dirty:
            group = EDGE_GROUP_INIT
            for eid in tree.node_edges[node]:
                group = edge_group_append(group, index.edge_hashes[eid])
            tree.edge_group_root[node] = group
        else:
            group = tree.edge_group_root[node]
        bounds = (
            tree.min_lon[node],
            tree.min_lat[node],
            tree.max_lon[node],
            tree.max_lat[node],
        )
        edges = tree.node_edges[node]
        if tree.node_type[node] == NODE_LEAF:
            node_hash = hash_leaf(*bounds, len(edges), group)
        else:
            left = int(tree.left_child[node])
            right = int(tree.right_child[node])
            left_hash = changed_hash.get(left, tree.node_hash[left])
            right_hash = changed_hash.get(right, tree.node_hash[right])
            node_hash = hash_internal(
                *bounds, left_hash, right_hash, len(edges), group
            )
        tree.node_hash[node] = node_hash
        changed_hash[node] = node_hash

    root = int(tree.root_index)
    index.root_hash = changed_hash.get(root, tree.node_hash[root])
    return index.root_hash


class _HashingWriter:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.count = 0

    def write(self, data: bytes) -> int:
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("short index write")
        self.hasher.update(data)
        self.count += written
        return written

    def flush(self) -> None:
        self.stream.flush()


def _save_index_streaming(index, path: Path) -> int:
    """Write BLIDX002 without retaining a second complete pickle in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb+") as stream:
        stream.write(b"\0" * _HEADER.size)
        writer = _HashingWriter(stream)
        pickle.dump(index, writer, protocol=5)
        writer.flush()
        stream.seek(0)
        stream.write(
            _HEADER.pack(MAGIC_INDEX, VERSION, writer.count, writer.hasher.digest())
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path.stat().st_size


def _job_map() -> dict[str, Path]:
    return {city: path for city, path in DEFAULT_JOBS}


def _base_paths(output_dir: Path, city: str) -> dict[str, Path]:
    directory = output_dir / "base_indexes" / city
    return {
        "directory": directory,
        "manifest": directory / f"{city}_060pct_manifest.dat",
        "edge_data": directory / f"{city}_060pct_edge_data.bin",
        "smt_data": directory / f"{city}_060pct_edge_smt_nodes.bin",
        "catalog": directory / f"{city}_060pct_trajectory_catalog.dat",
        "metrics": directory / f"{city}_060pct_metrics.json",
        "delta": directory / f"{city}_next_10000_cell50_delta.pkl",
    }


def _total_trajectories(city: str, cfg: dict, source_road, partition) -> tuple[int, str]:
    total = _metrics_total(cfg)
    if total is not None:
        return total, "existing full-build metrics"
    return (
        _count_source_trajectories(city, cfg, source_road, partition),
        "source scan",
    )


def build_base(city: str, config_path: Path, output_dir: Path) -> dict:
    """Persist the complete 60% Cell50 base index used by every update point."""
    cfg = load_config(str(config_path))
    source_road, partition, grid = _build_partition(cfg)
    total, total_source = _total_trajectories(city, cfg, source_road, partition)
    base_count = math.floor(total * BASE_PERCENT / 100)
    paths = _base_paths(output_dir, city)
    paths["directory"].mkdir(parents=True, exist_ok=True)

    storage = cfg.get("disk_storage", {})
    shard_count = int(storage.get("construction_shards", 512))
    block_cache_bytes = int(
        float(storage.get("edge_block_cache_mib", 512)) * 1024 * 1024
    )
    smt_cache_bytes = int(
        float(storage.get("edge_smt_cache_mib", 1024)) * 1024 * 1024
    )

    wall_started = perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f".{city}_update_base_spool_", dir=paths["directory"]
    ) as spool_directory:
        spool = ShardedEntrySpool(spool_directory, shard_count=shard_count)
        spool_stats = spool_cell_trajectories(
            city=city,
            source_road=source_road,
            partition=partition,
            trajectory_globs=list(cfg["trajectory_globs"]),
            config_dir=Path(cfg["_config_dir"]),
            spool=spool,
            catalog_file=paths["catalog"],
            expected_file_count=cfg.get("expected_trajectory_file_count"),
            max_trajectories=base_count,
        )
        if int(spool_stats["trajectory_count"]) != base_count:
            raise RuntimeError(
                f"base selected {spool_stats['trajectory_count']}, expected {base_count}"
            )
        index, build_stats = build_disk_smt_baseline(
            road=partition.road,
            spool=spool,
            edge_data_file=paths["edge_data"],
            smt_data_file=paths["smt_data"],
            theta=int(cfg.get("theta", 64)),
            trajectory_files=spool_stats["files"],
            block_cache_bytes=block_cache_bytes,
            smt_cache_bytes=smt_cache_bytes,
        )

    manifest_size = _save_index_streaming(index, paths["manifest"])
    result = {
        "success": True,
        "format_version": BASE_FORMAT_VERSION,
        "city": city,
        "base_percent": BASE_PERCENT,
        "total_trajectory_count": total,
        "total_trajectory_count_source": total_source,
        "base_trajectory_count": base_count,
        "base_partitioned_entry_count": build_stats["record_count"],
        "base_smt_leaf_count": sum(index.store._unique_counts),
        "base_populated_edge_count": build_stats["populated_edge_count"],
        "source_edge_count": len(source_road.valid_edge_ids()),
        "partitioned_edge_count": len(partition.road.valid_edge_ids()),
        "theta": int(index.theta),
        "grid": grid.as_dict(),
        "manifest_size_bytes": manifest_size,
        "edge_data_size_bytes": build_stats["edge_data_size_bytes"],
        "smt_data_size_bytes": build_stats["smt_data_size_bytes"],
        "base_index_size_bytes": (
            manifest_size
            + int(build_stats["edge_data_size_bytes"])
            + int(build_stats["smt_data_size_bytes"])
        ),
        "base_build_wall_seconds_excluded": perf_counter() - wall_started,
        "base_root_hex": index.root_hash.hex(),
        "manifest_file": str(paths["manifest"]),
        "edge_data_file": str(paths["edge_data"]),
        "smt_data_file": str(paths["smt_data"]),
        "catalog_file": str(paths["catalog"]),
    }
    paths["metrics"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _base_is_valid(output_dir: Path, city: str) -> bool:
    paths = _base_paths(output_dir, city)
    required = ("manifest", "edge_data", "smt_data", "metrics")
    if not all(paths[name].is_file() for name in required):
        return False
    try:
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        metrics.get("success")
        and metrics.get("format_version") == BASE_FORMAT_VERSION
        and metrics.get("city") == city
        and int(metrics.get("base_percent", -1)) == BASE_PERCENT
    )


def _iter_cell_trajectories(city: str, cfg: dict, source_road, partition):
    files = collect_files(cfg["trajectory_globs"], Path(cfg["_config_dir"]))
    if city == "beijing":
        if len(files) != 1:
            raise ValueError(f"expected one Beijing CSV, got {len(files)}")
        _set_large_csv_field_limit()
        with files[0].open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for row_no, row in enumerate(reader, 2):
                points = _load_beijing_points(row["points_data"], row_no)
                parsed, _, _ = parse_cell_partitioned_beijing_points(
                    points, row_no, source_road, partition
                )
                if parsed is not None:
                    yield parsed
        return
    for file_index, path in enumerate(files, 1):
        print(f"[{file_index}/{len(files)}] prepare delta {path}", flush=True)
        data = load_relaxed_json(path)
        if not isinstance(data, list):
            raise ValueError(f"top-level JSON must be list: {path}")
        for trajectory in data:
            yield parse_cell_partitioned_trajectory(
                trajectory, source_road, partition
            )


def prepare_delta_cache(
    city: str,
    config_path: Path,
    output_dir: Path,
    max_insert_count: int = max(INSERT_COUNTS),
) -> dict:
    """Prepare the next trajectories once; all worker points use its prefix."""
    cfg = load_config(str(config_path))
    source_road, partition, _ = _build_partition(cfg)
    total, _ = _total_trajectories(city, cfg, source_road, partition)
    base_count = math.floor(total * BASE_PERCENT / 100)
    available = total - base_count
    if max_insert_count > available:
        raise ValueError(
            f"{city}: requested {max_insert_count} updates, only {available} remain"
        )
    selected: list[tuple[bytes, tuple[tuple[int, int, int], ...]]] = []
    seen: set[bytes] = set()
    for position, parsed in enumerate(
        _iter_cell_trajectories(city, cfg, source_road, partition)
    ):
        if parsed.trajectory_id in seen:
            raise ValueError(
                "duplicate trajectory_id: " + parsed.trajectory_id.hex()
            )
        seen.add(parsed.trajectory_id)
        if position < base_count:
            continue
        selected.append(
            (
                parsed.trajectory_id,
                tuple(
                    (int(eid), int(segment.start_time), int(segment.end_time))
                    for eid, segment in zip(parsed.eid_path, parsed.segments)
                ),
            )
        )
        if len(selected) >= max_insert_count:
            break
    if len(selected) != max_insert_count:
        raise RuntimeError(
            f"prepared {len(selected)} trajectories, expected {max_insert_count}"
        )

    paths = _base_paths(output_dir, city)
    payload = {
        "format_version": DELTA_FORMAT_VERSION,
        "city": city,
        "base_percent": BASE_PERCENT,
        "base_trajectory_count": base_count,
        "max_insert_count": max_insert_count,
        "records": selected,
    }
    paths["directory"].mkdir(parents=True, exist_ok=True)
    temporary = paths["delta"].with_suffix(paths["delta"].suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(paths["delta"])
    result = {
        key: payload[key]
        for key in (
            "format_version", "city", "base_percent",
            "base_trajectory_count", "max_insert_count",
        )
    }
    result["delta_cache_file"] = str(paths["delta"])
    result["delta_cache_size_bytes"] = paths["delta"].stat().st_size
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _delta_is_valid(output_dir: Path, city: str) -> bool:
    path = _base_paths(output_dir, city)["delta"]
    if not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            value = pickle.load(stream)
        return bool(
            value.get("format_version") == DELTA_FORMAT_VERSION
            and value.get("city") == city
            and int(value.get("base_percent", -1)) == BASE_PERCENT
            and int(value.get("max_insert_count", -1)) >= max(INSERT_COUNTS)
            and len(value.get("records", ())) >= max(INSERT_COUNTS)
        )
    except Exception:
        return False


class _DigestOverlay:
    """Mutable sparse overlay over an immutable DigestArray."""

    def __init__(self, base) -> None:
        self.base = base
        self.changed: dict[int, bytes] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> bytes:
        return self.changed.get(int(index), self.base[index])

    def __setitem__(self, index: int, value: bytes) -> None:
        value = bytes(value)
        if len(value) != 32:
            raise ValueError("digest must be 32 bytes")
        self.changed[int(index)] = value


class _MutableEdgeEntries:
    """One affected edge materialized in the original packed mutable layout."""

    __slots__ = ("keys", "starts", "ends")

    def __init__(self, block) -> None:
        self.keys = bytearray()
        self.starts = array("I")
        self.ends = array("I")
        for key, start, end in block.iter_entries():
            self.keys.extend(key)
            self.starts.append(int(start))
            self.ends.append(int(end))

    def __len__(self) -> int:
        return len(self.starts)

    def insert_by_start(self, key: bytes, start: int, end: int) -> None:
        position = bisect_right(self.starts, int(start))
        self.starts.insert(position, int(start))
        self.ends.insert(position, int(end))
        byte_position = position * 32
        self.keys[byte_position:byte_position] = key

    def get_entry(self, position: int) -> tuple[bytes, int, int]:
        offset = position * 32
        return (
            bytes(self.keys[offset:offset + 32]),
            int(self.starts[position]),
            int(self.ends[position]),
        )


def _load_edge_smt_uncached(index, eid: int) -> EdgeSMT:
    length = int(index.smt_lengths[eid]) if index.smt_lengths is not None else 0
    if index.smt_data_file is not None and length > 0:
        with open(index.smt_data_file, "rb") as stream:
            stream.seek(int(index.smt_offsets[eid]))
            smt = pickle.load(stream)
        if not isinstance(smt, EdgeSMT):
            raise ValueError(f"invalid persisted edge SMT object: {eid}")
    else:
        smt = EdgeSMT.build([])
    if smt.root_hash != index.edge_smt_roots[eid]:
        raise ValueError(f"base edge SMT root mismatch: {eid}")
    return smt


def update_worker(
    city: str,
    output_dir: Path,
    insert_count: int,
    result_path: Path,
    repeat: int,
) -> dict:
    if insert_count not in INSERT_COUNTS and not os.environ.get(
        "CELL50_UPDATE_ALLOW_TEST_COUNT"
    ):
        raise ValueError(f"insert_count must be one of {INSERT_COUNTS}")
    paths = _base_paths(output_dir, city)

    load_started = perf_counter()
    index = load_index(paths["manifest"])
    base_load_seconds = perf_counter() - load_started
    with paths["delta"].open("rb") as stream:
        delta_payload = pickle.load(stream)
    if delta_payload.get("format_version") != DELTA_FORMAT_VERSION:
        raise ValueError("incompatible Cell50 update delta cache")
    delta = list(delta_payload["records"][:insert_count])
    if len(delta) != insert_count:
        raise RuntimeError(f"loaded {len(delta)} delta trajectories, expected {insert_count}")

    preparation_started = perf_counter()
    affected: dict[int, list[tuple[bytes, int, int]]] = {}
    inserted_entry_count = 0
    for trajectory_id, intervals in delta:
        for eid, start, end in intervals:
            affected.setdefault(int(eid), []).append(
                (bytes(trajectory_id), int(start), int(end))
            )
            inserted_entry_count += 1
    affected_eids = sorted(affected)
    spatial_metadata = _build_spatial_metadata(index.tree, int(index.road.max_eid))
    preparation_seconds = perf_counter() - preparation_started

    index.edge_entry_roots = _DigestOverlay(index.edge_entry_roots)
    index.edge_smt_roots = _DigestOverlay(index.edge_smt_roots)
    index.edge_hashes = _DigestOverlay(index.edge_hashes)

    key_started = perf_counter()
    key_by_id = {trajectory_id: key_for_id(trajectory_id) for trajectory_id, _ in delta}
    ordered_insertion_seconds = perf_counter() - key_started
    edge_update_seconds = 0.0
    edge_data_load_seconds = 0.0
    smt_load_seconds = 0.0
    mutable_materialization_seconds = 0.0

    for position, eid in enumerate(affected_eids, 1):
        load_started = perf_counter()
        block = index.store._load(eid)
        edge_data_load_seconds += perf_counter() - load_started

        materialize_started = perf_counter()
        entries = _MutableEdgeEntries(block)
        mutable_materialization_seconds += perf_counter() - materialize_started
        index.store._cache.clear()
        index.store._cached_bytes = 0

        smt_started = perf_counter()
        smt = _load_edge_smt_uncached(index, eid)
        smt_load_seconds += perf_counter() - smt_started

        insert_started = perf_counter()
        for trajectory_id, start, end in affected[eid]:
            entries.insert_by_start(key_by_id[trajectory_id], start, end)
        ordered_insertion_seconds += perf_counter() - insert_started

        update_started = perf_counter()
        for trajectory_id in sorted({row[0] for row in affected[eid]}):
            smt.insert(trajectory_id)
        state = ENTRY_LIST_TAIL
        for entry_index in range(len(entries) - 1, -1, -1):
            key, start, end = entries.get_entry(entry_index)
            state = entry_list_prepend(hash_entry(key, start, end), state)
        count = len(entries)
        index.store.entry_counts[eid] = count
        index.store._unique_counts[eid] = len(smt.values)
        index.edge_entry_roots[eid] = state
        index.edge_smt_roots[eid] = smt.root_hash
        index.edge_hashes[eid] = hash_smt_edge(
            eid, count, state, smt.root_hash
        )
        edge_update_seconds += perf_counter() - update_started

        del entries, smt, block
        if position % max(1, len(affected_eids) // 20) == 0:
            print(
                f"updated affected edges {position}/{len(affected_eids)}",
                flush=True,
            )

    spatial_started = perf_counter()
    updated_root = _update_spatial_paths(index, affected_eids, spatial_metadata)
    spatial_seconds = perf_counter() - spatial_started
    update_seconds = (
        ordered_insertion_seconds + edge_update_seconds + spatial_seconds
    )

    verification_started = perf_counter()
    reference_root = authenticate_spatial_tree(index.tree, index.edge_hashes)
    verification_seconds = perf_counter() - verification_started
    if reference_root != updated_root:
        raise RuntimeError(
            "incremental spatial root mismatch: "
            f"got={updated_root.hex()}, reference={reference_root.hex()}"
        )

    base_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    base_count = int(base_metrics["base_trajectory_count"])
    result = {
        "success": True,
        "method": "cell50-per-edge-smt",
        "update_algorithm": UPDATE_ALGORITHM,
        "city": city,
        "repeat": repeat,
        "base_percent": BASE_PERCENT,
        "total_trajectory_count": base_metrics["total_trajectory_count"],
        "base_trajectory_count": base_count,
        "insert_trajectory_count": insert_count,
        "final_trajectory_count": base_count + insert_count,
        "inserted_entry_count": inserted_entry_count,
        "affected_edge_count": len(affected_eids),
        "update_time_s": update_seconds,
        "ordered_insertion_time_s": ordered_insertion_seconds,
        "affected_edge_smt_and_commitment_time_s": edge_update_seconds,
        "spatial_auth_path_update_time_s": spatial_seconds,
        "base_index_load_time_excluded_s": base_load_seconds,
        "delta_and_path_preparation_time_excluded_s": preparation_seconds,
        "edge_data_load_time_excluded_s": edge_data_load_seconds,
        "edge_smt_load_time_excluded_s": smt_load_seconds,
        "mutable_edge_materialization_time_excluded_s": mutable_materialization_seconds,
        "full_root_validation_time_excluded_s": verification_seconds,
        "updated_root_hex": updated_root.hex(),
        "reference_root_match": True,
        "selection_policy": (
            "next N valid Cell50-transformed trajectories immediately after the "
            "deterministic 60% prefix; every point independently loads the same base"
        ),
        "timing_scope": (
            "stable ordered edge insertion + incremental affected-edge Patricia "
            "paths + complete affected-edge time-commitment recomputation + "
            "affected spatial authentication paths"
        ),
        "excluded_from_time": [
            "60% base construction and loading",
            "source loading, parsing and Cell50 trajectory transformation",
            "delta selection and affected spatial-path mapping",
            "affected edge-data reads and mutable materialization",
            "affected edge-SMT reads and deserialization",
            "full-root validation",
            "updated index persistence and result writing",
        ],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del index
    gc.collect()
    return result


def _collect_rows(output_dir: Path) -> list[dict]:
    rows = []
    cells = output_dir / "cells"
    if not cells.is_dir():
        return rows
    for path in sorted(cells.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            row.get("success")
            and row.get("update_algorithm") == UPDATE_ALGORITHM
            and row.get("city") in _job_map()
            and int(row.get("insert_trajectory_count", -1)) in INSERT_COUNTS
            and int(row.get("repeat", -1)) >= 1
        ):
            rows.append(row)
    city_order = {city: position for position, (city, _) in enumerate(DEFAULT_JOBS)}
    rows.sort(key=lambda row: (
        city_order[row["city"]], int(row["insert_trajectory_count"])
    ))
    return rows


def aggregate(output_dir: Path, repeats: int) -> None:
    raw_rows = _collect_rows(output_dir)
    timing_fields = (
        "update_time_s",
        "ordered_insertion_time_s",
        "affected_edge_smt_and_commitment_time_s",
        "spatial_auth_path_update_time_s",
    )
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in raw_rows:
        key = (row["city"], int(row["insert_trajectory_count"]))
        groups.setdefault(key, []).append(row)
    rows = []
    for _key, runs in sorted(groups.items()):
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
        "city", "base_percent", "total_trajectory_count",
        "base_trajectory_count", "insert_trajectory_count",
        "final_trajectory_count", "inserted_entry_count",
        "affected_edge_count", "update_time_s", "ordered_insertion_time_s",
        "affected_edge_smt_and_commitment_time_s",
        "spatial_auth_path_update_time_s", "updated_root_hex",
        "reference_root_match",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "update_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lookup = {
        (row["city"], int(row["insert_trajectory_count"])): row
        for row in rows
    }
    with (output_dir / "update_time_matrix_s.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["city", *INSERT_COUNTS])
        for city, _ in DEFAULT_JOBS:
            writer.writerow([
                city,
                *[
                    lookup[(city, count)]["update_time_s"]
                    if (city, count) in lookup else ""
                    for count in INSERT_COUNTS
                ],
            ])
    (output_dir / "update_results.json").write_text(
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


def _parse_cities(raw: str) -> set[str]:
    allowed = set(_job_map())
    values = {value.strip().lower() for value in raw.split(",") if value.strip()}
    unknown = values - allowed
    if unknown:
        raise ValueError("unknown cities: " + ", ".join(sorted(unknown)))
    return values


def run_parent(output_dir: Path, cities: set[str], force: bool, repeats: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for city, config_path in DEFAULT_JOBS:
        if city not in cities:
            continue
        paths = _base_paths(output_dir, city)
        if force or not _base_is_valid(output_dir, city):
            print(f"BUILD CELL50 BASE {city} 60%", flush=True)
            command = [
                sys.executable, str(Path(__file__).resolve()), "--build-base",
                "--city", city, "--config", str(config_path),
                "--output-dir", str(output_dir),
            ]
            if subprocess.run(command, cwd=ROOT).returncode:
                raise SystemExit(1)
        else:
            print(f"RESUME CELL50 BASE {city} 60%", flush=True)

        if force or not _delta_is_valid(output_dir, city):
            print(f"PREPARE CELL50 DELTA {city}", flush=True)
            command = [
                sys.executable, str(Path(__file__).resolve()), "--prepare-delta",
                "--city", city, "--config", str(config_path),
                "--output-dir", str(output_dir),
            ]
            if subprocess.run(command, cwd=ROOT).returncode:
                raise SystemExit(1)
        else:
            print(f"RESUME CELL50 DELTA {city}", flush=True)

        for count in INSERT_COUNTS:
            for repeat in range(1, repeats + 1):
                cell = output_dir / "cells" / f"{city}_{count:05d}_r{repeat}.json"
                if cell.is_file() and not force:
                    result = json.loads(cell.read_text(encoding="utf-8"))
                    if (
                        result.get("success")
                        and result.get("update_algorithm") == UPDATE_ALGORITHM
                        and int(result.get("insert_trajectory_count", -1)) == count
                        and int(result.get("repeat", -1)) == repeat
                    ):
                        print(f"RESUME {city} +{count} r{repeat}", flush=True)
                        continue
                print(f"START {city} +{count} r{repeat}/{repeats}", flush=True)
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--city", city, "--output-dir", str(output_dir),
                    "--insert", str(count), "--result", str(cell),
                    "--repeat", str(repeat),
                ]
                if subprocess.run(command, cwd=ROOT).returncode:
                    raise SystemExit(1)
                aggregate(output_dir, repeats)
                print(f"DONE {city} +{count} r{repeat}", flush=True)
    aggregate(output_dir, repeats)
    print(f"ALL DONE: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "update_scaling_experiment",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cities", default="chengdu,xian,beijing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build-base", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-delta", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--city", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--insert", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.build_base:
        if not args.city or not args.config:
            parser.error("--build-base requires --city and --config")
        build_base(args.city.lower(), args.config.resolve(), output_dir)
        return
    if args.prepare_delta:
        if not args.city or not args.config:
            parser.error("--prepare-delta requires --city and --config")
        prepare_delta_cache(args.city.lower(), args.config.resolve(), output_dir)
        return
    if args.worker:
        if not args.city or args.insert is None or args.result is None or args.repeat is None:
            parser.error("--worker requires --city, --insert and --result")
        update_worker(
            args.city.lower(), output_dir, int(args.insert), args.result.resolve(),
            int(args.repeat),
        )
        return
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    run_parent(output_dir, _parse_cities(args.cities), args.force, args.repeats)


if __name__ == "__main__":
    main()
