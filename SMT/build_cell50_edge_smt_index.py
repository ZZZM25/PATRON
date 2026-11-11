"""Build the preserved Edge-SMT method on a 50 m cell-partitioned road.

Only the input road/trajectory segmentation changes.  The binary spatial tree,
time list, independent per-edge SMTs, edge commitments, VO and verifier are the
existing implementations.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

from baseline.cell_partition import (
    grid_from_road,
    split_road_network,
    write_partition_artifacts,
)
from baseline.config import ensure_parent, load_config, resolve_path
from baseline.disk_smt import (
    ShardedEntrySpool,
    build_disk_smt_baseline,
    spool_cell_trajectories,
)
from baseline.persistence import save_index
from baseline.road_network import RoadNetwork


def _required_path(cfg: dict, key: str) -> Path:
    if key not in cfg:
        raise ValueError(f"cell-partitioned config missing {key}")
    return resolve_path(cfg, cfg[key])


def _write_json(path: Path, value: dict) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--partition-only",
        action="store_true",
        help="write the split road/cell mapping without parsing trajectories",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    city = str(cfg["city"]).lower()
    cell_cfg = cfg.get("cell_partition")
    if not isinstance(cell_cfg, dict):
        raise ValueError("cell_partition object is required")

    source_road = RoadNetwork()
    source_road.load(
        str(_required_path(cfg, "source_node_file")),
        str(_required_path(cfg, "source_edge_file")),
    )
    cell_size = float(cell_cfg.get("size_m", 50.0))
    grid = grid_from_road(
        source_road,
        cell_width_m=float(cell_cfg.get("width_m", cell_size)),
        cell_height_m=float(cell_cfg.get("height_m", cell_size)),
        reference_lon=(
            float(cell_cfg["reference_lon"])
            if "reference_lon" in cell_cfg
            else None
        ),
        reference_lat=(
            float(cell_cfg["reference_lat"])
            if "reference_lat" in cell_cfg
            else None
        ),
        origin_x_m=float(cell_cfg.get("origin_x_m", 0.0)),
        origin_y_m=float(cell_cfg.get("origin_y_m", 0.0)),
    )

    partition_started = perf_counter()
    partition = split_road_network(source_road, grid)
    partition_seconds = perf_counter() - partition_started
    node_file = _required_path(cfg, "node_file")
    edge_file = _required_path(cfg, "edge_file")
    edge_cell_file = _required_path(cfg, "edge_cell_file")
    partition_metadata_file = _required_path(cfg, "partition_metadata_file")
    write_partition_artifacts(
        partition,
        node_file,
        edge_file,
        edge_cell_file,
        partition_metadata_file,
    )

    source_nodes = len(source_road.valid_node_ids())
    source_edges = len(source_road.valid_edge_ids())
    partitioned_nodes = len(partition.road.valid_node_ids())
    partitioned_edges = len(partition.road.valid_edge_ids())
    print(
        f"{city}: source road {source_nodes} nodes/{source_edges} edges -> "
        f"partitioned road {partitioned_nodes} nodes/{partitioned_edges} edges",
        flush=True,
    )
    print(
        "one-edge-one-cell validation: PASS; "
        f"partition time: {partition_seconds:.9f} s",
        flush=True,
    )
    if args.partition_only:
        print("partition-only build complete", flush=True)
        return

    index_file = _required_path(cfg, "index_file")
    catalog_file = _required_path(cfg, "catalog_file")
    if cfg.get("edge_data_file"):
        edge_data_file = resolve_path(cfg, cfg["edge_data_file"])
    else:
        edge_data_file = index_file.with_name(index_file.stem + "_edge_data.bin")
    if cfg.get("smt_data_file"):
        smt_data_file = resolve_path(cfg, cfg["smt_data_file"])
    else:
        smt_data_file = index_file.with_name(index_file.stem + "_smt_nodes.bin")
    storage_cfg = cfg.get("disk_storage", {})
    shard_count = int(storage_cfg.get("construction_shards", 512))
    block_cache_bytes = int(
        float(storage_cfg.get("edge_block_cache_mib", 512)) * 1024 * 1024
    )
    smt_cache_bytes = int(
        float(storage_cfg.get("edge_smt_cache_mib", 1024)) * 1024 * 1024
    )

    # The temporary shards live beside the final outputs so the multi-gigabyte
    # construction stream stays on the same drive and is removed on completion.
    spool_parent = edge_data_file.parent
    spool_parent.mkdir(parents=True, exist_ok=True)
    spool_started = perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f".{city}_cell50_spool_", dir=spool_parent
    ) as spool_directory:
        spool = ShardedEntrySpool(spool_directory, shard_count=shard_count)
        spool_stats = spool_cell_trajectories(
            city=city,
            source_road=source_road,
            partition=partition,
            trajectory_globs=list(cfg["trajectory_globs"]),
            config_dir=Path(cfg["_config_dir"]),
            spool=spool,
            catalog_file=catalog_file,
            expected_file_count=cfg.get("expected_trajectory_file_count"),
        )
        spool_seconds = perf_counter() - spool_started
        print(
            f"trajectory spool complete: {spool_stats['record_count']} records "
            f"in {spool_seconds:.3f} s",
            flush=True,
        )

        index, disk_build_stats = build_disk_smt_baseline(
            road=partition.road,
            spool=spool,
            edge_data_file=edge_data_file,
            theta=int(cfg.get("theta", 64)),
            trajectory_files=spool_stats["files"],
            block_cache_bytes=block_cache_bytes,
            smt_cache_bytes=smt_cache_bytes,
            smt_data_file=smt_data_file,
        )

    construction_seconds = (
        partition_seconds
        + spool_seconds
        + disk_build_stats["edge_build_seconds"]
        + disk_build_stats["spatial_tree_seconds"]
    )

    trusted_root_file = _required_path(cfg, "trusted_root_file")
    manifest_size = save_index(index, index_file)
    edge_data_size = int(disk_build_stats["edge_data_size_bytes"])
    smt_data_size = int(disk_build_stats["smt_data_size_bytes"])
    logical_index_size = manifest_size + edge_data_size + smt_data_size
    ensure_parent(trusted_root_file)
    trusted_root_file.write_text(index.root_hash.hex() + "\n", encoding="ascii")

    metrics = {
        "mode": "DISK-BACKED PER-EDGE-SMT; authenticated query algorithm unchanged",
        "city": city,
        "grid": grid.as_dict(),
        "source_node_count": source_nodes,
        "source_edge_count": source_edges,
        "partitioned_node_count": partitioned_nodes,
        "partitioned_edge_count": partitioned_edges,
        "added_edge_count": partitioned_edges - source_edges,
        "trajectory_count": spool_stats["trajectory_count"],
        "source_trajectory_segment_count": spool_stats["source_segment_count"],
        "partitioned_trajectory_segment_count": spool_stats["partitioned_segment_count"],
        "added_trajectory_segment_count": (
            spool_stats["partitioned_segment_count"]
            - spool_stats["source_segment_count"]
        ),
        "smt_leaf_count": sum(index.store._unique_counts),
        "populated_edge_count": disk_build_stats["populated_edge_count"],
        "one_edge_one_cell_validated": True,
        "partition_time_s": partition_seconds,
        "trajectory_transform_and_spool_time_s": spool_seconds,
        "per_edge_chain_and_smt_build_time_s": disk_build_stats["edge_build_seconds"],
        "smt_node_persistence_time_s": disk_build_stats["smt_persistence_seconds"],
        "spatial_tree_build_time_s": disk_build_stats["spatial_tree_seconds"],
        "index_construction_time_s": construction_seconds,
        "index_manifest_size_bytes": manifest_size,
        "edge_data_size_bytes": edge_data_size,
        "smt_data_size_bytes": smt_data_size,
        "index_size_bytes": logical_index_size,
        "edge_data_sha256": disk_build_stats["edge_data_sha256"],
        "catalog_size_bytes": spool_stats["catalog_size_bytes"],
        "construction_shards": shard_count,
        "query_edge_block_cache_bytes": block_cache_bytes,
        "query_edge_smt_cache_bytes": smt_cache_bytes,
        "query_timing_policy": (
            "lazy edge-data and persisted-SMT loading/deserialization measured during traversal "
            "but excluded from reported query_time_s"
        ),
        "trusted_root_hex": index.root_hash.hex(),
        "source_node_file": str(_required_path(cfg, "source_node_file")),
        "source_edge_file": str(_required_path(cfg, "source_edge_file")),
        "node_file": str(node_file),
        "edge_file": str(edge_file),
        "edge_cell_file": str(edge_cell_file),
        "index_file": str(index_file),
        "edge_data_file": str(edge_data_file),
        "smt_data_file": str(smt_data_file),
        "catalog_file": str(catalog_file),
        "trusted_root_file": str(trusted_root_file),
    }
    metrics_file = resolve_path(
        cfg,
        cfg.get(
            "build_metrics_file",
            str(Path(cfg["index_file"]).with_name("build_metrics.json")),
        ),
    )
    _write_json(metrics_file, metrics)

    print("BUILD_SUCCESS", flush=True)
    print("mode: disk-backed existing PER-EDGE-SMT; cryptographic structure unchanged")
    print("trajectories:", spool_stats["trajectory_count"])
    print("source trajectory segments:", spool_stats["source_segment_count"])
    print("partitioned trajectory segments:", spool_stats["partitioned_segment_count"])
    print("SMT leaves:", metrics["smt_leaf_count"])
    print(f"Index Construction Time: {construction_seconds:.9f} s")
    print(f"Index Manifest Size: {manifest_size / 1024 / 1024:.3f} MiB")
    print(f"Edge Data Size: {edge_data_size / 1024 / 1024:.3f} MiB")
    print(f"Persisted SMT Nodes Size: {smt_data_size / 1024 / 1024:.3f} MiB")
    print(f"Logical Index Size: {logical_index_size / 1024 / 1024:.3f} MiB")
    print("trusted root:", index.root_hash.hex())
    print("metrics:", metrics_file)


if __name__ == "__main__":
    main()
