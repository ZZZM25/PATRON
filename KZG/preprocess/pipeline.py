"""End-to-end orchestration for network splitting and trajectory conversion."""

from __future__ import annotations

import json
from pathlib import Path

from cell_index.index_builder import build_authenticated_index
from config import PipelineConfig
from geohash.encoder import GeohashGrid
from network.edge_builder import build_split_network, write_split_network
from network.reader import read_network
from trajectory.json_reader import load_trajectories
from trajectory.processor import TrajectoryProcessor, write_trajectory_csv
from validation import validate_split_network, validate_trajectory_rows


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    network = read_network(
        config.nodes_path,
        config.edges_path,
        coordinate_scale=config.coordinate_scale,
    )
    reference_latitude = sum(node.lat for node in network.nodes.values()) / len(network.nodes)
    if config.geohash_bits is None:
        grid = GeohashGrid.auto(
            target_cell_size_m=config.target_cell_size_m,
            reference_latitude=reference_latitude,
            coordinate_scale=config.coordinate_scale,
        )
    else:
        grid = GeohashGrid.fixed(
            bits=config.geohash_bits,
            reference_latitude=reference_latitude,
            coordinate_scale=config.coordinate_scale,
        )

    split_network = build_split_network(
        network,
        grid,
        new_edge_id_start=config.new_edge_id_start,
    )
    network_paths = write_split_network(
        split_network,
        config.output_dir,
        coordinate_digits=config.output_coordinate_digits,
    )
    network_summary = validate_split_network(network, split_network, grid)
    authenticated_index, index_paths = build_authenticated_index(
        split_network, config.output_dir / "cell_index"
    )

    trajectories = load_trajectories(config.trajectories_path)
    processor = TrajectoryProcessor(
        network,
        split_network,
        timezone_offset_hours=config.timezone_offset_hours,
        trajectory_index_start=config.trajectory_index_start,
        fail_on_unmapped_pair=config.fail_on_unmapped_pair,
    )
    rows = processor.process_all(trajectories)
    trajectory_path = write_trajectory_csv(
        rows, config.output_dir / "processed_trajectories.csv"
    )
    trajectory_summary = validate_trajectory_rows(rows, split_network)

    if processor.unmapped_pairs:
        unmapped_path = config.output_dir / "unmapped_trajectory_pairs.json"
        with unmapped_path.open("w", encoding="utf-8") as handle:
            json.dump(processor.unmapped_pairs, handle, ensure_ascii=False, indent=2)
    else:
        unmapped_path = None

    summary: dict[str, object] = {
        "input": {
            "nodes": str(config.nodes_path),
            "edges": str(config.edges_path),
            "trajectories": str(config.trajectories_path),
        },
        "grid": grid.describe(),
        "network": network_summary,
        "trajectories": trajectory_summary,
        "authenticated_index": {
            "root_hash": authenticated_index.root_hash_hex,
            "nonempty_cell_count": authenticated_index.manifest["nonempty_cell_count"],
            "indexed_edge_count": authenticated_index.manifest["indexed_edge_count"],
        },
        "unmapped_pairs": len(processor.unmapped_pairs),
        "outputs": {
            **network_paths,
            "processed_trajectories": trajectory_path,
            "cell_index": index_paths,
        },
    }
    if unmapped_path is not None:
        summary["outputs"] = {**summary["outputs"], "unmapped_pairs": str(unmapped_path)}

    summary_path = config.output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    summary["outputs"] = {**summary["outputs"], "run_summary": str(summary_path)}
    return summary
