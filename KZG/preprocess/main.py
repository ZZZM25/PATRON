"""Run the road-cell splitting and trajectory conversion pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PipelineConfig
from pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    default_input = project_root / "input"
    parser = argparse.ArgumentParser(
        description=(
            "Split each road edge at Geohash cell boundaries, then expand "
            "nested JSON trajectories into the new-edge CSV format."
        )
    )
    parser.add_argument(
        "--nodes",
        default=str(default_input / "chengdu_nodes.txt"),
        help="original node text file",
    )
    parser.add_argument(
        "--edges",
        default=str(default_input / "chengdu_edges.txt"),
        help="original edge text file",
    )
    parser.add_argument(
        "--trajectories",
        default=str(default_input / "trajectories.json"),
        help="original trajectory JSON",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "output"),
        help="output directory",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=50.0,
        help="target Geohash cell geometric-mean size in metres (default: 50)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="fixed binary-Geohash bit depth; omit to select it automatically",
    )
    parser.add_argument(
        "--timezone-offset-hours",
        type=int,
        default=8,
        help="UTC offset used for trajectory ID dates (default: 8)",
    )
    parser.add_argument(
        "--trajectory-index-start",
        type=int,
        default=0,
        help="first numeric component of trajectory_id (default: 0)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig.from_paths(
        nodes_path=args.nodes,
        edges_path=args.edges,
        trajectories_path=args.trajectories,
        output_dir=args.output,
        target_cell_size_m=args.cell_size,
        geohash_bits=args.bits,
        timezone_offset_hours=args.timezone_offset_hours,
        trajectory_index_start=args.trajectory_index_start,
    )
    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
