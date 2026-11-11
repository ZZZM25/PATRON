from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import prepare_rsg_dataset as prepare


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map one daily trajectory source onto an existing 50 m RSG road network"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_date = date.fromisoformat(args.date)
    dataset_dir = args.dataset_dir.resolve()
    source = args.source.resolve()
    cleaned_edges = dataset_dir / "source_edges_nondegenerate.txt"
    for path in (
        source,
        cleaned_edges,
        dataset_dir / "subedges_grouped.csv",
        dataset_dir / "new_edges.csv",
        dataset_dir / "new_nodes.csv",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    suffix = dataset_date.strftime("%m-%d")
    summary = prepare.map_trajectories_streaming(
        cleaned_edges,
        source,
        dataset_dir,
        output_name=f"traj-{suffix}.csv",
        failed_name=f"mapped_trajectories_failed-{suffix}.csv",
        summary_name=f"trajectory_mapping_summary-{suffix}.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
