from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import prepare_rsg_dataset as prepare


ROOT = Path(__file__).resolve().parent
DEFAULT_REAL2 = ROOT / "data" / "xian" / "source"
DEFAULT_ROAD_DIR = ROOT / "data" / "xian" / "road"
DEFAULT_OUTPUT = ROOT / "rsg_50m_xian_2018-10-01-roadtxt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备西安 2018-10-01 的精确 50 米 RSG 数据")
    parser.add_argument("--real2-dir", type=Path, default=DEFAULT_REAL2)
    parser.add_argument(
        "--nodes-file", type=Path, default=DEFAULT_ROAD_DIR / "xian_nodes.txt"
    )
    parser.add_argument(
        "--edges-file", type=Path, default=DEFAULT_ROAD_DIR / "xian_edges.txt"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real2_dir = args.real2_dir.resolve()
    output_dir = args.output_dir.resolve()
    nodes_path = args.nodes_file.resolve()
    edges_path = args.edges_file.resolve()
    trajectory_path = real2_dir / "trajectories" / "traj_mapped_xian_xian10-1.json"
    for path in (nodes_path, edges_path, trajectory_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    cleaned_edges_path, edge_audit = prepare.prepare_non_degenerate_edges(
        nodes_path, edges_path, output_dir
    )
    prepare.build_exact_50m_network(nodes_path, cleaned_edges_path, output_dir)
    grid_metadata = prepare.write_spatial_index(nodes_path, output_dir)
    mapping_summary = prepare.map_trajectories_streaming(
        cleaned_edges_path,
        trajectory_path,
        output_dir,
        output_name="traj-10-01.csv",
    )
    manifest = {
        "dataset_city": "xian",
        "dataset_date": "2018-10-01",
        "timezone": "UTC",
        "cell_size_m": 50.0,
        "source_nodes_file": str(nodes_path),
        "source_edges_file": str(edges_path),
        "source_trajectory_file": str(trajectory_path),
        "output_dir": str(output_dir),
        "source_edge_audit": edge_audit,
        "grid": grid_metadata,
        "mapping": mapping_summary,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[西安数据准备完成] {output_dir}")


if __name__ == "__main__":
    main()
