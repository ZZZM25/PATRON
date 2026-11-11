"""Default settings for the road-cell splitting pipeline.

The program can be run from PyCharm with ``main.py`` or from a terminal with
the command-line options described in ``README.md``.  The defaults below are
intentionally conservative: the input files are supplied through CLI options
and the cell size is approximately 50 metres.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Runtime configuration shared by all pipeline stages."""

    nodes_path: Path
    edges_path: Path
    trajectories_path: Path
    output_dir: Path

    # If geohash_bits is None, the closest standard binary-Geohash precision
    # is selected automatically from target_cell_size_m and the network
    # centre latitude.  For the supplied Chengdu data, 50 m normally selects
    # 38 bits (about 65 m x 38 m at this latitude; geometric mean about 50 m).
    target_cell_size_m: float = 50.0
    geohash_bits: Optional[int] = None

    # Unix timestamps in the supplied data are treated as China local time
    # when the trajectory ID date is formed.  Change to 0 for UTC.
    timezone_offset_hours: int = 8
    trajectory_index_start: int = 0

    # Coordinates are converted to fixed-point degrees for all grid decisions.
    coordinate_scale: int = 10_000_000
    output_coordinate_digits: int = 10

    # New edge IDs are independent of the original edge IDs.  This avoids
    # collisions when one original edge becomes many sub-edges.
    new_edge_id_start: int = 0

    # A malformed or unmappable trajectory pair should stop the run.  This is
    # preferable for paper experiments because silent row loss is dangerous.
    fail_on_unmapped_pair: bool = True

    @classmethod
    def from_paths(
        cls,
        nodes_path: str | Path,
        edges_path: str | Path,
        trajectories_path: str | Path,
        output_dir: str | Path,
        **kwargs,
    ) -> "PipelineConfig":
        return cls(
            nodes_path=Path(nodes_path),
            edges_path=Path(edges_path),
            trajectories_path=Path(trajectories_path),
            output_dir=Path(output_dir),
            **kwargs,
        )
