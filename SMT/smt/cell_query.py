"""Cell-cover spatial query policy for Cell50 road networks.

The original rectangle is interpreted as a request for the smallest contiguous
set of fixed grid cells whose union covers that rectangle.  Spatial-tree
pruning uses the outer rectangle of those cells, while edge expansion is based
only on the authenticated road edge's pre-agreed cell assignment.  No
per-edge segment/rectangle intersection test is used in this mode.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .cell_partition import CellPartitionedRoad, MetricCellGrid
from .config import resolve_path
from .query_utils import normalize_rectangle


CELL_COVER_QUERY_MODE = "minimum-covering-cells-v1"


@dataclass(frozen=True, slots=True)
class CellCover:
    """Compact rectangular cell range selected for one spatial query."""

    original_bounds: tuple[float, float, float, float]
    min_cell_x: int
    min_cell_y: int
    max_cell_x: int
    max_cell_y: int
    effective_bounds: tuple[float, float, float, float]

    @property
    def cell_count(self) -> int:
        return (
            (self.max_cell_x - self.min_cell_x + 1)
            * (self.max_cell_y - self.min_cell_y + 1)
        )

    def contains_cell(self, cell_x: int, cell_y: int) -> bool:
        return (
            self.min_cell_x <= cell_x <= self.max_cell_x
            and self.min_cell_y <= cell_y <= self.max_cell_y
        )


@dataclass(frozen=True, slots=True)
class CellCoverPolicy:
    """Verifier-visible grid and deterministic edge-to-cell assignment."""

    grid: MetricCellGrid
    edge_cell_x: tuple[int, ...]
    edge_cell_y: tuple[int, ...]
    mode: str = CELL_COVER_QUERY_MODE

    @classmethod
    def from_partition(cls, partition: CellPartitionedRoad) -> "CellCoverPolicy":
        return cls(
            partition.grid,
            tuple(partition.edge_cell_x),
            tuple(partition.edge_cell_y),
        )

    @classmethod
    def from_csv(
        cls,
        grid: MetricCellGrid,
        edge_cell_file: str | Path,
        road,
        *,
        validate_geometry: bool = True,
    ) -> "CellCoverPolicy":
        """Load and audit the complete ``sub_eid -> cell`` assignment.

        The CSV is not accepted as an unchecked server hint.  It must contain
        every road edge exactly once, and by default each assignment is
        independently recomputed from the split edge's interior midpoint.
        """
        edge_count = road.max_eid + 1
        cell_x: list[int | None] = [None] * edge_count
        cell_y: list[int | None] = [None] * edge_count
        path = Path(edge_cell_file)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"sub_eid", "cell_x", "cell_y"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"invalid edge-cell mapping header: {path}")
            for row in reader:
                eid = int(row["sub_eid"])
                if not 0 <= eid < edge_count or not road.edge_exists(eid):
                    raise ValueError(f"edge-cell mapping has invalid eid {eid}")
                if cell_x[eid] is not None:
                    raise ValueError(f"duplicate edge-cell mapping for eid {eid}")
                cx, cy = int(row["cell_x"]), int(row["cell_y"])
                if validate_geometry:
                    u, v = road.get_edge_nodes(eid)
                    lon0, lat0 = road.node_lon[u], road.node_lat[u]
                    lon1, lat1 = road.node_lon[v], road.node_lat[v]
                    for fraction in (0.25, 0.5, 0.75):
                        lon = lon0 + (lon1 - lon0) * fraction
                        lat = lat0 + (lat1 - lat0) * fraction
                        if grid.cell_for_lon_lat(lon, lat) != (cx, cy):
                            raise ValueError(
                                "edge-cell mapping disagrees with split road "
                                f"geometry for eid {eid}"
                            )
                cell_x[eid], cell_y[eid] = cx, cy

        missing = [
            eid for eid in road.valid_edge_ids()
            if cell_x[eid] is None or cell_y[eid] is None
        ]
        if missing:
            raise ValueError(
                f"edge-cell mapping is incomplete; first missing eid {missing[0]}"
            )
        # Invalid/nonexistent EIDs are never queried; zero placeholders keep
        # direct integer indexing compact and deterministic.
        return cls(
            grid,
            tuple(0 if value is None else value for value in cell_x),
            tuple(0 if value is None else value for value in cell_y),
        )

    def cover_rectangle(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> CellCover:
        original = normalize_rectangle(min_lon, min_lat, max_lon, max_lat)
        qmin_lon, qmin_lat, qmax_lon, qmax_lat = original
        min_x_m, min_y_m = self.grid.project(qmin_lon, qmin_lat)
        max_x_m, max_y_m = self.grid.project(qmax_lon, qmax_lat)
        min_cell_x, min_cell_y = self.grid.cell_for_xy(min_x_m, min_y_m)
        max_cell_x, max_cell_y = self.grid.cell_for_xy(max_x_m, max_y_m)

        outer_min_x = self.grid.origin_x_m + min_cell_x * self.grid.cell_width_m
        outer_min_y = self.grid.origin_y_m + min_cell_y * self.grid.cell_height_m
        outer_max_x = (
            self.grid.origin_x_m + (max_cell_x + 1) * self.grid.cell_width_m
        )
        outer_max_y = (
            self.grid.origin_y_m + (max_cell_y + 1) * self.grid.cell_height_m
        )
        effective_min_lon, effective_min_lat = self.grid.unproject(
            outer_min_x, outer_min_y
        )
        effective_max_lon, effective_max_lat = self.grid.unproject(
            outer_max_x, outer_max_y
        )
        return CellCover(
            original,
            min_cell_x,
            min_cell_y,
            max_cell_x,
            max_cell_y,
            (
                effective_min_lon,
                effective_min_lat,
                effective_max_lon,
                effective_max_lat,
            ),
        )

    def edge_matches(self, eid: int, cover: CellCover) -> bool:
        if not 0 <= eid < len(self.edge_cell_x):
            return False
        return cover.contains_cell(self.edge_cell_x[eid], self.edge_cell_y[eid])


def cell_cover_policy_from_config(config: dict, road) -> CellCoverPolicy:
    """Build the agreed Cell50 policy from an already resolved JSON config."""
    cell = config.get("cell_partition")
    if not isinstance(cell, dict):
        raise ValueError("cell_partition object is required for Cell-cover query mode")
    size = float(cell.get("size_m", 50.0))
    width = float(cell.get("width_m", size))
    height = float(cell.get("height_m", size))
    grid = MetricCellGrid(
        reference_lon=float(cell["reference_lon"]),
        reference_lat=float(cell["reference_lat"]),
        cell_width_m=width,
        cell_height_m=height,
        origin_x_m=float(cell.get("origin_x_m", 0.0)),
        origin_y_m=float(cell.get("origin_y_m", 0.0)),
    )
    mapping = config.get("edge_cell_file")
    if not mapping:
        raise ValueError("edge_cell_file is required for Cell-cover query mode")
    return CellCoverPolicy.from_csv(grid, resolve_path(config, mapping), road)
