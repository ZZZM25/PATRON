"""Deterministic metric-cell subdivision for road networks.

This module is deliberately independent from the authenticated index.  It
turns one source ``RoadNetwork`` into another, larger ``RoadNetwork`` whose
edges are cut at every metric-grid boundary.  The existing spatial tree, time
list, per-edge SMT, VO, and verifier can therefore consume the subdivided road
without knowing anything about cells.

The projection is a local WGS84 tangent-plane linearization.  It is affine in
longitude/latitude, which is important here: a point interpolated on a source
edge remains exactly on the projected straight segment.  At city scale the
configured cells are approximately square on the ellipsoid and exactly square
in this local metric plane.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .models import TrajectorySegment
from .road_network import RoadNetwork


_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.6943799901413165e-3
_T_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class MetricCellGrid:
    """A deterministic local metric grid with half-open cells.

    Cell ``(i, j)`` represents ``[i*w, (i+1)*w) x [j*h, (j+1)*h)`` after
    subtracting the configured metric origin.  An edge lying exactly on a grid
    line is assigned by this same half-open rule, so it still has one cell.
    """

    reference_lon: float
    reference_lat: float
    cell_width_m: float = 50.0
    cell_height_m: float = 50.0
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0

    def __post_init__(self) -> None:
        if not (-180.0 <= self.reference_lon <= 180.0):
            raise ValueError("reference_lon outside [-180, 180]")
        if not (-90.0 < self.reference_lat < 90.0):
            raise ValueError("reference_lat outside (-90, 90)")
        if self.cell_width_m <= 0.0 or self.cell_height_m <= 0.0:
            raise ValueError("cell dimensions must be positive")

    @property
    def _scales(self) -> tuple[float, float]:
        """Return metres per radian for longitude and latitude at the origin."""
        phi = math.radians(self.reference_lat)
        sin_phi = math.sin(phi)
        denominator = math.sqrt(1.0 - _WGS84_E2 * sin_phi * sin_phi)
        prime_vertical_radius = _WGS84_A / denominator
        meridional_radius = (
            _WGS84_A * (1.0 - _WGS84_E2) / (denominator ** 3)
        )
        return prime_vertical_radius * math.cos(phi), meridional_radius

    def project(self, lon: float, lat: float) -> tuple[float, float]:
        lon_scale, lat_scale = self._scales
        return (
            math.radians(float(lon) - self.reference_lon) * lon_scale,
            math.radians(float(lat) - self.reference_lat) * lat_scale,
        )

    def unproject(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Invert :meth:`project` using the same affine city-scale model."""
        lon_scale, lat_scale = self._scales
        return (
            self.reference_lon + math.degrees(float(x_m) / lon_scale),
            self.reference_lat + math.degrees(float(y_m) / lat_scale),
        )

    def cell_for_xy(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (
            math.floor((x_m - self.origin_x_m) / self.cell_width_m),
            math.floor((y_m - self.origin_y_m) / self.cell_height_m),
        )

    def cell_for_lon_lat(self, lon: float, lat: float) -> tuple[int, int]:
        return self.cell_for_xy(*self.project(lon, lat))

    def as_dict(self) -> dict[str, float | str]:
        return {
            "projection": "WGS84 local tangent-plane affine linearization",
            "cell_boundary_rule": "half-open [min,max) on x and y",
            "reference_lon": self.reference_lon,
            "reference_lat": self.reference_lat,
            "cell_width_m": self.cell_width_m,
            "cell_height_m": self.cell_height_m,
            "origin_x_m": self.origin_x_m,
            "origin_y_m": self.origin_y_m,
        }


@dataclass(slots=True)
class CellPartitionedRoad:
    """A split road plus the complete source-edge/sub-edge correspondence."""

    source_road: RoadNetwork
    road: RoadNetwork
    grid: MetricCellGrid
    original_edge_to_subedges: list[tuple[int, ...]]
    edge_original_eid: list[int]
    edge_source_t0: list[float]
    edge_source_t1: list[float]
    edge_cell_x: list[int]
    edge_cell_y: list[int]

    def cell_of_edge(self, eid: int) -> tuple[int, int]:
        if not self.road.edge_exists(eid):
            raise KeyError(eid)
        return self.edge_cell_x[eid], self.edge_cell_y[eid]

    def validate(self) -> None:
        """Check mapping completeness and the one-edge/one-cell invariant."""
        edge_count = self.road.max_eid + 1
        fields = (
            self.edge_original_eid,
            self.edge_source_t0,
            self.edge_source_t1,
            self.edge_cell_x,
            self.edge_cell_y,
        )
        if any(len(field) != edge_count for field in fields):
            raise ValueError("partition edge metadata length mismatch")

        seen: set[int] = set()
        for original_eid in self.source_road.valid_edge_ids():
            subedges = self.original_edge_to_subedges[original_eid]
            if not subedges:
                raise ValueError(f"source edge {original_eid} has no sub-edge")
            previous_t = 0.0
            source_u, source_v = self.source_road.get_edge_nodes(original_eid)
            previous_node = source_u
            for position, eid in enumerate(subedges):
                if eid in seen:
                    raise ValueError(f"sub-edge {eid} appears more than once")
                seen.add(eid)
                if self.edge_original_eid[eid] != original_eid:
                    raise ValueError(f"sub-edge {eid} source mapping mismatch")
                t0 = self.edge_source_t0[eid]
                t1 = self.edge_source_t1[eid]
                if abs(t0 - previous_t) > 1e-10 or not t0 < t1:
                    raise ValueError(f"source edge {original_eid} has a parameter gap")
                u, v = self.road.get_edge_nodes(eid)
                if u != previous_node:
                    raise ValueError(f"source edge {original_eid} is not continuous")

                # Every open sub-edge lies in one cell.  Sampling both halves
                # also catches accidental missed grid intersections.
                source_lon_u = self.source_road.node_lon[source_u]
                source_lat_u = self.source_road.node_lat[source_u]
                source_lon_v = self.source_road.node_lon[source_v]
                source_lat_v = self.source_road.node_lat[source_v]
                expected = self.cell_of_edge(eid)
                for fraction in (0.25, 0.5, 0.75):
                    t = t0 + (t1 - t0) * fraction
                    lon = source_lon_u + (source_lon_v - source_lon_u) * t
                    lat = source_lat_u + (source_lat_v - source_lat_u) * t
                    if self.grid.cell_for_lon_lat(lon, lat) != expected:
                        raise ValueError(
                            f"sub-edge {eid} crosses a cell boundary: {expected}"
                        )
                previous_t = t1
                previous_node = v
                if position == len(subedges) - 1 and v != source_v:
                    raise ValueError(f"source edge {original_eid} ends at wrong node")
            if abs(previous_t - 1.0) > 1e-10:
                raise ValueError(f"source edge {original_eid} does not cover [0,1]")

        if seen != set(self.road.valid_edge_ids()):
            raise ValueError("partition has unmapped sub-edges")


def grid_from_road(
    road: RoadNetwork,
    *,
    cell_width_m: float = 50.0,
    cell_height_m: float | None = None,
    reference_lon: float | None = None,
    reference_lat: float | None = None,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
) -> MetricCellGrid:
    node_ids = road.valid_node_ids()
    if not node_ids:
        raise ValueError("cannot create a grid for an empty road network")
    if reference_lon is None:
        reference_lon = (
            min(road.node_lon[nid] for nid in node_ids)
            + max(road.node_lon[nid] for nid in node_ids)
        ) / 2.0
    if reference_lat is None:
        reference_lat = (
            min(road.node_lat[nid] for nid in node_ids)
            + max(road.node_lat[nid] for nid in node_ids)
        ) / 2.0
    if cell_height_m is None:
        cell_height_m = cell_width_m
    return MetricCellGrid(
        float(reference_lon),
        float(reference_lat),
        float(cell_width_m),
        float(cell_height_m),
        float(origin_x_m),
        float(origin_y_m),
    )


def _interior_grid_parameters(
    a: float,
    b: float,
    cell_size: float,
    origin: float,
) -> list[float]:
    delta = b - a
    if delta == 0.0:
        return []
    lo, hi = sorted((a, b))
    first = math.floor((lo - origin) / cell_size) + 1
    last = math.ceil((hi - origin) / cell_size) - 1
    parameters: list[float] = []
    for grid_index in range(first, last + 1):
        boundary = origin + grid_index * cell_size
        t = (boundary - a) / delta
        if _T_EPSILON < t < 1.0 - _T_EPSILON:
            parameters.append(t)
    return parameters


def _deduplicate_parameters(values: list[float]) -> list[float]:
    values.sort()
    out: list[float] = []
    for value in values:
        if not out or abs(value - out[-1]) > _T_EPSILON:
            out.append(value)
    return out


def split_road_network(
    source_road: RoadNetwork,
    grid: MetricCellGrid,
) -> CellPartitionedRoad:
    """Split every source edge at all x/y cell boundaries.

    Original road nodes keep their IDs.  New virtual nodes start at
    ``source_road.max_nid + 1``.  Sub-edge IDs are dense and deterministic:
    source EID order first, then increasing parameter along its stored u->v
    orientation.
    """
    road = RoadNetwork()
    road.node_lon = list(source_road.node_lon)
    road.node_lat = list(source_road.node_lat)
    road.node_present = list(source_road.node_present)
    road.max_nid = source_road.max_nid
    road.edge_u = []
    road.edge_v = []
    road.edge_present = []
    road.pair_to_eids = {}

    original_edge_to_subedges: list[tuple[int, ...]] = [
        tuple() for _ in range(source_road.max_eid + 1)
    ]
    edge_original_eid: list[int] = []
    edge_source_t0: list[float] = []
    edge_source_t1: list[float] = []
    edge_cell_x: list[int] = []
    edge_cell_y: list[int] = []

    for original_eid in source_road.valid_edge_ids():
        source_u, source_v = source_road.get_edge_nodes(original_eid)
        lon0, lat0 = source_road.node_lon[source_u], source_road.node_lat[source_u]
        lon1, lat1 = source_road.node_lon[source_v], source_road.node_lat[source_v]
        x0, y0 = grid.project(lon0, lat0)
        x1, y1 = grid.project(lon1, lat1)
        cuts = _deduplicate_parameters(
            [0.0, 1.0]
            + _interior_grid_parameters(
                x0, x1, grid.cell_width_m, grid.origin_x_m
            )
            + _interior_grid_parameters(
                y0, y1, grid.cell_height_m, grid.origin_y_m
            )
        )

        nodes = [source_u]
        for t in cuts[1:-1]:
            node_id = len(road.node_lon)
            road.node_lon.append(lon0 + (lon1 - lon0) * t)
            road.node_lat.append(lat0 + (lat1 - lat0) * t)
            road.node_present.append(True)
            nodes.append(node_id)
        nodes.append(source_v)

        subedges: list[int] = []
        for part_index, (t0, t1) in enumerate(zip(cuts, cuts[1:])):
            u, v = nodes[part_index], nodes[part_index + 1]
            eid = len(road.edge_u)
            road.edge_u.append(u)
            road.edge_v.append(v)
            road.edge_present.append(True)
            road.pair_to_eids.setdefault(road._canonical_pair(u, v), []).append(eid)
            midpoint_t = (t0 + t1) / 2.0
            midpoint_x = x0 + (x1 - x0) * midpoint_t
            midpoint_y = y0 + (y1 - y0) * midpoint_t
            cell_x, cell_y = grid.cell_for_xy(midpoint_x, midpoint_y)

            edge_original_eid.append(original_eid)
            edge_source_t0.append(t0)
            edge_source_t1.append(t1)
            edge_cell_x.append(cell_x)
            edge_cell_y.append(cell_y)
            subedges.append(eid)
        original_edge_to_subedges[original_eid] = tuple(subedges)

    road.max_nid = len(road.node_lon) - 1
    road.max_eid = len(road.edge_u) - 1
    result = CellPartitionedRoad(
        source_road=source_road,
        road=road,
        grid=grid,
        original_edge_to_subedges=original_edge_to_subedges,
        edge_original_eid=edge_original_eid,
        edge_source_t0=edge_source_t0,
        edge_source_t1=edge_source_t1,
        edge_cell_x=edge_cell_x,
        edge_cell_y=edge_cell_y,
    )
    result.validate()
    return result


def _interpolate_timestamp(start: int, end: int, fraction: float) -> int:
    if end < start:
        raise ValueError(f"segment time decreases: {start}>{end}")
    if fraction <= 0.0:
        return int(start)
    if fraction >= 1.0:
        return int(end)
    # Round half up.  Reusing the same boundary fraction for adjacent parts
    # guarantees an identical shared timestamp and a gap-free time path.
    return int(start + math.floor((end - start) * fraction + 0.5))


def expand_trajectory_segment(
    partition: CellPartitionedRoad,
    source_eid: int,
    start_node: int,
    start_time: int,
    end_node: int,
    end_time: int,
) -> tuple[list[int], list[TrajectorySegment]]:
    """Replace one source-edge traversal with its ordered sub-edge traversal."""
    source_u, source_v = partition.source_road.get_edge_nodes(source_eid)
    if start_node == source_u and end_node == source_v:
        forward = True
    elif start_node == source_v and end_node == source_u:
        forward = False
    else:
        raise ValueError(
            f"trajectory segment {start_node}->{end_node} does not match "
            f"source edge {source_eid} ({source_u},{source_v})"
        )

    source_subedges = partition.original_edge_to_subedges[source_eid]
    ordered = source_subedges if forward else tuple(reversed(source_subedges))
    eid_path: list[int] = []
    segments: list[TrajectorySegment] = []
    for eid in ordered:
        stored_u, stored_v = partition.road.get_edge_nodes(eid)
        t0 = partition.edge_source_t0[eid]
        t1 = partition.edge_source_t1[eid]
        if forward:
            u, v = stored_u, stored_v
            fraction0, fraction1 = t0, t1
        else:
            u, v = stored_v, stored_u
            fraction0, fraction1 = 1.0 - t1, 1.0 - t0
        segment = TrajectorySegment(
            start_node=u,
            start_time=_interpolate_timestamp(start_time, end_time, fraction0),
            end_node=v,
            end_time=_interpolate_timestamp(start_time, end_time, fraction1),
        )
        eid_path.append(eid)
        segments.append(segment)

    if not segments:
        raise RuntimeError(f"source edge {source_eid} expanded to no segments")
    if segments[0].start_node != start_node or segments[-1].end_node != end_node:
        raise RuntimeError(f"source edge {source_eid} expansion changed endpoints")
    if segments[0].start_time != start_time or segments[-1].end_time != end_time:
        raise RuntimeError(f"source edge {source_eid} expansion changed time range")
    for left, right in zip(segments, segments[1:]):
        if left.end_node != right.start_node or left.end_time != right.start_time:
            raise RuntimeError(f"source edge {source_eid} expansion is discontinuous")
    return eid_path, segments


def write_partition_artifacts(
    partition: CellPartitionedRoad,
    node_file: str | Path,
    edge_file: str | Path,
    edge_cell_file: str | Path,
    metadata_file: str | Path,
) -> None:
    """Materialize the split road and its audit mapping for existing tools."""
    node_path = Path(node_file)
    edge_path = Path(edge_file)
    mapping_path = Path(edge_cell_file)
    metadata_path = Path(metadata_file)
    for path in (node_path, edge_path, mapping_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with node_path.open("w", encoding="utf-8", newline="\n") as stream:
        for nid in partition.road.valid_node_ids():
            stream.write(
                f"{nid} {partition.road.node_lon[nid]:.17g} "
                f"{partition.road.node_lat[nid]:.17g}\n"
            )
    with edge_path.open("w", encoding="utf-8", newline="\n") as stream:
        for eid in partition.road.valid_edge_ids():
            u, v = partition.road.get_edge_nodes(eid)
            stream.write(f"{eid} {u} {v}\n")

    with mapping_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sub_eid",
                "source_eid",
                "sub_u",
                "sub_v",
                "source_t0",
                "source_t1",
                "cell_x",
                "cell_y",
            ]
        )
        for eid in partition.road.valid_edge_ids():
            u, v = partition.road.get_edge_nodes(eid)
            writer.writerow(
                [
                    eid,
                    partition.edge_original_eid[eid],
                    u,
                    v,
                    format(partition.edge_source_t0[eid], ".17g"),
                    format(partition.edge_source_t1[eid], ".17g"),
                    partition.edge_cell_x[eid],
                    partition.edge_cell_y[eid],
                ]
            )

    source_edge_count = len(partition.source_road.valid_edge_ids())
    split_edge_count = len(partition.road.valid_edge_ids())
    metadata = {
        "format": "cell-partitioned-road-v1",
        "grid": partition.grid.as_dict(),
        "source_node_count": len(partition.source_road.valid_node_ids()),
        "source_edge_count": source_edge_count,
        "partitioned_node_count": len(partition.road.valid_node_ids()),
        "partitioned_edge_count": split_edge_count,
        "added_edge_count": split_edge_count - source_edge_count,
        "one_edge_one_cell_validated": True,
        "node_file": str(node_path.resolve()),
        "edge_file": str(edge_path.resolve()),
        "edge_cell_file": str(mapping_path.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
