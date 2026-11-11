"""Split one original straight-line edge at every Geohash grid boundary."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from geohash.encoder import GeohashGrid
from .reader import NetworkInput, OriginalEdge


@dataclass(frozen=True)
class SplitPiece:
    original_edge_id: int
    segment_index: int
    t_start: Fraction
    t_end: Fraction
    start_lon_scaled: Fraction
    start_lat_scaled: Fraction
    end_lon_scaled: Fraction
    end_lat_scaled: Fraction
    cell_id: str


def _unique_sorted(values: list[Fraction]) -> list[Fraction]:
    return sorted(set(values))


def split_original_edge(
    edge: OriginalEdge,
    network: NetworkInput,
    grid: GeohashGrid,
) -> list[SplitPiece]:
    """Return contiguous pieces in the original edge orientation."""

    start = network.nodes[edge.start_node]
    end = network.nodes[edge.end_node]
    parameters = [Fraction(0), Fraction(1)]
    parameters.extend(
        grid.boundary_parameters(
            start.lon_scaled,
            start.lat_scaled,
            end.lon_scaled,
            end.lat_scaled,
        )
    )
    parameters = _unique_sorted(parameters)

    pieces: list[SplitPiece] = []
    for index, (t_start, t_end) in enumerate(zip(parameters, parameters[1:])):
        if t_end <= t_start:
            continue
        start_lon, start_lat = grid.point_on_segment(
            start.lon_scaled,
            start.lat_scaled,
            end.lon_scaled,
            end.lat_scaled,
            t_start,
        )
        end_lon, end_lat = grid.point_on_segment(
            start.lon_scaled,
            start.lat_scaled,
            end.lon_scaled,
            end.lat_scaled,
            t_end,
        )
        cell_id = grid.cell_for_segment_midpoint(
            start.lon_scaled,
            start.lat_scaled,
            end.lon_scaled,
            end.lat_scaled,
            t_start,
            t_end,
        )
        pieces.append(
            SplitPiece(
                original_edge_id=edge.edge_id,
                segment_index=index,
                t_start=t_start,
                t_end=t_end,
                start_lon_scaled=start_lon,
                start_lat_scaled=start_lat,
                end_lon_scaled=end_lon,
                end_lat_scaled=end_lat,
                cell_id=cell_id,
            )
        )
    if not pieces:
        raise ValueError(f"original edge {edge.edge_id} produced no split pieces")
    return pieces
