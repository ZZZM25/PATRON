"""Post-build checks for topology, one-cell ownership, and time conservation."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

from geohash.encoder import GeohashGrid
from network.edge_builder import SplitNetwork
from network.reader import NetworkInput


def validate_split_network(
    network: NetworkInput, split_network: SplitNetwork, grid: GeohashGrid
) -> dict[str, int | float]:
    if not split_network.edges:
        raise AssertionError("split network has no edges")
    if len(split_network.by_id) != len(split_network.edges):
        raise AssertionError("new edge IDs are not unique")

    for edge in split_network.edges:
        if edge.start_node not in split_network.nodes or edge.end_node not in split_network.nodes:
            raise AssertionError(f"new edge {edge.edge_id} references a missing node")
        if not edge.cell_id or len(edge.cell_id) != grid.bits:
            raise AssertionError(f"new edge {edge.edge_id} has invalid cell ID")
        original = network.by_id[edge.original_edge_id]
        original_start = network.nodes[original.start_node]
        original_end = network.nodes[original.end_node]
        # Validate with the exact fractional positions stored in the mapping,
        # rather than converting a boundary point to decimal text and back.
        # The latter can move a point by a few nanodegrees and create a false
        # crossing at the same boundary.
        start_lon, start_lat = grid.point_on_segment(
            original_start.lon_scaled,
            original_start.lat_scaled,
            original_end.lon_scaled,
            original_end.lat_scaled,
            edge.t_start,
        )
        end_lon, end_lat = grid.point_on_segment(
            original_start.lon_scaled,
            original_start.lat_scaled,
            original_end.lon_scaled,
            original_end.lat_scaled,
            edge.t_end,
        )
        crossings = grid.boundary_parameters(start_lon, start_lat, end_lon, end_lat)
        if crossings:
            raise AssertionError(
                f"new edge {edge.edge_id} still crosses a grid boundary: {crossings[:3]}"
            )

    for original in network.edges:
        pieces = split_network.pieces_by_original_edge[original.edge_id]
        if pieces[0].t_start != 0 or pieces[-1].t_end != 1:
            raise AssertionError(f"original edge {original.edge_id} is not fully covered")
        for left, right in zip(pieces, pieces[1:]):
            if left.t_end != right.t_start:
                raise AssertionError(f"gap/overlap in original edge {original.edge_id}")
        if pieces[0].start_node != original.start_node or pieces[-1].end_node != original.end_node:
            raise AssertionError(f"endpoint mismatch for original edge {original.edge_id}")

    return {
        "original_nodes": len(network.nodes),
        "original_edges": len(network.edges),
        "new_nodes": len(split_network.nodes),
        "new_edges": len(split_network.edges),
        "cells_used": len(split_network.edges_by_cell),
        "split_original_edges": sum(
            1
            for pieces in split_network.pieces_by_original_edge.values()
            if len(pieces) > 1
        ),
        "max_segments_per_original_edge": max(
            len(pieces) for pieces in split_network.pieces_by_original_edge.values()
        ),
    }


def validate_trajectory_rows(
    rows: list[tuple[str, list[list[int]]]], split_network: SplitNetwork
) -> dict[str, int]:
    seen_ids: set[str] = set()
    record_count = 0
    for trajectory_id, records in rows:
        if trajectory_id in seen_ids:
            raise AssertionError(f"duplicate trajectory ID: {trajectory_id}")
        seen_ids.add(trajectory_id)
        previous_end: int | None = None
        for record in records:
            if len(record) != 5:
                raise AssertionError(f"trajectory {trajectory_id}: malformed record {record}")
            edge_id, start, start_time, end, end_time = record
            if edge_id not in split_network.by_id:
                raise AssertionError(f"trajectory {trajectory_id}: unknown edge {edge_id}")
            if start not in split_network.nodes or end not in split_network.nodes:
                raise AssertionError(f"trajectory {trajectory_id}: unknown node in {record}")
            if end_time < start_time:
                raise AssertionError(f"trajectory {trajectory_id}: negative time in {record}")
            if previous_end is not None and start != previous_end:
                raise AssertionError(f"trajectory {trajectory_id}: broken node chain")
            previous_end = end
            record_count += 1
    return {"trajectories": len(rows), "generated_trajectory_records": record_count}
