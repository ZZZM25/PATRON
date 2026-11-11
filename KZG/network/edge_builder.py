"""Build and write the split road network."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import cos, hypot, pi
from pathlib import Path

from geohash.encoder import GeohashGrid

from .edge_splitter import SplitPiece, split_original_edge
from .node_generator import GeneratedNode, NodeRegistry
from .reader import NetworkInput


METERS_PER_DEGREE_LAT = 110_574.0
METERS_PER_DEGREE_LON_EQUATOR = 111_320.0


@dataclass(frozen=True)
class NewEdge:
    edge_id: int
    start_node: int
    end_node: int
    cell_id: str
    original_edge_id: int
    segment_index: int
    segment_count: int
    t_start: Fraction
    t_end: Fraction
    length_m: float


class SplitNetwork:
    def __init__(
        self,
        nodes: dict[int, GeneratedNode],
        edges: list[NewEdge],
        pieces_by_original_edge: dict[int, list[NewEdge]],
        grid: GeohashGrid,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.by_id = {edge.edge_id: edge for edge in edges}
        self.pieces_by_original_edge = pieces_by_original_edge
        self.grid = grid

    @property
    def edges_by_cell(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = defaultdict(list)
        for edge in self.edges:
            result[edge.cell_id].append(edge.edge_id)
        return dict(sorted(result.items()))


def _piece_length_m(piece: SplitPiece, reference_latitude: float, scale: int) -> float:
    dlon = float(piece.end_lon_scaled - piece.start_lon_scaled) / scale
    dlat = float(piece.end_lat_scaled - piece.start_lat_scaled) / scale
    # A local equirectangular length is enough for a city-sized network and,
    # importantly, is linear along each straight input edge.
    meters_lon = METERS_PER_DEGREE_LON_EQUATOR * cos(reference_latitude * pi / 180.0)
    return hypot(dlon * meters_lon, dlat * METERS_PER_DEGREE_LAT)


def build_split_network(
    network: NetworkInput,
    grid: GeohashGrid,
    new_edge_id_start: int = 0,
) -> SplitNetwork:
    registry = NodeRegistry(network, coordinate_scale=grid.coordinate_scale)
    new_edges: list[NewEdge] = []
    pieces_by_original: dict[int, list[NewEdge]] = {}
    next_edge_id = new_edge_id_start

    for original in sorted(network.edges, key=lambda edge: edge.edge_id):
        pieces = split_original_edge(original, network, grid)
        segment_count = len(pieces)
        built: list[NewEdge] = []
        for piece in pieces:
            start_node = registry.node_for_piece_start(
                original.edge_id,
                piece.t_start,
                piece.start_lon_scaled,
                piece.start_lat_scaled,
                original.start_node,
            )
            end_node = registry.node_for_piece_end(
                original.edge_id,
                piece.t_end,
                piece.end_lon_scaled,
                piece.end_lat_scaled,
                original.end_node,
            )
            new_edge = NewEdge(
                edge_id=next_edge_id,
                start_node=start_node,
                end_node=end_node,
                cell_id=piece.cell_id,
                original_edge_id=original.edge_id,
                segment_index=len(built),
                segment_count=segment_count,
                t_start=piece.t_start,
                t_end=piece.t_end,
                length_m=_piece_length_m(
                    piece, grid.reference_latitude, grid.coordinate_scale
                ),
            )
            built.append(new_edge)
            new_edges.append(new_edge)
            next_edge_id += 1
        pieces_by_original[original.edge_id] = built

    return SplitNetwork(registry.nodes, new_edges, pieces_by_original, grid)


def _frac_text(value: Fraction) -> str:
    return f"{float(value):.12f}"


def write_split_network(
    split_network: SplitNetwork,
    output_dir: str | Path,
    coordinate_digits: int = 10,
) -> dict[str, str]:
    """Write compatible node/edge files plus auditable cell metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "new_nodes.txt"
    edges_path = output_dir / "new_edges.txt"
    edge_cells_path = output_dir / "edge_cells.csv"
    node_meta_path = output_dir / "node_metadata.csv"
    cells_path = output_dir / "edges_by_cell.json"
    mapping_path = output_dir / "original_to_new_edges.csv"

    with nodes_path.open("w", encoding="utf-8", newline="") as handle:
        for node_id in sorted(split_network.nodes):
            node = split_network.nodes[node_id]
            handle.write(
                f"{node.node_id} {node.lon:.{coordinate_digits}f} "
                f"{node.lat:.{coordinate_digits}f}\n"
            )

    with edges_path.open("w", encoding="utf-8", newline="") as handle:
        for edge in split_network.edges:
            handle.write(f"{edge.edge_id} {edge.start_node} {edge.end_node}\n")

    with edge_cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "new_edge_id",
                "start_node",
                "end_node",
                "cell_id",
                "original_edge_id",
                "segment_index",
                "segment_count",
                "length_m",
                "t_start",
                "t_end",
            ]
        )
        for edge in split_network.edges:
            writer.writerow(
                [
                    edge.edge_id,
                    edge.start_node,
                    edge.end_node,
                    edge.cell_id,
                    edge.original_edge_id,
                    edge.segment_index,
                    edge.segment_count,
                    f"{edge.length_m:.9f}",
                    _frac_text(edge.t_start),
                    _frac_text(edge.t_end),
                ]
            )

    with node_meta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["node_id", "lon", "lat", "node_type", "original_edge_id", "t_on_original_edge"]
        )
        for node_id in sorted(split_network.nodes):
            node = split_network.nodes[node_id]
            writer.writerow(
                [
                    node.node_id,
                    f"{node.lon:.{coordinate_digits}f}",
                    f"{node.lat:.{coordinate_digits}f}",
                    node.node_type,
                    "" if node.original_edge_id is None else node.original_edge_id,
                    "" if node.t_on_original_edge is None else f"{node.t_on_original_edge:.12f}",
                ]
            )

    with cells_path.open("w", encoding="utf-8") as handle:
        json.dump(split_network.edges_by_cell, handle, ensure_ascii=False, indent=2)

    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_edge_id", "new_edge_id", "segment_index", "cell_id"])
        for original_id in sorted(split_network.pieces_by_original_edge):
            for edge in split_network.pieces_by_original_edge[original_id]:
                writer.writerow(
                    [original_id, edge.edge_id, edge.segment_index, edge.cell_id]
                )

    return {
        "nodes": str(nodes_path),
        "edges": str(edges_path),
        "edge_cells": str(edge_cells_path),
        "node_metadata": str(node_meta_path),
        "edges_by_cell": str(cells_path),
        "original_to_new_edges": str(mapping_path),
    }
