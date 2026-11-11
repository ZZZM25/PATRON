"""Readers for the supplied Chengdu node and edge text files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RoadNode:
    node_id: int
    lon: float
    lat: float
    lon_scaled: int
    lat_scaled: int


@dataclass(frozen=True)
class OriginalEdge:
    edge_id: int
    start_node: int
    end_node: int


class NetworkInput:
    """Original graph plus deterministic pair-to-edge lookup."""

    def __init__(self, nodes: dict[int, RoadNode], edges: list[OriginalEdge]):
        self.nodes = nodes
        self.edges = edges
        self.by_id = {edge.edge_id: edge for edge in edges}
        if len(self.by_id) != len(edges):
            raise ValueError("duplicate original edge IDs found")

        self.directed: dict[tuple[int, int], list[OriginalEdge]] = defaultdict(list)
        self.undirected: dict[tuple[int, int], list[OriginalEdge]] = defaultdict(list)
        for edge in edges:
            self.directed[(edge.start_node, edge.end_node)].append(edge)
            key = tuple(sorted((edge.start_node, edge.end_node)))
            self.undirected[key].append(edge)

    def resolve_traversal(
        self, start_node: int, end_node: int
    ) -> tuple[OriginalEdge, bool]:
        """Resolve a trajectory node pair.

        Returns ``(edge, stored_forward)``.  ``stored_forward`` is true when
        the trajectory follows the orientation in the edge file; otherwise it
        follows the same physical edge in reverse.
        """

        direct = self.directed.get((start_node, end_node), [])
        if len(direct) == 1:
            return direct[0], True
        if len(direct) > 1:
            raise ValueError(
                f"multiple directed edges for trajectory pair "
                f"{start_node}->{end_node}: {[e.edge_id for e in direct]}"
            )

        reverse = self.directed.get((end_node, start_node), [])
        if len(reverse) == 1:
            return reverse[0], False
        if len(reverse) > 1:
            raise ValueError(
                f"multiple reverse edges for trajectory pair "
                f"{start_node}->{end_node}: {[e.edge_id for e in reverse]}"
            )
        raise KeyError(f"trajectory pair {start_node}->{end_node} is not in the network")


def _iter_data_lines(path: Path) -> Iterable[tuple[int, list[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            yield line_number, line.split()


def read_nodes(path: str | Path, coordinate_scale: int = 10_000_000) -> dict[int, RoadNode]:
    path = Path(path)
    nodes: dict[int, RoadNode] = {}
    for line_number, fields in _iter_data_lines(path):
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_number}: expected node_id lon lat")
        try:
            node_id = int(fields[0])
            lon = float(fields[1])
            lat = float(fields[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid node row {fields}") from exc
        if node_id in nodes:
            raise ValueError(f"{path}:{line_number}: duplicate node ID {node_id}")
        nodes[node_id] = RoadNode(
            node_id=node_id,
            lon=lon,
            lat=lat,
            lon_scaled=round(lon * coordinate_scale),
            lat_scaled=round(lat * coordinate_scale),
        )
    if not nodes:
        raise ValueError(f"no nodes found in {path}")
    return nodes


def read_edges(path: str | Path) -> list[OriginalEdge]:
    path = Path(path)
    edges: list[OriginalEdge] = []
    seen: set[int] = set()
    for line_number, fields in _iter_data_lines(path):
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_number}: expected edge_id start_node end_node")
        try:
            edge_id = int(fields[0])
            start_node = int(fields[1])
            end_node = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid edge row {fields}") from exc
        if edge_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate edge ID {edge_id}")
        seen.add(edge_id)
        edges.append(OriginalEdge(edge_id, start_node, end_node))
    if not edges:
        raise ValueError(f"no edges found in {path}")
    return edges


def read_network(
    nodes_path: str | Path,
    edges_path: str | Path,
    coordinate_scale: int = 10_000_000,
) -> NetworkInput:
    nodes = read_nodes(nodes_path, coordinate_scale=coordinate_scale)
    edges = read_edges(edges_path)
    for edge in edges:
        if edge.start_node not in nodes or edge.end_node not in nodes:
            raise ValueError(
                f"edge {edge.edge_id} references a missing node: "
                f"{edge.start_node}->{edge.end_node}"
            )
    return NetworkInput(nodes, edges)
