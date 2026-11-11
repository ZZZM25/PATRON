"""Generate stable IDs for boundary points created by edge splitting."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .edge_splitter import SplitPiece
from .reader import NetworkInput, RoadNode


@dataclass(frozen=True)
class GeneratedNode:
    node_id: int
    lon: float
    lat: float
    node_type: str
    original_edge_id: int | None = None
    t_on_original_edge: float | None = None


class NodeRegistry:
    """Keep original IDs and add one private chain of nodes per original edge.

    Interior cut points are intentionally keyed by ``(original_edge_id, t)``.
    This preserves the original graph topology: two road edges that merely
    cross at the same geometric point are not accidentally connected.
    """

    def __init__(self, network: NetworkInput, coordinate_scale: int) -> None:
        self.coordinate_scale = coordinate_scale
        self.nodes: dict[int, GeneratedNode] = {
            node_id: GeneratedNode(
                node_id=node_id,
                lon=node.lon,
                lat=node.lat,
                node_type="original",
            )
            for node_id, node in network.nodes.items()
        }
        self._next_id = max(self.nodes, default=-1) + 1
        self._interior: dict[tuple[int, Fraction], int] = {}

    def node_for_piece_start(
        self, original_edge_id: int, t: Fraction, lon_scaled: Fraction, lat_scaled: Fraction, original_start: int
    ) -> int:
        if t == 0:
            return original_start
        return self._interior_node(original_edge_id, t, lon_scaled, lat_scaled)

    def node_for_piece_end(
        self, original_edge_id: int, t: Fraction, lon_scaled: Fraction, lat_scaled: Fraction, original_end: int
    ) -> int:
        if t == 1:
            return original_end
        return self._interior_node(original_edge_id, t, lon_scaled, lat_scaled)

    def _interior_node(
        self, original_edge_id: int, t: Fraction, lon_scaled: Fraction, lat_scaled: Fraction
    ) -> int:
        key = (original_edge_id, t)
        existing = self._interior.get(key)
        if existing is not None:
            return existing
        node_id = self._next_id
        self._next_id += 1
        self._interior[key] = node_id
        self.nodes[node_id] = GeneratedNode(
            node_id=node_id,
            lon=float(lon_scaled / self.coordinate_scale),
            lat=float(lat_scaled / self.coordinate_scale),
            node_type="cell_boundary",
            original_edge_id=original_edge_id,
            t_on_original_edge=float(t),
        )
        return node_id
