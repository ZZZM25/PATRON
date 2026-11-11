from __future__ import annotations

from .crypto import time_overlap
from .models import VerificationEntry
from .query_utils import normalize_rectangle, rectangle_overlap, edge_intersects_query
from .spatial_tree import NODE_LEAF
from .vo import (
    BaselineVO, QueryStats, ServerQueryResponse,
    TOKEN_NODE_PRUNED_LEAF, TOKEN_NODE_PRUNED_INTERNAL, TOKEN_NODE_LEAF, TOKEN_NODE_INTERNAL,
    TOKEN_EDGE_OPAQUE, TOKEN_EDGE_PREFIX,
)


class BaselineServerQuery:
    """Query the sorted-list baseline.

    For a spatially matched edge, entries are sorted by start.  The server finds
    k = upper_bound(start <= query_end), expands ALL entries [0, k), and tests
    their end times.  Entry k, when present, is returned as a boundary witness
    proving the first excluded start is > query_end.  The remaining suffix is
    represented by one precomputed reverse-chain state.
    """

    def __init__(self, index, query_min_lon, query_min_lat, query_max_lon, query_max_lat, query_start, query_end):
        if query_start > query_end:
            raise ValueError("query_start > query_end")
        self.index = index
        self.road = index.road
        self.store = index.store
        self.tree = index.tree
        self.qminx, self.qminy, self.qmaxx, self.qmaxy = normalize_rectangle(
            query_min_lon, query_min_lat, query_max_lon, query_max_lat
        )
        self.qs = int(query_start)
        self.qe = int(query_end)
        self.tokens = []
        self.vs = []
        self.candidates = set()
        self.stats = QueryStats()

    def _node_overlaps(self, idx):
        t = self.tree
        return rectangle_overlap(
            t.min_lon[idx], t.min_lat[idx], t.max_lon[idx], t.max_lat[idx],
            self.qminx, self.qminy, self.qmaxx, self.qmaxy,
        )

    def _edge_spatial_match(self, eid: int) -> bool:
        """Return whether an edge belongs to this query's spatial result.

        The legacy mode uses exact segment/rectangle intersection.  Cell-cover
        queries override this hook with deterministic edge-to-cell membership.
        """
        return edge_intersects_query(
            eid, self.road, self.qminx, self.qminy, self.qmaxx, self.qmaxy
        )

    def _emit_pruned(self, idx):
        t = self.tree
        m = (t.min_lon[idx], t.min_lat[idx], t.max_lon[idx], t.max_lat[idx])
        edges = t.node_edges[idx]
        if t.node_type[idx] == NODE_LEAF:
            self.tokens.append((TOKEN_NODE_PRUNED_LEAF, *m, len(edges), t.edge_group_root[idx]))
        else:
            self.tokens.append((
                TOKEN_NODE_PRUNED_INTERNAL, *m,
                t.node_hash[t.left_child[idx]], t.node_hash[t.right_child[idx]],
                len(edges), t.edge_group_root[idx],
            ))

    def _emit_edge(self, eid: int, is_cross: bool):
        if is_cross:
            self.stats.checked_cross_edges += 1
        else:
            self.stats.checked_leaf_edges += 1

        count = self.store.entry_counts[eid]
        if not self._edge_spatial_match(eid):
            self.tokens.append((TOKEN_EDGE_OPAQUE, eid, count, self.index.edge_entry_roots[eid]))
            return

        self.stats.spatial_matched_edges += 1

        # Sorted-list temporal boundary: all entries before k have start <= qe.
        k = self.store.upper_bound_start(eid, self.qe)
        has_boundary = 1 if k < count else 0

        # If a boundary witness exists, the compact suffix starts AFTER it.
        suffix_position = k + 1 if has_boundary else k
        suffix_state = self.index.suffix_state(eid, suffix_position)
        self.tokens.append((TOKEN_EDGE_PREFIX, eid, count, k, has_boundary, suffix_state))

        # Every entry in the prefix is disclosed and hashed by the client,
        # including false positives with end < query_start.
        for tid, start, end in self.store.iter_prefix(eid, k):
            self.vs.append(VerificationEntry(eid, tid, start, end))
            self.stats.scanned_entries += 1
            if time_overlap(start, end, self.qs, self.qe):
                self.stats.time_matched_records += 1
                self.candidates.add(tid)

        # One first-excluded entry authenticates the right boundary.
        if has_boundary:
            tid, start, end = self.store.get_entry(eid, k)
            self.vs.append(VerificationEntry(eid, tid, start, end))
            self.stats.boundary_witness_entries += 1

    def _emit_node(self, idx: int):
        self.stats.visited_spatial_nodes += 1
        if not self._node_overlaps(idx):
            self.stats.pruned_spatial_nodes += 1
            self._emit_pruned(idx)
            return

        t = self.tree
        m = (t.min_lon[idx], t.min_lat[idx], t.max_lon[idx], t.max_lat[idx])
        edges = t.node_edges[idx]
        if t.node_type[idx] == NODE_LEAF:
            for eid in edges:
                self._emit_edge(eid, False)
            self.tokens.append((TOKEN_NODE_LEAF, *m, len(edges)))
        else:
            self._emit_node(t.left_child[idx])
            self._emit_node(t.right_child[idx])
            # Baseline keeps no aggregate MBR for cross_edges.
            for eid in edges:
                self._emit_edge(eid, True)
            self.tokens.append((TOKEN_NODE_INTERNAL, *m, len(edges)))

    def build(self):
        self._emit_node(self.tree.root_index)
        return ServerQueryResponse(
            sorted(self.candidates), BaselineVO(self.tokens, self.vs), self.stats
        )
