from __future__ import annotations

from dataclasses import dataclass

from .crypto import (
    ENTRY_LIST_TAIL, EDGE_GROUP_INIT,
    hash_entry, entry_list_prepend, hash_edge, edge_group_append,
    hash_leaf, hash_internal,
)
from .spatial_tree import NODE_LEAF, NODE_INTERNAL


@dataclass(slots=True)
class BaselineIndex:
    road: object
    store: object
    tree: object
    edge_entry_roots: list[bytes]
    edge_hashes: list[bytes]
    # For edge e with n entries, packed states contains state_0..state_n,
    # each 32 bytes, where state_n = ENTRY_LIST_TAIL and
    # state_i = H(entry_i || state_{i+1}).
    edge_suffix_states: list[bytes]
    root_hash: bytes
    theta: int
    trajectory_files: tuple[str, ...]

    def suffix_state(self, eid: int, position: int) -> bytes:
        count = self.store.entry_counts[eid]
        if not 0 <= position <= count:
            raise IndexError(position)
        raw = self.edge_suffix_states[eid]
        p = position * 32
        return raw[p:p + 32]


def build_edge_commitments(road, store, *, cache_suffix_states: bool = True):
    roots = [ENTRY_LIST_TAIL for _ in range(road.max_eid + 1)]
    hashes = [b"" for _ in range(road.max_eid + 1)]
    suffix_states = [b"" for _ in range(road.max_eid + 1)]

    for eid in range(road.max_eid + 1):
        if not road.edge_exists(eid):
            continue

        count = store.entry_counts[eid]
        # An SMT-mode build deliberately does NOT materialize the n+1 cached
        # reverse-chain suffix states.  The original baseline still does.
        packed = bytearray((count + 1) * 32) if cache_suffix_states else None
        if packed is not None:
            packed[count * 32:(count + 1) * 32] = ENTRY_LIST_TAIL
        state = ENTRY_LIST_TAIL

        for i in range(count - 1, -1, -1):
            tid, start, end = store.get_entry(eid, i)
            state = entry_list_prepend(hash_entry(tid, start, end), state)
            if packed is not None:
                p = i * 32
                packed[p:p + 32] = state

        root = state if count else ENTRY_LIST_TAIL
        roots[eid] = root
        hashes[eid] = hash_edge(eid, count, root)
        suffix_states[eid] = bytes(packed) if packed is not None else b""

    return roots, hashes, suffix_states


def _edge_group_root(edge_ids, edge_hashes):
    state = EDGE_GROUP_INIT
    for eid in edge_ids:
        state = edge_group_append(state, edge_hashes[eid])
    return state


def authenticate_spatial_tree(tree, edge_hashes):
    def visit(idx: int) -> bytes:
        edges = tree.node_edges[idx]
        group = _edge_group_root(edges, edge_hashes)
        tree.edge_group_root[idx] = group
        m = (tree.min_lon[idx], tree.min_lat[idx], tree.max_lon[idx], tree.max_lat[idx])
        if tree.node_type[idx] == NODE_LEAF:
            h = hash_leaf(*m, len(edges), group)
        else:
            left = visit(tree.left_child[idx])
            right = visit(tree.right_child[idx])
            h = hash_internal(*m, left, right, len(edges), group)
        tree.node_hash[idx] = h
        return h

    return visit(tree.root_index)
