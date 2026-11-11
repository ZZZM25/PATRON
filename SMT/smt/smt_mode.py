"""SMT-augmented version of the original sorted-entry spatial baseline."""
from __future__ import annotations

from dataclasses import dataclass

from .authenticated_index import build_edge_commitments, authenticate_spatial_tree
from .crypto import (
    ENTRY_LIST_TAIL, entry_list_prepend, hash_entry, hash_smt_edge, time_overlap,
)
from .edge_smt import EdgeSMT, key_for_id
from .entry_store import PackedEdgeEntryStore
from .models import VerificationEntry
from .server import BaselineServerQuery
from .spatial_tree import BinarySpatialTreeBuilder
from .vo import (
    BaselineVO, ServerQueryResponse, QueryStats,
    TOKEN_EDGE_OPAQUE_SMT, TOKEN_EDGE_PREFIX_SMT,
)

PER_EDGE_SMT_MODE = "per-edge-smt-spatial-root-v1"


def is_per_edge_smt_index(index) -> bool:
    """Accept both the original in-memory and disk-backed representations."""
    return (
        getattr(index, "proof_mode", None) == PER_EDGE_SMT_MODE
        and hasattr(index, "road")
        and hasattr(index, "store")
        and hasattr(index, "tree")
        and hasattr(index, "edge_smt_roots")
    )


@dataclass
class SmtBaselineIndex:
    road: object
    store: PackedEdgeEntryStore
    tree: object
    edge_entry_roots: list[bytes]
    edge_hashes: list[bytes]
    edge_suffix_states: list[bytes]  # Empty in this naive SMT mode: no cached suffixes.
    root_hash: bytes
    theta: int
    trajectory_files: tuple[str, ...]
    edge_smts: list[EdgeSMT]
    edge_smt_roots: list[bytes]
    proof_mode: str

    def suffix_state(self, eid: int, position: int) -> bytes:
        count = self.store.entry_counts[eid]
        if not 0 <= position <= count:
            raise IndexError(position)
        # Rehash the entire trailing portion of the time list for EVERY query.
        # This reconstructs state_position without storing any suffix states.
        state = ENTRY_LIST_TAIL
        for i in range(count - 1, position - 1, -1):
            key, start, end = self.store.get_entry(eid, i)
            state = entry_list_prepend(hash_entry(key, start, end), state)
        return state


def build_smt_baseline(road, original_store, theta: int, trajectory_files=()) -> SmtBaselineIndex:
    """Data owner builds trusted chain and SMT from the SAME original entries.

    This baseline purposefully uses a separate (unoptimized) SMT on every edge.
    Original IDs are NOT kept in the new per-edge time-sorted list.
    """
    store = PackedEdgeEntryStore(road.max_eid)
    smts: list[EdgeSMT] = []
    empty = EdgeSMT.build([])
    for eid in range(road.max_eid + 1):
        if not road.edge_exists(eid):
            smts.append(empty)
            continue
        by_key = set()
        for original_id, start, end in original_store.iter_entries(eid):
            original_id = bytes(original_id)
            k = key_for_id(original_id)
            by_key.add(original_id)
            store.insert_by_start(eid, k, start, end)
        smts.append(EdgeSMT.build(by_key) if by_key else empty)
    store.validate_start_order()
    # Build the same authenticated entry roots, but do not retain any
    # intermediate suffix chain hashes.  Query-time recomputation is mandatory.
    edge_roots, _, suffix_states = build_edge_commitments(
        road, store, cache_suffix_states=False
    )
    smt_roots = [smt.root_hash for smt in smts]
    hashes = [b"" for _ in smts]
    for eid in road.valid_edge_ids():
        hashes[eid] = hash_smt_edge(eid, store.entry_counts[eid], edge_roots[eid], smt_roots[eid])
    tree = BinarySpatialTreeBuilder(
        road, int(theta), edge_trajectory_counts=store.unique_trajectory_counts()
    ).build()
    spatial_root = authenticate_spatial_tree(tree, hashes)
    return SmtBaselineIndex(
        road, store, tree, edge_roots, hashes, suffix_states,
        spatial_root, tree.theta, tuple(map(str, trajectory_files)), smts, smt_roots,
        PER_EDGE_SMT_MODE,
    )


class SmtBaselineServerQuery(BaselineServerQuery):
    def __init__(self, index, *query_args):
        if getattr(index, "proof_mode", None) != PER_EDGE_SMT_MODE:
            raise ValueError(
                "obsolete/incompatible SMT index; rebuild it with one independent SMT per edge"
            )
        if hasattr(index, "begin_query"):
            # Disk-backed mode must discover target edges through the ordinary
            # spatial traversal; no prior query is allowed to preload them.
            index.begin_query()
        super().__init__(index, *query_args)
        self.required_pairs: set[tuple[int, bytes]] = set()

    def _emit_edge(self, eid: int, is_cross: bool):
        if is_cross:
            self.stats.checked_cross_edges += 1
        else:
            self.stats.checked_leaf_edges += 1
        count = self.store.entry_counts[eid]
        root = self.index.edge_smt_roots[eid]
        if not self._edge_spatial_match(eid):
            self.tokens.append((TOKEN_EDGE_OPAQUE_SMT, eid, count, self.index.edge_entry_roots[eid], root))
            return
        self.stats.spatial_matched_edges += 1
        k = self.store.linear_upper_bound_start(eid, self.qe)
        has_boundary = int(k < count)
        suffix_pos = k + has_boundary
        suffix = self.index.suffix_state(eid, suffix_pos)
        self.tokens.append((TOKEN_EDGE_PREFIX_SMT, eid, count, k, has_boundary, suffix, root))
        for key, start, end in self.store.iter_prefix(eid, k):
            self.vs.append(VerificationEntry(eid, key, start, end))
            self.stats.scanned_entries += 1
            if time_overlap(start, end, self.qs, self.qe):
                self.stats.time_matched_records += 1
                self.required_pairs.add((eid, key))
        if has_boundary:
            key, start, end = self.store.get_entry(eid, k)
            self.vs.append(VerificationEntry(eid, key, start, end))
            self.stats.boundary_witness_entries += 1

    def build(self) -> ServerQueryResponse:
        self._emit_node(self.tree.root_index)
        # One independent, complete 256-sibling membership proof for every
        # matched (edge, K).  Each edge SMT root is already committed through
        # hash_smt_edge() and the authenticated spatial tree; there is no
        # separate cross-edge SMT routing tree or second root.
        proof_pairs = sorted(self.required_pairs)
        if hasattr(self.index, "prove_pairs"):
            # Disk-backed Cell50 indexes group keys by edge and load/cache only
            # those edge SMTs.  Proofs are still freshly generated on every
            # query; warmup never caches completed proofs.
            proofs = self.index.prove_pairs(proof_pairs)
        else:
            proofs = [
                self.index.edge_smts[eid].prove(eid, key)
                for eid, key in proof_pairs
            ]
        if hasattr(self.index, "query_storage_times"):
            (
                self.stats.edge_data_load_time_s,
                self.stats.smt_materialization_time_s,
            ) = self.index.query_storage_times()
        ids = sorted({proof.trajectory_id for proof in proofs})
        return ServerQueryResponse(ids, BaselineVO(self.tokens, self.vs, proofs), self.stats)


class CellCoverSmtBaselineServerQuery(SmtBaselineServerQuery):
    """Per-edge SMT query over the minimum covering Cell50 cell set."""

    def __init__(self, index, cell_policy, *query_args):
        if len(query_args) != 6:
            raise TypeError("expected min_lon,min_lat,max_lon,max_lat,start,end")
        self.cell_policy = cell_policy
        self.original_query_bounds = tuple(float(value) for value in query_args[:4])
        self.cell_cover = cell_policy.cover_rectangle(*query_args[:4])
        effective_query = (
            *self.cell_cover.effective_bounds,
            query_args[4],
            query_args[5],
        )
        super().__init__(index, *effective_query)

    def _edge_spatial_match(self, eid: int) -> bool:
        return self.cell_policy.edge_matches(eid, self.cell_cover)
