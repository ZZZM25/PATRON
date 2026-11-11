from __future__ import annotations

from dataclasses import dataclass

from .crypto import (
    ENTRY_LIST_TAIL, EDGE_GROUP_INIT,
    hash_entry, entry_list_prepend, hash_edge, hash_smt_edge, edge_group_append,
    hash_leaf, hash_internal, time_overlap,
)
from .edge_smt import verify_proof
from .query_utils import normalize_rectangle, rectangle_overlap, edge_intersects_query
from .vo import (
    TOKEN_NODE_PRUNED_LEAF, TOKEN_NODE_PRUNED_INTERNAL, TOKEN_NODE_LEAF, TOKEN_NODE_INTERNAL,
    TOKEN_EDGE_OPAQUE, TOKEN_EDGE_PREFIX, TOKEN_EDGE_OPAQUE_SMT, TOKEN_EDGE_PREFIX_SMT,
)


@dataclass(slots=True)
class VerificationReport:
    success: bool
    reconstructed_root: bytes
    trusted_root: bytes
    candidate_ids: list[bytes]
    token_count: int
    verification_entry_count: int
    consumed_verification_entries: int


@dataclass(slots=True)
class _EdgeItem:
    eid: int
    edge_hash: bytes


@dataclass(slots=True)
class _NodeItem:
    node_hash: bytes


class VerificationError(RuntimeError):
    pass


def verify_vo(
    vo, road, trusted_root: bytes,
    query_min_lon, query_min_lat, query_max_lon, query_max_lat,
    query_start: int, query_end: int, claimed_candidate_ids=None,
    cell_policy=None,
):
    if cell_policy is None:
        qminx, qminy, qmaxx, qmaxy = normalize_rectangle(
            query_min_lon, query_min_lat, query_max_lon, query_max_lat
        )
        edge_matches = lambda eid: edge_intersects_query(
            eid, road, qminx, qminy, qmaxx, qmaxy
        )
    else:
        cell_cover = cell_policy.cover_rectangle(
            query_min_lon, query_min_lat, query_max_lon, query_max_lat
        )
        qminx, qminy, qmaxx, qmaxy = cell_cover.effective_bounds
        edge_matches = lambda eid: cell_policy.edge_matches(eid, cell_cover)
    if query_start > query_end:
        raise VerificationError("query_start > query_end")

    stack = []
    vs_index = 0
    candidates = set()
    matched_pairs = set()
    edge_smt_roots = {}
    smt_mode = vo.smt_proofs is not None

    def validate_mbr(m):
        if m[0] > m[2] or m[1] > m[3]:
            raise VerificationError("invalid MBR")

    def pop_edges(n):
        if n < 0 or len(stack) < n:
            raise VerificationError("edge stack underflow")
        items = []
        for _ in range(n):
            x = stack.pop()
            if not isinstance(x, _EdgeItem):
                raise VerificationError("expected edge item")
            items.append(x)
        items.reverse()
        eids = [x.eid for x in items]
        if eids != sorted(eids) or len(eids) != len(set(eids)):
            raise VerificationError("edge list is not canonical sorted unique order")
        state = EDGE_GROUP_INIT
        for x in items:
            state = edge_group_append(state, x.edge_hash)
        return state

    for token in vo.tokens:
        code = token[0]

        if code in (TOKEN_EDGE_OPAQUE, TOKEN_EDGE_OPAQUE_SMT):
            if (code == TOKEN_EDGE_OPAQUE_SMT) != smt_mode:
                raise VerificationError("mixed SMT / legacy edge tokens")
            if smt_mode:
                _, eid, count, entry_root, smt_root = token
            else:
                _, eid, count, entry_root = token
            if not road.edge_exists(eid):
                raise VerificationError(f"invalid eid {eid}")
            if edge_matches(eid):
                raise VerificationError("spatially matching edge was hidden as opaque")
            if smt_mode:
                stack.append(_EdgeItem(eid, hash_smt_edge(eid, count, entry_root, smt_root)))
            else:
                stack.append(_EdgeItem(eid, hash_edge(eid, count, entry_root)))

        elif code in (TOKEN_EDGE_PREFIX, TOKEN_EDGE_PREFIX_SMT):
            if (code == TOKEN_EDGE_PREFIX_SMT) != smt_mode:
                raise VerificationError("mixed SMT / legacy edge tokens")
            if smt_mode:
                _, eid, total_count, prefix_count, has_boundary, suffix_state, smt_root = token
                if eid in edge_smt_roots:
                    raise VerificationError("duplicate SMT edge")
                edge_smt_roots[eid] = smt_root
            else:
                _, eid, total_count, prefix_count, has_boundary, suffix_state = token
            if not road.edge_exists(eid):
                raise VerificationError(f"invalid eid {eid}")
            if not edge_matches(eid):
                raise VerificationError("nonmatching edge unnecessarily expanded")
            if not (0 <= prefix_count <= total_count):
                raise VerificationError("invalid prefix count")
            if has_boundary not in (0, 1):
                raise VerificationError("invalid boundary flag")
            if has_boundary != (1 if prefix_count < total_count else 0):
                raise VerificationError("boundary flag inconsistent with counts")
            if not isinstance(suffix_state, (bytes, bytearray)) or len(suffix_state) != 32:
                raise VerificationError("bad suffix state")

            needed = prefix_count + has_boundary
            if vs_index + needed > len(vo.verification_set):
                raise VerificationError("verification set truncated")

            disclosed = []
            previous_start = None

            # Expanded prefix: every Entry is authenticated and independently
            # time-tested, even when it is a non-result because end < qs.
            for _ in range(prefix_count):
                ent = vo.verification_set[vs_index]
                vs_index += 1
                if ent.eid != eid:
                    raise VerificationError("verification entry eid mismatch")
                if len(ent.trajectory_id) != 32:
                    raise VerificationError("bad trajectory id")
                if ent.start < 0 or ent.end < 0 or ent.start > 0xFFFFFFFF or ent.end > 0xFFFFFFFF:
                    raise VerificationError("bad timestamp")
                if ent.start > query_end:
                    raise VerificationError("prefix contains start > query_end")
                if previous_start is not None and ent.start < previous_start:
                    raise VerificationError("disclosed list is not sorted by start")
                previous_start = ent.start
                disclosed.append(ent)
                if time_overlap(ent.start, ent.end, query_start, query_end):
                    if smt_mode:
                        matched_pairs.add((eid, ent.trajectory_id))
                    else:
                        candidates.add(ent.trajectory_id)

            # First excluded Entry proves the right boundary start > query_end.
            if has_boundary:
                boundary = vo.verification_set[vs_index]
                vs_index += 1
                if boundary.eid != eid:
                    raise VerificationError("boundary eid mismatch")
                if len(boundary.trajectory_id) != 32:
                    raise VerificationError("bad boundary trajectory id")
                if boundary.start <= query_end:
                    raise VerificationError("invalid right boundary: start <= query_end")
                if previous_start is not None and boundary.start < previous_start:
                    raise VerificationError("boundary violates start ordering")
                disclosed.append(boundary)
            else:
                if suffix_state != ENTRY_LIST_TAIL:
                    raise VerificationError("complete list must end at list tail")

            # Reconstruct the complete sorted-list root from compact suffix state,
            # optional boundary witness, and ALL disclosed prefix entries.
            state = bytes(suffix_state)
            for ent in reversed(disclosed):
                state = entry_list_prepend(
                    hash_entry(ent.trajectory_id, ent.start, ent.end), state
                )
            edge_hash = (hash_smt_edge(eid, total_count, state, smt_root) if smt_mode
                         else hash_edge(eid, total_count, state))
            stack.append(_EdgeItem(eid, edge_hash))

        elif code == TOKEN_NODE_PRUNED_LEAF:
            _, a, b, c, d, count, group_root = token
            m = (a, b, c, d)
            validate_mbr(m)
            if rectangle_overlap(*m, qminx, qminy, qmaxx, qmaxy):
                raise VerificationError("overlapping leaf was spatially pruned")
            stack.append(_NodeItem(hash_leaf(*m, count, group_root)))

        elif code == TOKEN_NODE_PRUNED_INTERNAL:
            _, a, b, c, d, left_hash, right_hash, cross_count, cross_root = token
            m = (a, b, c, d)
            validate_mbr(m)
            if rectangle_overlap(*m, qminx, qminy, qmaxx, qmaxy):
                raise VerificationError("overlapping internal node was spatially pruned")
            stack.append(_NodeItem(hash_internal(*m, left_hash, right_hash, cross_count, cross_root)))

        elif code == TOKEN_NODE_LEAF:
            _, a, b, c, d, edge_count = token
            m = (a, b, c, d)
            validate_mbr(m)
            if not rectangle_overlap(*m, qminx, qminy, qmaxx, qmaxy):
                raise VerificationError("nonoverlapping leaf should have been pruned")
            group = pop_edges(edge_count)
            stack.append(_NodeItem(hash_leaf(*m, edge_count, group)))

        elif code == TOKEN_NODE_INTERNAL:
            _, a, b, c, d, cross_count = token
            m = (a, b, c, d)
            validate_mbr(m)
            if not rectangle_overlap(*m, qminx, qminy, qmaxx, qmaxy):
                raise VerificationError("nonoverlapping internal should have been pruned")
            cross_root = pop_edges(cross_count)
            if len(stack) < 2:
                raise VerificationError("node stack underflow")
            right = stack.pop()
            left = stack.pop()
            if not isinstance(left, _NodeItem) or not isinstance(right, _NodeItem):
                raise VerificationError("expected child nodes")
            stack.append(_NodeItem(hash_internal(*m, left.node_hash, right.node_hash, cross_count, cross_root)))

        else:
            raise VerificationError(f"unknown token {code}")

    if vs_index != len(vo.verification_set):
        raise VerificationError("unused verification entries")
    if len(stack) != 1 or not isinstance(stack[0], _NodeItem):
        raise VerificationError("VO did not reduce to one root")

    spatial_root = stack[0].node_hash
    root = spatial_root
    if smt_mode:
        if not isinstance(vo.smt_proofs, list):
            raise VerificationError("SMT VO must contain independent per-edge proofs")
        proof_pairs = [(p.eid, p.key) for p in vo.smt_proofs]
        if proof_pairs != sorted(proof_pairs):
            raise VerificationError("SMT proofs are not in canonical (edge,key) order")
        if len(proof_pairs) != len(set(proof_pairs)) or set(proof_pairs) != matched_pairs:
            raise VerificationError("SMT proof set does not equal matching (edge,key) pairs")
        for proof in vo.smt_proofs:
            expected_root = edge_smt_roots.get(proof.eid)
            if expected_root is None or not verify_proof(proof, expected_root):
                raise VerificationError("invalid per-edge SMT ID mapping / membership proof")
            candidates.add(proof.trajectory_id)
    if root != trusted_root:
        raise VerificationError("reconstructed root != trusted root")
    derived = sorted(candidates)
    if claimed_candidate_ids is not None and derived != sorted(claimed_candidate_ids):
        raise VerificationError("server candidate set differs from client-derived candidate set")

    return VerificationReport(
        True, root, trusted_root, derived,
        len(vo.tokens), len(vo.verification_set), vs_index,
    )
