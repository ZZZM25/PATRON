from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Mapping

from kzg.geohash_index import CellEdgeRecord, GeohashCellIndex


_EMPTY_LEAF = hashlib.sha256(b"KZG_GEOHASH_EMPTY_CELL_V1").digest()
_CELL_DOMAIN = b"KZG_GEOHASH_DAILY_CELL_V1"


def hash_pair(left: bytes, right: bytes) -> bytes:
    if len(left) != 32 or len(right) != 32:
        raise ValueError("Merkle child hashes must contain 32 bytes")
    return hashlib.sha256(left + right).digest()


def empty_leaf_hash() -> bytes:
    return _EMPTY_LEAF


def _prefix_bytes(cell_id: str, bits: int) -> bytes:
    if len(cell_id) != bits or any(bit not in "01" for bit in cell_id):
        raise ValueError(f"invalid {bits}-bit cell prefix: {cell_id!r}")
    return int(cell_id, 2).to_bytes((bits + 7) // 8, "big")


def _serialize_edge(record: CellEdgeRecord, commitment: bytes) -> bytes:
    if len(commitment) != 48:
        raise ValueError("edge KZG commitment must contain 48 bytes")
    return struct.pack(
        ">QQQQQQqqqq",
        record.edge_id,
        record.start_node,
        record.end_node,
        record.original_edge_id,
        record.segment_index,
        record.segment_count,
        record.start_lon_scaled,
        record.start_lat_scaled,
        record.end_lon_scaled,
        record.end_lat_scaled,
    ) + commitment


def daily_cell_leaf_hash(
    day: str,
    cell_id: str,
    records: list[CellEdgeRecord],
    commitment_by_edge: Mapping[int, bytes],
    bits: int,
) -> bytes:
    """Bind one complete spatial cell to its road edges and daily KZG commitments."""
    if not records:
        return empty_leaf_hash()
    ordered = sorted(records, key=lambda record: record.edge_id)
    payload = bytearray()
    payload.extend(_CELL_DOMAIN)
    payload.extend(struct.pack(">H", bits))
    payload.extend(_prefix_bytes(cell_id, bits))
    day_bytes = day.encode("ascii")
    payload.extend(struct.pack(">H", len(day_bytes)))
    payload.extend(day_bytes)
    payload.extend(struct.pack(">I", len(ordered)))
    for record in ordered:
        try:
            commitment = commitment_by_edge[record.edge_id]
        except KeyError as exc:
            raise KeyError(f"missing KZG commitment for edge {record.edge_id}") from exc
        payload.extend(_serialize_edge(record, commitment))
    return hashlib.sha256(payload).digest()


def build_daily_leaf_hashes(
    day: str,
    cell_index: GeohashCellIndex,
    commitment_by_edge: Mapping[int, bytes],
) -> dict[str, bytes]:
    """Build only non-empty spatial leaves; empty cells remain implicit."""
    index_edges = set(cell_index.edge_to_cell)
    commitment_edges = set(int(edge_id) for edge_id in commitment_by_edge)
    if index_edges != commitment_edges:
        missing = sorted(index_edges - commitment_edges)
        extra = sorted(commitment_edges - index_edges)
        raise ValueError(
            f"cell-index/commitment edge set mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    return {
        cell_id: daily_cell_leaf_hash(day, cell_id, records, commitment_by_edge, cell_index.bits)
        for cell_id, records in cell_index.cells.items()
    }


@dataclass(frozen=True)
class SparseMerkleMultiProof:
    depth: int
    # Only non-default external sibling subtrees are transmitted.  Missing
    # sibling prefixes are deterministically interpreted as default subtrees.
    nodes: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "nodes": [
                {"prefix": prefix, "hash": value}
                for prefix, value in self.nodes
            ],
        }

    @classmethod
    def from_mapping(cls, value: Mapping) -> "SparseMerkleMultiProof":
        depth = int(value.get("depth", -1))
        raw_nodes = value.get("nodes", [])
        if not isinstance(raw_nodes, list):
            raise ValueError("multiproof nodes must be a list")
        nodes: list[tuple[str, str]] = []
        for item in raw_nodes:
            if not isinstance(item, Mapping):
                raise ValueError("malformed multiproof node")
            nodes.append((str(item.get("prefix", "")), str(item.get("hash", ""))))
        return cls(depth, tuple(nodes))


class SparseMerkleTree:
    """Fixed-depth sparse Merkle tree keyed directly by Geohash prefix."""

    def __init__(self, depth: int, leaf_hashes: Mapping[str, bytes]):
        if depth < 1 or depth > 60:
            raise ValueError("Merkle depth must be between 1 and 60")
        self.depth = int(depth)
        self.default_by_height = self._make_default_hashes(self.depth)
        self._nondefault_nodes: dict[str, bytes] = {}
        active: set[str] = set()
        for cell_id, value in leaf_hashes.items():
            if len(cell_id) != self.depth or any(bit not in "01" for bit in cell_id):
                raise ValueError(f"invalid leaf prefix: {cell_id!r}")
            if len(value) != 32:
                raise ValueError("leaf hash must contain 32 bytes")
            active.add(cell_id)
            if value != self.default_by_height[0]:
                self._nondefault_nodes[cell_id] = value

        while active:
            parents = {prefix[:-1] for prefix in active}
            for parent in parents:
                left_prefix = parent + "0"
                right_prefix = parent + "1"
                left = self.node_hash(left_prefix)
                right = self.node_hash(right_prefix)
                value = hash_pair(left, right)
                if value != self.default_hash_for_prefix(len(parent)):
                    self._nondefault_nodes[parent] = value
            if "" in parents:
                break
            active = parents

    @staticmethod
    def _make_default_hashes(depth: int) -> list[bytes]:
        values = [empty_leaf_hash()]
        for _ in range(depth):
            values.append(hash_pair(values[-1], values[-1]))
        return values

    def default_hash_for_prefix(self, prefix_length: int) -> bytes:
        if not 0 <= prefix_length <= self.depth:
            raise ValueError("prefix length outside tree")
        return self.default_by_height[self.depth - prefix_length]

    def node_hash(self, prefix: str) -> bytes:
        if len(prefix) > self.depth or any(bit not in "01" for bit in prefix):
            raise ValueError(f"invalid tree prefix: {prefix!r}")
        return self._nondefault_nodes.get(prefix, self.default_hash_for_prefix(len(prefix)))

    @property
    def root_hash(self) -> bytes:
        return self.node_hash("")

    def multiproof(self, cell_ids: list[str]) -> SparseMerkleMultiProof:
        targets = sorted(set(cell_ids))
        if not targets:
            raise ValueError("at least one cell is required for a multiproof")
        for cell_id in targets:
            if len(cell_id) != self.depth or any(bit not in "01" for bit in cell_id):
                raise ValueError(f"invalid cell prefix: {cell_id!r}")

        current = {cell_id: self.node_hash(cell_id) for cell_id in targets}
        proof_nodes: dict[str, bytes] = {}

        for _ in range(self.depth):
            next_level: dict[str, bytes] = {}
            processed: set[str] = set()
            for prefix in sorted(current):
                if prefix in processed:
                    continue
                sibling = prefix[:-1] + ("1" if prefix[-1] == "0" else "0")
                parent = prefix[:-1]
                if sibling in current:
                    sibling_hash = current[sibling]
                    processed.add(sibling)
                else:
                    sibling_hash = self.node_hash(sibling)
                    if sibling_hash != self.default_hash_for_prefix(len(sibling)):
                        proof_nodes[sibling] = sibling_hash
                current_hash = current[prefix]
                if prefix[-1] == "0":
                    parent_hash = hash_pair(current_hash, sibling_hash)
                else:
                    parent_hash = hash_pair(sibling_hash, current_hash)
                existing = next_level.get(parent)
                if existing is not None and existing != parent_hash:
                    raise RuntimeError("inconsistent multiproof parent reconstruction")
                next_level[parent] = parent_hash
                processed.add(prefix)
            current = next_level

        if current.get("") != self.root_hash:
            raise RuntimeError("generated multiproof does not reconstruct the tree root")
        return SparseMerkleMultiProof(
            depth=self.depth,
            nodes=tuple((prefix, value.hex()) for prefix, value in sorted(proof_nodes.items())),
        )

    def verify_multiproof(
        self,
        leaf_hashes: Mapping[str, bytes],
        proof: SparseMerkleMultiProof | Mapping,
        expected_root: bytes,
    ) -> bool:
        """Verify queried leaves together, reusing all common upward paths."""
        try:
            if isinstance(proof, SparseMerkleMultiProof):
                parsed = proof
            else:
                parsed = SparseMerkleMultiProof.from_mapping(proof)
            if parsed.depth != self.depth or len(expected_root) != 32 or not leaf_hashes:
                return False

            current: dict[str, bytes] = {}
            for prefix, value in leaf_hashes.items():
                if len(prefix) != self.depth or any(bit not in "01" for bit in prefix):
                    return False
                if len(value) != 32 or prefix in current:
                    return False
                current[prefix] = value

            proof_nodes: dict[str, bytes] = {}
            for prefix, value_hex in parsed.nodes:
                if not prefix or len(prefix) > self.depth or any(bit not in "01" for bit in prefix):
                    return False
                if prefix in proof_nodes:
                    return False
                value = bytes.fromhex(value_hex)
                if len(value) != 32:
                    return False
                # Sending a default node is redundant and rejected so the proof
                # stays canonical/compact.
                if value == self.default_hash_for_prefix(len(prefix)):
                    return False
                proof_nodes[prefix] = value

            consumed: set[str] = set()
            for _ in range(self.depth):
                next_level: dict[str, bytes] = {}
                processed: set[str] = set()
                for prefix in sorted(current):
                    if prefix in processed:
                        continue
                    sibling = prefix[:-1] + ("1" if prefix[-1] == "0" else "0")
                    parent = prefix[:-1]
                    if sibling in current:
                        sibling_hash = current[sibling]
                        processed.add(sibling)
                    elif sibling in proof_nodes:
                        sibling_hash = proof_nodes[sibling]
                        consumed.add(sibling)
                    else:
                        sibling_hash = self.default_hash_for_prefix(len(sibling))
                    current_hash = current[prefix]
                    parent_hash = (
                        hash_pair(current_hash, sibling_hash)
                        if prefix[-1] == "0"
                        else hash_pair(sibling_hash, current_hash)
                    )
                    existing = next_level.get(parent)
                    if existing is not None and existing != parent_hash:
                        return False
                    next_level[parent] = parent_hash
                    processed.add(prefix)
                current = next_level

            return current.get("") == expected_root and consumed == set(proof_nodes)
        except (ValueError, TypeError):
            return False


def query_cell_payloads(
    day: str,
    cell_index: GeohashCellIndex,
    cell_ids: list[str],
    commitment_by_edge: Mapping[int, bytes],
) -> tuple[list[dict], dict[str, bytes]]:
    """Create authenticated query payloads and their leaf hashes."""
    cells: list[dict] = []
    leaf_hashes: dict[str, bytes] = {}
    for cell_id in cell_ids:
        records = cell_index.cell_records(cell_id)
        lon_index, lat_index = cell_index.grid.indices_from_prefix(cell_id)
        edges = []
        for record in records:
            commitment = commitment_by_edge.get(record.edge_id)
            if commitment is None:
                raise KeyError(f"missing KZG commitment for edge {record.edge_id}")
            item = record.to_dict()
            item["kzg_commitment"] = commitment.hex()
            edges.append(item)
        leaf = daily_cell_leaf_hash(day, cell_id, records, commitment_by_edge, cell_index.bits)
        leaf_hashes[cell_id] = leaf
        cells.append({
            "cell_id": cell_id,
            "lon_index": lon_index,
            "lat_index": lat_index,
            "edge_count": len(edges),
            "edges": edges,
        })
    return cells, leaf_hashes
