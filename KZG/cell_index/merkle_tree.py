"""Fixed-depth sparse Merkle tree with compact default-hash proofs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .serialization import empty_leaf_hash, hash_pair


@dataclass(frozen=True)
class MerkleProof:
    cell_id: str
    depth: int
    default_bitmap: str
    siblings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "depth": self.depth,
            "default_bitmap": self.default_bitmap,
            "siblings": self.siblings,
        }


class SparseMerkleTree:
    """Sparse tree over all binary prefixes of one fixed depth.

    Only non-default leaves are supplied.  Missing leaves and subtrees use the
    shared default hashes, so the theoretical 2**depth leaves are never
    materialized.
    """

    def __init__(self, depth: int, leaf_hashes: Mapping[str, bytes]):
        if depth < 1 or depth > 60:
            raise ValueError("Merkle depth must be between 1 and 60")
        self.depth = depth
        self.leaf_hashes = dict(leaf_hashes)
        for cell_id in self.leaf_hashes:
            if len(cell_id) != depth or any(bit not in "01" for bit in cell_id):
                raise ValueError(f"invalid leaf prefix: {cell_id!r}")
        self.default_by_height = self._make_default_hashes(depth)
        # Materialize only prefixes that lie on a path to a non-empty leaf.
        # The old recursive implementation tried to visit both children of an
        # absent node and therefore expanded the theoretical 2**depth space.
        # This bottom-up construction keeps the tree genuinely sparse.
        self._nondefault_nodes: dict[str, bytes] = {}
        for cell_id, value in self.leaf_hashes.items():
            if value != self.default_by_height[0]:
                self._nondefault_nodes[cell_id] = value
        active = set(self.leaf_hashes)
        while active:
            parents = {prefix[:-1] for prefix in active}
            for parent in parents:
                left_prefix = parent + "0"
                right_prefix = parent + "1"
                left = self._nondefault_nodes.get(
                    left_prefix, self.default_hash_for_prefix(len(left_prefix))
                )
                right = self._nondefault_nodes.get(
                    right_prefix, self.default_hash_for_prefix(len(right_prefix))
                )
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
        """Return a node hash, using default hashes for absent subtrees."""

        if len(prefix) > self.depth or any(bit not in "01" for bit in prefix):
            raise ValueError(f"invalid tree prefix: {prefix!r}")
        return self._nondefault_nodes.get(
            prefix, self.default_hash_for_prefix(len(prefix))
        )

    @property
    def root_hash(self) -> bytes:
        return self.node_hash("")

    def proof(self, cell_id: str) -> MerkleProof:
        if len(cell_id) != self.depth or any(bit not in "01" for bit in cell_id):
            raise ValueError(f"invalid cell prefix: {cell_id!r}")
        bitmap: list[str] = []
        siblings: list[str] = []
        current_prefix = cell_id
        for step in range(self.depth):
            current_bit = current_prefix[-1]
            sibling_bit = "1" if current_bit == "0" else "0"
            sibling_prefix = current_prefix[:-1] + sibling_bit
            sibling_hash = self.node_hash(sibling_prefix)
            is_default = sibling_hash == self.default_by_height[step]
            bitmap.append("1" if is_default else "0")
            if not is_default:
                siblings.append(sibling_hash.hex())
            current_prefix = current_prefix[:-1]
        return MerkleProof(
            cell_id=cell_id,
            depth=self.depth,
            default_bitmap="".join(bitmap),
            siblings=siblings,
        )

    def verify(
        self,
        cell_id: str,
        leaf_hash: bytes,
        proof: MerkleProof | Mapping[str, object],
        expected_root: bytes,
    ) -> bool:
        if isinstance(proof, MerkleProof):
            proof_dict = proof.to_dict()
        else:
            proof_dict = proof
        proof_cell = str(proof_dict.get("cell_id", ""))
        depth = int(proof_dict.get("depth", -1))
        bitmap = str(proof_dict.get("default_bitmap", ""))
        sibling_values = proof_dict.get("siblings", [])
        if proof_cell != cell_id or depth != self.depth or len(bitmap) != self.depth:
            return False
        if not isinstance(sibling_values, list):
            return False

        current = leaf_hash
        sibling_index = 0
        for step, bit in enumerate(bitmap):
            if bit == "1":
                sibling = self.default_by_height[step]
            elif bit == "0":
                if sibling_index >= len(sibling_values):
                    return False
                try:
                    sibling = bytes.fromhex(str(sibling_values[sibling_index]))
                except ValueError:
                    return False
                if len(sibling) != 32:
                    return False
                sibling_index += 1
            else:
                return False

            path_bit = cell_id[-1 - step]
            if path_bit == "0":
                current = hash_pair(current, sibling)
            else:
                current = hash_pair(sibling, current)
        return sibling_index == len(sibling_values) and current == expected_root
