"""Per-edge 256-bit sparse Merkle trees for the deliberately unoptimized baseline.

Key = SHA256(existing 32-byte trajectory ID); value = that original ID.
The tree stores only nonempty compressed branch nodes but emits a FULL 256-sibling
proof for every (edge, key) independently: no compressed/multi proofs or batching.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


def h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def key_for_id(trajectory_id: bytes) -> bytes:
    if not isinstance(trajectory_id, bytes) or len(trajectory_id) != 32:
        raise ValueError("first SHA-256 trajectory ID must be 32 bytes")
    return h(trajectory_id)


def leaf_hash(key: bytes, trajectory_id: bytes) -> bytes:
    if len(key) != 32 or len(trajectory_id) != 32:
        raise ValueError("invalid SMT leaf data")
    return h(b"EDGE-SMT-LEAF-v1" + key + trajectory_id)


def node_hash(left: bytes, right: bytes) -> bytes:
    return h(b"EDGE-SMT-NODE-v1" + left + right)


def default_hashes() -> tuple[bytes, ...]:
    defaults = [b""] * 257
    defaults[256] = h(b"EDGE-SMT-EMPTY-v1")
    for depth in range(255, -1, -1):
        defaults[depth] = node_hash(defaults[depth + 1], defaults[depth + 1])
    return tuple(defaults)


def bit(key: bytes, depth: int) -> int:
    return (key[depth // 8] >> (7 - depth % 8)) & 1


@dataclass
class _Node:
    depth: int
    key: bytes
    digest: bytes
    left: _Node | None = None
    right: _Node | None = None


@dataclass(frozen=True)
class EdgeSmtProof:
    eid: int
    key: bytes
    trajectory_id: bytes
    siblings: tuple[bytes, ...]  # indexed 0(root) -> 255(leaf)


@dataclass
class EdgeSMT:
    values: dict[bytes, bytes]
    root: _Node | None
    root_hash: bytes
    defaults: tuple[bytes, ...]

    def __getstate__(self):
        # The 257 default hashes are protocol constants, identical for every
        # edge.  Regenerate them on load instead of storing another copy in
        # every independently persisted edge tree.
        return self.values, self.root, self.root_hash

    def __setstate__(self, state):
        # Accept both the compact state above and legacy dataclass pickles so
        # indexes produced before this optimization remain readable.
        if isinstance(state, dict):
            self.values = state["values"]
            self.root = state["root"]
            self.root_hash = state["root_hash"]
            self.defaults = state.get("defaults") or default_hashes()
            return
        self.values, self.root, self.root_hash = state
        self.defaults = default_hashes()

    @classmethod
    def build(cls, trajectory_ids) -> EdgeSMT:
        data: dict[bytes, bytes] = {}
        for tid in trajectory_ids:
            tid = bytes(tid)
            k = key_for_id(tid)
            if k in data and data[k] != tid:
                raise ValueError("SHA-256 key collision")
            data[k] = tid
        defaults = default_hashes()
        records = sorted(data.items())

        def lift(node: _Node, target_depth: int) -> bytes:
            if target_depth > node.depth:
                raise ValueError("invalid SMT lift")
            digest = node.digest
            for d in range(node.depth - 1, target_depth - 1, -1):
                empty = defaults[d + 1]
                digest = node_hash(digest, empty) if bit(node.key, d) == 0 else node_hash(empty, digest)
            return digest

        def subtree(rows: list[tuple[bytes, bytes]]) -> _Node:
            if len(rows) == 1:
                k, tid = rows[0]
                return _Node(256, k, leaf_hash(k, tid))
            first, last = rows[0][0], rows[-1][0]
            diff = int.from_bytes(first, 'big') ^ int.from_bytes(last, 'big')
            depth = 256 - diff.bit_length()
            if depth == 256:
                raise ValueError("duplicate key")
            midpoint = 0
            while midpoint < len(rows) and bit(rows[midpoint][0], depth) == 0:
                midpoint += 1
            if midpoint == 0 or midpoint == len(rows):
                raise RuntimeError("invalid SMT split")
            left = subtree(rows[:midpoint])
            right = subtree(rows[midpoint:])
            digest = node_hash(lift(left, depth + 1), lift(right, depth + 1))
            return _Node(depth, first, digest, left, right)

        root = subtree(records) if records else None
        root_hash = lift(root, 0) if root else defaults[0]
        return cls(data, root, root_hash, defaults)

    def _lift(self, node: _Node, target_depth: int) -> bytes:
        digest = node.digest
        for d in range(node.depth - 1, target_depth - 1, -1):
            empty = self.defaults[d + 1]
            digest = node_hash(digest, empty) if bit(node.key, d) == 0 else node_hash(empty, digest)
        return digest

    def insert(self, trajectory_id: bytes) -> bool:
        """Incrementally insert one leaf and update only its Patricia path.

        Returns ``True`` for a new leaf and ``False`` when the exact existing
        trajectory is already present.  A SHA-256 key collision is rejected.
        """
        tid = bytes(trajectory_id)
        key = key_for_id(tid)
        existing = self.values.get(key)
        if existing is not None:
            if existing != tid:
                raise ValueError("SHA-256 key collision")
            return False

        new_leaf = _Node(256, key, leaf_hash(key, tid))
        if self.root is None:
            self.root = new_leaf
            self.values[key] = tid
            self.root_hash = self._lift(new_leaf, 0)
            return True

        def common_prefix(a: bytes, b: bytes) -> int:
            diff = int.from_bytes(a, "big") ^ int.from_bytes(b, "big")
            return 256 if diff == 0 else 256 - diff.bit_length()

        def add(node: _Node) -> _Node:
            shared = common_prefix(key, node.key)
            if shared < node.depth:
                if bit(key, shared) == 0:
                    left, right = new_leaf, node
                else:
                    left, right = node, new_leaf
                digest = node_hash(
                    self._lift(left, shared + 1),
                    self._lift(right, shared + 1),
                )
                return _Node(shared, left.key, digest, left, right)

            if node.depth == 256:
                raise RuntimeError("unexpected duplicate SMT key")
            direction = bit(key, node.depth)
            if direction == 0:
                if node.left is None:
                    raise RuntimeError("SMT structural inconsistency")
                node.left = add(node.left)
            else:
                if node.right is None:
                    raise RuntimeError("SMT structural inconsistency")
                node.right = add(node.right)
            if node.left is None or node.right is None:
                raise RuntimeError("SMT structural inconsistency")
            node.key = node.left.key
            node.digest = node_hash(
                self._lift(node.left, node.depth + 1),
                self._lift(node.right, node.depth + 1),
            )
            return node

        self.root = add(self.root)
        self.values[key] = tid
        self.root_hash = self._lift(self.root, 0)
        return True

    def prove(self, eid: int, key: bytes) -> EdgeSmtProof:
        if key not in self.values or self.root is None:
            raise KeyError("SMT key missing")
        siblings = [self.defaults[d + 1] for d in range(256)]
        node = self.root
        while node.depth < 256:
            d = node.depth
            if bit(key, d) == 0:
                child, other = node.left, node.right
            else:
                child, other = node.right, node.left
            if child is None or other is None:
                raise RuntimeError("SMT structural inconsistency")
            siblings[d] = self._lift(other, d + 1)
            node = child
        if node.key != key:
            raise RuntimeError("SMT leaf key mismatch")
        return EdgeSmtProof(eid, key, self.values[key], tuple(siblings))


def verify_proof(proof: EdgeSmtProof, root_hash: bytes) -> bool:
    if len(proof.key) != 32 or len(proof.trajectory_id) != 32 or len(proof.siblings) != 256 or len(root_hash) != 32:
        return False
    if key_for_id(proof.trajectory_id) != proof.key:
        return False
    digest = leaf_hash(proof.key, proof.trajectory_id)
    for d in range(255, -1, -1):
        sibling = proof.siblings[d]
        if len(sibling) != 32:
            return False
        digest = node_hash(digest, sibling) if bit(proof.key, d) == 0 else node_hash(sibling, digest)
    return digest == root_hash
