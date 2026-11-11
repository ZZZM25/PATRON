from __future__ import annotations

import hashlib
import hmac
import pickle
import struct
from array import array
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path


PID_DOMAIN = b"MRPT-SMT-PID-v1"
KEY_DOMAIN = b"MRPT-SMT-KEY-v1"
LEAF_DOMAIN = b"MRPT-SMT-LEAF-v1"
EMPTY_LEAF_DOMAIN = b"MRPT-SMT-EMPTY-LEAF-v1"
NODE_DOMAIN = b"MRPT-SMT-NODE-v1"
ROUND_DOMAIN = b"MRPT-SMT-ROUND-v1"
ROOT_DOMAIN = b"MRPT-SMT-COMBINED-ROOT-v1"

MAGIC = b"BLSMT001"
VERSION = 1
_HEADER = struct.Struct(">8sIQ32s")


def pseudonymize_trajectory_id(secret_key: bytes, trajectory_id: bytes) -> bytes:
    if len(secret_key) < 16:
        raise ValueError("HMAC key must contain at least 16 bytes")
    if len(trajectory_id) != 32:
        raise ValueError("trajectory_id must be 32 bytes")
    return hmac.new(secret_key, PID_DOMAIN + trajectory_id, hashlib.sha256).digest()


def smt_key(pid: bytes) -> bytes:
    if len(pid) != 32:
        raise ValueError("pid must be 32 bytes")
    return hashlib.sha256(KEY_DOMAIN + pid).digest()


def _round_hash(payload: bytes, rounds: int) -> bytes:
    if rounds < 1:
        raise ValueError("hash_rounds must be >= 1")
    value = hashlib.sha256(payload).digest()
    for round_index in range(1, rounds):
        value = hashlib.sha256(
            ROUND_DOMAIN + round_index.to_bytes(4, "big") + value
        ).digest()
    return value


def hash_leaf(pid: bytes, rounds: int) -> bytes:
    return _round_hash(LEAF_DOMAIN + pid, rounds)


def hash_empty_leaf(rounds: int) -> bytes:
    return _round_hash(EMPTY_LEAF_DOMAIN, rounds)


def hash_node(left_hash: bytes, right_hash: bytes, rounds: int) -> bytes:
    if len(left_hash) != 32 or len(right_hash) != 32:
        raise ValueError("SMT child hash must be 32 bytes")
    return _round_hash(NODE_DOMAIN + left_hash + right_hash, rounds)


def combine_roots(spatial_root: bytes, smt_root: bytes) -> bytes:
    if len(spatial_root) != 32 or len(smt_root) != 32:
        raise ValueError("root must be 32 bytes")
    return hashlib.sha256(ROOT_DOMAIN + spatial_root + smt_root).digest()


def _bit(key_int: int, depth: int) -> int:
    return (key_int >> (255 - depth)) & 1


def _common_prefix_depth(first: int, last: int) -> int:
    different = first ^ last
    if different == 0:
        return 256
    return 256 - different.bit_length()


def _default_hashes(rounds: int) -> tuple[bytes, ...]:
    values = [b""] * 257
    values[256] = hash_empty_leaf(rounds)
    for depth in range(255, -1, -1):
        values[depth] = hash_node(values[depth + 1], values[depth + 1], rounds)
    return tuple(values)


@dataclass(slots=True, frozen=True)
class SmtProof:
    pid: bytes
    siblings: tuple[bytes, ...]


@dataclass(slots=True)
class CompressedSparseMerkleTree:
    """A compact index that deliberately emits uncompressed 256-level proofs.

    Only branching nodes and leaves are materialized in the index.  Query proof
    generation expands every skipped path level, emits all default siblings and
    recomputes the complete path once before returning it.  This is intentionally
    conservative and slow, while remaining a valid SHA-256 sparse Merkle tree.
    """

    hash_rounds: int
    default_hashes: tuple[bytes, ...]
    node_depth: array
    node_left: array
    node_right: array
    node_keys: bytes
    node_hashes: bytes
    root_index: int
    root_hash: bytes
    leaf_count: int

    @classmethod
    def build(cls, pids, hash_rounds: int = 1):
        unique = sorted(set(bytes(pid) for pid in pids), key=smt_key)
        if any(len(pid) != 32 for pid in unique):
            raise ValueError("all pids must be 32 bytes")

        defaults = _default_hashes(hash_rounds)
        depths = array("H")
        lefts = array("i")
        rights = array("i")
        key_blob = bytearray()
        hash_blob = bytearray()

        if not unique:
            return cls(
                hash_rounds, defaults, depths, lefts, rights,
                bytes(key_blob), bytes(hash_blob), -1, defaults[0], 0,
            )

        records = []
        previous_key = None
        for pid in unique:
            key = smt_key(pid)
            if key == previous_key:
                raise ValueError("SMT key collision")
            previous_key = key
            records.append((int.from_bytes(key, "big"), key, pid))

        def append_node(depth: int, left: int, right: int, key: bytes, digest: bytes) -> int:
            idx = len(depths)
            depths.append(depth)
            lefts.append(left)
            rights.append(right)
            key_blob.extend(key)
            hash_blob.extend(digest)
            return idx

        def key_at(node_index: int) -> bytes:
            start = node_index * 32
            return bytes(key_blob[start:start + 32])

        def digest_at(node_index: int) -> bytes:
            start = node_index * 32
            return bytes(hash_blob[start:start + 32])

        def lift(node_index: int, target_depth: int) -> bytes:
            current_depth = int(depths[node_index])
            if target_depth > current_depth:
                raise ValueError("cannot lift below a node")
            digest = digest_at(node_index)
            key_int = int.from_bytes(key_at(node_index), "big")
            for parent_depth in range(current_depth - 1, target_depth - 1, -1):
                sibling = defaults[parent_depth + 1]
                if _bit(key_int, parent_depth) == 0:
                    digest = hash_node(digest, sibling, hash_rounds)
                else:
                    digest = hash_node(sibling, digest, hash_rounds)
            return digest

        def build_range(lo: int, hi: int) -> int:
            if hi - lo == 1:
                _, key, pid = records[lo]
                return append_node(256, -1, -1, key, hash_leaf(pid, hash_rounds))

            first = records[lo][0]
            last = records[hi - 1][0]
            split_depth = _common_prefix_depth(first, last)
            if split_depth >= 256:
                raise ValueError("duplicate SMT key")

            mid_lo, mid_hi = lo + 1, hi - 1
            while mid_lo <= mid_hi:
                mid = (mid_lo + mid_hi) // 2
                if _bit(records[mid][0], split_depth) == 0:
                    mid_lo = mid + 1
                else:
                    mid_hi = mid - 1
            split = mid_lo
            if split <= lo or split >= hi:
                raise RuntimeError("invalid SMT partition")

            left = build_range(lo, split)
            right = build_range(split, hi)
            left_hash = lift(left, split_depth + 1)
            right_hash = lift(right, split_depth + 1)
            digest = hash_node(left_hash, right_hash, hash_rounds)
            return append_node(split_depth, left, right, records[lo][1], digest)

        root_index = build_range(0, len(records))
        root_depth = int(depths[root_index])
        root_digest = lift(root_index, 0) if root_depth else digest_at(root_index)
        return cls(
            hash_rounds, defaults, depths, lefts, rights,
            bytes(key_blob), bytes(hash_blob), root_index, root_digest, len(records),
        )

    def _key_at(self, node_index: int) -> bytes:
        start = node_index * 32
        return self.node_keys[start:start + 32]

    def _hash_at(self, node_index: int) -> bytes:
        start = node_index * 32
        return self.node_hashes[start:start + 32]

    def _lift(self, node_index: int, target_depth: int, default_hashes=None) -> bytes:
        current_depth = int(self.node_depth[node_index])
        if target_depth > current_depth:
            raise ValueError("cannot lift below a node")
        defaults = self.default_hashes if default_hashes is None else default_hashes
        digest = self._hash_at(node_index)
        key_int = int.from_bytes(self._key_at(node_index), "big")
        for parent_depth in range(current_depth - 1, target_depth - 1, -1):
            sibling = defaults[parent_depth + 1]
            if _bit(key_int, parent_depth) == 0:
                digest = hash_node(digest, sibling, self.hash_rounds)
            else:
                digest = hash_node(sibling, digest, self.hash_rounds)
        return digest

    def prove(self, pid: bytes, recompute_default_hashes: bool = False) -> SmtProof:
        key = smt_key(pid)
        key_int = int.from_bytes(key, "big")
        # Deliberately unoptimized mode: rebuild all 257 empty-subtree hashes
        # for every proof instead of reusing the table stored in the index.
        if recompute_default_hashes:
            # The deliberately simple proof initializer and compressed-path
            # expander are independent components.  Neither shares its empty
            # subtree table with the other.
            sibling_defaults = _default_hashes(self.hash_rounds)
            lift_defaults = _default_hashes(self.hash_rounds)
        else:
            sibling_defaults = self.default_hashes
            lift_defaults = self.default_hashes
        siblings = [sibling_defaults[depth + 1] for depth in range(256)]
        node = self.root_index
        if node < 0:
            raise KeyError("pid is not in the empty SMT")

        while int(self.node_depth[node]) < 256:
            depth = int(self.node_depth[node])
            left = int(self.node_left[node])
            right = int(self.node_right[node])
            if _bit(key_int, depth) == 0:
                child, other = left, right
            else:
                child, other = right, left
            siblings[depth] = self._lift(other, depth + 1, lift_defaults)
            node = child

        if self._key_at(node) != key:
            raise KeyError("pid is not in the SMT")

        proof = SmtProof(bytes(pid), tuple(siblings))
        # Intentionally poor server behavior: validate every complete proof
        # before returning it instead of trusting the already-authenticated tree.
        if verify_smt_proof(proof, self.root_hash, self.hash_rounds) is False:
            raise RuntimeError("generated SMT proof does not match root")
        return proof


@dataclass(slots=True)
class PrivateSmtBundle:
    index: object
    smt: CompressedSparseMerkleTree
    spatial_root: bytes
    smt_root: bytes
    trusted_root: bytes
    key_fingerprint: str


def verify_smt_proof(proof: SmtProof, expected_root: bytes, hash_rounds: int) -> bool:
    if len(proof.pid) != 32 or len(proof.siblings) != 256:
        return False
    key_int = int.from_bytes(smt_key(proof.pid), "big")
    digest = hash_leaf(proof.pid, hash_rounds)
    for depth in range(255, -1, -1):
        sibling = proof.siblings[depth]
        if len(sibling) != 32:
            return False
        if _bit(key_int, depth) == 0:
            digest = hash_node(digest, sibling, hash_rounds)
        else:
            digest = hash_node(sibling, digest, hash_rounds)
    return hmac.compare_digest(digest, expected_root)


def build_proofs_bad(
    smt: CompressedSparseMerkleTree,
    pids,
    executor: Executor | None = None,
    recompute_default_hashes: bool = True,
):
    """Build one independent, uncompressed proof for each distinct disclosed PID."""
    unique = sorted(set(bytes(pid) for pid in pids))
    def prove_one(pid):
        return smt.prove(pid, recompute_default_hashes=recompute_default_hashes)

    if executor is None:
        return [prove_one(pid) for pid in unique]
    return list(executor.map(prove_one, unique))


def select_sampled_cover_pids(verification_entries, candidate_pids, stride: int = 1):
    """Select all result PIDs plus deterministic cover PIDs.

    With the default stride of 1, every distinct PID disclosed by the VO receives
    a full independent proof.  Larger stride values remain available only for
    diagnostic runs and are not the default experiment.
    """
    if stride < 1:
        raise ValueError("proof stride must be >= 1")
    entries = list(verification_entries)
    unique = sorted({bytes(entry.trajectory_id) for entry in entries})
    selected = {bytes(pid) for pid in candidate_pids}
    if not unique:
        return sorted(selected)

    cover_count = min(len(unique), max(1, (len(entries) + stride - 1) // stride))
    for cover_index in range(cover_count):
        unique_index = (cover_index * len(unique)) // cover_count
        selected.add(unique[unique_index])
    return sorted(selected)


def verify_proofs_bad(
    proofs,
    required_pids,
    expected_root: bytes,
    hash_rounds: int,
    executor: Executor | None = None,
) -> None:
    required = sorted(set(bytes(pid) for pid in required_pids))
    proof_list = list(proofs)
    if [proof.pid for proof in proof_list] != required:
        raise ValueError("SMT proof PID set does not match disclosed VO PID set")

    def verify_one(proof):
        return verify_smt_proof(proof, expected_root, hash_rounds)

    results = map(verify_one, proof_list) if executor is None else executor.map(verify_one, proof_list)
    if not all(results):
        raise ValueError("invalid SMT membership proof")


def build_edge_batch_map(tree, edge_batch_size: int = 16):
    """Assign every spatial-node edge block to an independent verifier batch."""
    if edge_batch_size < 1:
        raise ValueError("edge_batch_size must be >= 1")
    mapping = {}
    for node_index, edge_ids in enumerate(tree.node_edges):
        for position, eid in enumerate(edge_ids):
            mapping[int(eid)] = (node_index, position // edge_batch_size)
    return mapping


def verify_proofs_in_edge_batches_bad(
    proofs,
    verification_entries,
    edge_batch_by_eid,
    expected_root: bytes,
    hash_rounds: int,
    executor: Executor | None = None,
) -> int:
    """Verify with a local PID cache per edge batch and no global cache.

    Proof bytes remain unique per PID in the VO.  Independent verifier tasks
    process blocks of edges and repeat a PID verification when that PID occurs
    in another block.  This raises verification work without increasing VO.
    """
    proof_list = list(proofs)
    proof_by_pid = {proof.pid: proof for proof in proof_list}
    if len(proof_by_pid) != len(proof_list):
        raise ValueError("duplicate PID proof")

    entries = list(verification_entries)
    required_pids = {bytes(entry.trajectory_id) for entry in entries}
    if set(proof_by_pid) != required_pids:
        raise ValueError("SMT proof PID set does not match disclosed VO PID set")

    work = sorted({
        (edge_batch_by_eid[int(entry.eid)], bytes(entry.trajectory_id))
        for entry in entries
    })

    def verify_one(item):
        _, pid = item
        return verify_smt_proof(proof_by_pid[pid], expected_root, hash_rounds)

    results = map(verify_one, work) if executor is None else executor.map(verify_one, work)
    if not all(results):
        raise ValueError("invalid SMT membership proof")
    return len(work)


def serialized_bad_proof_size(proofs) -> int:
    proof_list = list(proofs)
    # Header count + PID + all 256 sibling hashes. No compression or sharing.
    return 8 + sum(32 + 256 * 32 for _ in proof_list)


def save_bundle(bundle: PrivateSmtBundle, path) -> int:
    payload = pickle.dumps(bundle, protocol=5)
    digest = hashlib.sha256(payload).digest()
    raw = _HEADER.pack(MAGIC, VERSION, len(payload), digest) + payload
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return len(raw)


def load_bundle(path) -> PrivateSmtBundle:
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER.size:
        raise ValueError("truncated SMT bundle")
    magic, version, length, digest = _HEADER.unpack(raw[:_HEADER.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("bad SMT bundle format/version")
    payload = raw[_HEADER.size:]
    if len(payload) != length:
        raise ValueError("SMT bundle payload length mismatch")
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("SMT bundle checksum mismatch")
    return pickle.loads(payload)
