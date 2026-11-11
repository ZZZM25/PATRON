from __future__ import annotations

import hashlib
import struct

_U32 = struct.Struct(">I")
_F64_4 = struct.Struct(">dddd")

ENTRY_DOMAIN = b"BL-ENTRY-v2"
ENTRY_LIST_TAIL = hashlib.sha256(b"BL-SORTED-LIST-TAIL-v2").digest()
ENTRY_LIST_DOMAIN = b"BL-SORTED-LIST-CHAIN-v2"
EDGE_DOMAIN = b"BL-EDGE-v2"
EDGE_GROUP_INIT = hashlib.sha256(b"BL-EDGE-GROUP-INIT-v2").digest()
EDGE_GROUP_DOMAIN = b"BL-EDGE-GROUP-v2"
LEAF_DOMAIN = b"BL-SPATIAL-LEAF-v2"
INTERNAL_DOMAIN = b"BL-SPATIAL-INTERNAL-v2"


def u32(value: int) -> bytes:
    if not 0 <= int(value) <= 0xFFFFFFFF:
        raise ValueError(f"uint32 overflow: {value}")
    return _U32.pack(int(value))


def hash_entry(trajectory_id: bytes, start: int, end: int) -> bytes:
    if not isinstance(trajectory_id, (bytes, bytearray)) or len(trajectory_id) != 32:
        raise ValueError("trajectory_id must be 32 bytes")
    return hashlib.sha256(
        ENTRY_DOMAIN + bytes(trajectory_id) + u32(start) + u32(end)
    ).digest()


def entry_list_prepend(entry_hash: bytes, suffix_state: bytes) -> bytes:
    """Reverse hash-chain step: state_i = H(entry_i || state_{i+1})."""
    if len(entry_hash) != 32 or len(suffix_state) != 32:
        raise ValueError("hash state must be 32 bytes")
    return hashlib.sha256(ENTRY_LIST_DOMAIN + entry_hash + suffix_state).digest()


def hash_edge(eid: int, entry_count: int, entry_list_root: bytes) -> bytes:
    return hashlib.sha256(
        EDGE_DOMAIN + u32(eid) + u32(entry_count) + entry_list_root
    ).digest()


def edge_group_append(state: bytes, edge_hash: bytes) -> bytes:
    return hashlib.sha256(EDGE_GROUP_DOMAIN + state + edge_hash).digest()


def pack_mbr(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bytes:
    return _F64_4.pack(float(min_lon), float(min_lat), float(max_lon), float(max_lat))


def hash_leaf(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    edge_count: int,
    edge_group_root: bytes,
) -> bytes:
    return hashlib.sha256(
        LEAF_DOMAIN
        + pack_mbr(min_lon, min_lat, max_lon, max_lat)
        + u32(edge_count)
        + edge_group_root
    ).digest()


def hash_internal(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    left_hash: bytes,
    right_hash: bytes,
    cross_count: int,
    cross_group_root: bytes,
) -> bytes:
    return hashlib.sha256(
        INTERNAL_DOMAIN
        + pack_mbr(min_lon, min_lat, max_lon, max_lat)
        + left_hash
        + right_hash
        + u32(cross_count)
        + cross_group_root
    ).digest()


def time_overlap(start: int, end: int, query_start: int, query_end: int) -> bool:
    return start <= query_end and end >= query_start


SMT_EDGE_DOMAIN = b"BL-EDGE-SMT-v1"

def hash_smt_edge(eid: int, entry_count: int, chain_root: bytes, smt_root: bytes) -> bytes:
    if len(chain_root) != 32 or len(smt_root) != 32:
        raise ValueError("SMT edge roots must be 32 bytes")
    return hashlib.sha256(
        SMT_EDGE_DOMAIN + u32(eid) + u32(entry_count) + chain_root + smt_root
    ).digest()
