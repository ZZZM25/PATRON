"""Canonical serialization for authenticated cell payloads."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class IndexedEdgeRecord:
    edge_id: int
    start_node: int
    end_node: int
    original_edge_id: int
    segment_index: int
    segment_count: int
    start_lon_scaled: int
    start_lat_scaled: int
    end_lon_scaled: int
    end_lat_scaled: int

    def to_dict(self) -> dict[str, int]:
        return {
            "edge_id": self.edge_id,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "original_edge_id": self.original_edge_id,
            "segment_index": self.segment_index,
            "segment_count": self.segment_count,
            "start_lon_scaled": self.start_lon_scaled,
            "start_lat_scaled": self.start_lat_scaled,
            "end_lon_scaled": self.end_lon_scaled,
            "end_lat_scaled": self.end_lat_scaled,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IndexedEdgeRecord":
        fields = (
            "edge_id",
            "start_node",
            "end_node",
            "original_edge_id",
            "segment_index",
            "segment_count",
            "start_lon_scaled",
            "start_lat_scaled",
            "end_lon_scaled",
            "end_lat_scaled",
        )
        missing = [field for field in fields if field not in value]
        if missing:
            raise ValueError(f"indexed edge record is missing fields: {missing}")
        return cls(*(int(value[field]) for field in fields))


def serialize_edge_record(record: IndexedEdgeRecord) -> bytes:
    """Serialize one edge with fixed-width little-endian fields."""

    return struct.pack(
        "<QQQQQQqqqq",
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
    )


def serialize_cell_payload(records: list[IndexedEdgeRecord]) -> bytes:
    """Canonical leaf payload: count followed by edge records by edge ID."""

    ordered = sorted(records, key=lambda record: record.edge_id)
    output = bytearray(struct.pack("<I", len(ordered)))
    for record in ordered:
        output.extend(serialize_edge_record(record))
    return bytes(output)


def _prefix_bytes(cell_id: str, bits: int) -> bytes:
    if len(cell_id) != bits or any(bit not in "01" for bit in cell_id):
        raise ValueError(f"invalid {bits}-bit cell prefix: {cell_id!r}")
    byte_length = (bits + 7) // 8
    return int(cell_id, 2).to_bytes(byte_length, byteorder="big")


def leaf_hash(cell_id: str, records: list[IndexedEdgeRecord], bits: int) -> bytes:
    """Hash a non-empty cell and bind the cell prefix to its contents."""

    payload = serialize_cell_payload(records)
    # The fixed bit depth is included to make the prefix encoding unambiguous.
    domain = struct.pack("<H", bits) + _prefix_bytes(cell_id, bits)
    return hashlib.sha256(domain + payload).digest()


def empty_leaf_hash() -> bytes:
    """Hash used for every empty fixed-depth cell."""

    return hashlib.sha256(b"PATRON_EMPTY_LEAF_v1").digest()


def hash_pair(left: bytes, right: bytes) -> bytes:
    """Hash an internal Merkle node exactly as SHA256(left || right)."""

    return hashlib.sha256(left + right).digest()
