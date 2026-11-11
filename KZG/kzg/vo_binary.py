from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


VO_FORMAT = "kzg-geohash-vo-binary-v2"
_V1_FORMAT = "kzg-geohash-vo-binary-v1"
_V1_MAGIC = b"KZGVO\x01\r\n"
_MAGIC = b"KZGVO\x02\r\n"
_HEADER = struct.Struct(">8sQ32s")
_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


def _compact_proof(proof: Mapping[str, Any]) -> dict:
    """Remove values that the verifier deterministically derives from records."""
    compact = dict(proof)
    raw_edges = proof.get("edges")
    if not isinstance(raw_edges, list):
        return compact
    compact_edges = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Mapping):
            compact_edges.append(raw_edge)
            continue
        edge = dict(raw_edge)
        raw_slots = raw_edge.get("slots")
        if isinstance(raw_slots, list):
            edge["slots"] = [
                {key: value for key, value in slot.items() if key != "value"}
                if isinstance(slot, Mapping)
                else slot
                for slot in raw_slots
            ]
        compact_edges.append(edge)
    compact["edges"] = compact_edges
    return compact


def _canonical_payload(proof: Mapping[str, Any], claimed: Mapping[str, Any]) -> bytes:
    envelope = {
        "format": VO_FORMAT,
        "proof": _compact_proof(proof),
        "claimed": dict(claimed),
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _encode_canonical_payload(raw: bytes) -> bytes:
    if len(raw) > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError("VO exceeds the maximum supported uncompressed size")
    digest = hashlib.sha256(raw).digest()
    compressed = zlib.compress(raw, level=9)
    return _HEADER.pack(_MAGIC, len(raw), digest) + compressed


def encode_vo(proof: Mapping[str, Any], claimed: Mapping[str, Any]) -> bytes:
    """Encode a proof and its claimed result as a safe compressed binary VO.

    JSON is used only as the deterministic value model inside the envelope.  The
    wire/file representation is binary and compressed; unlike pickle, decoding
    it never executes input-controlled Python objects.
    """
    if not isinstance(proof, Mapping) or not isinstance(claimed, Mapping):
        raise TypeError("proof and claimed result must be mappings")
    return _encode_canonical_payload(_canonical_payload(proof, claimed))


def decode_vo(encoded: bytes | bytearray | memoryview) -> tuple[dict, dict]:
    """Decode and integrity-check one KZG Geohash binary VO."""
    data = bytes(encoded)
    if len(data) < _HEADER.size:
        raise ValueError("truncated binary VO header")
    magic, raw_size, expected_digest = _HEADER.unpack_from(data)
    expected_format = {_MAGIC: VO_FORMAT, _V1_MAGIC: _V1_FORMAT}.get(magic)
    if expected_format is None:
        raise ValueError("unsupported binary VO format or version")
    if raw_size > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError("binary VO declares an excessive uncompressed size")

    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(data[_HEADER.size :], raw_size + 1)
    except zlib.error as exc:
        raise ValueError("invalid compressed binary VO payload") from exc
    if len(raw) != raw_size:
        raise ValueError("binary VO uncompressed size mismatch")
    if not decompressor.eof or decompressor.unconsumed_tail or decompressor.unused_data:
        raise ValueError("binary VO has truncated or trailing compressed data")
    if hashlib.sha256(raw).digest() != expected_digest:
        raise ValueError("binary VO payload digest mismatch")

    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("binary VO payload is not valid canonical data") from exc
    if not isinstance(envelope, dict) or envelope.get("format") != expected_format:
        raise ValueError("binary VO envelope format mismatch")
    if set(envelope) != {"format", "proof", "claimed"}:
        raise ValueError("binary VO envelope has unexpected fields")
    proof = envelope.get("proof")
    claimed = envelope.get("claimed")
    if not isinstance(proof, dict) or not isinstance(claimed, dict):
        raise ValueError("binary VO proof and claimed result must be objects")
    return proof, claimed


def write_vo(path: str | Path, proof: Mapping[str, Any], claimed: Mapping[str, Any]) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical_bytes = _canonical_payload(proof, claimed)
    encoded = _encode_canonical_payload(canonical_bytes)
    target.write_bytes(encoded)
    return {
        "format": VO_FORMAT,
        "path": str(target),
        "vo_bytes": len(encoded),
        "canonical_bytes": len(canonical_bytes),
        "size_reduction_percent": (
            100.0 * (1.0 - len(encoded) / len(canonical_bytes))
            if canonical_bytes
            else 0.0
        ),
    }


def read_vo(path: str | Path) -> tuple[dict, dict]:
    return decode_vo(Path(path).read_bytes())
