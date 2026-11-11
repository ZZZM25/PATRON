"""Canonical binary encoding for RSG verification objects.

The format is deliberately small and self-describing.  It does not use
pickle, so bytes received from an untrusted server never trigger object
construction or code execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MAGIC = b"RSGV"
VERSION = 1

TYPE_MEMBER_BATCH = 1
TYPE_COMPLETENESS = 2
TYPE_INVALID_ITEMS = 3
TYPE_PRIVATE_COMPLETENESS = 4

MAX_VO_BYTES = 64 * 1024 * 1024
MAX_ITEM_COUNT = 10_000_000
MAX_PLAINTEXT_BYTES = 1 * 1024 * 1024
MAX_TOKEN_BYTES = 64


class VOFormatError(ValueError):
    """Raised when an untrusted VO byte string is malformed."""


def _varint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError("varint cannot encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _varbytes(value: bytes) -> bytes:
    return _varint(len(value)) + value


def _uint_bytes(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError("unsigned integer cannot be negative")
    if value == 0:
        return b""
    return value.to_bytes((value.bit_length() + 7) // 8, "big")


def _unsigned(value: int) -> bytes:
    return _varbytes(_uint_bytes(value))


def _signed(value: int) -> bytes:
    value = int(value)
    sign = 1 if value < 0 else 0
    return bytes((sign,)) + _unsigned(abs(value))


@dataclass
class _Reader:
    data: memoryview
    offset: int = 0

    @classmethod
    def from_blob(cls, blob: bytes | bytearray | memoryview) -> "_Reader":
        raw = memoryview(blob)
        if len(raw) > MAX_VO_BYTES:
            raise VOFormatError("VO exceeds maximum size")
        return cls(raw)

    def read(self, size: int) -> bytes:
        size = int(size)
        if size < 0 or self.offset + size > len(self.data):
            raise VOFormatError("truncated VO")
        result = bytes(self.data[self.offset:self.offset + size])
        self.offset += size
        return result

    def u8(self) -> int:
        return self.read(1)[0]

    def varint(self) -> int:
        value = 0
        shift = 0
        encoded = bytearray()
        for _ in range(10):
            byte = self.u8()
            encoded.append(byte)
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                if bytes(encoded) != _varint(value):
                    raise VOFormatError("non-canonical varint")
                return value
            shift += 7
        raise VOFormatError("varint is too large")

    def varbytes(self, maximum: int = MAX_VO_BYTES) -> bytes:
        size = self.varint()
        if size > maximum:
            raise VOFormatError("length-prefixed field exceeds maximum size")
        return self.read(size)

    def unsigned(self) -> int:
        raw = self.varbytes()
        if raw and raw[0] == 0:
            raise VOFormatError("non-canonical unsigned integer")
        return int.from_bytes(raw, "big") if raw else 0

    def signed(self) -> int:
        sign = self.u8()
        if sign not in (0, 1):
            raise VOFormatError("invalid integer sign")
        magnitude = self.unsigned()
        if sign and magnitude == 0:
            raise VOFormatError("negative zero is not canonical")
        return -magnitude if sign else magnitude

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise VOFormatError("trailing bytes after VO")


def _header(vo_type: int) -> bytes:
    return MAGIC + bytes((VERSION, vo_type))


def _open(blob: bytes | bytearray | memoryview, expected_type: int) -> _Reader:
    reader = _Reader.from_blob(blob)
    if reader.read(len(MAGIC)) != MAGIC:
        raise VOFormatError("invalid VO magic")
    if reader.u8() != VERSION:
        raise VOFormatError("unsupported VO version")
    if reader.u8() != expected_type:
        raise VOFormatError("unexpected VO type")
    return reader


def encode_member_batch(vo: dict) -> bytes:
    status = vo.get("status")
    if status == "empty_member_batch":
        return _header(TYPE_MEMBER_BATCH) + bytes((0,)) + _varint(0)
    if status != "member_batch":
        raise ValueError(f"unsupported member status: {status}")

    items = list(vo.get("member_items", []))
    out = bytearray(_header(TYPE_MEMBER_BATCH))
    out.append(1)
    out.extend(_varint(len(items)))
    for item in items:
        plaintext = str(item["plaintext"]).encode("utf-8")
        if len(plaintext) > MAX_PLAINTEXT_BYTES:
            raise ValueError("member plaintext is too large")
        out.extend(_varint(int(item["traj_id"])))
        out.extend(_varbytes(plaintext))
    out.extend(_unsigned(int(vo["w"])))
    if len(out) > MAX_VO_BYTES:
        raise ValueError("member VO exceeds maximum size")
    return bytes(out)


def decode_member_batch(blob: bytes | bytearray | memoryview) -> dict:
    reader = _open(blob, TYPE_MEMBER_BATCH)
    status_code = reader.u8()
    count = reader.varint()
    if count > MAX_ITEM_COUNT:
        raise VOFormatError("member item count exceeds maximum")
    if status_code == 0:
        if count != 0:
            raise VOFormatError("empty member VO has nonzero count")
        reader.finish()
        return {"status": "empty_member_batch", "member_items": []}
    if status_code != 1:
        raise VOFormatError("unknown member status")

    items = []
    for _ in range(count):
        trajectory_id = reader.varint()
        try:
            plaintext = reader.varbytes(MAX_PLAINTEXT_BYTES).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VOFormatError("member plaintext is not UTF-8") from exc
        items.append({"traj_id": trajectory_id, "plaintext": plaintext})
    witness = reader.unsigned()
    reader.finish()
    return {
        "status": "member_batch",
        "member_items": items,
        "w": witness,
        "proof_mode": "binary",
    }


def encode_completeness(vo: dict) -> bytes:
    status = vo.get("status")
    status_code = {
        "batch_nonmembership": 1,
        "detected_incomplete_claim": 2,
    }.get(status)
    if status_code is None:
        raise ValueError(f"unsupported completeness status: {status}")
    try:
        digest = bytes.fromhex(str(vo["binding_digest"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid completeness binding digest") from exc
    if len(digest) != 32:
        raise ValueError("completeness binding digest must be 32 bytes")

    out = bytearray(_header(TYPE_COMPLETENESS))
    out.append(status_code)
    out.extend(_unsigned(int(vo["x_batch"])))
    out.extend(digest)
    if status_code == 1:
        out.extend(_signed(int(vo["a"])))
        out.extend(_unsigned(int(vo["d"])))
    if len(out) > MAX_VO_BYTES:
        raise ValueError("completeness VO exceeds maximum size")
    return bytes(out)


def decode_completeness(blob: bytes | bytearray | memoryview) -> dict:
    reader = _open(blob, TYPE_COMPLETENESS)
    status_code = reader.u8()
    if status_code not in (1, 2):
        raise VOFormatError("unknown completeness status")
    x_batch = reader.unsigned()
    digest = reader.read(32).hex()
    result = {
        "status": (
            "batch_nonmembership"
            if status_code == 1
            else "detected_incomplete_claim"
        ),
        "x_batch": x_batch,
        "binding_digest": digest,
    }
    if status_code == 1:
        result["a"] = reader.signed()
        result["d"] = reader.unsigned()
    reader.finish()
    return result


def encode_private_completeness(vo: dict) -> bytes:
    """Encode the privacy-preserving count/token completeness proof.

    Non-result trajectories are represented only by fixed opaque tokens.  The
    client reconstructs the time representative from those tokens; x_batch is
    intentionally not transported.
    """
    status = vo.get("status")
    status_code = {
        "batch_nonmembership": 1,
        "detected_incomplete_claim": 2,
    }.get(status)
    if status_code is None:
        raise ValueError(f"unsupported private completeness status: {status}")

    tokens = [bytes(value) for value in vo.get("omitted_tokens", [])]
    out = bytearray(_header(TYPE_PRIVATE_COMPLETENESS))
    out.append(status_code)
    out.extend(_varint(int(vo["total_traj_count"])))
    out.extend(_varint(len(tokens)))
    for token in tokens:
        if not token or len(token) > MAX_TOKEN_BYTES:
            raise ValueError("invalid opaque token length")
        out.extend(_varbytes(token))
    out.extend(_unsigned(int(vo["edge_ac"])))
    out.extend(_unsigned(int(vo["root_witness"])))
    out.extend(_unsigned(int(vo["identity_witness"])))
    if status_code == 1:
        out.extend(_signed(int(vo["a"])))
        out.extend(_unsigned(int(vo["d"])))
    if len(out) > MAX_VO_BYTES:
        raise ValueError("private completeness VO exceeds maximum size")
    return bytes(out)


def decode_private_completeness(blob: bytes | bytearray | memoryview) -> dict:
    reader = _open(blob, TYPE_PRIVATE_COMPLETENESS)
    status_code = reader.u8()
    if status_code not in (1, 2):
        raise VOFormatError("unknown private completeness status")
    total_traj_count = reader.varint()
    token_count = reader.varint()
    if total_traj_count > MAX_ITEM_COUNT or token_count > MAX_ITEM_COUNT:
        raise VOFormatError("private completeness count exceeds maximum")
    tokens = []
    for _ in range(token_count):
        token = reader.varbytes(MAX_TOKEN_BYTES)
        if not token:
            raise VOFormatError("empty opaque token")
        tokens.append(token)
    result = {
        "status": (
            "batch_nonmembership"
            if status_code == 1
            else "detected_incomplete_claim"
        ),
        "total_traj_count": total_traj_count,
        "omitted_tokens": tokens,
        "edge_ac": reader.unsigned(),
        "root_witness": reader.unsigned(),
        "identity_witness": reader.unsigned(),
    }
    if status_code == 1:
        result["a"] = reader.signed()
        result["d"] = reader.unsigned()
    reader.finish()
    return result


def encode_invalid_items(items: Iterable[dict]) -> bytes:
    values = list(items)
    out = bytearray(_header(TYPE_INVALID_ITEMS))
    out.extend(_varint(len(values)))
    status_codes = {"trajectory_absent": 1, "time_mismatch": 2}
    for item in values:
        status_code = status_codes.get(item.get("status"))
        if status_code is None:
            raise ValueError(f"unsupported invalid-item status: {item.get('status')}")
        out.extend(_varint(int(item["traj_id"])))
        out.append(status_code)
        proof = item.get("proof")
        out.append(0 if proof is None else 1)
        if proof is not None:
            out.extend(_signed(int(proof["a"])))
            out.extend(_unsigned(int(proof["d"])))
    if len(out) > MAX_VO_BYTES:
        raise ValueError("invalid-items VO exceeds maximum size")
    return bytes(out)


def decode_invalid_items(blob: bytes | bytearray | memoryview) -> list[dict]:
    reader = _open(blob, TYPE_INVALID_ITEMS)
    count = reader.varint()
    if count > MAX_ITEM_COUNT:
        raise VOFormatError("invalid-item count exceeds maximum")
    statuses = {1: "trajectory_absent", 2: "time_mismatch"}
    result = []
    for _ in range(count):
        trajectory_id = reader.varint()
        status_code = reader.u8()
        if status_code not in statuses:
            raise VOFormatError("unknown invalid-item status")
        has_proof = reader.u8()
        if has_proof not in (0, 1):
            raise VOFormatError("invalid proof-presence flag")
        proof = None
        if has_proof:
            proof = {"a": reader.signed(), "d": reader.unsigned()}
        result.append(
            {
                "traj_id": trajectory_id,
                "status": statuses[status_code],
                "proof": proof,
            }
        )
    reader.finish()
    return result
