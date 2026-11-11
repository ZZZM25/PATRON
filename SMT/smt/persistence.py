from __future__ import annotations

import hashlib
import pickle
import struct
from pathlib import Path

MAGIC_INDEX = b"BLIDX002"
MAGIC_CATALOG = b"BLCAT002"
VERSION = 2
_HEADER = struct.Struct(">8sIQ32s")


def _save(obj, path, magic):
    payload = pickle.dumps(obj, protocol=5)
    digest = hashlib.sha256(payload).digest()
    raw = _HEADER.pack(magic, VERSION, len(payload), digest) + payload
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)
    return len(raw)


def _load(path, magic):
    index_path = Path(path)
    file_size = index_path.stat().st_size
    with index_path.open("rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) < _HEADER.size:
            raise ValueError("truncated file")
        got_magic, version, n, digest = _HEADER.unpack(header)
        if got_magic != magic or version != VERSION:
            raise ValueError("bad file format/version; rebuild baseline index with v2")
        if file_size != _HEADER.size + n:
            raise ValueError("payload length mismatch")

        # Verify the payload in bounded chunks.  The previous implementation
        # used Path.read_bytes() and then sliced the payload, transiently
        # holding two complete copies of multi-gigabyte index files in RAM.
        hasher = hashlib.sha256()
        remaining = n
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("truncated payload")
            hasher.update(chunk)
            remaining -= len(chunk)
        if hasher.digest() != digest:
            raise ValueError("payload checksum mismatch")

        # pickle.load reads directly from the file stream; no full payload
        # bytes object is retained alongside the deserialized index.
        stream.seek(_HEADER.size)
        return pickle.load(stream)


def save_index(index, path):
    return _save(index, path, MAGIC_INDEX)


def load_index(path):
    return _load(path, MAGIC_INDEX)


def save_catalog(catalog, path):
    return _save(catalog, path, MAGIC_CATALOG)


def load_catalog(path):
    return _load(path, MAGIC_CATALOG)
