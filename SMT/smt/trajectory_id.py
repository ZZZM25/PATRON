from __future__ import annotations

import hashlib
import struct

_U32 = struct.Struct(">I")


def compute_trajectory_id(eid_path: list[int], trajectory_start: int, trajectory_end: int) -> bytes:
    h = hashlib.sha256()
    for eid in eid_path:
        h.update(_U32.pack(int(eid)))
    h.update(_U32.pack(int(trajectory_start)))
    h.update(_U32.pack(int(trajectory_end)))
    return h.digest()
