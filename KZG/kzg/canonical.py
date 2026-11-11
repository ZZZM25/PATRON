import struct
from collections.abc import Iterable
from datetime import date
from .field import hash_to_field
from .models import SegmentRecord
# Q：无符号64位整数，保存 traj_id
# I：无符号32位整数，保存 edge_id
# q：有符号64位整数，保存时间戳
_RECORD_STRUCT = struct.Struct(">QIqq")

def normalize_records(records:Iterable[SegmentRecord]) -> list[SegmentRecord]:
    return sorted(set(records), key=lambda r: (r.traj_id, r.start_ts, r.end_ts))

def encode_cell_records(day: str, edge_id: int, position: int, records: Iterable[SegmentRecord]) -> bytes:
    """编码某条道路边在某个位置的轨迹记录。"""
    try:
        parsed_day = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError("day must use YYYY-MM-DD format") from exc
    if parsed_day.isoformat() != day:
        raise ValueError("day must use YYYY-MM-DD format")
    if not 0 <= edge_id <= 0xFFFFFFFF:
        raise ValueError("edge ID must fit an unsigned 32-bit integer")
    if not 0 <= position <= 0xFFFFFFFF:
        raise ValueError("position must fit an unsigned 32-bit integer")
    normalized = normalize_records(records)
    output = bytearray()
    output.extend(day.encode("ascii"))
    output.extend(struct.pack(">II", position, len(normalized)))
    for record in normalized:
        if record.traj_id < 0:
            raise ValueError("trajectory ID must not be negative")
        output.extend(_RECORD_STRUCT.pack(record.traj_id, edge_id, record.start_ts, record.end_ts))
    return bytes(output)

def cell_value(day: str, edge_id: int, position: int, records: Iterable[SegmentRecord]) -> int:
    """计算某条边的分钟槽有限域值。"""
    encoded = encode_cell_records(day, edge_id, position, records)
    return hash_to_field(encoded)