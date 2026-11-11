from datetime import date, datetime, time, timezone
from .config import SLOT_SECONDS, SLOTS_PER_DAY

def day_start_timestamp(day:str) -> int:
    """返回某个 UTC 日期零点的时间戳。"""
    parsed_day = date.fromisoformat(day)
    day_start = datetime.combine(parsed_day,time.min,tzinfo=timezone.utc)
    return int(day_start.timestamp())

def infer_day(timestamp: int) -> str:
    """根据 UTC 时间戳推断日期。"""
    value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return value.date().isoformat()

def slot_interval(
    day_start_ts:int,
    slot:int,
    slot_seconds:int = SLOT_SECONDS,
    slot_count:int = SLOTS_PER_DAY,
) -> tuple[int, int]:
    """返回某个时间槽对应的左闭右开时间区间。"""
    if slot_seconds <= 0 or slot_count <= 0:
        raise ValueError("slot configuration must be positive")
    if not 0 <= slot < slot_count:
        raise ValueError(f"invalid time slot: {slot}")
    start_ts = day_start_ts + slot * slot_seconds
    end_ts = start_ts + slot_seconds
    return start_ts, end_ts

def slots_covering_interval(
    day_start_ts: int,
    start_ts: int,
    end_ts: int,
    slot_seconds: int = SLOT_SECONDS,
    slot_count: int = SLOTS_PER_DAY,
) -> list[int]:
    """返回左闭右开时间区间覆盖的全部时间槽。"""

    if end_ts <= start_ts:
        raise ValueError(
            "end timestamp must be greater than start timestamp"
        )

    if slot_seconds <= 0 or slot_count <= 0:
        raise ValueError("slot configuration must be positive")

    day_end_ts = day_start_ts + slot_count * slot_seconds

    if start_ts < day_start_ts or end_ts > day_end_ts:
        raise ValueError(
            "time interval is outside the selected day"
        )

    first_slot = (
        start_ts - day_start_ts
    ) // slot_seconds

    last_slot = (
        end_ts - 1 - day_start_ts
    ) // slot_seconds

    return list(
        range(
            int(first_slot),
            int(last_slot) + 1,
        )
    )


def complete_slots_for_interval(
    period_start_ts: int,
    start_ts: int,
    end_ts: int,
    slot_seconds: int = SLOT_SECONDS,
    slot_count: int = SLOTS_PER_DAY,
) -> list[int]:
    """Return slots only when the query is exactly a union of complete slots.

    Construction still uses ``slots_covering_interval`` because trajectory
    segments may start or end inside a slot. Query and verifier entry points
    use this stricter function so no partial boundary slot enters a VO.
    """
    if end_ts <= start_ts:
        raise ValueError("end timestamp must be greater than start timestamp")
    if slot_seconds <= 0 or slot_count <= 0:
        raise ValueError("slot configuration must be positive")

    period_end_ts = period_start_ts + slot_count * slot_seconds
    if start_ts < period_start_ts or end_ts > period_end_ts:
        raise ValueError("time interval is outside the authenticated period")

    start_offset = start_ts - period_start_ts
    end_offset = end_ts - period_start_ts
    if start_offset % slot_seconds != 0 or end_offset % slot_seconds != 0:
        raise ValueError(
            "query interval must start and end on authenticated time-slot boundaries"
        )

    first_slot = start_offset // slot_seconds
    end_slot = end_offset // slot_seconds
    return list(range(int(first_slot), int(end_slot)))


def partition_slots(slots:list[int],maximum_size:int) -> list[list[int]]:
    """将完整查询槽确定性地拆成不超过SRS多点上限的连续分片。"""
    normalized = [int(slot) for slot in slots]

    if maximum_size <= 0:
        raise ValueError("maximum slot chunk size must be positive")

    if normalized != sorted(set(normalized)):
        raise ValueError("query slots must be sorted and unique")

    return [normalized[index:index + maximum_size] for index in range(0,len(normalized),maximum_size)]
