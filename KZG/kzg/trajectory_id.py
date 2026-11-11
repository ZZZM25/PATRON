from __future__ import annotations

from datetime import date

_TRAJECTORY_SEQUENCE_BASE = 1_000_000


def encode_daily_trajectory_id(day: str, ordinal: int) -> int:
    """Encode YYYY-MM-DD + 1-based daily ordinal into one uint64-compatible ID."""
    parsed = date.fromisoformat(day)
    if parsed.isoformat() != day:
        raise ValueError("day must use YYYY-MM-DD format")
    ordinal = int(ordinal)
    if not 1 <= ordinal < _TRAJECTORY_SEQUENCE_BASE:
        raise ValueError("daily trajectory ordinal must be in [1, 999999]")
    date_number = int(day.replace("-", ""))
    return date_number * _TRAJECTORY_SEQUENCE_BASE + ordinal


def format_daily_trajectory_id(trajectory_id: int) -> str:
    value = int(trajectory_id)
    date_number, ordinal = divmod(value, _TRAJECTORY_SEQUENCE_BASE)
    return f"{date_number:08d}_{ordinal:06d}"


def format_trajectory_id(trajectory_id: int, scheme: str) -> str:
    """Format an authenticated ID according to the dataset's declared scheme."""
    if scheme == "global-source-row-ordinal-1based":
        return str(int(trajectory_id))
    return format_daily_trajectory_id(trajectory_id)
