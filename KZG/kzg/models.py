from __future__ import annotations
from dataclasses import dataclass
from types import ClassMethodDescriptorType


@dataclass(frozen=True, order=True)
class SegmentRecord:
    """一条轨迹在一条道路边上的时间记录。"""
    traj_id: int
    start_ts: int
    end_ts: int

    def as_list(self) -> list[int]:
        return [self.traj_id, self.start_ts, self.end_ts]

    @classmethod
    def from_list(cls, value: list[int]) -> SegmentRecord:
        if len(value) != 3:
            raise ValueError(
                f"segment record must contain 3 fields, got {value!r}"
            )
        return cls(*(int(item) for item in value))

    def overlaps(self, start_ts:int, end_ts:int) -> bool:
        return self.start_ts < end_ts and self.end_ts > start_ts

@dataclass(frozen=True)
class Edge:
    edge_id: int
    start_node: int
    end_node: int


@dataclass(frozen=True)
class Rect:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def normalized(self) -> "Rect":
        return Rect(
            min_lon=min(self.min_lon, self.max_lon),
            min_lat=min(self.min_lat, self.max_lat),
            max_lon=max(self.min_lon, self.max_lon),
            max_lat=max(self.min_lat, self.max_lat),
        )