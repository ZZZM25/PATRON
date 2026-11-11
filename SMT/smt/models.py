from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Entry:
    trajectory_id: bytes
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class TrajectorySegment:
    start_node: int
    start_time: int
    end_node: int
    end_time: int


@dataclass(slots=True)
class ParsedTrajectory:
    trajectory_id: bytes
    eid_path: list[int]
    segments: list[TrajectorySegment]


@dataclass(slots=True, frozen=True)
class VerificationEntry:
    eid: int
    trajectory_id: bytes
    start: int
    end: int
