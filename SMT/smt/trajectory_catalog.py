from __future__ import annotations

from array import array

from .models import TrajectorySegment
from .query_utils import normalize_rectangle, segment_intersects_rectangle


class PackedTrajectoryCatalog:
    __slots__ = (
        "offset_count", "start_nodes", "start_times", "end_nodes", "end_times"
    )

    def __init__(self) -> None:
        self.offset_count: dict[bytes, tuple[int, int]] = {}
        self.start_nodes = array("I")
        self.start_times = array("I")
        self.end_nodes = array("I")
        self.end_times = array("I")

    def add(self, trajectory_id: bytes, segments: list[TrajectorySegment]) -> None:
        if trajectory_id in self.offset_count:
            raise ValueError("duplicate trajectory_id: " + trajectory_id.hex())
        offset = len(self.start_nodes)
        for seg in segments:
            for value in (seg.start_node, seg.start_time, seg.end_node, seg.end_time):
                if not 0 <= int(value) <= 0xFFFFFFFF:
                    raise ValueError(f"catalog value outside uint32: {value}")
            self.start_nodes.append(seg.start_node)
            self.start_times.append(seg.start_time)
            self.end_nodes.append(seg.end_node)
            self.end_times.append(seg.end_time)
        self.offset_count[trajectory_id] = (offset, len(segments))

    def iter_segments(self, trajectory_id: bytes):
        try:
            offset, count = self.offset_count[trajectory_id]
        except KeyError as e:
            raise KeyError("trajectory not in catalog: " + trajectory_id.hex()) from e
        for i in range(offset, offset + count):
            yield (
                int(self.start_nodes[i]), int(self.start_times[i]),
                int(self.end_nodes[i]), int(self.end_times[i]),
            )

    def time_range(self) -> tuple[int, int]:
        if not self.start_times:
            raise ValueError("empty catalog")
        return min(self.start_times), max(self.end_times)


def trajectory_matches_exact_query(catalog, trajectory_id: bytes, road,
                                   query_min_lon, query_min_lat, query_max_lon, query_max_lat,
                                   query_start: int, query_end: int) -> bool:
    query_min_lon, query_min_lat, query_max_lon, query_max_lat = normalize_rectangle(
        query_min_lon, query_min_lat, query_max_lon, query_max_lat
    )
    for u, start, v, end in catalog.iter_segments(trajectory_id):
        overlap_start = max(start, query_start)
        overlap_end = min(end, query_end)
        if overlap_start > overlap_end:
            continue
        x1, y1 = road.node_lon[u], road.node_lat[u]
        x2, y2 = road.node_lon[v], road.node_lat[v]
        duration = end - start
        # Keep the Chengdu/Xi'an semantics used by the existing project.
        if duration == 0:
            if query_min_lon <= x1 <= query_max_lon and query_min_lat <= y1 <= query_max_lat:
                return True
            continue
        a0 = (overlap_start - start) / duration
        a1 = (overlap_end - start) / duration
        sx, sy = x1 + (x2-x1)*a0, y1 + (y2-y1)*a0
        ex, ey = x1 + (x2-x1)*a1, y1 + (y2-y1)*a1
        if segment_intersects_rectangle(
            sx, sy, ex, ey,
            query_min_lon, query_min_lat, query_max_lon, query_max_lat,
        ):
            return True
    return False


def fine_filter_candidates(candidate_ids, catalog, road,
                           query_min_lon, query_min_lat, query_max_lon, query_max_lat,
                           query_start: int, query_end: int) -> list[bytes]:
    out = []
    for tid in candidate_ids:
        if trajectory_matches_exact_query(
            catalog, tid, road,
            query_min_lon, query_min_lat, query_max_lon, query_max_lat,
            query_start, query_end,
        ):
            out.append(tid)
    return out
