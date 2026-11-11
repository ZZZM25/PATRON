from __future__ import annotations

from array import array
from bisect import bisect_right


class PackedEdgeEntryStore:
    """Per-edge Entry lists kept sorted by start time during insertion.

    Every Entry is inserted directly into its time-ordered position according
    to ``start``.  Equal-start entries are inserted after existing equal-start
    entries, preserving arrival order among ties.

    The list has no min/max temporal envelope, interval tree, MMR, bucket
    hierarchy, or other temporal subtree metadata.
    """

    __slots__ = ("trajectory_id_bytes", "starts", "ends", "entry_counts")

    def __init__(self, max_eid: int) -> None:
        n = max_eid + 1
        self.trajectory_id_bytes: list[bytearray] = [bytearray() for _ in range(n)]
        self.starts: list[array] = [array("I") for _ in range(n)]
        self.ends: list[array] = [array("I") for _ in range(n)]
        self.entry_counts: list[int] = [0] * n

    def insert_by_start(self, eid: int, trajectory_id: bytes, start: int, end: int) -> None:
        """Insert one Entry into its start-time ordered position.

        This is the baseline's temporal organization step.  It happens during
        index construction rather than as a final batch sort.
        """
        if len(trajectory_id) != 32:
            raise ValueError("trajectory_id must be 32 bytes")
        if not (0 <= start <= 0xFFFFFFFF and 0 <= end <= 0xFFFFFFFF):
            raise ValueError("timestamp outside uint32")

        starts = self.starts[eid]
        pos = bisect_right(starts, int(start))

        self.starts[eid].insert(pos, int(start))
        self.ends[eid].insert(pos, int(end))

        raw = self.trajectory_id_bytes[eid]
        byte_pos = pos * 32
        raw[byte_pos:byte_pos] = trajectory_id

        self.entry_counts[eid] += 1

    def append(self, eid: int, trajectory_id: bytes, start: int, end: int) -> None:
        """Compatibility alias: insertion is time-ordered, not tail append."""
        self.insert_by_start(eid, trajectory_id, start, end)

    def get_entry(self, eid: int, index: int) -> tuple[bytes, int, int]:
        count = self.entry_counts[eid]
        if not 0 <= index < count:
            raise IndexError(index)
        pos = index * 32
        tid = bytes(self.trajectory_id_bytes[eid][pos:pos + 32])
        return tid, int(self.starts[eid][index]), int(self.ends[eid][index])

    def iter_entries(self, eid: int):
        raw = self.trajectory_id_bytes[eid]
        starts = self.starts[eid]
        ends = self.ends[eid]
        for i in range(self.entry_counts[eid]):
            p = i * 32
            yield bytes(raw[p:p + 32]), int(starts[i]), int(ends[i])

    def iter_prefix(self, eid: int, count: int):
        if not 0 <= count <= self.entry_counts[eid]:
            raise ValueError("invalid prefix count")
        raw = self.trajectory_id_bytes[eid]
        starts = self.starts[eid]
        ends = self.ends[eid]
        for i in range(count):
            p = i * 32
            yield bytes(raw[p:p + 32]), int(starts[i]), int(ends[i])

    def upper_bound_start(self, eid: int, query_end: int) -> int:
        """Return number of entries whose start <= query_end."""
        return bisect_right(self.starts[eid], int(query_end))

    def linear_upper_bound_start(self, eid: int, query_end: int) -> int:
        """Naive query boundary lookup: scan starts from the beginning.

        Only the SMT baseline uses this method.  Stop at the first start
        greater than query_end; the pre-existing binary-search method remains
        available for the original baseline.
        """
        count = 0
        for start in self.starts[eid]:
            if int(start) > int(query_end):
                break
            count += 1
        return count

    def validate_start_order(self) -> None:
        """Fail if any edge list is not nondecreasing by start time."""
        for eid, count in enumerate(self.entry_counts):
            starts = self.starts[eid]
            for i in range(1, count):
                if starts[i - 1] > starts[i]:
                    raise RuntimeError(f"edge {eid} Entry list is not start-time ordered")

    def total_entries(self) -> int:
        return sum(self.entry_counts)

    def unique_trajectory_counts(self) -> list[int]:
        """Return the number of distinct trajectories stored on every edge.

        A trajectory can traverse the same edge more than once, producing more
        than one Entry.  Spatial partition weights count that trajectory once
        for the edge rather than treating repeated traversals as independent
        trajectories.
        """
        counts = [0] * len(self.entry_counts)
        for eid, entry_count in enumerate(self.entry_counts):
            if entry_count == 0:
                continue
            raw = self.trajectory_id_bytes[eid]
            counts[eid] = len(
                {
                    bytes(raw[pos:pos + 32])
                    for pos in range(0, entry_count * 32, 32)
                }
            )
        return counts
