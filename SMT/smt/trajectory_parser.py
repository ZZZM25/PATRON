from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ParsedTrajectory, TrajectorySegment
from .trajectory_id import compute_trajectory_id


def load_relaxed_json(path: str | Path):
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    obj, end = json.JSONDecoder().raw_decode(stripped)
    trailing = stripped[end:].strip()
    if trailing and set(trailing) != {"]"}:
        raise ValueError(f"unexpected trailing data in {path}: {trailing[:100]!r}")
    return obj


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def collect_files(globs: list[str], config_dir: Path) -> list[Path]:
    out: list[Path] = []
    import glob
    for pattern in globs:
        p = Path(pattern)
        pattern_abs = str(p if p.is_absolute() else (config_dir / p))
        out.extend(Path(x) for x in glob.glob(pattern_abs))
    unique = sorted(set(x.resolve() for x in out), key=natural_key)
    if not unique:
        raise FileNotFoundError(f"no trajectory files match: {globs}")
    return unique


def parse_trajectory(trajectory, road) -> ParsedTrajectory:
    if not isinstance(trajectory, list) or len(trajectory) < 3:
        raise ValueError("trajectory must be a list with at least 3 fields")
    node_pairs = trajectory[1]
    point_groups = trajectory[2]
    if len(node_pairs) != len(point_groups):
        raise ValueError("node_pairs and point_groups size mismatch")
    if not node_pairs:
        raise ValueError("empty trajectory")

    eid_path: list[int] = []
    segments: list[TrajectorySegment] = []
    previous_group_end: int | None = None

    for i, (pair, points) in enumerate(zip(node_pairs, point_groups)):
        if len(pair) < 2 or not points:
            raise ValueError(f"malformed trajectory segment {i}")
        u, v = int(pair[0]), int(pair[1])
        current_first = int(points[0][0])
        current_last = int(points[-1][0])
        start = current_first if i == 0 else previous_group_end
        if start is None:
            raise RuntimeError("missing previous segment end")
        end = current_last
        if end < start:
            raise ValueError(f"segment time decreases: {start}>{end}")
        eid = road.resolve_edge(u, v)
        eid_path.append(eid)
        segments.append(TrajectorySegment(u, int(start), v, int(end)))
        previous_group_end = current_last

    trajectory_start = segments[0].start_time
    trajectory_end = segments[-1].end_time
    trajectory_id = compute_trajectory_id(eid_path, trajectory_start, trajectory_end)
    return ParsedTrajectory(trajectory_id, eid_path, segments)


def parse_cell_partitioned_trajectory(trajectory, source_road, partition) -> ParsedTrajectory:
    """Parse a Chengdu/Xi'an record against a cell-partitioned road.

    Input validation and source-segment time semantics intentionally match
    :func:`parse_trajectory`.  The only change is that every resolved source
    edge is expanded into the ordered sub-edges produced by ``partition``.
    The trajectory ID is then computed with the same formula over the new EID
    path, because the partitioned road is the road network for this variant.
    """
    from .cell_partition import expand_trajectory_segment

    if not isinstance(trajectory, list) or len(trajectory) < 3:
        raise ValueError("trajectory must be a list with at least 3 fields")
    node_pairs = trajectory[1]
    point_groups = trajectory[2]
    if len(node_pairs) != len(point_groups):
        raise ValueError("node_pairs and point_groups size mismatch")
    if not node_pairs:
        raise ValueError("empty trajectory")

    eid_path: list[int] = []
    segments: list[TrajectorySegment] = []
    previous_group_end: int | None = None

    for i, (pair, points) in enumerate(zip(node_pairs, point_groups)):
        if len(pair) < 2 or not points:
            raise ValueError(f"malformed trajectory segment {i}")
        u, v = int(pair[0]), int(pair[1])
        current_first = int(points[0][0])
        current_last = int(points[-1][0])
        start = current_first if i == 0 else previous_group_end
        if start is None:
            raise RuntimeError("missing previous segment end")
        end = current_last
        if end < start:
            raise ValueError(f"segment time decreases: {start}>{end}")
        source_eid = source_road.resolve_edge(u, v)
        expanded_eids, expanded_segments = expand_trajectory_segment(
            partition,
            source_eid,
            u,
            int(start),
            v,
            int(end),
        )
        eid_path.extend(expanded_eids)
        segments.extend(expanded_segments)
        previous_group_end = current_last

    trajectory_start = segments[0].start_time
    trajectory_end = segments[-1].end_time
    trajectory_id = compute_trajectory_id(eid_path, trajectory_start, trajectory_end)
    return ParsedTrajectory(trajectory_id, eid_path, segments)
