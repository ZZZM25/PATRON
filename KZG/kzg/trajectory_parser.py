from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from kzg.config import SLOT_SECONDS,SLOTS_PER_DAY
from kzg.models import SegmentRecord
from kzg.road_network import RoadNetwork
from kzg.time_slots import day_start_timestamp,infer_day,slots_covering_interval
from kzg.trajectory_id import encode_daily_trajectory_id


def load_trajectory_json(path: str | Path) -> list:
    """读取轨迹JSON文件。"""
    with Path(path).open("r",encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data,list):
        raise ValueError("top-level trajectory JSON must be a list")

    return data


def _trajectory_parts(trajectory: object,traj_id: int) -> tuple[list,list]:
    """取出一条轨迹中的节点对和带时间戳的轨迹段。"""
    if not isinstance(trajectory,list) or len(trajectory) < 3:
        raise ValueError(f"trajectory {traj_id} must contain at least 3 parts")

    node_pairs = trajectory[1]
    segments = trajectory[2]

    if not isinstance(node_pairs,list) or not isinstance(segments,list):
        raise ValueError(f"trajectory {traj_id} has invalid node pairs or segments")

    if len(node_pairs) != len(segments):
        raise ValueError(f"trajectory {traj_id} has {len(node_pairs)} node pairs but {len(segments)} segments")

    for index,(node_pair,points) in enumerate(zip(node_pairs,segments)):
        if not isinstance(node_pair,list) or len(node_pair) != 2:
            raise ValueError(f"trajectory {traj_id}, segment {index}: invalid node pair")

        if not isinstance(points,list) or not points:
            raise ValueError(f"trajectory {traj_id}, segment {index}: empty trajectory segment")

        if any(not isinstance(point,list) or len(point) < 3 for point in points):
            raise ValueError(f"trajectory {traj_id}, segment {index}: invalid trajectory point")

    return node_pairs,segments


def _parse_trajectory_json_segments(trajectory_path: str | Path,road: RoadNetwork,day: str | None = None,parallel_policy: str = "error") -> tuple[str,dict[int,dict[int,list[SegmentRecord]]],int,int]:
    """Parse legacy JSON trajectories and assign date+ordinal trajectory IDs."""
    trajectories = load_trajectory_json(trajectory_path)
    slot_records = defaultdict(lambda: defaultdict(list))
    detected_day = day
    day_start_ts = day_start_timestamp(day) if day is not None else None
    segment_count = 0

    for ordinal, trajectory in enumerate(trajectories, start=1):
        node_pairs, segments = _trajectory_parts(trajectory, ordinal)
        if not segments:
            continue
        first_ts = int(segments[0][0][0])
        row_day = infer_day(first_ts)
        if detected_day is None:
            detected_day = row_day
            day_start_ts = day_start_timestamp(detected_day)
        if row_day != detected_day:
            raise ValueError(f"trajectory {ordinal} belongs to {row_day}, expected {detected_day}")
        traj_id = encode_daily_trajectory_id(detected_day, ordinal)
        day_end_ts = day_start_ts + SLOTS_PER_DAY * SLOT_SECONDS

        for index, (node_pair, points) in enumerate(zip(node_pairs, segments)):
            start_node = int(node_pair[0])
            end_node = int(node_pair[1])
            edge_id = road.resolve_edge(start_node, end_node, parallel_policy)
            start_ts = int(points[0][0])
            end_ts = int(segments[index + 1][0][0]) if index + 1 < len(segments) else int(points[-1][0])
            if end_ts <= start_ts:
                end_ts = start_ts + 1
            if infer_day(start_ts) != detected_day or start_ts < day_start_ts or end_ts > day_end_ts:
                raise ValueError(f"trajectory {traj_id}, segment {index}: interval is outside UTC day {detected_day}")
            record = SegmentRecord(traj_id, start_ts, end_ts)
            for slot in slots_covering_interval(day_start_ts, start_ts, end_ts):
                slot_records[edge_id][slot].append(record)
            segment_count += 1

    if detected_day is None:
        raise ValueError("cannot infer day from an empty trajectory file")
    result = {edge_id: dict(slots) for edge_id, slots in slot_records.items()}
    return detected_day, result, segment_count, len(trajectories)

def _csv_trajectory_rows(path: str | Path):
    """Read source trajectory rows in deterministic file order.

    The source trajectory_id is treated only as an input label.  The
    authenticated trajectory ID is reassigned later as date + daily ordinal.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("trajectory CSV is missing a header")
        required_fields = {"trajectory_id", "trajectory_data"}
        missing_fields = required_fields.difference(reader.fieldnames)
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            raise ValueError(f"trajectory CSV is missing required columns: {names}")
        seen_source_ids = set()
        for line_number, row in enumerate(reader, start=2):
            if not row or all(value is None or not str(value).strip() for value in row.values()):
                continue
            source_id = str(row.get("trajectory_id", "")).strip()
            if not source_id:
                raise ValueError(f"trajectory CSV line {line_number}: empty trajectory_id")
            if source_id in seen_source_ids:
                raise ValueError(f"trajectory CSV line {line_number}: duplicate trajectory_id {source_id}")
            seen_source_ids.add(source_id)
            try:
                segments = json.loads(row["trajectory_data"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"trajectory CSV line {line_number}: trajectory_data is not valid JSON") from exc
            if not isinstance(segments, list):
                raise ValueError(f"trajectory CSV line {line_number}: trajectory_data must be a list")
            yield line_number, source_id, segments

def _csv_segment_parts(segment: object,line_number: int,traj_id: int,segment_index: int) -> tuple[int,int,int,int,int]:
    """解析CSV轨迹中的[edge_id,start_node,start_ts,end_node,end_ts]五元组。"""
    if not isinstance(segment,list) or len(segment) != 5:
        raise ValueError(f"trajectory CSV line {line_number}, trajectory {traj_id}, segment {segment_index}: expected 5 fields")

    try:
        edge_id,start_node,start_ts,end_node,end_ts = (int(value) for value in segment)
    except (TypeError,ValueError) as exc:
        raise ValueError(f"trajectory CSV line {line_number}, trajectory {traj_id}, segment {segment_index}: fields must be integers") from exc

    if end_ts < start_ts:
        raise ValueError(f"trajectory CSV line {line_number}, trajectory {traj_id}, segment {segment_index}: end timestamp must not be earlier than start timestamp")

    if end_ts == start_ts:
        end_ts = start_ts + 1

    return edge_id,start_node,start_ts,end_node,end_ts


def _validated_csv_edge_id(road: RoadNetwork,declared_edge_id: int,start_node: int,end_node: int,line_number: int,traj_id: int,segment_index: int) -> int:
    """验证CSV声明的边ID确实连接给定的两个无向节点。"""
    candidate_edge_ids = road.edge_ids_between(start_node,end_node)

    if not candidate_edge_ids:
        raise ValueError(f"trajectory CSV line {line_number}, trajectory {traj_id}, segment {segment_index}: no road edge between nodes {start_node} and {end_node}")

    if declared_edge_id not in candidate_edge_ids:
        raise ValueError(f"trajectory CSV line {line_number}, trajectory {traj_id}, segment {segment_index}: declared edge {declared_edge_id} does not match nodes {start_node} and {end_node}; expected one of {candidate_edge_ids}")

    return declared_edge_id


def _parse_trajectory_csv_segments(trajectory_path: str | Path,road: RoadNetwork,day: str | None = None) -> tuple[str,dict[int,dict[int,list[SegmentRecord]]],int,int]:
    """Parse processed 5-tuple trajectories using UTC Unix timestamps.

    Authenticated trajectory IDs are deterministic: YYYYMMDD * 1,000,000 +
    the 1-based row ordinal within that UTC-day dataset.
    """
    slot_records = defaultdict(lambda: defaultdict(list))
    detected_day = day
    day_start_ts = day_start_timestamp(day) if day is not None else None
    segment_count = 0
    trajectory_count = 0

    for line_number, source_id, segments in _csv_trajectory_rows(trajectory_path):
        if not segments:
            raise ValueError(f"trajectory CSV line {line_number}: trajectory has no segments")
        first = _csv_segment_parts(segments[0], line_number, 0, 0)
        row_day = infer_day(first[2])
        if detected_day is None:
            detected_day = row_day
            day_start_ts = day_start_timestamp(detected_day)
        if row_day != detected_day:
            raise ValueError(
                f"trajectory CSV line {line_number}: source trajectory {source_id} belongs to {row_day}, expected {detected_day}"
            )
        trajectory_count += 1
        traj_id = encode_daily_trajectory_id(detected_day, trajectory_count)
        day_end_ts = day_start_ts + SLOTS_PER_DAY * SLOT_SECONDS

        for segment_index, segment in enumerate(segments):
            declared_edge_id, start_node, start_ts, end_node, end_ts = _csv_segment_parts(
                segment, line_number, traj_id, segment_index
            )
            edge_id = _validated_csv_edge_id(
                road, declared_edge_id, start_node, end_node, line_number, traj_id, segment_index
            )
            if infer_day(start_ts) != detected_day or start_ts < day_start_ts or end_ts > day_end_ts:
                raise ValueError(
                    f"trajectory CSV line {line_number}, trajectory {traj_id}, segment {segment_index}: interval is outside UTC day {detected_day}"
                )
            record = SegmentRecord(traj_id, start_ts, end_ts)
            for slot in slots_covering_interval(day_start_ts, start_ts, end_ts):
                slot_records[edge_id][slot].append(record)
            segment_count += 1

    if detected_day is None:
        raise ValueError("cannot infer day from an empty trajectory file")
    result = {edge_id: dict(slots) for edge_id, slots in slot_records.items()}
    return detected_day, result, segment_count, trajectory_count

def parse_trajectory_segments(trajectory_path: str | Path,road: RoadNetwork,day: str | None = None,parallel_policy: str = "error") -> tuple[str,dict[int,dict[int,list[SegmentRecord]]],int,int]:
    """根据文件扩展名自动解析原有JSON或五元组CSV轨迹。"""
    suffix = Path(trajectory_path).suffix.lower()

    if suffix == ".json":
        return _parse_trajectory_json_segments(trajectory_path,road,day,parallel_policy)

    if suffix == ".csv":
        return _parse_trajectory_csv_segments(trajectory_path,road,day)

    raise ValueError(f"unsupported trajectory file format: {suffix or '<none>'}; expected .json or .csv")
