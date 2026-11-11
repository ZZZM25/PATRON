from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from gmpy2 import mpz

import rsa_query as proof
from run_rsg_50m_experiment_matrix import resolve_spatial_range


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day_seconds(
    date_str: str, start_timestamp: int, end_timestamp: int
) -> tuple[int, int]:
    """Convert a UTC [start, end) interval to inclusive seconds in one day."""
    start_timestamp = int(start_timestamp)
    end_timestamp = int(end_timestamp)
    if end_timestamp <= start_timestamp:
        raise ValueError("查询结束时间必须晚于开始时间")
    day_start = int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    if start_timestamp < day_start or end_timestamp > day_start + 86400:
        raise ValueError("当前单日索引只支持同一 UTC 日期内的查询")
    return start_timestamp - day_start, end_timestamp - day_start - 1


def _continuous_time_slots(
    time_axis: dict, start_timestamp: int, end_timestamp: int
) -> tuple[int, int]:
    """Map a UTC Unix [start,end) interval to inclusive bimonthly slots."""
    start_timestamp = int(start_timestamp)
    end_timestamp = int(end_timestamp)
    origin = int(time_axis["start_timestamp"])
    axis_end = int(time_axis["end_timestamp_exclusive"])
    unit_seconds = int(time_axis["time_unit_seconds"])
    slot_count = int(time_axis["time_slot_count"])
    if unit_seconds <= 0 or slot_count <= 0:
        raise ValueError("非法时间轴参数")
    if end_timestamp <= start_timestamp:
        raise ValueError("查询结束时间必须晚于开始时间")
    if start_timestamp < origin or end_timestamp > axis_end:
        raise ValueError("查询时间超出当前双月索引范围")
    qs = (start_timestamp - origin) // unit_seconds
    qe = (end_timestamp - 1 - origin) // unit_seconds
    if qs < 0 or qe < qs or qe >= slot_count:
        raise ValueError("查询时间槽超出索引范围")
    return qs, qe


def _query_time_units(
    date_str: str,
    start_timestamp: int,
    end_timestamp: int,
    time_axis: dict | None,
) -> tuple[int, int]:
    if time_axis is None:
        return _day_seconds(date_str, start_timestamp, end_timestamp)
    return _continuous_time_slots(time_axis, start_timestamp, end_timestamp)


def load_client_context(
    dataset_dir: Path, trusted_manifest_path: Path, expected_date: str
) -> dict:
    """Load only owner-pinned public parameters; never load the server ADS."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads(Path(trusted_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("date") != expected_date:
        raise ValueError("可信清单日期不匹配")
    records = {item["role"]: item for item in manifest.get("files", [])}
    for role, name in (
        ("grid_metadata", "grid_metadata.json"),
        ("edge_cells", "edge_cells.csv"),
    ):
        path = dataset_dir / name
        record = records.get(role)
        if record is None or record.get("name") != name:
            raise ValueError(f"可信清单缺少 {role}")
        if int(record.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"{role} 文件大小不匹配")
        if record.get("sha256") != _sha256_file(path):
            raise ValueError(f"{role} 文件哈希不匹配")

    rsa = manifest["rsa"]
    return {
        "rsa": {
            "n": mpz(rsa["n"]),
            "g": mpz(rsa["g"]),
            "hprime_bits": int(rsa["hprime_bits"]),
            "depth": int(rsa["depth"]),
            "format_version": int(rsa["format_version"]),
            "token_bytes": int(rsa["opaque_token_bytes"]),
            "base": int(rsa.get("base", 2)),
            "day_seconds": int(rsa.get("day_seconds", 86400)),
        },
        "trusted_root": {"data_ac": mpz(manifest["data_root"]["data_ac"])},
        "time_axis": manifest.get("time_axis"),
        "data": {},
    }


def query_spatiotemporal_range(
    dataset_dir: Path,
    index_file: Path,
    date_str: str,
    spatial_bounds: Sequence[float],
    start_timestamp: int,
    end_timestamp: int,
    time_axis: dict | None = None,
) -> dict:
    """Untrusted-server operation: range in, matching results and VO out."""
    if len(spatial_bounds) != 4:
        raise ValueError("spatial_bounds 必须为 [min_lon,min_lat,max_lon,max_lat]")
    bounds = [float(value) for value in spatial_bounds]
    spatial, _metadata = resolve_spatial_range(
        Path(dataset_dir), bounds[0], bounds[1], bounds[2], bounds[3]
    )
    qs, qe = _query_time_units(
        date_str, start_timestamp, end_timestamp, time_axis
    )
    server_db = proof.load_binary_selected_edges(
        str(index_file), date_str, set(spatial["edge_ids"])
    )
    response = proof.query_edges_by_range(
        server_db, date_str, spatial["edge_ids"], qs, qe
    )
    response["query"].update({
        "spatial_bounds": bounds,
        "start_timestamp": int(start_timestamp),
        "end_timestamp": int(end_timestamp),
    })
    response["spatial_cover"] = {
        key: spatial[key]
        for key in (
            "cell_col_range",
            "cell_row_range",
            "covered_cell_count",
            "road_edge_count",
        )
    }
    return response


def verify_spatiotemporal_response(
    client_db: dict,
    dataset_dir: Path,
    response: dict,
    date_str: str,
    spatial_bounds: Sequence[float],
    start_timestamp: int,
    end_timestamp: int,
    time_axis: dict | None = None,
) -> dict:
    """Client operation: independently derive query coverage and verify results + VO."""
    if len(spatial_bounds) != 4:
        raise ValueError("spatial_bounds 必须包含四个值")
    bounds = [float(value) for value in spatial_bounds]
    spatial, _metadata = resolve_spatial_range(
        Path(dataset_dir), bounds[0], bounds[1], bounds[2], bounds[3]
    )
    qs, qe = _query_time_units(
        date_str, start_timestamp, end_timestamp, time_axis
    )
    actual_query = response.get("query", {})
    spatial_binding_ok = (
        actual_query.get("spatial_bounds") == bounds
        and int(actual_query.get("start_timestamp", -1)) == int(start_timestamp)
        and int(actual_query.get("end_timestamp", -1)) == int(end_timestamp)
    )
    verdict = proof.verify_edges_range_response(
        client_db,
        response,
        {
            "date": date_str,
            "edge_ids": spatial["edge_ids"],
            "qs": qs,
            "qe": qe,
        },
    )
    verdict["spatial_binding_ok"] = bool(spatial_binding_ok)
    verdict["all_ok"] = bool(verdict["all_ok"] and spatial_binding_ok)
    return verdict
