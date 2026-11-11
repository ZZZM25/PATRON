"""Reader for the supplied four-part nested trajectory JSON format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RawTrajectory:
    source_index: int
    node_sequence: list[int]
    node_pairs: list[tuple[int, int]]
    gps_groups: list[list[tuple[int, float, float]]]
    simplified_nodes: list[int]


def _parse_one(raw: Any, source_index: int) -> RawTrajectory:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(
            f"trajectory {source_index}: expected [node_sequence, node_pairs, "
            "gps_groups, simplified_nodes]"
        )
    node_sequence_raw, pairs_raw, gps_raw, simplified_raw = raw
    if not isinstance(node_sequence_raw, list) or not isinstance(pairs_raw, list):
        raise ValueError(f"trajectory {source_index}: node sections must be lists")
    if len(node_sequence_raw) != len(pairs_raw) + 1:
        raise ValueError(
            f"trajectory {source_index}: node_sequence length must equal "
            "len(node_pairs)+1"
        )
    if len(pairs_raw) != len(gps_raw):
        raise ValueError(
            f"trajectory {source_index}: node_pairs and gps_groups lengths differ"
        )

    node_sequence = [int(value) for value in node_sequence_raw]
    node_pairs: list[tuple[int, int]] = []
    for pair_index, pair in enumerate(pairs_raw):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"trajectory {source_index}, pair {pair_index}: expected two nodes")
        expected = (node_sequence[pair_index], node_sequence[pair_index + 1])
        actual = (int(pair[0]), int(pair[1]))
        if actual != expected:
            raise ValueError(
                f"trajectory {source_index}, pair {pair_index}: pair {actual} "
                f"does not match node sequence {expected}"
            )
        node_pairs.append(actual)

    gps_groups: list[list[tuple[int, float, float]]] = []
    for pair_index, group in enumerate(gps_raw):
        if not isinstance(group, list) or not group:
            raise ValueError(f"trajectory {source_index}, pair {pair_index}: empty GPS group")
        parsed_group: list[tuple[int, float, float]] = []
        previous_time: int | None = None
        for point_index, point in enumerate(group):
            if not isinstance(point, list) or len(point) < 3:
                raise ValueError(
                    f"trajectory {source_index}, pair {pair_index}, point {point_index}: "
                    "expected [timestamp, lat, lon]"
                )
            timestamp = int(point[0])
            latitude = float(point[1])
            longitude = float(point[2])
            if previous_time is not None and timestamp < previous_time:
                raise ValueError(
                    f"trajectory {source_index}, pair {pair_index}: GPS timestamps decrease"
                )
            previous_time = timestamp
            parsed_group.append((timestamp, latitude, longitude))
        gps_groups.append(parsed_group)

    if not isinstance(simplified_raw, list):
        raise ValueError(f"trajectory {source_index}: simplified node section is not a list")
    simplified_nodes = [int(value) for value in simplified_raw]
    return RawTrajectory(
        source_index=source_index,
        node_sequence=node_sequence,
        node_pairs=node_pairs,
        gps_groups=gps_groups,
        simplified_nodes=simplified_nodes,
    )


def load_trajectories(path: str | Path) -> list[RawTrajectory]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("trajectory JSON root must be a list")
    return [_parse_one(item, index) for index, item in enumerate(raw)]
