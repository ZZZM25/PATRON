"""Expand raw trajectories onto the split network and write CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from network.edge_builder import NewEdge, SplitNetwork
from network.reader import NetworkInput

from .edge_time import allocate_boundary_times, interval_from_gps_group
from .json_reader import RawTrajectory


@dataclass(frozen=True)
class TraversedPiece:
    new_edge_id: int
    start_node: int
    end_node: int
    fraction_start: Fraction
    fraction_end: Fraction


def _traversal_pieces(
    pieces: list[NewEdge], stored_forward: bool
) -> list[TraversedPiece]:
    if stored_forward:
        return [
            TraversedPiece(
                new_edge_id=edge.edge_id,
                start_node=edge.start_node,
                end_node=edge.end_node,
                fraction_start=edge.t_start,
                fraction_end=edge.t_end,
            )
            for edge in pieces
        ]
    return [
        TraversedPiece(
            new_edge_id=edge.edge_id,
            start_node=edge.end_node,
            end_node=edge.start_node,
            fraction_start=1 - edge.t_end,
            fraction_end=1 - edge.t_start,
        )
        for edge in reversed(pieces)
    ]


def _trajectory_date(first_timestamp: int, timezone_offset_hours: int) -> str:
    dt = datetime.fromtimestamp(first_timestamp, tz=timezone.utc)
    dt += timedelta(hours=timezone_offset_hours)
    return dt.strftime("%Y%m%d")


def _first_timestamp(trajectory: RawTrajectory) -> int:
    for group in trajectory.gps_groups:
        if group:
            return int(group[0][0])
    raise ValueError(f"trajectory {trajectory.source_index} contains no GPS timestamp")


class TrajectoryProcessor:
    def __init__(
        self,
        network: NetworkInput,
        split_network: SplitNetwork,
        timezone_offset_hours: int = 8,
        trajectory_index_start: int = 0,
        fail_on_unmapped_pair: bool = True,
    ) -> None:
        self.network = network
        self.split_network = split_network
        self.timezone_offset_hours = timezone_offset_hours
        self.trajectory_index_start = trajectory_index_start
        self.fail_on_unmapped_pair = fail_on_unmapped_pair
        self.unmapped_pairs: list[dict[str, object]] = []

    def process_one(self, trajectory: RawTrajectory) -> tuple[str, list[list[int]]]:
        records: list[list[int]] = []
        previous_end: int | None = None
        for pair_index, (pair, gps_group) in enumerate(
            zip(trajectory.node_pairs, trajectory.gps_groups)
        ):
            start_node, end_node = pair
            try:
                original_edge, stored_forward = self.network.resolve_traversal(
                    start_node, end_node
                )
            except (KeyError, ValueError) as exc:
                problem = {
                    "trajectory_index": trajectory.source_index,
                    "pair_index": pair_index,
                    "start_node": start_node,
                    "end_node": end_node,
                    "error": str(exc),
                }
                self.unmapped_pairs.append(problem)
                if self.fail_on_unmapped_pair:
                    raise ValueError(problem) from exc
                continue

            pieces = self.split_network.pieces_by_original_edge[original_edge.edge_id]
            traversed = _traversal_pieces(pieces, stored_forward)
            interval = interval_from_gps_group(gps_group)
            fractions = [piece.fraction_start for piece in traversed]
            fractions.append(traversed[-1].fraction_end)
            times = allocate_boundary_times(interval, fractions)

            for piece_index, piece in enumerate(traversed):
                if previous_end is not None and piece.start_node != previous_end:
                    raise ValueError(
                        f"trajectory {trajectory.source_index}: broken adjacency at "
                        f"pair {pair_index}, generated edge {piece.new_edge_id}"
                    )
                record = [
                    piece.new_edge_id,
                    piece.start_node,
                    times[piece_index],
                    piece.end_node,
                    times[piece_index + 1],
                ]
                if record[4] < record[2]:
                    raise ValueError(f"negative generated duration: {record}")
                records.append(record)
                previous_end = piece.end_node

        output_index = self.trajectory_index_start + trajectory.source_index
        date = _trajectory_date(
            _first_timestamp(trajectory), self.timezone_offset_hours
        )
        trajectory_id = f"{output_index}_{date}"
        return trajectory_id, records

    def process_all(
        self, trajectories: Iterable[RawTrajectory]
    ) -> list[tuple[str, list[list[int]]]]:
        return [self.process_one(trajectory) for trajectory in trajectories]


def write_trajectory_csv(
    rows: Iterable[tuple[str, list[list[int]]]], output_path: str | Path
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "trajectory_data"])
        for trajectory_id, records in rows:
            writer.writerow([trajectory_id, json.dumps(records, separators=(",", ":"))])
    return str(output_path)
