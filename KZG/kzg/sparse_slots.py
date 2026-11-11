from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import date

from kzg.config import SLOT_SECONDS,SLOTS_PER_DAY
from kzg.field import fr_from_bytes,hash_to_field
from kzg.time_slots import complete_slots_for_interval


SPARSE_SLOT_ENCODING = "sparse-nonempty-v1"
_SLOT_HEADER_STRUCT = struct.Struct(">II")
_RECORD_STRUCT = struct.Struct(">QIqq")


@dataclass(frozen=True,slots=True)
class SparseSlotReconstructionContext:
    """Query-wide constants shared by every authenticated edge."""

    expected_slot_set: frozenset[int]
    day_bytes: bytes
    day_start_ts: int
    day_end_ts: int
    slot_seconds: int
    slot_count: int


def prepare_sparse_slot_reconstruction(
    expected_slots: list[int],
    day: str,
    base_evaluation_values: list[int],
    day_start_ts: int,
    day_end_ts: int,
    query_start_ts: int | None = None,
    query_end_ts: int | None = None,
    slot_seconds: int = SLOT_SECONDS,
    slot_count: int = SLOTS_PER_DAY,
) -> SparseSlotReconstructionContext:
    """Validate query-wide sparse-slot inputs once, before the edge loop."""
    if (query_start_ts is None) != (query_end_ts is None):
        raise ValueError("query start and end timestamps must be provided together")
    if slot_seconds <= 0 or slot_count <= 0:
        raise ValueError("slot configuration must be positive")
    if expected_slots and max(expected_slots) >= len(base_evaluation_values):
        raise ValueError("base evaluation vector is smaller than the requested slot range")
    if query_start_ts is not None:
        complete_slots = complete_slots_for_interval(
            int(day_start_ts),
            int(query_start_ts),
            int(query_end_ts),
            int(slot_seconds),
            int(slot_count),
        )
        if [int(slot) for slot in expected_slots] != complete_slots:
            raise ValueError("expected slots do not exactly match the complete query slots")
    try:
        parsed_day = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError("day must use YYYY-MM-DD format") from exc
    if parsed_day.isoformat() != day:
        raise ValueError("day must use YYYY-MM-DD format")
    return SparseSlotReconstructionContext(
        expected_slot_set=frozenset(expected_slots),
        day_bytes=day.encode("ascii"),
        day_start_ts=int(day_start_ts),
        day_end_ts=int(day_end_ts),
        slot_seconds=int(slot_seconds),
        slot_count=int(slot_count),
    )


def reconstruct_sparse_slots_prepared(
    slot_packages: object,
    edge_id: int,
    context: SparseSlotReconstructionContext,
) -> tuple[dict[int,int],set[int]]:
    """Validate one edge using query-wide constants prepared once."""
    if not isinstance(slot_packages,list):
        raise ValueError(f"slot packages for edge {edge_id} must be a list")
    if not 0 <= edge_id <= 0xFFFFFFFF:
        raise ValueError("edge ID must fit an unsigned 32-bit integer")

    normalized_slot_packages = []
    previous_slot = None
    for slot_package in slot_packages:
        if not isinstance(slot_package,dict):
            raise ValueError(f"malformed sparse slot package for edge {edge_id}")
        slot = int(slot_package["slot"])
        if previous_slot is not None and slot <= previous_slot:
            raise ValueError(
                f"sparse slots for edge {edge_id} must be strictly increasing and unique"
            )
        previous_slot = slot
        if slot not in context.expected_slot_set:
            raise ValueError(
                f"sparse slots for edge {edge_id} contain a slot outside the query interval"
            )
        if not 0 <= slot < context.slot_count:
            raise ValueError(f"invalid minute slot: {slot}")
        normalized_slot_packages.append((slot,slot_package))

    claimed_value_by_slot: dict[int,int] = {}
    derived_trajectory_ids = set()
    day_start_ts = context.day_start_ts
    day_end_ts = context.day_end_ts

    for slot,slot_package in normalized_slot_packages:
        raw_records = slot_package["records"]
        if not isinstance(raw_records,list):
            raise ValueError(f"records for edge {edge_id}, slot {slot} must be a list")
        if not raw_records:
            raise ValueError(
                f"sparse slot package for edge {edge_id}, slot {slot} "
                "must not contain an empty record list"
            )

        current_slot_start = day_start_ts + slot * context.slot_seconds
        current_slot_end = current_slot_start + context.slot_seconds
        encoded = bytearray(context.day_bytes)
        encoded.extend(_SLOT_HEADER_STRUCT.pack(slot,len(raw_records)))
        previous_record = None

        for raw_record in raw_records:
            if not isinstance(raw_record,list) or len(raw_record) != 3:
                raise ValueError(
                    f"malformed trajectory record on edge {edge_id}, slot {slot}"
                )
            trajectory_id = int(raw_record[0])
            start_timestamp = int(raw_record[1])
            end_timestamp = int(raw_record[2])
            record_key = (trajectory_id,start_timestamp,end_timestamp)
            if previous_record is not None and record_key <= previous_record:
                raise ValueError(
                    f"records for edge {edge_id}, slot {slot} are not canonical"
                )
            previous_record = record_key
            if trajectory_id < 0:
                raise ValueError(f"negative trajectory ID on edge {edge_id}, slot {slot}")
            if end_timestamp <= start_timestamp:
                raise ValueError(f"invalid trajectory interval on edge {edge_id}, slot {slot}")
            if start_timestamp < day_start_ts or end_timestamp > day_end_ts:
                raise ValueError(
                    f"trajectory interval is outside the anchored day on edge {edge_id}, slot {slot}"
                )
            if not (
                start_timestamp < current_slot_end
                and end_timestamp > current_slot_start
            ):
                raise ValueError(
                    f"trajectory record does not belong to edge {edge_id}, slot {slot}"
                )
            # Complete-slot queries include every authenticated record in the
            # selected slot. There is intentionally no sub-slot filtering.
            derived_trajectory_ids.add(trajectory_id)
            encoded.extend(
                _RECORD_STRUCT.pack(
                    trajectory_id,edge_id,start_timestamp,end_timestamp
                )
            )

        recomputed_value = hash_to_field(bytes(encoded))
        if "value" not in slot_package:
            # Binary VO v2 omits this redundant 32-byte value. The verifier
            # already derives it from the canonical records before using it in
            # the Fiat-Shamir transcript and KZG equation.
            claimed_value = recomputed_value
        else:
            raw_value = slot_package["value"]
            if not isinstance(raw_value,str):
                raise ValueError(
                    f"value for edge {edge_id}, slot {slot} must be a hexadecimal string"
                )
            try:
                encoded_value = bytes.fromhex(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"value for edge {edge_id}, slot {slot} is not valid hexadecimal data"
                ) from exc
            if len(encoded_value) != 32:
                raise ValueError(
                    f"value for edge {edge_id}, slot {slot} must contain 32 bytes"
                )
            claimed_value = fr_from_bytes(encoded_value)
            if recomputed_value != claimed_value:
                raise ValueError(f"cell value mismatch for edge {edge_id}, slot {slot}")
        claimed_value_by_slot[slot] = claimed_value

    return claimed_value_by_slot,derived_trajectory_ids


def reconstruct_sparse_slots(
    slot_packages: object,
    expected_slots: list[int],
    day: str,
    edge_id: int,
    base_evaluation_values: list[int],
    day_start_ts: int,
    day_end_ts: int,
    query_start_ts: int | None = None,
    query_end_ts: int | None = None,
    slot_seconds: int = SLOT_SECONDS,
    slot_count: int = SLOTS_PER_DAY,
) -> tuple[dict[int,int],set[int]]:
    """Validate nonempty slots and return only their claimed field values.

    Empty queried slots are authenticated by the shared base vector and are
    filled lazily by the verifier. Keeping this result sparse avoids creating
    one Python dictionary entry for every ``edge x queried slot`` pair.
    """
    context = prepare_sparse_slot_reconstruction(
        expected_slots,
        day,
        base_evaluation_values,
        day_start_ts,
        day_end_ts,
        query_start_ts,
        query_end_ts,
        slot_seconds,
        slot_count,
    )
    return reconstruct_sparse_slots_prepared(slot_packages,edge_id,context)
