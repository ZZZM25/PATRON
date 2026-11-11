from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, order=True)
class CellRect:
    lon_index_min: int
    lon_index_max: int
    lat_index_min: int
    lat_index_max: int

    def normalized(self) -> "CellRect":
        return CellRect(
            min(self.lon_index_min, self.lon_index_max),
            max(self.lon_index_min, self.lon_index_max),
            min(self.lat_index_min, self.lat_index_max),
            max(self.lat_index_min, self.lat_index_max),
        )

    def to_dict(self) -> dict[str, int]:
        normalized = self.normalized()
        return {
            "lon_index_min": normalized.lon_index_min,
            "lon_index_max": normalized.lon_index_max,
            "lat_index_min": normalized.lat_index_min,
            "lat_index_max": normalized.lat_index_max,
            "cell_count": (
                (normalized.lon_index_max - normalized.lon_index_min + 1)
                * (normalized.lat_index_max - normalized.lat_index_min + 1)
            ),
        }


@dataclass(frozen=True)
class CellEdgeRecord:
    edge_id: int
    start_node: int
    end_node: int
    original_edge_id: int
    segment_index: int
    segment_count: int
    start_lon_scaled: int
    start_lat_scaled: int
    end_lon_scaled: int
    end_lat_scaled: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CellEdgeRecord":
        names = (
            "edge_id",
            "start_node",
            "end_node",
            "original_edge_id",
            "segment_index",
            "segment_count",
            "start_lon_scaled",
            "start_lat_scaled",
            "end_lon_scaled",
            "end_lat_scaled",
        )
        missing = [name for name in names if name not in value]
        if missing:
            raise ValueError(f"cell edge record missing fields: {missing}")
        return cls(*(int(value[name]) for name in names))

    def to_dict(self) -> dict[str, int]:
        return {
            "edge_id": self.edge_id,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "original_edge_id": self.original_edge_id,
            "segment_index": self.segment_index,
            "segment_count": self.segment_count,
            "start_lon_scaled": self.start_lon_scaled,
            "start_lat_scaled": self.start_lat_scaled,
            "end_lon_scaled": self.end_lon_scaled,
            "end_lat_scaled": self.end_lat_scaled,
        }


class GeohashGrid:
    WORLD_LON_MIN = -180
    WORLD_LON_MAX = 180
    WORLD_LAT_MIN = -90
    WORLD_LAT_MAX = 90

    def __init__(self, bits: int, coordinate_scale: int) -> None:
        if bits < 1 or bits > 60:
            raise ValueError("bits must be between 1 and 60")
        self.bits = int(bits)
        self.coordinate_scale = int(coordinate_scale)
        self.longitude_bits = (self.bits + 1) // 2
        self.latitude_bits = self.bits // 2
        self.longitude_bins = 1 << self.longitude_bits
        self.latitude_bins = 1 << self.latitude_bits
        self.lon_min_scaled = self.WORLD_LON_MIN * self.coordinate_scale
        self.lon_max_scaled = self.WORLD_LON_MAX * self.coordinate_scale
        self.lat_min_scaled = self.WORLD_LAT_MIN * self.coordinate_scale
        self.lat_max_scaled = self.WORLD_LAT_MAX * self.coordinate_scale
        self.lon_step = Fraction(self.lon_max_scaled - self.lon_min_scaled, self.longitude_bins)
        self.lat_step = Fraction(self.lat_max_scaled - self.lat_min_scaled, self.latitude_bins)

    def prefix_from_indices(self, lon_index: int, lat_index: int) -> str:
        if not 0 <= lon_index < self.longitude_bins:
            raise ValueError("longitude cell index outside grid")
        if not 0 <= lat_index < self.latitude_bins:
            raise ValueError("latitude cell index outside grid")
        out: list[str] = []
        for bit_position in range(max(self.longitude_bits, self.latitude_bits)):
            if bit_position < self.longitude_bits:
                shift = self.longitude_bits - 1 - bit_position
                out.append(str((lon_index >> shift) & 1))
            if bit_position < self.latitude_bits:
                shift = self.latitude_bits - 1 - bit_position
                out.append(str((lat_index >> shift) & 1))
        return "".join(out)

    def indices_from_prefix(self, cell_id: str) -> tuple[int, int]:
        if len(cell_id) != self.bits or any(bit not in "01" for bit in cell_id):
            raise ValueError(f"invalid {self.bits}-bit cell prefix: {cell_id!r}")
        lon_index = 0
        lat_index = 0
        position = 0
        for bit_position in range(max(self.longitude_bits, self.latitude_bits)):
            if bit_position < self.longitude_bits:
                lon_index = (lon_index << 1) | int(cell_id[position])
                position += 1
            if bit_position < self.latitude_bits:
                lat_index = (lat_index << 1) | int(cell_id[position])
                position += 1
        return lon_index, lat_index

    def bounds_from_indices(self, lon_index: int, lat_index: int) -> dict[str, float]:
        self.prefix_from_indices(lon_index, lat_index)
        lon_min = Fraction(self.lon_min_scaled) + self.lon_step * lon_index
        lon_max = lon_min + self.lon_step
        lat_min = Fraction(self.lat_min_scaled) + self.lat_step * lat_index
        lat_max = lat_min + self.lat_step
        scale = self.coordinate_scale
        return {
            "min_lon": float(lon_min / scale),
            "max_lon": float(lon_max / scale),
            "min_lat": float(lat_min / scale),
            "max_lat": float(lat_max / scale),
        }

    @staticmethod
    def _fraction(value: int | float | str | Fraction) -> Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        return Fraction(str(value))

    def _axis_cover(
        self,
        minimum,
        maximum,
        *,
        world_min_degrees: int,
        world_max_degrees: int,
        world_min_scaled: int,
        step_scaled: Fraction,
        bin_count: int,
        axis_name: str,
    ) -> tuple[int, int]:
        minimum = self._fraction(minimum)
        maximum = self._fraction(maximum)
        if minimum > maximum:
            raise ValueError(f"minimum {axis_name} cannot exceed maximum {axis_name}")
        if minimum < world_min_degrees or maximum > world_max_degrees:
            raise ValueError(
                f"{axis_name} range must stay within [{world_min_degrees}, {world_max_degrees}]"
            )
        minimum_scaled = minimum * self.coordinate_scale
        maximum_scaled = maximum * self.coordinate_scale
        min_index = floor((minimum_scaled - world_min_scaled) / step_scaled)
        if minimum == maximum:
            max_index = min_index
        else:
            # Half-open upper boundary for a non-zero interval.  If max is
            # exactly on a grid line, the cell on the other side is not added.
            max_index = ceil((maximum_scaled - world_min_scaled) / step_scaled) - 1
        min_index = max(0, min(bin_count - 1, min_index))
        max_index = max(0, min(bin_count - 1, max_index))
        return min_index, max_index

    def geographic_cover(self, min_lon, max_lon, min_lat, max_lat) -> CellRect:
        lon_min, lon_max = self._axis_cover(
            min_lon,
            max_lon,
            world_min_degrees=self.WORLD_LON_MIN,
            world_max_degrees=self.WORLD_LON_MAX,
            world_min_scaled=self.lon_min_scaled,
            step_scaled=self.lon_step,
            bin_count=self.longitude_bins,
            axis_name="longitude",
        )
        lat_min, lat_max = self._axis_cover(
            min_lat,
            max_lat,
            world_min_degrees=self.WORLD_LAT_MIN,
            world_max_degrees=self.WORLD_LAT_MAX,
            world_min_scaled=self.lat_min_scaled,
            step_scaled=self.lat_step,
            bin_count=self.latitude_bins,
            axis_name="latitude",
        )
        return CellRect(lon_min, lon_max, lat_min, lat_max)



def _static_hash_pair(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def _static_empty_leaf_hash() -> bytes:
    return hashlib.sha256(b"PATRON_EMPTY_LEAF_v1").digest()


def _static_leaf_hash(cell_id: str, records: list[CellEdgeRecord], bits: int) -> bytes:
    ordered = sorted(records, key=lambda record: record.edge_id)
    payload = bytearray(struct.pack("<I", len(ordered)))
    for record in ordered:
        payload.extend(struct.pack(
            "<QQQQQQqqqq",
            record.edge_id, record.start_node, record.end_node,
            record.original_edge_id, record.segment_index, record.segment_count,
            record.start_lon_scaled, record.start_lat_scaled,
            record.end_lon_scaled, record.end_lat_scaled,
        ))
    prefix_bytes = int(cell_id, 2).to_bytes((bits + 7) // 8, "big")
    domain = struct.pack("<H", bits) + prefix_bytes
    return hashlib.sha256(domain + payload).digest()


def _static_sparse_root(depth: int, leaves: Mapping[str, bytes]) -> bytes:
    defaults = [_static_empty_leaf_hash()]
    for _ in range(depth):
        defaults.append(_static_hash_pair(defaults[-1], defaults[-1]))
    nodes = {prefix: value for prefix, value in leaves.items() if value != defaults[0]}
    active = set(leaves)
    while active:
        parents = {prefix[:-1] for prefix in active}
        for parent in parents:
            plen = len(parent)
            left_prefix = parent + "0"
            right_prefix = parent + "1"
            child_default = defaults[depth - (plen + 1)]
            left = nodes.get(left_prefix, child_default)
            right = nodes.get(right_prefix, child_default)
            value = _static_hash_pair(left, right)
            parent_default = defaults[depth - plen]
            if value != parent_default:
                nodes[parent] = value
        if "" in parents:
            break
        active = parents
    return nodes.get("", defaults[depth])

class GeohashCellIndex:
    """Static road-cell mapping produced by the road preprocessing project.

    The Geohash tree is implicit.  `cell_id`/prefix is the direct key for a
    fixed-depth leaf, so lookup never traverses a spatial tree.
    """

    def __init__(self, index_dir: Path, manifest: dict[str, Any], cells: dict[str, list[CellEdgeRecord]]):
        self.index_dir = Path(index_dir)
        self.manifest = manifest
        self.bits = int(manifest["bits"])
        self.coordinate_scale = int(manifest["coordinate_scale"])
        self.grid = GeohashGrid(self.bits, self.coordinate_scale)
        self.cells = {
            cell_id: sorted(records, key=lambda record: record.edge_id)
            for cell_id, records in cells.items()
        }
        self.edge_to_cell: dict[int, str] = {}
        for cell_id, records in self.cells.items():
            if len(cell_id) != self.bits or any(bit not in "01" for bit in cell_id):
                raise ValueError(f"invalid cell prefix in index: {cell_id!r}")
            for record in records:
                if record.edge_id in self.edge_to_cell:
                    raise ValueError(f"edge {record.edge_id} belongs to more than one cell")
                self.edge_to_cell[record.edge_id] = cell_id

        expected_edge_count = int(manifest.get("indexed_edge_count", len(self.edge_to_cell)))
        if expected_edge_count != len(self.edge_to_cell):
            raise ValueError(
                f"cell index edge count mismatch: {len(self.edge_to_cell)} != {expected_edge_count}"
            )

    @classmethod
    def load(cls, index_dir: str | Path) -> "GeohashCellIndex":
        index_dir = Path(index_dir)
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        payload_name = manifest.get("files", {}).get("cell_payloads", "cell_payloads.json")
        raw_cells = json.loads((index_dir / payload_name).read_text(encoding="utf-8"))
        cells = {
            str(cell_id): [CellEdgeRecord.from_mapping(value) for value in values]
            for cell_id, values in raw_cells.items()
        }
        bits = int(manifest["bits"])
        leaf_name = manifest.get("files", {}).get("cell_leaf_hashes", "cell_leaf_hashes.json")
        leaf_path = index_dir / leaf_name
        if leaf_path.exists():
            raw_leaves = json.loads(leaf_path.read_text(encoding="utf-8"))
            if set(raw_leaves) != set(cells):
                raise ValueError("static cell payload and leaf-hash key sets differ")
            verified_leaves: dict[str, bytes] = {}
            for cell_id, records in cells.items():
                computed = _static_leaf_hash(cell_id, records, bits)
                stored = bytes.fromhex(str(raw_leaves[cell_id]))
                if computed != stored:
                    raise ValueError(f"static cell payload hash mismatch: {cell_id}")
                verified_leaves[cell_id] = computed
            computed_root = _static_sparse_root(bits, verified_leaves).hex()
            if computed_root != str(manifest.get("root_hash", "")):
                raise ValueError("static cell index root does not match its manifest")
        return cls(index_dir, manifest, cells)

    @property
    def static_root_hash(self) -> str:
        return str(self.manifest.get("root_hash", ""))

    @property
    def edge_count(self) -> int:
        return len(self.edge_to_cell)

    def cell_records(self, cell_id: str) -> list[CellEdgeRecord]:
        return list(self.cells.get(cell_id, []))

    def cell_ids_for_rect(self, rect: CellRect) -> list[str]:
        normalized = rect.normalized()
        if not 0 <= normalized.lon_index_min <= normalized.lon_index_max < self.grid.longitude_bins:
            raise ValueError("longitude cell rectangle outside grid")
        if not 0 <= normalized.lat_index_min <= normalized.lat_index_max < self.grid.latitude_bins:
            raise ValueError("latitude cell rectangle outside grid")
        result: list[str] = []
        for lat_index in range(normalized.lat_index_min, normalized.lat_index_max + 1):
            for lon_index in range(normalized.lon_index_min, normalized.lon_index_max + 1):
                result.append(self.grid.prefix_from_indices(lon_index, lat_index))
        return result

    def edge_ids_for_rect(self, rect: CellRect) -> list[int]:
        edge_ids: list[int] = []
        for cell_id in self.cell_ids_for_rect(rect):
            edge_ids.extend(record.edge_id for record in self.cell_records(cell_id))
        return sorted(edge_ids)

    def covered_geo_rectangle(self, rect: CellRect) -> dict[str, float]:
        normalized = rect.normalized()
        bottom_left = self.grid.bounds_from_indices(normalized.lon_index_min, normalized.lat_index_min)
        top_right = self.grid.bounds_from_indices(normalized.lon_index_max, normalized.lat_index_max)
        return {
            "min_lon": bottom_left["min_lon"],
            "max_lon": top_right["max_lon"],
            "min_lat": bottom_left["min_lat"],
            "max_lat": top_right["max_lat"],
        }

    def fingerprint(self) -> str:
        """Bind the daily dataset to one immutable static cell index."""
        hasher = hashlib.sha256()
        hasher.update(b"KZG_GEOHASH_STATIC_INDEX_V1")
        hasher.update(json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        # The manifest points at the static root and grid parameters; binding it
        # is enough to reject accidental use of another preprocessing output.
        return hasher.hexdigest()
