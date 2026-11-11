"""Complete-cell rectangle queries and client-side Merkle verification."""

from __future__ import annotations

from fractions import Fraction
from math import ceil, floor
from typing import Any, Mapping

from .index_builder import AuthenticatedCellIndex
from .merkle_tree import MerkleProof
from .serialization import IndexedEdgeRecord, empty_leaf_hash, leaf_hash


def _record_list(records: list[IndexedEdgeRecord]) -> list[dict[str, int]]:
    return [record.to_dict() for record in records]


def _coordinate_as_fraction(value: int | float | str | Fraction) -> Fraction:
    """Parse a degree coordinate without introducing binary-float rounding.

    CLI values are passed in as strings, so decimal inputs such as ``104.05``
    retain their exact decimal value during the cell-range calculation.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _minimal_axis_cover(
    minimum_degrees: int | float | str | Fraction,
    maximum_degrees: int | float | str | Fraction,
    *,
    coordinate_scale: int,
    world_min_degrees: int,
    world_max_degrees: int,
    world_min_scaled: int,
    step_scaled: Fraction,
    bin_count: int,
    axis_name: str,
) -> tuple[int, int]:
    """Return the smallest inclusive cell-index interval covering an axis range.

    For a non-zero interval, the upper index is computed with ``ceil(...) - 1``.
    Therefore a maximum coordinate exactly on a grid boundary does not pull in
    the next cell unnecessarily.  A zero-width interval is treated as a point
    query and maps to the single cell containing that point.
    """

    minimum = _coordinate_as_fraction(minimum_degrees)
    maximum = _coordinate_as_fraction(maximum_degrees)
    world_min = Fraction(world_min_degrees)
    world_max = Fraction(world_max_degrees)

    if minimum > maximum:
        raise ValueError(f"minimum {axis_name} cannot exceed maximum {axis_name}")
    if minimum < world_min or maximum > world_max:
        raise ValueError(
            f"{axis_name} range must stay within [{world_min_degrees}, {world_max_degrees}]"
        )

    minimum_scaled = minimum * coordinate_scale
    maximum_scaled = maximum * coordinate_scale
    min_index = floor((minimum_scaled - world_min_scaled) / step_scaled)

    if minimum == maximum:
        max_index = min_index
    else:
        max_index = ceil((maximum_scaled - world_min_scaled) / step_scaled) - 1

    min_index = max(0, min(bin_count - 1, min_index))
    max_index = max(0, min(bin_count - 1, max_index))
    return min_index, max_index


def geographic_rectangle_to_cell_rectangle(
    index: AuthenticatedCellIndex,
    min_lon: int | float | str | Fraction,
    max_lon: int | float | str | Fraction,
    min_lat: int | float | str | Fraction,
    max_lat: int | float | str | Fraction,
) -> dict[str, Any]:
    """Convert a geographic rectangle to its minimal complete-cell cover.

    The returned cell ranges are inclusive.  Every point of the requested
    longitude/latitude rectangle is covered by the selected complete cells,
    while no completely disjoint outer cell is added.  When a non-zero range's
    maximum lies exactly on a grid line, the cell beyond that line is not
    included merely because of the shared boundary.
    """

    grid = index.grid
    lon_index_min, lon_index_max = _minimal_axis_cover(
        min_lon,
        max_lon,
        coordinate_scale=grid.coordinate_scale,
        world_min_degrees=grid.WORLD_LON_MIN,
        world_max_degrees=grid.WORLD_LON_MAX,
        world_min_scaled=grid.lon_min_scaled,
        step_scaled=grid.lon_step,
        bin_count=grid.longitude_bins,
        axis_name="longitude",
    )
    lat_index_min, lat_index_max = _minimal_axis_cover(
        min_lat,
        max_lat,
        coordinate_scale=grid.coordinate_scale,
        world_min_degrees=grid.WORLD_LAT_MIN,
        world_max_degrees=grid.WORLD_LAT_MAX,
        world_min_scaled=grid.lat_min_scaled,
        step_scaled=grid.lat_step,
        bin_count=grid.latitude_bins,
        axis_name="latitude",
    )

    lower_left = grid.bounds_from_indices(lon_index_min, lat_index_min)
    upper_right = grid.bounds_from_indices(lon_index_max, lat_index_max)
    scale = grid.coordinate_scale

    requested_min_lon = float(_coordinate_as_fraction(min_lon))
    requested_max_lon = float(_coordinate_as_fraction(max_lon))
    requested_min_lat = float(_coordinate_as_fraction(min_lat))
    requested_max_lat = float(_coordinate_as_fraction(max_lat))

    return {
        "requested_geo_rectangle": {
            "min_lon": requested_min_lon,
            "max_lon": requested_max_lon,
            "min_lat": requested_min_lat,
            "max_lat": requested_max_lat,
        },
        "cell_rectangle": {
            "lon_index_min": lon_index_min,
            "lon_index_max": lon_index_max,
            "lat_index_min": lat_index_min,
            "lat_index_max": lat_index_max,
            "column_count": lon_index_max - lon_index_min + 1,
            "row_count": lat_index_max - lat_index_min + 1,
            "cell_count": (lon_index_max - lon_index_min + 1)
            * (lat_index_max - lat_index_min + 1),
        },
        "covered_geo_rectangle": {
            "min_lon": float(lower_left.lon_min / scale),
            "max_lon": float(upper_right.lon_max / scale),
            "min_lat": float(lower_left.lat_min / scale),
            "max_lat": float(upper_right.lat_max / scale),
        },
    }


def query_geographic_rectangle(
    index: AuthenticatedCellIndex,
    min_lon: int | float | str | Fraction,
    max_lon: int | float | str | Fraction,
    min_lat: int | float | str | Fraction,
    max_lat: int | float | str | Fraction,
) -> dict[str, Any]:
    """Query the minimal complete-cell rectangle covering geographic bounds."""

    coverage = geographic_rectangle_to_cell_rectangle(
        index, min_lon, max_lon, min_lat, max_lat
    )
    cell_rectangle = coverage["cell_rectangle"]
    result = query_complete_cell_rectangle(
        index,
        int(cell_rectangle["lon_index_min"]),
        int(cell_rectangle["lon_index_max"]),
        int(cell_rectangle["lat_index_min"]),
        int(cell_rectangle["lat_index_max"]),
    )
    result["format"] = "geographic_complete_cell_cover_query_v1"
    result["geographic_query"] = coverage["requested_geo_rectangle"]
    result["covered_geo_rectangle"] = coverage["covered_geo_rectangle"]
    return result


def query_complete_cell_rectangle(
    index: AuthenticatedCellIndex,
    lon_index_min: int,
    lon_index_max: int,
    lat_index_min: int,
    lat_index_max: int,
) -> dict[str, Any]:
    """Directly enumerate a rectangle of complete cells.

    No tree traversal is used for lookup.  The rectangle is represented by
    inclusive longitude/latitude cell-index ranges, and each leaf is fetched
    directly by its fixed-depth prefix.
    """

    if lon_index_min > lon_index_max or lat_index_min > lat_index_max:
        raise ValueError("rectangle minimum index cannot exceed maximum index")
    if not (0 <= lon_index_min <= lon_index_max < index.grid.longitude_bins):
        raise ValueError("longitude cell index range is outside the grid")
    if not (0 <= lat_index_min <= lat_index_max < index.grid.latitude_bins):
        raise ValueError("latitude cell index range is outside the grid")

    cells: list[dict[str, Any]] = []
    candidate_edges: list[dict[str, int]] = []
    for lat_index in range(lat_index_min, lat_index_max + 1):
        for lon_index in range(lon_index_min, lon_index_max + 1):
            cell_id = index.grid.prefix_from_indices(lon_index, lat_index)
            records = index.cell_records(cell_id)
            proof = index.tree.proof(cell_id)
            item = {
                "cell_id": cell_id,
                "lon_index": lon_index,
                "lat_index": lat_index,
                "edge_count": len(records),
                "edges": _record_list(records),
                "proof": proof.to_dict(),
            }
            cells.append(item)
            candidate_edges.extend(item["edges"])

    return {
        "format": "complete_cell_rectangle_query_v1",
        "root_hash": index.root_hash_hex,
        "query": {
            "lon_index_min": lon_index_min,
            "lon_index_max": lon_index_max,
            "lat_index_min": lat_index_min,
            "lat_index_max": lat_index_max,
            "cell_count": len(cells),
        },
        "cells": cells,
        "candidate_edges": candidate_edges,
    }


def verify_query_result(
    result: Mapping[str, Any],
    index: AuthenticatedCellIndex,
    expected_root_hex: str | None = None,
) -> dict[str, Any]:
    """Verify every returned cell and its proof against a trusted root."""

    expected_root = bytes.fromhex(expected_root_hex or index.root_hash_hex)
    if str(result.get("root_hash", "")) != expected_root.hex():
        raise ValueError("query result root hash is not the trusted root")
    query = result.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("query result has no query rectangle")
    required = (
        "lon_index_min",
        "lon_index_max",
        "lat_index_min",
        "lat_index_max",
    )
    if any(name not in query for name in required):
        raise ValueError("query result rectangle is incomplete")
    expected_cells: list[str] = []
    expected_positions: dict[str, tuple[int, int]] = {}
    for lat_index in range(int(query["lat_index_min"]), int(query["lat_index_max"]) + 1):
        for lon_index in range(int(query["lon_index_min"]), int(query["lon_index_max"]) + 1):
            cell_id = index.grid.prefix_from_indices(lon_index, lat_index)
            expected_cells.append(cell_id)
            expected_positions[cell_id] = (lon_index, lat_index)

    raw_cells = result.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != len(expected_cells):
        raise ValueError("query result does not contain exactly the requested cells")

    seen: set[str] = set()
    verified_edges: list[dict[str, int]] = []
    for item in raw_cells:
        if not isinstance(item, Mapping):
            raise ValueError("malformed cell result")
        cell_id = str(item.get("cell_id", ""))
        if cell_id in seen or cell_id not in expected_positions:
            raise ValueError(f"unexpected or duplicate cell in result: {cell_id}")
        seen.add(cell_id)
        expected_lon, expected_lat = expected_positions[cell_id]
        if int(item.get("lon_index", -1)) != expected_lon or int(item.get("lat_index", -1)) != expected_lat:
            raise ValueError(f"cell index does not match prefix: {cell_id}")
        raw_edges = item.get("edges")
        if not isinstance(raw_edges, list):
            raise ValueError(f"cell {cell_id} has malformed edge list")
        records = [IndexedEdgeRecord.from_mapping(edge) for edge in raw_edges]
        if len({record.edge_id for record in records}) != len(records):
            raise ValueError(f"cell {cell_id} contains duplicate edge IDs")
        actual_leaf = (
            empty_leaf_hash()
            if not records
            else leaf_hash(cell_id, records, index.grid.bits)
        )
        proof_value = item.get("proof")
        if not isinstance(proof_value, Mapping):
            raise ValueError(f"cell {cell_id} has no Merkle proof")
        proof = MerkleProof(
            cell_id=str(proof_value.get("cell_id", "")),
            depth=int(proof_value.get("depth", -1)),
            default_bitmap=str(proof_value.get("default_bitmap", "")),
            siblings=[str(value) for value in proof_value.get("siblings", [])],
        )
        if not index.tree.verify(cell_id, actual_leaf, proof, expected_root):
            raise ValueError(f"Merkle verification failed for cell {cell_id}")
        verified_edges.extend(record.to_dict() for record in records)

    if seen != set(expected_cells):
        raise ValueError("one or more requested cells are missing")
    supplied_candidates = result.get("candidate_edges", [])
    if supplied_candidates != verified_edges:
        raise ValueError("candidate_edges does not match verified cell contents")
    return {
        "verified": True,
        "root_hash": expected_root.hex(),
        "verified_cell_count": len(seen),
        "verified_edge_count": len(verified_edges),
        "empty_cell_count": sum(
            1 for item in raw_cells if not item.get("edges")
        ),
    }
