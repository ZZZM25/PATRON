"""Build and load a directly addressable authenticated cell index."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geohash.encoder import GeohashGrid
from network.edge_builder import SplitNetwork

from .merkle_tree import SparseMerkleTree
from .serialization import IndexedEdgeRecord, leaf_hash


def _record_from_edge(edge: Any, nodes: dict[int, Any], coordinate_scale: int) -> IndexedEdgeRecord:
    start = nodes[edge.start_node]
    end = nodes[edge.end_node]
    return IndexedEdgeRecord(
        edge_id=edge.edge_id,
        start_node=edge.start_node,
        end_node=edge.end_node,
        original_edge_id=edge.original_edge_id,
        segment_index=edge.segment_index,
        segment_count=edge.segment_count,
        start_lon_scaled=round(start.lon * coordinate_scale),
        start_lat_scaled=round(start.lat * coordinate_scale),
        end_lon_scaled=round(end.lon * coordinate_scale),
        end_lat_scaled=round(end.lat * coordinate_scale),
    )


@dataclass
class AuthenticatedCellIndex:
    index_dir: Path
    manifest: dict[str, Any]
    cells: dict[str, list[IndexedEdgeRecord]]
    tree: SparseMerkleTree
    grid: GeohashGrid

    @property
    def root_hash(self) -> bytes:
        return self.tree.root_hash

    @property
    def root_hash_hex(self) -> str:
        return self.root_hash.hex()

    def cell_records(self, cell_id: str) -> list[IndexedEdgeRecord]:
        return list(self.cells.get(cell_id, []))


def build_authenticated_index(
    split_network: SplitNetwork,
    output_dir: str | Path,
) -> tuple[AuthenticatedCellIndex, dict[str, str]]:
    """Group complete split edges by cell and build the sparse Merkle root."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_scale = split_network.grid.coordinate_scale
    cells: dict[str, list[IndexedEdgeRecord]] = defaultdict(list)
    for edge in split_network.edges:
        cells[edge.cell_id].append(
            _record_from_edge(edge, split_network.nodes, coordinate_scale)
        )
    cells = {
        cell_id: sorted(records, key=lambda record: record.edge_id)
        for cell_id, records in sorted(cells.items())
    }
    leaf_hashes = {
        cell_id: leaf_hash(cell_id, records, split_network.grid.bits)
        for cell_id, records in cells.items()
    }
    tree = SparseMerkleTree(split_network.grid.bits, leaf_hashes)

    cells_path = output_dir / "cell_payloads.json"
    leaves_path = output_dir / "cell_leaf_hashes.json"
    manifest_path = output_dir / "manifest.json"
    root_path = output_dir / "root_hash.txt"

    with cells_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                cell_id: [record.to_dict() for record in records]
                for cell_id, records in cells.items()
            },
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    with leaves_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {cell_id: value.hex() for cell_id, value in leaf_hashes.items()},
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    root_path.write_text(tree.root_hash.hex() + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "format": "authenticated_geohash_cell_index_v1",
        "bits": split_network.grid.bits,
        "coordinate_scale": coordinate_scale,
        "reference_latitude": split_network.grid.reference_latitude,
        "grid": split_network.grid.describe(),
        "root_hash": tree.root_hash.hex(),
        "nonempty_cell_count": len(cells),
        "indexed_edge_count": len(split_network.edges),
        "query_semantics": "inclusive rectangle of complete cell indices",
        "empty_cell_leaf": "SHA256(PATRON_EMPTY_LEAF_v1)",
        "files": {
            "cell_payloads": cells_path.name,
            "cell_leaf_hashes": leaves_path.name,
            "root_hash": root_path.name,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    index = AuthenticatedCellIndex(
        index_dir=output_dir,
        manifest=manifest,
        cells=cells,
        tree=tree,
        grid=split_network.grid,
    )
    return index, {
        "index_dir": str(output_dir),
        "manifest": str(manifest_path),
        "cell_payloads": str(cells_path),
        "cell_leaf_hashes": str(leaves_path),
        "root_hash": str(root_path),
    }


def load_authenticated_index(index_dir: str | Path) -> AuthenticatedCellIndex:
    index_dir = Path(index_dir)
    with (index_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (index_dir / "cell_payloads.json").open("r", encoding="utf-8") as handle:
        raw_cells = json.load(handle)
    with (index_dir / "cell_leaf_hashes.json").open("r", encoding="utf-8") as handle:
        raw_leaves = json.load(handle)

    bits = int(manifest["bits"])
    coordinate_scale = int(manifest["coordinate_scale"])
    reference_latitude = float(manifest["reference_latitude"])
    grid = GeohashGrid.fixed(bits, reference_latitude, coordinate_scale)
    cells = {
        cell_id: sorted(
            [IndexedEdgeRecord.from_mapping(value) for value in values],
            key=lambda record: record.edge_id,
        )
        for cell_id, values in raw_cells.items()
    }
    leaf_hashes = {cell_id: bytes.fromhex(value) for cell_id, value in raw_leaves.items()}
    if set(cells) != set(leaf_hashes):
        raise ValueError("cell payload and leaf-hash key sets differ")
    for cell_id, records in cells.items():
        expected = leaf_hash(cell_id, records, bits)
        if expected != leaf_hashes[cell_id]:
            raise ValueError(f"cell payload hash mismatch while loading {cell_id}")
    tree = SparseMerkleTree(bits, leaf_hashes)
    if tree.root_hash.hex() != str(manifest["root_hash"]):
        raise ValueError("stored root hash does not match reconstructed tree")
    return AuthenticatedCellIndex(index_dir, manifest, cells, tree, grid)
