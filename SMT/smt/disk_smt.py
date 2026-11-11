"""Disk-backed construction and querying for very large Cell50 Edge-SMT indexes.

The authenticated data structure is unchanged.  This module changes only its
physical representation:

* trajectory records are first written to fixed-size temporary shards;
* one shard and one edge SMT are built at a time;
* final per-edge records live in one compact, seekable data file;
* the spatial tree, roots, counts and edge offsets form a small manifest;
* queries load only spatially matched edges and keep bounded LRU caches.

The stable ``(start, arrival_sequence)`` ordering is identical to repeated
``bisect_right`` insertion used by ``PackedEdgeEntryStore``.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import os
import pickle
import struct
from array import array
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .authenticated_index import authenticate_spatial_tree
from .cell_dataset_builder import (
    _load_beijing_points,
    _set_large_csv_field_limit,
    parse_cell_partitioned_beijing_points,
)
from .crypto import (
    ENTRY_LIST_TAIL,
    entry_list_prepend,
    hash_entry,
    hash_smt_edge,
)
from .edge_smt import EdgeSMT, key_for_id
from .spatial_tree import BinarySpatialTreeBuilder
from .trajectory_parser import (
    collect_files,
    load_relaxed_json,
    parse_cell_partitioned_trajectory,
)


DISK_PER_EDGE_SMT_MODE = "per-edge-smt-spatial-root-v1"
EDGE_DATA_MAGIC = b"EDGDATA1"
SMT_DATA_MAGIC = b"SMTDAT01"
CATALOG_MAGIC = b"CELLCAT1"
_EDGE_DATA_HEADER = struct.Struct(">8sI")
_SMT_DATA_HEADER = struct.Struct(">8sI")
_BLOCK_HEADER = struct.Struct(">II")
_ENTRY = struct.Struct(">32sII")
_SPOOL_RECORD = struct.Struct(">IIIQ32s")
_CATALOG_HEADER = struct.Struct(">8sQ")
_CATALOG_TRAJECTORY = struct.Struct(">32sI")
_CATALOG_SEGMENT = struct.Struct(">IIII")


class DigestArray:
    """Compact random-access sequence of 32-byte digests."""

    __slots__ = ("raw",)

    def __init__(self, raw: bytes | bytearray):
        if len(raw) % 32:
            raise ValueError("digest array byte length is not divisible by 32")
        self.raw = bytes(raw)

    def __len__(self) -> int:
        return len(self.raw) // 32

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            return [self[i] for i in range(start, stop)]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = index * 32
        return self.raw[position:position + 32]


@dataclass(slots=True)
class _LoadedEdgeBlock:
    entry_count: int
    unique_count: int
    entry_bytes: bytes
    trajectory_id_bytes: bytes

    def get_entry(self, index: int) -> tuple[bytes, int, int]:
        if not 0 <= index < self.entry_count:
            raise IndexError(index)
        return _ENTRY.unpack_from(self.entry_bytes, index * _ENTRY.size)

    def iter_entries(self, start: int = 0, stop: int | None = None):
        if stop is None:
            stop = self.entry_count
        if not 0 <= start <= stop <= self.entry_count:
            raise ValueError("invalid entry range")
        for index in range(start, stop):
            yield _ENTRY.unpack_from(self.entry_bytes, index * _ENTRY.size)

    def trajectory_ids(self):
        for position in range(0, len(self.trajectory_id_bytes), 32):
            yield self.trajectory_id_bytes[position:position + 32]

    @property
    def byte_size(self) -> int:
        return len(self.entry_bytes) + len(self.trajectory_id_bytes) + 64


class DiskEdgeEntryStore:
    """Read-only packed edge records addressed by manifest offsets."""

    def __init__(
        self,
        data_file: str | Path,
        entry_counts: array,
        unique_counts: array,
        offsets: array,
        lengths: array,
        block_hashes: DigestArray,
        cache_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.data_file = str(Path(data_file).resolve())
        self.entry_counts = entry_counts
        self._unique_counts = unique_counts
        self.offsets = offsets
        self.lengths = lengths
        self.block_hashes = block_hashes
        self.cache_bytes = int(cache_bytes)
        self._cache: OrderedDict[int, _LoadedEdgeBlock] = OrderedDict()
        self._cached_bytes = 0
        self.query_edge_data_load_time_s = 0.0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cached_bytes"] = 0
        state["query_edge_data_load_time_s"] = 0.0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _load(self, eid: int) -> _LoadedEdgeBlock:
        if not 0 <= eid < len(self.entry_counts):
            raise IndexError(eid)
        cached = self._cache.pop(eid, None)
        if cached is not None:
            self._cache[eid] = cached
            return cached
        count = int(self.entry_counts[eid])
        unique_count = int(self._unique_counts[eid])
        if count == 0:
            return _LoadedEdgeBlock(0, 0, b"", b"")
        load_started = perf_counter()
        offset = int(self.offsets[eid])
        length = int(self.lengths[eid])
        with open(self.data_file, "rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"truncated disk edge block {eid}")
        if hashlib.sha256(raw).digest() != self.block_hashes[eid]:
            raise ValueError(f"disk edge block checksum mismatch: {eid}")
        stored_count, stored_unique = _BLOCK_HEADER.unpack_from(raw, 0)
        if stored_count != count or stored_unique != unique_count:
            raise ValueError(f"disk edge block metadata mismatch: {eid}")
        entry_end = _BLOCK_HEADER.size + count * _ENTRY.size
        expected_length = entry_end + unique_count * 32
        if expected_length != length:
            raise ValueError(f"disk edge block length mismatch: {eid}")
        block = _LoadedEdgeBlock(
            count,
            unique_count,
            raw[_BLOCK_HEADER.size:entry_end],
            raw[entry_end:],
        )
        self.query_edge_data_load_time_s += perf_counter() - load_started
        if block.byte_size <= self.cache_bytes:
            self._cache[eid] = block
            self._cached_bytes += block.byte_size
            while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._cached_bytes -= evicted.byte_size
        return block

    def begin_query(self) -> None:
        """Drop previous-query data so no target edge is preloaded."""
        self._cache.clear()
        self._cached_bytes = 0
        self.query_edge_data_load_time_s = 0.0

    def get_entry(self, eid: int, index: int) -> tuple[bytes, int, int]:
        return self._load(eid).get_entry(index)

    def iter_entries(self, eid: int):
        yield from self._load(eid).iter_entries()

    def iter_prefix(self, eid: int, count: int):
        yield from self._load(eid).iter_entries(0, count)

    def linear_upper_bound_start(self, eid: int, query_end: int) -> int:
        count = 0
        for _, start, _ in self._load(eid).iter_entries():
            if int(start) > int(query_end):
                break
            count += 1
        return count

    def unique_trajectory_counts(self) -> list[int]:
        return list(self._unique_counts)

    def unique_trajectory_ids(self, eid: int):
        yield from self._load(eid).trajectory_ids()

    def total_entries(self) -> int:
        return sum(self.entry_counts)


class DiskSmtBaselineIndex:
    """Small manifest plus lazy access to compact per-edge data."""

    def __init__(
        self,
        *,
        road,
        store: DiskEdgeEntryStore,
        tree,
        edge_entry_roots: DigestArray,
        edge_hashes: DigestArray,
        root_hash: bytes,
        theta: int,
        trajectory_files,
        edge_smt_roots: DigestArray,
        smt_cache_bytes: int = 1024 * 1024 * 1024,
        smt_data_file: str | Path | None = None,
        smt_offsets: array | None = None,
        smt_lengths: array | None = None,
    ) -> None:
        self.road = road
        self.store = store
        self.tree = tree
        self.edge_entry_roots = edge_entry_roots
        self.edge_hashes = edge_hashes
        self.edge_suffix_states = []
        self.root_hash = root_hash
        self.theta = int(theta)
        self.trajectory_files = tuple(map(str, trajectory_files))
        self.edge_smt_roots = edge_smt_roots
        self.proof_mode = DISK_PER_EDGE_SMT_MODE
        self.disk_backed = True
        self.smt_cache_bytes = int(smt_cache_bytes)
        self.smt_data_file = (
            str(Path(smt_data_file).resolve()) if smt_data_file is not None else None
        )
        self.smt_offsets = smt_offsets
        self.smt_lengths = smt_lengths
        self._smt_cache: OrderedDict[int, tuple[EdgeSMT, int]] = OrderedDict()
        self._smt_cached_bytes = 0
        self.query_smt_materialization_time_s = 0.0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_smt_cache"] = OrderedDict()
        state["_smt_cached_bytes"] = 0
        state["query_smt_materialization_time_s"] = 0.0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def suffix_state(self, eid: int, position: int) -> bytes:
        count = int(self.store.entry_counts[eid])
        if not 0 <= position <= count:
            raise IndexError(position)
        block = self.store._load(eid)
        state = ENTRY_LIST_TAIL
        for index in range(count - 1, position - 1, -1):
            key, start, end = block.get_entry(index)
            state = entry_list_prepend(hash_entry(key, start, end), state)
        return state

    def begin_query(self) -> None:
        """Ensure every query discovers and loads its edges during traversal."""
        self.store.begin_query()
        self._smt_cache.clear()
        self._smt_cached_bytes = 0
        self.query_smt_materialization_time_s = 0.0

    def query_storage_times(self) -> tuple[float, float]:
        return (
            self.store.query_edge_data_load_time_s,
            self.query_smt_materialization_time_s,
        )

    def _edge_smt(self, eid: int) -> EdgeSMT:
        cached = self._smt_cache.pop(eid, None)
        if cached is not None:
            self._smt_cache[eid] = cached
            return cached[0]
        materialization_started = perf_counter()
        smt_data_file = getattr(self, "smt_data_file", None)
        smt_offsets = getattr(self, "smt_offsets", None)
        smt_lengths = getattr(self, "smt_lengths", None)
        if (
            smt_data_file is not None
            and smt_offsets is not None
            and smt_lengths is not None
            and int(smt_lengths[eid]) > 0
        ):
            with open(smt_data_file, "rb") as stream:
                stream.seek(int(smt_offsets[eid]))
                smt = pickle.load(stream)
            if not isinstance(smt, EdgeSMT):
                raise ValueError(f"invalid persisted edge SMT object: {eid}")
        else:
            # Backward-compatible path for compact indexes built before full
            # per-edge SMT nodes were materialized on disk.
            smt = EdgeSMT.build(self.store.unique_trajectory_ids(eid))
        if smt.root_hash != self.edge_smt_roots[eid]:
            raise ValueError(f"disk edge SMT root mismatch: {eid}")
        self.query_smt_materialization_time_s += (
            perf_counter() - materialization_started
        )
        # Conservative estimate for Python dict + Patricia nodes.  The cache is
        # a performance aid only; correctness never depends on this estimate.
        estimated_bytes = max(1, len(smt.values)) * 320 + 1024
        if estimated_bytes <= self.smt_cache_bytes:
            self._smt_cache[eid] = (smt, estimated_bytes)
            self._smt_cached_bytes += estimated_bytes
            while (
                self._smt_cached_bytes > self.smt_cache_bytes
                and len(self._smt_cache) > 1
            ):
                _, (_, size) = self._smt_cache.popitem(last=False)
                self._smt_cached_bytes -= size
        return smt

    def prove_pairs(self, pairs: list[tuple[int, bytes]]):
        """Generate fresh proofs, grouping keys so each edge tree loads once."""
        proofs = []
        for eid, group in itertools.groupby(pairs, key=lambda item: item[0]):
            smt = self._edge_smt(eid)
            for _, key in group:
                proofs.append(smt.prove(eid, key))
        return proofs


class ShardedEntrySpool:
    """Fixed-record temporary shards used only during construction."""

    def __init__(self, directory: str | Path, shard_count: int = 512) -> None:
        if shard_count < 1:
            raise ValueError("shard_count must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_count = int(shard_count)
        self.paths = [
            self.directory / f"entries_{index:04d}.bin"
            for index in range(self.shard_count)
        ]
        self._streams = [
            path.open("wb", buffering=32 * 1024) for path in self.paths
        ]
        self.record_count = 0
        self.closed = False

    def append(
        self,
        eid: int,
        start: int,
        end: int,
        sequence: int,
        trajectory_id: bytes,
    ) -> None:
        if self.closed:
            raise ValueError("spool is closed")
        self._streams[eid % self.shard_count].write(
            _SPOOL_RECORD.pack(eid, start, end, sequence, trajectory_id)
        )
        self.record_count += 1

    def close(self) -> None:
        if self.closed:
            return
        for stream in self._streams:
            stream.close()
        self.closed = True


class StreamingTrajectoryCatalogWriter:
    """Compact materialization of transformed trajectories for audit/reuse."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.destination.with_suffix(self.destination.suffix + ".tmp")
        self.stream = self.temporary.open("w+b")
        self.stream.write(_CATALOG_HEADER.pack(CATALOG_MAGIC, 0))
        self.count = 0

    def append(self, parsed) -> None:
        self.stream.write(
            _CATALOG_TRAJECTORY.pack(parsed.trajectory_id, len(parsed.segments))
        )
        for segment in parsed.segments:
            self.stream.write(
                _CATALOG_SEGMENT.pack(
                    segment.start_node,
                    segment.start_time,
                    segment.end_node,
                    segment.end_time,
                )
            )
        self.count += 1

    def close(self) -> int:
        self.stream.seek(0)
        self.stream.write(_CATALOG_HEADER.pack(CATALOG_MAGIC, self.count))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        size = self.stream.seek(0, os.SEEK_END)
        self.stream.close()
        self.temporary.replace(self.destination)
        return size


def spool_cell_trajectories(
    *,
    city: str,
    source_road,
    partition,
    trajectory_globs: list[str],
    config_dir: Path,
    spool: ShardedEntrySpool,
    catalog_file: str | Path,
    expected_file_count: int | None = None,
    max_trajectories: int | None = None,
):
    """Transform trajectories and stream records to disk shards.

    ``max_trajectories`` selects the first N valid trajectories in the same
    natural file/record order used by the original scaling benchmark.  It is
    intentionally applied after parsing because malformed/empty Beijing rows
    are not trajectories in the authenticated data set.
    """
    if max_trajectories is not None and int(max_trajectories) < 1:
        raise ValueError("max_trajectories must be positive")
    limit = int(max_trajectories) if max_trajectories is not None else None
    files = collect_files(trajectory_globs, config_dir)
    if expected_file_count is not None and len(files) != int(expected_file_count):
        raise ValueError(
            f"expected {expected_file_count} trajectory files, got {len(files)}"
        )
    catalog = StreamingTrajectoryCatalogWriter(catalog_file)
    seen: set[bytes] = set()
    trajectory_count = 0
    source_segments = 0
    partitioned_segments = 0
    arrival_sequence = 0
    spool_append_seconds = 0.0

    def consume(parsed, source_segment_count: int) -> None:
        nonlocal trajectory_count, source_segments, partitioned_segments
        nonlocal arrival_sequence, spool_append_seconds
        if parsed.trajectory_id in seen:
            raise ValueError("duplicate trajectory_id: " + parsed.trajectory_id.hex())
        seen.add(parsed.trajectory_id)
        catalog.append(parsed)
        append_started = perf_counter()
        for eid, segment in zip(parsed.eid_path, parsed.segments):
            spool.append(
                eid,
                segment.start_time,
                segment.end_time,
                arrival_sequence,
                parsed.trajectory_id,
            )
            arrival_sequence += 1
        spool_append_seconds += perf_counter() - append_started
        trajectory_count += 1
        source_segments += source_segment_count
        partitioned_segments += len(parsed.segments)

    try:
        if city.lower() == "beijing":
            if len(files) != 1:
                raise ValueError(f"expected one Beijing CSV, got {len(files)}")
            _set_large_csv_field_limit()
            with files[0].open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                for row_no, row in enumerate(reader, 2):
                    if limit is not None and trajectory_count >= limit:
                        break
                    points = _load_beijing_points(row["points_data"], row_no)
                    parsed, movement_count, _ = (
                        parse_cell_partitioned_beijing_points(
                            points, row_no, source_road, partition
                        )
                    )
                    if parsed is None:
                        continue
                    consume(parsed, movement_count)
                    if trajectory_count % 10_000 == 0:
                        print(
                            f"spooled {trajectory_count} Beijing trajectories; "
                            f"{partitioned_segments} partitioned records",
                            flush=True,
                        )
        else:
            for file_index, path in enumerate(files, 1):
                if limit is not None and trajectory_count >= limit:
                    break
                print(f"[{file_index}/{len(files)}] spool {path}", flush=True)
                data = load_relaxed_json(path)
                if not isinstance(data, list):
                    raise ValueError(f"top-level JSON must be list: {path}")
                for trajectory in data:
                    if limit is not None and trajectory_count >= limit:
                        break
                    parsed = parse_cell_partitioned_trajectory(
                        trajectory, source_road, partition
                    )
                    consume(parsed, len(trajectory[1]))
                del data
        spool.close()
        catalog_size = catalog.close()
    except BaseException:
        spool.close()
        try:
            catalog.stream.close()
        except Exception:
            pass
        raise
    return {
        "files": files,
        "trajectory_count": trajectory_count,
        "source_segment_count": source_segments,
        "partitioned_segment_count": partitioned_segments,
        "record_count": arrival_sequence,
        "spool_append_seconds": spool_append_seconds,
        "catalog_size_bytes": catalog_size,
    }


def _set_digest(raw: bytearray, index: int, digest: bytes) -> None:
    if len(digest) != 32:
        raise ValueError("digest must be 32 bytes")
    position = index * 32
    raw[position:position + 32] = digest


def build_disk_smt_baseline(
    *,
    road,
    spool: ShardedEntrySpool,
    edge_data_file: str | Path,
    theta: int,
    trajectory_files=(),
    block_cache_bytes: int = 512 * 1024 * 1024,
    smt_cache_bytes: int = 1024 * 1024 * 1024,
    smt_data_file: str | Path | None = None,
):
    """Build the unchanged authenticated structure one disk shard at a time."""
    spool.close()
    edge_count = road.max_eid + 1
    entry_counts = array("I", [0]) * edge_count
    unique_counts = array("I", [0]) * edge_count
    offsets = array("Q", [0]) * edge_count
    lengths = array("Q", [0]) * edge_count
    entry_roots_raw = bytearray(32 * edge_count)
    smt_roots_raw = bytearray(32 * edge_count)
    edge_hashes_raw = bytearray(32 * edge_count)
    block_hashes_raw = bytearray(32 * edge_count)
    smt_offsets = array("Q", [0]) * edge_count
    smt_lengths = array("Q", [0]) * edge_count
    empty_smt_root = EdgeSMT.build([]).root_hash

    for eid in road.valid_edge_ids():
        _set_digest(entry_roots_raw, eid, ENTRY_LIST_TAIL)
        _set_digest(smt_roots_raw, eid, empty_smt_root)
        _set_digest(
            edge_hashes_raw,
            eid,
            hash_smt_edge(eid, 0, ENTRY_LIST_TAIL, empty_smt_root),
        )

    destination = Path(edge_data_file).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    smt_destination = (
        Path(smt_data_file).resolve() if smt_data_file is not None else None
    )
    smt_temporary = (
        smt_destination.with_suffix(smt_destination.suffix + ".tmp")
        if smt_destination is not None
        else None
    )
    if smt_destination is not None:
        smt_destination.parent.mkdir(parents=True, exist_ok=True)
    processed_records = 0
    populated_edges = 0
    data_hasher = hashlib.sha256()
    edge_build_started = perf_counter()
    smt_persistence_seconds = 0.0

    smt_context = (
        smt_temporary.open("wb") if smt_temporary is not None else nullcontext(None)
    )
    with temporary.open("wb") as output, smt_context as smt_output:
        header = _EDGE_DATA_HEADER.pack(EDGE_DATA_MAGIC, 1)
        output.write(header)
        data_hasher.update(header)
        if smt_output is not None:
            smt_output.write(_SMT_DATA_HEADER.pack(SMT_DATA_MAGIC, 1))

        for shard_index, path in enumerate(spool.paths, 1):
            raw = path.read_bytes()
            if len(raw) % _SPOOL_RECORD.size:
                raise ValueError(f"truncated spool shard: {path}")
            grouped: dict[int, list[tuple[int, int, int, bytes]]] = {}
            for eid, start, end, sequence, trajectory_id in (
                _SPOOL_RECORD.iter_unpack(raw)
            ):
                grouped.setdefault(eid, []).append(
                    (start, sequence, end, trajectory_id)
                )
            del raw

            for eid in sorted(grouped):
                rows = grouped[eid]
                rows.sort(key=lambda row: (row[0], row[1]))
                unique_ids = set(row[3] for row in rows)
                smt = EdgeSMT.build(unique_ids)
                if smt_output is not None:
                    persistence_started = perf_counter()
                    smt_offsets[eid] = smt_output.tell()
                    pickle.dump(smt, smt_output, protocol=5)
                    smt_lengths[eid] = smt_output.tell() - smt_offsets[eid]
                    smt_persistence_seconds += (
                        perf_counter() - persistence_started
                    )
                keyed_entries = [
                    (key_for_id(trajectory_id), start, end)
                    for start, _, end, trajectory_id in rows
                ]
                del rows
                state = ENTRY_LIST_TAIL
                for key, start, end in reversed(keyed_entries):
                    state = entry_list_prepend(hash_entry(key, start, end), state)

                sorted_ids = sorted(unique_ids, key=key_for_id)
                count = len(keyed_entries)
                unique_count = len(sorted_ids)
                if count > 0xFFFFFFFF or unique_count > 0xFFFFFFFF:
                    raise ValueError(f"edge {eid} count exceeds uint32")
                entry_counts[eid] = count
                unique_counts[eid] = unique_count
                offsets[eid] = output.tell()
                lengths[eid] = (
                    _BLOCK_HEADER.size
                    + count * _ENTRY.size
                    + unique_count * 32
                )

                block_hasher = hashlib.sha256()

                def write_piece(piece: bytes) -> None:
                    output.write(piece)
                    data_hasher.update(piece)
                    block_hasher.update(piece)

                write_piece(_BLOCK_HEADER.pack(count, unique_count))
                for key, start, end in keyed_entries:
                    write_piece(_ENTRY.pack(key, start, end))
                for trajectory_id in sorted_ids:
                    write_piece(trajectory_id)

                _set_digest(block_hashes_raw, eid, block_hasher.digest())
                _set_digest(entry_roots_raw, eid, state)
                _set_digest(smt_roots_raw, eid, smt.root_hash)
                _set_digest(
                    edge_hashes_raw,
                    eid,
                    hash_smt_edge(eid, count, state, smt.root_hash),
                )
                processed_records += count
                populated_edges += 1

            if shard_index % max(1, spool.shard_count // 20) == 0:
                print(
                    f"edge build shards {shard_index}/{spool.shard_count}; "
                    f"records={processed_records}; populated_edges={populated_edges}",
                    flush=True,
                )
            del grouped

        output.flush()
        os.fsync(output.fileno())
        if smt_output is not None:
            persistence_started = perf_counter()
            smt_output.flush()
            os.fsync(smt_output.fileno())
            smt_persistence_seconds += perf_counter() - persistence_started

    if processed_records != spool.record_count:
        raise RuntimeError(
            f"processed {processed_records} records, spooled {spool.record_count}"
        )
    temporary.replace(destination)
    if smt_destination is not None and smt_temporary is not None:
        smt_temporary.replace(smt_destination)
    edge_build_seconds = perf_counter() - edge_build_started

    spatial_started = perf_counter()
    edge_hashes = DigestArray(edge_hashes_raw)
    tree = BinarySpatialTreeBuilder(
        road,
        int(theta),
        edge_trajectory_counts=unique_counts,
    ).build()
    spatial_root = authenticate_spatial_tree(tree, edge_hashes)
    spatial_seconds = perf_counter() - spatial_started

    store = DiskEdgeEntryStore(
        destination,
        entry_counts,
        unique_counts,
        offsets,
        lengths,
        DigestArray(block_hashes_raw),
        cache_bytes=block_cache_bytes,
    )
    index = DiskSmtBaselineIndex(
        road=road,
        store=store,
        tree=tree,
        edge_entry_roots=DigestArray(entry_roots_raw),
        edge_hashes=edge_hashes,
        root_hash=spatial_root,
        theta=tree.theta,
        trajectory_files=trajectory_files,
        edge_smt_roots=DigestArray(smt_roots_raw),
        smt_cache_bytes=smt_cache_bytes,
        smt_data_file=smt_destination,
        smt_offsets=smt_offsets if smt_destination is not None else None,
        smt_lengths=smt_lengths if smt_destination is not None else None,
    )
    smt_data_size = (
        smt_destination.stat().st_size if smt_destination is not None else 0
    )
    return index, {
        "edge_build_seconds": edge_build_seconds,
        "spatial_tree_seconds": spatial_seconds,
        "edge_data_size_bytes": destination.stat().st_size,
        "smt_data_size_bytes": smt_data_size,
        "smt_persistence_seconds": smt_persistence_seconds,
        "edge_data_sha256": data_hasher.hexdigest(),
        "populated_edge_count": populated_edges,
        "record_count": processed_records,
    }
