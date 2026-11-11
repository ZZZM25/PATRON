from __future__ import annotations

import json
import sqlite3
import struct
import zlib
from pathlib import Path

from kzg.canonical import normalize_records
from kzg.config import SLOTS_PER_DAY
from kzg.field import FR_MODULUS,fr_from_bytes,fr_to_bytes
from kzg.models import SegmentRecord


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS edge_commitments (
    edge_id INTEGER PRIMARY KEY,
    commitment BLOB NOT NULL,
    is_empty INTEGER NOT NULL,
    nonempty_slots INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS slot_records (
    edge_id INTEGER NOT NULL,
    slot INTEGER NOT NULL,
    records BLOB NOT NULL,
    slot_value BLOB NOT NULL,
    PRIMARY KEY(edge_id,slot)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS edge_polynomials (
    edge_id INTEGER PRIMARY KEY,
    coefficients BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS shared_polynomials (
    name TEXT PRIMARY KEY,
    coefficients BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_cell_leaves (
    cell_id TEXT PRIMARY KEY,
    leaf_hash BLOB NOT NULL
);
"""

_SQLITE_EDGE_CHUNK_SIZE = 900


class DailyStore:
    """保存某一天的轨迹数据、多项式和KZG承诺。"""

    def __init__(self,path: str | Path,readonly: bool = False):
        self.path = Path(path)

        if readonly:
            uri = self.path.resolve().as_uri() + "?mode=ro&immutable=1"
            self.conn = sqlite3.connect(uri,uri=True,check_same_thread=False)
            self.conn.execute("PRAGMA query_only=ON")
        else:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.conn = sqlite3.connect(self.path,check_same_thread=False)
            self.conn.executescript(_SCHEMA)

        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """关闭数据库连接。"""
        self.conn.close()

    def __enter__(self) -> DailyStore:
        return self

    def __exit__(self,exc_type,exc,tb) -> None:
        self.close()

    def commit(self) -> None:
        """提交当前事务。"""
        self.conn.commit()

    def _configured_slot_count(self) -> int:
        """Return the authenticated dataset slot count, with daily compatibility."""
        return int(self.get_metadata_or("slot_count", SLOTS_PER_DAY))

    def set_metadata(self,key: str,value) -> None:
        """保存一项JSON格式的元数据。"""
        raw = json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
        self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",(key,raw))

    def get_metadata(self,key: str):
        """读取一项元数据。"""
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?",(key,)).fetchone()

        if row is None:
            raise KeyError(key)

        return json.loads(bytes(row["value"]).decode("utf-8"))

    def get_metadata_or(self,key: str,default=None):
        """读取可选元数据；旧数据库没有该键时返回默认值。"""
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?",(key,)).fetchone()

        if row is None:
            return default

        return json.loads(bytes(row["value"]).decode("utf-8"))

    @staticmethod
    def pack_records(records: list[SegmentRecord]) -> bytes:
        """规范化并压缩轨迹记录。"""
        normalized = normalize_records(records)
        raw = json.dumps([record.as_list() for record in normalized],separators=(",",":")).encode("utf-8")
        return zlib.compress(raw,level=6)

    @staticmethod
    def unpack_records(blob: bytes) -> list[SegmentRecord]:
        """解压并恢复轨迹记录。"""
        values = json.loads(zlib.decompress(blob).decode("utf-8"))
        return [SegmentRecord.from_list(value) for value in values]

    @staticmethod
    def pack_coefficients(coefficients: list[int]) -> bytes:
        """把有限域多项式系数保存成固定长度二进制。"""
        if not coefficients:
            raise ValueError("polynomial coefficients cannot be empty")

        normalized = [int(value) % FR_MODULUS for value in coefficients]
        return struct.pack(">I",len(normalized)) + b"".join(fr_to_bytes(value) for value in normalized)

    @staticmethod
    def unpack_coefficients(blob: bytes) -> list[int]:
        """从新格式原始二进制或旧格式zlib数据恢复多项式系数。"""
        raw = bytes(blob)

        if len(raw) < 4 or 4 + struct.unpack(">I",raw[:4])[0] * 32 != len(raw):
            raw = zlib.decompress(raw)

        if len(raw) < 4:
            raise ValueError("truncated polynomial blob")

        count = struct.unpack(">I",raw[:4])[0]
        body = raw[4:]

        if count == 0:
            raise ValueError("polynomial coefficient list cannot be empty")

        if len(body) != count * 32:
            raise ValueError("invalid polynomial coefficient blob length")

        return [fr_from_bytes(body[index:index + 32]) for index in range(0,len(body),32)]

    def put_slot(self,edge_id: int,slot: int,records: list[SegmentRecord],value: int) -> None:
        """保存一个非空时间槽。"""
        if not 0 <= slot < self._configured_slot_count():
            raise ValueError(f"invalid time slot: {slot}")

        self.conn.execute("INSERT OR REPLACE INTO slot_records(edge_id,slot,records,slot_value) VALUES (?,?,?,?)",(int(edge_id),slot,self.pack_records(records),fr_to_bytes(value)))

    def get_slot(self,edge_id: int,slot: int) -> tuple[list[SegmentRecord],int] | None:
        """读取一个时间槽；没有保存时返回None。"""
        row = self.conn.execute("SELECT records,slot_value FROM slot_records WHERE edge_id=? AND slot=?",(int(edge_id),int(slot))).fetchone()

        if row is None:
            return None

        records = self.unpack_records(bytes(row["records"]))
        value = fr_from_bytes(bytes(row["slot_value"]))
        return records,value

    def get_edge_slots(self,edge_id: int) -> dict[int,tuple[list[SegmentRecord],int]]:
        """读取一条道路的全部非空时间槽。"""
        rows = self.conn.execute("SELECT slot,records,slot_value FROM slot_records WHERE edge_id=? ORDER BY slot",(int(edge_id),))
        return {int(row["slot"]):(self.unpack_records(bytes(row["records"])),fr_from_bytes(bytes(row["slot_value"]))) for row in rows}

    def get_edge_slot_values(self,edge_ids: list[int]) -> dict[int,dict[int,int]]:
        """批量读取道路全天的非空槽值，不解压轨迹记录。

        稀疏证明只需要这些有限域值。返回结果也包含没有非空槽的道路，
        其值为空字典，方便调用方完整检查预加载状态。
        """
        normalized_edge_ids = sorted(set(int(edge_id) for edge_id in edge_ids))
        result: dict[int,dict[int,int]] = {edge_id: {} for edge_id in normalized_edge_ids}

        for offset in range(0,len(normalized_edge_ids),_SQLITE_EDGE_CHUNK_SIZE):
            edge_chunk = normalized_edge_ids[offset:offset + _SQLITE_EDGE_CHUNK_SIZE]
            if not edge_chunk:
                continue
            placeholders = ",".join("?" for _ in edge_chunk)
            sql = f"SELECT edge_id,slot,slot_value FROM slot_records WHERE edge_id IN ({placeholders}) ORDER BY edge_id,slot"
            for row in self.conn.execute(sql,edge_chunk):
                edge_id = int(row["edge_id"])
                result[edge_id][int(row["slot"])] = fr_from_bytes(bytes(row["slot_value"]))

        return result

    def get_slots(self,edge_ids: list[int],slots: list[int]) -> dict[tuple[int,int],tuple[list[SegmentRecord],int]]:
        """批量读取多条道路在指定分钟槽内的全部非空记录。"""
        normalized_edge_ids = sorted(set(int(edge_id) for edge_id in edge_ids))
        normalized_slots = sorted(set(int(slot) for slot in slots))

        if not normalized_edge_ids or not normalized_slots:
            return {}

        if (
            normalized_slots[0] < 0
            or normalized_slots[-1] >= self._configured_slot_count()
        ):
            raise ValueError("invalid minute slot in bulk request")

        requested_slots = set(normalized_slots)
        minimum_slot = normalized_slots[0]
        maximum_slot = normalized_slots[-1]
        result = {}

        for offset in range(0,len(normalized_edge_ids),_SQLITE_EDGE_CHUNK_SIZE):
            edge_chunk = normalized_edge_ids[offset:offset + _SQLITE_EDGE_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in edge_chunk)
            sql = f"SELECT edge_id,slot,records,slot_value FROM slot_records WHERE edge_id IN ({placeholders}) AND slot BETWEEN ? AND ? ORDER BY edge_id,slot"
            parameters = [*edge_chunk,minimum_slot,maximum_slot]

            for row in self.conn.execute(sql,parameters):
                edge_id = int(row["edge_id"])
                slot = int(row["slot"])

                if slot not in requested_slots:
                    continue

                records = self.unpack_records(bytes(row["records"]))
                value = fr_from_bytes(bytes(row["slot_value"]))
                result[(edge_id,slot)] = (records,value)

        return result

    def put_edge_commitment(self,edge_id: int,commitment: bytes,is_empty: bool,nonempty_slots: int) -> None:
        """保存一条道路的KZG承诺。"""
        if len(commitment) != 48:
            raise ValueError("compressed G1 commitment must contain 48 bytes")

        self.conn.execute("INSERT OR REPLACE INTO edge_commitments(edge_id,commitment,is_empty,nonempty_slots) VALUES (?,?,?,?)",(int(edge_id),bytes(commitment),int(is_empty),int(nonempty_slots)))

    def get_edge_commitment_info(self,edge_id: int) -> tuple[bytes,bool,int]:
        """读取道路承诺、是否为空以及非空槽数量。"""
        row = self.conn.execute("SELECT commitment,is_empty,nonempty_slots FROM edge_commitments WHERE edge_id=?",(int(edge_id),)).fetchone()

        if row is None:
            raise KeyError(f"edge commitment not found: {edge_id}")

        return bytes(row["commitment"]),bool(row["is_empty"]),int(row["nonempty_slots"])

    def all_edge_commitments(self) -> list[tuple[int,bytes]]:
        """按edge_id顺序读取全部道路承诺。"""
        rows = self.conn.execute("SELECT edge_id,commitment FROM edge_commitments ORDER BY edge_id")
        return [(int(row["edge_id"]),bytes(row["commitment"])) for row in rows]

    def put_edge_polynomial(self,edge_id: int,coefficients: list[int]) -> None:
        """保存一条非空道路的多项式系数。"""
        packed = self.pack_coefficients(coefficients)
        self.put_edge_polynomial_blob(edge_id,packed)

    def put_edge_polynomial_blob(self,edge_id: int,packed_coefficients: bytes) -> None:
        """保存已经由工作进程规范打包的多项式系数。"""
        packed = bytes(packed_coefficients)

        if not packed:
            raise ValueError("packed polynomial coefficients cannot be empty")

        self.conn.execute("INSERT OR REPLACE INTO edge_polynomials(edge_id,coefficients) VALUES (?,?)",(int(edge_id),packed))

    def put_shared_polynomial(self,name: str,coefficients: list[int]) -> None:
        """保存可供多条道路共享的多项式。"""
        packed = self.pack_coefficients(coefficients)
        self.conn.execute("INSERT OR REPLACE INTO shared_polynomials(name,coefficients) VALUES (?,?)",(name,packed))

    def get_shared_polynomial(self,name: str) -> list[int]:
        """读取共享多项式。"""
        row = self.conn.execute("SELECT coefficients FROM shared_polynomials WHERE name=?",(name,)).fetchone()

        if row is None:
            raise KeyError(f"shared polynomial not found: {name}")

        return self.unpack_coefficients(bytes(row["coefficients"]))

    def get_edge_polynomial(self,edge_id: int) -> list[int]:
        """读取道路多项式，空道路自动使用共享空多项式。"""
        row = self.conn.execute("SELECT coefficients FROM edge_polynomials WHERE edge_id=?",(int(edge_id),)).fetchone()

        if row is not None:
            return self.unpack_coefficients(bytes(row["coefficients"]))

        commitment,is_empty,nonempty_slots = self.get_edge_commitment_info(edge_id)

        if is_empty:
            return self.get_shared_polynomial("empty")

        raise KeyError(f"active edge polynomial not found: {edge_id}")

    def get_edge_polynomials(self,edge_ids: list[int]) -> dict[int,list[int]]:
        """批量读取道路多项式，空道路复用同一个共享空多项式。"""
        normalized_edge_ids = sorted(set(int(edge_id) for edge_id in edge_ids))

        if not normalized_edge_ids:
            return {}

        result = {}
        empty_coefficients = None

        for offset in range(0,len(normalized_edge_ids),_SQLITE_EDGE_CHUNK_SIZE):
            edge_chunk = normalized_edge_ids[offset:offset + _SQLITE_EDGE_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in edge_chunk)
            sql = f"SELECT ec.edge_id,ec.is_empty,ep.coefficients FROM edge_commitments AS ec LEFT JOIN edge_polynomials AS ep ON ep.edge_id=ec.edge_id WHERE ec.edge_id IN ({placeholders}) ORDER BY ec.edge_id"

            for row in self.conn.execute(sql,edge_chunk):
                edge_id = int(row["edge_id"])
                coefficients_blob = row["coefficients"]

                if coefficients_blob is not None:
                    result[edge_id] = self.unpack_coefficients(bytes(coefficients_blob))
                    continue

                if bool(row["is_empty"]):
                    if empty_coefficients is None:
                        empty_coefficients = self.get_shared_polynomial("empty")

                    result[edge_id] = empty_coefficients
                    continue

                raise KeyError(f"active edge polynomial not found: {edge_id}")

        missing_edge_ids = [edge_id for edge_id in normalized_edge_ids if edge_id not in result]

        if missing_edge_ids:
            raise KeyError(f"edge commitment not found: {missing_edge_ids[0]}")

        return result
    def put_daily_cell_leaf(self, cell_id: str, leaf_hash: bytes) -> None:
        """Store one non-empty Geohash cell leaf for the daily authenticated root."""
        if not cell_id or any(bit not in "01" for bit in cell_id):
            raise ValueError("cell_id must be a binary Geohash prefix")
        if len(leaf_hash) != 32:
            raise ValueError("cell leaf hash must contain 32 bytes")
        self.conn.execute(
            "INSERT OR REPLACE INTO daily_cell_leaves(cell_id,leaf_hash) VALUES (?,?)",
            (cell_id, bytes(leaf_hash)),
        )

    def all_daily_cell_leaves(self) -> dict[str, bytes]:
        """Read all materialized non-empty daily cell leaves by prefix."""
        rows = self.conn.execute("SELECT cell_id,leaf_hash FROM daily_cell_leaves ORDER BY cell_id")
        return {str(row["cell_id"]): bytes(row["leaf_hash"]) for row in rows}

