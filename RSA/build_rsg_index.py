from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import secrets
import time
from datetime import date, timedelta
from pathlib import Path


BUILDER_MODULE = "rsa_index_builder"


def validate_binary(path: Path, builder) -> dict:
    counts = {
        "date_count": 0,
        "edge_count": 0,
        "edge_trajectory_count": 0,
        "occurrence_count": 0,
        "dates": [],
    }
    epoch = date(1970, 1, 1)
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"RSGA":
            raise ValueError(f"非法索引 magic: {magic!r}")
        version = builder.read_u16(f)
        base = builder.read_u8(f)
        token_bytes = builder.read_u8(f)
        day_seconds = builder.read_u32(f)
        timezone_offset_seconds = builder.read_i32(f)
        hprime_bits = builder.read_u16(f)
        _reserved2 = builder.read_u16(f)
        n = builder.bytes_to_mpz(builder.read_varbytes(f))
        g = builder.bytes_to_mpz(builder.read_varbytes(f))
        modulus_bytes = (n.bit_length() + 7) // 8
        occurrence_record_bytes = 5 + (modulus_bytes if version >= 2 else 0)
        date_count = builder.read_varint(f)
        counts["date_count"] = date_count

        for _ in range(date_count):
            day_idx = builder.read_varint(f)
            day_string = (epoch + timedelta(days=day_idx)).isoformat()
            counts["dates"].append(day_string)
            _data_xp = builder.read_varbytes(f)
            _data_ac = builder.read_varbytes(f)
            edge_count = builder.read_varint(f)
            counts["edge_count"] += edge_count
            for _ in range(edge_count):
                _edge_id = builder.read_varint(f)
                _edge_xp = builder.read_varbytes(f)
                _edge_ac = builder.read_varbytes(f)
                if version >= 3:
                    edge_witnesses = f.read(2 * modulus_bytes)
                    if len(edge_witnesses) != 2 * modulus_bytes:
                        raise EOFError("v3 edge witness 数据被截断")
                trajectory_count = builder.read_varint(f)
                counts["edge_trajectory_count"] += trajectory_count
                for _ in range(trajectory_count):
                    _trajectory_id = builder.read_varint(f)
                    if version >= 3:
                        token = f.read(token_bytes)
                        if len(token) != token_bytes:
                            raise EOFError("v3 opaque token 数据被截断")
                    occurrence_count = builder.read_varint(f)
                    counts["occurrence_count"] += occurrence_count
                    expected_bytes = occurrence_count * occurrence_record_bytes
                    payload = f.read(expected_bytes)
                    if len(payload) != expected_bytes:
                        raise EOFError("索引 occurrence 数据被截断")
        trailing = f.read(1)
        if trailing:
            raise ValueError("索引末尾存在未解析数据")

    counts.update(
        {
            "magic": magic.decode("ascii"),
            "version": version,
            "base": base,
            "day_seconds": day_seconds,
            "timezone_offset_seconds": timezone_offset_seconds,
            "hprime_bits": hprime_bits,
            "n_bits": n.bit_length(),
            "g_bits": g.bit_length(),
            "has_member_witnesses": version >= 2,
            "has_private_tokens": version >= 3,
            "opaque_token_bytes": token_bytes if version >= 3 else 0,
            "member_witness_bytes": modulus_bytes if version >= 2 else 0,
            "file_size_bytes": path.stat().st_size,
        }
    )
    return counts


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an RSA/RSG 50 m index for any prepared dataset unit"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--date", required=True, help="dataset unit date: YYYY-MM-DD")
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        help="映射后的轨迹 CSV；默认使用数据目录中的 traj-MM-DD.csv",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-name",
        default=None,
        help="输出文件名；默认生成隐私令牌、可信数量和离线 witness 的 v3 索引",
    )
    parser.add_argument(
        "--token-key-file",
        type=Path,
        help="数据拥有者的 32 字节令牌密钥；默认保存在数据目录且不发布给客户端",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_date = date.fromisoformat(args.date).isoformat()
    dataset_dir = args.dataset_dir.resolve()
    month_day = date.fromisoformat(dataset_date).strftime("%m-%d")
    trajectory_csv = (
        args.trajectory_csv.resolve()
        if args.trajectory_csv is not None
        else dataset_dir / f"traj-{month_day}.csv"
    )
    edge_csv = dataset_dir / "new_edges.csv"
    output_name = args.output_name or f"{dataset_date}.rsg-50m-private-token.bin"
    output_bin = dataset_dir / output_name
    manifest_path = dataset_dir / f"{output_bin.stem}.manifest.json"
    token_key_path = (
        args.token_key_file.resolve()
        if args.token_key_file is not None
        else dataset_dir / f"{dataset_date}.owner-token-key.bin"
    )

    for path in (trajectory_csv, edge_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_bin.exists() and not args.force:
        raise FileExistsError(f"索引已存在；如需覆盖请加 --force: {output_bin}")

    builder = importlib.import_module(BUILDER_MODULE)
    builder.TRAJ_CSV_FILES = [str(trajectory_csv)]
    builder.OUTPUT_BIN = str(output_bin)
    builder.TIMEZONE = "UTC"
    builder.TZ_OFFSET_SECONDS = builder.get_tz_offset_seconds(builder.TIMEZONE)
    builder.NUM_WORKERS = max(1, args.workers)
    builder.CHUNK_SIZE = 32
    builder.INCLUDE_EMPTY_EDGES = False
    builder.EDGE_CSV = str(edge_csv)
    builder.PRINT_PROGRESS_EVERY_TRAJ = 1000
    builder.FUSE_PREPARE_AND_ACCUMULATOR = True
    builder.FUSED_CHUNK_SIZE = 1
    builder.PRECOMPUTE_MEMBER_WITNESSES = True
    builder.BINARY_VERSION = 3
    builder.OPAQUE_TOKEN_BYTES = 16
    if token_key_path.exists():
        token_key = token_key_path.read_bytes()
    else:
        token_key = secrets.token_bytes(32)
        token_key_path.write_bytes(token_key)
    if len(token_key) < 32:
        raise ValueError("token key 至少需要 32 字节")
    builder.OPAQUE_TOKEN_KEY = token_key
    # 将空边也认证到日根，客户端可验证查询窗口中的整条边没有被省略。
    builder.INCLUDE_EMPTY_EDGES = True
    # The original full-load check duplicates a very large index in memory.
    # A streaming structural validation is performed below instead.
    builder.CHECK_AFTER_WRITE = False

    print(f"========== 构建 {dataset_date} RSG 50m 索引 ==========")
    print(f"trajectory_csv={trajectory_csv}")
    print(f"edge_csv={edge_csv}")
    print(f"output_bin={output_bin}")
    print(f"workers={builder.NUM_WORKERS}")
    print(f"token_key_file={token_key_path} (owner-only, not included in manifest)")
    t0 = time.perf_counter()
    builder.main()
    build_seconds = time.perf_counter() - t0

    print("========== 流式校验索引 ==========")
    validation = validate_binary(output_bin, builder)
    if validation["dates"] != [dataset_date]:
        raise ValueError(f"索引日期不符合预期: {validation['dates']}")
    digest = sha256_file(output_bin)
    manifest = {
        "dataset_date": dataset_date,
        "timezone": "UTC",
        "cell_size_m": 50.0,
        "trajectory_csv": str(trajectory_csv),
        "edge_csv": str(edge_csv),
        "index_file": str(output_bin),
        "workers": builder.NUM_WORKERS,
        "privacy_scheme": "opaque-edge-trajectory-token-v1",
        "owner_token_key_file": str(token_key_path),
        "build_seconds_outer": build_seconds,
        "sha256": digest,
        "validation": validation,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[索引构建完成] {output_bin}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
