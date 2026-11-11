# -*- coding: utf-8 -*-
"""
one_edge_claim_verify_member_batch.py

范围查询验证版本：
1) 客户端只给日期、空间范围和时间范围，服务端从索引生成结果集合 R
2) member 部分改成 batch membership：
   - 每个 member 只取 1 条满足查询区间的明文 tr_time
   - x_member_batch = ∏ prime(tr_time_plaintext)
   - w_member_batch = g^(edge_xp / x_member_batch) mod n
3) 正常查询不会接收客户端给定的 R，因此不再返回错误 claim item 证明
4) 不在 R 里的所有真实轨迹统一做一个 completeness batch：
   - X_batch = ∏ x_traj
   - gcd(edge_xp, X_batch)=1 -> batch non-membership witness
   - gcd(edge_xp, X_batch)!=1 -> claim 不完整

重要说明
--------
member 批量这里按你的要求做成：
- 明文就返回已经存好的 tr_time: edge|traj|{start_leaf,end_leaf}
- 每个明文只做一次 hash_to_prime
- 不再对 member 去恢复 pair 再乘进去

这在你当前 ADS 下是成立的，因为 edge_xp 里本来就包含了每条 tr_time 的 prime 因子。
验证 member_batch 时：
- 先检查每条明文 tr_time 解析出来的 [s,e] 是否确实与查询区间相交
- 再令 x_member_batch = ∏ prime(tr_time)
- 检查 w_member_batch^(x_member_batch) mod n == edge_ac
"""

import struct
import hashlib
import time
import os
import json
import heapq
import multiprocessing as mp
from datetime import date, timedelta
from typing import Dict, List, Set, Optional, Tuple

import gmpy2
from gmpy2 import mpz, next_prime, gcd, gcdext, powmod, invert

from vo_binary import (
    VOFormatError,
    decode_completeness,
    decode_private_completeness,
    decode_invalid_items,
    decode_member_batch,
    encode_completeness,
    encode_private_completeness,
    encode_invalid_items,
    encode_member_batch,
)


# ============================================================
# 输入区：你只改这里
# ============================================================

INPUT_BIN = "traj-10-1-rsa-fast-with-loader.bin"

CLAIM_DATE = "2018-10-01"
CLAIM_QS = 36789
CLAIM_QE = 58999

# 手动指定多条边；留空则自动从 CLAIM_DATE 里挑轨迹数量最多的边
QUERY_EDGE_IDS = []
AUTO_PICK_EDGE_COUNT = 10

NUM_WORKERS = 12

# 下面两个配置只供 simulate_claim_r_from_edge 攻击/回归测试使用；
# 正常查询始终返回全部真实命中轨迹。
# "matched_first_k"  : 从真实命中轨迹里按 traj_id 排序后取前 K 个
# "matched_all"      : 直接把所有真实命中轨迹都当作 R
AUTO_R_MODE = "matched_first_k"
AUTO_R_K = 50

# 输出摘要
SUMMARY_JSON = "multi_edge_claim_verify_member_batch_parallel_balanced_minimal_vo_summary.json"
PRINT_EDGE_ASSIGNMENT = True

# 加速开关：不改变验证数学逻辑，只减少重复计算和无用 VO 体积
# 原始 completeness 的 omitted_detail 里保存 query_pair_seeds，但 verify_completeness_batch 实际只用 x_traj。
# 100 条边时，保存所有 seed 字符串会非常慢、非常占内存；需要完整调试 VO 时再改成 True。
STORE_QUERY_PAIR_SEEDS_IN_VO = False  # 最简 VO 必须保持 False

# query-pair seed 基本不会重复，给它们进 mapper.cache 反而浪费内存和字典开销。
# 这里直接 hash_to_prime，不缓存；member / invalid 仍然用原 mapper 逻辑。
USE_UNCACHED_QUERY_PAIR_PRIME = True

# 安全默认值：验证端必须从受信任 ADS、客户端查询和结果集合独立重建
# completeness 的 X_batch。关闭后只验证服务端自报 X_batch 的代数关系，
# 会重新引入 X=1, a=0, d=g 的完整性绕过漏洞。
STRICT_COMPLETENESS_BINDING = True

BIGINT_AS_HEX = False  # 重要：验证不需要 bigint 字符串，必须关掉，省内存
EPOCH_DATE = date(1970, 1, 1)
EDGE_META_KEYS = {
    "edge_xp",
    "edge_ac",
    "edge_xp_str",
    "edge_ac_str",
    "traj_count",
    "root_witness",
    "identity_witness",
}


# ============================================================
# 二进制 ADS 读取
# ============================================================

def read_u8(f) -> int:
    return struct.unpack(">B", f.read(1))[0]


def read_u16(f) -> int:
    return struct.unpack(">H", f.read(2))[0]


def read_u32(f) -> int:
    return struct.unpack(">I", f.read(4))[0]


def read_i32(f) -> int:
    return struct.unpack(">i", f.read(4))[0]


def read_varint(f) -> int:
    shift = 0
    result = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("读取 varint 时提前结束")
        b = b[0]
        result |= ((b & 0x7F) << shift)
        if not (b & 0x80):
            return result
        shift += 7


def read_varbytes(f) -> bytes:
    n = read_varint(f)
    return f.read(n)


def bytes_to_mpz(b: bytes) -> mpz:
    return mpz(int.from_bytes(b, "big"))


def unpack_occurrence(b: bytes):
    if len(b) != 5:
        raise ValueError("occurrence 必须是 5 字节")
    v = int.from_bytes(b, "big")
    start_second = (v >> 17) & ((1 << 17) - 1)
    end_second = v & ((1 << 17) - 1)
    return start_second, end_second


def bigint_to_str(x, as_hex=True) -> str:
    x = mpz(x)
    return x.digits(16) if as_hex else x.digits(10)


def calc_depth(domain_size: int, base: int) -> int:
    depth = 0
    size = 1
    while size < domain_size:
        size *= base
        depth += 1
    return depth


def to_base_fixed(value: int, base: int, depth: int) -> str:
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value < 0:
        raise ValueError(f"value 不能小于 0: {value}")
    if not (2 <= base <= len(digits)):
        raise ValueError("base 必须在 2~36 之间")
    arr = []
    x = value
    for _ in range(depth):
        arr.append(digits[x % base])
        x //= base
    if x != 0:
        raise ValueError(f"value={value} 超出 depth={depth}, base={base} 可表示范围")
    return "".join(reversed(arr))


def edge_tr_seed(edge_id, trajectory_id):
    return f"{edge_id}|{trajectory_id}"


def tr_time_seed(edge_id, trajectory_id, start_code, end_code):
    return f"{edge_id}|{trajectory_id}|{{{start_code},{end_code}}}"


def private_identity_seed(day_idx: int, edge_id: int, token_hex: str) -> str:
    return f"RSG-ID-V3|{int(day_idx)}|{int(edge_id)}|{token_hex}"


def private_occurrence_seed(
        day_idx: int,
        edge_id: int,
        traj_id: int,
        token_hex: str,
        start_code: str,
        end_code: str,
) -> str:
    return (
        f"RSG-OCC-V3|{int(day_idx)}|{int(edge_id)}|{int(traj_id)}|"
        f"{token_hex}|{{{start_code},{end_code}}}"
    )


def private_time_pair_seed(
        day_idx: int,
        edge_id: int,
        token_hex: str,
        start_prefix: str,
        end_prefix: str,
) -> str:
    return (
        f"RSG-TIME-V3|{int(day_idx)}|{int(edge_id)}|{token_hex}|"
        f"{start_prefix}|{end_prefix}"
    )


def private_edge_descriptor_seed(
        day_idx: int,
        edge_id: int,
        traj_count: int,
        edge_ac: mpz,
) -> str:
    return (
        f"RSG-EDGE-DESCRIPTOR-V3|{int(day_idx)}|{int(edge_id)}|"
        f"{int(traj_count)}|{mpz(edge_ac).digits(16)}"
    )


def skip_varbytes(f) -> int:
    n = read_varint(f)
    if n:
        f.seek(n, 1)
    return n


def skip_occurrences(f, occ_count: int, rsa: dict) -> None:
    if occ_count:
        record_bytes = 5 + (rsa["modulus_bytes"] if rsa["format_version"] >= 2 else 0)
        f.seek(occ_count * record_bytes, 1)


def date_to_day_idx(date_str: str) -> int:
    return (date.fromisoformat(date_str) - EPOCH_DATE).days


def read_header_keep_rsa(f) -> dict:
    magic = f.read(4)
    if magic != b"RSGA":
        raise ValueError(f"不是本程序生成的二进制文件，magic={magic!r}")

    version = read_u16(f)
    if version not in (1, 2, 3):
        raise ValueError(f"暂不支持 version={version}")

    base = read_u8(f)
    token_bytes = read_u8(f)
    day_seconds = read_u32(f)
    tz_offset_seconds = read_i32(f)
    hprime_bits = read_u16(f)
    _reserved2 = read_u16(f)

    n = bytes_to_mpz(read_varbytes(f))
    g = bytes_to_mpz(read_varbytes(f))
    date_count = read_varint(f)
    depth = calc_depth(day_seconds, base)

    return {
        "n": n,
        "g": g,
        "format_version": version,
        "modulus_bytes": (n.bit_length() + 7) // 8,
        "has_member_witnesses": version >= 2,
        "has_private_tokens": version >= 3,
        "token_bytes": int(token_bytes) if version >= 3 else 0,
        "base": base,
        "day_seconds": day_seconds,
        "tz_offset_seconds": tz_offset_seconds,
        "hprime_bits": hprime_bits,
        "depth": depth,
        "date_count": date_count,
        "bigint_encoding": "mpz_only_no_string",
    }


def skip_one_edge(f, rsa: dict) -> None:
    _edge_id = read_varint(f)
    skip_varbytes(f)  # edge_xp
    skip_varbytes(f)  # edge_ac
    if rsa["format_version"] >= 3:
        f.seek(2 * rsa["modulus_bytes"], 1)  # root witness + identity witness
    traj_count = read_varint(f)
    for _ in range(traj_count):
        _traj_id = read_varint(f)
        if rsa["format_version"] >= 3:
            f.seek(rsa["token_bytes"], 1)
        occ_count = read_varint(f)
        skip_occurrences(f, occ_count, rsa)


def scan_edges_for_date(path: str, target_date: str) -> Tuple[dict, List[dict]]:
    """
    轻量扫描：只统计目标日期每条边的 traj_count / occ_count。
    不读取 edge_xp 大整数，不转字符串。
    用于自动挑边和按轨迹数量负载均衡。
    """
    target_day_idx = date_to_day_idx(target_date)

    with open(path, "rb") as f:
        rsa = read_header_keep_rsa(f)
        edge_infos = []

        for _ in range(rsa["date_count"]):
            day_idx = read_varint(f)
            skip_varbytes(f)  # data_xp
            skip_varbytes(f)  # data_ac
            edge_count = read_varint(f)

            if day_idx != target_day_idx:
                for _ in range(edge_count):
                    skip_one_edge(f, rsa)
                continue

            for _ in range(edge_count):
                edge_id = read_varint(f)
                edge_xp_bytes = skip_varbytes(f)  # edge_xp
                edge_ac_bytes = skip_varbytes(f)  # edge_ac
                if rsa["format_version"] >= 3:
                    f.seek(2 * rsa["modulus_bytes"], 1)
                traj_count = read_varint(f)

                occ_total = 0
                for _ in range(traj_count):
                    _traj_id = read_varint(f)
                    if rsa["format_version"] >= 3:
                        f.seek(rsa["token_bytes"], 1)
                    occ_count = read_varint(f)
                    occ_total += occ_count
                    skip_occurrences(f, occ_count, rsa)

                edge_infos.append({
                    "edge_id": int(edge_id),
                    "traj_count": int(traj_count),
                    "occ_total": int(occ_total),
                    "edge_xp_bytes": int(edge_xp_bytes),
                    "edge_ac_bytes": int(edge_ac_bytes),
                    "weight": int(max(1, traj_count)),
                })
            return rsa, edge_infos

    raise KeyError(f"date 不存在: {target_date}")


def load_binary_selected_edges(path: str, date_str: str, selected_edge_ids: Set[int]) -> dict:
    """
    只加载当前 worker 负责的边。
    这是这版最快/最省内存的关键：
    - 不加载无关日期
    - 不加载无关边
    - 不生成 edge_xp_str / edge_ac_str / data_xp_str 等字符串副本
    """
    target_day_idx = date_to_day_idx(date_str)
    selected_edge_ids = set(int(x) for x in selected_edge_ids)

    with open(path, "rb") as f:
        rsa = read_header_keep_rsa(f)
        result = {
            "rsa": {k: v for k, v in rsa.items() if k != "date_count"},
            "data": {date_str: {}},
            "trusted_root": None,
        }
        date_obj = result["data"][date_str]

        for _ in range(rsa["date_count"]):
            day_idx = read_varint(f)
            data_xp = bytes_to_mpz(read_varbytes(f))
            data_ac = bytes_to_mpz(read_varbytes(f))
            edge_count = read_varint(f)

            if day_idx != target_day_idx:
                for _ in range(edge_count):
                    skip_one_edge(f, rsa)
                continue

            result["trusted_root"] = {
                "date": date_str,
                "data_xp": data_xp,
                "data_ac": data_ac,
            }

            for _ in range(edge_count):
                edge_id = read_varint(f)

                if edge_id not in selected_edge_ids:
                    skip_varbytes(f)  # edge_xp
                    skip_varbytes(f)  # edge_ac
                    if rsa["format_version"] >= 3:
                        f.seek(2 * rsa["modulus_bytes"], 1)
                    traj_count = read_varint(f)
                    for _ in range(traj_count):
                        _traj_id = read_varint(f)
                        if rsa["format_version"] >= 3:
                            f.seek(rsa["token_bytes"], 1)
                        occ_count = read_varint(f)
                        skip_occurrences(f, occ_count, rsa)
                    continue

                edge_xp = bytes_to_mpz(read_varbytes(f))
                edge_ac = bytes_to_mpz(read_varbytes(f))
                root_witness = None
                identity_witness = None
                if rsa["format_version"] >= 3:
                    root_bytes = f.read(rsa["modulus_bytes"])
                    identity_bytes = f.read(rsa["modulus_bytes"])
                    if len(root_bytes) != rsa["modulus_bytes"] or len(identity_bytes) != rsa["modulus_bytes"]:
                        raise EOFError("v3 edge witness 数据被截断")
                    root_witness = bytes_to_mpz(root_bytes)
                    identity_witness = bytes_to_mpz(identity_bytes)
                traj_count = read_varint(f)

                edge_obj = {
                    "edge_xp": edge_xp,
                    "edge_ac": edge_ac,
                    "traj_count": int(traj_count),
                }
                if rsa["format_version"] >= 3:
                    edge_obj["root_witness"] = root_witness
                    edge_obj["identity_witness"] = identity_witness

                for _ in range(traj_count):
                    traj_id = read_varint(f)
                    token = b""
                    if rsa["format_version"] >= 3:
                        token = f.read(rsa["token_bytes"])
                        if len(token) != rsa["token_bytes"]:
                            raise EOFError("opaque token 数据被截断")
                    occ_count = read_varint(f)
                    detail_tr = []

                    for _ in range(occ_count):
                        occ = f.read(5)
                        start_second, end_second = unpack_occurrence(occ)
                        member_witness = None
                        if rsa["format_version"] >= 2:
                            witness_bytes = f.read(rsa["modulus_bytes"])
                            if len(witness_bytes) != rsa["modulus_bytes"]:
                                raise EOFError("member witness 数据被截断")
                            member_witness = bytes_to_mpz(witness_bytes)
                        start_code = to_base_fixed(start_second, rsa["base"], rsa["depth"])
                        end_code = to_base_fixed(end_second, rsa["base"], rsa["depth"])
                        tr_time = (
                            private_occurrence_seed(
                                target_day_idx,
                                edge_id,
                                traj_id,
                                token.hex(),
                                start_code,
                                end_code,
                            )
                            if rsa["format_version"] >= 3
                            else tr_time_seed(edge_id, traj_id, start_code, end_code)
                        )
                        detail = {
                            "start_second": start_second,
                            "end_second": end_second,
                            "tr_time": tr_time,
                        }
                        if member_witness is not None:
                            detail["member_witness"] = member_witness
                        detail_tr.append(detail)

                    edge_obj[str(traj_id)] = {
                        "edge_tr": (
                            private_identity_seed(target_day_idx, edge_id, token.hex())
                            if rsa["format_version"] >= 3
                            else edge_tr_seed(edge_id, traj_id)
                        ),
                        "opaque_token": token,
                        "detail_tr": detail_tr,
                    }

                date_obj[str(edge_id)] = edge_obj

            return result

    raise KeyError(f"date 不存在: {date_str}")


# 兼容原函数名，但这里只用于调试；多边并行主流程使用 load_binary_selected_edges
def load_binary_to_dict(path: str, as_hex: bool = BIGINT_AS_HEX) -> dict:
    rsa, edge_infos = scan_edges_for_date(path, CLAIM_DATE)
    all_edges = {x["edge_id"] for x in edge_infos}
    return load_binary_selected_edges(path, CLAIM_DATE, all_edges)


# ============================================================
# Prime mapper / 乘法 / 幂 / Bézout
# ============================================================

class PrimeMapper:
    def __init__(self, bits: int):
        self.bits = bits
        self.cache: Dict[str, mpz] = {}

    def hash_to_prime(self, seed: str) -> mpz:
        seed = str(seed)
        if seed in self.cache:
            return self.cache[seed]

        mask = (1 << self.bits) - 1
        counter = 0
        while True:
            digest = hashlib.sha256(f"{seed}|{counter}".encode("utf-8")).digest()
            x = int.from_bytes(digest, "big") & mask
            x |= (1 << (self.bits - 1))
            x |= 1
            p = next_prime(mpz(x))
            if p.bit_length() == self.bits:
                self.cache[seed] = p
                return p
            counter += 1


def product_tree(nums: List[mpz]) -> mpz:
    if not nums:
        return mpz(1)
    level = [mpz(x) for x in nums]
    while len(level) > 1:
        nxt = []
        n = len(level)
        for i in range(0, n - 1, 2):
            nxt.append(level[i] * level[i + 1])
        if n % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def signed_powmod(base: mpz, exp: int, mod: mpz) -> mpz:
    base = mpz(base)
    mod = mpz(mod)
    exp = int(exp)
    if exp >= 0:
        return powmod(base, exp, mod)
    inv = invert(base, mod)
    if inv == 0:
        raise ZeroDivisionError("base 在模 n 下不可逆")
    return powmod(inv, -exp, mod)


def bezout_nonmembership_witness(edge_xp: mpz, x: mpz, g: mpz, n: mpz) -> dict:
    gval, a, b = gcdext(edge_xp, x)
    if gval != 1:
        raise ValueError("gcd(edge_xp, x) != 1，无法做 non-membership")
    d = signed_powmod(g, int(b), n)
    return {"a": int(a), "b": int(b), "d": d}


def aggregate_precomputed_member_witnesses(
        representatives: List[mpz],
        witnesses: List[mpz],
        n: mpz,
) -> Optional[mpz]:
    """
    Aggregate W_A=g^(X/X_A) and W_B=g^(X/X_B) without knowing p/q.
    For u*X_B + v*X_A = 1:
        W_(A union B) = W_A^u * W_B^v mod n.
    Returns None on the negligible hash-to-prime collision/non-coprime case so
    the caller can fall back to the original exact quotient exponentiation.
    """
    if len(representatives) != len(witnesses):
        raise ValueError("representatives / witnesses 数量不一致")
    if not representatives:
        return None

    level = [(mpz(x), mpz(w)) for x, w in zip(representatives, witnesses)]
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level) - 1, 2):
            x_a, w_a = level[i]
            x_b, w_b = level[i + 1]
            gval, u, v = gcdext(x_b, x_a)  # u*x_b + v*x_a = 1
            if gval != 1:
                return None
            combined = (
                signed_powmod(w_a, int(u), n)
                * signed_powmod(w_b, int(v), n)
            ) % n
            next_level.append((x_a * x_b, combined))
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0][1]


# ============================================================
# 编码 / pair / parsing
# ============================================================

def optimized_start_path(code: str):
    result = []
    for i in range(1, len(code) + 1):
        prefix = code[:i]
        if i == 1 or code[i - 1] == "0":
            result.append(prefix)
    return result


def optimized_end_path(code: str):
    result = []
    for i in range(1, len(code) + 1):
        prefix = code[:i]
        if i == 1 or code[i - 1] == "1":
            result.append(prefix)
    return result


def canonical_cover_interval(l: int, r: int, depth: int) -> List[str]:
    if l > r:
        return []
    full_max = (1 << depth) - 1
    l = max(0, l)
    r = min(full_max, r)
    result = []

    def dfs(node_l: int, node_r: int, prefix: str):
        if r < node_l or node_r < l:
            return
        if l <= node_l and node_r <= r:
            result.append(prefix)
            return
        if node_l == node_r:
            result.append(prefix)
            return
        mid = (node_l + node_r) >> 1
        dfs(node_l, mid, prefix + "0")
        dfs(mid + 1, node_r, prefix + "1")

    dfs(0, full_max, "")
    return result


def query_pair_seeds(edge_id: int, traj_id: int, qs: int, qe: int, day_max: int, depth: int) -> List[str]:
    left_cover = canonical_cover_interval(0, qe, depth)
    right_cover = canonical_cover_interval(qs, day_max, depth)

    seen = set()
    out = []
    eid = str(edge_id)
    tid = str(traj_id)
    for lp in left_cover:
        for rp in right_cover:
            seed = f"{eid}|{tid}|{lp}|{rp}"
            if seed not in seen:
                seen.add(seed)
                out.append(seed)
    return out


def parse_tr_time_plaintext(tr_time: str) -> Tuple[int, int, str, str]:
    """
    'edge|traj|{start_code,end_code}'
    """
    p1 = tr_time.find("|")
    p2 = tr_time.find("|", p1 + 1)
    if p1 == -1 or p2 == -1:
        raise ValueError(f"非法 tr_time: {tr_time}")

    edge_id = int(tr_time[:p1])
    traj_id = int(tr_time[p1 + 1:p2])

    body = tr_time[p2 + 1:]
    if not (body.startswith("{") and body.endswith("}")):
        raise ValueError(f"非法 tr_time body: {tr_time}")

    inside = body[1:-1]
    comma = inside.find(",")
    if comma == -1:
        raise ValueError(f"非法 tr_time code body: {tr_time}")

    start_code = inside[:comma]
    end_code = inside[comma + 1:]
    return edge_id, traj_id, start_code, end_code


def parse_private_occurrence_plaintext(
        plaintext: str,
) -> Tuple[int, int, int, str, str, str]:
    """Parse RSG-OCC-V3|day|edge|traj|token|{start,end}."""
    parts = plaintext.split("|", 5)
    if len(parts) != 6 or parts[0] != "RSG-OCC-V3":
        raise ValueError("非法 v3 occurrence plaintext")
    day_idx = int(parts[1])
    edge_id = int(parts[2])
    traj_id = int(parts[3])
    token_hex = parts[4]
    try:
        token = bytes.fromhex(token_hex)
    except ValueError as exc:
        raise ValueError("非法 opaque token") from exc
    if not token or token.hex() != token_hex.lower():
        raise ValueError("opaque token 不是规范十六进制")
    body = parts[5]
    if not (body.startswith("{") and body.endswith("}")):
        raise ValueError("非法 v3 occurrence code")
    values = body[1:-1].split(",", 1)
    if len(values) != 2:
        raise ValueError("非法 v3 occurrence code pair")
    return day_idx, edge_id, traj_id, token_hex.lower(), values[0], values[1]


# ============================================================
# 只检查“一条边”的 claim
# ============================================================

def intervals_intersect(a1: int, a2: int, b1: int, b2: int) -> bool:
    return not (a2 < b1 or b2 < a1)


def inspect_one_edge_claim(db: dict, date_str: str, edge_id: int, qs: int, qe: int, claimed_result_traj_ids: List[int]) -> dict:
    if date_str not in db["data"]:
        raise KeyError(f"date 不存在: {date_str}")

    date_obj = db["data"][date_str]
    edge_key = str(edge_id)
    claimed_set = set(int(x) for x in claimed_result_traj_ids)

    if edge_key not in date_obj:
        return {
            "edge_exists": False,
            "claimed_set": claimed_set,
            "all_present_traj_ids": set(),
            "matched_by_traj": {},
            "nonmatched_by_traj": {},
            "omitted_all": set(),
        }

    edge_obj = date_obj[edge_key]
    all_present: Set[int] = set()
    matched_by_traj: Dict[int, List[dict]] = {}
    nonmatched_by_traj: Dict[int, List[dict]] = {}

    for k, traj_obj in edge_obj.items():
        if k in EDGE_META_KEYS:
            continue

        traj_id = int(k)
        all_present.add(traj_id)

        matched = []
        nonmatched = []

        for detail in traj_obj["detail_tr"]:
            s = detail["start_second"]
            e = detail["end_second"]
            if intervals_intersect(s, e, qs, qe):
                matched.append(detail)
            else:
                nonmatched.append(detail)

        if matched:
            matched_by_traj[traj_id] = matched
        if nonmatched:
            nonmatched_by_traj[traj_id] = nonmatched

    omitted_all = all_present - claimed_set

    return {
        "edge_exists": True,
        "claimed_set": claimed_set,
        "all_present_traj_ids": all_present,
        "matched_by_traj": matched_by_traj,
        "nonmatched_by_traj": nonmatched_by_traj,
        "omitted_all": omitted_all,
    }


def find_matching_trajectory_ids(
        db: dict, date_str: str, edge_id: int, qs: int, qe: int
) -> List[int]:
    """Return the complete server-side result set for one edge and time range."""
    inspect = inspect_one_edge_claim(
        db=db,
        date_str=date_str,
        edge_id=edge_id,
        qs=qs,
        qe=qe,
        claimed_result_traj_ids=[],
    )

    if not inspect["edge_exists"]:
        return []

    return sorted(inspect["matched_by_traj"].keys())


def simulate_claim_r_from_edge(db: dict, date_str: str, edge_id: int, qs: int, qe: int) -> List[int]:
    """Legacy test helper that can intentionally truncate the real result set."""
    matched_traj_ids = find_matching_trajectory_ids(
        db, date_str, edge_id, qs, qe
    )

    if AUTO_R_MODE == "matched_all":
        return matched_traj_ids

    if AUTO_R_MODE == "matched_first_k":
        return matched_traj_ids[:AUTO_R_K]

    raise ValueError(f"未知 AUTO_R_MODE: {AUTO_R_MODE}")


# ============================================================
# x 构造
# ============================================================

def member_plaintext_x(mapper: PrimeMapper, plaintext: str) -> mpz:
    return mapper.hash_to_prime(plaintext)


def trajectory_absence_x(mapper: PrimeMapper, edge_id: int, traj_id: int) -> mpz:
    return mapper.hash_to_prime(edge_tr_seed(edge_id, traj_id))


def trajectory_query_time_x(mapper: PrimeMapper, edge_id: int, traj_id: int, qs: int, qe: int, day_max: int, depth: int):
    seeds = query_pair_seeds(edge_id, traj_id, qs, qe, day_max, depth)
    primes = [mapper.hash_to_prime(s) for s in seeds]
    x = product_tree(primes)
    return x, seeds


def hash_to_prime_uncached(seed: str, bits: int) -> mpz:
    """
    与 PrimeMapper.hash_to_prime 完全同样的映射规则，只是不写入 cache。
    对 query-pair 这种一次性 seed 更快、更省内存。
    """
    seed = str(seed)
    mask = (1 << bits) - 1
    counter = 0
    while True:
        digest = hashlib.sha256(f"{seed}|{counter}".encode("utf-8")).digest()
        x = int.from_bytes(digest, "big") & mask
        x |= (1 << (bits - 1))
        x |= 1
        p = next_prime(mpz(x))
        if p.bit_length() == bits:
            return p
        counter += 1


def build_query_pair_template(qs: int, qe: int, day_max: int, depth: int) -> List[Tuple[str, str]]:
    """
    查询区间固定时，left_cover/right_cover 对所有 edge/traj 都一样。
    原代码每个 traj 都重复 canonical_cover_interval，这里提前生成模板，不计入四个核心时间。
    """
    left_cover = canonical_cover_interval(0, qe, depth)
    right_cover = canonical_cover_interval(qs, day_max, depth)
    return [(lp, rp) for lp in left_cover for rp in right_cover]


def trajectory_query_time_x_from_template(
        mapper: PrimeMapper,
        edge_id: int,
        traj_id: int,
        pair_template: List[Tuple[str, str]],
        hprime_bits: int,
):
    prefix = f"{edge_id}|{traj_id}|"
    primes = []
    qpair_seeds = [] if STORE_QUERY_PAIR_SEEDS_IN_VO else None

    if USE_UNCACHED_QUERY_PAIR_PRIME:
        for lp, rp in pair_template:
            seed = prefix + lp + "|" + rp
            if qpair_seeds is not None:
                qpair_seeds.append(seed)
            primes.append(hash_to_prime_uncached(seed, hprime_bits))
    else:
        for lp, rp in pair_template:
            seed = prefix + lp + "|" + rp
            if qpair_seeds is not None:
                qpair_seeds.append(seed)
            primes.append(mapper.hash_to_prime(seed))

    return product_tree(primes), ([] if qpair_seeds is None else qpair_seeds)


def private_trajectory_query_time_x_from_template(
        edge_id: int,
        token: bytes,
        pair_template: List[Tuple[str, str]],
        hprime_bits: int,
        day_idx: int,
) -> mpz:
    token_hex = bytes(token).hex()
    primes = [
        hash_to_prime_uncached(
            private_time_pair_seed(day_idx, edge_id, token_hex, lp, rp),
            hprime_bits,
        )
        for lp, rp in pair_template
    ]
    return product_tree(primes)


# ============================================================
# proof 生成
# ============================================================

def pick_one_member_plaintext(detail_list: List[dict]) -> str:
    # 现在就取第一条命中段
    return detail_list[0]["tr_time"]



def generate_member_batch_vo(db: dict, mapper: PrimeMapper, date_str: str, edge_id: int, member_items: List[dict]) -> dict:
    """
    最简 member VO：
    - 必须返回：member_items 里的 traj_id + plaintext，w
    - 不返回：date / edge_id / edge_ac / x_batch / matched_occurrence_count
    验证者根据 claim + ADS 自己取 edge_ac，并重新计算 x_batch。
    """
    edge_obj = db["data"][date_str][str(edge_id)]
    edge_xp = edge_obj["edge_xp"]
    n = db["rsa"]["n"]
    g = db["rsa"]["g"]

    minimal_member_items = [
        {
            "traj_id": int(item["traj_id"]),
            "plaintext": item["plaintext"],
        }
        for item in member_items
    ]

    plaintexts = [item["plaintext"] for item in minimal_member_items]
    x_parts = [member_plaintext_x(mapper, pt) for pt in plaintexts]
    x_batch = product_tree(x_parts) if x_parts else mpz(1)

    if x_batch == 1:
        return {
            "status": "empty_member_batch",
            "member_items": [],
        }

    if edge_xp % x_batch != 0:
        raise ValueError("x_member_batch 不能整除 edge_xp")

    static_witnesses = [item.get("member_witness") for item in member_items]
    proof_mode = "online_full_exponent"
    w = None
    if static_witnesses and all(value is not None for value in static_witnesses):
        w = aggregate_precomputed_member_witnesses(
            representatives=x_parts,
            witnesses=[mpz(value) for value in static_witnesses],
            n=n,
        )
        if w is not None:
            proof_mode = "offline_witness_aggregate"

    if w is None:
        w = powmod(g, edge_xp // x_batch, n)

    return {
        "status": "member_batch",
        "member_items": minimal_member_items,
        "w": w,
        "proof_mode": proof_mode,
    }


def generate_absence_vo(db: dict, mapper: PrimeMapper, date_str: str, edge_id: int, traj_id: int) -> dict:
    """
    最简 trajectory_absent VO：只返回 Bézout 证明 (a, d)。
    x = H(edge|traj) 和 edge_ac 都由验证者根据 claim + ADS 重新得到。
    """
    edge_obj = db["data"][date_str][str(edge_id)]
    edge_xp = edge_obj["edge_xp"]
    n = db["rsa"]["n"]
    g = db["rsa"]["g"]

    x = trajectory_absence_x(mapper, edge_id, traj_id)
    wit = bezout_nonmembership_witness(edge_xp, x, g, n)

    return {
        "a": wit["a"],
        "d": wit["d"],
    }


def generate_time_mismatch_vo(db: dict, mapper: PrimeMapper, date_str: str, edge_id: int, traj_id: int, qs: int, qe: int) -> dict:
    """
    最简 time_mismatch VO：只返回 Bézout 证明 (a, d)。
    query-pair seeds / x / edge_ac / qs / qe 都不放进 VO。
    """
    edge_obj = db["data"][date_str][str(edge_id)]
    edge_xp = edge_obj["edge_xp"]
    n = db["rsa"]["n"]
    g = db["rsa"]["g"]
    depth = db["rsa"]["depth"]
    day_max = db["rsa"]["day_seconds"] - 1

    x, _qpair_seeds = trajectory_query_time_x(mapper, edge_id, traj_id, qs, qe, day_max, depth)
    wit = bezout_nonmembership_witness(edge_xp, x, g, n)

    return {
        "a": wit["a"],
        "d": wit["d"],
    }

def generate_completeness_batch(db: dict, mapper: PrimeMapper, date_str: str, edge_id: int, qs: int, qe: int, omitted_all: Set[int]) -> dict:
    edge_obj = db["data"][date_str][str(edge_id)]
    edge_xp = edge_obj["edge_xp"]
    edge_ac = edge_obj["edge_ac"]
    n = db["rsa"]["n"]
    g = db["rsa"]["g"]
    depth = db["rsa"]["depth"]
    day_max = db["rsa"]["day_seconds"] - 1

    x_parts = []
    omitted_detail = []

    for traj_id in sorted(omitted_all):
        x_traj, qpair_seeds = trajectory_query_time_x(mapper, edge_id, traj_id, qs, qe, day_max, depth)
        x_parts.append(x_traj)
        omitted_detail.append({
            "traj_id": traj_id,
            "query_pair_seeds": qpair_seeds,
            "x_traj": x_traj,
        })

    x_batch = product_tree(x_parts) if x_parts else mpz(1)
    # gcdext already returns gcd(edge_xp, x_batch).  Running gcd first and
    # gcdext immediately afterwards repeats the expensive Euclidean pass over
    # the same very large integers.  Keep the exact same Bezout proof while
    # doing that work only once.
    gval, bezout_a, bezout_b = gcdext(edge_xp, x_batch)
    all_present = {
        int(key)
        for key in edge_obj
        if key not in EDGE_META_KEYS
    }
    binding_claim = {
        "date": date_str,
        "edge_id": int(edge_id),
        "qs": int(qs),
        "qe": int(qe),
        "claimed_result_traj_ids": sorted(all_present - set(int(x) for x in omitted_all)),
        "claimed_result_traj_count": len(all_present - set(int(x) for x in omitted_all)),
    }
    binding_digest = completeness_binding_digest(
        db, binding_claim, sorted(int(x) for x in omitted_all), x_batch
    )

    if gval == 1:
        d = signed_powmod(g, int(bezout_b), n)
        return {
            "status": "batch_nonmembership",
            "date": date_str,
            "edge_id": edge_id,
            "qs": qs,
            "qe": qe,
            "omitted_traj_ids": sorted(omitted_all),
            "omitted_detail": omitted_detail,
            "x_batch": x_batch,
            "binding_digest": binding_digest,
            "gcd": mpz(1),
            "a": int(bezout_a),
            "d": d,
            "edge_ac": edge_ac,
        }

    return {
        "status": "detected_incomplete_claim",
        "date": date_str,
        "edge_id": edge_id,
        "qs": qs,
        "qe": qe,
        "omitted_traj_ids": sorted(omitted_all),
        "omitted_detail": omitted_detail,
        "x_batch": x_batch,
        "binding_digest": binding_digest,
        "gcd": gval,
        "edge_ac": edge_ac,
    }



def compute_omitted_ids_from_claim_and_db(db: dict, claim: dict) -> List[int]:
    """
    最简 VO 口径：omitted_traj_ids 不放进 VO。
    验证端根据 ADS 中该边出现过的所有轨迹 all_present，以及 claim 里的 R，自己恢复 omitted_all。
    """
    date_str = claim["date"]
    edge_id = int(claim["edge_id"])
    if date_str not in db["data"] or str(edge_id) not in db["data"][date_str]:
        return []

    edge_obj = db["data"][date_str][str(edge_id)]
    claimed_set = set(int(x) for x in claim.get("claimed_result_traj_ids", []))
    all_present = set()
    for k in edge_obj.keys():
        if k in EDGE_META_KEYS:
            continue
        all_present.add(int(k))
    return sorted(all_present - claimed_set)


def validate_claim_result_ids(claim: dict) -> Optional[List[int]]:
    """Return canonical result ids, or None for duplicates/count mismatch."""
    raw_ids = [int(x) for x in claim.get("claimed_result_traj_ids", [])]
    canonical = sorted(set(raw_ids))
    declared_count = int(claim.get("claimed_result_traj_count", len(raw_ids)))
    if len(canonical) != len(raw_ids) or declared_count != len(raw_ids):
        return None
    return canonical


def completeness_binding_digest(
        db: dict,
        claim: dict,
        omitted_ids: List[int],
        x_batch: mpz,
) -> str:
    """Bind X_batch to the trusted ADS root, query, edge, R and U-R."""
    result_ids = validate_claim_result_ids(claim)
    if result_ids is None:
        raise ValueError("claim 结果集合包含重复 id 或数量不一致")
    trusted_root = db.get("trusted_root") or {}
    payload = {
        "domain": "RSG-COMPLETENESS-BINDING-V1",
        "trusted_data_ac_hex": mpz(trusted_root.get("data_ac", 0)).digits(16),
        "date": str(claim["date"]),
        "edge_id": int(claim["edge_id"]),
        "qs": int(claim["qs"]),
        "qe": int(claim["qe"]),
        "result_ids": result_ids,
        "omitted_ids": [int(x) for x in omitted_ids],
        "x_batch_hex": mpz(x_batch).digits(16),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recompute_expected_completeness_x(
        db: dict,
        claim: dict,
) -> Tuple[mpz, List[int], List[Tuple[str, str]]]:
    """Rebuild the semantic X* from trusted U, client query Q and claimed R."""
    result_ids = validate_claim_result_ids(claim)
    if result_ids is None:
        raise ValueError("claim 结果集合包含重复 id 或数量不一致")

    date_str = str(claim["date"])
    edge_id = int(claim["edge_id"])
    qs = int(claim["qs"])
    qe = int(claim["qe"])
    if qs < 0 or qe < qs or qe >= int(db["rsa"]["day_seconds"]):
        raise ValueError("查询时间范围非法")

    omitted_ids = compute_omitted_ids_from_claim_and_db(db, claim)
    pair_template = build_query_pair_template(
        qs,
        qe,
        int(db["rsa"]["day_seconds"]) - 1,
        int(db["rsa"]["depth"]),
    )
    mapper = PrimeMapper(bits=int(db["rsa"]["hprime_bits"]))
    x_parts = []
    for traj_id in omitted_ids:
        x_traj, _seeds = trajectory_query_time_x_from_template(
            mapper=mapper,
            edge_id=edge_id,
            traj_id=traj_id,
            pair_template=pair_template,
            hprime_bits=int(db["rsa"]["hprime_bits"]),
        )
        x_parts.append(x_traj)
    return (product_tree(x_parts) if x_parts else mpz(1)), omitted_ids, pair_template


def generate_completeness_batch_fast(
        db: dict,
        mapper: PrimeMapper,
        date_str: str,
        edge_id: int,
        qs: int,
        qe: int,
        omitted_all: Set[int],
        pair_template: Optional[List[Tuple[str, str]]] = None,
) -> dict:
    """
    快速验证版 completeness VO：
    - batch_nonmembership: 返回 status + X_batch + Bézout witness (a, d)
    - detected_incomplete_claim: 返回 status + X_batch
    - 不返回 omitted_traj_ids / x_traj / edge_ac / edge_xp / gcd / query_pair_seeds / date / edge_id / qs / qe

    生成端仍完整计算 X_batch 和 Bézout 证明；
    验证端直接用 VO 里的 X_batch 验证，不再重新计算所有 omitted 轨迹的 prime。
    """
    edge_obj = db["data"][date_str][str(edge_id)]
    edge_xp = edge_obj["edge_xp"]
    n = db["rsa"]["n"]
    g = db["rsa"]["g"]
    depth = db["rsa"]["depth"]
    day_max = db["rsa"]["day_seconds"] - 1
    hprime_bits = db["rsa"]["hprime_bits"]

    if pair_template is None:
        pair_template = build_query_pair_template(qs, qe, day_max, depth)

    omitted_ids = sorted(int(x) for x in omitted_all)
    x_parts = []

    for traj_id in omitted_ids:
        x_traj, _qpair_seeds = trajectory_query_time_x_from_template(
            mapper=mapper,
            edge_id=edge_id,
            traj_id=traj_id,
            pair_template=pair_template,
            hprime_bits=hprime_bits,
        )
        x_parts.append(x_traj)

    x_batch = product_tree(x_parts) if x_parts else mpz(1)
    all_present = {
        int(key)
        for key in edge_obj
        if key not in EDGE_META_KEYS
    }
    omitted_set = set(omitted_ids)
    binding_claim = {
        "date": date_str,
        "edge_id": int(edge_id),
        "qs": int(qs),
        "qe": int(qe),
        "claimed_result_traj_ids": sorted(all_present - omitted_set),
        "claimed_result_traj_count": len(all_present - omitted_set),
    }
    binding_digest = completeness_binding_digest(
        db, binding_claim, omitted_ids, x_batch
    )

    # A single gcdext gives both the gcd test and the Bezout coefficients.
    # The former implementation performed gcd + gcdext on the same operands.
    gval, a, b = gcdext(edge_xp, x_batch)
    if gval == 1:
        d = signed_powmod(g, int(b), n)
        return {
            "status": "batch_nonmembership",
            "x_batch": x_batch,
            "binding_digest": binding_digest,
            "a": int(a),
            "d": d,
        }

    return {
        "status": "detected_incomplete_claim",
        "x_batch": x_batch,
        "binding_digest": binding_digest,
    }


def generate_private_completeness_batch(
        db: dict,
        date_str: str,
        edge_id: int,
        omitted_all: Set[int],
        pair_template: List[Tuple[str, str]],
) -> dict:
    """Server-side v3 proof generation over opaque non-result tokens."""
    edge_obj = db["data"][date_str][str(edge_id)]
    edge_xp = edge_obj["edge_xp"]
    day_idx = date_to_day_idx(date_str)
    hprime_bits = int(db["rsa"]["hprime_bits"])
    omitted_tokens = [
        bytes(edge_obj[str(traj_id)]["opaque_token"])
        for traj_id in sorted(int(value) for value in omitted_all)
    ]
    x_parts = [
        private_trajectory_query_time_x_from_template(
            edge_id=edge_id,
            token=token,
            pair_template=pair_template,
            hprime_bits=hprime_bits,
            day_idx=day_idx,
        )
        for token in omitted_tokens
    ]
    x_batch = product_tree(x_parts) if x_parts else mpz(1)
    gval, a, b = gcdext(edge_xp, x_batch)
    common = {
        "total_traj_count": int(edge_obj["traj_count"]),
        "omitted_tokens": omitted_tokens,
        "edge_ac": edge_obj["edge_ac"],
        "root_witness": edge_obj["root_witness"],
        "identity_witness": edge_obj["identity_witness"],
    }
    if gval != 1:
        return {"status": "detected_incomplete_claim", **common}
    return {
        "status": "batch_nonmembership",
        **common,
        "a": int(a),
        "d": signed_powmod(db["rsa"]["g"], int(b), db["rsa"]["n"]),
    }


def query_edge_by_range(
        db: dict,
        date_str: str,
        edge_id: int,
        qs: int,
        qe: int,
) -> dict:
    """Server query API: derive results from the range and return results + binary VO.

    Unlike ``generate_claim_package``, this public query path does not accept a
    caller-supplied result set.  Therefore the normal response has no
    ``invalid_claim_items`` branch: every returned item was selected directly
    from the indexed edge by the server-side range predicate.
    """
    edge_id = int(edge_id)
    qs = int(qs)
    qe = int(qe)
    if qs < 0 or qe < qs or qe >= int(db["rsa"]["day_seconds"]):
        raise ValueError("非法日内查询时间范围")
    if date_str not in db["data"]:
        raise KeyError(f"date 不存在: {date_str}")
    if str(edge_id) not in db["data"][date_str]:
        raise KeyError(f"edge 不存在: {edge_id}")

    result_ids = find_matching_trajectory_ids(db, date_str, edge_id, qs, qe)
    inspect = inspect_one_edge_claim(
        db=db,
        date_str=date_str,
        edge_id=edge_id,
        qs=qs,
        qe=qe,
        claimed_result_traj_ids=result_ids,
    )
    mapper = PrimeMapper(bits=db["rsa"]["hprime_bits"])
    member_items = []
    for traj_id in result_ids:
        matched_details = inspect["matched_by_traj"].get(int(traj_id), [])
        if not matched_details:
            raise AssertionError("范围查询产生了不满足时间条件的结果")
        member_items.append({
            "traj_id": int(traj_id),
            "plaintext": matched_details[0]["tr_time"],
            "member_witness": matched_details[0].get("member_witness"),
        })

    member_batch = generate_member_batch_vo(
        db, mapper, date_str, edge_id, member_items
    )
    pair_template = build_query_pair_template(
        qs,
        qe,
        int(db["rsa"]["day_seconds"]) - 1,
        int(db["rsa"]["depth"]),
    )
    if int(db["rsa"].get("format_version", 0)) >= 3:
        completeness = generate_private_completeness_batch(
            db=db,
            date_str=date_str,
            edge_id=edge_id,
            omitted_all=inspect["omitted_all"],
            pair_template=pair_template,
        )
        completeness_vo = encode_private_completeness(completeness)
    else:
        completeness = generate_completeness_batch_fast(
            db=db,
            mapper=mapper,
            date_str=date_str,
            edge_id=edge_id,
            qs=qs,
            qe=qe,
            omitted_all=inspect["omitted_all"],
            pair_template=pair_template,
        )
        completeness_vo = encode_completeness(completeness)

    return {
        "query": {
            "date": str(date_str),
            "edge_id": edge_id,
            "qs": qs,
            "qe": qe,
        },
        "results": [
            {"traj_id": int(item["traj_id"]), "plaintext": str(item["plaintext"])}
            for item in member_items
        ],
        "vo": {
            "member_batch": encode_member_batch(member_batch),
            "completeness": completeness_vo,
        },
    }


def verify_edge_range_response(
        client_db: dict,
        response: dict,
        expected_query: Optional[dict] = None,
) -> dict:
    """Client verification for a range-only edge query response."""
    try:
        query = response["query"]
        canonical_query = {
            "date": str(query["date"]),
            "edge_id": int(query["edge_id"]),
            "qs": int(query["qs"]),
            "qe": int(query["qe"]),
        }
        if expected_query is not None:
            expected = {
                "date": str(expected_query["date"]),
                "edge_id": int(expected_query["edge_id"]),
                "qs": int(expected_query["qs"]),
                "qe": int(expected_query["qe"]),
            }
            if canonical_query != expected:
                return {"all_ok": False, "query_binding_ok": False}

        results = [
            {"traj_id": int(item["traj_id"]), "plaintext": str(item["plaintext"])}
            for item in response.get("results", [])
        ]
        result_ids = [item["traj_id"] for item in results]
        result_binding_ok = (
            len(set(result_ids)) == len(result_ids)
            and result_ids == sorted(result_ids)
        )
        claim = {
            **canonical_query,
            "claimed_result_traj_ids": result_ids,
            "claimed_result_traj_count": len(result_ids),
        }
        member_batch = decode_member_batch(response["vo"]["member_batch"])
        decoded_results = [
            {"traj_id": int(item["traj_id"]), "plaintext": str(item["plaintext"])}
            for item in member_batch.get("member_items", [])
        ]
        result_binding_ok = result_binding_ok and decoded_results == results
        if int(client_db["rsa"].get("format_version", 0)) >= 3:
            completeness = decode_private_completeness(
                response["vo"]["completeness"]
            )
            mapper = PrimeMapper(bits=client_db["rsa"]["hprime_bits"])
            edge_auth_ok = verify_private_edge_auth(
                client_db, mapper, claim, completeness
            )
            if edge_auth_ok and result_binding_ok:
                member_ok, result_tokens = verify_private_member_batch_vo(
                    client_db,
                    mapper,
                    claim,
                    member_batch,
                    completeness["edge_ac"],
                )
            else:
                member_ok, result_tokens = False, []
            pair_template = build_query_pair_template(
                canonical_query["qs"],
                canonical_query["qe"],
                int(client_db["rsa"]["day_seconds"]) - 1,
                int(client_db["rsa"]["depth"]),
            )
            completeness_ok = (
                result_binding_ok
                and verify_private_completeness_batch(
                    client_db,
                    mapper,
                    claim,
                    completeness,
                    result_tokens,
                    pair_template,
                )
            )
        else:
            mapper = PrimeMapper(bits=client_db["rsa"]["hprime_bits"])
            completeness = decode_completeness(response["vo"]["completeness"])
            member_ok = result_binding_ok and verify_member_batch_vo(
                client_db, mapper, claim, member_batch
            )
            completeness_ok = result_binding_ok and verify_completeness_batch(
                client_db, claim, completeness
            )
            edge_auth_ok = True
    except (KeyError, TypeError, ValueError, VOFormatError):
        return {
            "all_ok": False,
            "query_binding_ok": False,
            "result_binding_ok": False,
            "member_ok": False,
            "completeness_ok": False,
        }

    return {
        "all_ok": bool(
            result_binding_ok and edge_auth_ok and member_ok and completeness_ok
        ),
        "query_binding_ok": True,
        "result_binding_ok": bool(result_binding_ok),
        "edge_auth_ok": bool(edge_auth_ok),
        "member_ok": bool(member_ok),
        "completeness_ok": bool(completeness_ok),
        "result_count": len(results),
    }


def query_edges_by_range(
        db: dict,
        date_str: str,
        edge_ids: List[int],
        qs: int,
        qe: int,
) -> dict:
    """Server query API for all edges covered by one spatial range."""
    canonical_edges = sorted(set(int(value) for value in edge_ids))
    edge_responses = [
        query_edge_by_range(db, date_str, edge_id, qs, qe)
        for edge_id in canonical_edges
    ]
    result_ids = sorted({
        int(item["traj_id"])
        for edge_response in edge_responses
        for item in edge_response["results"]
    })
    return {
        "query": {
            "date": str(date_str),
            "edge_ids": canonical_edges,
            "qs": int(qs),
            "qe": int(qe),
        },
        "result_trajectory_ids": result_ids,
        "edges": edge_responses,
    }


def verify_edges_range_response(
        client_db: dict,
        response: dict,
        expected_query: dict,
) -> dict:
    """Verify exact edge coverage, returned results and every edge VO."""
    try:
        expected_edges = sorted(set(int(value) for value in expected_query["edge_ids"]))
        actual_query = response["query"]
        query_binding_ok = (
            str(actual_query["date"]) == str(expected_query["date"])
            and int(actual_query["qs"]) == int(expected_query["qs"])
            and int(actual_query["qe"]) == int(expected_query["qe"])
            and [int(value) for value in actual_query["edge_ids"]] == expected_edges
        )
        edge_responses = list(response.get("edges", []))
        returned_edges = [int(item["query"]["edge_id"]) for item in edge_responses]
        edge_coverage_ok = (
            len(returned_edges) == len(expected_edges)
            and len(set(returned_edges)) == len(returned_edges)
            and sorted(returned_edges) == expected_edges
        )
        verdicts = []
        if query_binding_ok and edge_coverage_ok:
            for edge_response in edge_responses:
                verdicts.append(verify_edge_range_response(
                    client_db,
                    edge_response,
                    {
                        "date": expected_query["date"],
                        "edge_id": edge_response["query"]["edge_id"],
                        "qs": expected_query["qs"],
                        "qe": expected_query["qe"],
                    },
                ))
        union_ids = sorted({
            int(item["traj_id"])
            for edge_response in edge_responses
            for item in edge_response.get("results", [])
        })
        result_union_ok = (
            [int(value) for value in response.get("result_trajectory_ids", [])]
            == union_ids
        )
    except (KeyError, TypeError, ValueError):
        return {
            "all_ok": False,
            "query_binding_ok": False,
            "edge_coverage_ok": False,
            "result_union_ok": False,
            "edge_verdicts": [],
        }
    all_ok = (
        query_binding_ok
        and edge_coverage_ok
        and result_union_ok
        and len(verdicts) == len(expected_edges)
        and all(item.get("all_ok", False) for item in verdicts)
    )
    return {
        "all_ok": bool(all_ok),
        "query_binding_ok": bool(query_binding_ok),
        "edge_coverage_ok": bool(edge_coverage_ok),
        "result_union_ok": bool(result_union_ok),
        "edge_verdicts": verdicts,
    }


def generate_claim_package(db: dict, date_str: str, edge_id: int, qs: int, qe: int, claimed_result_traj_ids: List[int]) -> dict:
    mapper = PrimeMapper(bits=db["rsa"]["hprime_bits"])

    inspect = inspect_one_edge_claim(
        db=db,
        date_str=date_str,
        edge_id=edge_id,
        qs=qs,
        qe=qe,
        claimed_result_traj_ids=claimed_result_traj_ids,
    )

    t0 = time.perf_counter()
    member_items = []
    invalid_claim_items = []

    for traj_id in claimed_result_traj_ids:
        traj_id = int(traj_id)

        if not inspect["edge_exists"] or traj_id not in inspect["all_present_traj_ids"]:
            invalid_claim_items.append({
                "traj_id": traj_id,
                "status": "trajectory_absent",
                "proof": generate_absence_vo(db, mapper, date_str, edge_id, traj_id) if inspect["edge_exists"] else None,
            })
            continue

        if traj_id in inspect["matched_by_traj"]:
            matched_details = inspect["matched_by_traj"][traj_id]
            selected_detail = matched_details[0]
            plaintext = selected_detail["tr_time"]
            member_items.append({
                "traj_id": traj_id,
                "plaintext": plaintext,
                "matched_occurrence_count": len(matched_details),
                "member_witness": selected_detail.get("member_witness"),
            })
        else:
            invalid_claim_items.append({
                "traj_id": traj_id,
                "status": "time_mismatch",
                "proof": generate_time_mismatch_vo(db, mapper, date_str, edge_id, traj_id, qs, qe),
            })
    member_batch = generate_member_batch_vo(db, mapper, date_str, edge_id, member_items)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    completeness = None
    if inspect["edge_exists"]:
        completeness = generate_completeness_batch(
            db=db,
            mapper=mapper,
            date_str=date_str,
            edge_id=edge_id,
            qs=qs,
            qe=qe,
            omitted_all=inspect["omitted_all"],
        )
    t3 = time.perf_counter()

    return {
        "claim": {
            "date": date_str,
            "edge_id": edge_id,
            "qs": qs,
            "qe": qe,
            "claimed_result_traj_ids": [int(x) for x in claimed_result_traj_ids],
            "claimed_result_traj_count": len([int(x) for x in claimed_result_traj_ids]),
        },
        "inspect": {
            "edge_exists": inspect["edge_exists"],
            "all_present_traj_ids": sorted(inspect["all_present_traj_ids"]),
            "omitted_all": sorted(inspect["omitted_all"]),
            "batch_traj_count": len(inspect["omitted_all"]),
        },
        "vo": {
            "member_batch": member_batch,
            "invalid_claim_items": invalid_claim_items,
            "completeness": completeness,
        },
        "timing": {
            "generation": {
                "member_batch_gen_time": t1 - t0,
                "completeness_gen_time": t3 - t2,
            }
        }
    }


# ============================================================
# proof 验证
# ============================================================


def verify_private_edge_auth(
        db: dict,
        mapper: PrimeMapper,
        claim: dict,
        completeness: dict,
) -> bool:
    """Authenticate (date, edge, count, edge_ac) against the owner-pinned root."""
    trusted_root = db.get("trusted_root") or {}
    if not trusted_root or "data_ac" not in trusted_root:
        return False
    count = int(completeness.get("total_traj_count", -1))
    edge_ac = mpz(completeness.get("edge_ac", 0))
    root_witness = mpz(completeness.get("root_witness", 0))
    if count < 0 or edge_ac <= 0 or root_witness <= 0:
        return False
    seed = private_edge_descriptor_seed(
        date_to_day_idx(str(claim["date"])),
        int(claim["edge_id"]),
        count,
        edge_ac,
    )
    representative = mapper.hash_to_prime(seed)
    return (
        powmod(root_witness, representative, db["rsa"]["n"])
        == mpz(trusted_root["data_ac"])
    )


def verify_private_member_batch_vo(
        db: dict,
        mapper: PrimeMapper,
        claim: dict,
        member_batch: dict,
        edge_ac: mpz,
) -> Tuple[bool, List[bytes]]:
    """Client v3 result correctness; returns authenticated result tokens."""
    claimed_ids = validate_claim_result_ids(claim)
    if claimed_ids is None:
        return False, []
    member_items = member_batch.get("member_items", [])
    if not member_items:
        return (
            not claimed_ids
            and member_batch.get("status") == "empty_member_batch",
            [],
        )
    if member_batch.get("status") != "member_batch":
        return False, []
    item_ids = [int(item["traj_id"]) for item in member_items]
    if len(set(item_ids)) != len(item_ids) or sorted(item_ids) != claimed_ids:
        return False, []

    expected_day_idx = date_to_day_idx(str(claim["date"]))
    token_bytes = int(db["rsa"].get("token_bytes", 0))
    representatives = []
    result_tokens = []
    try:
        for item in member_items:
            plaintext = str(item["plaintext"])
            day_idx, edge_id, traj_id, token_hex, start_code, end_code = (
                parse_private_occurrence_plaintext(plaintext)
            )
            token = bytes.fromhex(token_hex)
            if day_idx != expected_day_idx or edge_id != int(claim["edge_id"]):
                return False, []
            if traj_id != int(item["traj_id"]) or len(token) != token_bytes:
                return False, []
            start_second = int(start_code, int(db["rsa"]["base"]))
            end_second = int(end_code, int(db["rsa"]["base"]))
            if not intervals_intersect(
                    start_second,
                    end_second,
                    int(claim["qs"]),
                    int(claim["qe"]),
            ):
                return False, []
            representatives.append(member_plaintext_x(mapper, plaintext))
            result_tokens.append(token)
    except (TypeError, ValueError):
        return False, []

    x_batch = product_tree(representatives) if representatives else mpz(1)
    ok = powmod(member_batch["w"], x_batch, db["rsa"]["n"]) == mpz(edge_ac)
    return bool(ok), (result_tokens if ok else [])


def verify_private_completeness_batch(
        db: dict,
        mapper: PrimeMapper,
        claim: dict,
        completeness: Optional[dict],
        result_tokens: List[bytes],
        pair_template: List[Tuple[str, str]],
) -> bool:
    """Client v3 coverage + opaque-token time non-membership verification."""
    if completeness is None or completeness.get("status") != "batch_nonmembership":
        return False
    if not verify_private_edge_auth(db, mapper, claim, completeness):
        return False

    token_bytes = int(db["rsa"].get("token_bytes", 0))
    omitted_tokens = [bytes(value) for value in completeness.get("omitted_tokens", [])]
    if any(len(value) != token_bytes for value in result_tokens + omitted_tokens):
        return False
    if len(set(result_tokens)) != len(result_tokens):
        return False
    if len(set(omitted_tokens)) != len(omitted_tokens):
        return False
    if set(result_tokens).intersection(omitted_tokens):
        return False
    total_count = int(completeness.get("total_traj_count", -1))
    if len(result_tokens) + len(omitted_tokens) != total_count:
        return False

    day_idx = date_to_day_idx(str(claim["date"]))
    edge_id = int(claim["edge_id"])
    identity_parts = [
        mapper.hash_to_prime(private_identity_seed(day_idx, edge_id, token.hex()))
        for token in result_tokens + omitted_tokens
    ]
    identity_x = product_tree(identity_parts) if identity_parts else mpz(1)
    edge_ac = mpz(completeness["edge_ac"])
    if powmod(
            completeness["identity_witness"],
            identity_x,
            db["rsa"]["n"],
    ) != edge_ac:
        return False

    time_parts = [
        private_trajectory_query_time_x_from_template(
            edge_id=edge_id,
            token=token,
            pair_template=pair_template,
            hprime_bits=int(db["rsa"]["hprime_bits"]),
            day_idx=day_idx,
        )
        for token in omitted_tokens
    ]
    x_batch = product_tree(time_parts) if time_parts else mpz(1)
    left = signed_powmod(edge_ac, int(completeness["a"]), db["rsa"]["n"])
    right = powmod(completeness["d"], x_batch, db["rsa"]["n"])
    return (left * right) % db["rsa"]["n"] == db["rsa"]["g"]


def verify_member_batch_vo(db: dict, mapper: PrimeMapper, claim: dict, member_batch: dict) -> bool:
    member_items = member_batch.get("member_items", [])
    claimed_ids = validate_claim_result_ids(claim)
    if claimed_ids is None:
        return False
    if not member_items:
        return (
            not claimed_ids
            and member_batch.get("status") == "empty_member_batch"
        )

    if member_batch.get("status") != "member_batch":
        raise ValueError(f"未知 member_batch status: {member_batch.get('status')}")

    item_ids = [int(item["traj_id"]) for item in member_items]
    if len(set(item_ids)) != len(item_ids) or sorted(item_ids) != claimed_ids:
        return False

    n = db["rsa"]["n"]
    edge_obj = db["data"][claim["date"]][str(claim["edge_id"])]
    edge_ac = edge_obj["edge_ac"]
    x_parts = []

    for item in member_items:
        plaintext = item["plaintext"]
        tr_edge_id, tr_traj_id, start_code, end_code = parse_tr_time_plaintext(plaintext)

        if tr_edge_id != claim["edge_id"]:
            return False
        if tr_traj_id != item["traj_id"]:
            return False

        start_second = int(start_code, db["rsa"]["base"])
        end_second = int(end_code, db["rsa"]["base"])

        if not intervals_intersect(start_second, end_second, claim["qs"], claim["qe"]):
            return False

        x_parts.append(member_plaintext_x(mapper, plaintext))

    x_batch = product_tree(x_parts) if x_parts else mpz(1)
    return powmod(member_batch["w"], x_batch, n) == edge_ac


def verify_nonmembership_vo(db: dict, claim: dict, proof: dict, x: mpz) -> bool:
    """
    非成员验证的正确口径：
    - x 已经由当前待验证对象给定/计算得到；
      例如 trajectory_absent 的 x=H(edge|traj)，
      time_mismatch 的 x=对应 query-pair product，
      completeness batch 的 x=VO 中给出的 X_batch。
    - 验证端不重新生成 omitted 集合。
    - 若本地 ADS 里有 edge_xp，可以先检查 gcd(edge_xp, x)==1。
    - 然后验证 Bézout witness: edge_ac^a * d^x mod n == g。
    """
    n = db["rsa"]["n"]
    g = db["rsa"]["g"]
    edge_obj = db["data"][claim["date"]][str(claim["edge_id"])]
    edge_xp = edge_obj["edge_xp"]
    edge_ac = edge_obj["edge_ac"]

    x = mpz(x)
    if gcd(edge_xp, x) != 1:
        return False

    return verify_nonmembership_equation(db, claim, proof, x)


def verify_nonmembership_equation(db: dict, claim: dict, proof: dict, x: mpz) -> bool:
    """Verify the RSA non-membership equation after coprimality was checked."""
    n = db["rsa"]["n"]
    g = db["rsa"]["g"]
    edge_obj = db["data"][claim["date"]][str(claim["edge_id"])]
    edge_ac = edge_obj["edge_ac"]

    left = signed_powmod(edge_ac, int(proof["a"]), n)
    right = powmod(proof["d"], x, n)
    return (left * right) % n == g


def verify_invalid_claim_item(db: dict, mapper: PrimeMapper, claim: dict, item: dict) -> bool:
    status = item["status"]
    traj_id = int(item["traj_id"])
    proof = item.get("proof")

    if status == "trajectory_absent":
        if proof is None:
            return True
        x = trajectory_absence_x(mapper, claim["edge_id"], traj_id)
        return verify_nonmembership_vo(db, claim, proof, x)

    if status == "time_mismatch":
        x, _ = trajectory_query_time_x(
            mapper=mapper,
            edge_id=claim["edge_id"],
            traj_id=traj_id,
            qs=claim["qs"],
            qe=claim["qe"],
            day_max=db["rsa"]["day_seconds"] - 1,
            depth=db["rsa"]["depth"],
        )
        return verify_nonmembership_vo(db, claim, proof, x)

    raise ValueError(f"未知 invalid claim status: {status}")


def verify_completeness_batch(
        db: dict,
        claim: dict,
        completeness: Optional[dict],
        pair_template: Optional[List[Tuple[str, str]]] = None,
) -> bool:
    """
    安全验证版：
        1) 从受信任 ADS 的 U、客户端查询 Q 和声明结果 R 重建 O=U-R；
        2) 独立重建 X*=F(edge,Q,O)，拒绝服务端替换的 X_batch；
        3) 检查绑定摘要；
        4) 最后验证 RSA Bézout 非成员方程。

    STRICT_COMPLETENESS_BINDING=False 仅用于旧性能对照，不适用于不可信服务器。
    """
    if completeness is None:
        return False

    edge_id = int(claim["edge_id"])

    if claim["date"] not in db["data"] or str(edge_id) not in db["data"][claim["date"]]:
        return completeness.get("status") == "detected_incomplete_claim"

    if "x_batch" not in completeness:
        return False

    x_batch = mpz(completeness["x_batch"])
    edge_obj = db["data"][claim["date"]][str(edge_id)]
    edge_xp = edge_obj["edge_xp"]

    if STRICT_COMPLETENESS_BINDING:
        if not db.get("trusted_root"):
            return False
        try:
            expected_x, expected_omitted_ids, _expected_pairs = (
                recompute_expected_completeness_x(db, claim)
            )
            expected_digest = completeness_binding_digest(
                db, claim, expected_omitted_ids, expected_x
            )
        except (KeyError, TypeError, ValueError):
            return False
        if x_batch != expected_x:
            return False
        if completeness.get("binding_digest") != expected_digest:
            return False

    gval = gcd(edge_xp, x_batch)

    if completeness["status"] == "batch_nonmembership":
        if gval != 1:
            return False
        # gval was just checked above; do not repeat the same giant-integer gcd.
        return verify_nonmembership_equation(db, claim, completeness, x_batch)

    if completeness["status"] == "detected_incomplete_claim":
        # 证明可以正确检测出缺失结果，但该查询声明本身必须判失败。
        return False

    raise ValueError(f"未知 completeness status: {completeness['status']}")


def verify_claim_package(db: dict, package: dict) -> dict:
    mapper = PrimeMapper(bits=db["rsa"]["hprime_bits"])

    t0 = time.perf_counter()
    member_batch_ok = verify_member_batch_vo(db, mapper, package["claim"], package["vo"]["member_batch"])
    invalid_items_proofs_ok = True
    for item in package["vo"]["invalid_claim_items"]:
        invalid_items_proofs_ok = invalid_items_proofs_ok and verify_invalid_claim_item(db, mapper, package["claim"], item)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    completeness_ok = verify_completeness_batch(db, package["claim"], package["vo"]["completeness"])
    t3 = time.perf_counter()

    # 真正的 soundness：只要 claim 里出现非 member 项，就应失败
    invalid_claim_count = len(package["vo"]["invalid_claim_items"])
    soundness_ok = member_batch_ok and invalid_items_proofs_ok and (invalid_claim_count == 0)

    return {
        "member_batch_ok": member_batch_ok,
        "invalid_items_proofs_ok": invalid_items_proofs_ok,
        "invalid_claim_count": invalid_claim_count,
        "soundness_ok": soundness_ok,
        "completeness_ok": completeness_ok,
        "completeness_status": None if package["vo"]["completeness"] is None else package["vo"]["completeness"]["status"],
        "all_ok": soundness_ok and completeness_ok,
        "timing": {
            "verification": {
                "member_batch_and_invalid_items_verify_time": t1 - t0,
                "completeness_verify_time": t3 - t2,
            }
        }
    }


# ============================================================
# 多边并行调度：固定 worker + 负载均衡 + 每个 worker 只加载一次 ADS
# ============================================================

def choose_query_edges(edge_infos: List[dict]) -> List[dict]:
    by_id = {int(x["edge_id"]): x for x in edge_infos}

    if QUERY_EDGE_IDS:
        selected = []
        for eid in QUERY_EDGE_IDS:
            eid = int(eid)
            selected.append(by_id.get(eid, {
                "edge_id": eid,
                "traj_count": 0,
                "occ_total": 0,
                "weight": 1,
                "missing_in_scan": True,
            }))
        return selected

    # 自动挑轨迹数量最多的边。包含轨迹数量为 0 的边没有验证压力，默认不选。
    candidates = [x for x in edge_infos if x["traj_count"] > 0]
    candidates.sort(key=lambda x: (x["traj_count"], x["occ_total"], x["edge_id"]), reverse=True)
    return candidates[:AUTO_PICK_EDGE_COUNT]


def balance_edges_by_weight(selected_edges: List[dict], num_workers: int) -> List[List[dict]]:
    """
    LPT 贪心负载均衡：重边优先放到当前负载最小的 worker。
    效果：轨迹数量多的边所在 worker 分到的边更少；轨迹数量少的边会被合并到一起。
    """
    if not selected_edges:
        return []

    num_workers = max(1, min(int(num_workers), len(selected_edges)))
    buckets = [(0, i, []) for i in range(num_workers)]  # (load, worker_id, edges)
    heapq.heapify(buckets)

    for edge in sorted(selected_edges, key=lambda x: x.get("weight", 1), reverse=True):
        load, wid, arr = heapq.heappop(buckets)
        arr.append(edge)
        load += int(edge.get("weight", 1))
        heapq.heappush(buckets, (load, wid, arr))

    buckets.sort(key=lambda x: x[1])
    return [arr for _load, _wid, arr in buckets if arr]


def make_claim_dict(edge_id: int, used_r: List[int]) -> dict:
    return {
        "date": CLAIM_DATE,
        "edge_id": int(edge_id),
        "qs": int(CLAIM_QS),
        "qe": int(CLAIM_QE),
        "claimed_result_traj_ids": [int(x) for x in used_r],
        "claimed_result_traj_count": len([int(x) for x in used_r]),
    }


def worker_prepare_edge_context(db: dict, edge_id: int) -> dict:
    # 正常查询只接收空间/时间范围；结果集合由服务端扫描索引产生。
    used_r = find_matching_trajectory_ids(
        db=db,
        date_str=CLAIM_DATE,
        edge_id=edge_id,
        qs=CLAIM_QS,
        qe=CLAIM_QE,
    )

    # inspect 也不计时：它只是为了生成 R 后准备验证上下文
    inspect = inspect_one_edge_claim(
        db=db,
        date_str=CLAIM_DATE,
        edge_id=edge_id,
        qs=CLAIM_QS,
        qe=CLAIM_QE,
        claimed_result_traj_ids=used_r,
    )

    return {
        "edge_id": int(edge_id),
        "used_r": used_r,
        "inspect": inspect,
        "mapper": PrimeMapper(bits=db["rsa"]["hprime_bits"]),
        "qpair_template": build_query_pair_template(
            CLAIM_QS,
            CLAIM_QE,
            db["rsa"]["day_seconds"] - 1,
            db["rsa"]["depth"],
        ),
        "claim": make_claim_dict(edge_id, used_r),
        "member_batch_vo": None,
        "invalid_claim_items_vo": None,
        "completeness_vo": None,
        "completeness_status": None,
    }


def worker_member_gen(db: dict, ctx: dict) -> dict:
    edge_id = ctx["edge_id"]
    mapper = ctx["mapper"]
    inspect = ctx["inspect"]
    claimed_result_traj_ids = ctx["used_r"]

    member_items = []
    # used_r 由服务端范围查询直接产生，因此这里只会有真实命中项。
    for traj_id in claimed_result_traj_ids:
        traj_id = int(traj_id)
        matched_details = inspect["matched_by_traj"].get(traj_id, [])
        if not matched_details:
            raise AssertionError("服务端范围查询返回了不满足条件的轨迹")
        selected_detail = matched_details[0]
        member_items.append({
            "traj_id": traj_id,
            "plaintext": selected_detail["tr_time"],
            "matched_occurrence_count": len(matched_details),
            "member_witness": selected_detail.get("member_witness"),
        })

    member_batch = generate_member_batch_vo(db, mapper, CLAIM_DATE, edge_id, member_items)
    # Store/transport the VO only in the canonical binary format.  The Python
    # dictionaries are generation-time temporaries and are not shared with the
    # verifier.
    member_batch_vo = encode_member_batch(member_batch)
    ctx["member_batch_vo"] = member_batch_vo
    ctx["invalid_claim_items_vo"] = None

    return {
        "edge_id": edge_id,
        "member_batch_status": member_batch["status"],
        "member_batch_count": len(member_batch.get("member_items", [])),
        "member_proof_mode": member_batch.get("proof_mode", "empty"),
        "invalid_claim_count": 0,
        "member_vo_bytes": len(member_batch_vo),
        "invalid_items_vo_bytes": 0,
    }


def worker_completeness_gen(db: dict, ctx: dict) -> dict:
    edge_id = ctx["edge_id"]
    mapper = ctx["mapper"]
    inspect = ctx["inspect"]

    completeness = None
    if inspect["edge_exists"]:
        if int(db["rsa"].get("format_version", 0)) >= 3:
            completeness = generate_private_completeness_batch(
                db=db,
                date_str=CLAIM_DATE,
                edge_id=edge_id,
                omitted_all=inspect["omitted_all"],
                pair_template=ctx["qpair_template"],
            )
        else:
            completeness = generate_completeness_batch_fast(
                db=db,
                mapper=mapper,
                date_str=CLAIM_DATE,
                edge_id=edge_id,
                qs=CLAIM_QS,
                qe=CLAIM_QE,
                omitted_all=inspect["omitted_all"],
                pair_template=ctx["qpair_template"],
            )
    if completeness is None:
        completeness_vo = None
    elif int(db["rsa"].get("format_version", 0)) >= 3:
        completeness_vo = encode_private_completeness(completeness)
    else:
        completeness_vo = encode_completeness(completeness)
    ctx["completeness_vo"] = completeness_vo
    ctx["completeness_status"] = None if completeness is None else completeness["status"]

    return {
        "edge_id": edge_id,
        "completeness_status": None if completeness is None else completeness["status"],
        "omitted_count": len(inspect["omitted_all"]),
        "completeness_vo_bytes": 0 if completeness_vo is None else len(completeness_vo),
    }


def worker_member_verify(db: dict, ctx: dict) -> dict:
    # 严格对应原 verify_claim_package 的第一段：member_batch + invalid_claim_items
    mapper = PrimeMapper(bits=db["rsa"]["hprime_bits"])
    try:
        member_batch = decode_member_batch(ctx["member_batch_vo"])
        invalid_claim_items = (
            []
            if ctx.get("invalid_claim_items_vo") is None
            else decode_invalid_items(ctx["invalid_claim_items_vo"])
        )
        private_completeness = (
            decode_private_completeness(ctx["completeness_vo"])
            if int(db["rsa"].get("format_version", 0)) >= 3
            else None
        )
    except (TypeError, VOFormatError, ValueError):
        member_batch = {"status": "malformed", "member_items": []}
        invalid_claim_items = []
        member_batch_ok = False
        invalid_items_proofs_ok = False
    else:
        if int(db["rsa"].get("format_version", 0)) >= 3:
            edge_auth_ok = verify_private_edge_auth(
                db, mapper, ctx["claim"], private_completeness
            )
            if edge_auth_ok:
                member_batch_ok, result_tokens = verify_private_member_batch_vo(
                    db,
                    mapper,
                    ctx["claim"],
                    member_batch,
                    private_completeness["edge_ac"],
                )
            else:
                member_batch_ok, result_tokens = False, []
            ctx["verified_result_tokens"] = result_tokens
        else:
            member_batch_ok = verify_member_batch_vo(db, mapper, ctx["claim"], member_batch)

        if int(db["rsa"].get("format_version", 0)) >= 3:
            # v3 客户端不持有 edge_xp/轨迹全集。声明中出现任何无法给出
            # 成员证明的项就直接拒绝，不需要再验证服务端的“无效原因”。
            invalid_items_proofs_ok = not invalid_claim_items
        else:
            invalid_items_proofs_ok = True
            for item in invalid_claim_items:
                invalid_items_proofs_ok = invalid_items_proofs_ok and verify_invalid_claim_item(db, mapper, ctx["claim"], item)

    invalid_claim_count = len(invalid_claim_items)
    soundness_ok = member_batch_ok and invalid_items_proofs_ok and (invalid_claim_count == 0)

    ctx["member_batch_ok"] = member_batch_ok
    ctx["invalid_items_proofs_ok"] = invalid_items_proofs_ok
    ctx["invalid_claim_count"] = invalid_claim_count
    ctx["soundness_ok"] = soundness_ok

    return {
        "edge_id": ctx["edge_id"],
        "member_batch_ok": member_batch_ok,
        "invalid_items_proofs_ok": invalid_items_proofs_ok,
        "invalid_claim_count": invalid_claim_count,
        "soundness_ok": soundness_ok,
    }


def worker_completeness_verify(db: dict, ctx: dict) -> dict:
    try:
        if ctx["completeness_vo"] is None:
            completeness = None
        elif int(db["rsa"].get("format_version", 0)) >= 3:
            completeness = decode_private_completeness(ctx["completeness_vo"])
        else:
            completeness = decode_completeness(ctx["completeness_vo"])
    except (TypeError, VOFormatError, ValueError):
        completeness = None
    if int(db["rsa"].get("format_version", 0)) >= 3:
        completeness_ok = verify_private_completeness_batch(
            db,
            PrimeMapper(bits=db["rsa"]["hprime_bits"]),
            ctx["claim"],
            completeness,
            ctx.get("verified_result_tokens", []),
            ctx.get("qpair_template") or [],
        )
    else:
        completeness_ok = verify_completeness_batch(
            db, ctx["claim"], completeness, ctx.get("qpair_template")
        )
    all_ok = bool(ctx.get("soundness_ok", False)) and completeness_ok

    return {
        "edge_id": ctx["edge_id"],
        "completeness_ok": completeness_ok,
        "completeness_status": ctx.get("completeness_status"),
        "all_ok": all_ok,
    }


def edge_group_worker(conn, input_bin: str, edge_ids: List[int]):
    """
    一个 worker 进程：
    1) 只加载自己负责的边，一次 ADS
    2) 初始化时获取 R + inspect，不计时
    3) 等主进程发四个阶段命令
    """
    try:
        db = load_binary_selected_edges(input_bin, CLAIM_DATE, set(edge_ids))
        contexts = [worker_prepare_edge_context(db, eid) for eid in edge_ids]

        init_summary = []
        for ctx in contexts:
            inspect = ctx["inspect"]
            init_summary.append({
                "edge_id": ctx["edge_id"],
                "edge_exists": inspect["edge_exists"],
                "all_present_traj_count": len(inspect["all_present_traj_ids"]),
                "R_claim_count": len(ctx["used_r"]),
                "omitted_count": len(inspect["omitted_all"]),
            })
        conn.send({"ok": True, "event": "ready", "summary": init_summary})

        while True:
            cmd = conn.recv()
            if cmd == "member_gen":
                conn.send({"ok": True, "event": cmd, "results": [worker_member_gen(db, ctx) for ctx in contexts]})
            elif cmd == "completeness_gen":
                conn.send({"ok": True, "event": cmd, "results": [worker_completeness_gen(db, ctx) for ctx in contexts]})
            elif cmd == "member_verify":
                conn.send({"ok": True, "event": cmd, "results": [worker_member_verify(db, ctx) for ctx in contexts]})
            elif cmd == "completeness_verify":
                conn.send({"ok": True, "event": cmd, "results": [worker_completeness_verify(db, ctx) for ctx in contexts]})
            elif cmd == "close":
                conn.send({"ok": True, "event": "closed"})
                break
            else:
                conn.send({"ok": False, "event": cmd, "error": f"未知命令: {cmd}"})
    except BaseException as e:
        try:
            conn.send({"ok": False, "event": "error", "error": repr(e)})
        except Exception:
            pass
        raise
    finally:
        conn.close()


def run_stage(conns, cmd: str) -> Tuple[float, List[dict]]:
    t0 = time.perf_counter()
    for conn in conns:
        conn.send(cmd)

    results = []
    for conn in conns:
        msg = conn.recv()
        if not msg.get("ok"):
            raise RuntimeError(f"worker 执行 {cmd} 失败: {msg.get('error')}")
        results.extend(msg.get("results", []))
    t1 = time.perf_counter()
    return t1 - t0, results


def close_workers(conns, procs):
    for conn in conns:
        try:
            conn.send("close")
        except Exception:
            pass
    for conn in conns:
        try:
            conn.recv()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
    for p in procs:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()


def main():
    print("========== 多边 claim 查询验证：固定进程 + 智能负载均衡 + 正确快速验证口径 ==========")
    print(f"INPUT_BIN={INPUT_BIN}")
    print(f"CLAIM_DATE={CLAIM_DATE}")
    print(f"range=[{CLAIM_QS},{CLAIM_QE}]")
    print("RESULT_SOURCE=server_range_scan")

    print("\n========== 轻量扫描目标日期边信息（准备阶段，不计入四个核心时间）==========")
    rsa_meta, edge_infos = scan_edges_for_date(INPUT_BIN, CLAIM_DATE)
    print(f"base={rsa_meta['base']}, depth={rsa_meta['depth']}, hprime_bits={rsa_meta['hprime_bits']}")
    print(f"date_edge_count={len(edge_infos)}")

    selected = choose_query_edges(edge_infos)
    if not selected:
        raise ValueError("没有选中任何边：请检查 QUERY_EDGE_IDS 或 AUTO_PICK_EDGE_COUNT")

    actual_workers = max(1, min(int(NUM_WORKERS), len(selected)))
    assignments = balance_edges_by_weight(selected, actual_workers)

    print("\n========== 选择的查询边 ==========")
    print(f"selected_edge_count={len(selected)}")
    print(f"NUM_WORKERS_CONFIG={NUM_WORKERS}")
    print(f"actual_worker_count={len(assignments)}")
    print("top_selected_edges_by_traj_count=")
    for item in sorted(selected, key=lambda x: x.get("traj_count", 0), reverse=True)[:20]:
        print(f"  edge={item['edge_id']}, traj_count={item.get('traj_count', 0)}, occ_total={item.get('occ_total', 0)}")

    if PRINT_EDGE_ASSIGNMENT:
        print("\n========== 智能分配结果 ==========")
        for i, group in enumerate(assignments):
            load = sum(int(x.get("weight", 1)) for x in group)
            eids = [int(x["edge_id"]) for x in group]
            print(f"worker_{i}: load≈{load}, edge_count={len(group)}, edges={eids}")

    print("\n========== 启动 worker，加载各自负责的 ADS 边 ==========")
    ctx = mp.get_context("spawn")
    conns = []
    procs = []

    try:
        for group in assignments:
            edge_ids = [int(x["edge_id"]) for x in group]
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=edge_group_worker, args=(child_conn, INPUT_BIN, edge_ids))
            p.start()
            child_conn.close()
            conns.append(parent_conn)
            procs.append(p)

        init_summaries = []
        for conn in conns:
            msg = conn.recv()
            if not msg.get("ok"):
                raise RuntimeError(f"worker 初始化失败: {msg.get('error')}")
            init_summaries.extend(msg.get("summary", []))
        print("worker_load_prepare_R_and_inspect_done=True")
        print("说明: worker 加载自己负责的边、自动生成 R、inspect、负载均衡都属于准备阶段，不做计时，也不计入下面四个核心时间。")

        print("\n========== 每条边准备情况 ==========")
        for item in sorted(init_summaries, key=lambda x: x["edge_id"]):
            print(
                f"edge={item['edge_id']}, exists={item['edge_exists']}, "
                f"all_present={item['all_present_traj_count']}, "
                f"R_claim_count={item['R_claim_count']}, omitted_count={item['omitted_count']}"
            )

        print("\n========== 生成 VO ==========")
        member_batch_gen_time, member_gen_results = run_stage(conns, "member_gen")
        completeness_gen_time, comp_gen_results = run_stage(conns, "completeness_gen")

        print("\n========== 验证 VO ==========")
        member_batch_verify_time, member_ver_results = run_stage(conns, "member_verify")
        completeness_verify_time, comp_ver_results = run_stage(conns, "completeness_verify")

        all_ok = all(x.get("soundness_ok", False) for x in member_ver_results) and all(x.get("completeness_ok", False) for x in comp_ver_results)

        print("\n========== 汇总结果 ==========")
        print(f"edge_count = {len(selected)}")
        print(f"worker_count = {len(assignments)}")
        print(f"all_ok = {all_ok}")

        print("\n========== 生成时间 ==========")
        print(f"member_batch_gen_time = {member_batch_gen_time:.6f}s")
        print(f"completeness_gen_time = {completeness_gen_time:.6f}s")

        print("\n========== 验证时间 ==========")
        print(f"member_batch_verify_time = {member_batch_verify_time:.6f}s")
        print(f"completeness_verify_time = {completeness_verify_time:.6f}s")

        summary = {
            "input_bin": INPUT_BIN,
            "claim_date": CLAIM_DATE,
            "claim_range": [CLAIM_QS, CLAIM_QE],
            "result_source": "server_range_scan",
            "selected_edges": selected,
            "assignments": [
                {
                    "worker_id": i,
                    "load": sum(int(x.get("weight", 1)) for x in group),
                    "edge_ids": [int(x["edge_id"]) for x in group],
                }
                for i, group in enumerate(assignments)
            ],
            "init_summaries": init_summaries,
            "member_gen_results": member_gen_results,
            "completeness_gen_results": comp_gen_results,
            "member_verify_results": member_ver_results,
            "completeness_verify_results": comp_ver_results,
            "timing": {
                "member_batch_gen_time": member_batch_gen_time,
                "completeness_gen_time": completeness_gen_time,
                "member_batch_verify_time": member_batch_verify_time,
                "completeness_verify_time": completeness_verify_time,
            },
            "all_ok": all_ok,
        }

        with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nsummary_written={SUMMARY_JSON}")

    finally:
        close_workers(conns, procs)


if __name__ == "__main__":
    mp.freeze_support()
    main()
