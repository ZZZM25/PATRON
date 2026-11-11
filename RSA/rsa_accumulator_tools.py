# -*- coding: utf-8 -*-
"""
rsa_accumulator_tools.py

从 multi_edge_claim_verify_member_batch_parallel_v2.py 中抽离出来的底层工具函数。

包含：
- 二进制读取小工具
- bigint / 编码转换工具
- 字符串到定长大素数映射 PrimeMapper
- product_tree 快速乘法
- signed_powmod 快速幂，支持负指数
- Bézout 非成员见证生成
- 查询 pair seed 构造
- x 构造工具

注意：本文件不放具体查询 claim、验证流程、多进程调度。
"""

import struct
import hashlib
from typing import Dict, List, Tuple

from gmpy2 import mpz, next_prime, gcdext, powmod, invert


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


def intervals_intersect(a1: int, a2: int, b1: int, b2: int) -> bool:
    return not (a2 < b1 or b2 < a1)


def member_plaintext_x(mapper: PrimeMapper, plaintext: str) -> mpz:
    return mapper.hash_to_prime(plaintext)


def trajectory_absence_x(mapper: PrimeMapper, edge_id: int, traj_id: int) -> mpz:
    return mapper.hash_to_prime(edge_tr_seed(edge_id, traj_id))


def trajectory_query_time_x(mapper: PrimeMapper, edge_id: int, traj_id: int, qs: int, qe: int, day_max: int, depth: int):
    seeds = query_pair_seeds(edge_id, traj_id, qs, qe, day_max, depth)
    primes = [mapper.hash_to_prime(s) for s in seeds]
    x = product_tree(primes)
    return x, seeds


def shorten_bigint(x: mpz, as_hex: bool = True, keep: int = 24) -> str:
    s = bigint_to_str(x, as_hex)
    if len(s) <= keep * 2:
        return s
    return s[:keep] + "..." + s[-keep:]
