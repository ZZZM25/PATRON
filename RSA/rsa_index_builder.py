# -*- coding: utf-8 -*-
import ast
import csv
import hashlib
import hmac
import multiprocessing as mp
import secrets
import struct
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta

import gmpy2
from gmpy2 import mpz, next_prime, gcd, powmod, invert

# =========================
# 配置区
# =========================
TRAJ_CSV_FILES = ["traj-10-2.csv"]
OUTPUT_BIN = "traj-10-2-rsa-fast-with-loader.bin"

TIMEZONE = "UTC"
# TIMEZONE = "UTC+8"

BASE = 2
DAY_SECONDS = 86400
REAL_MAX_SECOND = DAY_SECONDS - 1

# 这里直接内置一组固定的模拟 RSA 参数，方便你直接跑。
# 这是构建端/数据拥有者使用的测试参数，不需要手填。
USE_EXISTING_RSA = True
EXISTING_P = "7846549115030320064050237400219339536752286899539720005594939671131577842772150376845047818662417918520544227280021192478154645419851622759404419438153841"
EXISTING_Q = "11945327180965262971701330923063012540339609572585568473568948898130851562050333336711728312930071420927612619334257779428434339889103693243983036293160413"
EXISTING_N = "93729596420550612101573256736102135512350085310873858514115969575805342667476451979576268447291352865083680843665091563121580364469547042456049270615007175406718403916463386273618925683856146895449233017404589693205113426306834172953435303323701722590297461337125038759853971648112068440200598755683985096333"
EXISTING_G = "3898915189748737476828018217113662789263339032608491572964167767165564269697509610511935058512175946414072267190976707377205635453770965465985113410370733316731502257266348145585435079210823661666231906091920611377887532154878788351804796125445160103275703502296553764633623089861270491450966814450093474096"

RSA_P_BITS = 512
RSA_Q_BITS = 512
HPRIME_BITS = 64

NUM_WORKERS = 6
CHUNK_SIZE = 32

INCLUDE_EMPTY_EDGES = False
EDGE_CSV = "new_edges.csv"

PRINT_TIMING = True
PRINT_PROGRESS_EVERY_TRAJ = 5000
CHECK_AFTER_WRITE = True

# 大数据集建议开启：在同一工作进程内完成 prime 准备和累加器计算，
# 避免把巨大的 factor_list 序列化回主进程后再发送一次。
FUSE_PREPARE_AND_ACCUMULATOR = True
FUSED_CHUNK_SIZE = 1

LOAD_BIGINT_AS_HEX_STR = True
LOAD_REBUILD_QUERY_PAIR = False

# v2 索引会在可信构建端为每个时间 occurrence 预计算 RSA membership witness。
# p/q 只在本文件的构建流程中使用；输出文件仍然只包含 n、g、累加器和公开 witness。
PRECOMPUTE_MEMBER_WITNESSES = True
BINARY_VERSION = 2 if PRECOMPUTE_MEMBER_WITNESSES else 1

# v3 隐私索引：数据拥有者为每个 (date, edge, trajectory) 生成固定长度、
# 不可逆的不透明令牌。令牌进入身份/时间素数，客户端只看到令牌而看不到
# 非结果轨迹 id。密钥只在可信构建端使用，不写入 ADS。
OPAQUE_TOKEN_BYTES = 16
OPAQUE_TOKEN_KEY = None


# =========================
# 时间与基础工具
# =========================
def get_tz_offset_seconds(tz_text: str) -> int:
    tz = tz_text.strip().upper()
    if tz == "UTC":
        return 0
    if tz.startswith("UTC+"):
        return int(tz.replace("UTC+", "")) * 3600
    if tz.startswith("UTC-"):
        return -int(tz.replace("UTC-", "")) * 3600
    raise ValueError("TIMEZONE 只能写成 'UTC'、'UTC+8'、'UTC-5'")


TZ_OFFSET_SECONDS = get_tz_offset_seconds(TIMEZONE)
_EPOCH_DATE = date(1970, 1, 1)


def clean_key(k):
    if k is None:
        return None
    return k.strip().lstrip("\ufeff")


def parse_trajectory_data(cell: str):
    cell = cell.strip()
    try:
        import json
        return json.loads(cell)
    except Exception:
        return ast.literal_eval(cell)


def calc_depth(domain_size: int, base: int) -> int:
    depth = 0
    size = 1
    while size < domain_size:
        size *= base
        depth += 1
    return depth


DEPTH = calc_depth(DAY_SECONDS, BASE)


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


def local_day_and_second(ts: int, day_cache: dict):
    local_ts = ts + TZ_OFFSET_SECONDS
    day_idx = local_ts // DAY_SECONDS
    second = local_ts % DAY_SECONDS
    if second < 0:
        second += DAY_SECONDS
        day_idx -= 1
    day_str = day_cache.get(day_idx)
    if day_str is None:
        day_str = (_EPOCH_DATE + timedelta(days=day_idx)).isoformat()
        day_cache[day_idx] = day_str
    return day_idx, day_str, second


# =========================
# 二进制编码工具
# =========================
def mpz_to_bytes(x) -> bytes:
    x = mpz(x)
    if x == 0:
        return b"\x00"
    nbytes = (x.bit_length() + 7) // 8
    return int(x).to_bytes(nbytes, "big")


def bytes_to_mpz(b: bytes) -> mpz:
    return mpz(int.from_bytes(b, "big"))


def bigint_to_str(x, as_hex=True) -> str:
    x = mpz(x)
    return x.digits(16) if as_hex else x.digits(10)


def write_u8(f, x: int):
    f.write(struct.pack(">B", x))


def write_u16(f, x: int):
    f.write(struct.pack(">H", x))


def write_u32(f, x: int):
    f.write(struct.pack(">I", x))


def write_i32(f, x: int):
    f.write(struct.pack(">i", x))


def read_u8(f) -> int:
    return struct.unpack(">B", f.read(1))[0]


def read_u16(f) -> int:
    return struct.unpack(">H", f.read(2))[0]


def read_u32(f) -> int:
    return struct.unpack(">I", f.read(4))[0]


def read_i32(f) -> int:
    return struct.unpack(">i", f.read(4))[0]


def write_varint(f, x: int):
    if x < 0:
        raise ValueError("这里只支持非负 varint")
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            f.write(bytes([b | 0x80]))
        else:
            f.write(bytes([b]))
            break


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


def write_varbytes(f, b: bytes):
    write_varint(f, len(b))
    f.write(b)


def read_varbytes(f) -> bytes:
    n = read_varint(f)
    return f.read(n)


def pack_occurrence(start_second: int, end_second: int) -> bytes:
    if not (0 <= start_second <= REAL_MAX_SECOND):
        raise ValueError(f"start_second 越界: {start_second}")
    if not (0 <= end_second <= REAL_MAX_SECOND):
        raise ValueError(f"end_second 越界: {end_second}")
    v = (start_second << 17) | end_second
    return int(v).to_bytes(5, "big")


def unpack_occurrence(b: bytes):
    if len(b) != 5:
        raise ValueError("occurrence 必须是 5 字节")
    v = int.from_bytes(b, "big")
    start_second = (v >> 17) & ((1 << 17) - 1)
    end_second = v & ((1 << 17) - 1)
    return start_second, end_second


# =========================
# RSA / hash-to-prime
# =========================
def random_prime(bits: int) -> mpz:
    while True:
        x = secrets.randbits(bits)
        x |= (1 << (bits - 1))
        x |= 1
        p = next_prime(mpz(x))
        if p.bit_length() == bits:
            return p


def generate_rsa_params():
    p = random_prime(RSA_P_BITS)
    q = random_prime(RSA_Q_BITS)
    while q == p:
        q = random_prime(RSA_Q_BITS)
    n = p * q
    while True:
        h = mpz(secrets.randbits(n.bit_length() - 1)) % n
        if h > 1 and gcd(h, n) == 1:
            g = powmod(h, 2, n)
            if g > 1:
                return p, q, n, g


def ensure_rsa_params():
    if USE_EXISTING_RSA:
        if not EXISTING_P or not EXISTING_Q or not EXISTING_N or not EXISTING_G:
            raise ValueError("数据拥有者快速构建模式下，USE_EXISTING_RSA=True 时必须填写 EXISTING_P、EXISTING_Q、EXISTING_N、EXISTING_G")
        p = mpz(EXISTING_P)
        q = mpz(EXISTING_Q)
        n = mpz(EXISTING_N)
        g = mpz(EXISTING_G)
        if p * q != n:
            raise ValueError("EXISTING_P * EXISTING_Q 必须等于 EXISTING_N")
        return p, q, n, g
    return generate_rsa_params()


def hash_to_prime_local(seed: str, cache: dict, bits: int = HPRIME_BITS) -> mpz:
    seed = str(seed)
    p = cache.get(seed)
    if p is not None:
        return p
    mask = (1 << bits) - 1
    counter = 0
    while True:
        digest = hashlib.sha256(f"{seed}|{counter}".encode("utf-8")).digest()
        x = int.from_bytes(digest, "big") & mask
        x |= (1 << (bits - 1))
        x |= 1
        p = next_prime(mpz(x))
        if p.bit_length() == bits:
            cache[seed] = p
            return p
        counter += 1


def product_tree(nums):
    if not nums:
        return mpz(1)
    level = [mpz(x) for x in nums]
    while len(level) > 1:
        next_level = []
        n = len(level)
        for i in range(0, n - 1, 2):
            next_level.append(level[i] * level[i + 1])
        if n % 2 == 1:
            next_level.append(level[-1])
        level = next_level
    return level[0]


# =========================
# 前缀 / tr_time / query_pair 动态恢复
# =========================
def edge_tr_seed(edge_id, trajectory_id):
    return f"{edge_id}|{trajectory_id}"


def tr_time_seed(edge_id, trajectory_id, start_code, end_code):
    return f"{edge_id}|{trajectory_id}|{{{start_code},{end_code}}}"


def private_edge_seed(day_idx: int, edge_id: int) -> str:
    return f"RSG-EDGE-V3|{int(day_idx)}|{int(edge_id)}"


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


def opaque_trajectory_token(day_idx: int, edge_id: int, traj_id: int) -> bytes:
    key = WORKER_CTX.get("opaque_token_key")
    if not key:
        raise ValueError("v3 构建需要数据拥有者 OPAQUE_TOKEN_KEY")
    message = f"RSG-TOKEN-V1|{int(day_idx)}|{int(edge_id)}|{int(traj_id)}".encode("ascii")
    return hmac.new(key, message, hashlib.sha256).digest()[:OPAQUE_TOKEN_BYTES]


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


def rebuild_query_pair_from_occ(edge_id: int, traj_id: int, start_second: int, end_second: int):
    start_code = to_base_fixed(start_second, BASE, DEPTH)
    end_code = to_base_fixed(end_second, BASE, DEPTH)
    s_path = optimized_start_path(start_code)
    e_path = optimized_end_path(end_code)

    pair_list = []
    seen = set()
    eid = str(edge_id)
    tid = str(traj_id)

    for sp in s_path:
        for ep in e_path:
            seed = f"{eid}|{tid}|{sp}|{ep}"
            if seed not in seen:
                seen.add(seed)
                pair_list.append(seed)
    return pair_list


# =========================
# 第一阶段：从 traj CSV 聚合最小信息
# =========================
def ensure_edge_bucket(data, day_idx, edge_id):
    date_bucket = data.setdefault(day_idx, {})
    edge_bucket = date_bucket.setdefault(int(edge_id), defaultdict(list))
    return edge_bucket


def add_segment(data, day_idx, edge_id, traj_id, start_second, end_second):
    edge_bucket = ensure_edge_bucket(data, day_idx, edge_id)
    edge_bucket[int(traj_id)].append((int(start_second), int(end_second)))


def load_all_edge_ids(edge_csv: str):
    edge_ids = []
    with open(edge_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{edge_csv} 文件为空或者没有表头")
        reader.fieldnames = [clean_key(x) for x in reader.fieldnames]
        edge_id_col = None
        for c in ["edge_id", "new_edge_id", "id"]:
            if c in reader.fieldnames:
                edge_id_col = c
                break
        if edge_id_col is None:
            raise KeyError(f"{edge_csv} 找不到 edge_id 列")
        for row in reader:
            row = {clean_key(k): v for k, v in row.items() if k is not None}
            edge_ids.append(int(row[edge_id_col]))
    return edge_ids


def process_trajectory_file(traj_csv, data):
    stats = Counter()
    traj_line_count = 0
    day_cache = {}

    max_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_field_size)
            break
        except OverflowError:
            max_field_size //= 10

    with open(traj_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{traj_csv} 文件为空或者没有表头")
        reader.fieldnames = [clean_key(x) for x in reader.fieldnames]

        if "trajectory_id" not in reader.fieldnames:
            raise KeyError(f"{traj_csv} 缺少 trajectory_id 列")
        if "trajectory_data" not in reader.fieldnames:
            raise KeyError(f"{traj_csv} 缺少 trajectory_data 列")

        for line_no, row in enumerate(reader, start=2):
            traj_line_count += 1
            if PRINT_PROGRESS_EVERY_TRAJ and traj_line_count % PRINT_PROGRESS_EVERY_TRAJ == 0:
                print(f"  已处理 trajectory 行数: {traj_line_count:,}")

            row = {clean_key(k): v for k, v in row.items() if k is not None}
            traj_id = int(row["trajectory_id"])

            try:
                trajectory_data = parse_trajectory_data(row["trajectory_data"])
            except Exception as e:
                print(f"第 {line_no} 行 trajectory_data 解析失败: {e}")
                stats["parse_failed_rows"] += 1
                continue

            for seg in trajectory_data:
                stats["total_segments"] += 1
                if len(seg) != 5:
                    stats["bad_segments"] += 1
                    continue

                edge_id, start_node, start_ts, end_node, end_ts = seg
                edge_id = int(edge_id)
                start_ts = int(start_ts)
                end_ts = int(end_ts)

                if end_ts < start_ts:
                    stats["end_ts_less_than_start_ts"] += 1
                    continue

                start_day_idx, _, start_second = local_day_and_second(start_ts, day_cache)
                end_day_idx, _, end_second = local_day_and_second(end_ts, day_cache)

                if start_day_idx != end_day_idx:
                    stats["cross_day_segments_skipped"] += 1
                    continue

                add_segment(data, start_day_idx, edge_id, traj_id, start_second, end_second)
                stats["same_day_segments"] += 1

    return stats


# =========================
# 第二阶段：并行按 edge 计算 edge_xp / edge_ac
# =========================
WORKER_CTX = {}


def worker_init(
        p_bytes: bytes,
        q_bytes: bytes,
        n_bytes: bytes,
        g_bytes: bytes,
        opaque_token_key: bytes = b"",
        binary_version: int = None,
        opaque_token_bytes: int = None,
        time_domain_size: int = None,
):
    global BINARY_VERSION, OPAQUE_TOKEN_BYTES, DAY_SECONDS, REAL_MAX_SECOND, DEPTH
    if binary_version is not None:
        BINARY_VERSION = int(binary_version)
    if opaque_token_bytes is not None:
        OPAQUE_TOKEN_BYTES = int(opaque_token_bytes)
    if time_domain_size is not None:
        DAY_SECONDS = int(time_domain_size)
        REAL_MAX_SECOND = DAY_SECONDS - 1
        DEPTH = calc_depth(DAY_SECONDS, BASE)
    p = bytes_to_mpz(p_bytes)
    q = bytes_to_mpz(q_bytes)
    n = bytes_to_mpz(n_bytes)
    g = bytes_to_mpz(g_bytes)

    WORKER_CTX["p"] = p
    WORKER_CTX["q"] = q
    WORKER_CTX["n"] = n
    WORKER_CTX["g"] = g
    WORKER_CTX["g_mod_p"] = g % p
    WORKER_CTX["g_mod_q"] = g % q
    WORKER_CTX["qr_order_p"] = (p - 1) // 2
    WORKER_CTX["qr_order_q"] = (q - 1) // 2
    WORKER_CTX["p_inv_mod_q"] = invert(p, q)
    WORKER_CTX["opaque_token_key"] = bytes(opaque_token_key)
    WORKER_CTX["prime_cache"] = {}
    WORKER_CTX["pair_prime_cache"] = {}


def build_pair_prime_list_from_occ(
        edge_id: int,
        traj_id: int,
        start_second: int,
        end_second: int,
        *,
        day_idx: int = None,
        token_hex: str = None,
):
    key = (day_idx, edge_id, token_hex, traj_id, start_second, end_second)
    cache = WORKER_CTX["pair_prime_cache"]
    cached = cache.get(key)
    if cached is not None:
        return cached

    start_code = to_base_fixed(start_second, BASE, DEPTH)
    end_code = to_base_fixed(end_second, BASE, DEPTH)

    s_path = optimized_start_path(start_code)
    e_path = optimized_end_path(end_code)

    pair_prime_set = set()
    pair_prime_list = []
    local_prime_cache = WORKER_CTX["prime_cache"]

    for sp in s_path:
        for ep in e_path:
            if BINARY_VERSION >= 3:
                if day_idx is None or token_hex is None:
                    raise ValueError("v3 query-pair 需要 day_idx 和 opaque token")
                seed = private_time_pair_seed(day_idx, edge_id, token_hex, sp, ep)
            else:
                seed = f"{edge_id}|{traj_id}|{sp}|{ep}"
            p = int(hash_to_prime_local(seed, local_prime_cache))
            if p not in pair_prime_set:
                pair_prime_set.add(p)
                pair_prime_list.append(p)

    cache[key] = pair_prime_list
    return pair_prime_list


def compute_edge_prepare_task(task):
    if BINARY_VERSION >= 3:
        day_idx, edge_id, traj_map = task
    else:
        edge_id, traj_map = task
        day_idx = None

    local_prime_cache = WORKER_CTX["prime_cache"]

    edge_seed = (
        private_edge_seed(day_idx, edge_id)
        if BINARY_VERSION >= 3
        else str(edge_id)
    )
    factor_list = [int(hash_to_prime_local(edge_seed, local_prime_cache))]
    identity_primes = []
    tr_time_counter = Counter()
    pair_prime_set = set()
    traj_rows = []

    for traj_id, occs in traj_map.items():
        if BINARY_VERSION >= 3:
            token = opaque_trajectory_token(day_idx, edge_id, traj_id)
            token_hex = token.hex()
            edge_tr = private_identity_seed(day_idx, edge_id, token_hex)
        else:
            token = b""
            token_hex = ""
            edge_tr = edge_tr_seed(edge_id, traj_id)
        identity_prime = int(hash_to_prime_local(edge_tr, local_prime_cache))
        factor_list.append(identity_prime)
        identity_primes.append(identity_prime)

        packed_occs = []
        for start_second, end_second in occs:
            start_code = to_base_fixed(start_second, BASE, DEPTH)
            end_code = to_base_fixed(end_second, BASE, DEPTH)
            tr_time = (
                private_occurrence_seed(
                    day_idx, edge_id, traj_id, token_hex, start_code, end_code
                )
                if BINARY_VERSION >= 3
                else tr_time_seed(edge_id, traj_id, start_code, end_code)
            )
            tr_time_counter[tr_time] += 1
            tr_time_prime = int(hash_to_prime_local(tr_time, local_prime_cache))
            packed_occs.append((pack_occurrence(start_second, end_second), tr_time_prime))

            pair_prime_set.update(
                build_pair_prime_list_from_occ(
                    edge_id,
                    traj_id,
                    start_second,
                    end_second,
                    day_idx=day_idx,
                    token_hex=token_hex,
                )
            )

        if BINARY_VERSION >= 3:
            traj_rows.append((int(traj_id), token, packed_occs))
        else:
            traj_rows.append((int(traj_id), packed_occs))

    for tr_time, count in tr_time_counter.items():
        p = hash_to_prime_local(tr_time, local_prime_cache)
        factor_list.append(int(p if count == 1 else p ** count))

    factor_list.extend(int(x) for x in pair_prime_set)

    return {
        "edge_id": int(edge_id),
        "factor_list": factor_list,
        "identity_xp": product_tree(identity_primes),
        "traj_rows": traj_rows,
        "traj_count": len(traj_rows),
        "tr_time_unique_count": len(tr_time_counter),
        "pair_unique_count": len(pair_prime_set),
    }


def compute_edge_accumulator_task(task):
    edge_id, factor_list = task

    p = WORKER_CTX["p"]
    q = WORKER_CTX["q"]
    g_mod_p = WORKER_CTX["g_mod_p"]
    g_mod_q = WORKER_CTX["g_mod_q"]
    qr_order_p = WORKER_CTX["qr_order_p"]
    qr_order_q = WORKER_CTX["qr_order_q"]
    p_inv_mod_q = WORKER_CTX["p_inv_mod_q"]

    edge_xp = product_tree(factor_list)

    exp_p = edge_xp % qr_order_p
    exp_q = edge_xp % qr_order_q

    ac_p = powmod(g_mod_p, exp_p, p)
    ac_q = powmod(g_mod_q, exp_q, q)

    t = ((ac_q - ac_p) * p_inv_mod_q) % q
    edge_ac = ac_p + p * t

    return {
        "edge_id": int(edge_id),
        "edge_xp_bytes": mpz_to_bytes(edge_xp),
        "edge_ac_bytes": mpz_to_bytes(edge_ac),
        "xp_bits": edge_xp.bit_length(),
        "exp_p_bits": mpz(exp_p).bit_length(),
        "exp_q_bits": mpz(exp_q).bit_length(),
    }


def compute_member_witness(edge_xp: mpz, representative: mpz) -> mpz:
    """Use the trusted RSA trapdoor to remove one representative from edge_xp."""
    p = WORKER_CTX["p"]
    q = WORKER_CTX["q"]
    g_mod_p = WORKER_CTX["g_mod_p"]
    g_mod_q = WORKER_CTX["g_mod_q"]
    order_p = WORKER_CTX["qr_order_p"]
    order_q = WORKER_CTX["qr_order_q"]
    p_inv_mod_q = WORKER_CTX["p_inv_mod_q"]
    representative = mpz(representative)

    def quotient_mod_order(order: mpz) -> mpz:
        inv = invert(representative % order, order)
        if inv != 0:
            return ((edge_xp % order) * inv) % order
        # 极小概率下 representative 与群阶不互素；退回精确除法仍然正确。
        if edge_xp % representative != 0:
            raise ValueError("member representative 不能整除 edge_xp")
        return (edge_xp // representative) % order

    witness_p = powmod(g_mod_p, quotient_mod_order(order_p), p)
    witness_q = powmod(g_mod_q, quotient_mod_order(order_q), q)
    t = ((witness_q - witness_p) * p_inv_mod_q) % q
    return witness_p + p * t


def root_witness_worker_init(
        p_bytes: bytes,
        q_bytes: bytes,
        n_bytes: bytes,
        g_bytes: bytes,
        opaque_token_key: bytes,
        binary_version: int,
        opaque_token_bytes: int,
        root_xp_bytes: bytes,
):
    """Initialize one worker for parallel edge-to-root membership witnesses."""
    worker_init(
        p_bytes,
        q_bytes,
        n_bytes,
        g_bytes,
        opaque_token_key,
        binary_version,
        opaque_token_bytes,
    )
    WORKER_CTX["root_xp"] = bytes_to_mpz(root_xp_bytes)


def compute_root_witness_task(task):
    edge_id, representative = task
    witness = compute_member_witness(WORKER_CTX["root_xp"], mpz(representative))
    modulus_bytes = (WORKER_CTX["n"].bit_length() + 7) // 8
    return int(edge_id), int(witness).to_bytes(modulus_bytes, "big")


def attach_member_witnesses(traj_rows, edge_xp: mpz):
    """Attach one fixed-width public witness to every occurrence row."""
    if not PRECOMPUTE_MEMBER_WITNESSES:
        if BINARY_VERSION >= 3:
            return [
                (traj_id, token, [occ for occ, _representative in occs])
                for traj_id, token, occs in traj_rows
            ]
        return [
            (traj_id, [occ for occ, _representative in occs])
            for traj_id, occs in traj_rows
        ]

    modulus_bytes = (WORKER_CTX["n"].bit_length() + 7) // 8
    witness_cache = {}
    output_rows = []
    normalized_rows = (
        traj_rows
        if BINARY_VERSION >= 3
        else [(traj_id, b"", occs) for traj_id, occs in traj_rows]
    )
    for traj_id, token, occs in normalized_rows:
        output_occs = []
        for occ, representative in occs:
            witness = witness_cache.get(representative)
            if witness is None:
                witness = compute_member_witness(edge_xp, mpz(representative))
                witness_cache[representative] = witness
            output_occs.append((occ, int(witness).to_bytes(modulus_bytes, "big")))
        if BINARY_VERSION >= 3:
            output_rows.append((traj_id, token, output_occs))
        else:
            output_rows.append((traj_id, output_occs))
    return output_rows


def compute_edge_fused_task(task):
    """内存友好的 edge 构建：factor_list 始终留在同一个子进程。"""
    WORKER_CTX["prime_cache"].clear()
    WORKER_CTX["pair_prime_cache"].clear()
    prepared = compute_edge_prepare_task(task)
    acc = compute_edge_accumulator_task((prepared["edge_id"], prepared["factor_list"]))
    edge_xp = bytes_to_mpz(acc["edge_xp_bytes"])
    identity_witness = None
    if BINARY_VERSION >= 3:
        identity_witness = compute_member_witness(edge_xp, prepared["identity_xp"])
    result = {
        "edge_id": prepared["edge_id"],
        "edge_xp_bytes": acc["edge_xp_bytes"],
        "edge_ac_bytes": acc["edge_ac_bytes"],
        "traj_rows": attach_member_witnesses(prepared["traj_rows"], edge_xp),
        "identity_witness_bytes": (
            None
            if identity_witness is None
            else int(identity_witness).to_bytes((WORKER_CTX["n"].bit_length() + 7) // 8, "big")
        ),
        "traj_count": prepared["traj_count"],
        "tr_time_unique_count": prepared["tr_time_unique_count"],
        "pair_unique_count": prepared["pair_unique_count"],
        "xp_bits": acc["xp_bits"],
    }
    del prepared
    WORKER_CTX["prime_cache"].clear()
    WORKER_CTX["pair_prime_cache"].clear()
    return result


# =========================
# 第三阶段：写二进制文件
# =========================
def edge_ac_seed(
        edge_id: str,
        edge_ac: mpz,
        *,
        day_idx: int = None,
        traj_count: int = None,
) -> str:
    if BINARY_VERSION >= 3:
        if day_idx is None or traj_count is None:
            raise ValueError("v3 edge descriptor 需要 day_idx 和 traj_count")
        return (
            f"RSG-EDGE-DESCRIPTOR-V3|{int(day_idx)}|{int(edge_id)}|"
            f"{int(traj_count)}|{mpz(edge_ac).digits(16)}"
        )
    return str(edge_ac)


def write_binary(output_path: str, n: mpz, g: mpz, data_results: dict):
    with open(output_path, "wb") as f:
        f.write(b"RSGA")
        write_u16(f, BINARY_VERSION)
        write_u8(f, BASE)
        write_u8(f, OPAQUE_TOKEN_BYTES if BINARY_VERSION >= 3 else 0)
        write_u32(f, DAY_SECONDS)
        write_i32(f, TZ_OFFSET_SECONDS)
        write_u16(f, HPRIME_BITS)
        write_u16(f, 0)

        write_varbytes(f, mpz_to_bytes(n))
        write_varbytes(f, mpz_to_bytes(g))

        day_items = sorted(data_results.items(), key=lambda x: x[0])
        write_varint(f, len(day_items))

        for day_idx, day_obj in day_items:
            write_varint(f, day_idx)
            write_varbytes(f, day_obj["data_xp_bytes"])
            write_varbytes(f, day_obj["data_ac_bytes"])

            edges = sorted(day_obj["edges"], key=lambda x: x["edge_id"])
            write_varint(f, len(edges))

            for edge_obj in edges:
                write_varint(f, edge_obj["edge_id"])
                write_varbytes(f, edge_obj["edge_xp_bytes"])
                write_varbytes(f, edge_obj["edge_ac_bytes"])
                if BINARY_VERSION >= 3:
                    root_witness = edge_obj.get("root_witness_bytes")
                    identity_witness = edge_obj.get("identity_witness_bytes")
                    if not root_witness or not identity_witness:
                        raise ValueError("v3 edge 缺少 root/identity witness")
                    f.write(root_witness)
                    f.write(identity_witness)

                traj_rows = sorted(edge_obj["traj_rows"], key=lambda x: x[0])
                write_varint(f, len(traj_rows))

                normalized_rows = (
                    traj_rows
                    if BINARY_VERSION >= 3
                    else [(traj_id, b"", occs) for traj_id, occs in traj_rows]
                )
                for traj_id, token, occs in normalized_rows:
                    write_varint(f, traj_id)
                    if BINARY_VERSION >= 3:
                        if len(token) != OPAQUE_TOKEN_BYTES:
                            raise ValueError("opaque token 长度错误")
                        f.write(token)
                    write_varint(f, len(occs))
                    for item in occs:
                        if BINARY_VERSION >= 2:
                            occ, member_witness = item
                            f.write(occ)
                            f.write(member_witness)
                        else:
                            f.write(item)


# =========================
# 二进制完整加载：还原成 dict
# =========================
def load_binary_to_dict(path: str, as_hex: bool = LOAD_BIGINT_AS_HEX_STR, rebuild_query_pair: bool = LOAD_REBUILD_QUERY_PAIR):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"RSGA":
            raise ValueError(f"不是本程序生成的二进制文件，magic={magic!r}")

        version = read_u16(f)
        if version not in (1, 2):
            raise ValueError(f"暂不支持 version={version}")

        base = read_u8(f)
        _reserved = read_u8(f)
        day_seconds = read_u32(f)
        tz_offset_seconds = read_i32(f)
        hprime_bits = read_u16(f)
        _reserved2 = read_u16(f)

        n = bytes_to_mpz(read_varbytes(f))
        g = bytes_to_mpz(read_varbytes(f))
        modulus_bytes = (n.bit_length() + 7) // 8
        date_count = read_varint(f)

        result = {
            "rsa": {
                "n": bigint_to_str(n, as_hex),
                "g": bigint_to_str(g, as_hex),
                "base": base,
                "day_seconds": day_seconds,
                "tz_offset_seconds": tz_offset_seconds,
                "hprime_bits": hprime_bits,
                "bigint_encoding": "hex" if as_hex else "decimal",
            },
            "data": {}
        }

        for _ in range(date_count):
            day_idx = read_varint(f)
            date_str = (_EPOCH_DATE + timedelta(days=day_idx)).isoformat()

            data_xp = bytes_to_mpz(read_varbytes(f))
            data_ac = bytes_to_mpz(read_varbytes(f))
            edge_count = read_varint(f)

            date_obj = {
                "data_xp": bigint_to_str(data_xp, as_hex),
                "data_ac": bigint_to_str(data_ac, as_hex),
            }

            for _ in range(edge_count):
                edge_id = read_varint(f)
                edge_xp = bytes_to_mpz(read_varbytes(f))
                edge_ac = bytes_to_mpz(read_varbytes(f))
                traj_count = read_varint(f)

                edge_obj = {
                    "edge_xp": bigint_to_str(edge_xp, as_hex),
                    "edge_ac": bigint_to_str(edge_ac, as_hex),
                }

                for _ in range(traj_count):
                    traj_id = read_varint(f)
                    occ_count = read_varint(f)

                    detail_tr = []
                    for _ in range(occ_count):
                        occ = f.read(5)
                        start_second, end_second = unpack_occurrence(occ)
                        member_witness = None
                        if version >= 2:
                            witness_bytes = f.read(modulus_bytes)
                            if len(witness_bytes) != modulus_bytes:
                                raise EOFError("member witness 数据被截断")
                            member_witness = bytes_to_mpz(witness_bytes)

                        start_code = to_base_fixed(start_second, base, DEPTH)
                        end_code = to_base_fixed(end_second, base, DEPTH)
                        tr_time = tr_time_seed(edge_id, traj_id, start_code, end_code)

                        detail = {
                            "start_second": start_second,
                            "end_second": end_second,
                            "tr_time": tr_time,
                        }
                        if member_witness is not None:
                            detail["member_witness"] = bigint_to_str(member_witness, as_hex)

                        if rebuild_query_pair:
                            detail["query_pair"] = rebuild_query_pair_from_occ(edge_id, traj_id, start_second, end_second)

                        detail_tr.append(detail)

                    edge_obj[str(traj_id)] = {
                        "edge_tr": edge_tr_seed(edge_id, traj_id),
                        "detail_tr": detail_tr,
                    }

                date_obj[str(edge_id)] = edge_obj

            result["data"][date_str] = date_obj

        return result


def read_binary_summary(path: str, max_dates=2, max_edges=2, max_traj=2):
    with open(path, "rb") as f:
        magic = f.read(4)
        version = read_u16(f)
        base = read_u8(f)
        _reserved = read_u8(f)
        day_seconds = read_u32(f)
        tz_offset = read_i32(f)
        hprime_bits = read_u16(f)
        _reserved2 = read_u16(f)

        n = bytes_to_mpz(read_varbytes(f))
        g = bytes_to_mpz(read_varbytes(f))
        modulus_bytes = (n.bit_length() + 7) // 8
        occurrence_record_bytes = 5 + (modulus_bytes if version >= 2 else 0)
        date_count = read_varint(f)

        print("========== 二进制摘要 ==========")
        print(f"magic={magic!r}")
        print(f"version={version}")
        print(f"base={base}")
        print(f"day_seconds={day_seconds}")
        print(f"tz_offset_seconds={tz_offset}")
        print(f"hprime_bits={hprime_bits}")
        print(f"n_bits={n.bit_length()}")
        print(f"g_bits={g.bit_length()}")
        print(f"date_count={date_count}")

        for di in range(min(date_count, max_dates)):
            day_idx = read_varint(f)
            data_xp = bytes_to_mpz(read_varbytes(f))
            data_ac = bytes_to_mpz(read_varbytes(f))
            edge_count = read_varint(f)

            day_str = (_EPOCH_DATE + timedelta(days=day_idx)).isoformat()
            print(f"\n[date {di}] day_idx={day_idx} date={day_str} edge_count={edge_count}")
            print(f"  data_xp_bits={data_xp.bit_length()}")
            print(f"  data_ac_bits={data_ac.bit_length()}")

            for ei in range(min(edge_count, max_edges)):
                edge_id = read_varint(f)
                edge_xp = bytes_to_mpz(read_varbytes(f))
                edge_ac = bytes_to_mpz(read_varbytes(f))
                traj_count = read_varint(f)

                print(f"  [edge {ei}] edge_id={edge_id} traj_count={traj_count}")
                print(f"    edge_xp_bits={edge_xp.bit_length()}")
                print(f"    edge_ac_bits={edge_ac.bit_length()}")

                for ti in range(min(traj_count, max_traj)):
                    traj_id = read_varint(f)
                    occ_count = read_varint(f)
                    print(f"    [traj {ti}] traj_id={traj_id} occ_count={occ_count}")
                    for oi in range(min(occ_count, 2)):
                        occ = f.read(5)
                        s, e = unpack_occurrence(occ)
                        if version >= 2:
                            witness = f.read(modulus_bytes)
                            if len(witness) != modulus_bytes:
                                raise EOFError("member witness 数据被截断")
                        print(
                            f"      occ[{oi}] start_second={s}, end_second={e}, "
                            f"member_witness={'yes' if version >= 2 else 'no'}"
                        )
                    remain = occ_count - min(occ_count, 2)
                    if remain > 0:
                        f.seek(remain * occurrence_record_bytes, 1)

                remain_traj = traj_count - min(traj_count, max_traj)
                for _ in range(remain_traj):
                    _traj_id = read_varint(f)
                    occ_count = read_varint(f)
                    f.seek(occ_count * occurrence_record_bytes, 1)

            remain_edge = edge_count - min(edge_count, max_edges)
            for _ in range(remain_edge):
                _edge_id = read_varint(f)
                _ = read_varbytes(f)
                _ = read_varbytes(f)
                traj_count = read_varint(f)
                for _ in range(traj_count):
                    _traj_id = read_varint(f)
                    occ_count = read_varint(f)
                    f.seek(occ_count * occurrence_record_bytes, 1)


# =========================
# 主流程
# =========================
def main():
    print("========== 一步构建前缀 + RSA 累加器（二进制 + 可完整加载版）==========")
    print(f"TRAJ_CSV_FILES={TRAJ_CSV_FILES}")
    print(f"OUTPUT_BIN={OUTPUT_BIN}")
    print(f"TIMEZONE={TIMEZONE}")
    print(f"BINARY_VERSION={BINARY_VERSION}")
    print(f"PRECOMPUTE_MEMBER_WITNESSES={PRECOMPUTE_MEMBER_WITNESSES}")
    if BINARY_VERSION >= 3:
        if not isinstance(OPAQUE_TOKEN_KEY, (bytes, bytearray)) or len(OPAQUE_TOKEN_KEY) < 32:
            raise ValueError("v3 需要至少 32 字节的数据拥有者 OPAQUE_TOKEN_KEY")
        if not (16 <= int(OPAQUE_TOKEN_BYTES) <= 32):
            raise ValueError("OPAQUE_TOKEN_BYTES 必须在 16..32")
        print(f"OPAQUE_TOKEN_BYTES={OPAQUE_TOKEN_BYTES}")
    if PRECOMPUTE_MEMBER_WITNESSES and not FUSE_PREPARE_AND_ACCUMULATOR:
        raise ValueError("预计算 member witness 需要 FUSE_PREPARE_AND_ACCUMULATOR=True")
    print(f"BASE={BASE}")
    print(f"DEPTH={DEPTH}")
    print(f"HPRIME_BITS={HPRIME_BITS}")
    print(f"RSA_P_BITS={RSA_P_BITS}")
    print(f"RSA_Q_BITS={RSA_Q_BITS}")
    print(f"NUM_WORKERS={NUM_WORKERS}")
    print(f"CHUNK_SIZE={CHUNK_SIZE}")
    print(f"INCLUDE_EMPTY_EDGES={INCLUDE_EMPTY_EDGES}")

    t0 = time.perf_counter()
    p, q, n, g = ensure_rsa_params()
    print(f"p bits={p.bit_length()}")
    print(f"q bits={q.bit_length()}")
    print(f"n bits={n.bit_length()}")
    print(f"g bits={g.bit_length()}")

    data = {}
    total_stats = Counter()

    t1 = time.perf_counter()
    for traj_csv in TRAJ_CSV_FILES:
        stats = process_trajectory_file(traj_csv, data)
        total_stats.update(stats)
    t2 = time.perf_counter()

    print("\n========== 轨迹处理统计 ==========")
    for k, v in total_stats.items():
        print(f"{k}: {v:,}")

    data_results = {}
    p_bytes = mpz_to_bytes(p)
    q_bytes = mpz_to_bytes(q)
    n_bytes = mpz_to_bytes(n)
    g_bytes = mpz_to_bytes(g)

    edge_prime_prepare_wall_time = 0.0
    pure_edge_xp_ac_wall_time = 0.0

    for day_idx, date_bucket in data.items():
        day_str = (_EPOCH_DATE + timedelta(days=day_idx)).isoformat()
        if PRINT_TIMING:
            print(f"\n处理日期: {day_str}")

        tasks = []
        for edge_id, traj_map in date_bucket.items():
            tasks.append(
                (day_idx, edge_id, traj_map)
                if BINARY_VERSION >= 3
                else (edge_id, traj_map)
            )

        if INCLUDE_EMPTY_EDGES:
            present = set(int(eid) for eid in date_bucket.keys())
            for edge_id in load_all_edge_ids(EDGE_CSV):
                if edge_id not in present:
                    tasks.append(
                        (day_idx, edge_id, {})
                        if BINARY_VERSION >= 3
                        else (edge_id, {})
                    )

        with mp.Pool(
                processes=NUM_WORKERS,
                initializer=worker_init,
                initargs=(
                    p_bytes,
                    q_bytes,
                    n_bytes,
                    g_bytes,
                    bytes(OPAQUE_TOKEN_KEY or b""),
                    BINARY_VERSION,
                    OPAQUE_TOKEN_BYTES,
                    DAY_SECONDS,
                ),
        ) as pool:
            if FUSE_PREPARE_AND_ACCUMULATOR:
                fused_t0 = time.perf_counter()
                edge_results = []
                completed_edges = 0
                for result in pool.imap_unordered(
                    compute_edge_fused_task, tasks, chunksize=FUSED_CHUNK_SIZE
                ):
                    edge_results.append(result)
                    completed_edges += 1
                    if PRINT_TIMING and completed_edges % 1000 == 0:
                        print(f"  [融合构建进度] edge={completed_edges:,}/{len(tasks):,}")
                fused_t1 = time.perf_counter()
                day_edge_prime_prepare_wall_time = 0.0
                day_pure_edge_xp_ac_wall_time = fused_t1 - fused_t0
                pure_edge_xp_ac_wall_time += day_pure_edge_xp_ac_wall_time
            else:
                prep_t0 = time.perf_counter()
                prepared_results = []
                for result in pool.imap_unordered(compute_edge_prepare_task, tasks, chunksize=CHUNK_SIZE):
                    prepared_results.append(result)
                prep_t1 = time.perf_counter()
                day_edge_prime_prepare_wall_time = prep_t1 - prep_t0
                edge_prime_prepare_wall_time += day_edge_prime_prepare_wall_time

                arith_tasks = [(item['edge_id'], item['factor_list']) for item in prepared_results]
                arith_t0 = time.perf_counter()
                acc_results = []
                for result in pool.imap_unordered(compute_edge_accumulator_task, arith_tasks, chunksize=CHUNK_SIZE):
                    acc_results.append(result)
                arith_t1 = time.perf_counter()
                day_pure_edge_xp_ac_wall_time = arith_t1 - arith_t0
                pure_edge_xp_ac_wall_time += day_pure_edge_xp_ac_wall_time

                acc_by_edge = {item['edge_id']: item for item in acc_results}
                edge_results = []
                for prepared in prepared_results:
                    acc = acc_by_edge[prepared['edge_id']]
                    edge_results.append({
                        'edge_id': prepared['edge_id'],
                        'edge_xp_bytes': acc['edge_xp_bytes'],
                        'edge_ac_bytes': acc['edge_ac_bytes'],
                        'traj_rows': prepared['traj_rows'],
                        'identity_witness_bytes': None,
                        'traj_count': prepared['traj_count'],
                        'tr_time_unique_count': prepared['tr_time_unique_count'],
                        'pair_unique_count': prepared['pair_unique_count'],
                        'xp_bits': acc['xp_bits'],
                    })

        t3 = time.perf_counter()
        data_prime_list = []
        data_prime_by_edge = {}
        main_prime_cache = {}
        for edge_obj in edge_results:
            edge_ac = bytes_to_mpz(edge_obj["edge_ac_bytes"])
            seed = edge_ac_seed(
                str(edge_obj["edge_id"]),
                edge_ac,
                day_idx=day_idx,
                traj_count=edge_obj["traj_count"],
            )
            descriptor_prime = hash_to_prime_local(seed, main_prime_cache)
            data_prime_list.append(descriptor_prime)
            data_prime_by_edge[int(edge_obj["edge_id"])] = descriptor_prime

        data_xp = product_tree(data_prime_list)
        t4 = time.perf_counter()
        data_ac = powmod(g, data_xp, n)
        t5 = time.perf_counter()

        if BINARY_VERSION >= 3:
            # 数据拥有者离线生成每条边到日根的静态成员 witness。客户端只需
            # data_ac 和这个 witness，不需要读取其它边或完整 ADS。
            witness_tasks = [
                (int(edge_obj["edge_id"]), data_prime_by_edge[int(edge_obj["edge_id"])])
                for edge_obj in edge_results
            ]
            edge_result_by_id = {
                int(edge_obj["edge_id"]): edge_obj for edge_obj in edge_results
            }
            completed_witnesses = 0
            with mp.Pool(
                    processes=NUM_WORKERS,
                    initializer=root_witness_worker_init,
                    initargs=(
                        p_bytes,
                        q_bytes,
                        n_bytes,
                        g_bytes,
                        bytes(OPAQUE_TOKEN_KEY),
                        BINARY_VERSION,
                        OPAQUE_TOKEN_BYTES,
                        mpz_to_bytes(data_xp),
                    ),
            ) as witness_pool:
                for edge_id, witness_bytes in witness_pool.imap_unordered(
                    compute_root_witness_task, witness_tasks, chunksize=32
                ):
                    edge_result_by_id[edge_id]["root_witness_bytes"] = witness_bytes
                    completed_witnesses += 1
                    if PRINT_TIMING and completed_witnesses % 10000 == 0:
                        print(
                            f"  [根 witness 进度] "
                            f"edge={completed_witnesses:,}/{len(witness_tasks):,}"
                        )

        if PRINT_TIMING:
            print(f"  当日 prime准备墙钟时间={day_edge_prime_prepare_wall_time:.4f}s")
            print(f"  当日你要的 edge_xp+edge_ac 总墙钟时间={day_pure_edge_xp_ac_wall_time:.4f}s")
            print(
                f"  data_xp_bits={data_xp.bit_length()}, "
                f"data_mul_time={t4 - t3:.4f}s, "
                f"data_ac_time={t5 - t4:.4f}s"
            )

        data_results[day_idx] = {
            "edges": edge_results,
            "data_xp_bytes": mpz_to_bytes(data_xp),
            "data_ac_bytes": mpz_to_bytes(data_ac),
        }

    t6 = time.perf_counter()
    write_binary(OUTPUT_BIN, n, g, data_results)
    t7 = time.perf_counter()

    if CHECK_AFTER_WRITE:
        read_binary_summary(OUTPUT_BIN, max_dates=2, max_edges=2, max_traj=2)
        restored = load_binary_to_dict(OUTPUT_BIN, as_hex=LOAD_BIGINT_AS_HEX_STR, rebuild_query_pair=LOAD_REBUILD_QUERY_PAIR)
        print("\n========== 完整加载检查 ==========")
        print(f"restored_date_count={len(restored['data'])}")
        total_edge = 0
        total_traj = 0
        total_occ = 0
        for date_obj in restored["data"].values():
            for k, edge_obj in date_obj.items():
                if k in ("data_xp", "data_ac"):
                    continue
                total_edge += 1
                for tk, traj_obj in edge_obj.items():
                    if tk in ("edge_xp", "edge_ac"):
                        continue
                    total_traj += 1
                    total_occ += len(traj_obj["detail_tr"])
        print(f"restored_edge_count={total_edge}")
        print(f"restored_traj_count={total_traj}")
        print(f"restored_occ_count={total_occ}")

    print("\n========== 耗时 ==========")
    print(f"trajectory_process_time={t2 - t1:.4f}s")
    print(f"edge_prime_prepare_wall_time={edge_prime_prepare_wall_time:.4f}s")
    print(f"pure_edge_xp_ac_wall_time={pure_edge_xp_ac_wall_time:.4f}s")
    print(f"说明: 这是所有prime已准备完成后，仅计算各edge的edge_xp与edge_ac的真实墙钟时间（数据拥有者CRT+指数约减优化）")
    print(f"accumulator_build_time_all_in={t6 - t2:.4f}s")
    print(f"write_bin_time={t7 - t6:.4f}s")
    print(f"total_time={t7 - t0:.4f}s")
    print(f"已保存输出文件: {OUTPUT_BIN}")

    # Expose machine-readable timings to experiment drivers.  The primary
    # construction metric deliberately starts after trajectory loading (t2)
    # and ends before binary serialization (t6).
    timings = {
        "trajectory_process_time": t2 - t1,
        "edge_prime_prepare_wall_time": edge_prime_prepare_wall_time,
        "pure_edge_xp_ac_wall_time": pure_edge_xp_ac_wall_time,
        "accumulator_build_time_all_in": t6 - t2,
        "write_bin_time": t7 - t6,
        "total_time": t7 - t0,
    }
    globals()["LAST_BUILD_TIMINGS"] = timings
    return timings


if __name__ == "__main__":
    mp.freeze_support()
    main()
