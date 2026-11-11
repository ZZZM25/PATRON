from __future__ import annotations

import math
import struct

from kzg.field import FR_MODULUS,fr_to_bytes,hash_to_field
from kzg.kzg import msm_g1

try:
    from kzg._native import aggregate_polynomials as _native_aggregate_polynomials
except ImportError:
    _native_aggregate_polynomials = None


BATCH_PROOF_MODE = "edge-batch-kzg-v2"
DEFAULT_EDGE_BATCH_SIZE = 256
AUTO_EDGE_BATCH_SIZE = 0
NATIVE_POLYNOMIAL_AGGREGATION = _native_aggregate_polynomials is not None
_TRANSCRIPT_MAGIC = b"KZG-TRAJECTORY-EDGE-BATCH-V2"


def resolve_edge_batch_size(edge_count: int,batch_size: int) -> int:
    """Resolve ``0`` to one spatial batch containing every queried edge."""
    normalized_edge_count = int(edge_count)
    normalized_batch_size = int(batch_size)
    if normalized_edge_count < 0:
        raise ValueError("edge count cannot be negative")
    if normalized_batch_size == AUTO_EDGE_BATCH_SIZE:
        return max(1,normalized_edge_count)
    if normalized_batch_size < 0:
        raise ValueError("edge batch size cannot be negative")
    return normalized_batch_size


def partition_edge_ids(edge_ids: list[int],batch_size: int = DEFAULT_EDGE_BATCH_SIZE) -> list[list[int]]:
    """将排序且唯一的道路ID确定性地划分为多个批次。"""
    normalized = [int(edge_id) for edge_id in edge_ids]

    batch_size = resolve_edge_batch_size(len(normalized),batch_size)

    if normalized != sorted(set(normalized)):
        raise ValueError("edge IDs must be sorted and unique")

    return [normalized[index:index + batch_size] for index in range(0,len(normalized),batch_size)]


def canonical_batch_transcript(date: str,daily_root: bytes,srs_fingerprint: bytes,domain_size: int,rect_values: list[float],start_ts: int,end_ts: int,batch_index: int,slots: list[int],edge_ids: list[int],commitments: list[bytes],values_by_edge: list[list[int]]) -> bytes:
    """规范编码一个批次的查询、承诺和声明值。"""
    if len(daily_root) != 32:
        raise ValueError("daily root must contain 32 bytes")

    if len(srs_fingerprint) != 32:
        raise ValueError("SRS fingerprint must contain 32 bytes")

    if domain_size <= 0:
        raise ValueError("domain size must be positive")

    if len(rect_values) != 4 or not all(math.isfinite(float(value)) for value in rect_values):
        raise ValueError("query rectangle must contain four finite coordinates")

    if batch_index < 0:
        raise ValueError("batch index cannot be negative")

    normalized_slots = [int(slot) for slot in slots]
    normalized_edge_ids = [int(edge_id) for edge_id in edge_ids]

    if normalized_slots != sorted(set(normalized_slots)):
        raise ValueError("query slots must be sorted and unique")

    if normalized_edge_ids != sorted(set(normalized_edge_ids)):
        raise ValueError("batch edge IDs must be sorted and unique")

    if len(normalized_edge_ids) != len(commitments) or len(normalized_edge_ids) != len(values_by_edge):
        raise ValueError("batch edges, commitments and value rows must have the same length")

    date_bytes = str(date).encode("utf-8")

    if len(date_bytes) > 65535:
        raise ValueError("date encoding is too long")

    encoded = bytearray()
    encoded.extend(_TRANSCRIPT_MAGIC)
    encoded.extend(struct.pack(">H",len(date_bytes)))
    encoded.extend(date_bytes)
    encoded.extend(daily_root)
    encoded.extend(srs_fingerprint)
    encoded.extend(struct.pack(">I",int(domain_size)))
    encoded.extend(struct.pack(">dddd",*(float(value) for value in rect_values)))
    encoded.extend(struct.pack(">qqI",int(start_ts),int(end_ts),int(batch_index)))
    encoded.extend(struct.pack(">I",len(normalized_slots)))

    for slot in normalized_slots:
        if slot < 0:
            raise ValueError("query slot cannot be negative")

        encoded.extend(struct.pack(">I",slot))

    encoded.extend(struct.pack(">I",len(normalized_edge_ids)))

    for edge_id,commitment,values in zip(normalized_edge_ids,commitments,values_by_edge):
        if edge_id < 0:
            raise ValueError("edge ID cannot be negative")

        if len(commitment) != 48:
            raise ValueError("edge commitment must contain 48 bytes")

        if len(values) != len(normalized_slots):
            raise ValueError("each edge must contain one value for every query slot")

        encoded.extend(struct.pack(">Q",edge_id))
        encoded.extend(commitment)

        for value in values:
            encoded.extend(fr_to_bytes(value))

    return bytes(encoded)


def batch_challenge(date: str,daily_root: bytes,srs_fingerprint: bytes,domain_size: int,rect_values: list[float],start_ts: int,end_ts: int,batch_index: int,slots: list[int],edge_ids: list[int],commitments: list[bytes],values_by_edge: list[list[int]]) -> int:
    """从完整批次transcript确定性地产生非零Fiat-Shamir挑战。"""
    transcript = canonical_batch_transcript(date,daily_root,srs_fingerprint,domain_size,rect_values,start_ts,end_ts,batch_index,slots,edge_ids,commitments,values_by_edge)
    counter = 0

    while True:
        challenge = hash_to_field(transcript + struct.pack(">I",counter))

        if challenge != 0:
            return challenge

        counter += 1


def batch_weights(challenge: int,count: int) -> list[int]:
    """返回1、r、r²等批次随机线性组合权重。"""
    if count < 0:
        raise ValueError("weight count cannot be negative")

    normalized_challenge = int(challenge) % FR_MODULUS

    if normalized_challenge == 0:
        raise ValueError("batch challenge cannot be zero")

    weights = []
    current = 1

    for _ in range(count):
        weights.append(current)
        current = current * normalized_challenge % FR_MODULUS

    return weights


def _aggregate_polynomials_python(polynomials: list[list[int]],weights: list[int]) -> list[int]:
    """高速线性聚合道路多项式。

    热路径优化：每个输出系数只做一次模约简，而不是对每条边的每个
    ``multiply-add`` 都执行一次 ``% FR_MODULUS``。典型 256x2048 batch
    因此把约 524k 次模运算降到约 2048 次。
    """
    if not polynomials:
        raise ValueError("at least one polynomial is required")
    if len(polynomials) != len(weights):
        raise ValueError("polynomials and weights must have the same length")

    maximum_length = max(len(polynomial) for polynomial in polynomials)
    if maximum_length == 0:
        raise ValueError("polynomials cannot be empty")

    normalized_weights = [int(weight) % FR_MODULUS for weight in weights]
    result = [0] * maximum_length
    # Accumulate with Python big integers and reduce once per coefficient.  With
    # the configured batch sizes the temporary integer growth is modest and this
    # is substantially cheaper than a modular reduction in the inner loop.
    for polynomial, weight in zip(polynomials, normalized_weights):
        local_result = result
        for index, value in enumerate(polynomial):
            local_result[index] += weight * int(value)

    modulus = FR_MODULUS
    for index, value in enumerate(result):
        result[index] = value % modulus
    return result


def aggregate_polynomials(polynomials: list[list[int]],weights: list[int]) -> list[int]:
    """优先使用原生有限域聚合，不可用或输入非规范时回退 Python。"""
    if _native_aggregate_polynomials is not None:
        try:
            return _native_aggregate_polynomials(polynomials, weights)
        except (TypeError, ValueError):
            pass
    return _aggregate_polynomials_python(polynomials, weights)


def aggregate_values(values_by_edge: list[list[int]],weights: list[int]) -> list[int]:
    """高速聚合多条道路在相同查询点上的值（每个输出只取模一次）。"""
    if not values_by_edge:
        raise ValueError("at least one edge value row is required")
    if len(values_by_edge) != len(weights):
        raise ValueError("value rows and weights must have the same length")

    value_count = len(values_by_edge[0])
    if value_count == 0:
        raise ValueError("edge value rows cannot be empty")
    if any(len(values) != value_count for values in values_by_edge):
        raise ValueError("all edge value rows must have the same length")

    if _native_aggregate_polynomials is not None:
        try:
            return _native_aggregate_polynomials(values_by_edge,weights)
        except (TypeError,ValueError):
            pass

    normalized_weights = [int(weight) % FR_MODULUS for weight in weights]
    result = [0] * value_count
    for values, weight in zip(values_by_edge, normalized_weights):
        for index, value in enumerate(values):
            result[index] += weight * int(value)
    modulus = FR_MODULUS
    for index, value in enumerate(result):
        result[index] = value % modulus
    return result


def aggregate_commitments(commitments,weights: list[int],prepared_points=None):
    """利用KZG线性性质聚合多个G1道路承诺。"""
    if not commitments:
        raise ValueError("at least one commitment is required")

    if len(commitments) != len(weights):
        raise ValueError("commitments and weights must have the same length")

    return msm_g1(commitments,weights,prepared_points)

GEOHASH_BATCH_PROOF_MODE = "geohash-edge-batch-kzg-v3"
_GEOHASH_TRANSCRIPT_MAGIC = b"KZG-GEOHASH-EDGE-BATCH-V3"


def _canonical_geohash_batch_transcript(
    date: str,
    daily_root: bytes,
    srs_fingerprint: bytes,
    domain_size: int,
    cell_rect_values: list[int],
    start_ts: int,
    end_ts: int,
    batch_index: int,
    slots: list[int],
    edge_ids: list[int],
    cell_prefixes: list[str],
    commitments: list[bytes],
    value_rows,
    *,
    packed_rows: bool,
) -> bytes:
    """Encode the shared Geohash transcript protocol for integer or packed rows."""
    if len(daily_root) != 32 or len(srs_fingerprint) != 32:
        raise ValueError("daily root and SRS fingerprint must contain 32 bytes")
    if domain_size <= 0:
        raise ValueError("domain size must be positive")
    if len(cell_rect_values) != 4:
        raise ValueError("cell rectangle must contain four indices")
    normalized_slots = [int(slot) for slot in slots]
    normalized_edge_ids = [int(edge_id) for edge_id in edge_ids]
    if normalized_slots != sorted(set(normalized_slots)):
        raise ValueError("query slots must be sorted and unique")
    if normalized_edge_ids != sorted(set(normalized_edge_ids)):
        raise ValueError("batch edge IDs must be sorted and unique")
    count = len(normalized_edge_ids)
    if not (count == len(cell_prefixes) == len(commitments) == len(value_rows)):
        raise ValueError("batch edges, cell prefixes, commitments and values must align")
    date_bytes = str(date).encode("ascii")
    encoded = bytearray(_GEOHASH_TRANSCRIPT_MAGIC)
    encoded.extend(struct.pack(">H", len(date_bytes)))
    encoded.extend(date_bytes)
    encoded.extend(daily_root)
    encoded.extend(srs_fingerprint)
    encoded.extend(struct.pack(">I", int(domain_size)))
    encoded.extend(struct.pack(">qqqq", *(int(value) for value in cell_rect_values)))
    encoded.extend(struct.pack(">qqI", int(start_ts), int(end_ts), int(batch_index)))
    encoded.extend(struct.pack(">I", len(normalized_slots)))
    for slot in normalized_slots:
        if slot < 0:
            raise ValueError("query slot cannot be negative")
        encoded.extend(struct.pack(">I", slot))
    encoded.extend(struct.pack(">I", count))
    encoded_value_cache: dict[int,bytes] = {}
    packed_row_length = len(normalized_slots) * 32
    for edge_id, cell_prefix, commitment, value_row in zip(
        normalized_edge_ids, cell_prefixes, commitments, value_rows
    ):
        prefix_bytes = str(cell_prefix).encode("ascii")
        if not prefix_bytes or any(byte not in (48, 49) for byte in prefix_bytes):
            raise ValueError("cell prefix must be a non-empty binary string")
        if len(prefix_bytes) > 255:
            raise ValueError("cell prefix too long")
        if len(commitment) != 48:
            raise ValueError("edge commitment must contain 48 bytes")
        packed_view = None
        if packed_rows:
            try:
                packed_view = memoryview(value_row).cast("B")
            except (TypeError,ValueError) as exc:
                raise ValueError("each packed edge value row must be bytes-like") from exc
            if packed_view.nbytes != packed_row_length:
                raise ValueError(
                    "each packed edge value row must contain exactly "
                    f"{packed_row_length} bytes"
                )
        elif len(value_row) != len(normalized_slots):
            raise ValueError("each edge needs one value for every query slot")
        encoded.extend(struct.pack(">Q", edge_id))
        encoded.extend(struct.pack(">B", len(prefix_bytes)))
        encoded.extend(prefix_bytes)
        encoded.extend(commitment)
        if packed_view is not None:
            encoded.extend(packed_view)
        else:
            for value in value_row:
                normalized_value = int(value)
                if normalized_value < 0 or normalized_value >= FR_MODULUS:
                    normalized_value %= FR_MODULUS
                encoded_value = encoded_value_cache.get(normalized_value)
                if encoded_value is None:
                    encoded_value = fr_to_bytes(normalized_value)
                    encoded_value_cache[normalized_value] = encoded_value
                encoded.extend(encoded_value)
    return bytes(encoded)


def canonical_geohash_batch_transcript(
    date: str,
    daily_root: bytes,
    srs_fingerprint: bytes,
    domain_size: int,
    cell_rect_values: list[int],
    start_ts: int,
    end_ts: int,
    batch_index: int,
    slots: list[int],
    edge_ids: list[int],
    cell_prefixes: list[str],
    commitments: list[bytes],
    values_by_edge: list[list[int]],
) -> bytes:
    """Bind a cross-cell KZG batch to its authenticated Geohash membership."""
    return _canonical_geohash_batch_transcript(
        date,
        daily_root,
        srs_fingerprint,
        domain_size,
        cell_rect_values,
        start_ts,
        end_ts,
        batch_index,
        slots,
        edge_ids,
        cell_prefixes,
        commitments,
        values_by_edge,
        packed_rows=False,
    )


def canonical_geohash_batch_transcript_packed(
    date: str,
    daily_root: bytes,
    srs_fingerprint: bytes,
    domain_size: int,
    cell_rect_values: list[int],
    start_ts: int,
    end_ts: int,
    batch_index: int,
    slots: list[int],
    edge_ids: list[int],
    cell_prefixes: list[str],
    commitments: list[bytes],
    packed_values_by_edge: list[bytes],
) -> bytes:
    """Encode prepacked canonical field rows using the unchanged V3 protocol."""
    return _canonical_geohash_batch_transcript(
        date,
        daily_root,
        srs_fingerprint,
        domain_size,
        cell_rect_values,
        start_ts,
        end_ts,
        batch_index,
        slots,
        edge_ids,
        cell_prefixes,
        commitments,
        packed_values_by_edge,
        packed_rows=True,
    )


def _geohash_challenge_from_transcript(transcript: bytes) -> int:
    counter = 0
    while True:
        challenge = hash_to_field(transcript + struct.pack(">I", counter))
        if challenge != 0:
            return challenge
        counter += 1


def geohash_batch_challenge(
    date: str,
    daily_root: bytes,
    srs_fingerprint: bytes,
    domain_size: int,
    cell_rect_values: list[int],
    start_ts: int,
    end_ts: int,
    batch_index: int,
    slots: list[int],
    edge_ids: list[int],
    cell_prefixes: list[str],
    commitments: list[bytes],
    values_by_edge: list[list[int]],
) -> int:
    transcript = canonical_geohash_batch_transcript(
        date,
        daily_root,
        srs_fingerprint,
        domain_size,
        cell_rect_values,
        start_ts,
        end_ts,
        batch_index,
        slots,
        edge_ids,
        cell_prefixes,
        commitments,
        values_by_edge,
    )
    return _geohash_challenge_from_transcript(transcript)


def geohash_batch_challenge_packed(
    date: str,
    daily_root: bytes,
    srs_fingerprint: bytes,
    domain_size: int,
    cell_rect_values: list[int],
    start_ts: int,
    end_ts: int,
    batch_index: int,
    slots: list[int],
    edge_ids: list[int],
    cell_prefixes: list[str],
    commitments: list[bytes],
    packed_values_by_edge: list[bytes],
) -> int:
    transcript = canonical_geohash_batch_transcript_packed(
        date,
        daily_root,
        srs_fingerprint,
        domain_size,
        cell_rect_values,
        start_ts,
        end_ts,
        batch_index,
        slots,
        edge_ids,
        cell_prefixes,
        commitments,
        packed_values_by_edge,
    )
    return _geohash_challenge_from_transcript(transcript)
