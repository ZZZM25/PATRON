"""KZG SRS、椭圆曲线点和多项式承诺。"""
import hashlib
import struct
from pathlib import Path
import secrets
from dataclasses import dataclass
from .polynomial import divmod_poly, interpolate, quotient_by_roots, sub, trim, vanishing
from pyblst import BlstP1Element, BlstP2Element, final_verify, miller_loop
from py_ecc.bls.point_compression import compress_G1 as ecc_compress_G1
from py_ecc.bls.point_compression import compress_G2 as ecc_compress_G2
from py_ecc.optimized_bls12_381 import G1 as ECC_G1
from py_ecc.optimized_bls12_381 import G2 as ECC_G2

from .field import FR_MODULUS
from .ntt import group_ntt
from .polynomial import trim
_SRS_MAGIC = b"KZGSRS01"

try:
    from kzg._native import (
        msm_g1_prepared as _native_msm_g1_prepared,
        prepare_g1_points as _native_prepare_g1_points,
    )
except ImportError:
    _native_msm_g1_prepared = None
    _native_prepare_g1_points = None

NATIVE_G1_MSM = (
    _native_msm_g1_prepared is not None and _native_prepare_g1_points is not None
)

_G1_COMPRESSED = int(ecc_compress_G1(ECC_G1)).to_bytes(48, "big")
_G2_FIRST, _G2_SECOND = ecc_compress_G2(ECC_G2)
_G2_COMPRESSED = int(_G2_FIRST).to_bytes(48, "big") + int(_G2_SECOND).to_bytes(48, "big")

G1_GENERATOR = BlstP1Element.uncompress(_G1_COMPRESSED)
G2_GENERATOR = BlstP2Element.uncompress(_G2_COMPRESSED)


def g1_to_bytes(point: BlstP1Element) -> bytes:
    """把G1点压缩为48字节。"""
    return bytes(point.compress())


def g1_from_bytes(data: bytes) -> BlstP1Element:
    """从48字节恢复G1点。"""
    if len(data) != 48:
        raise ValueError("compressed G1 point must contain 48 bytes")

    return BlstP1Element.uncompress(data)


def prepare_g1_points(points: list[BlstP1Element]):
    """Prepare a fixed G1 point set once for repeated native MSM operations."""
    if not NATIVE_G1_MSM or not points:
        return None
    try:
        return _native_prepare_g1_points(
            b"".join(g1_to_bytes(point) for point in points)
        )
    except (TypeError,ValueError):
        return None


def g2_to_bytes(point: BlstP2Element) -> bytes:
    """把G2点压缩为96字节。"""
    return bytes(point.compress())


def g2_from_bytes(data: bytes) -> BlstP2Element:
    """从96字节恢复G2点。"""
    if len(data) != 96:
        raise ValueError("compressed G2 point must contain 96 bytes")

    return BlstP2Element.uncompress(data)


# def msm_g1(points: list[BlstP1Element], scalars: list[int]) -> BlstP1Element:
#     """执行G1上的简单多标量乘法。"""
#     if len(points) != len(scalars):
#         raise ValueError("points and scalars must have the same length")
#
#     if not points:
#         raise ValueError("MSM requires at least one point")
#
#     result = points[0].scalar_mul(0)
#
#     for point, scalar in zip(points, scalars):
#         normalized_scalar = int(scalar) % FR_MODULUS
#
#         if normalized_scalar != 0:
#             result = result + point.scalar_mul(normalized_scalar)
#
#     return result
#
#
# def msm_g2(points: list[BlstP2Element], scalars: list[int]) -> BlstP2Element:
#     """执行G2上的简单多标量乘法。"""
#     if len(points) != len(scalars):
#         raise ValueError("points and scalars must have the same length")
#
#     if not points:
#         raise ValueError("MSM requires at least one point")
#
#     result = points[0].scalar_mul(0)
#
#     for point, scalar in zip(points, scalars):
#         normalized_scalar = int(scalar) % FR_MODULUS
#
#         if normalized_scalar != 0:
#             result = result + point.scalar_mul(normalized_scalar)
#
#     return result

def _choose_msm_window(active_count: int) -> int | None:
    """根据非零标量数量选择MSM算法和窗口大小。"""
    if active_count <= 96:
        return None

    if active_count <= 192:
        return 5

    if active_count <= 768:
        return 7

    if active_count <= 4096:
        return 8

    return 10


def _naive_msm(points,scalars,zero):
    """对较小输入执行逐项MSM。"""
    result = zero

    for point,scalar in zip(points,scalars):
        if scalar != 0:
            result = result + point.scalar_mul(scalar)

    return result


def _pippenger_msm(points,scalars,zero,window_size: int):
    """使用分桶Pippenger算法执行大规模MSM。"""
    maximum_bits = max((scalar.bit_length() for scalar in scalars),default=0)

    if maximum_bits == 0:
        return zero

    window_count = (maximum_bits + window_size - 1) // window_size
    bucket_count = 1 << window_size
    window_mask = bucket_count - 1
    result = zero

    for window_index in range(window_count - 1,-1,-1):
        if window_index != window_count - 1:
            for _ in range(window_size):
                result = result + result

        buckets = [zero for _ in range(bucket_count)]
        shift = window_index * window_size

        for point,scalar in zip(points,scalars):
            digit = (scalar >> shift) & window_mask

            if digit != 0:
                buckets[digit] = buckets[digit] + point

        running_sum = zero

        for digit in range(window_mask,0,-1):
            running_sum = running_sum + buckets[digit]
            result = result + running_sum

    return result


def _msm(points,scalars):
    """根据输入规模自动选择逐项MSM或Pippenger MSM。"""
    if len(points) != len(scalars):
        raise ValueError("points and scalars must have the same length")

    if not points:
        raise ValueError("MSM requires at least one point")

    zero = points[0].scalar_mul(0)
    normalized_scalars = [int(scalar) % FR_MODULUS for scalar in scalars]
    active_pairs = [(point,scalar) for point,scalar in zip(points,normalized_scalars) if scalar != 0]

    if not active_pairs:
        return zero

    active_points = [point for point,scalar in active_pairs]
    active_scalars = [scalar for point,scalar in active_pairs]
    window_size = _choose_msm_window(len(active_pairs))

    if window_size is None:
        return _naive_msm(active_points,active_scalars,zero)

    return _pippenger_msm(active_points,active_scalars,zero,window_size)


def msm_g1(
    points: list[BlstP1Element],
    scalars: list[int],
    prepared_points=None,
) -> BlstP1Element:
    """对已准备的固定点集使用BLST原生Pippenger，否则使用Python实现。"""
    if len(points) != len(scalars):
        raise ValueError("points and scalars must have the same length")
    if not points:
        raise ValueError("MSM requires at least one point")

    normalized_scalars = [int(scalar) % FR_MODULUS for scalar in scalars]
    if NATIVE_G1_MSM and prepared_points is not None:
        try:
            encoded_result = _native_msm_g1_prepared(
                prepared_points, normalized_scalars
            )
            return g1_from_bytes(encoded_result)
        except (TypeError, ValueError):
            pass
    return _msm(points, normalized_scalars)


def msm_g2(points: list[BlstP2Element],scalars: list[int]) -> BlstP2Element:
    """在G2上执行自动优化的多标量乘法。"""
    return _msm(points,scalars)

@dataclass
class SRS:
    """KZG结构化参考字符串。"""

    domain_size: int
    max_multi_points: int
    g1_powers: list[BlstP1Element]
    g2_powers: list[BlstP2Element]

    @classmethod
    def generate(cls, domain_size: int, max_multi_points: int) -> "SRS":
        """生成论文仿真使用的单方测试SRS。"""
        if domain_size <= 1:
            raise ValueError("domain size must be greater than one")

        if max_multi_points <= 0 or max_multi_points >= domain_size:
            raise ValueError("max multi points must be between one and domain size minus one")

        tau = secrets.randbelow(FR_MODULUS - 1) + 1
        g1_powers = []
        g2_powers = []
        current_power = 1

        for _ in range(domain_size):
            g1_powers.append(G1_GENERATOR.scalar_mul(current_power))
            current_power = current_power * tau % FR_MODULUS

        current_power = 1

        for _ in range(max_multi_points + 1):
            g2_powers.append(G2_GENERATOR.scalar_mul(current_power))
            current_power = current_power * tau % FR_MODULUS

        return cls(domain_size, max_multi_points, g1_powers, g2_powers)

    @property
    def g2(self) -> BlstP2Element:
        """返回G2生成元。"""
        return self.g2_powers[0]

    def save(self, path: str | Path) -> None:
        """把SRS保存为二进制文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("wb") as file:
            file.write(_SRS_MAGIC)
            file.write(struct.pack(">II", self.domain_size, self.max_multi_points))

            for point in self.g1_powers:
                file.write(g1_to_bytes(point))

            for point in self.g2_powers:
                file.write(g2_to_bytes(point))

    @classmethod
    def load(cls, path: str | Path) -> "SRS":
        """从二进制文件加载SRS。"""
        raw = Path(path).read_bytes()
        header_size = len(_SRS_MAGIC) + 8

        if len(raw) < header_size:
            raise ValueError("SRS file is truncated")

        if raw[:len(_SRS_MAGIC)] != _SRS_MAGIC:
            raise ValueError("invalid SRS file header")

        offset = len(_SRS_MAGIC)
        domain_size, max_multi_points = struct.unpack(">II", raw[offset:offset + 8])
        offset += 8

        expected_size = offset + domain_size * 48 + (max_multi_points + 1) * 96

        if len(raw) != expected_size:
            raise ValueError("invalid SRS file length")

        g1_end = offset + domain_size * 48
        g2_end = g1_end + (max_multi_points + 1) * 96
        g1_powers = [g1_from_bytes(raw[index:index + 48]) for index in range(offset, g1_end, 48)]
        g2_powers = [g2_from_bytes(raw[index:index + 96]) for index in range(g1_end, g2_end, 96)]

        return cls(domain_size, max_multi_points, g1_powers, g2_powers)

    @staticmethod
    def fingerprint(path: str | Path) -> str:
        """计算SRS文件的SHA-256指纹。"""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lagrange_g1_powers(srs: SRS) -> list[BlstP1Element]:
    """把单项式G1 SRS转换成同一求值域上的拉格朗日G1 SRS。"""
    return group_ntt(srs.g1_powers,inverse=True)


def commit_sparse_evaluations(base_commitment: BlstP1Element,base_evaluations: list[int],changed_values: dict[int,int],lagrange_powers: list[BlstP1Element]) -> BlstP1Element:
    """从公共基础向量出发，只对发生变化的求值位置更新承诺。"""
    if len(base_evaluations) != len(lagrange_powers):
        raise ValueError("base evaluations and Lagrange powers must have the same length")

    points = []
    scalars = []

    for position,value in sorted(changed_values.items()):
        normalized_position = int(position)

        if not 0 <= normalized_position < len(base_evaluations):
            raise ValueError(f"evaluation position is outside the domain: {position}")

        delta = (int(value) - int(base_evaluations[normalized_position])) % FR_MODULUS

        if delta != 0:
            points.append(lagrange_powers[normalized_position])
            scalars.append(delta)

    if not points:
        return base_commitment

    return base_commitment + msm_g1(points,scalars)

@dataclass(frozen=True)
class MultiOpening:
    """KZG多点证明结果。"""

    proof: BlstP1Element
    interpolation: list[int]
    vanishing: list[int]



class KZG:
    """KZG多项式承诺对象。"""

    def __init__(self, srs: SRS):
        self.srs = srs
        self._native_g1_points = prepare_g1_points(self.srs.g1_powers)

    def commit_g1(self, coefficients: list[int]) -> BlstP1Element:
        """在G1上承诺一个多项式。"""
        normalized = trim(coefficients)

        if len(normalized) > len(self.srs.g1_powers):
            raise ValueError("polynomial degree exceeds the G1 SRS")

        points = self.srs.g1_powers[:len(normalized)]
        return msm_g1(points, normalized, self._native_g1_points)

    def commit_g2(self, coefficients: list[int]) -> BlstP2Element:
        """在G2上承诺一个多项式。"""
        normalized = trim(coefficients)

        if len(normalized) > len(self.srs.g2_powers):
            raise ValueError("polynomial degree exceeds the G2 SRS")

        return msm_g2(self.srs.g2_powers[:len(normalized)], normalized)

    def open_multi(self, coefficients: list[int], points: list[int], claimed_values: list[int]) -> MultiOpening:
        """为多个查询点生成一个KZG多点证明。"""
        if not points:
            raise ValueError("multi-point opening requires at least one point")

        if len(points) != len(claimed_values):
            raise ValueError("points and claimed values must have the same length")

        if len(points) > self.srs.max_multi_points:
            raise ValueError("number of opening points exceeds the SRS limit")

        normalized_points = [int(point) % FR_MODULUS for point in points]

        if len(set(normalized_points)) != len(normalized_points):
            raise ValueError("opening points must be distinct")

        interpolation_polynomial = interpolate(normalized_points, claimed_values)
        vanishing_polynomial = vanishing(normalized_points)
        numerator = sub(coefficients, interpolation_polynomial)
        quotient, remainder = divmod_poly(numerator, vanishing_polynomial)

        if trim(remainder) != [0]:
            raise ValueError("claimed values do not match the polynomial")

        proof = self.commit_g1(quotient)

        return MultiOpening(proof, interpolation_polynomial, vanishing_polynomial)

    def open_multi_fast(self, coefficients: list[int], points: list[int], claimed_values: list[int]) -> MultiOpening:
        """Optimized server-side multi-opening with the same proof semantics.

        The KZG witness quotient is the quotient of ``P`` by the vanishing
        polynomial ``Z=prod(X-x_i)``.  For honest claimed values, the remainder
        is exactly their interpolation polynomial.  The server therefore does
        not need to construct that interpolation polynomial merely to obtain the
        quotient.  We compute the quotient directly by synthetic division and
        keep the legacy ``open_multi`` implementation for compatibility/tests.
        """
        if not points:
            raise ValueError("multi-point opening requires at least one point")
        if len(points) != len(claimed_values):
            raise ValueError("points and claimed values must have the same length")
        if len(points) > self.srs.max_multi_points:
            raise ValueError("number of opening points exceeds the SRS limit")
        normalized_points = [int(point) % FR_MODULUS for point in points]
        if len(set(normalized_points)) != len(normalized_points):
            raise ValueError("opening points must be distinct")

        quotient = quotient_by_roots(coefficients, normalized_points)
        proof = self.commit_g1(quotient)
        # These two polynomials are not needed by the query-server fast path.
        # They are returned empty to preserve the MultiOpening shape.
        return MultiOpening(proof, [], [])

    def verify_multi(self, commitment: BlstP1Element, points: list[int], values: list[int], proof: BlstP1Element) -> bool:
        """使用双线性配对验证KZG多点证明。"""
        try:
            if not points or len(points) != len(values):
                return False

            if len(points) > self.srs.max_multi_points:
                return False

            normalized_points = [int(point) % FR_MODULUS for point in points]

            if len(set(normalized_points)) != len(normalized_points):
                return False

            interpolation_polynomial = interpolate(normalized_points, values)
            vanishing_polynomial = vanishing(normalized_points)
            interpolation_commitment = self.commit_g1(interpolation_polynomial)
            vanishing_commitment = self.commit_g2(vanishing_polynomial)
            left_g1 = commitment + -interpolation_commitment
            left_pairing = miller_loop(left_g1, self.srs.g2)
            right_pairing = miller_loop(proof, vanishing_commitment)

            return bool(final_verify(left_pairing, right_pairing))
        except Exception:
            return False
