import struct
from functools import lru_cache

from .field import FR_MODULUS, fr_to_bytes

try:
    from kzg._native import (
        inverse_ntt_packed as _native_inverse_ntt_packed,
        ntt as _native_ntt,
    )
except (ImportError, AttributeError):
    _native_inverse_ntt_packed = None
    _native_ntt = None


NATIVE_NTT = _native_ntt is not None and _native_inverse_ntt_packed is not None
def is_power_of_two(value: int) -> bool:
    """判断一个整数是否为2的幂。"""
    return value > 0 and value & (value - 1) == 0

@lru_cache(maxsize=None)
def primitive_root_of_unity(size: int) -> int:
    """寻找指定阶数的单位根。"""
    if not is_power_of_two(size):
        raise ValueError("NTT size must be a power of two")
    if (FR_MODULUS - 1) % size != 0:
        raise ValueError(f"field does not support a {size}-th root of unity")
    exponent = (FR_MODULUS - 1) // size
    candidate = 2
    while True:
        root = pow(candidate, exponent, FR_MODULUS)
        if pow(root, size, FR_MODULUS) == 1 and (size == 1 or pow(root, size // 2, FR_MODULUS) != 1):
            return root
        candidate += 1


def _ntt_python(values: list[int], inverse: bool = False) -> list[int]:
    """执行正向NTT或逆NTT。"""
    size = len(values)
    if not is_power_of_two(size):
        raise ValueError("NTT length must be a power of two")
    result = [int(value) % FR_MODULUS for value in values]
    reversed_index = 0
    for current_index in range(1, size):
        bit = size >> 1
        while reversed_index & bit:
            reversed_index ^= bit
            bit >>= 1
        reversed_index ^= bit
        if current_index < reversed_index:
            result[current_index], result[reversed_index] = result[reversed_index], result[current_index]
    root = primitive_root_of_unity(size)
    if inverse:
        root = pow(root, FR_MODULUS - 2, FR_MODULUS)
    current_length = 2
    while current_length <= size:
        step_root = pow(root, size // current_length, FR_MODULUS)
        half_length = current_length // 2
        for block_start in range(0, size, current_length):
            current_root = 1
            for offset in range(half_length):
                left_value = result[block_start + offset]
                right_value = result[block_start + offset + half_length] * current_root % FR_MODULUS
                result[block_start + offset] = (left_value + right_value) % FR_MODULUS
                result[block_start + offset + half_length] = (left_value - right_value) % FR_MODULUS
                current_root = current_root * step_root % FR_MODULUS
        current_length <<= 1
    if inverse:
        inverse_size = pow(size, FR_MODULUS - 2, FR_MODULUS)
        result = [value * inverse_size % FR_MODULUS for value in result]
    return result


@lru_cache(maxsize=None)
def _stage_roots(size: int, inverse: bool) -> list[int]:
    root = primitive_root_of_unity(size)
    if inverse:
        root = pow(root, FR_MODULUS - 2, FR_MODULUS)
    roots = []
    current_length = 2
    while current_length <= size:
        roots.append(pow(root, size // current_length, FR_MODULUS))
        current_length <<= 1
    return roots


def ntt(values: list[int], inverse: bool = False) -> list[int]:
    """执行正向NTT或逆NTT，优先使用原生有限域实现。"""
    size = len(values)
    if not is_power_of_two(size):
        raise ValueError("NTT length must be a power of two")
    if NATIVE_NTT:
        scale = pow(size, FR_MODULUS - 2, FR_MODULUS) if inverse else 1
        try:
            return _native_ntt(values, _stage_roots(size, inverse), scale)
        except (TypeError, ValueError):
            # Preserve the reference implementation's normalization behavior
            # for unusual callers that provide negative or non-canonical ints.
            pass
    return _ntt_python(values, inverse)


def evaluations_to_packed_coefficients(evaluations: list[int]) -> bytes:
    """把求值逆NTT后直接编码为DailyStore的系数二进制格式。"""
    size = len(evaluations)
    if not is_power_of_two(size):
        raise ValueError("NTT length must be a power of two")
    if NATIVE_NTT:
        scale = pow(size, FR_MODULUS - 2, FR_MODULUS)
        try:
            return _native_inverse_ntt_packed(
                evaluations, _stage_roots(size, True), scale
            )
        except (TypeError, ValueError):
            pass
    coefficients = _ntt_python(evaluations, inverse=True)
    return struct.pack(">I", len(coefficients)) + b"".join(
        fr_to_bytes(value) for value in coefficients
    )


def group_ntt(points: list,inverse: bool = False) -> list:
    """在支持加法和标量乘法的素数阶群上执行NTT。"""
    size = len(points)

    if not is_power_of_two(size):
        raise ValueError("group NTT length must be a power of two")

    if not points:
        raise ValueError("group NTT requires at least one point")

    result = list(points)
    reversed_index = 0

    for current_index in range(1,size):
        bit = size >> 1

        while reversed_index & bit:
            reversed_index ^= bit
            bit >>= 1

        reversed_index ^= bit

        if current_index < reversed_index:
            result[current_index],result[reversed_index] = result[reversed_index],result[current_index]

    root = primitive_root_of_unity(size)

    if inverse:
        root = pow(root,FR_MODULUS - 2,FR_MODULUS)

    current_length = 2

    while current_length <= size:
        step_root = pow(root,size // current_length,FR_MODULUS)
        half_length = current_length // 2

        for block_start in range(0,size,current_length):
            current_root = 1

            for offset in range(half_length):
                left_point = result[block_start + offset]
                right_point = result[block_start + offset + half_length].scalar_mul(current_root)
                result[block_start + offset] = left_point + right_point
                result[block_start + offset + half_length] = left_point + right_point.scalar_mul(-1)
                current_root = current_root * step_root % FR_MODULUS

        current_length <<= 1

    if inverse:
        inverse_size = pow(size,FR_MODULUS - 2,FR_MODULUS)
        result = [point.scalar_mul(inverse_size) for point in result]

    return result


def evaluations_to_coefficients(evaluations: list[int]) -> list[int]:
    """把单位根上的求值转换为多项式系数。"""
    return ntt(evaluations, inverse=True)


def coefficients_to_evaluations(coefficients: list[int]) -> list[int]:
    """把多项式系数转换为单位根上的求值。"""
    return ntt(coefficients, inverse=False)


@lru_cache(maxsize=None)
def domain_points(size: int) -> tuple[int, ...]:
    """返回NTT域中的全部单位根查询点。"""
    root = primitive_root_of_unity(size)
    points = [1]
    for _ in range(1, size):
        points.append(points[-1] * root % FR_MODULUS)
    return tuple(points)
