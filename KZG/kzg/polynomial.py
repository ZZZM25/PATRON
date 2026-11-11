from functools import lru_cache

from .field import FR_MODULUS
def trim(poly: list[int]) -> list[int]:
    """删除多项式末尾无意义的零系数。"""
    result = [int(value) % FR_MODULUS for value in poly]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]

def add(first: list[int], second: list[int]) -> list[int]:
    """有限域多项式加法。"""
    size = max(len(first), len(second))
    result = [0] * size
    for index in range(size):
        first_value = first[index] if index < len(first) else 0
        second_value = second[index] if index < len(second) else 0
        result[index] = (first_value + second_value) % FR_MODULUS
    return trim(result)


def sub(first: list[int], second: list[int]) -> list[int]:
    """有限域多项式减法。"""
    size = max(len(first), len(second))
    result = [0] * size
    for index in range(size):
        first_value = first[index] if index < len(first) else 0
        second_value = second[index] if index < len(second) else 0
        result[index] = (first_value - second_value) % FR_MODULUS
    return trim(result)

def scale(poly: list[int], scalar: int) -> list[int]:
    """多项式的每个系数同时乘以一个有限域元素。"""
    normalized_scalar = scalar % FR_MODULUS
    return trim([(coefficient * normalized_scalar) % FR_MODULUS for coefficient in poly])

def mul(first: list[int], second: list[int]) -> list[int]:
    """有限域多项式乘法。"""
    if not first or not second:
        return [0]
    result = [0] * (len(first) + len(second) - 1)
    for first_index, first_value in enumerate(first):
        for second_index, second_value in enumerate(second):
            result_index = first_index + second_index
            result[result_index] = (result[result_index] + first_value * second_value) % FR_MODULUS
    return trim(result)

def evaluate(poly: list[int], point: int) -> int:
    """使用 Horner 方法计算多项式在某一点的值。"""
    normalized_point = point % FR_MODULUS
    result = 0
    for coefficient in reversed(poly):
        result = (result * normalized_point + coefficient) % FR_MODULUS
    return result

def divmod_poly(numerator: list[int], denominator: list[int]) -> tuple[list[int], list[int]]:
    """使用原地倒序长除法返回有限域多项式的商和余数。"""
    normalized_numerator = trim(numerator)
    normalized_denominator = trim(denominator)

    if normalized_denominator == [0]:
        raise ZeroDivisionError("polynomial division by zero")

    if len(normalized_numerator) < len(normalized_denominator):
        return [0], normalized_numerator

    quotient = [0] * (len(normalized_numerator) - len(normalized_denominator) + 1)
    remainder = normalized_numerator[:]
    denominator_degree = len(normalized_denominator) - 1
    leading_coefficient = normalized_denominator[-1]
    inverse_leading = 1 if leading_coefficient == 1 else pow(leading_coefficient,FR_MODULUS - 2,FR_MODULUS)

    for current_degree in range(len(normalized_numerator) - 1,denominator_degree - 1,-1):
        leading_value = remainder[current_degree]

        if leading_value == 0:
            continue

        shift = current_degree - denominator_degree
        factor = leading_value * inverse_leading % FR_MODULUS
        quotient[shift] = factor

        for index,coefficient in enumerate(normalized_denominator[:-1]):
            remainder[index + shift] = (remainder[index + shift] - factor * coefficient) % FR_MODULUS

        remainder[current_degree] = 0

    remainder_limit = denominator_degree
    return trim(quotient),trim(remainder[:remainder_limit])


def quotient_by_roots(poly: list[int], roots: list[int]) -> list[int]:
    """Return ``poly / prod(X-root)`` with deferred field reductions.

    The divisor is monic, so each quotient coefficient only needs one reduction.
    Updates to the lower remainder coefficients use Python integers and are reduced
    when they become a later quotient coefficient.  This avoids a field reduction
    for every multiply-add while preserving the exact quotient used by the KZG
    witness.  The remainder is intentionally discarded; the verifier rejects
    inconsistent claimed values.
    """
    if not roots:
        return trim(poly)

    normalized_roots = [int(root) % FR_MODULUS for root in roots]
    current = trim([int(value) % FR_MODULUS for value in poly])
    divisor_degree = len(normalized_roots)
    if len(current) <= divisor_degree:
        return [0]

    divisor = vanishing(normalized_roots)
    lower_divisor = divisor[:-1]
    quotient = [0] * (len(current) - divisor_degree)

    for shift in range(len(quotient) - 1, -1, -1):
        factor = current[shift + divisor_degree] % FR_MODULUS
        quotient[shift] = factor
        if factor != 0:
            for index, coefficient in enumerate(lower_divisor):
                current[shift + index] -= factor * coefficient

    return trim(quotient)


@lru_cache(maxsize=256)
def _cached_vanishing(normalized_points: tuple[int,...]) -> tuple[int,...]:
    """缓存固定查询点集合对应的消失多项式。"""
    result = [1]

    for point in normalized_points:
        next_result = [0] * (len(result) + 1)

        for index,coefficient in enumerate(result):
            next_result[index] = (next_result[index] - point * coefficient) % FR_MODULUS
            next_result[index + 1] = (next_result[index + 1] + coefficient) % FR_MODULUS

        result = next_result

    return tuple(result)


def _divide_by_linear(poly: tuple[int,...],root: int) -> list[int]:
    """对已知含因子(X-root)的首一多项式执行综合除法。"""
    quotient = [0] * (len(poly) - 1)
    quotient[-1] = int(poly[-1]) % FR_MODULUS

    for index in range(len(quotient) - 2,-1,-1):
        quotient[index] = (int(poly[index + 1]) + root * quotient[index + 1]) % FR_MODULUS

    remainder = (int(poly[0]) + root * quotient[0]) % FR_MODULUS

    if remainder != 0:
        raise ValueError("polynomial is not divisible by the specified linear factor")

    return quotient


def _batch_inverse(values: list[int]) -> list[int]:
    """使用一次有限域求逆同时计算多个非零元素的逆。"""
    if not values:
        return []

    prefixes = [1] * len(values)
    product = 1

    for index,value in enumerate(values):
        normalized_value = int(value) % FR_MODULUS

        if normalized_value == 0:
            raise ZeroDivisionError("cannot invert zero")

        prefixes[index] = product
        product = product * normalized_value % FR_MODULUS

    inverse_product = pow(product,FR_MODULUS - 2,FR_MODULUS)
    inverses = [0] * len(values)

    for index in range(len(values) - 1,-1,-1):
        normalized_value = int(values[index]) % FR_MODULUS
        inverses[index] = inverse_product * prefixes[index] % FR_MODULUS
        inverse_product = inverse_product * normalized_value % FR_MODULUS

    return inverses


@lru_cache(maxsize=256)
def _cached_lagrange_basis(normalized_points: tuple[int,...]) -> tuple[tuple[int,...],...]:
    """缓存固定查询点集合的全部拉格朗日基多项式。"""
    if not normalized_points:
        raise ValueError("at least one interpolation point is required")

    if len(set(normalized_points)) != len(normalized_points):
        raise ValueError("interpolation points must be distinct")

    vanishing_polynomial = _cached_vanishing(normalized_points)
    numerators = [_divide_by_linear(vanishing_polynomial,point) for point in normalized_points]
    denominators = [evaluate(numerator,point) for numerator,point in zip(numerators,normalized_points)]
    inverse_denominators = _batch_inverse(denominators)
    bases = []

    for numerator,inverse_denominator in zip(numerators,inverse_denominators):
        bases.append(tuple(coefficient * inverse_denominator % FR_MODULUS for coefficient in numerator))

    return tuple(bases)

def vanishing(points: list[int]) -> list[int]:
    """构造并缓存全部指定点上的消失多项式。"""
    normalized_points = tuple(int(point) % FR_MODULUS for point in points)
    return list(_cached_vanishing(normalized_points))

def interpolate(points: list[int], values: list[int]) -> list[int]:
    """使用缓存的拉格朗日基构造经过全部查询点的多项式。"""
    if len(points) != len(values):
        raise ValueError("points and values must have the same length")

    if not points:
        raise ValueError("at least one interpolation point is required")

    normalized_points = [int(point) % FR_MODULUS for point in points]

    if len(set(normalized_points)) != len(normalized_points):
        raise ValueError("interpolation points must be distinct")

    bases = _cached_lagrange_basis(tuple(normalized_points))
    result = [0] * len(normalized_points)

    for value,basis in zip(values,bases):
        normalized_value = int(value) % FR_MODULUS

        for index,coefficient in enumerate(basis):
            result[index] = (result[index] + normalized_value * coefficient) % FR_MODULUS

    return trim(result)
