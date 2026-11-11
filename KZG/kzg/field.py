import hashlib
from py_ecc.optimized_bls12_381 import curve_order

FR_MODULUS = curve_order
FR_BYTES = 32

def fr_to_bytes(value:int) -> bytes:
    """把整数转换成规范的32字节有限域元素。"""
    normalized = int(value)
    if normalized < 0 or normalized >= FR_MODULUS:
        normalized %= FR_MODULUS
    return normalized.to_bytes(FR_BYTES, byteorder="big")

def fr_from_bytes(value: bytes) -> int:
    """从规范的32字节编码恢复有限域元素。"""
    if len(value) != FR_BYTES:
        raise ValueError(f"field element must contain {FR_BYTES} bytes")
    result = int.from_bytes(value, byteorder="big",)
    if result >= FR_MODULUS:
        raise ValueError("non-canonical field element")
    return result


def hash_to_field(message: bytes) -> int:
    """使用SHA-256将任意消息映射到BLS12-381标量域。"""
    two_to_256 = 1 << 256
    acceptance_limit = (two_to_256 - two_to_256 % FR_MODULUS)
    counter = 0
    while True:
        digest = hashlib.sha256(counter.to_bytes(4, "big") + message).digest()
        candidate = int.from_bytes(digest,byteorder="big")
        if candidate < acceptance_limit:
            return candidate % FR_MODULUS
        counter += 1

if __name__ == "__main__":

    # 测试hash_to_field函数
    test_message = b"Hello, world!"
    field_element = hash_to_field(test_message)
    print(field_element)
    assert 0 <= field_element < FR_MODULUS, "field element out of range"
