from .canonical import cell_value
from .config import DEFAULT_DOMAIN_SIZE, SLOTS_PER_DAY
from .models import SegmentRecord

def base_evaluations(
    day: str,
    domain_size: int = DEFAULT_DOMAIN_SIZE,
    slot_count: int = SLOTS_PER_DAY,
) -> list[int]:
    """构造所有位置均为空记录的基础向量。"""
    if domain_size < slot_count:
        raise ValueError("domain size is smaller than the number of time slots")
    if domain_size & (domain_size - 1) != 0:
        raise ValueError("domain size must be a power of two")
    return [cell_value(day, 0, position, []) for position in range(domain_size)]

def edge_evaluations(
    day: str,
    edge_id: int,
    slots: dict[int, list[SegmentRecord]],
    domain_size: int = DEFAULT_DOMAIN_SIZE,
    slot_count: int = SLOTS_PER_DAY,
) -> list[int]:
    """构造某条道路边的完整求值向量。"""
    for slot in slots:
        if not 0 <= slot < slot_count:
            raise ValueError(f"invalid time slot: {slot}")
    evaluations = base_evaluations(day, domain_size, slot_count)
    for slot, records in slots.items():
        evaluations[slot] = cell_value(day, edge_id, slot, records)
    return evaluations
