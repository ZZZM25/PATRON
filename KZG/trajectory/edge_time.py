"""Time extraction and integer timestamp allocation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class TimeInterval:
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


def interval_from_gps_group(group: list[tuple[int, float, float]]) -> TimeInterval:
    if not group:
        raise ValueError("a GPS group cannot be empty")
    start = int(group[0][0])
    end = int(group[-1][0])
    if end < start:
        raise ValueError(f"GPS group has negative duration: {start}->{end}")
    return TimeInterval(start, end)


def round_fraction_half_up(value: Fraction) -> int:
    """Round a non-negative exact fraction to the nearest integer."""

    if value.denominator == 1:
        return value.numerator
    if value < 0:
        # Unix timestamps in this project are positive, but keeping the helper
        # mathematically sensible costs little.
        return -round_fraction_half_up(-value)
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + (1 if 2 * remainder >= value.denominator else 0)


def allocate_boundary_times(
    interval: TimeInterval, fractions: list[Fraction]
) -> list[int]:
    """Allocate timestamps at cumulative distance fractions.

    The first and last values remain exactly the original GPS timestamps;
    internal values are rounded to integer Unix seconds and clamped to preserve
    monotonicity when several short pieces share one second.
    """

    if not fractions or fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("fractions must start at 0 and end at 1")
    if any(a > b for a, b in zip(fractions, fractions[1:])):
        raise ValueError("fractions must be nondecreasing")
    duration = interval.duration
    values: list[int] = []
    previous = interval.start
    for index, fraction in enumerate(fractions):
        if index == 0:
            value = interval.start
        elif index == len(fractions) - 1:
            value = interval.end
        else:
            exact = Fraction(interval.start) + Fraction(duration) * fraction
            value = round_fraction_half_up(exact)
            value = max(previous, min(interval.end, value))
        values.append(value)
        previous = value
    return values
