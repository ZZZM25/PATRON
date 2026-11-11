"""Select a standard binary-Geohash depth for a target local cell size."""

from dataclasses import dataclass
from math import cos, log, pi, sqrt


METERS_PER_DEGREE_LAT = 110_574.0
METERS_PER_DEGREE_LON_EQUATOR = 111_320.0


@dataclass(frozen=True)
class PrecisionSelection:
    bits: int
    longitude_bits: int
    latitude_bits: int
    width_m: float
    height_m: float
    geometric_mean_m: float


def cell_dimensions_m(bits: int, reference_latitude: float) -> tuple[float, float]:
    """Return approximate longitude and latitude cell sizes in metres."""

    if bits < 1:
        raise ValueError("Geohash bit depth must be positive")
    longitude_bits = (bits + 1) // 2
    latitude_bits = bits // 2
    width_deg = 360.0 / (2**longitude_bits)
    height_deg = 180.0 / (2**latitude_bits)
    width_m = width_deg * METERS_PER_DEGREE_LON_EQUATOR * cos(
        reference_latitude * pi / 180.0
    )
    height_m = height_deg * METERS_PER_DEGREE_LAT
    return width_m, height_m


def choose_geohash_bits(
    target_cell_size_m: float,
    reference_latitude: float,
    min_bits: int = 4,
    max_bits: int = 60,
) -> PrecisionSelection:
    """Choose the bit depth whose geometric-mean size is closest to target.

    Standard Geohash cells are rectangles rather than perfect squares.  The
    geometric mean is used as the target size so that the selected resolution
    is stable and meaningful for both dimensions.
    """

    if target_cell_size_m <= 0:
        raise ValueError("target_cell_size_m must be positive")
    if not (1 <= min_bits <= max_bits <= 60):
        raise ValueError("precision range must satisfy 1 <= min <= max <= 60")

    best: tuple[float, PrecisionSelection] | None = None
    target_log = log(target_cell_size_m)
    for bits in range(min_bits, max_bits + 1):
        width_m, height_m = cell_dimensions_m(bits, reference_latitude)
        geometric_mean_m = sqrt(width_m * height_m)
        # Primary objective: geometric mean close to target.  A tiny tie-break
        # favours the less elongated rectangle.
        score = abs(log(geometric_mean_m) - target_log) + 1e-6 * abs(
            log(width_m / height_m)
        )
        selection = PrecisionSelection(
            bits=bits,
            longitude_bits=(bits + 1) // 2,
            latitude_bits=bits // 2,
            width_m=width_m,
            height_m=height_m,
            geometric_mean_m=geometric_mean_m,
        )
        if best is None or score < best[0]:
            best = (score, selection)
    assert best is not None
    return best[1]
