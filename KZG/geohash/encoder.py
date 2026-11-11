"""Integer/fixed-point binary Geohash grid operations.

The grid uses the standard Geohash convention: longitude and latitude bits
are interleaved, starting with longitude.  The implementation works in
scaled integer degrees (normally 1e-7 degree input precision), and uses
``fractions.Fraction`` for boundary parameters.  Therefore a line crossing a
cell boundary is split deterministically without floating-point decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor
from typing import Iterable

from .precision_selector import PrecisionSelection, choose_geohash_bits


@dataclass(frozen=True)
class CellBounds:
    cell_id: str
    lon_index: int
    lat_index: int
    lon_min: Fraction
    lon_max: Fraction
    lat_min: Fraction
    lat_max: Fraction


class GeohashGrid:
    """A fixed-depth binary-Geohash partition of the whole world."""

    WORLD_LON_MIN = -180
    WORLD_LON_MAX = 180
    WORLD_LAT_MIN = -90
    WORLD_LAT_MAX = 90

    def __init__(
        self,
        bits: int,
        reference_latitude: float,
        coordinate_scale: int = 10_000_000,
        selection: PrecisionSelection | None = None,
    ) -> None:
        if bits < 1 or bits > 60:
            raise ValueError("bits must be between 1 and 60")
        self.bits = bits
        self.reference_latitude = reference_latitude
        self.coordinate_scale = coordinate_scale
        self.longitude_bits = (bits + 1) // 2
        self.latitude_bits = bits // 2
        self.longitude_bins = 1 << self.longitude_bits
        self.latitude_bins = 1 << self.latitude_bits
        self.lon_min_scaled = self.WORLD_LON_MIN * coordinate_scale
        self.lon_max_scaled = self.WORLD_LON_MAX * coordinate_scale
        self.lat_min_scaled = self.WORLD_LAT_MIN * coordinate_scale
        self.lat_max_scaled = self.WORLD_LAT_MAX * coordinate_scale
        self.lon_step = Fraction(
            self.lon_max_scaled - self.lon_min_scaled, self.longitude_bins
        )
        self.lat_step = Fraction(
            self.lat_max_scaled - self.lat_min_scaled, self.latitude_bins
        )
        self.selection = selection

    @classmethod
    def auto(
        cls,
        target_cell_size_m: float,
        reference_latitude: float,
        coordinate_scale: int = 10_000_000,
    ) -> "GeohashGrid":
        selection = choose_geohash_bits(target_cell_size_m, reference_latitude)
        return cls(
            bits=selection.bits,
            reference_latitude=reference_latitude,
            coordinate_scale=coordinate_scale,
            selection=selection,
        )

    @classmethod
    def fixed(
        cls,
        bits: int,
        reference_latitude: float,
        coordinate_scale: int = 10_000_000,
    ) -> "GeohashGrid":
        return cls(
            bits=bits,
            reference_latitude=reference_latitude,
            coordinate_scale=coordinate_scale,
        )

    @staticmethod
    def _as_fraction(value: int | float | Fraction) -> Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        return Fraction(str(value))

    def _clamp_index(self, value: int, count: int) -> int:
        return max(0, min(count - 1, value))

    def indices_for_scaled_point(
        self, lon_scaled: int | Fraction, lat_scaled: int | Fraction
    ) -> tuple[int, int]:
        """Return the leaf indices containing a fixed-point point.

        The upper world boundary is assigned to the final cell.  Other
        boundaries use the usual half-open convention.  Split segments use a
        midpoint, so this convention gives every output edge exactly one cell
        even when a segment lies on a grid line.
        """

        lon = self._as_fraction(lon_scaled)
        lat = self._as_fraction(lat_scaled)
        lon_index = floor((lon - self.lon_min_scaled) / self.lon_step)
        lat_index = floor((lat - self.lat_min_scaled) / self.lat_step)
        return (
            self._clamp_index(lon_index, self.longitude_bins),
            self._clamp_index(lat_index, self.latitude_bins),
        )

    def prefix_from_indices(self, lon_index: int, lat_index: int) -> str:
        """Interleave longitude and latitude index bits into a prefix."""

        lon_index = self._clamp_index(lon_index, self.longitude_bins)
        lat_index = self._clamp_index(lat_index, self.latitude_bins)
        out: list[str] = []
        for bit_position in range(max(self.longitude_bits, self.latitude_bits)):
            if bit_position < self.longitude_bits:
                shift = self.longitude_bits - 1 - bit_position
                out.append(str((lon_index >> shift) & 1))
            if bit_position < self.latitude_bits:
                shift = self.latitude_bits - 1 - bit_position
                out.append(str((lat_index >> shift) & 1))
        return "".join(out)

    def cell_id_for_scaled_point(
        self, lon_scaled: int | Fraction, lat_scaled: int | Fraction
    ) -> str:
        lon_index, lat_index = self.indices_for_scaled_point(lon_scaled, lat_scaled)
        return self.prefix_from_indices(lon_index, lat_index)

    def bounds_from_indices(self, lon_index: int, lat_index: int) -> CellBounds:
        cell_id = self.prefix_from_indices(lon_index, lat_index)
        lon_min = self.lon_min_scaled + self.lon_step * lon_index
        lon_max = lon_min + self.lon_step
        lat_min = self.lat_min_scaled + self.lat_step * lat_index
        lat_max = lat_min + self.lat_step
        return CellBounds(
            cell_id=cell_id,
            lon_index=lon_index,
            lat_index=lat_index,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
        )

    def bounds_for_cell(self, cell_id: str) -> CellBounds:
        if len(cell_id) != self.bits or any(c not in "01" for c in cell_id):
            raise ValueError(f"invalid {self.bits}-bit cell id: {cell_id!r}")
        lon_index = 0
        lat_index = 0
        position = 0
        for bit_position in range(max(self.longitude_bits, self.latitude_bits)):
            if bit_position < self.longitude_bits:
                lon_index = (lon_index << 1) | int(cell_id[position])
                position += 1
            if bit_position < self.latitude_bits:
                lat_index = (lat_index << 1) | int(cell_id[position])
                position += 1
        return self.bounds_from_indices(lon_index, lat_index)

    def boundary_parameters(
        self,
        lon1_scaled: int,
        lat1_scaled: int,
        lon2_scaled: int,
        lat2_scaled: int,
    ) -> list[Fraction]:
        """Return all interior t values where a segment crosses a grid line."""

        x1 = Fraction(lon1_scaled)
        y1 = Fraction(lat1_scaled)
        x2 = Fraction(lon2_scaled)
        y2 = Fraction(lat2_scaled)
        parameters: set[Fraction] = set()

        def add_axis(
            first: Fraction,
            second: Fraction,
            world_min: int,
            step: Fraction,
            bin_count: int,
        ) -> None:
            delta = second - first
            if delta == 0:
                return
            lo = min(first, second)
            hi = max(first, second)
            # Interior grid lines only.  A line coinciding with an endpoint is
            # harmless but is deliberately filtered by the 0<t<1 test.
            start_index = max(
                1, floor((lo - world_min) / step) + 1
            )
            end_index = min(bin_count - 1, floor((hi - world_min) / step))
            for index in range(start_index, end_index + 1):
                boundary = world_min + step * index
                t = (boundary - first) / delta
                if 0 < t < 1:
                    parameters.add(t)

        add_axis(
            x1,
            x2,
            self.lon_min_scaled,
            self.lon_step,
            self.longitude_bins,
        )
        add_axis(
            y1,
            y2,
            self.lat_min_scaled,
            self.lat_step,
            self.latitude_bins,
        )
        return sorted(parameters)

    @staticmethod
    def point_on_segment(
        lon1_scaled: int,
        lat1_scaled: int,
        lon2_scaled: int,
        lat2_scaled: int,
        t: Fraction,
    ) -> tuple[Fraction, Fraction]:
        return (
            Fraction(lon1_scaled) + (lon2_scaled - lon1_scaled) * t,
            Fraction(lat1_scaled) + (lat2_scaled - lat1_scaled) * t,
        )

    def cell_for_segment_midpoint(
        self,
        lon1_scaled: int,
        lat1_scaled: int,
        lon2_scaled: int,
        lat2_scaled: int,
        t_start: Fraction,
        t_end: Fraction,
    ) -> str:
        midpoint = (t_start + t_end) / 2
        lon, lat = self.point_on_segment(
            lon1_scaled, lat1_scaled, lon2_scaled, lat2_scaled, midpoint
        )
        return self.cell_id_for_scaled_point(lon, lat)

    def approximate_dimensions_m(self) -> tuple[float, float]:
        if self.selection is not None:
            return self.selection.width_m, self.selection.height_m
        # Avoid a dependency cycle and keep fixed-bit mode informative.
        from .precision_selector import cell_dimensions_m

        return cell_dimensions_m(self.bits, self.reference_latitude)

    def describe(self) -> dict[str, object]:
        width_m, height_m = self.approximate_dimensions_m()
        return {
            "bits": self.bits,
            "longitude_bits": self.longitude_bits,
            "latitude_bits": self.latitude_bits,
            "reference_latitude": self.reference_latitude,
            "approx_width_m": width_m,
            "approx_height_m": height_m,
            "approx_geometric_mean_m": (width_m * height_m) ** 0.5,
            "coordinate_scale": self.coordinate_scale,
        }
