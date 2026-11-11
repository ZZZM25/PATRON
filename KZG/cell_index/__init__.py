"""Direct cell lookup plus sparse-Merkle authentication for the split network."""

from .index_builder import AuthenticatedCellIndex, build_authenticated_index, load_authenticated_index
from .query_engine import (
    geographic_rectangle_to_cell_rectangle,
    query_complete_cell_rectangle,
    query_geographic_rectangle,
    verify_query_result,
)

__all__ = [
    "AuthenticatedCellIndex",
    "build_authenticated_index",
    "load_authenticated_index",
    "geographic_rectangle_to_cell_rectangle",
    "query_complete_cell_rectangle",
    "query_geographic_rectangle",
    "verify_query_result",
]
