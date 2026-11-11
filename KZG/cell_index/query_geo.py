"""Command-line geographic rectangle query with complete-cell coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index_builder import load_authenticated_index
from .query_engine import query_geographic_rectangle, verify_query_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query the minimal complete Geohash-cell rectangle covering a "
            "longitude/latitude rectangle and verify every Merkle proof."
        )
    )
    parser.add_argument("--index", required=True, help="directory containing manifest.json")
    parser.add_argument("--min-lon", required=True, help="minimum longitude in degrees")
    parser.add_argument("--max-lon", required=True, help="maximum longitude in degrees")
    parser.add_argument("--min-lat", required=True, help="minimum latitude in degrees")
    parser.add_argument("--max-lat", required=True, help="maximum latitude in degrees")
    parser.add_argument("--output", default="query_result.json")
    parser.add_argument(
        "--trusted-root",
        default=None,
        help="trusted Root Hash hex string or a file containing the hex string",
    )
    parser.add_argument(
        "--tamper-field",
        default=None,
        help="test helper: change one returned edge field before verification",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index = load_authenticated_index(args.index)
    try:
        result = query_geographic_rectangle(
            index,
            args.min_lon,
            args.max_lon,
            args.min_lat,
            args.max_lat,
        )
    except ValueError as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    if args.tamper_field:
        edge = next(
            (
                edge
                for cell in result["cells"]
                for edge in cell["edges"]
            ),
            None,
        )
        if edge is None:
            raise SystemExit("the selected rectangle has no edge to tamper with")
        if args.tamper_field not in edge:
            raise SystemExit(f"unknown edge field for tamper test: {args.tamper_field}")
        original_edge_id = edge["edge_id"]
        edge[args.tamper_field] += 1
        # Keep candidate_edges synchronized so the intended failure is the
        # Merkle proof check rather than the redundant candidate-list check.
        candidate = next(
            candidate
            for candidate in result["candidate_edges"]
            if candidate["edge_id"] == original_edge_id
        )
        candidate[args.tamper_field] += 1

    trusted_root = args.trusted_root
    if trusted_root:
        trusted_root_path = Path(trusted_root)
        if trusted_root_path.is_file():
            trusted_root = trusted_root_path.read_text(encoding="utf-8").strip()

    try:
        verification = verify_query_result(
            result, index, expected_root_hex=trusted_root
        )
    except ValueError as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    result["verification"] = verification
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    summary = {
        **verification,
        "requested_geo_rectangle": result["geographic_query"],
        "cell_rectangle": result["query"],
        "covered_geo_rectangle": result["covered_geo_rectangle"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
