"""Command-line complete-cell rectangle query and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index_builder import load_authenticated_index
from .query_engine import query_complete_cell_rectangle, verify_query_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query a complete rectangle of Geohash cells and verify every proof."
    )
    parser.add_argument("--index", required=True, help="directory containing manifest.json")
    parser.add_argument("--lon-index-min", type=int, required=True)
    parser.add_argument("--lon-index-max", type=int, required=True)
    parser.add_argument("--lat-index-min", type=int, required=True)
    parser.add_argument("--lat-index-max", type=int, required=True)
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
    result = query_complete_cell_rectangle(
        index,
        args.lon_index_min,
        args.lon_index_max,
        args.lat_index_min,
        args.lat_index_max,
    )
    if args.tamper_field:
        if not result["cells"] or not result["cells"][0]["edges"]:
            raise SystemExit("the selected rectangle has no edge to tamper with")
        result["cells"][0]["edges"][0][args.tamper_field] += 1
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
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
