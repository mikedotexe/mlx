#!/usr/bin/env python3

# Copyright © 2026 Apple Inc.

from __future__ import annotations

import argparse
import json

from _safetensors_repack_helper import (
    default_base_alignment,
    format_repack_summary_text,
    repack_safetensors_file,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite one safetensors file into an alignment-friendly layout for "
            "MLX mmap experiments."
        )
    )
    parser.add_argument("input_file", help="Single .safetensors file to repack.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. Defaults to a sibling file named "
            "<stem>.aligned.safetensors."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--base-alignment",
        type=int,
        default=default_base_alignment(),
        help=(
            "Absolute byte alignment target for the first tensor payload "
            f"(default: {default_base_alignment()})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = repack_safetensors_file(
        args.input_file,
        output_path=args.output,
        base_alignment=args.base_alignment,
        force=args.force,
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_repack_summary_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
