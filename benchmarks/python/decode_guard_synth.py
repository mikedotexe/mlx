#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

import argparse
import time

import mlx.core as mx


def _to_array_list(loaded) -> list[mx.array]:
    arrays = loaded[0] if isinstance(loaded, tuple) else loaded
    if isinstance(arrays, dict):
        return list(arrays.values())
    if isinstance(arrays, list):
        return arrays
    if isinstance(arrays, tuple):
        return list(arrays)
    return [arrays]


def _numel(arr: mx.array) -> int:
    return int(arr.size)


def main() -> int:
    parser = argparse.ArgumentParser(
        "Synthetic decode-like tok/s guard for mapped-vs-copy parity checks."
    )
    parser.add_argument("--file", required=True)
    parser.add_argument("--format", default="auto")
    parser.add_argument("--memory-map", default="0")
    parser.add_argument("--gguf-nvfp4-compat", action="store_true")
    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
        help="Number of synthetic decode steps (default: 256).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=16,
        help="Warmup iterations excluded from timing (default: 16).",
    )
    parser.add_argument(
        "--max-elems",
        type=int,
        default=4_000_000,
        help="Maximum tensor elements to touch per step (default: 4,000,000).",
    )

    args = parser.parse_args()
    fmt = args.format if args.format and args.format != "auto" else None

    loaded = mx.load(
        args.file,
        format=fmt,
        memory_map=args.memory_map == "1",
        return_metadata=True,
        gguf_nvfp4_compat=args.gguf_nvfp4_compat,
    )
    tensors = _to_array_list(loaded)
    if not tensors:
        print("0.0")
        return 0

    target = max(tensors, key=_numel)
    flat = mx.reshape(target, (-1,))
    elems = min(int(flat.size), max(int(args.max_elems), 1))
    block = flat[:elems]

    for _ in range(max(int(args.warmup), 0)):
        mx.eval(mx.sum(block))
    mx.synchronize()

    steps = max(int(args.tokens), 1)
    tic = time.perf_counter()
    for _ in range(steps):
        mx.eval(mx.sum(block))
    mx.synchronize()
    toc = time.perf_counter()

    elapsed = toc - tic
    tok_s = steps / elapsed if elapsed > 0 else 0.0
    print(f"{tok_s:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
