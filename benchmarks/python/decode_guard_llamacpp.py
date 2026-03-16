#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Real decode tok/s guard using llama.cpp llama-bench JSON output."
        )
    )
    p.add_argument("--binary", required=True, help="Path to llama-bench binary.")
    p.add_argument("--file", required=True, help="Path to GGUF model file.")
    p.add_argument(
        "--memory-map",
        choices=["0", "1"],
        required=True,
        help="Whether to use mmap for llama.cpp model loading.",
    )
    p.add_argument(
        "--n-prompt",
        type=int,
        default=16,
        help="Prompt tokens for the prompt-processing leg (default: 16).",
    )
    p.add_argument(
        "--n-gen",
        type=int,
        default=64,
        help="Generated tokens for the decode leg (default: 64).",
    )
    p.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="llama-bench repetitions (default: 1).",
    )
    p.add_argument(
        "--n-gpu-layers",
        type=int,
        default=99,
        help="llama.cpp GPU layers setting (default: 99).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional threads override for llama-bench.",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Prefilled context depth for the generation leg (default: 0).",
    )
    p.add_argument(
        "--keep-warmup",
        action="store_true",
        help="Keep llama-bench warmup runs (default: disabled).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the parsed generation row as JSON instead of a float.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    binary = Path(args.binary).expanduser().resolve()
    file_path = Path(args.file).expanduser().resolve()
    cmd = [
        str(binary),
        "-m",
        str(file_path),
        "-p",
        str(max(args.n_prompt, 0)),
        "-n",
        str(max(args.n_gen, 0)),
        "-d",
        str(max(args.depth, 0)),
        "-r",
        str(max(args.repetitions, 1)),
        "-o",
        "json",
        "-mmp",
        args.memory_map,
        "-ngl",
        str(args.n_gpu_layers),
    ]
    if args.threads is not None:
        cmd += ["-t", str(max(args.threads, 1))]
    if not args.keep_warmup:
        cmd.append("--no-warmup")

    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "llama-bench failed:\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = json.loads(completed.stdout)
    generation_rows = [
        row
        for row in payload
        if int(row.get("n_gen", 0) or 0) > 0 and int(row.get("n_prompt", 0) or 0) == 0
    ]
    if not generation_rows:
        raise RuntimeError("llama-bench JSON did not contain a pure generation row")
    generation = generation_rows[0]

    if args.json:
        print(json.dumps(generation, indent=2, sort_keys=True))
    else:
        print(float(generation["avg_ts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
