#!/usr/bin/env python3

# Copyright © 2026 Apple Inc.

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


DEFAULT_WARMUP = 5
DEFAULT_RUNS = 30
_CASE_ORDER = (
    "docs_gpu_gpu",
    "docs_gpu_cpu",
    "cpu_to_gpu_chain",
    "gpu_to_cpu_chain",
    "np_view_from_mlx",
    "np_copy_from_mlx",
    "mlx_from_numpy_copy",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark unified-memory execution and zero-copy host-view behavior "
            "for MLX on Apple silicon."
        )
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--wheel-python",
        default=None,
        help=(
            "Python executable for a Metal-enabled MLX wheel runtime. If omitted, "
            "the wheel lane is skipped."
        ),
    )
    parser.add_argument(
        "--repo-pythonpath",
        default=None,
        help=(
            "PYTHONPATH pointing at the repo runtime (typically <repo>/python) used "
            "for the mmap diagnostics lane."
        ),
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help=(
            "Optional real safetensors or gguf model file for the repo mmap probe."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations for benchmarked cases (default: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Measured iterations for benchmarked cases (default: {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--child-mode",
        choices=("wheel", "repo_mmap"),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _summarize_ns(samples: list[int]) -> dict[str, int]:
    if not samples:
        raise ValueError("expected at least one timing sample")
    return {
        "median_ns": int(statistics.median(samples)),
        "min_ns": int(min(samples)),
        "max_ns": int(max(samples)),
        "mean_ns": int(sum(samples) / len(samples)),
    }


def _build_child_command(
    *,
    python_executable: str,
    child_mode: str,
    output_format: str,
    warmup: int,
    runs: int,
    model_file: str | None = None,
) -> list[str]:
    cmd = [
        python_executable,
        str(Path(__file__).resolve()),
        "--child-mode",
        child_mode,
        "--format",
        output_format,
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
    ]
    if model_file:
        cmd.extend(["--model-file", model_file])
    return cmd


def _build_child_env(
    *,
    lane: str,
    repo_pythonpath: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if lane == "wheel":
        env["PYTHONPATH"] = ""
        return env
    if lane == "repo_mmap":
        if repo_pythonpath:
            extra = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                repo_pythonpath if not extra else f"{repo_pythonpath}{os.pathsep}{extra}"
            )
        return env
    raise ValueError(f"unsupported lane: {lane}")


def _run_child(
    *,
    python_executable: str,
    lane: str,
    repo_pythonpath: str | None,
    warmup: int,
    runs: int,
    model_file: str | None,
) -> dict:
    cmd = _build_child_command(
        python_executable=python_executable,
        child_mode=lane,
        output_format="json",
        warmup=warmup,
        runs=runs,
        model_file=model_file,
    )
    env = _build_child_env(lane=lane, repo_pythonpath=repo_pythonpath)
    completed = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": (
                f"child failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip() or 'no output'}"
            ),
            "command": " ".join(shlex.quote(token) for token in cmd),
        }
    return json.loads(completed.stdout)


def _format_ns(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,} ns"


def _format_text(payload: dict) -> str:
    meta = payload["meta"]
    wheel = payload["results"]["wheel"]
    repo_probe = payload["results"]["repo_mmap_probe"]

    lines = [
        "Unified Memory / Zero-Copy Bench",
        f"warmup={meta['warmup']} runs={meta['runs']}",
        "",
        "Wheel lane:",
    ]
    if not wheel["available"]:
        lines.append(f"  skipped: {wheel['reason']}")
    else:
        lines.append(
            f"  metal_available={wheel['metal_available']} "
            f"default_device={wheel['default_device']}"
        )
        if wheel.get("device_name"):
            lines.append(f"  device={wheel['device_name']}")
        for case_id in _CASE_ORDER:
            stats = wheel["cases"][case_id]
            lines.append(
                f"  {case_id}: median={_format_ns(stats['median_ns'])} "
                f"min={_format_ns(stats['min_ns'])} max={_format_ns(stats['max_ns'])}"
            )
        semantics = wheel.get("semantics", {})
        if semantics:
            lines.append(
                "  semantics: "
                f"np_view_owndata={semantics.get('np_view_owndata')} "
                f"np_view_writeable={semantics.get('np_view_writeable')} "
                f"np_view_mutation_reflects={semantics.get('np_view_mutation_reflects')}"
            )

    lines.extend(["", "Repo mmap probe:"])
    if not repo_probe["available"]:
        lines.append(f"  skipped: {repo_probe['reason']}")
    else:
        lines.append(f"  load median={_format_ns(repo_probe['load_ns']['median_ns'])}")
        stats = repo_probe.get("mmap_stats", {})
        lines.append(
            "  mmap: "
            f"mapped_bytes={stats.get('mapped_bytes')} "
            f"copied_bytes={stats.get('copied_bytes')} "
            f"fallback_tensors={stats.get('fallback_tensors')}"
        )
        if stats.get("fallback_reasons"):
            lines.append(f"  fallback_reasons={stats['fallback_reasons']}")
    return "\n".join(lines)


def _run_wheel_cases(*, warmup: int, runs: int) -> dict:
    import time

    import mlx.core as mx
    import numpy as np

    def bench_ns(fn):
        for _ in range(max(warmup, 0)):
            fn()
        samples = []
        for _ in range(max(runs, 1)):
            start = time.perf_counter_ns()
            fn()
            end = time.perf_counter_ns()
            samples.append(end - start)
        return _summarize_ns(samples)

    result = {
        "available": True,
        "metal_available": bool(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "cases": {},
        "semantics": {},
    }
    if not result["metal_available"]:
        result["available"] = False
        result["reason"] = "Metal is not available in the wheel lane."
        return result

    try:
        device_info = mx.device_info(mx.gpu)
    except Exception as exc:  # pragma: no cover - best-effort metadata
        device_info = {"device_info_error": repr(exc)}
    result["device_info"] = device_info
    result["device_name"] = device_info.get("device_name")

    a = mx.random.uniform(shape=(4096, 512), dtype=mx.float32)
    b = mx.random.uniform(shape=(512, 4), dtype=mx.float32)
    small = mx.random.uniform(shape=(32768,), dtype=mx.float32)
    arr = mx.arange(1_000_000, dtype=mx.float32)
    np_src = np.arange(1_000_000, dtype=np.float32)
    mx.eval(a, b, small, arr)
    mx.synchronize()

    def docs_case(device_1, device_2):
        x = mx.matmul(a, b, stream=device_1)
        y = b
        for _ in range(500):
            y = mx.exp(y, stream=device_2)
        mx.eval(x, y)
        mx.synchronize()

    def cpu_to_gpu_chain():
        c = mx.add(small, 1.0, stream=mx.cpu)
        d = mx.log1p(c, stream=mx.gpu)
        mx.eval(d)
        mx.synchronize()

    def gpu_to_cpu_chain():
        c = mx.add(small, 1.0, stream=mx.gpu)
        d = mx.log1p(c, stream=mx.cpu)
        mx.eval(d)
        mx.synchronize()

    def np_view_from_mlx():
        view = np.array(arr, copy=False)
        _ = view[0]

    def np_copy_from_mlx():
        copied = np.array(arr)
        _ = copied[0]

    def mlx_from_numpy_copy():
        converted = mx.array(np_src)
        mx.eval(converted)
        mx.synchronize()

    result["cases"]["docs_gpu_gpu"] = bench_ns(lambda: docs_case(mx.gpu, mx.gpu))
    result["cases"]["docs_gpu_cpu"] = bench_ns(lambda: docs_case(mx.gpu, mx.cpu))
    result["cases"]["cpu_to_gpu_chain"] = bench_ns(cpu_to_gpu_chain)
    result["cases"]["gpu_to_cpu_chain"] = bench_ns(gpu_to_cpu_chain)
    result["cases"]["np_view_from_mlx"] = bench_ns(np_view_from_mlx)
    result["cases"]["np_copy_from_mlx"] = bench_ns(np_copy_from_mlx)
    result["cases"]["mlx_from_numpy_copy"] = bench_ns(mlx_from_numpy_copy)

    view = np.array(arr, copy=False)
    before = int(arr[0].item())
    view[0] = before + 7
    mx.synchronize()
    after = int(arr[0].item())
    result["semantics"] = {
        "np_view_owndata": bool(view.flags.owndata),
        "np_view_writeable": bool(view.flags.writeable),
        "np_view_base_type": type(view.base).__name__ if view.base is not None else None,
        "np_view_mutation_reflects": after == before + 7,
    }
    return result


def _run_repo_mmap_probe(*, warmup: int, runs: int, model_file: str | None) -> dict:
    import time

    import mlx.core as mx

    if not model_file:
        return {
            "available": False,
            "reason": "--model-file is required for repo_mmap_probe.",
        }
    if not hasattr(mx, "last_mmap_load_stats"):
        return {
            "available": False,
            "reason": "Repo runtime does not expose mx.last_mmap_load_stats().",
        }

    samples = []
    last_stats = None
    for _ in range(max(warmup, 0)):
        mx.last_mmap_load_stats(clear=True)
        _ = mx.load(model_file, memory_map=True)
        mx.last_mmap_load_stats(clear=True)
    for _ in range(max(runs, 1)):
        mx.last_mmap_load_stats(clear=True)
        start = time.perf_counter_ns()
        loaded = mx.load(model_file, memory_map=True)
        end = time.perf_counter_ns()
        samples.append(end - start)
        last_stats = mx.last_mmap_load_stats(clear=True)
        del loaded

    return {
        "available": True,
        "load_ns": _summarize_ns(samples),
        "mmap_stats": last_stats,
    }


def _run_child_mode(args: argparse.Namespace) -> int:
    if args.child_mode == "wheel":
        payload = _run_wheel_cases(warmup=args.warmup, runs=args.runs)
    elif args.child_mode == "repo_mmap":
        payload = _run_repo_mmap_probe(
            warmup=args.warmup,
            runs=args.runs,
            model_file=args.model_file,
        )
    else:  # pragma: no cover - argparse constrains this already
        raise ValueError(f"unsupported child mode: {args.child_mode}")

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        text_payload = {
            "meta": {
                "warmup": args.warmup,
                "runs": args.runs,
                "wheel_python": None,
                "repo_pythonpath": None,
                "model_file": args.model_file,
            },
            "results": {
                "wheel": (
                    payload
                    if args.child_mode == "wheel"
                    else {"available": False, "reason": "child mode is repo_mmap"}
                ),
                "repo_mmap_probe": (
                    payload
                    if args.child_mode == "repo_mmap"
                    else {"available": False, "reason": "child mode is wheel"}
                ),
            },
        }
        print(_format_text(text_payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.child_mode:
        return _run_child_mode(args)

    payload = {
        "meta": {
            "warmup": args.warmup,
            "runs": args.runs,
            "wheel_python": args.wheel_python,
            "repo_pythonpath": args.repo_pythonpath,
            "model_file": args.model_file,
        },
        "results": {
            "wheel": {
                "available": False,
                "reason": "--wheel-python not provided.",
            },
            "repo_mmap_probe": {
                "available": False,
                "reason": "--repo-pythonpath not provided.",
            },
        },
    }

    if args.wheel_python:
        payload["results"]["wheel"] = _run_child(
            python_executable=args.wheel_python,
            lane="wheel",
            repo_pythonpath=None,
            warmup=args.warmup,
            runs=args.runs,
            model_file=None,
        )
    if args.repo_pythonpath:
        payload["results"]["repo_mmap_probe"] = _run_child(
            python_executable=sys.executable,
            lane="repo_mmap",
            repo_pythonpath=args.repo_pythonpath,
            warmup=args.warmup,
            runs=args.runs,
            model_file=args.model_file,
        )

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
