# Copyright © 2026 Apple Inc.

import argparse
import datetime as dt
import gc
import json
import math
import mmap
import os
import re
import resource
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

from find_benchmark_models import find_candidates


_MMAP_DEBUG_RE = re.compile(
    r"^\[io mmap\] (?P<tag>\S+) file=(?P<file>.*?) "
    r"mapped_bytes=(?P<mapped_bytes>\d+) "
    r"copied_bytes=(?P<copied_bytes>\d+) "
    r"fallback_tensors=(?P<fallback_tensors>\d+)"
    r"(?: fallback_reasons=\{(?P<fallback_reasons>[^}]*)\})?$"
)


def _rss_bytes_from_ru_maxrss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)


def _fault_counts() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(getattr(usage, "ru_minflt", 0)), int(getattr(usage, "ru_majflt", 0))


def _prepare_file_cache(path: str, cache_mode: str) -> None:
    if cache_mode in {"inherit", "steady-state"}:
        return

    if cache_mode == "warm":
        chunk = bytearray(8 * 1024 * 1024)
        view = memoryview(chunk)
        with open(path, "rb", buffering=0) as f:
            while True:
                read = f.readinto(view)
                if not read:
                    break
        return

    if cache_mode == "cold-best-effort":
        with open(path, "rb", buffering=0) as f:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            try:
                mm.madvise(mmap.MADV_DONTNEED)
            finally:
                mm.close()
        return

    raise ValueError(f"unsupported cache mode: {cache_mode}")


def _parse_fallback_reasons(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    reasons: dict[str, int] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        key, value = token.rsplit(":", maxsplit=1)
        try:
            reasons[key] = int(value)
        except ValueError:
            continue
    return reasons


def _parse_mmap_debug_stats(stderr: str) -> dict | None:
    for line in reversed(stderr.splitlines()):
        match = _MMAP_DEBUG_RE.match(line.strip())
        if not match:
            continue
        return {
            "tag": match.group("tag"),
            "file": match.group("file"),
            "mapped_bytes": int(match.group("mapped_bytes")),
            "copied_bytes": int(match.group("copied_bytes")),
            "fallback_tensors": int(match.group("fallback_tensors")),
            "fallback_reasons": _parse_fallback_reasons(
                match.group("fallback_reasons")
            ),
        }
    return None


def _resolve_format(path: str, fmt: str | None) -> str | None:
    if fmt and fmt != "auto":
        return fmt
    _, ext = os.path.splitext(path)
    if not ext:
        return None
    return ext[1:]


def _to_array_list(loaded) -> list[mx.array]:
    arrays = loaded[0] if isinstance(loaded, tuple) else loaded
    if isinstance(arrays, dict):
        return list(arrays.values())
    if isinstance(arrays, list):
        return arrays
    if isinstance(arrays, tuple):
        return list(arrays)
    return [arrays]


def _run_synth_decode(
    arrays: list[mx.array], tokens: int, max_elems: int, repeats: int
) -> float:
    if not arrays:
        return 0.0

    target = max(arrays, key=lambda arr: int(arr.size))
    flat = mx.reshape(target, (-1,))
    elems = min(int(flat.size), max(max_elems, 1))
    block = flat[:elems]

    for _ in range(16):
        mx.eval(mx.sum(block))
    mx.synchronize()

    steps = max(tokens, 1)
    rates: list[float] = []
    for _ in range(max(repeats, 1)):
        tic = time.perf_counter()
        for _ in range(steps):
            mx.eval(mx.sum(block))
        mx.synchronize()
        toc = time.perf_counter()
        elapsed = toc - tic
        rates.append(steps / elapsed if elapsed > 0 else 0.0)
    return statistics.median(rates)


def _run_worker(
    path: str,
    fmt: str | None,
    memory_map: bool,
    gguf_nvfp4_compat: bool,
    decode_synth_tokens: int,
    decode_synth_max_elems: int,
    decode_synth_repeats: int,
    worker_prime_loads: int,
) -> dict:
    mx.clear_cache()

    for _ in range(max(worker_prime_loads, 0)):
        primed = mx.load(
            path,
            format=fmt,
            memory_map=memory_map,
            return_metadata=True,
            gguf_nvfp4_compat=gguf_nvfp4_compat,
        )
        primed_arrays = _to_array_list(primed)
        mx.eval(primed_arrays)
        mx.synchronize()
        del primed_arrays
        del primed
        gc.collect()
        mx.clear_cache()

    mx.reset_peak_memory()
    load_faults_before = _fault_counts()

    tic = time.perf_counter()
    loaded = mx.load(
        path,
        format=fmt,
        memory_map=memory_map,
        return_metadata=True,
        gguf_nvfp4_compat=gguf_nvfp4_compat,
    )
    array_list = _to_array_list(loaded)

    mx.eval(array_list)
    mx.synchronize()
    toc = time.perf_counter()
    load_faults_after = _fault_counts()

    result = {
        "elapsed_s": toc - tic,
        "peak_rss_bytes": _rss_bytes_from_ru_maxrss(),
        "load_minor_faults": load_faults_after[0] - load_faults_before[0],
        "load_major_faults": load_faults_after[1] - load_faults_before[1],
    }
    if decode_synth_tokens > 0:
        result["decode_tok_s"] = _run_synth_decode(
            arrays=array_list,
            tokens=decode_synth_tokens,
            max_elems=decode_synth_max_elems,
            repeats=decode_synth_repeats,
        )
    total_faults_after = _fault_counts()
    result["total_minor_faults"] = total_faults_after[0] - load_faults_before[0]
    result["total_major_faults"] = total_faults_after[1] - load_faults_before[1]
    return result


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _cv_pct(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0.0:
        return 0.0
    return 100.0 * statistics.pstdev(values) / mean


def _median_regression_pct(
    copy_values: list[float], mmap_values: list[float], pairwise: bool
) -> float:
    if pairwise:
        regressions = []
        for c, m in zip(copy_values, mmap_values):
            if c > 0:
                regressions.append(100.0 * (c - m) / c)
        if regressions:
            return _median(regressions)
        return 0.0
    copy_med = _median(copy_values)
    mmap_med = _median(mmap_values)
    return 100.0 * (copy_med - mmap_med) / copy_med if copy_med > 0 else 0.0


def _aa_noise_floor_pct(values: list[float]) -> float:
    # Estimate noise floor using two independent subsets of the same mode.
    if len(values) < 2:
        return 0.0
    left = values[0::2]
    right = values[1::2]
    if not left or not right:
        return 0.0
    left_med = _median(left)
    right_med = _median(right)
    if left_med <= 0:
        return 0.0
    return abs(100.0 * (left_med - right_med) / left_med)


def _effective_decode_regression_gate_pct(
    base_gate_pct: float,
    copy_values: list[float],
    mmap_values: list[float],
    enable_aa_noise_floor: bool,
    aa_multiplier: float,
    aa_margin_pct: float,
) -> tuple[float, float]:
    if not enable_aa_noise_floor:
        return base_gate_pct, 0.0
    aa_floor = max(_aa_noise_floor_pct(copy_values), _aa_noise_floor_pct(mmap_values))
    adaptive_floor = aa_floor * aa_multiplier + aa_margin_pct
    return max(base_gate_pct, adaptive_floor), aa_floor


def _read_optional_int_sysctl(name: str) -> int | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
    try:
        return int(out)
    except ValueError:
        return None


def _read_memory_pressure_state() -> str | None:
    try:
        out = subprocess.check_output(
            ["memory_pressure", "-Q"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return None
    for line in out.splitlines():
        if "System-wide memory pressure" in line:
            return line.split(":", maxsplit=1)[-1].strip().lower()
    return None


def _collect_attempt_metadata() -> dict:
    load_1m = None
    try:
        load_1m = float(os.getloadavg()[0])
    except Exception:
        pass
    return {
        "loadavg_1m": load_1m,
        "cpu_count": os.cpu_count() or 0,
        "thermal_pressure": _read_optional_int_sysctl("kern.thermal_pressure"),
        "memory_pressure_state": _read_memory_pressure_state(),
    }


def _attempt_noise_reasons(
    metadata: dict,
    loadavg_multiplier: float,
    thermal_threshold: int | None,
    noisy_memory_levels: set[str],
) -> list[str]:
    reasons: list[str] = []
    load_1m = metadata.get("loadavg_1m")
    cpu_count = int(metadata.get("cpu_count", 0) or 0)
    if load_1m is not None and cpu_count > 0:
        if load_1m > cpu_count * loadavg_multiplier:
            reasons.append(
                f"loadavg_1m={load_1m:.2f} exceeds {loadavg_multiplier:.2f}x cpu_count"
            )
    thermal = metadata.get("thermal_pressure")
    if thermal_threshold is not None and thermal is not None:
        if int(thermal) >= thermal_threshold:
            reasons.append(f"thermal_pressure={thermal} >= {thermal_threshold}")
    mem_state = metadata.get("memory_pressure_state")
    if mem_state and mem_state in noisy_memory_levels:
        reasons.append(f"memory_pressure_state={mem_state}")
    return reasons


def _load_history_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return records
    return records


def _append_history_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")


def _trend_bounds_mad(values: list[float], mad_mult: float) -> tuple[float, float]:
    med = _median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = _median(abs_dev) if abs_dev else 0.0
    band = mad_mult * mad
    return med - band, med + band


def _build_llama_bench_decode_cmd(
    llama_bench_binary: str,
    prompt_tokens: int,
    gen_tokens: int,
    repetitions: int,
    gpu_layers: int,
    threads: int | None,
    depth: int,
    keep_warmup: bool,
) -> str:
    helper = Path(__file__).with_name("decode_guard_llamacpp.py")
    cmd = [
        shlex.quote(sys.executable),
        shlex.quote(str(helper)),
        "--binary",
        shlex.quote(str(Path(llama_bench_binary).expanduser().resolve())),
        "--file",
        "{file}",
        "--memory-map",
        "{memory_map}",
        "--n-prompt",
        str(max(prompt_tokens, 0)),
        "--n-gen",
        str(max(gen_tokens, 0)),
        "--repetitions",
        str(max(repetitions, 1)),
        "--n-gpu-layers",
        str(gpu_layers),
        "--depth",
        str(max(depth, 0)),
    ]
    if threads is not None:
        cmd += ["--threads", str(max(threads, 1))]
    if keep_warmup:
        cmd.append("--keep-warmup")
    return " ".join(cmd)


def _provided_cli_flags(argv: list[str]) -> set[str]:
    flags: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if token.startswith("--"):
            flags.add(token.split("=", maxsplit=1)[0])
    return flags


def _discover_default_llama_bench_binary() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root.parent
        / "mikeconsciouness"
        / "nvfp4_quantization"
        / "ollama_fork"
        / "llama.cpp"
        / "build-codex"
        / "bin"
        / "llama-bench",
        repo_root.parent
        / "mikeconsciouness"
        / "nvfp4_quantization"
        / "ollama_fork"
        / "llama.cpp"
        / "build"
        / "bin"
        / "llama-bench",
        Path.home()
        / "other"
        / "mikeconsciouness"
        / "nvfp4_quantization"
        / "ollama_fork"
        / "llama.cpp"
        / "build-codex"
        / "bin"
        / "llama-bench",
        Path.home()
        / "other"
        / "mikeconsciouness"
        / "nvfp4_quantization"
        / "ollama_fork"
        / "llama.cpp"
        / "build"
        / "bin"
        / "llama-bench",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _apply_profile_defaults(args: argparse.Namespace, provided_flags: set[str]) -> None:
    if args.profile is None:
        return

    if args.profile == "ollama-gguf-real-decode":
        if not args.files and "--discover-models" not in provided_flags:
            args.discover_models = True
        if "--discover-ollama-blobs" not in provided_flags:
            args.discover_ollama_blobs = True
        if "--discover-model-id-regex" not in provided_flags:
            args.discover_model_id_regex = r"(?:phi3|llama2)"
        if "--discover-min-size-mib" not in provided_flags:
            args.discover_min_size_mib = 1024.0
        if "--discover-max-results" not in provided_flags:
            args.discover_max_results = 2
        if "--runs" not in provided_flags:
            args.runs = 1
        if "--warmup-runs" not in provided_flags:
            args.warmup_runs = 0
        if "--attempts" not in provided_flags:
            args.attempts = 1
        if "--decode-phase" not in provided_flags:
            args.decode_phase = "soft"
        if "--decode-runs" not in provided_flags:
            args.decode_runs = 1
        if (
            "--decode-cmd" not in provided_flags
            and "--decode-llama-bench-binary" not in provided_flags
            and args.decode_llama_bench_binary is None
        ):
            args.decode_llama_bench_binary = _discover_default_llama_bench_binary()
        if "--coverage-probe" not in provided_flags and "--no-coverage-probe" not in provided_flags:
            args.coverage_probe = True


def _assess_cold_cache_credibility(
    cold_copy_time_s: float,
    cold_copy_major_faults: float,
    warm_copy_time_s: float,
    warm_copy_major_faults: float,
    min_time_ratio: float,
    min_major_fault_delta: int,
) -> dict:
    time_ratio = (
        (cold_copy_time_s / warm_copy_time_s) if warm_copy_time_s > 0 else math.inf
    )
    major_fault_delta = int(cold_copy_major_faults - warm_copy_major_faults)
    credible = (
        time_ratio >= min_time_ratio or major_fault_delta >= min_major_fault_delta
    )
    return {
        "credible": credible,
        "time_ratio": time_ratio,
        "major_fault_delta": major_fault_delta,
        "warm_copy_time_s": warm_copy_time_s,
        "warm_copy_major_faults": warm_copy_major_faults,
    }


def _run_worker_subprocess(
    script_path: str,
    path: str,
    fmt: str | None,
    memory_map: bool,
    gguf_nvfp4_compat: bool,
    debug_io: bool,
    decode_synth_tokens: int,
    decode_synth_max_elems: int,
    decode_synth_repeats: int,
    cache_mode: str,
    worker_prime_loads: int,
    collect_mmap_stats: bool,
) -> dict:
    _prepare_file_cache(path, cache_mode)

    cmd = [
        sys.executable,
        script_path,
        "--worker",
        "--file",
        path,
        "--memory-map",
        "1" if memory_map else "0",
        "--worker-prime-loads",
        str(worker_prime_loads),
    ]
    if fmt:
        cmd += ["--format", fmt]
    if gguf_nvfp4_compat:
        cmd.append("--gguf-nvfp4-compat")
    if decode_synth_tokens > 0:
        cmd += ["--worker-decode-synth-tokens", str(decode_synth_tokens)]
        cmd += ["--worker-decode-synth-max-elems", str(decode_synth_max_elems)]
        cmd += ["--worker-decode-synth-repeats", str(decode_synth_repeats)]

    env = os.environ.copy()
    if debug_io or collect_mmap_stats:
        env["MLX_DEBUG_IO_MEMORY_MAP"] = "1"

    completed = subprocess.run(
        cmd,
        text=True,
        env=env,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "worker benchmark failed:\n"
            f"cmd={' '.join(shlex.quote(part) for part in cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if debug_io and completed.stderr.strip():
        print(completed.stderr.rstrip())

    trial = json.loads(completed.stdout)
    mmap_stats = _parse_mmap_debug_stats(completed.stderr)
    if mmap_stats is not None:
        trial["mmap_stats"] = mmap_stats
    return trial


def _run_trials(
    script_path: str,
    path: str,
    fmt: str | None,
    memory_map: bool,
    gguf_nvfp4_compat: bool,
    runs: int,
    warmup_runs: int,
    debug_io: bool,
    decode_synth_tokens: int,
    decode_synth_max_elems: int,
    decode_synth_repeats: int,
    cache_mode: str,
    worker_prime_loads: int,
    collect_mmap_stats: bool = False,
) -> list[dict]:
    trials = []
    total_runs = max(warmup_runs, 0) + max(runs, 0)
    for run_idx in range(total_runs):
        trial = _run_worker_subprocess(
            script_path=script_path,
            path=path,
            fmt=fmt,
            memory_map=memory_map,
            gguf_nvfp4_compat=gguf_nvfp4_compat,
            debug_io=debug_io,
            decode_synth_tokens=decode_synth_tokens,
            decode_synth_max_elems=decode_synth_max_elems,
            decode_synth_repeats=decode_synth_repeats,
            cache_mode=cache_mode,
            worker_prime_loads=worker_prime_loads,
            collect_mmap_stats=collect_mmap_stats,
        )
        if run_idx >= max(warmup_runs, 0):
            trials.append(trial)
    return trials


def _run_interleaved_trials(
    script_path: str,
    path: str,
    fmt: str | None,
    gguf_nvfp4_compat: bool,
    runs: int,
    warmup_runs: int,
    debug_io: bool,
    decode_synth_tokens: int,
    decode_synth_max_elems: int,
    decode_synth_repeats: int,
    cache_mode: str,
    worker_prime_loads: int,
) -> tuple[list[dict], list[dict]]:
    copy_trials: list[dict] = []
    mmap_trials: list[dict] = []

    total_runs = max(warmup_runs, 0) + max(runs, 0)
    for run_idx in range(total_runs):
        # Alternate mode order each iteration to reduce thermal/order bias.
        mode_order = (False, True) if (run_idx % 2 == 0) else (True, False)
        results: dict[bool, dict] = {}

        for memory_map in mode_order:
            mode_trials = _run_trials(
                script_path=script_path,
                path=path,
                fmt=fmt,
                memory_map=memory_map,
                gguf_nvfp4_compat=gguf_nvfp4_compat,
                runs=1,
                warmup_runs=0,
                debug_io=debug_io,
                decode_synth_tokens=decode_synth_tokens,
                decode_synth_max_elems=decode_synth_max_elems,
                decode_synth_repeats=decode_synth_repeats,
                cache_mode=cache_mode,
                worker_prime_loads=worker_prime_loads,
            )
            results[memory_map] = mode_trials[0]

        if run_idx >= max(warmup_runs, 0):
            copy_trials.append(results[False])
            mmap_trials.append(results[True])

    return copy_trials, mmap_trials


def _parse_first_float(text: str) -> float:
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        raise ValueError("decode command did not print a float")
    return float(match.group(0))


def _run_decode_command(
    template: str,
    mode: str,
    memory_map: bool,
    file_path: str,
    fmt: str | None,
) -> float:
    cmd = template.format(
        mode=mode,
        memory_map="1" if memory_map else "0",
        file=shlex.quote(file_path),
        format=shlex.quote(fmt or ""),
    )
    out = subprocess.check_output(cmd, shell=True, text=True)
    return _parse_first_float(out.strip())


def _run_decode_trials(
    template: str,
    mode: str,
    memory_map: bool,
    file_path: str,
    fmt: str | None,
    runs: int,
) -> list[float]:
    trials = []
    for _ in range(max(runs, 1)):
        trials.append(
            _run_decode_command(
                template=template,
                mode=mode,
                memory_map=memory_map,
                file_path=file_path,
                fmt=fmt,
            )
        )
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(
        "Mapped-vs-copy load benchmark with median/CV contract gates."
    )
    parser.add_argument(
        "--profile",
        choices=["ollama-gguf-real-decode"],
        default=None,
        help=(
            "Apply a canned benchmark profile. Explicit CLI flags still win over "
            "profile defaults."
        ),
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--format", default="auto")
    parser.add_argument("--memory-map", default="0")
    parser.add_argument("--gguf-nvfp4-compat", action="store_true")
    parser.add_argument(
        "--worker-decode-synth-tokens",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-decode-synth-max-elems",
        type=int,
        default=4_000_000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-decode-synth-repeats",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-prime-loads",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Model files (.safetensors/.gguf) to benchmark.",
    )
    parser.add_argument(
        "--discover-models",
        action="store_true",
        help=(
            "Auto-discover benchmark-worthy model files using curated filters "
            "(excludes vocab/tokenizer artifacts and tiny files)."
        ),
    )
    parser.add_argument(
        "--discover-root",
        action="append",
        default=[],
        help=(
            "Root directory/file for model discovery. Repeat for multiple roots. "
            "Defaults to current directory when --discover-models is set."
        ),
    )
    parser.add_argument(
        "--discover-min-size-mib",
        type=float,
        default=256.0,
        help="Minimum discovered file size in MiB (default: 256).",
    )
    parser.add_argument(
        "--discover-max-results",
        type=int,
        default=20,
        help="Maximum discovered files to include (default: 20).",
    )
    parser.add_argument(
        "--discover-all-per-dir",
        action="store_true",
        help="When discovering, keep all candidates per directory (default keeps largest only).",
    )
    parser.add_argument(
        "--discover-ollama-blobs",
        action="store_true",
        help="Also discover benchmark-worthy GGUF blobs from ~/.ollama.",
    )
    parser.add_argument(
        "--discover-model-id-regex",
        default=None,
        help=(
            "Regex filter applied to discovered model IDs. "
            "Primarily useful with --discover-ollama-blobs."
        ),
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help=(
            "Number of full benchmark attempts per file for consensus gating "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--min-pass",
        type=int,
        default=None,
        help=(
            "Minimum passing attempts required per file. "
            "Default: majority of --attempts."
        ),
    )
    parser.add_argument(
        "--max-invalid-attempts",
        type=int,
        default=None,
        help=(
            "Maximum extra retries for attempts marked invalid due to host noise. "
            "Default: same as --attempts."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Warmup runs per mode excluded from stats (default: 3).",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["inherit", "warm", "cold-best-effort", "steady-state"],
        default="inherit",
        help=(
            "File-cache / allocator temperature for each worker run: "
            "inherit=use ambient cache state; warm=pre-read the file before each run; "
            "cold-best-effort=ask the kernel to drop file-backed pages before each run; "
            "steady-state=prime one unmeasured load+eval in the worker before timing."
        ),
    )
    parser.add_argument(
        "--show-trials",
        action="store_true",
        help="Print raw per-trial load time and RSS values for each mode.",
    )
    parser.add_argument(
        "--coverage-probe",
        dest="coverage_probe",
        action="store_true",
        help=(
            "Run one extra mapped-load probe per file and parse MLX mmap coverage "
            "stats (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-coverage-probe",
        dest="coverage_probe",
        action="store_false",
        help="Disable the extra mapped-load coverage probe.",
    )
    parser.add_argument(
        "--validate-cold-cache",
        action="store_true",
        help=(
            "When --cache-mode cold-best-effort is used, run a warm copy probe "
            "and assess whether the cold-start claim is supported by timing or "
            "major-fault evidence."
        ),
    )
    parser.add_argument(
        "--cold-claim-hard-fail",
        action="store_true",
        help="Fail the file when cold-cache validation says the run was not credibly cold.",
    )
    parser.add_argument(
        "--cold-time-ratio-threshold",
        type=float,
        default=1.5,
        help=(
            "Cold-cache claim is credible if cold copy time is at least this many "
            "times the warm-copy baseline (default: 1.5)."
        ),
    )
    parser.add_argument(
        "--cold-major-fault-threshold",
        type=int,
        default=1,
        help=(
            "Cold-cache claim is credible if cold copy major faults exceed the warm "
            "baseline by at least this amount (default: 1)."
        ),
    )
    parser.add_argument(
        "--interleave-modes",
        action="store_true",
        help=(
            "Run copy/mmap trials in alternating order each iteration to "
            "reduce thermal/order bias."
        ),
    )
    parser.add_argument("--debug-io", action="store_true")
    parser.add_argument("--load-gate-pct", type=float, default=15.0)
    parser.add_argument("--rss-gate-pct", type=float, default=20.0)
    parser.add_argument("--cv-gate-pct", type=float, default=10.0)
    parser.add_argument("--decode-regression-gate-pct", type=float, default=3.0)
    parser.add_argument(
        "--decode-phase",
        choices=["soft", "hard"],
        default="soft",
        help=(
            "Decode policy: soft=warn and only fail on persistent decode issues; "
            "hard=decode must pass quorum like phase-1 gates."
        ),
    )
    parser.add_argument(
        "--decode-persistent-fail-runs",
        type=int,
        default=None,
        help=(
            "In --decode-phase soft, fail only if decode fails this many valid "
            "attempts (default: --min-pass threshold)."
        ),
    )
    parser.add_argument(
        "--decode-runs",
        type=int,
        default=3,
        help="Decode guard trial count per mode when --decode-cmd is set (default: 3).",
    )
    parser.add_argument(
        "--decode-cv-gate-pct",
        type=float,
        default=None,
        help=(
            "Optional maximum decode trial CV%% when decode guard is enabled "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--decode-cmd",
        default=None,
        help=(
            "Optional command template for decode tok/s guard. "
            "Supports placeholders {mode}, {memory_map}, {file}, {format}."
        ),
    )
    parser.add_argument(
        "--decode-llama-bench-binary",
        default=None,
        help=(
            "Shortcut decode backend: path to llama.cpp llama-bench binary. "
            "Builds an internal --decode-cmd around decode_guard_llamacpp.py."
        ),
    )
    parser.add_argument(
        "--decode-llama-bench-prompt-tokens",
        type=int,
        default=16,
        help="Prompt tokens used by the built-in llama-bench decode shortcut (default: 16).",
    )
    parser.add_argument(
        "--decode-llama-bench-gen-tokens",
        type=int,
        default=64,
        help="Generated tokens used by the built-in llama-bench decode shortcut (default: 64).",
    )
    parser.add_argument(
        "--decode-llama-bench-repetitions",
        type=int,
        default=1,
        help="llama-bench repetitions for the built-in decode shortcut (default: 1).",
    )
    parser.add_argument(
        "--decode-llama-bench-gpu-layers",
        type=int,
        default=99,
        help="llama-bench GPU layers for the built-in decode shortcut (default: 99).",
    )
    parser.add_argument(
        "--decode-llama-bench-threads",
        type=int,
        default=None,
        help="Optional thread override for the built-in llama-bench decode shortcut.",
    )
    parser.add_argument(
        "--decode-llama-bench-depth",
        type=int,
        default=0,
        help="Prefilled context depth for the built-in llama-bench decode shortcut (default: 0).",
    )
    parser.add_argument(
        "--decode-llama-bench-keep-warmup",
        action="store_true",
        help="Keep llama-bench warmup runs in the built-in decode shortcut.",
    )
    parser.add_argument(
        "--decode-aa-noise-floor",
        action="store_true",
        help=(
            "Use A/A split noise floor to auto-relax decode regression gate when "
            "measurement jitter is high."
        ),
    )
    parser.add_argument(
        "--decode-aa-multiplier",
        type=float,
        default=1.0,
        help="Multiplier for A/A noise floor when adapting decode regression gate.",
    )
    parser.add_argument(
        "--decode-aa-margin-pct",
        type=float,
        default=0.5,
        help="Additional safety margin in percent for adaptive decode gate.",
    )
    parser.add_argument(
        "--decode-synth",
        action="store_true",
        help=(
            "Use built-in synthetic decode guard in worker trials "
            "(avoids external decode command jitter)."
        ),
    )
    parser.add_argument(
        "--decode-synth-tokens",
        type=int,
        default=256,
        help="Synthetic decode steps per trial when --decode-synth is set (default: 256).",
    )
    parser.add_argument(
        "--decode-synth-max-elems",
        type=int,
        default=4_000_000,
        help=(
            "Max tensor elements touched per synthetic decode step when "
            "--decode-synth is set (default: 4,000,000)."
        ),
    )
    parser.add_argument(
        "--decode-synth-repeats",
        type=int,
        default=3,
        help=(
            "Timed repeats per synthetic decode trial; median is used "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--reject-noisy-attempts",
        dest="reject_noisy_attempts",
        action="store_true",
        help="Mark attempts invalid when host load/thermal/memory pressure is high.",
    )
    parser.add_argument(
        "--no-reject-noisy-attempts",
        dest="reject_noisy_attempts",
        action="store_false",
        help="Disable host-noise invalidation.",
    )
    parser.add_argument(
        "--noisy-loadavg-multiplier",
        type=float,
        default=2.0,
        help=(
            "Attempt is noisy if loadavg_1m exceeds this multiple of CPU count "
            "(default: 2.0)."
        ),
    )
    parser.add_argument(
        "--noisy-thermal-threshold",
        type=int,
        default=2,
        help=(
            "Attempt is noisy if kern.thermal_pressure is >= this value "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--noisy-memory-levels",
        default="warning,critical",
        help=(
            "Comma-separated memory pressure levels treated as noisy "
            "(default: warning,critical)."
        ),
    )
    parser.add_argument(
        "--history-json",
        default=str(Path(__file__).with_name("load_mmap_history.jsonl")),
        help="Path to JSONL benchmark history for trend analysis.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=20,
        help="Recent history window size used for trend detection (default: 20).",
    )
    parser.add_argument(
        "--trend-min-samples",
        type=int,
        default=5,
        help="Minimum prior samples required for trend checks (default: 5).",
    )
    parser.add_argument(
        "--trend-mad-mult",
        type=float,
        default=3.0,
        help="MAD band multiplier for trend bounds (default: 3.0).",
    )
    parser.add_argument(
        "--trend-hard-fail",
        action="store_true",
        help="Fail benchmark when trend checks detect significant regressions.",
    )
    parser.set_defaults(reject_noisy_attempts=True, coverage_probe=True)

    raw_argv = sys.argv[1:]
    provided_flags = _provided_cli_flags(raw_argv)
    args = parser.parse_args()
    _apply_profile_defaults(args, provided_flags)

    if args.worker:
        fmt = _resolve_format(args.file, args.format)
        result = _run_worker(
            args.file,
            fmt,
            args.memory_map == "1",
            args.gguf_nvfp4_compat,
            args.worker_decode_synth_tokens,
            args.worker_decode_synth_max_elems,
            args.worker_decode_synth_repeats,
            args.worker_prime_loads,
        )
        print(json.dumps(result))
        return 0

    if args.attempts < 1:
        raise ValueError("--attempts must be >= 1")
    if args.decode_cmd is not None and args.decode_llama_bench_binary is not None:
        raise ValueError(
            "--decode-cmd and --decode-llama-bench-binary are mutually exclusive"
        )
    if (
        args.profile == "ollama-gguf-real-decode"
        and args.decode_llama_bench_binary is None
        and args.decode_cmd is None
        and not args.decode_synth
    ):
        raise ValueError(
            "profile ollama-gguf-real-decode could not find llama-bench; "
            "set --decode-llama-bench-binary explicitly"
        )
    if args.decode_llama_bench_binary is not None:
        decode_binary = Path(args.decode_llama_bench_binary).expanduser().resolve()
        if not decode_binary.exists():
            raise ValueError(
                f"--decode-llama-bench-binary not found: {decode_binary}"
            )
    consensus_min_pass = (
        args.min_pass if args.min_pass is not None else (args.attempts // 2 + 1)
    )
    if consensus_min_pass < 1 or consensus_min_pass > args.attempts:
        raise ValueError("--min-pass must be between 1 and --attempts")
    max_invalid_attempts = (
        args.max_invalid_attempts if args.max_invalid_attempts is not None else args.attempts
    )
    if max_invalid_attempts < 0:
        raise ValueError("--max-invalid-attempts must be >= 0")
    noisy_memory_levels = {
        token.strip().lower()
        for token in str(args.noisy_memory_levels).split(",")
        if token.strip()
    }
    history_path = Path(args.history_json).expanduser() if args.history_json else None
    prior_history = _load_history_jsonl(history_path) if history_path else []
    if args.attempts > 1:
        print(
            "Consensus mode enabled: "
            f"attempts={args.attempts} min_pass={consensus_min_pass}"
        )
    if args.profile:
        print(f"Profile enabled: {args.profile}")
    if args.cache_mode != "inherit":
        print(f"Cache mode enabled: {args.cache_mode}")

    file_paths = list(args.files)
    discovered_formats: dict[str, str] = {}
    if args.discover_models:
        discover_roots = (
            [Path(root).expanduser().resolve() for root in args.discover_root]
            if args.discover_root
            else [Path.cwd()]
        )
        discover_candidates = find_candidates(
            roots=discover_roots,
            min_size_bytes=int(args.discover_min_size_mib * 1024 * 1024),
            one_per_dir=not args.discover_all_per_dir,
            include_ollama_blobs=args.discover_ollama_blobs,
            model_id_regex=args.discover_model_id_regex,
        )
        discovered_files = [
            c.path for c in discover_candidates[: max(args.discover_max_results, 0)]
        ]
        for candidate in discover_candidates[: max(args.discover_max_results, 0)]:
            if candidate.format:
                discovered_formats[candidate.path] = candidate.format
        existing = set(file_paths)
        for path in discovered_files:
            if path not in existing:
                file_paths.append(path)
                existing.add(path)

        print(
            "Discovery selected "
            f"{len(discovered_files)} model file(s) from {len(discover_roots)} root(s)."
        )
        for candidate in discover_candidates[: max(args.discover_max_results, 0)]:
            suffix = f" [{candidate.model_id}]" if candidate.model_id else ""
            print(f"  discovered: {candidate.path}{suffix}")

    if not file_paths:
        raise ValueError(
            "provide at least one file path or use --discover-models "
            "with --discover-root"
        )

    script_path = os.path.abspath(__file__)
    failed = False

    for file_path in file_paths:
        discovered_format = discovered_formats.get(file_path)
        fmt = (
            discovered_format
            if args.format == "auto" and discovered_format is not None
            else _resolve_format(file_path, args.format)
        )
        decode_cmd_template = args.decode_cmd
        decode_skip_reason = None
        if args.decode_llama_bench_binary is not None:
            if fmt == "gguf":
                decode_cmd_template = _build_llama_bench_decode_cmd(
                    llama_bench_binary=args.decode_llama_bench_binary,
                    prompt_tokens=args.decode_llama_bench_prompt_tokens,
                    gen_tokens=args.decode_llama_bench_gen_tokens,
                    repetitions=args.decode_llama_bench_repetitions,
                    gpu_layers=args.decode_llama_bench_gpu_layers,
                    threads=args.decode_llama_bench_threads,
                    depth=args.decode_llama_bench_depth,
                    keep_warmup=args.decode_llama_bench_keep_warmup,
                )
            else:
                decode_skip_reason = "built-in llama-bench decode skipped (format is not gguf)"
        worker_prime_loads = 1 if args.cache_mode == "steady-state" else 0
        target_valid_attempts = args.attempts
        max_total_attempts = target_valid_attempts + max_invalid_attempts
        valid_attempts = 0
        invalid_attempts = 0
        total_attempts = 0
        phase1_passes = 0
        hard_passes = 0
        decode_failures = 0
        decode_enabled = bool(decode_cmd_template or args.decode_synth)
        load_improve_history: list[float] = []
        rss_reduce_history: list[float] = []
        decode_regression_history: list[float] = []
        copy_minor_fault_history: list[float] = []
        mmap_minor_fault_history: list[float] = []
        copy_major_fault_history: list[float] = []
        mmap_major_fault_history: list[float] = []

        while valid_attempts < target_valid_attempts and total_attempts < max_total_attempts:
            total_attempts += 1
            if max_total_attempts > 1:
                print(
                    f"\nFile: {file_path} "
                    f"[attempt {total_attempts}/{max_total_attempts}]"
                )
            else:
                print(f"\nFile: {file_path}")
            if decode_skip_reason:
                print(f"  {decode_skip_reason}")

            attempt_meta = _collect_attempt_metadata()
            if args.show_trials:
                print(
                    "  host "
                    f"loadavg_1m={attempt_meta.get('loadavg_1m')} "
                    f"thermal={attempt_meta.get('thermal_pressure')} "
                    f"mem_pressure={attempt_meta.get('memory_pressure_state')}"
                )

            if args.interleave_modes:
                copy_trials, mmap_trials = _run_interleaved_trials(
                    script_path=script_path,
                    path=file_path,
                    fmt=fmt,
                    gguf_nvfp4_compat=args.gguf_nvfp4_compat,
                    runs=args.runs,
                    warmup_runs=args.warmup_runs,
                    debug_io=args.debug_io,
                    decode_synth_tokens=args.decode_synth_tokens if args.decode_synth else 0,
                    decode_synth_max_elems=args.decode_synth_max_elems,
                    decode_synth_repeats=args.decode_synth_repeats,
                    cache_mode=args.cache_mode,
                    worker_prime_loads=worker_prime_loads,
                )
            else:
                copy_trials = _run_trials(
                    script_path,
                    file_path,
                    fmt,
                    False,
                    args.gguf_nvfp4_compat,
                    args.runs,
                    args.warmup_runs,
                    args.debug_io,
                    args.decode_synth_tokens if args.decode_synth else 0,
                    args.decode_synth_max_elems,
                    args.decode_synth_repeats,
                    args.cache_mode,
                    worker_prime_loads,
                )
                mmap_trials = _run_trials(
                    script_path,
                    file_path,
                    fmt,
                    True,
                    args.gguf_nvfp4_compat,
                    args.runs,
                    args.warmup_runs,
                    args.debug_io,
                    args.decode_synth_tokens if args.decode_synth else 0,
                    args.decode_synth_max_elems,
                    args.decode_synth_repeats,
                    args.cache_mode,
                    worker_prime_loads,
                )

            copy_time = [x["elapsed_s"] for x in copy_trials]
            mmap_time = [x["elapsed_s"] for x in mmap_trials]
            copy_rss = [x["peak_rss_bytes"] for x in copy_trials]
            mmap_rss = [x["peak_rss_bytes"] for x in mmap_trials]
            copy_minor_faults = [x["load_minor_faults"] for x in copy_trials]
            mmap_minor_faults = [x["load_minor_faults"] for x in mmap_trials]
            copy_major_faults = [x["load_major_faults"] for x in copy_trials]
            mmap_major_faults = [x["load_major_faults"] for x in mmap_trials]

            copy_time_med = _median(copy_time)
            mmap_time_med = _median(mmap_time)
            copy_rss_med = _median(copy_rss)
            mmap_rss_med = _median(mmap_rss)
            copy_minor_faults_med = _median(copy_minor_faults)
            mmap_minor_faults_med = _median(mmap_minor_faults)
            copy_major_faults_med = _median(copy_major_faults)
            mmap_major_faults_med = _median(mmap_major_faults)

            load_improve_pct = (
                100.0 * (copy_time_med - mmap_time_med) / copy_time_med
                if copy_time_med > 0
                else 0.0
            )
            rss_reduce_pct = (
                100.0 * (copy_rss_med - mmap_rss_med) / copy_rss_med
                if copy_rss_med > 0
                else 0.0
            )

            copy_time_cv = _cv_pct(copy_time)
            mmap_time_cv = _cv_pct(mmap_time)
            copy_rss_cv = _cv_pct(copy_rss)
            mmap_rss_cv = _cv_pct(mmap_rss)
            max_cv = max(copy_time_cv, mmap_time_cv, copy_rss_cv, mmap_rss_cv)

            load_ok = load_improve_pct >= args.load_gate_pct
            rss_ok = rss_reduce_pct >= args.rss_gate_pct
            cv_ok = max_cv <= args.cv_gate_pct
            phase1_ok = load_ok and rss_ok and cv_ok

            if args.warmup_runs > 0:
                print(f"  warmup runs per mode: {args.warmup_runs}")
            if args.show_trials:
                print(
                    "  trials copy_time_s="
                    f"{[round(v, 4) for v in copy_time]} "
                    "mmap_time_s="
                    f"{[round(v, 4) for v in mmap_time]}"
                )
                print(
                    "  trials copy_rss_mib="
                    f"{[round(v / (1024**2), 2) for v in copy_rss]} "
                    "mmap_rss_mib="
                    f"{[round(v / (1024**2), 2) for v in mmap_rss]}"
                )
                print(
                    "  trials load_faults copy(min/maj)="
                    f"{list(zip(copy_minor_faults, copy_major_faults))} "
                    "mmap(min/maj)="
                    f"{list(zip(mmap_minor_faults, mmap_major_faults))}"
                )
            print(f"  load median copy={copy_time_med:.4f}s mmap={mmap_time_med:.4f}s")
            print(
                f"  rss median  copy={copy_rss_med / (1024**2):.2f}MiB "
                f"mmap={mmap_rss_med / (1024**2):.2f}MiB"
            )
            print(
                "  load faults median "
                f"copy(min/maj)=({copy_minor_faults_med:.0f}/{copy_major_faults_med:.0f}) "
                f"mmap(min/maj)=({mmap_minor_faults_med:.0f}/{mmap_major_faults_med:.0f})"
            )
            print(
                f"  cv% time(copy/mmap)=({copy_time_cv:.2f}/{mmap_time_cv:.2f}) "
                f"rss(copy/mmap)=({copy_rss_cv:.2f}/{mmap_rss_cv:.2f})"
            )
            print(
                f"  gates load>={args.load_gate_pct:.1f}% rss>={args.rss_gate_pct:.1f}% "
                f"cv<={args.cv_gate_pct:.1f}% => "
                f"load={load_ok} rss={rss_ok} cv={cv_ok}"
            )

            decode_ok = True
            decode_regression_pct = None
            decode_regression_ok = True
            decode_cv_ok = True
            decode_gate_effective_pct = args.decode_regression_gate_pct
            decode_aa_floor = 0.0

            if decode_cmd_template:
                copy_tok_s_trials = _run_decode_trials(
                    template=decode_cmd_template,
                    mode="copy",
                    memory_map=False,
                    file_path=file_path,
                    fmt=fmt,
                    runs=args.decode_runs,
                )
                mmap_tok_s_trials = _run_decode_trials(
                    template=decode_cmd_template,
                    mode="mapped",
                    memory_map=True,
                    file_path=file_path,
                    fmt=fmt,
                    runs=args.decode_runs,
                )
                copy_tok_s = _median(copy_tok_s_trials)
                mmap_tok_s = _median(mmap_tok_s_trials)
                copy_tok_cv = _cv_pct(copy_tok_s_trials)
                mmap_tok_cv = _cv_pct(mmap_tok_s_trials)
                decode_cv = max(copy_tok_cv, mmap_tok_cv)
                decode_regression_pct = _median_regression_pct(
                    copy_values=copy_tok_s_trials,
                    mmap_values=mmap_tok_s_trials,
                    pairwise=args.interleave_modes,
                )
                decode_gate_effective_pct, decode_aa_floor = (
                    _effective_decode_regression_gate_pct(
                        base_gate_pct=args.decode_regression_gate_pct,
                        copy_values=copy_tok_s_trials,
                        mmap_values=mmap_tok_s_trials,
                        enable_aa_noise_floor=args.decode_aa_noise_floor,
                        aa_multiplier=args.decode_aa_multiplier,
                        aa_margin_pct=args.decode_aa_margin_pct,
                    )
                )
                decode_regression_ok = (
                    decode_regression_pct <= decode_gate_effective_pct
                )
                decode_cv_ok = (
                    True
                    if args.decode_cv_gate_pct is None
                    else decode_cv <= args.decode_cv_gate_pct
                )
                decode_ok = decode_regression_ok and decode_cv_ok
                if args.show_trials:
                    print(
                        "  decode trials copy_tok_s="
                        f"{[round(v, 3) for v in copy_tok_s_trials]} "
                        "mmap_tok_s="
                        f"{[round(v, 3) for v in mmap_tok_s_trials]}"
                    )
                print(
                    f"  decode tok/s copy={copy_tok_s:.3f} mmap={mmap_tok_s:.3f} "
                    f"regression={decode_regression_pct:.2f}% "
                    f"(gate <= {decode_gate_effective_pct:.2f}%)"
                )
                if args.decode_aa_noise_floor:
                    print(f"  decode A/A noise floor={decode_aa_floor:.2f}%")
                print(
                    "  decode cv% "
                    f"copy={copy_tok_cv:.2f} mmap={mmap_tok_cv:.2f} "
                    + (
                        "(gate disabled) => "
                        if args.decode_cv_gate_pct is None
                        else f"(gate <= {args.decode_cv_gate_pct:.1f}%) => "
                    )
                    + f"regression={decode_regression_ok} cv={decode_cv_ok}"
                )
            elif args.decode_synth:
                copy_tok_s_trials = [x["decode_tok_s"] for x in copy_trials]
                mmap_tok_s_trials = [x["decode_tok_s"] for x in mmap_trials]
                copy_tok_s = _median(copy_tok_s_trials)
                mmap_tok_s = _median(mmap_tok_s_trials)
                copy_tok_cv = _cv_pct(copy_tok_s_trials)
                mmap_tok_cv = _cv_pct(mmap_tok_s_trials)
                decode_cv = max(copy_tok_cv, mmap_tok_cv)
                decode_regression_pct = _median_regression_pct(
                    copy_values=copy_tok_s_trials,
                    mmap_values=mmap_tok_s_trials,
                    pairwise=args.interleave_modes,
                )
                decode_gate_effective_pct, decode_aa_floor = (
                    _effective_decode_regression_gate_pct(
                        base_gate_pct=args.decode_regression_gate_pct,
                        copy_values=copy_tok_s_trials,
                        mmap_values=mmap_tok_s_trials,
                        enable_aa_noise_floor=args.decode_aa_noise_floor,
                        aa_multiplier=args.decode_aa_multiplier,
                        aa_margin_pct=args.decode_aa_margin_pct,
                    )
                )
                decode_regression_ok = (
                    decode_regression_pct <= decode_gate_effective_pct
                )
                decode_cv_ok = (
                    True
                    if args.decode_cv_gate_pct is None
                    else decode_cv <= args.decode_cv_gate_pct
                )
                decode_ok = decode_regression_ok and decode_cv_ok
                if args.show_trials:
                    print(
                        "  decode trials copy_tok_s="
                        f"{[round(v, 3) for v in copy_tok_s_trials]} "
                        "mmap_tok_s="
                        f"{[round(v, 3) for v in mmap_tok_s_trials]}"
                    )
                print(
                    f"  decode tok/s copy={copy_tok_s:.3f} mmap={mmap_tok_s:.3f} "
                    f"regression={decode_regression_pct:.2f}% "
                    f"(gate <= {decode_gate_effective_pct:.2f}%)"
                )
                if args.decode_aa_noise_floor:
                    print(f"  decode A/A noise floor={decode_aa_floor:.2f}%")
                print(
                    "  decode cv% "
                    f"copy={copy_tok_cv:.2f} mmap={mmap_tok_cv:.2f} "
                    + (
                        "(gate disabled) => "
                        if args.decode_cv_gate_pct is None
                        else f"(gate <= {args.decode_cv_gate_pct:.1f}%) => "
                    )
                    + f"regression={decode_regression_ok} cv={decode_cv_ok}"
                )
            else:
                print("  decode tok/s gate skipped (set --decode-cmd or --decode-synth)")

            noise_reasons = _attempt_noise_reasons(
                metadata=attempt_meta,
                loadavg_multiplier=args.noisy_loadavg_multiplier,
                thermal_threshold=args.noisy_thermal_threshold,
                noisy_memory_levels=noisy_memory_levels,
            )
            attempt_invalid = args.reject_noisy_attempts and bool(noise_reasons)
            if attempt_invalid:
                invalid_attempts += 1
                print(f"  attempt invalid due to host-noise: {', '.join(noise_reasons)}")
                print("  attempt excluded from quorum")
            else:
                valid_attempts += 1
                load_improve_history.append(load_improve_pct)
                rss_reduce_history.append(rss_reduce_pct)
                copy_minor_fault_history.append(copy_minor_faults_med)
                mmap_minor_fault_history.append(mmap_minor_faults_med)
                copy_major_fault_history.append(copy_major_faults_med)
                mmap_major_fault_history.append(mmap_major_faults_med)
                if decode_regression_pct is not None:
                    decode_regression_history.append(decode_regression_pct)
                if phase1_ok:
                    phase1_passes += 1
                hard_attempt_ok = phase1_ok and decode_ok
                if hard_attempt_ok:
                    hard_passes += 1
                if decode_enabled and not decode_ok:
                    decode_failures += 1
                if args.decode_phase == "hard":
                    attempt_pass = hard_attempt_ok
                else:
                    attempt_pass = phase1_ok
                if max_total_attempts > 1:
                    print(f"  attempt pass={attempt_pass}")

            quorum_passes = hard_passes if args.decode_phase == "hard" else phase1_passes
            remaining_valid_slots = target_valid_attempts - valid_attempts
            if quorum_passes >= consensus_min_pass:
                break
            if quorum_passes + remaining_valid_slots < consensus_min_pass:
                break
            if valid_attempts + (max_total_attempts - total_attempts) < target_valid_attempts:
                break

        quorum_passes = hard_passes if args.decode_phase == "hard" else phase1_passes
        file_ok = quorum_passes >= consensus_min_pass
        if valid_attempts == 0:
            file_ok = False
            print("No valid attempts collected after noise filtering.")
        decode_persistent_fail = False
        decode_persistent_threshold = (
            args.decode_persistent_fail_runs
            if args.decode_persistent_fail_runs is not None
            else consensus_min_pass
        )
        if args.decode_phase == "soft" and decode_enabled:
            decode_persistent_fail = decode_failures >= decode_persistent_threshold
            if decode_failures > 0:
                print(
                    "Decode soft-phase summary: "
                    f"failures={decode_failures}/{valid_attempts} "
                    f"persistent_threshold={decode_persistent_threshold}"
                )
            if decode_persistent_fail:
                print("Decode soft-phase escalated to failure (persistent regressions).")
                file_ok = False

        cold_cache_validation = None
        if args.cache_mode == "cold-best-effort" and args.validate_cold_cache:
            warm_copy_probe = _run_trials(
                script_path,
                file_path,
                fmt,
                False,
                args.gguf_nvfp4_compat,
                1,
                0,
                args.debug_io,
                0,
                args.decode_synth_max_elems,
                args.decode_synth_repeats,
                "warm",
                0,
                collect_mmap_stats=False,
            )[0]
            cold_cache_validation = _assess_cold_cache_credibility(
                cold_copy_time_s=copy_time_med,
                cold_copy_major_faults=copy_major_faults_med,
                warm_copy_time_s=float(warm_copy_probe["elapsed_s"]),
                warm_copy_major_faults=float(warm_copy_probe["load_major_faults"]),
                min_time_ratio=args.cold_time_ratio_threshold,
                min_major_fault_delta=args.cold_major_fault_threshold,
            )
            print(
                "  cold validation "
                f"warm_copy={cold_cache_validation['warm_copy_time_s']:.4f}s "
                f"cold_copy={copy_time_med:.4f}s "
                f"time_ratio={cold_cache_validation['time_ratio']:.2f}x "
                f"major_fault_delta={cold_cache_validation['major_fault_delta']} "
                f"credible={cold_cache_validation['credible']}"
            )
            if args.cold_claim_hard_fail and not cold_cache_validation["credible"]:
                print("Cold-cache hard-fail enabled: failing file due to weak cold evidence.")
                file_ok = False

        mmap_coverage = None
        mmap_coverage_mapped_ratio_pct = None
        if args.coverage_probe:
            probe = _run_trials(
                script_path,
                file_path,
                fmt,
                True,
                args.gguf_nvfp4_compat,
                1,
                0,
                args.debug_io,
                0,
                args.decode_synth_max_elems,
                args.decode_synth_repeats,
                "inherit",
                0,
                collect_mmap_stats=True,
            )[0]
            mmap_coverage = probe.get("mmap_stats")
            if mmap_coverage is not None:
                total_bytes = (
                    mmap_coverage["mapped_bytes"] + mmap_coverage["copied_bytes"]
                )
                mmap_coverage_mapped_ratio_pct = (
                    100.0 * mmap_coverage["mapped_bytes"] / total_bytes
                    if total_bytes > 0
                    else 0.0
                )
                print(
                    "  mmap coverage "
                    f"mapped={mmap_coverage['mapped_bytes'] / (1024**2):.2f}MiB "
                    f"copied={mmap_coverage['copied_bytes'] / (1024**2):.2f}MiB "
                    f"mapped_ratio={mmap_coverage_mapped_ratio_pct:.2f}% "
                    f"fallback_tensors={mmap_coverage['fallback_tensors']}"
                )
                if mmap_coverage["fallback_reasons"]:
                    fallback_summary = ", ".join(
                        f"{key}={value}"
                        for key, value in sorted(
                            mmap_coverage["fallback_reasons"].items()
                        )
                    )
                    print(f"  mmap fallback reasons {fallback_summary}")
            else:
                print("  mmap coverage unavailable (no MLX mmap stats emitted)")

        summary_payload = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "file": file_path,
            "phase": args.decode_phase,
            "cache_mode": args.cache_mode,
            "valid_attempts": valid_attempts,
            "invalid_attempts": invalid_attempts,
            "requested_attempts": target_valid_attempts,
            "required_passes": consensus_min_pass,
            "phase1_passes": phase1_passes,
            "hard_passes": hard_passes,
            "decode_failures": decode_failures,
            "decode_persistent_threshold": decode_persistent_threshold if decode_enabled else None,
            "decode_persistent_fail": decode_persistent_fail if decode_enabled else None,
            "file_ok": file_ok,
            "load_improve_pct_median": _median(load_improve_history)
            if load_improve_history
            else None,
            "rss_reduce_pct_median": _median(rss_reduce_history) if rss_reduce_history else None,
            "copy_load_minor_faults_median": _median(copy_minor_fault_history)
            if copy_minor_fault_history
            else None,
            "mmap_load_minor_faults_median": _median(mmap_minor_fault_history)
            if mmap_minor_fault_history
            else None,
            "copy_load_major_faults_median": _median(copy_major_fault_history)
            if copy_major_fault_history
            else None,
            "mmap_load_major_faults_median": _median(mmap_major_fault_history)
            if mmap_major_fault_history
            else None,
            "decode_regression_pct_median": _median(decode_regression_history)
            if decode_regression_history
            else None,
            "cold_cache_validation": cold_cache_validation,
            "mmap_coverage_mapped_bytes": mmap_coverage["mapped_bytes"]
            if mmap_coverage is not None
            else None,
            "mmap_coverage_copied_bytes": mmap_coverage["copied_bytes"]
            if mmap_coverage is not None
            else None,
            "mmap_coverage_mapped_ratio_pct": mmap_coverage_mapped_ratio_pct,
            "mmap_coverage_fallback_tensors": mmap_coverage["fallback_tensors"]
            if mmap_coverage is not None
            else None,
            "mmap_coverage_fallback_reasons": mmap_coverage["fallback_reasons"]
            if mmap_coverage is not None
            else None,
            "mmap_coverage": mmap_coverage,
        }

        trend_regressions: list[str] = []
        if history_path:
            file_history = [h for h in prior_history if h.get("file") == file_path]
            recent = file_history[-max(args.history_window, 1) :]

            def _check_trend(metric_key: str, higher_is_better: bool) -> None:
                current = summary_payload.get(metric_key)
                if current is None:
                    return
                values = [r.get(metric_key) for r in recent]
                values = [v for v in values if isinstance(v, (int, float))]
                if len(values) < args.trend_min_samples:
                    return
                lower, upper = _trend_bounds_mad(values, args.trend_mad_mult)
                if higher_is_better and current < lower:
                    trend_regressions.append(
                        f"{metric_key}={current:.2f} < trend_lower={lower:.2f}"
                    )
                if (not higher_is_better) and current > upper:
                    trend_regressions.append(
                        f"{metric_key}={current:.2f} > trend_upper={upper:.2f}"
                    )

            _check_trend("load_improve_pct_median", higher_is_better=True)
            _check_trend("rss_reduce_pct_median", higher_is_better=True)
            _check_trend("decode_regression_pct_median", higher_is_better=False)

            if trend_regressions:
                print("Trend warnings:")
                for warning in trend_regressions:
                    print(f"  - {warning}")
                if args.trend_hard_fail:
                    file_ok = False
                    print("Trend hard-fail enabled: failing file due to trend regression.")

            _append_history_jsonl(history_path, summary_payload)
            prior_history.append(summary_payload)

        print(
            "Consensus result: "
            f"passes={quorum_passes}/{valid_attempts} "
            f"required={consensus_min_pass} "
            f"valid_attempts={valid_attempts} invalid_attempts={invalid_attempts} "
            f"=> {file_ok}"
        )

        if not file_ok:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
