# Copyright © 2026 Apple Inc.

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import mmap
import os
import platform
import re
import resource
import shlex
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from find_benchmark_models import find_candidates


_MMAP_DEBUG_RE = re.compile(
    r"^\[io mmap\] (?P<tag>\S+) file=(?P<file>.*?) "
    r"mapped_bytes=(?P<mapped_bytes>\d+) "
    r"copied_bytes=(?P<copied_bytes>\d+) "
    r"fallback_tensors=(?P<fallback_tensors>\d+)"
    r"(?: fallback_reasons=\{(?P<fallback_reasons>[^}]*)\})?"
    r"(?: fallback_reason_bytes=\{(?P<fallback_reason_bytes>[^}]*)\})?"
    r"(?: fallback_reason_source_bytes=\{(?P<fallback_reason_source_bytes>[^}]*)\})?$"
)
_QUANTIZATION_HINT_RE = re.compile(r"(?:^|[-_.])(nvfp4|fp4|q[2-8]|iq\d*|int[48]|quant)")
_DENSE_HINT_RE = re.compile(r"(?:^|[-_.])(f16|fp16|bf16|f32|fp32|float16|float32)")
_MX = None
_ALLOWED_MMAP_PREFETCH_STRATEGIES = ("sequential", "willneed", "none")
_ROUTE_ID_BY_DISPLAY = {
    "Quantized Direct Map": "quantized_direct_map",
    "Lazy Dtype Conversion": "lazy_dtype_conversion",
    "Alignment-Friendly Mapping": "alignment_friendly_mapping",
    "Coverage-Targeted Fallback Work": "coverage_targeted_fallback_work",
    "Hotset Promotion": "hotset_promotion",
    "Adaptive Default Policy": "adaptive_default_policy",
    "Hybrid mmap Policy": "hybrid_mmap_policy",
    "Prefetch / Readahead Tuning": "prefetch_readahead_tuning",
    "Allocator / Dispatch Audit": "allocator_dispatch_audit",
    "Stability Audit": "stability_audit",
    "Broaden the Matrix": "broaden_the_matrix",
}
_ROUTE_DISPLAY_BY_ID = {value: key for key, value in _ROUTE_ID_BY_DISPLAY.items()}
_OUTCOME_CHOICES = ("auto", "win", "loss", "mixed", "inconclusive")
_DEMO_PRESET_CHOICES = (
    "history-replay",
    "adaptation-ladder",
    "golden-guardrail",
    "regression-forensics",
    "persistence-memory",
)
_COMPARISON_METRIC_SPECS = (
    {
        "key": "load_improve_pct_median",
        "label": "load improve",
        "higher_is_better": True,
    },
    {
        "key": "rss_reduce_pct_median",
        "label": "RSS reduce",
        "higher_is_better": True,
    },
    {
        "key": "load_peak_memory_reduce_pct_median",
        "label": "load peak memory reduce",
        "higher_is_better": True,
    },
    {
        "key": "total_peak_memory_reduce_pct_median",
        "label": "total peak memory reduce",
        "higher_is_better": True,
    },
    {
        "key": "load_call_improve_pct_median",
        "label": "load-call improve",
        "higher_is_better": True,
    },
    {
        "key": "parse_improve_pct_median",
        "label": "parse improve",
        "higher_is_better": True,
    },
    {
        "key": "tensor_setup_improve_pct_median",
        "label": "tensor-setup improve",
        "higher_is_better": True,
    },
    {
        "key": "first_eval_improve_pct_median",
        "label": "first-eval improve",
        "higher_is_better": True,
    },
    {
        "key": "decode_regression_pct_median",
        "label": "steady decode regression",
        "higher_is_better": False,
    },
    {
        "key": "decode_first_token_regression_pct_median",
        "label": "first-token regression",
        "higher_is_better": False,
    },
)
_COMPARISON_METRIC_BY_KEY = {
    spec["key"]: spec for spec in _COMPARISON_METRIC_SPECS
}
_DEMO_POLICY_SCORE_WEIGHTS = {
    "load_improve_pct_median": 1.0,
    "rss_reduce_pct_median": 0.8,
    "load_peak_memory_reduce_pct_median": 0.4,
    "total_peak_memory_reduce_pct_median": 0.3,
    "load_call_improve_pct_median": 0.5,
    "parse_improve_pct_median": 0.25,
    "tensor_setup_improve_pct_median": 0.25,
    "first_eval_improve_pct_median": 0.4,
    "decode_regression_pct_median": -1.0,
    "decode_first_token_regression_pct_median": -1.1,
}


def _get_mx():
    global _MX
    if _MX is None:
        import mlx.core as mx

        _MX = mx
    return _MX


def _safe_check_output(cmd: list[str], timeout: float = 2.0) -> str | None:
    try:
        out = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        ).strip()
    except Exception:
        return None
    return out or None


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


def _parse_reason_counts(raw: str | None) -> dict[str, int]:
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
            "fallback_reasons": _parse_reason_counts(
                match.group("fallback_reasons")
            ),
            "fallback_reason_bytes": _parse_reason_counts(
                match.group("fallback_reason_bytes")
            ),
            "fallback_reason_source_bytes": _parse_reason_counts(
                match.group("fallback_reason_source_bytes")
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
) -> dict:
    mx = _get_mx()
    if not arrays:
        return {
            "decode_tok_s": 0.0,
            "decode_first_token_s": None,
            "decode_first_token_minor_faults": None,
            "decode_first_token_major_faults": None,
            "decode_steady_minor_faults": None,
            "decode_steady_major_faults": None,
        }

    target = max(arrays, key=lambda arr: int(arr.size))
    flat = mx.reshape(target, (-1,))
    elems = min(int(flat.size), max(max_elems, 1))
    block = flat[:elems]

    first_faults_before = _fault_counts()
    first_tic = time.perf_counter()
    mx.eval(mx.sum(block))
    mx.synchronize()
    first_toc = time.perf_counter()
    first_faults_after = _fault_counts()

    steps = max(tokens - 1, 1)
    rates: list[float] = []
    steady_faults_before = _fault_counts()
    for _ in range(max(repeats, 1)):
        tic = time.perf_counter()
        for _ in range(steps):
            mx.eval(mx.sum(block))
        mx.synchronize()
        toc = time.perf_counter()
        elapsed = toc - tic
        rates.append(steps / elapsed if elapsed > 0 else 0.0)
    steady_faults_after = _fault_counts()
    return {
        "decode_tok_s": statistics.median(rates),
        "decode_first_token_s": first_toc - first_tic,
        "decode_first_token_minor_faults": first_faults_after[0]
        - first_faults_before[0],
        "decode_first_token_major_faults": first_faults_after[1]
        - first_faults_before[1],
        "decode_steady_minor_faults": steady_faults_after[0] - steady_faults_before[0],
        "decode_steady_major_faults": steady_faults_after[1] - steady_faults_before[1],
    }


def _run_worker(
    path: str,
    fmt: str | None,
    memory_map: bool,
    gguf_nvfp4_compat: bool,
    mmap_small_tensor_copy_max_bytes: int | None,
    mmap_hotset_promotion_top_k: int | None,
    mmap_hotset_promotion_min_bytes: int | None,
    mmap_prefetch_strategy: str,
    decode_synth_tokens: int,
    decode_synth_max_elems: int,
    decode_synth_repeats: int,
    worker_prime_loads: int,
) -> dict:
    mx = _get_mx()
    mx.clear_cache()
    last_mmap_load_stats = getattr(mx, "last_mmap_load_stats", None)
    last_load_phase_stats = getattr(mx, "last_load_phase_stats", None)
    get_peak_memory = getattr(mx, "get_peak_memory", None)
    if callable(last_mmap_load_stats):
        last_mmap_load_stats(clear=True)
    if callable(last_load_phase_stats):
        last_load_phase_stats(clear=True)

    for _ in range(max(worker_prime_loads, 0)):
        primed = mx.load(
            path,
            format=fmt,
            memory_map=memory_map,
            return_metadata=True,
            gguf_nvfp4_compat=gguf_nvfp4_compat,
            mmap_small_tensor_copy_max_bytes=mmap_small_tensor_copy_max_bytes,
            mmap_hotset_promotion_top_k=mmap_hotset_promotion_top_k,
            mmap_hotset_promotion_min_bytes=mmap_hotset_promotion_min_bytes,
            mmap_prefetch_strategy=mmap_prefetch_strategy,
        )
        primed_arrays = _to_array_list(primed)
        mx.eval(primed_arrays)
        mx.synchronize()
        del primed_arrays
        del primed
        gc.collect()
        mx.clear_cache()

    if callable(last_mmap_load_stats):
        last_mmap_load_stats(clear=True)
    if callable(last_load_phase_stats):
        last_load_phase_stats(clear=True)
    mx.reset_peak_memory()
    load_faults_before = _fault_counts()

    load_tic = time.perf_counter()
    loaded = mx.load(
        path,
        format=fmt,
        memory_map=memory_map,
        return_metadata=True,
        gguf_nvfp4_compat=gguf_nvfp4_compat,
        mmap_small_tensor_copy_max_bytes=mmap_small_tensor_copy_max_bytes,
        mmap_hotset_promotion_top_k=mmap_hotset_promotion_top_k,
        mmap_hotset_promotion_min_bytes=mmap_hotset_promotion_min_bytes,
        mmap_prefetch_strategy=mmap_prefetch_strategy,
    )
    load_toc = time.perf_counter()
    load_call_faults_after = _fault_counts()
    load_phase_stats = None
    if callable(last_load_phase_stats):
        load_phase_stats = last_load_phase_stats(clear=True)
    array_list = _to_array_list(loaded)

    eval_tic = time.perf_counter()
    mx.eval(array_list)
    mx.synchronize()
    eval_toc = time.perf_counter()
    load_faults_after = _fault_counts()
    load_peak_memory_bytes = (
        int(get_peak_memory()) if callable(get_peak_memory) else None
    )

    result = {
        "elapsed_s": (load_toc - load_tic) + (eval_toc - eval_tic),
        "load_call_s": load_toc - load_tic,
        "first_eval_s": eval_toc - eval_tic,
        "peak_rss_bytes": _rss_bytes_from_ru_maxrss(),
        "load_minor_faults": load_faults_after[0] - load_faults_before[0],
        "load_major_faults": load_faults_after[1] - load_faults_before[1],
        "load_call_minor_faults": load_call_faults_after[0] - load_faults_before[0],
        "load_call_major_faults": load_call_faults_after[1] - load_faults_before[1],
        "first_eval_minor_faults": load_faults_after[0] - load_call_faults_after[0],
        "first_eval_major_faults": load_faults_after[1] - load_call_faults_after[1],
        "load_peak_memory_bytes": load_peak_memory_bytes,
    }
    if load_phase_stats is not None:
        result["load_phase_stats"] = load_phase_stats
    if callable(last_mmap_load_stats):
        mmap_stats = last_mmap_load_stats(clear=True)
        if mmap_stats is not None:
            result["mmap_stats"] = mmap_stats
    if decode_synth_tokens > 0:
        result.update(
            _run_synth_decode(
            arrays=array_list,
            tokens=decode_synth_tokens,
            max_elems=decode_synth_max_elems,
            repeats=decode_synth_repeats,
            )
        )
    total_faults_after = _fault_counts()
    result["total_minor_faults"] = total_faults_after[0] - load_faults_before[0]
    result["total_major_faults"] = total_faults_after[1] - load_faults_before[1]
    result["total_peak_memory_bytes"] = (
        int(get_peak_memory()) if callable(get_peak_memory) else None
    )
    return result


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _median_optional(values: list[int | float | None]) -> int | float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return statistics.median(present)


def _optional_trial_metric(
    trials: list[dict], key: str
) -> list[int | float | None]:
    return [trial.get(key) for trial in trials]


def _optional_nested_trial_metric(
    trials: list[dict], parent_key: str, key: str
) -> list[int | float | None]:
    values: list[int | float | None] = []
    for trial in trials:
        parent = trial.get(parent_key)
        if isinstance(parent, dict):
            values.append(parent.get(key))
        else:
            values.append(None)
    return values


def _improvement_pct(copy_value: float, mmap_value: float) -> float:
    return 100.0 * (copy_value - mmap_value) / copy_value if copy_value > 0 else 0.0


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
                regressions.append(_improvement_pct(c, m))
        if regressions:
            return _median(regressions)
        return 0.0
    copy_med = _median(copy_values)
    mmap_med = _median(mmap_values)
    return _improvement_pct(copy_med, mmap_med)


def _median_latency_regression_pct(
    copy_values: list[float], mmap_values: list[float], pairwise: bool
) -> float:
    if pairwise:
        regressions = []
        for c, m in zip(copy_values, mmap_values):
            if c > 0:
                regressions.append(100.0 * (m - c) / c)
        if regressions:
            return _median(regressions)
        return 0.0
    copy_med = _median(copy_values)
    mmap_med = _median(mmap_values)
    return 100.0 * (mmap_med - copy_med) / copy_med if copy_med > 0 else 0.0


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


def _slugify_identifier(raw: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower())
    return token.strip("_") or "unknown"


def _route_id_for_display(route: str) -> str:
    return _ROUTE_ID_BY_DISPLAY.get(route, _slugify_identifier(route))


def _normalize_route_reference(raw: str | None) -> tuple[str, str] | None:
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None
    if token in _ROUTE_DISPLAY_BY_ID:
        return token, _ROUTE_DISPLAY_BY_ID[token]
    if token in _ROUTE_ID_BY_DISPLAY:
        return _ROUTE_ID_BY_DISPLAY[token], token
    route_id = _slugify_identifier(token)
    return route_id, _ROUTE_DISPLAY_BY_ID.get(route_id, token)


def _normalize_outcome_label(raw: str | None) -> str:
    token = str(raw or "auto").strip().lower().replace("-", "_")
    if token not in _OUTCOME_CHOICES:
        raise ValueError(
            "--route-outcome must be one of " + ", ".join(_OUTCOME_CHOICES)
        )
    return token


def _extract_code_state(environment_metadata: dict | None) -> dict:
    environment_metadata = environment_metadata or {}
    return {
        "git_head": environment_metadata.get("git_head"),
        "git_head_short": environment_metadata.get("git_head_short"),
        "git_branch": environment_metadata.get("git_branch"),
        "git_dirty": environment_metadata.get("git_dirty"),
    }


def _make_run_id(
    file_path: str,
    *,
    when: dt.datetime | None = None,
    git_head_short: str | None = None,
) -> str:
    when = when or dt.datetime.now(dt.timezone.utc)
    stamp = when.strftime("%Y%m%dT%H%M%S.%fZ")
    stem = _slugify_identifier(Path(file_path).stem)[:24]
    head = _slugify_identifier(git_head_short or "unknown")[:12]
    return f"{stamp}-{head}-{stem}-p{os.getpid()}"


def _parse_optional_byte_threshold(raw: str) -> int | None:
    token = str(raw).strip().lower()
    if token in {"", "none", "default", "off"}:
        return None
    value = int(token)
    if value < 0:
        raise ValueError("byte threshold must be >= 0")
    return value


def _parse_optional_byte_threshold_list(raw: str | None) -> list[int | None]:
    if raw is None:
        return []
    values: list[int | None] = []
    seen: set[int | None] = set()
    for token in str(raw).split(","):
        value = _parse_optional_byte_threshold(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _normalize_prefetch_strategy(raw: str) -> str:
    strategy = str(raw).strip().lower()
    if strategy not in _ALLOWED_MMAP_PREFETCH_STRATEGIES:
        raise ValueError(
            "prefetch strategy must be one of "
            + ", ".join(_ALLOWED_MMAP_PREFETCH_STRATEGIES)
        )
    return strategy


def _parse_prefetch_strategy_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for token in str(raw).split(","):
        value = _normalize_prefetch_strategy(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _normalize_hotset_promotion_top_k(raw: int | None) -> int | None:
    if raw is None:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError("hotset promotion top-k must be >= 0")
    return value if value > 0 else None


def _normalize_hotset_promotion_min_bytes(raw: int | None) -> int | None:
    if raw is None:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError("hotset promotion minimum bytes must be >= 0")
    return value if value > 0 else None


def _parse_optional_nonnegative_int(raw: str) -> int | None:
    token = str(raw).strip().lower()
    if token in {"", "none", "default", "off"}:
        return None
    value = int(token)
    if value < 0:
        raise ValueError("value must be >= 0")
    return value if value > 0 else None


def _parse_optional_nonnegative_int_list(raw: str | None) -> list[int | None]:
    if raw is None:
        return []
    values: list[int | None] = []
    seen: set[int | None] = set()
    for token in str(raw).split(","):
        value = _parse_optional_nonnegative_int(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _parse_hotset_promotion_min_bytes_list(raw: str | None) -> list[int | None]:
    if raw is None:
        return []
    values: list[int | None] = []
    seen: set[int | None] = set()
    for token in str(raw).split(","):
        value = _normalize_hotset_promotion_min_bytes(
            _parse_optional_byte_threshold(token)
        )
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _format_byte_threshold(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "default"
    if num_bytes >= 1024**2 and num_bytes % (1024**2) == 0:
        return f"{num_bytes // (1024**2)}MiB"
    if num_bytes >= 1024 and num_bytes % 1024 == 0:
        return f"{num_bytes // 1024}KiB"
    return f"{num_bytes}B"


def _build_policy_knobs(
    mmap_small_tensor_copy_max_bytes: int | None,
    mmap_prefetch_strategy: str = "sequential",
    mmap_hotset_promotion_top_k: int | None = None,
    mmap_hotset_promotion_min_bytes: int | None = None,
) -> dict:
    top_k = _normalize_hotset_promotion_top_k(mmap_hotset_promotion_top_k)
    min_bytes = _normalize_hotset_promotion_min_bytes(
        mmap_hotset_promotion_min_bytes
    )
    return {
        "mmap_small_tensor_copy_max_bytes": (
            int(mmap_small_tensor_copy_max_bytes)
            if mmap_small_tensor_copy_max_bytes is not None
            else None
        ),
        "mmap_hotset_promotion_top_k": top_k,
        "mmap_hotset_promotion_min_bytes": (
            int(min_bytes) if top_k is not None and min_bytes is not None else None
        ),
        "mmap_prefetch_strategy": _normalize_prefetch_strategy(mmap_prefetch_strategy),
    }


def _policy_signature(policy_knobs: dict | None) -> str:
    policy_knobs = policy_knobs or {}
    strategy = _normalize_prefetch_strategy(
        str(policy_knobs.get("mmap_prefetch_strategy") or "sequential")
    )
    threshold = policy_knobs.get("mmap_small_tensor_copy_max_bytes")
    hotset_top_k = _normalize_hotset_promotion_top_k(
        policy_knobs.get("mmap_hotset_promotion_top_k")
    )
    hotset_min_bytes = _normalize_hotset_promotion_min_bytes(
        policy_knobs.get("mmap_hotset_promotion_min_bytes")
    )
    parts: list[str] = []
    if strategy != "sequential":
        parts.append(f"prefetch={strategy}")
    if threshold is not None:
        parts.append(f"small_tensor_copy_lte={int(threshold)}B")
    if hotset_top_k is not None:
        parts.append(f"hotset_topk={int(hotset_top_k)}")
        if hotset_min_bytes is not None:
            parts.append(f"hotset_min_bytes={int(hotset_min_bytes)}B")
    if not parts:
        return "default"
    return ",".join(parts)


def _policy_summary(policy_knobs: dict | None) -> str:
    policy_knobs = policy_knobs or {}
    strategy = _normalize_prefetch_strategy(
        str(policy_knobs.get("mmap_prefetch_strategy") or "sequential")
    )
    threshold = policy_knobs.get("mmap_small_tensor_copy_max_bytes")
    hotset_top_k = _normalize_hotset_promotion_top_k(
        policy_knobs.get("mmap_hotset_promotion_top_k")
    )
    hotset_min_bytes = _normalize_hotset_promotion_min_bytes(
        policy_knobs.get("mmap_hotset_promotion_min_bytes")
    )
    parts: list[str] = []
    if strategy != "sequential":
        parts.append(f"mmap prefetch {strategy}")
    if threshold is not None:
        parts.append(
            "copy mapped-eligible tensors at or below "
            f"{_format_byte_threshold(int(threshold))}"
        )
    if hotset_top_k is not None:
        hotset_summary = (
            f"promote top {int(hotset_top_k)} mapped-eligible tensor"
            f"{'' if int(hotset_top_k) == 1 else 's'}"
        )
        if hotset_min_bytes is not None:
            hotset_summary += (
                f" at or above {_format_byte_threshold(int(hotset_min_bytes))}"
            )
        hotset_summary += " into owned buffers"
        parts.append(hotset_summary)
    if not parts:
        return "default mmap policy"
    return "; ".join(parts)


def _classify_effective_policy_mode(
    memory_map: bool, policy_knobs: dict | None
) -> str:
    if not memory_map:
        return "copy"
    return "mapped" if _policy_signature(policy_knobs) == "default" else "hybrid"


def _effective_policy_signature(
    *, memory_map: bool, policy_knobs: dict | None
) -> str:
    mode = _classify_effective_policy_mode(memory_map, policy_knobs)
    if mode == "copy":
        return "copy"
    return f"{mode}:{_policy_signature(policy_knobs)}"


def _effective_policy_summary(
    *, memory_map: bool, policy_knobs: dict | None
) -> str:
    mode = _classify_effective_policy_mode(memory_map, policy_knobs)
    if mode == "copy":
        return "copy-only policy"
    if mode == "mapped":
        return "mapped default policy"
    return _policy_summary(policy_knobs)


def _build_policy_context(
    *,
    fmt: str | None,
    model_class: str,
    cache_mode: str,
    decode_mode: str,
) -> dict[str, str]:
    return {
        "format": str(fmt or "unknown"),
        "model_class": str(model_class or "unknown"),
        "cache_mode": str(cache_mode or "inherit"),
        "decode_mode": str(decode_mode or "none"),
    }


def _policy_context_key(context: dict[str, str]) -> str:
    return "|".join(
        [
            f"format={context['format']}",
            f"class={context['model_class']}",
            f"cache={context['cache_mode']}",
            f"decode={context['decode_mode']}",
        ]
    )


def _record_policy_context(record: dict) -> dict[str, str]:
    return _build_policy_context(
        fmt=record.get("format"),
        model_class=str(record.get("model_class") or _record_model_class(record)),
        cache_mode=str(record.get("cache_mode") or "inherit"),
        decode_mode=str(record.get("decode_mode") or "none"),
    )


def _matches_policy_context(record: dict, context: dict[str, str]) -> bool:
    return _record_policy_context(record) == context


def _record_effective_policy_mode(record: dict) -> str:
    mode = record.get("effective_policy_mode")
    if isinstance(mode, str) and mode:
        return mode
    memory_map = record.get("effective_memory_map")
    if memory_map is False:
        return "copy"
    policy_knobs = record.get("effective_policy_knobs") or record.get("policy_knobs")
    return _classify_effective_policy_mode(True, policy_knobs)


def _record_effective_policy_signature(record: dict) -> str:
    signature = record.get("effective_policy_signature")
    if isinstance(signature, str) and signature:
        return signature
    mode = _record_effective_policy_mode(record)
    if mode == "copy":
        return "copy"
    policy_knobs = record.get("effective_policy_knobs") or record.get("policy_knobs")
    return f"{mode}:{record.get('policy_signature') or _policy_signature(policy_knobs)}"


def _policy_candidate_definitions(cache_mode: str) -> list[dict]:
    candidates = [
        {
            "name": "copy",
            "memory_map": False,
            "policy_knobs": _build_policy_knobs(None),
            "label": "Copy Only",
        },
        {
            "name": "mapped_default",
            "memory_map": True,
            "policy_knobs": _build_policy_knobs(None),
            "label": "Mapped Default",
        },
        {
            "name": "hybrid_small",
            "memory_map": True,
            "policy_knobs": _build_policy_knobs(65536),
            "label": "Hybrid Small-Tensor",
        },
        {
            "name": "hybrid_hotset",
            "memory_map": True,
            "policy_knobs": _build_policy_knobs(
                None,
                mmap_hotset_promotion_top_k=2,
                mmap_hotset_promotion_min_bytes=32 * 1024 * 1024,
            ),
            "label": "Hybrid Hotset",
        },
        {
            "name": "hybrid_balanced",
            "memory_map": True,
            "policy_knobs": _build_policy_knobs(
                65536,
                mmap_hotset_promotion_top_k=2,
                mmap_hotset_promotion_min_bytes=32 * 1024 * 1024,
            ),
            "label": "Hybrid Balanced",
        },
    ]
    if cache_mode == "cold-best-effort":
        candidates.extend(
            [
                {
                    "name": "mapped_willneed",
                    "memory_map": True,
                    "policy_knobs": _build_policy_knobs(
                        None, mmap_prefetch_strategy="willneed"
                    ),
                    "label": "Mapped WILLNEED",
                },
                {
                    "name": "hybrid_balanced_willneed",
                    "memory_map": True,
                    "policy_knobs": _build_policy_knobs(
                        65536,
                        mmap_prefetch_strategy="willneed",
                        mmap_hotset_promotion_top_k=2,
                        mmap_hotset_promotion_min_bytes=32 * 1024 * 1024,
                    ),
                    "label": "Hybrid Balanced WILLNEED",
                },
            ]
        )
    for candidate in candidates:
        candidate["effective_mode"] = _classify_effective_policy_mode(
            candidate["memory_map"], candidate["policy_knobs"]
        )
        candidate["effective_policy_signature"] = _effective_policy_signature(
            memory_map=candidate["memory_map"], policy_knobs=candidate["policy_knobs"]
        )
        candidate["effective_policy_summary"] = _effective_policy_summary(
            memory_map=candidate["memory_map"], policy_knobs=candidate["policy_knobs"]
        )
    return candidates


def _outcome_score(outcome: str | None) -> float:
    normalized = _normalize_outcome_label(outcome or "inconclusive")
    return {
        "win": 1.0,
        "mixed": 0.35,
        "inconclusive": 0.0,
        "loss": -1.0,
        "auto": 0.0,
    }.get(normalized, 0.0)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _decode_priority_weights(
    cache_mode: str, decode_mode: str
) -> tuple[float, float]:
    if decode_mode == "none":
        return 0.0, 0.0
    if cache_mode in {"warm", "steady-state"}:
        return 0.65, 1.5
    if cache_mode == "cold-best-effort":
        return 0.45, 0.8
    return 0.5, 1.0


def _build_auto_policy_probe_decode_summary(probe_trial: dict | None) -> dict | None:
    if not isinstance(probe_trial, dict):
        return None
    decode_tok_s = probe_trial.get("decode_tok_s")
    decode_first_token_s = probe_trial.get("decode_first_token_s")
    if not isinstance(decode_first_token_s, (int, float)) and not isinstance(
        decode_tok_s, (int, float)
    ):
        return None
    load_call_s = probe_trial.get("load_call_s")
    first_eval_s = probe_trial.get("first_eval_s")
    first_token_minor = probe_trial.get("decode_first_token_minor_faults")
    first_token_major = probe_trial.get("decode_first_token_major_faults")
    steady_minor = probe_trial.get("decode_steady_minor_faults")
    steady_major = probe_trial.get("decode_steady_major_faults")
    first_token_faults = int(first_token_minor or 0) + int(first_token_major or 0)
    steady_faults = int(steady_minor or 0) + int(steady_major or 0)
    first_token_vs_load_pct = None
    if isinstance(load_call_s, (int, float)) and load_call_s > 0 and isinstance(
        decode_first_token_s, (int, float)
    ):
        first_token_vs_load_pct = 100.0 * float(decode_first_token_s) / float(load_call_s)
    first_token_vs_eval_pct = None
    if isinstance(first_eval_s, (int, float)) and first_eval_s > 0 and isinstance(
        decode_first_token_s, (int, float)
    ):
        first_token_vs_eval_pct = 100.0 * float(decode_first_token_s) / float(first_eval_s)
    return {
        "decode_tok_s": float(decode_tok_s) if isinstance(decode_tok_s, (int, float)) else None,
        "decode_first_token_s": (
            float(decode_first_token_s)
            if isinstance(decode_first_token_s, (int, float))
            else None
        ),
        "load_call_s": float(load_call_s) if isinstance(load_call_s, (int, float)) else None,
        "first_eval_s": float(first_eval_s) if isinstance(first_eval_s, (int, float)) else None,
        "first_token_minor_faults": (
            int(first_token_minor) if isinstance(first_token_minor, (int, float)) else None
        ),
        "first_token_major_faults": (
            int(first_token_major) if isinstance(first_token_major, (int, float)) else None
        ),
        "steady_minor_faults": (
            int(steady_minor) if isinstance(steady_minor, (int, float)) else None
        ),
        "steady_major_faults": (
            int(steady_major) if isinstance(steady_major, (int, float)) else None
        ),
        "first_token_faults": first_token_faults,
        "steady_faults": steady_faults,
        "first_token_vs_load_pct": (
            round(first_token_vs_load_pct, 2)
            if isinstance(first_token_vs_load_pct, (int, float))
            else None
        ),
        "first_token_vs_eval_pct": (
            round(first_token_vs_eval_pct, 2)
            if isinstance(first_token_vs_eval_pct, (int, float))
            else None
        ),
    }


def _auto_policy_from_history(
    candidate: dict,
    *,
    matching_records: list[dict],
    same_file_records: list[dict],
    route_stats: dict[str, dict],
    cache_mode: str,
    decode_mode: str,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    steady_decode_weight, first_token_weight = _decode_priority_weights(
        cache_mode, decode_mode
    )
    if matching_records:
        outcome_values = [
            _outcome_score(
                record.get("route_outcome")
                or (record.get("comparison") or {}).get("auto_outcome")
            )
            for record in matching_records
        ]
        avg_outcome = _mean_or_none(outcome_values)
        if avg_outcome is not None:
            score += 35.0 * avg_outcome
            reasons.append(
                f"history outcome avg={avg_outcome:+.2f} across {len(matching_records)} run(s)"
            )
        load_values = [
            float(record.get("load_improve_pct_median"))
            for record in matching_records
            if isinstance(record.get("load_improve_pct_median"), (int, float))
        ]
        rss_values = [
            float(record.get("rss_reduce_pct_median"))
            for record in matching_records
            if isinstance(record.get("rss_reduce_pct_median"), (int, float))
        ]
        decode_values = [
            float(record.get("decode_regression_pct_median"))
            for record in matching_records
            if isinstance(record.get("decode_regression_pct_median"), (int, float))
        ]
        first_token_values = [
            float(record.get("decode_first_token_regression_pct_median"))
            for record in matching_records
            if isinstance(
                record.get("decode_first_token_regression_pct_median"), (int, float)
            )
        ]
        if load_values:
            load_med = float(_median(load_values))
            score += 0.4 * load_med
            reasons.append(f"history load improve median={load_med:.1f}%")
        if rss_values:
            rss_med = float(_median(rss_values))
            score += 0.3 * rss_med
            reasons.append(f"history RSS reduce median={rss_med:.1f}%")
        if decode_values:
            decode_med = float(_median(decode_values))
            score -= steady_decode_weight * decode_med
            reasons.append(f"history decode regression median={decode_med:.1f}%")
        if first_token_values:
            first_token_med = float(_median(first_token_values))
            score -= first_token_weight * first_token_med
            reasons.append(
                f"history first-token regression median={first_token_med:.1f}%"
            )
    if same_file_records:
        score += 8.0
        reasons.append(f"same-file support={len(same_file_records)} run(s)")

    if candidate["effective_mode"] == "hybrid":
        for route_id in ("hotset_promotion", "hybrid_mmap_policy"):
            stats = route_stats.get(route_id)
            if not stats:
                continue
            success_rate = stats.get("success_rate")
            if isinstance(success_rate, (int, float)):
                boost = (float(success_rate) - 0.5) * 16.0
                score += boost
                reasons.append(
                    f"route {route_id} hit_rate={100.0 * float(success_rate):.1f}%"
                )
        if decode_mode != "none" and cache_mode in {"warm", "steady-state"}:
            stats = route_stats.get("hotset_promotion")
            if stats:
                success_rate = stats.get("success_rate")
                if isinstance(success_rate, (int, float)):
                    boost = (float(success_rate) - 0.5) * 20.0
                    score += boost
                    reasons.append(
                        "decode-aware hotset route "
                        f"hit_rate={100.0 * float(success_rate):.1f}%"
                    )
    if candidate["name"].endswith("willneed"):
        stats = route_stats.get("prefetch_readahead_tuning")
        if stats:
            success_rate = stats.get("success_rate")
            if isinstance(success_rate, (int, float)):
                boost = (float(success_rate) - 0.5) * 16.0
                score += boost
                reasons.append(
                    f"route prefetch_readahead_tuning hit_rate={100.0 * float(success_rate):.1f}%"
                )
    return score, reasons


def _auto_policy_from_probe(
    candidate: dict,
    *,
    model_metadata: dict,
    cache_mode: str,
    decode_mode: str,
    probe_stats: dict | None,
    probe_decode_summary: dict | None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    fmt = str(model_metadata.get("format") or "unknown")
    model_class = str(model_metadata.get("model_class") or "unknown")
    weight_class = str(model_metadata.get("weight_class") or "unknown")
    mapped_ratio = None
    fallback_tensors = 0
    quantized_share = 0.0
    copied_bytes = 0
    if probe_stats is not None:
        mapped = int(probe_stats.get("mapped_bytes") or 0)
        copied_bytes = int(probe_stats.get("copied_bytes") or 0)
        total = mapped + copied_bytes
        mapped_ratio = (100.0 * mapped / total) if total > 0 else 0.0
        fallback_tensors = int(probe_stats.get("fallback_tensors") or 0)
        fallback_reason_bytes = probe_stats.get("fallback_reason_bytes") or {}
        quantized_bytes = int(fallback_reason_bytes.get("quantized_conversion") or 0)
        if copied_bytes > 0:
            quantized_share = quantized_bytes / copied_bytes
    first_token_ratio = None
    first_token_faults = 0
    steady_faults = 0
    if probe_decode_summary is not None:
        first_token_ratio = probe_decode_summary.get("first_token_vs_load_pct")
        first_token_faults = int(probe_decode_summary.get("first_token_faults") or 0)
        steady_faults = int(probe_decode_summary.get("steady_faults") or 0)

    if candidate["effective_mode"] == "copy":
        if mapped_ratio is not None and mapped_ratio < 35.0:
            score += 24.0
            reasons.append(f"probe mapped_ratio={mapped_ratio:.1f}% is low")
        if quantized_share >= 0.6:
            score += 20.0
            reasons.append(
                f"quantized fallback share={100.0 * quantized_share:.1f}% of copied bytes"
            )
        if cache_mode == "warm" and decode_mode != "none":
            score += 8.0
            reasons.append("warm decode path favors avoiding risky mmap regressions")
        if isinstance(first_token_ratio, (int, float)) and first_token_ratio >= 50.0:
            score += min(10.0, 0.12 * float(first_token_ratio))
            reasons.append(
                f"probe first-token/load ratio={float(first_token_ratio):.1f}% is high"
            )
        if first_token_faults >= 4:
            score += min(8.0, 1.25 * float(first_token_faults))
            reasons.append(f"probe first-token faults={first_token_faults}")
        if fmt == "safetensors" and weight_class == "dense" and mapped_ratio is not None and mapped_ratio >= 90.0:
            score -= 18.0
    elif candidate["effective_mode"] == "mapped":
        if fmt == "safetensors":
            score += 16.0
            reasons.append("safetensors generally maps cleanly")
        if weight_class == "dense":
            score += 10.0
            reasons.append("dense weights favor direct mapped views")
        if cache_mode == "cold-best-effort":
            score += 10.0
            reasons.append("cold cache favors mapped startup paths")
        if mapped_ratio is not None and mapped_ratio >= 90.0:
            score += 20.0
            reasons.append(f"probe mapped_ratio={mapped_ratio:.1f}% is high")
        if mapped_ratio is not None and mapped_ratio < 50.0:
            score -= 16.0
        if quantized_share >= 0.5:
            score -= 18.0
        if (
            decode_mode != "none"
            and cache_mode in {"warm", "steady-state"}
            and probe_decode_summary is None
        ):
            score -= 30.0
            reasons.append(
                "warm decode has no direct probe signal yet, so mapped default stays conservative"
            )
        if isinstance(first_token_ratio, (int, float)) and first_token_ratio >= 40.0:
            penalty = min(18.0, 0.22 * float(first_token_ratio))
            score -= penalty
            reasons.append(
                f"probe first-token/load ratio={float(first_token_ratio):.1f}% looks costly"
            )
        if first_token_faults >= 4:
            penalty = min(12.0, 1.5 * float(first_token_faults))
            score -= penalty
            reasons.append(f"probe first-token faults={first_token_faults}")
        if steady_faults >= 6:
            score -= min(8.0, float(steady_faults))
            reasons.append(f"probe steady decode faults={steady_faults}")
        if (
            decode_mode != "none"
            and isinstance(first_token_ratio, (int, float))
            and float(first_token_ratio) <= 20.0
            and first_token_faults <= 1
            and steady_faults <= 2
        ):
            score += 6.0
            reasons.append("probe decode path looks light enough for direct mapping")
    else:
        if mapped_ratio is not None and 40.0 <= mapped_ratio < 90.0:
            score += 16.0
            reasons.append(f"probe mapped_ratio={mapped_ratio:.1f}% suggests hybrid room")
        if fallback_tensors >= 8:
            score += 8.0
            reasons.append(f"probe fallback_tensors={fallback_tensors}")
        if decode_mode != "none" and cache_mode in {"warm", "steady-state"}:
            score += 10.0
            reasons.append("decode-aware cache mode favors some promotion")
        if candidate["name"] == "hybrid_small" and copied_bytes > 0 and fallback_tensors >= 16:
            score += 6.0
        if candidate["name"] in {"hybrid_hotset", "hybrid_balanced", "hybrid_balanced_willneed"}:
            if model_class == "quantized_gguf":
                score += 8.0
                reasons.append("quantized GGUF often benefits from hybrid tuning")
        hotset_enabled = (
            candidate["policy_knobs"].get("mmap_hotset_promotion_top_k") is not None
        )
        if hotset_enabled and isinstance(first_token_ratio, (int, float)) and first_token_ratio >= 35.0:
            boost = min(18.0, 0.2 * float(first_token_ratio))
            score += boost
            reasons.append(
                "probe first-token/load ratio suggests targeted hotset promotion"
            )
        if hotset_enabled and first_token_faults >= 4:
            boost = min(12.0, 1.5 * float(first_token_faults))
            score += boost
            reasons.append(f"probe first-token faults={first_token_faults}")
        if (
            candidate["name"] == "hybrid_hotset"
            and hotset_enabled
            and isinstance(mapped_ratio, (int, float))
            and mapped_ratio >= 95.0
            and fallback_tensors <= 4
            and isinstance(first_token_ratio, (int, float))
            and first_token_ratio >= 35.0
        ):
            score += 3.0
            reasons.append("mapped coverage is already high, so hotset-only promotion is cleaner")
        if (
            candidate["name"] == "hybrid_balanced"
            and isinstance(mapped_ratio, (int, float))
            and mapped_ratio >= 95.0
            and fallback_tensors <= 4
            and isinstance(first_token_ratio, (int, float))
            and first_token_ratio >= 35.0
        ):
            score -= 2.0
            reasons.append("small-tensor copying looks broader than the current first-touch issue")
        if candidate["name"] == "hybrid_small" and steady_faults >= 6:
            score += min(8.0, float(steady_faults))
            reasons.append(f"probe steady decode faults={steady_faults}")
        if candidate["name"].endswith("willneed") and cache_mode == "cold-best-effort":
            score += 8.0
            reasons.append("cold cache may benefit from WILLNEED prefetch")
            if first_token_faults >= 4:
                score += 4.0
                reasons.append("probe first-touch faults strengthen the WILLNEED case")
        if mapped_ratio is not None and mapped_ratio >= 95.0:
            score -= 10.0
        if (
            hotset_enabled
            and isinstance(first_token_ratio, (int, float))
            and float(first_token_ratio) <= 20.0
            and first_token_faults == 0
        ):
            score -= 4.0

    return score, reasons


def _choose_auto_policy(
    *,
    file_path: str,
    fmt: str | None,
    model_metadata: dict,
    cache_mode: str,
    decode_mode: str,
    prior_history: list[dict],
    probe_stats: dict | None,
    probe_trial: dict | None = None,
) -> dict:
    context = _build_policy_context(
        fmt=fmt,
        model_class=str(model_metadata.get("model_class") or "unknown"),
        cache_mode=cache_mode,
        decode_mode=decode_mode,
    )
    context_records = [
        record for record in prior_history if _matches_policy_context(record, context)
    ]
    route_stats = _route_outcome_stats(context_records)
    probe_decode_summary = _build_auto_policy_probe_decode_summary(probe_trial)
    candidates: list[dict] = []
    for candidate in _policy_candidate_definitions(cache_mode):
        candidate_records = [
            record
            for record in context_records
            if _record_effective_policy_signature(record)
            == candidate["effective_policy_signature"]
        ]
        same_file_records = [
            record for record in candidate_records if str(record.get("file") or "") == file_path
        ]
        history_score, history_reasons = _auto_policy_from_history(
            candidate,
            matching_records=candidate_records,
            same_file_records=same_file_records,
            route_stats=route_stats,
            cache_mode=cache_mode,
            decode_mode=decode_mode,
        )
        heuristic_score, heuristic_reasons = _auto_policy_from_probe(
            candidate,
            model_metadata=model_metadata,
            cache_mode=cache_mode,
            decode_mode=decode_mode,
            probe_stats=probe_stats,
            probe_decode_summary=probe_decode_summary,
        )
        score = history_score + heuristic_score
        candidates.append(
            {
                **candidate,
                "score": round(score, 2),
                "history_score": round(history_score, 2),
                "heuristic_score": round(heuristic_score, 2),
                "history_support": len(candidate_records),
                "same_file_support": len(same_file_records),
                "reasons": history_reasons + heuristic_reasons,
            }
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate["score"],
            -candidate["history_support"],
            candidate["name"],
        )
    )
    selected = candidates[0]
    return {
        "requested_policy_mode": "auto",
        "effective_policy_mode": selected["effective_mode"],
        "effective_memory_map": bool(selected["memory_map"]),
        "effective_policy_knobs": selected["policy_knobs"],
        "effective_policy_signature": selected["effective_policy_signature"],
        "effective_policy_summary": selected["effective_policy_summary"],
        "selected_candidate": selected["name"],
        "selected_label": selected["label"],
        "context": context,
        "context_key": _policy_context_key(context),
        "probe_stats": probe_stats,
        "probe_decode_summary": probe_decode_summary,
        "candidates": candidates,
        "summary": (
            f"{selected['label']} ({selected['effective_policy_signature']}) "
            f"score={selected['score']:.2f}"
        ),
    }


def _remove_cli_option(argv: list[str], option: str, *, takes_value: bool) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == option:
            skip_next = takes_value
            continue
        if takes_value and token.startswith(f"{option}="):
            continue
        cleaned.append(token)
    return cleaned


def _trend_bounds_mad(values: list[float], mad_mult: float) -> tuple[float, float]:
    med = _median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = _median(abs_dev) if abs_dev else 0.0
    band = mad_mult * mad
    return med - band, med + band


def _safe_file_size(path: str) -> int | None:
    try:
        return int(Path(path).expanduser().stat().st_size)
    except OSError:
        return None


def _infer_storage_source(
    file_path: str, discovered_source: str | None = None
) -> str:
    if discovered_source:
        return discovered_source
    try:
        resolved = str(Path(file_path).expanduser().resolve())
    except OSError:
        resolved = str(file_path)
    if ".ollama/models/blobs/" in resolved:
        return "ollama"
    return "filesystem"


def _infer_weight_class(
    file_path: str,
    fmt: str | None,
    fallback_reasons: dict[str, int] | None = None,
) -> str:
    path_l = str(file_path).lower()
    if fallback_reasons and int(fallback_reasons.get("quantized_conversion", 0)) > 0:
        return "quantized"
    if fmt == "safetensors":
        return "quantized" if _QUANTIZATION_HINT_RE.search(path_l) else "dense"
    if fmt == "gguf":
        if _QUANTIZATION_HINT_RE.search(path_l):
            return "quantized"
        if _DENSE_HINT_RE.search(path_l):
            return "dense"
    return "unknown"


def _model_class_name(fmt: str | None, weight_class: str) -> str:
    base = fmt or "unknown"
    return f"{weight_class}_{base}" if weight_class != "unknown" else f"unknown_{base}"


def _infer_model_metadata(
    file_path: str,
    fmt: str | None,
    fallback_reasons: dict[str, int] | None = None,
    discovered_info: dict | None = None,
) -> dict:
    discovered_info = discovered_info or {}
    weight_class = _infer_weight_class(file_path, fmt, fallback_reasons)
    return {
        "format": fmt or "unknown",
        "weight_class": weight_class,
        "model_class": _model_class_name(fmt, weight_class),
        "source": _infer_storage_source(
            file_path, discovered_source=discovered_info.get("source")
        ),
        "model_id": discovered_info.get("model_id"),
        "size_bytes": discovered_info.get("size_bytes") or _safe_file_size(file_path),
        "label": discovered_info.get("model_id") or Path(file_path).name,
    }


def _collect_environment_metadata(repo_root: Path) -> dict:
    git_head = _safe_check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    git_head_short = _safe_check_output(
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"]
    )
    git_branch = _safe_check_output(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"]
    )
    git_dirty = bool(
        _safe_check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--short",
                "--untracked-files=no",
            ]
        )
    )
    cpu_brand = (
        _safe_check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if sys.platform == "darwin"
        else None
    )
    hw_model = (
        _safe_check_output(["sysctl", "-n", "hw.model"])
        if sys.platform == "darwin"
        else None
    )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "git_head": git_head,
        "git_head_short": git_head_short,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "cpu_brand": cpu_brand,
        "hw_model": hw_model,
        "host_chip": cpu_brand or hw_model or platform.machine(),
    }


def _build_history_bucket(summary_payload: dict) -> dict[str, str]:
    return {
        "format": str(summary_payload.get("format") or "unknown"),
        "model_class": str(summary_payload.get("model_class") or "unknown"),
        "cache_mode": str(summary_payload.get("cache_mode") or "inherit"),
        "decode_mode": str(summary_payload.get("decode_mode") or "none"),
        "policy": str(
            summary_payload.get("effective_policy_signature")
            or summary_payload.get("policy_signature")
            or "default"
        ),
    }


def _history_bucket_key(bucket: dict[str, str]) -> str:
    return "|".join(
        [
            f"format={bucket['format']}",
            f"class={bucket['model_class']}",
            f"cache={bucket['cache_mode']}",
            f"decode={bucket['decode_mode']}",
            f"policy={bucket['policy']}",
        ]
    )


def _record_history_bucket(record: dict) -> dict[str, str]:
    bucket = record.get("history_bucket")
    if isinstance(bucket, dict):
        return {
            "format": str(bucket.get("format") or record.get("format") or "unknown"),
            "model_class": str(
                bucket.get("model_class") or record.get("model_class") or "unknown"
            ),
            "cache_mode": str(
                bucket.get("cache_mode") or record.get("cache_mode") or "inherit"
            ),
            "decode_mode": str(
                bucket.get("decode_mode") or record.get("decode_mode") or "none"
            ),
            "policy": str(
                bucket.get("policy")
                or record.get("effective_policy_signature")
                or record.get("policy_signature")
                or "default"
            ),
        }
    return _build_history_bucket(record)


def _matches_history_bucket(record: dict, bucket: dict[str, str]) -> bool:
    return _record_history_bucket(record) == bucket


def _select_trend_history(
    prior_history: list[dict],
    summary_payload: dict,
    *,
    file_path: str,
    history_window: int,
    trend_min_samples: int,
) -> dict:
    bucket = _build_history_bucket(summary_payload)
    bucket_key = _history_bucket_key(bucket)
    file_history = [h for h in prior_history if h.get("file") == file_path]
    bucket_history = [h for h in prior_history if _matches_history_bucket(h, bucket)]
    window = max(history_window, 1)
    bucket_recent = bucket_history[-window:]
    file_recent = file_history[-window:]
    if len(bucket_recent) >= trend_min_samples:
        return {
            "scope": "bucket",
            "records": bucket_recent,
            "bucket": bucket,
            "bucket_key": bucket_key,
            "bucket_samples": len(bucket_recent),
            "file_samples": len(file_recent),
        }
    if len(file_recent) >= trend_min_samples:
        return {
            "scope": "file",
            "records": file_recent,
            "bucket": bucket,
            "bucket_key": bucket_key,
            "bucket_samples": len(bucket_recent),
            "file_samples": len(file_recent),
        }
    return {
        "scope": "bucket",
        "records": bucket_recent,
        "bucket": bucket,
        "bucket_key": bucket_key,
        "bucket_samples": len(bucket_recent),
        "file_samples": len(file_recent),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _bucket_match_summary(bucket: dict[str, str]) -> str:
    return (
        f"format={bucket['format']}, "
        f"model_class={bucket['model_class']}, "
        f"cache_mode={bucket['cache_mode']}, "
        f"decode_mode={bucket['decode_mode']}, "
        f"policy={bucket['policy']}"
    )


def _describe_trend_context(trend_context: dict) -> str:
    bucket_summary = _bucket_match_summary(trend_context["bucket"])
    bucket_samples = int(trend_context.get("bucket_samples", 0) or 0)
    file_samples = int(trend_context.get("file_samples", 0) or 0)
    if trend_context.get("scope") == "bucket":
        if file_samples > 0:
            return (
                f"Using comparable-run history because it matches {bucket_summary} "
                f"and has {bucket_samples} sample(s); same-file history only has "
                f"{file_samples} sample(s)."
            )
        return (
            f"Using comparable-run history because it matches {bucket_summary} "
            f"and has {bucket_samples} sample(s)."
        )
    return (
        f"Falling back to same-file history because the comparable-run bucket "
        f"for {bucket_summary} only has {bucket_samples} sample(s), while this "
        f"file has {file_samples} sample(s)."
    )


def _collect_trend_regressions(
    summary_payload: dict,
    recent: list[dict],
    *,
    trend_min_samples: int,
    trend_mad_mult: float,
) -> list[str]:
    regressions: list[str] = []
    for spec in _COMPARISON_METRIC_SPECS:
        metric_key = str(spec["key"])
        current = summary_payload.get(metric_key)
        if current is None:
            continue
        values = [record.get(metric_key) for record in recent]
        values = [value for value in values if isinstance(value, (int, float))]
        if len(values) < trend_min_samples:
            continue
        lower, upper = _trend_bounds_mad(values, trend_mad_mult)
        if spec["higher_is_better"] and current < lower:
            regressions.append(f"{metric_key}={current:.2f} < trend_lower={lower:.2f}")
        if (not spec["higher_is_better"]) and current > upper:
            regressions.append(f"{metric_key}={current:.2f} > trend_upper={upper:.2f}")
    return regressions


def _reference_profile(records: list[dict]) -> dict[str, float]:
    profile: dict[str, float] = {}
    for spec in _COMPARISON_METRIC_SPECS:
        key = spec["key"]
        values = [record.get(key) for record in records]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        if numeric:
            profile[key] = float(_median(numeric))
    return profile


def _record_matches_git_head(record: dict, requested_git_head: str) -> bool:
    requested = str(requested_git_head).strip()
    if not requested:
        return False
    environment = record.get("environment") or {}
    code_state = record.get("code_state") or {}
    for candidate in (
        environment.get("git_head"),
        environment.get("git_head_short"),
        code_state.get("git_head"),
        code_state.get("git_head_short"),
    ):
        if candidate and str(candidate).startswith(requested):
            return True
    return False


def _select_explicit_baseline(
    prior_history: list[dict],
    summary_payload: dict,
    *,
    file_path: str,
    baseline_run_id: str | None,
    baseline_git_head: str | None,
) -> dict | None:
    if baseline_run_id:
        candidates = [
            record
            for record in prior_history
            if str(record.get("run_id") or "") == str(baseline_run_id)
        ]
        if not candidates:
            raise ValueError(
                f"--baseline-run-id did not match any history record: {baseline_run_id}"
            )
        record = candidates[-1]
        return {
            "basis": "baseline_run",
            "summary": f"Compared against explicit baseline run {baseline_run_id}.",
            "sample_count": 1,
            "reference_record": record,
            "reference_profile": _reference_profile([record]),
            "reference_run_id": record.get("run_id"),
            "reference_git_head": (
                (record.get("environment") or {}).get("git_head_short")
                or (record.get("environment") or {}).get("git_head")
            ),
        }

    if not baseline_git_head:
        return None

    candidates = [
        record
        for record in prior_history
        if _record_matches_git_head(record, baseline_git_head)
    ]
    if not candidates:
        raise ValueError(
            f"--baseline-git-head did not match any history record: {baseline_git_head}"
        )

    bucket = _build_history_bucket(summary_payload)
    same_bucket = [record for record in candidates if _matches_history_bucket(record, bucket)]
    if same_bucket:
        record = same_bucket[-1]
        basis = "baseline_git_bucket"
        summary = (
            "Compared against the latest explicit baseline commit match in the "
            "same benchmark bucket."
        )
    else:
        same_file = [record for record in candidates if record.get("file") == file_path]
        if same_file:
            record = same_file[-1]
            basis = "baseline_git_file"
            summary = (
                "Compared against the latest explicit baseline commit match for "
                "the same file."
            )
        else:
            record = candidates[-1]
            basis = "baseline_git_head"
            summary = (
                "Compared against the latest explicit baseline commit match in history."
            )

    return {
        "basis": basis,
        "summary": summary,
        "sample_count": 1,
        "reference_record": record,
        "reference_profile": _reference_profile([record]),
        "reference_run_id": record.get("run_id"),
        "reference_git_head": (
            (record.get("environment") or {}).get("git_head_short")
            or (record.get("environment") or {}).get("git_head")
        ),
    }


def _comparison_metric_entry(
    key: str,
    *,
    current: float,
    reference: float,
    higher_is_better: bool,
    label: str,
    deadband_pct: float = 1.0,
) -> dict:
    delta = float(current) - float(reference)
    if higher_is_better:
        if delta >= deadband_pct:
            status = "win"
        elif delta <= -deadband_pct:
            status = "loss"
        else:
            status = "neutral"
    else:
        if delta <= -deadband_pct:
            status = "win"
        elif delta >= deadband_pct:
            status = "loss"
        else:
            status = "neutral"
    return {
        "key": key,
        "label": label,
        "current": float(current),
        "reference": float(reference),
        "delta": delta,
        "higher_is_better": higher_is_better,
        "status": status,
    }


def _build_comparison_summary(
    summary_payload: dict,
    *,
    prior_history: list[dict],
    file_path: str,
    trend_context: dict,
    baseline_run_id: str | None,
    baseline_git_head: str | None,
) -> dict | None:
    explicit_baseline = _select_explicit_baseline(
        prior_history,
        summary_payload,
        file_path=file_path,
        baseline_run_id=baseline_run_id,
        baseline_git_head=baseline_git_head,
    )
    if explicit_baseline is not None:
        basis = explicit_baseline["basis"]
        summary = explicit_baseline["summary"]
        sample_count = int(explicit_baseline.get("sample_count", 1) or 1)
        reference_profile = explicit_baseline["reference_profile"]
        reference_record = explicit_baseline.get("reference_record")
    else:
        reference_records = list(trend_context.get("records") or [])
        if not reference_records:
            return None
        basis = (
            "bucket_recent_median"
            if trend_context.get("scope") == "bucket"
            else "file_recent_median"
        )
        bucket_key = trend_context.get("bucket_key")
        sample_count = len(reference_records)
        reference_profile = _reference_profile(reference_records)
        reference_record = None
        if basis == "bucket_recent_median":
            summary = (
                f"Compared against the recent comparable-run bucket median "
                f"({sample_count} sample(s), {bucket_key})."
            )
        else:
            summary = (
                f"Compared against the recent same-file median ({sample_count} sample(s))."
            )

    metrics: dict[str, dict] = {}
    wins = 0
    losses = 0
    neutrals = 0
    for spec in _COMPARISON_METRIC_SPECS:
        key = spec["key"]
        current = summary_payload.get(key)
        reference = reference_profile.get(key)
        if not isinstance(current, (int, float)) or not isinstance(reference, (int, float)):
            continue
        metric_entry = _comparison_metric_entry(
            key,
            current=float(current),
            reference=float(reference),
            higher_is_better=bool(spec["higher_is_better"]),
            label=str(spec["label"]),
        )
        metrics[key] = metric_entry
        if metric_entry["status"] == "win":
            wins += 1
        elif metric_entry["status"] == "loss":
            losses += 1
        else:
            neutrals += 1

    if not metrics:
        return None

    if wins > 0 and losses == 0:
        auto_outcome = "win"
    elif losses > 0 and wins == 0:
        auto_outcome = "loss"
    elif wins > 0 or losses > 0:
        auto_outcome = "mixed"
    else:
        auto_outcome = "inconclusive"

    comparison = {
        "basis": basis,
        "summary": summary,
        "sample_count": sample_count,
        "metrics": metrics,
        "scorecard": {
            "wins": wins,
            "losses": losses,
            "neutral": neutrals,
        },
        "auto_outcome": auto_outcome,
        "reference_run_id": reference_record.get("run_id") if reference_record else None,
        "reference_file": reference_record.get("file") if reference_record else None,
        "reference_git_head": (
            (reference_record.get("environment") or {}).get("git_head_short")
            if reference_record
            else None
        ),
        "reference_bucket_key": (
            trend_context.get("bucket_key")
            if explicit_baseline is None and trend_context.get("scope") == "bucket"
            else None
        ),
    }
    return comparison


def _resolve_route_outcome(
    *,
    requested_outcome: str,
    comparison: dict | None,
    file_ok: bool,
) -> tuple[str, str]:
    if requested_outcome != "auto":
        return requested_outcome, "explicit"
    if comparison is None:
        return ("loss", "auto") if not file_ok else ("inconclusive", "auto")
    auto_outcome = str(comparison.get("auto_outcome") or "inconclusive")
    if auto_outcome == "inconclusive" and not file_ok:
        return "loss", "auto"
    return auto_outcome, "auto"


def _route_outcome_stats(records: list[dict]) -> dict[str, dict]:
    stats_by_route: dict[str, dict] = {}
    for record in records:
        route_ref = _normalize_route_reference(
            record.get("attempted_route_id") or record.get("attempted_route")
        )
        if route_ref is None:
            continue
        route_id, route_display = route_ref
        route_stats = stats_by_route.setdefault(
            route_id,
            {
                "route_id": route_id,
                "route": route_display,
                "win": 0,
                "loss": 0,
                "mixed": 0,
                "inconclusive": 0,
                "samples": 0,
            },
        )
        outcome = _normalize_outcome_label(
            record.get("route_outcome")
            or (record.get("comparison") or {}).get("auto_outcome")
            or "inconclusive"
        )
        route_stats[outcome] += 1
        route_stats["samples"] += 1

    for route_stats in stats_by_route.values():
        scored = int(route_stats["win"] + route_stats["loss"] + route_stats["mixed"])
        route_stats["scored_samples"] = scored
        route_stats["success_rate"] = (
            (route_stats["win"] + 0.5 * route_stats["mixed"]) / scored
            if scored > 0
            else None
        )
        route_stats["net_wins"] = route_stats["win"] - route_stats["loss"]
    return stats_by_route


def _format_route_history_summary(route_stats: dict, *, scope: str) -> str:
    success_rate = route_stats.get("success_rate")
    hit_rate = (
        f"{100.0 * float(success_rate):.1f}%"
        if isinstance(success_rate, (int, float))
        else "n/a"
    )
    return (
        f"{scope} history: win/loss/mixed/inconclusive="
        f"{route_stats['win']}/{route_stats['loss']}/{route_stats['mixed']}/"
        f"{route_stats['inconclusive']} hit_rate={hit_rate}"
    )


def _calibrate_route_suggestions(
    route_suggestions: list[dict],
    prior_history: list[dict],
    *,
    bucket_key: str,
) -> list[dict]:
    bucket_records = [
        record
        for record in prior_history
        if str(record.get("history_bucket_key") or "") == str(bucket_key)
    ]
    bucket_stats = _route_outcome_stats(bucket_records)
    global_stats = _route_outcome_stats(prior_history)
    calibrated: list[dict] = []
    for suggestion in route_suggestions:
        item = dict(suggestion)
        route_stats = bucket_stats.get(item["route_id"])
        scope = "bucket"
        if route_stats is None:
            route_stats = global_stats.get(item["route_id"])
            scope = "global"
        if route_stats is not None:
            success_rate = route_stats.get("success_rate")
            scored_samples = int(route_stats.get("scored_samples", 0) or 0)
            adjustment = 0.0
            if isinstance(success_rate, (int, float)) and scored_samples > 0:
                adjustment_scale = min(0.12, 0.03 * scored_samples)
                adjustment = (float(success_rate) - 0.5) * 2.0 * adjustment_scale
            item["history_scope"] = scope
            item["history_support"] = route_stats
            item["history_summary"] = _format_route_history_summary(
                route_stats, scope=scope
            )
            item["confidence"] = round(_clamp01(item["confidence"] + adjustment), 2)
            item["confidence_label"] = _confidence_label(item["confidence"])
            item["confidence_adjustment"] = round(adjustment, 2)
        calibrated.append(item)
    calibrated.sort(key=lambda entry: (-entry["priority"], entry["route"]))
    return calibrated


def _report_policy_outcome(record: dict) -> str:
    outcome = record.get("route_outcome")
    if outcome:
        return _normalize_outcome_label(outcome)
    comparison = record.get("comparison") or {}
    return _normalize_outcome_label(comparison.get("auto_outcome") or "inconclusive")


def _build_history_report(records: list[dict], *, top_n: int = 3) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        bucket_key = str(
            record.get("history_bucket_key")
            or _history_bucket_key(_record_history_bucket(record))
        )
        grouped[bucket_key].append(record)

    bucket_reports: list[dict] = []
    for bucket_key, bucket_records in sorted(
        grouped.items(), key=lambda item: len(item[1]), reverse=True
    ):
        fallback_bytes = Counter()
        regression_counts: dict[str, list[float]] = defaultdict(list)
        policy_stats: dict[str, dict] = {}
        for record in bucket_records:
            for reason, value in (record.get("mmap_coverage_fallback_reason_bytes") or {}).items():
                fallback_bytes[str(reason)] += int(value)
            comparison = record.get("comparison") or {}
            for metric_key, entry in (comparison.get("metrics") or {}).items():
                if entry.get("status") == "loss":
                    regression_counts[metric_key].append(abs(float(entry["delta"])))
            policy = str(
                record.get("effective_policy_signature")
                or record.get("policy_signature")
                or "default"
            )
            bucket_policy = policy_stats.setdefault(
                policy,
                {
                    "policy": policy,
                    "samples": 0,
                    "win": 0,
                    "loss": 0,
                    "mixed": 0,
                    "inconclusive": 0,
                },
            )
            outcome = _report_policy_outcome(record)
            bucket_policy["samples"] += 1
            bucket_policy[outcome] += 1

        for bucket_policy in policy_stats.values():
            scored = bucket_policy["win"] + bucket_policy["loss"] + bucket_policy["mixed"]
            bucket_policy["success_rate"] = (
                (bucket_policy["win"] + 0.5 * bucket_policy["mixed"]) / scored
                if scored > 0
                else None
            )
            bucket_policy["net_wins"] = bucket_policy["win"] - bucket_policy["loss"]

        route_stats = _sorted_route_outcome_stats(_route_outcome_stats(bucket_records))
        regressions = sorted(
            (
                {
                    "metric_key": metric_key,
                    "label": _COMPARISON_METRIC_BY_KEY.get(metric_key, {}).get(
                        "label", metric_key
                    ),
                    "count": len(values),
                    "magnitude_median": float(_median(values)),
                }
                for metric_key, values in regression_counts.items()
            ),
            key=lambda item: (-item["count"], -item["magnitude_median"], item["metric_key"]),
        )
        policies = sorted(
            policy_stats.values(),
            key=lambda item: (
                -(item.get("success_rate") if item.get("success_rate") is not None else -1.0),
                -item["net_wins"],
                item["policy"],
            ),
        )
        losers = sorted(
            policy_stats.values(),
            key=lambda item: (
                item.get("success_rate") if item.get("success_rate") is not None else 2.0,
                item["net_wins"],
                item["policy"],
            ),
        )
        winners = policies[:top_n]
        winner_policies = {entry["policy"] for entry in winners}
        loser_entries: list[dict] = []
        for entry in losers:
            if entry["policy"] in winner_policies:
                continue
            loser_entries.append(entry)
            if len(loser_entries) >= top_n:
                break
        bucket_reports.append(
            {
                "bucket_key": bucket_key,
                "bucket": _record_history_bucket(bucket_records[-1]),
                "samples": len(bucket_records),
                "latest_timestamp": bucket_records[-1].get("timestamp"),
                "route_hit_rate": route_stats[:top_n],
                "top_fallback_reasons_by_bytes": [
                    {"reason": reason, "bytes": byte_count}
                    for reason, byte_count in fallback_bytes.most_common(top_n)
                ],
                "top_phase_regressions": regressions[:top_n],
                "policy_winners": winners,
                "policy_losers": loser_entries,
            }
        )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": len(records),
        "buckets": bucket_reports,
    }


def _print_history_report(report: dict) -> None:
    print(
        "History report: "
        f"records={report['records']} buckets={len(report['buckets'])}"
    )
    for bucket_report in report["buckets"]:
        print(f"\nBucket: {bucket_report['bucket_key']}")
        print(
            "  samples="
            f"{bucket_report['samples']} latest={bucket_report['latest_timestamp']}"
        )
        if bucket_report["route_hit_rate"]:
            print("  route hit-rate:")
            for route in bucket_report["route_hit_rate"]:
                success_rate = route.get("success_rate")
                hit_rate = (
                    f"{100.0 * float(success_rate):.1f}%"
                    if isinstance(success_rate, (int, float))
                    else "n/a"
                )
                print(
                    f"    {route['route']} ({route['route_id']}): "
                    f"{route['win']}W/{route['loss']}L/{route['mixed']}M/"
                    f"{route['inconclusive']}I hit_rate={hit_rate}"
                )
        if bucket_report["top_fallback_reasons_by_bytes"]:
            print("  top fallback bytes:")
            for entry in bucket_report["top_fallback_reasons_by_bytes"]:
                print(f"    {entry['reason']}: {_format_mib(entry['bytes'])}")
        if bucket_report["top_phase_regressions"]:
            print("  top regressions:")
            for entry in bucket_report["top_phase_regressions"]:
                print(
                    f"    {entry['label']}: count={entry['count']} "
                    f"median_delta={entry['magnitude_median']:.2f}pp"
                )
        if bucket_report["policy_winners"]:
            print("  policy winners:")
            for entry in bucket_report["policy_winners"]:
                success_rate = entry.get("success_rate")
                hit_rate = (
                    f"{100.0 * float(success_rate):.1f}%"
                    if isinstance(success_rate, (int, float))
                    else "n/a"
                )
                print(
                    f"    {entry['policy']}: samples={entry['samples']} "
                    f"hit_rate={hit_rate}"
                )
        if bucket_report["policy_losers"]:
            print("  policy losers:")
            for entry in bucket_report["policy_losers"]:
                success_rate = entry.get("success_rate")
                hit_rate = (
                    f"{100.0 * float(success_rate):.1f}%"
                    if isinstance(success_rate, (int, float))
                    else "n/a"
                )
                print(
                    f"    {entry['policy']}: samples={entry['samples']} "
                    f"hit_rate={hit_rate}"
                )


def _write_history_report_csv(report: dict) -> str:
    rows: list[dict[str, str]] = []
    for bucket_report in report["buckets"]:
        rows.append(
            {
                "bucket_key": bucket_report["bucket_key"],
                "samples": str(bucket_report["samples"]),
                "latest_timestamp": str(bucket_report["latest_timestamp"] or ""),
                "routes": "; ".join(
                    (
                        f"{route['route_id']}="
                        f"{route['win']}W/{route['loss']}L/{route['mixed']}M/"
                        f"{route['inconclusive']}I"
                    )
                    for route in bucket_report["route_hit_rate"]
                ),
                "fallback_reasons": "; ".join(
                    f"{entry['reason']}={entry['bytes']}"
                    for entry in bucket_report["top_fallback_reasons_by_bytes"]
                ),
                "regressions": "; ".join(
                    f"{entry['metric_key']}:{entry['count']}@{entry['magnitude_median']:.2f}"
                    for entry in bucket_report["top_phase_regressions"]
                ),
                "policy_winners": "; ".join(
                    f"{entry['policy']}:{entry['samples']}"
                    for entry in bucket_report["policy_winners"]
                ),
                "policy_losers": "; ".join(
                    f"{entry['policy']}:{entry['samples']}"
                    for entry in bucket_report["policy_losers"]
                ),
            }
        )
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    output_lines: list[str] = []
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().rstrip()


def _top_fallback_reason(reasons: dict[str, int] | None) -> tuple[str | None, int]:
    if not reasons:
        return None, 0
    reason, count = max(sorted(reasons.items()), key=lambda item: item[1])
    return reason, int(count)


def _top_fallback_bytes_reason(
    reason_bytes: dict[str, int] | None,
) -> tuple[str | None, int]:
    if not reason_bytes:
        return None, 0
    reason, byte_count = max(sorted(reason_bytes.items()), key=lambda item: item[1])
    return reason, int(byte_count)


def _bytes_to_mib(num_bytes: int | float | None) -> float:
    return float(num_bytes or 0) / (1024.0**2)


def _format_mib(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{_bytes_to_mib(num_bytes):.2f}MiB"


def _format_seconds(seconds: int | float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{float(seconds):.4f}s"


def _format_ms(seconds: int | float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{1000.0 * float(seconds):.2f}ms"


def _format_pct(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _format_fault_pair(minor: int | float | None, major: int | float | None) -> str:
    if minor is None or major is None:
        return "(n/a)"
    return f"({float(minor):.0f}/{float(major):.0f})"


def _fallback_priority_summary(fallback_priority_basis: str) -> str:
    if fallback_priority_basis == "count":
        return (
            "count-first using fallback tensor count; materialized destination "
            "bytes break ties"
        )
    return (
        "bytes-first using materialized destination bytes; fallback tensor "
        "count breaks ties"
    )


def _fallback_reason_ranks(
    fallback_reasons: dict[str, int] | None,
    fallback_reason_bytes: dict[str, int] | None,
    *,
    fallback_priority_basis: str,
) -> dict[str, int]:
    reasons = set((fallback_reasons or {}).keys()) | set((fallback_reason_bytes or {}).keys())
    if not reasons:
        return {}

    def key(reason: str) -> tuple[int, int, str]:
        count = int((fallback_reasons or {}).get(reason, 0))
        materialized_bytes = int((fallback_reason_bytes or {}).get(reason, 0))
        if fallback_priority_basis == "count":
            return (-count, -materialized_bytes, reason)
        return (-materialized_bytes, -count, reason)

    ordered = sorted(reasons, key=key)
    return {reason: rank for rank, reason in enumerate(ordered)}


def _fallback_priority_bonus(rank: int | None) -> int:
    if rank is None:
        return 0
    if rank <= 0:
        return 10
    if rank == 1:
        return 6
    if rank == 2:
        return 3
    return 0


def _format_fallback_priority_order(
    fallback_reasons: dict[str, int] | None,
    fallback_reason_bytes: dict[str, int] | None,
    *,
    fallback_priority_basis: str,
) -> str | None:
    rank_map = _fallback_reason_ranks(
        fallback_reasons,
        fallback_reason_bytes,
        fallback_priority_basis=fallback_priority_basis,
    )
    if not rank_map:
        return None
    ordered = sorted(rank_map.items(), key=lambda item: item[1])
    formatted: list[str] = []
    for reason, _ in ordered:
        count = int((fallback_reasons or {}).get(reason, 0))
        materialized_bytes = int((fallback_reason_bytes or {}).get(reason, 0))
        formatted.append(
            f"{reason} ({count} tensors, {_bytes_to_mib(materialized_bytes):.1f} MiB)"
        )
    return " > ".join(formatted)


def _build_route_suggestions(
    summary_payload: dict,
    *,
    decode_enabled: bool,
    decode_gate_pct: float,
    fallback_priority_basis: str,
    trend_regressions: list[str],
) -> list[dict]:
    suggestions: list[dict] = []
    seen_routes: set[str] = set()

    def add(
        route: str,
        why: str,
        try_next: str,
        priority: int,
        confidence: float,
        confidence_why: str,
    ) -> None:
        if route in seen_routes:
            return
        seen_routes.add(route)
        confidence = round(_clamp01(confidence), 2)
        suggestions.append(
            {
                "route_id": _route_id_for_display(route),
                "route": route,
                "why": why,
                "try_next": try_next,
                "priority": int(priority),
                "confidence": confidence,
                "confidence_label": _confidence_label(confidence),
                "confidence_why": confidence_why,
            }
        )

    load_improve = summary_payload.get("load_improve_pct_median")
    rss_reduce = summary_payload.get("rss_reduce_pct_median")
    decode_regression = summary_payload.get("decode_regression_pct_median")
    mapped_ratio = summary_payload.get("mmap_coverage_mapped_ratio_pct")
    fallback_tensors = summary_payload.get("mmap_coverage_fallback_tensors")
    fallback_reasons = summary_payload.get("mmap_coverage_fallback_reasons") or {}
    fallback_reason_bytes = (
        summary_payload.get("mmap_coverage_fallback_reason_bytes") or {}
    )
    fallback_reason_source_bytes = (
        summary_payload.get("mmap_coverage_fallback_reason_source_bytes") or {}
    )
    fallback_rank_map = _fallback_reason_ranks(
        fallback_reasons,
        fallback_reason_bytes,
        fallback_priority_basis=fallback_priority_basis,
    )
    copy_major_faults = summary_payload.get("copy_load_major_faults_median")
    mmap_major_faults = summary_payload.get("mmap_load_major_faults_median")
    cold_validation = summary_payload.get("cold_cache_validation") or {}

    quantized_fallbacks = int(fallback_reasons.get("quantized_conversion", 0))
    dtype_fallbacks = int(fallback_reasons.get("dtype_conversion", 0))
    misaligned_fallbacks = int(fallback_reasons.get("misaligned_offset", 0))
    top_reason, top_count = _top_fallback_reason(fallback_reasons)
    top_reason_bytes, top_byte_count = _top_fallback_bytes_reason(fallback_reason_bytes)

    if quantized_fallbacks > 0:
        quantized_count_share = quantized_fallbacks / max(
            int(fallback_tensors or 0), quantized_fallbacks, 1
        )
        quantized_byte_share = (
            int(fallback_reason_bytes.get("quantized_conversion", 0))
            / max(int(summary_payload.get("mmap_coverage_copied_bytes") or 0), 1)
        )
        quantized_share = max(quantized_count_share, quantized_byte_share)
        confidence = (
            0.6
            + 0.2 * quantized_share
            + (0.1 if quantized_fallbacks >= 16 else 0.0)
            + (
                0.05
                if isinstance(mapped_ratio, (int, float)) and mapped_ratio < 50.0
                else 0.0
            )
        )
        add(
            route="Quantized Direct Map",
            why=(
                "Coverage is still falling back on quantized conversion "
                f"({quantized_fallbacks} tensors)."
            ),
            try_next=(
                "Prototype direct mapped quantized views or a lazy dequant cache "
                "and watch copied bytes plus warm decode tok/s."
            ),
            priority=86 + _fallback_priority_bonus(
                fallback_rank_map.get("quantized_conversion")
            ),
            confidence=confidence,
            confidence_why=(
                f"quantized_conversion is {quantized_fallbacks}/"
                f"{max(int(fallback_tensors or 0), quantized_fallbacks)} fallback tensors "
                f"and {_bytes_to_mib(fallback_reason_bytes.get('quantized_conversion', 0)):.1f} MiB "
                f"of materialized fallback data from "
                f"{_bytes_to_mib(fallback_reason_source_bytes.get('quantized_conversion', 0)):.1f} MiB "
                f"of source data"
            ),
        )

    if dtype_fallbacks > 0:
        dtype_bytes = int(fallback_reason_bytes.get("dtype_conversion", 0))
        confidence = (
            0.45
            + min(0.25, 0.04 * dtype_fallbacks)
            + min(0.15, 0.1 * (dtype_bytes > 8 * 1024 * 1024))
            + (
                0.1
                if isinstance(mapped_ratio, (int, float)) and mapped_ratio < 80.0
                else 0.0
            )
        )
        add(
            route="Lazy Dtype Conversion",
            why=(
                "Coverage is paying eager dtype conversion "
                f"for {dtype_fallbacks} tensors."
            ),
            try_next=(
                "Delay dtype conversion until first consumer or batch it after "
                "load to keep mapped coverage high."
            ),
            priority=85
            + _fallback_priority_bonus(fallback_rank_map.get("dtype_conversion")),
            confidence=confidence,
            confidence_why=(
                f"dtype_conversion appears on {dtype_fallbacks} fallback tensor(s) "
                f"covering {_bytes_to_mib(dtype_bytes):.1f} MiB of materialized data "
                f"from {_bytes_to_mib(fallback_reason_source_bytes.get('dtype_conversion', 0)):.1f} MiB "
                f"of source data"
            ),
        )

    if misaligned_fallbacks > 0:
        misaligned_bytes = int(fallback_reason_bytes.get("misaligned_offset", 0))
        confidence = (
            0.5
            + min(0.2, 0.03 * misaligned_fallbacks)
            + min(0.15, 0.1 * (misaligned_bytes > 8 * 1024 * 1024))
            + (
                0.1
                if isinstance(mapped_ratio, (int, float)) and mapped_ratio < 80.0
                else 0.0
            )
        )
        add(
            route="Alignment-Friendly Mapping",
            why=(
                "Misaligned tensor offsets are blocking some mapped views "
                f"({misaligned_fallbacks} tensors)."
            ),
            try_next=(
                "Test alignment-aware exports or a hybrid copier that only "
                "materializes the misaligned tensors."
            ),
            priority=83
            + _fallback_priority_bonus(fallback_rank_map.get("misaligned_offset")),
            confidence=confidence,
            confidence_why=(
                f"misaligned offsets account for {misaligned_fallbacks} fallback tensor(s) "
                f"covering {_bytes_to_mib(misaligned_bytes):.1f} MiB of materialized data"
            ),
        )

    if top_reason is not None and top_reason not in {
        "quantized_conversion",
        "dtype_conversion",
        "misaligned_offset",
    }:
        top_reason_bytes_value = int(fallback_reason_bytes.get(top_reason, 0))
        confidence = (
            0.45
            + min(0.25, 0.02 * top_count)
            + min(0.15, 0.1 * (top_reason_bytes_value > 8 * 1024 * 1024))
            + (
                0.1
                if isinstance(mapped_ratio, (int, float)) and mapped_ratio < 80.0
                else 0.0
            )
        )
        add(
            route="Coverage-Targeted Fallback Work",
            why=(
                f"The dominant fallback reason is {top_reason} "
                f"({top_count} tensors)."
            ),
            try_next=(
                "Profile that fallback path directly and see if a small targeted "
                "fast path buys more mapped coverage than broader changes."
            ),
            priority=82 + _fallback_priority_bonus(fallback_rank_map.get(top_reason)),
            confidence=confidence,
            confidence_why=(
                f"{top_reason} is the dominant fallback reason at {top_count} tensor(s) "
                f"covering {_bytes_to_mib(top_reason_bytes_value):.1f} MiB of materialized data"
            ),
        )

    if (
        decode_enabled
        and isinstance(load_improve, (int, float))
        and isinstance(rss_reduce, (int, float))
        and isinstance(decode_regression, (int, float))
        and load_improve >= 10.0
        and rss_reduce >= 20.0
        and decode_regression >= max(1.0, 0.5 * decode_gate_pct)
    ):
        hotset_confidence = (
            0.55
            + (0.1 if load_improve >= 20.0 else 0.0)
            + (0.1 if rss_reduce >= 50.0 else 0.0)
            + (0.1 if decode_regression > decode_gate_pct else 0.0)
            + (
                0.05
                if isinstance(mapped_ratio, (int, float)) and mapped_ratio >= 70.0
                else 0.0
            )
        )
        add(
            route="Hotset Promotion",
            why=(
                "mmap is clearly helping load/RSS "
                f"(load +{load_improve:.1f}%, RSS +{rss_reduce:.1f}%) "
                f"but decode regresses {decode_regression:.1f}%."
            ),
            try_next=(
                "Start tensors mapped, then copy only the repeatedly touched "
                "decode tensors or hot layers into owned buffers."
            ),
            priority=94 if decode_regression > decode_gate_pct else 87,
            confidence=hotset_confidence,
            confidence_why=(
                f"load/RSS wins are strong while decode regresses "
                f"{decode_regression:.1f}% against a {decode_gate_pct:.1f}% gate"
            ),
        )
        adaptive_confidence = (
            0.55
            + (0.1 if load_improve >= 10.0 else 0.0)
            + (0.1 if rss_reduce >= 20.0 else 0.0)
            + (0.1 if decode_regression > decode_gate_pct else 0.0)
        )
        add(
            route="Adaptive Default Policy",
            why=(
                "This run is mixed across axes: mmap wins load/RSS but not decode."
            ),
            try_next=(
                "Gate mmap by format, model class, cache mode, or mapped ratio "
                "instead of flipping a universal default."
            ),
            priority=82,
            confidence=adaptive_confidence,
            confidence_why=(
                "the run has a real tradeoff across load, RSS, and decode rather "
                "than a single directionally clean winner"
            ),
        )

    if (
        isinstance(mapped_ratio, (int, float))
        and isinstance(fallback_tensors, (int, float))
        and mapped_ratio < 80.0
        and fallback_tensors > 0
    ):
        confidence = (
            0.55
            + 0.2 * (1.0 - min(mapped_ratio, 100.0) / 100.0)
            + (0.1 if int(fallback_tensors) >= 16 else 0.0)
        )
        add(
            route="Hybrid mmap Policy",
            why=(
                f"Only {mapped_ratio:.1f}% of bytes mapped and "
                f"{int(fallback_tensors)} tensors still fall back."
            ),
            try_next=(
                "Map the easy large tensors first, copy the rest, and test a "
                "simple byte-size threshold."
            ),
            priority=88 if mapped_ratio < 50.0 else 76,
            confidence=confidence,
            confidence_why=(
                f"mapped coverage is only {mapped_ratio:.1f}% with "
                f"{int(fallback_tensors)} fallback tensor(s) and "
                f"{_bytes_to_mib(summary_payload.get('mmap_coverage_copied_bytes')):.1f} MiB copied"
            ),
        )

    if (
        summary_payload.get("cache_mode") == "cold-best-effort"
        and cold_validation.get("credible")
        and isinstance(copy_major_faults, (int, float))
        and isinstance(mmap_major_faults, (int, float))
        and max(copy_major_faults, mmap_major_faults) > 0
    ):
        confidence = (
            0.6
            + min(0.15, 0.05 * float(cold_validation.get("time_ratio", 0.0)))
            + (
                0.1
                if int(cold_validation.get("major_fault_delta", 0) or 0) >= 4
                else 0.0
            )
        )
        add(
            route="Prefetch / Readahead Tuning",
            why=(
                "Cold-start pressure looks real "
                f"(time_ratio={cold_validation['time_ratio']:.2f}x, "
                f"major_fault_delta={cold_validation['major_fault_delta']})."
            ),
            try_next=(
                "Compare the current sequential advice against selective "
                "WILLNEED or a tiny pre-touch of the first decode hotset."
            ),
            priority=84,
            confidence=confidence,
            confidence_why=(
                f"cold validation is credible with time_ratio="
                f"{cold_validation['time_ratio']:.2f}x and major_fault_delta="
                f"{cold_validation['major_fault_delta']}"
            ),
        )

    if (
        isinstance(mapped_ratio, (int, float))
        and mapped_ratio >= 95.0
        and isinstance(load_improve, (int, float))
        and isinstance(rss_reduce, (int, float))
        and load_improve < 5.0
        and rss_reduce < 5.0
    ):
        confidence = (
            0.55
            + (0.1 if mapped_ratio >= 99.0 else 0.0)
            + (0.1 if load_improve < 2.0 and rss_reduce < 2.0 else 0.0)
        )
        add(
            route="Allocator / Dispatch Audit",
            why=(
                f"Mapped coverage is already high ({mapped_ratio:.1f}%) but the "
                "end-to-end gain is still small."
            ),
            try_next=(
                "Look past mapping itself: allocator churn, eval batching, or "
                "decode access pattern may be the real bottleneck."
            ),
            priority=72,
            confidence=confidence,
            confidence_why=(
                f"coverage is already {mapped_ratio:.1f}% so the remaining "
                "bottleneck is likely outside the mapping decision itself"
            ),
        )

    if trend_regressions:
        confidence = 0.55 + min(0.2, 0.08 * len(trend_regressions))
        add(
            route="Stability Audit",
            why=(
                "Recent trend warnings suggest the bottleneck is moving or the "
                "run is not yet stable enough to trust."
            ),
            try_next=(
                "Repeat on one dense safetensors model and one quantized GGUF "
                "model before broadening the change."
            ),
            priority=68,
            confidence=confidence,
            confidence_why=f"{len(trend_regressions)} trend regression warning(s) fired",
        )

    if not suggestions:
        add(
            route="Broaden the Matrix",
            why=(
                "This run does not isolate one dominant bottleneck strongly "
                "enough to point at a single fix."
            ),
            try_next=(
                "Compare a dense safetensors model and a quantized GGUF model "
                "across warm and cold cache modes to force separation."
            ),
            priority=1,
            confidence=0.35,
            confidence_why="no single bottleneck has separated itself clearly yet",
        )

    if (
        top_reason_bytes is not None
        and top_reason_bytes not in {
            "quantized_conversion",
            "dtype_conversion",
            "misaligned_offset",
        }
        and top_reason_bytes != top_reason
    ):
        suggestions = [
            item
            for item in suggestions
            if item["route"] != "Coverage-Targeted Fallback Work"
        ]
        seen_routes.discard("Coverage-Targeted Fallback Work")
        add(
            route="Coverage-Targeted Fallback Work",
            why=(
                f"The heaviest fallback by bytes is {top_reason_bytes} "
                f"({_bytes_to_mib(top_byte_count):.1f} MiB materialized)."
            ),
            try_next=(
                "Profile that byte-heavy fallback path directly and see if a small "
                "fast path buys more mapped coverage than broader changes."
            ),
            priority=82
            + _fallback_priority_bonus(fallback_rank_map.get(top_reason_bytes)),
            confidence=0.6
            + min(0.2, _bytes_to_mib(top_byte_count) / 512.0)
            + (
                0.1
                if isinstance(mapped_ratio, (int, float)) and mapped_ratio < 80.0
                else 0.0
            ),
            confidence_why=(
                f"{top_reason_bytes} dominates copied fallback bytes at "
                f"{_bytes_to_mib(top_byte_count):.1f} MiB of materialized data"
            ),
        )

    suggestions.sort(key=lambda item: (-item["priority"], item["route"]))
    return suggestions[:4]


def _record_model_class(record: dict) -> str:
    model_class = record.get("model_class")
    if model_class:
        return str(model_class)
    fmt = record.get("format")
    fallback_reasons = record.get("mmap_coverage_fallback_reasons")
    weight_class = _infer_weight_class(str(record.get("file") or ""), fmt, fallback_reasons)
    return _model_class_name(fmt, weight_class)


def _latest_record_catalog(records: list[dict]) -> list[dict]:
    catalog: list[dict] = []
    seen_files: set[str] = set()
    for record in reversed(records):
        file_path = str(record.get("file") or "")
        if not file_path or file_path in seen_files:
            continue
        if not Path(file_path).exists():
            continue
        seen_files.add(file_path)
        catalog.append(record)
    return catalog


def _find_catalog_target(
    catalog: list[dict], *, fmt: str, model_class: str
) -> dict | None:
    for record in catalog:
        if str(record.get("format") or "unknown") != fmt:
            continue
        if _record_model_class(record) != model_class:
            continue
        return record
    return None


def _build_catalog_record(
    file_path: str,
    fmt: str | None,
    *,
    discovered_info: dict | None = None,
) -> dict:
    metadata = _infer_model_metadata(
        file_path,
        fmt,
        discovered_info=discovered_info,
    )
    return {"file": file_path, **metadata}


def _build_input_catalog(
    *,
    prior_history: list[dict],
    file_paths: list[str],
    requested_format: str,
    discovered_formats: dict[str, str],
    discovered_details: dict[str, dict],
) -> list[dict]:
    records = list(prior_history)
    for file_path in file_paths:
        fmt = (
            discovered_formats.get(file_path)
            if requested_format == "auto" and discovered_formats.get(file_path) is not None
            else _resolve_format(file_path, requested_format)
        )
        records.append(
            _build_catalog_record(
                file_path,
                fmt,
                discovered_info=discovered_details.get(file_path),
            )
        )
    return _latest_record_catalog(records)


def _resolve_input_files(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str], dict[str, dict], list[dict]]:
    file_paths = list(args.files)
    discovered_formats: dict[str, str] = {}
    discovered_details: dict[str, dict] = {}
    discovered_candidates: list[dict] = []
    if not args.discover_models:
        return file_paths, discovered_formats, discovered_details, discovered_candidates

    discover_roots = (
        [Path(root).expanduser().resolve() for root in args.discover_root]
        if args.discover_root
        else [Path.cwd()]
    )
    candidates = find_candidates(
        roots=discover_roots,
        min_size_bytes=int(args.discover_min_size_mib * 1024 * 1024),
        one_per_dir=not args.discover_all_per_dir,
        include_ollama_blobs=args.discover_ollama_blobs,
        model_id_regex=args.discover_model_id_regex,
    )
    limit = max(args.discover_max_results, 0)
    selected = candidates[:limit]
    discovered_candidates = [
        {
            "path": candidate.path,
            "format": candidate.format,
            "model_id": candidate.model_id,
            "size_bytes": candidate.size_bytes,
            "source": candidate.source,
            "parent_dir": candidate.parent_dir,
        }
        for candidate in selected
    ]
    for candidate in selected:
        if candidate.format:
            discovered_formats[candidate.path] = candidate.format
        discovered_details[candidate.path] = {
            "source": candidate.source,
            "model_id": candidate.model_id,
            "size_bytes": candidate.size_bytes,
            "parent_dir": candidate.parent_dir,
        }
    existing = set(file_paths)
    for candidate in selected:
        if candidate.path not in existing:
            file_paths.append(candidate.path)
            existing.add(candidate.path)
    return file_paths, discovered_formats, discovered_details, discovered_candidates


def _select_matrix_targets(catalog: list[dict]) -> tuple[list[tuple[str, dict]], list[str]]:
    selected_targets: list[tuple[str, dict]] = []
    notes: list[str] = []
    for title, fmt, model_class in [
        ("Dense Safetensors", "safetensors", "dense_safetensors"),
        ("Quantized GGUF", "gguf", "quantized_gguf"),
    ]:
        target = _find_catalog_target(catalog, fmt=fmt, model_class=model_class)
        if target is None:
            notes.append(
                f"No {title.lower()} target is available yet for the compact policy matrix."
            )
            continue
        selected_targets.append((title, target))
    return selected_targets, notes


def _write_dense_safetensors_fixture(path: Path) -> None:
    tensor_payloads = {
        "dense.weight": struct.pack("<6f", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        "dense.bias": struct.pack("<8h", 3, -1, 7, 2, -5, 8, 13, -21),
    }
    offset = 0
    header = {
        "__metadata__": {
            "fixture": "golden_dense_safetensors",
        }
    }
    for name, payload in tensor_payloads.items():
        dtype = "F32" if name.endswith("weight") else "I16"
        shape = [2, 3] if name.endswith("weight") else [8]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    payload = b"".join(tensor_payloads.values())
    with path.open("wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(payload)


def _write_misaligned_safetensors_fixture(path: Path) -> None:
    header = b'{"x":{"dtype":"I16","shape":[2],"data_offsets":[1,5]}}'
    payload = b"\x00" + struct.pack("<2h", 123, -456)
    with path.open("wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
        f.write(payload)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _write_quantized_gguf_fixture(path: Path) -> None:
    tensor_name = b"quant.weight"
    header = bytearray()
    header.extend(b"GGUF")
    header.extend(struct.pack("<I", 3))
    header.extend(struct.pack("<Q", 1))  # tensor_count
    header.extend(struct.pack("<Q", 0))  # metadata_kv_count
    header.extend(struct.pack("<Q", len(tensor_name)))
    header.extend(tensor_name)
    header.extend(struct.pack("<I", 1))  # ndim
    header.extend(struct.pack("<Q", 32))  # dim0
    header.extend(struct.pack("<I", 2))  # GGUF_TYPE_Q4_0
    header.extend(struct.pack("<Q", 0))  # data offset from data section
    padded_header_size = _align_up(len(header), 32)
    header.extend(b"\x00" * (padded_header_size - len(header)))
    block = struct.pack("<e", 0.5) + bytes(range(16))
    with path.open("wb") as f:
        f.write(header)
        f.write(block)


def _build_golden_matrix_targets(temp_root: Path) -> list[tuple[str, dict]]:
    dense_path = temp_root / "golden_dense.safetensors"
    quantized_path = temp_root / "golden_quantized_q4_0.gguf"
    misaligned_path = temp_root / "golden_misaligned.safetensors"
    _write_dense_safetensors_fixture(dense_path)
    _write_quantized_gguf_fixture(quantized_path)
    _write_misaligned_safetensors_fixture(misaligned_path)
    return [
        (
            "Dense Safetensors",
            {
                "file": str(dense_path),
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "label": "golden_dense_safetensors",
            },
        ),
        (
            "Quantized GGUF",
            {
                "file": str(quantized_path),
                "format": "gguf",
                "model_class": "quantized_gguf",
                "label": "golden_quantized_q4_0",
            },
        ),
        (
            "Misaligned Safetensors",
            {
                "file": str(misaligned_path),
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "label": "golden_misaligned_safetensors",
            },
        ),
    ]


def _compact_policy_matrix_cases(
    args: argparse.Namespace,
) -> tuple[list[dict], list[str]]:
    threshold_values = _parse_optional_byte_threshold_list(
        args.policy_matrix_thresholds or "none,65536"
    )
    prefetch_values = _parse_prefetch_strategy_list(
        args.policy_matrix_prefetch_strategies or "sequential,willneed"
    )
    cases: list[dict] = []
    labels_seen: set[str] = set()
    for prefetch_strategy in prefetch_values:
        for threshold in threshold_values:
            knobs = _build_policy_knobs(
                threshold,
                mmap_prefetch_strategy=prefetch_strategy,
                mmap_hotset_promotion_top_k=args.mmap_hotset_promotion_top_k,
                mmap_hotset_promotion_min_bytes=args.mmap_hotset_promotion_min_bytes,
            )
            signature = _policy_signature(knobs)
            label = signature if signature != "default" else "default"
            if label in labels_seen:
                continue
            labels_seen.add(label)
            cases.append(
                {
                    "label": label,
                    "policy_knobs": knobs,
                }
            )
    return cases, ["warm", "cold-best-effort"]


def _build_experiment_command(
    *,
    script_path: str,
    file_path: str,
    fmt: str,
    cache_mode: str,
    args: argparse.Namespace,
    prefer_synth_decode: bool,
    policy_knobs: dict | None = None,
    requested_policy_mode: str | None = None,
    policy_matrix_preset: str | None = None,
    policy_matrix_case: str | None = None,
    golden_matrix_preset: str | None = None,
    golden_matrix_case: str | None = None,
    demo_parent: str | None = None,
    demo_case: str | None = None,
    history_json_path: str | None = None,
    include_baseline_flags: bool = True,
) -> str:
    policy_knobs = policy_knobs or _build_policy_knobs(
        args.mmap_small_tensor_copy_max_bytes,
        mmap_prefetch_strategy=args.mmap_prefetch_strategy,
        mmap_hotset_promotion_top_k=args.mmap_hotset_promotion_top_k,
        mmap_hotset_promotion_min_bytes=args.mmap_hotset_promotion_min_bytes,
    )
    threshold = policy_knobs.get("mmap_small_tensor_copy_max_bytes")
    prefetch_strategy = str(policy_knobs.get("mmap_prefetch_strategy") or "sequential")
    hotset_top_k = _normalize_hotset_promotion_top_k(
        policy_knobs.get("mmap_hotset_promotion_top_k")
    )
    hotset_min_bytes = _normalize_hotset_promotion_min_bytes(
        policy_knobs.get("mmap_hotset_promotion_min_bytes")
    )
    requested_policy_mode = requested_policy_mode or args.policy_mode
    parts = [
        sys.executable,
        script_path,
        file_path,
        "--format",
        fmt if fmt != "unknown" else "auto",
        "--runs",
        str(max(args.runs, 1)),
        "--warmup-runs",
        str(max(args.warmup_runs, 0)),
        "--attempts",
        str(max(args.attempts, 1)),
        "--cache-mode",
        cache_mode,
        "--coverage-probe",
        "--interleave-modes",
    ]
    if requested_policy_mode != "manual":
        parts.extend(["--policy-mode", requested_policy_mode])
    default_history_json = str(Path(script_path).with_name("load_mmap_history.jsonl"))
    resolved_history_json = (
        str(history_json_path) if history_json_path is not None else str(args.history_json)
    )
    if resolved_history_json != default_history_json:
        parts.extend(["--history-json", resolved_history_json])
    if args.min_pass is not None:
        parts.extend(["--min-pass", str(args.min_pass)])
    if args.max_invalid_attempts is not None:
        parts.extend(["--max-invalid-attempts", str(args.max_invalid_attempts)])
    if args.gguf_nvfp4_compat:
        parts.append("--gguf-nvfp4-compat")
    if threshold is not None:
        parts.extend(
            [
                "--mmap-small-tensor-copy-max-bytes",
                str(int(threshold)),
            ]
        )
    if hotset_top_k is not None:
        parts.extend(
            [
                "--mmap-hotset-promotion-top-k",
                str(int(hotset_top_k)),
            ]
        )
    if hotset_min_bytes is not None:
        parts.extend(
            [
                "--mmap-hotset-promotion-min-bytes",
                str(int(hotset_min_bytes)),
            ]
        )
    if prefetch_strategy != "sequential":
        parts.extend(["--mmap-prefetch-strategy", prefetch_strategy])
    if policy_matrix_preset is not None:
        parts.extend(["--policy-matrix-parent", policy_matrix_preset])
    if policy_matrix_case is not None:
        parts.extend(["--policy-matrix-case", policy_matrix_case])
    if golden_matrix_preset is not None:
        parts.extend(["--golden-matrix-parent", golden_matrix_preset])
    if golden_matrix_case is not None:
        parts.extend(["--golden-matrix-case", golden_matrix_case])
    if demo_parent is not None:
        parts.extend(["--demo-parent", demo_parent])
    if demo_case is not None:
        parts.extend(["--demo-case", demo_case])
    if include_baseline_flags and args.baseline_run_id is not None:
        parts.extend(["--baseline-run-id", args.baseline_run_id])
    if include_baseline_flags and args.baseline_git_head is not None:
        parts.extend(["--baseline-git-head", args.baseline_git_head])
    if args.fallback_priority_basis != "bytes":
        parts.extend(["--fallback-priority-basis", args.fallback_priority_basis])
    if args.reject_noisy_attempts:
        parts.append("--reject-noisy-attempts")
    else:
        parts.append("--no-reject-noisy-attempts")
    if args.debug_io:
        parts.append("--debug-io")
    if args.show_trials:
        parts.append("--show-trials")
    if cache_mode == "cold-best-effort":
        parts.append("--validate-cold-cache")
    if args.history_window != 20:
        parts.extend(["--history-window", str(args.history_window)])
    if args.trend_min_samples != 5:
        parts.extend(["--trend-min-samples", str(args.trend_min_samples)])
    if args.trend_mad_mult != 3.0:
        parts.extend(["--trend-mad-mult", str(args.trend_mad_mult)])
    if args.trend_hard_fail:
        parts.append("--trend-hard-fail")
    if args.decode_phase != "soft":
        parts.extend(["--decode-phase", args.decode_phase])
    if args.decode_cv_gate_pct is not None:
        parts.extend(["--decode-cv-gate-pct", str(args.decode_cv_gate_pct)])
    if args.decode_regression_gate_pct != 3.0:
        parts.extend(
            ["--decode-regression-gate-pct", str(args.decode_regression_gate_pct)]
        )
    if args.decode_aa_noise_floor:
        parts.append("--decode-aa-noise-floor")
    if args.decode_aa_multiplier != 1.0:
        parts.extend(["--decode-aa-multiplier", str(args.decode_aa_multiplier)])
    if args.decode_aa_margin_pct != 0.5:
        parts.extend(["--decode-aa-margin-pct", str(args.decode_aa_margin_pct)])

    use_llama_bench = (
        (not prefer_synth_decode)
        and args.decode_llama_bench_binary is not None
        and fmt == "gguf"
    )
    if use_llama_bench:
        parts.extend(
            [
                "--decode-llama-bench-binary",
                str(Path(args.decode_llama_bench_binary).expanduser().resolve()),
                "--decode-llama-bench-prompt-tokens",
                str(args.decode_llama_bench_prompt_tokens),
                "--decode-llama-bench-gen-tokens",
                str(args.decode_llama_bench_gen_tokens),
                "--decode-llama-bench-repetitions",
                str(args.decode_llama_bench_repetitions),
                "--decode-llama-bench-gpu-layers",
                str(args.decode_llama_bench_gpu_layers),
                "--decode-llama-bench-depth",
                str(args.decode_llama_bench_depth),
            ]
        )
        if args.decode_llama_bench_threads is not None:
            parts.extend(
                ["--decode-llama-bench-threads", str(args.decode_llama_bench_threads)]
            )
        if args.decode_llama_bench_keep_warmup:
            parts.append("--decode-llama-bench-keep-warmup")
    elif (not prefer_synth_decode) and args.decode_cmd is not None:
        parts.extend(["--decode-cmd", args.decode_cmd, "--decode-runs", str(args.decode_runs)])
    else:
        parts.extend(
            [
                "--decode-synth",
                "--decode-synth-tokens",
                str(args.decode_synth_tokens),
                "--decode-synth-max-elems",
                str(args.decode_synth_max_elems),
                "--decode-synth-repeats",
                str(args.decode_synth_repeats),
            ]
        )

    return " ".join(shlex.quote(part) for part in parts)


def _build_next_experiments(
    summary_payload: dict,
    route_suggestions: list[dict],
    *,
    prior_history: list[dict],
    script_path: str,
    args: argparse.Namespace,
) -> tuple[list[dict], list[str]]:
    route_names = {item["route"] for item in route_suggestions}
    catalog = _latest_record_catalog(prior_history + [summary_payload])
    experiments: list[dict] = []
    notes: list[str] = []
    selected_targets: list[tuple[str, dict]] = []

    for title, fmt, model_class in [
        ("Dense Safetensors", "safetensors", "dense_safetensors"),
        ("Quantized GGUF", "gguf", "quantized_gguf"),
    ]:
        current_matches = (
            str(summary_payload.get("format") or "unknown") == fmt
            and str(summary_payload.get("model_class") or "unknown") == model_class
        )
        target = summary_payload if current_matches else _find_catalog_target(
            catalog, fmt=fmt, model_class=model_class
        )
        if target is None:
            notes.append(f"No {title.lower()} target is available yet for the follow-up matrix.")
            continue
        selected_targets.append((title, target))

    if not selected_targets:
        notes.append(
            "Falling back to the current file because the history does not yet "
            "contain a dense safetensors or quantized GGUF example."
        )
        selected_targets.append(("Current File", summary_payload))

    for title, target in selected_targets:
        target_fmt = str(target.get("format") or summary_payload.get("format") or "unknown")
        target_file = str(target.get("file") or summary_payload.get("file") or "")
        prefer_synth_decode = target_fmt != "gguf"
        for cache_mode in ["warm", "cold-best-effort"]:
            goal_parts: list[str] = []
            if title == "Quantized GGUF":
                if "Quantized Direct Map" in route_names:
                    goal_parts.append("Stress quantized fallback coverage.")
                else:
                    goal_parts.append("Measure quantized mmap coverage and decode parity.")
            elif title == "Dense Safetensors":
                goal_parts.append("Keep a dense baseline for comparison.")
            else:
                goal_parts.append("Re-run the current file with a cleaner matrix.")

            if cache_mode == "warm":
                if {"Hotset Promotion", "Adaptive Default Policy"} & route_names:
                    goal_parts.append("Warm cache is the fastest way to inspect decode regressions.")
                else:
                    goal_parts.append("Warm cache gives the lower-noise steady-state view.")
            else:
                if "Prefetch / Readahead Tuning" in route_names:
                    goal_parts.append("Cold cache is where readahead and fault pressure show up.")
                else:
                    goal_parts.append("Cold cache keeps the load-path signal visible.")

            experiments.append(
                {
                    "title": f"{title} / {cache_mode}",
                    "goal": " ".join(goal_parts),
                    "command": _build_experiment_command(
                        script_path=script_path,
                        file_path=target_file,
                        fmt=target_fmt,
                        cache_mode=cache_mode,
                        args=args,
                        prefer_synth_decode=prefer_synth_decode,
                        requested_policy_mode=(
                            "manual" if args.policy_mode == "auto" else args.policy_mode
                        ),
                    ),
                    "target_file": target_file,
                    "target_label": str(target.get("label") or Path(target_file).name),
                    "target_class": _record_model_class(target),
                }
            )

    return experiments[:4], notes


def _demo_fixture_history_path() -> Path:
    return Path(__file__).resolve().parent / "testdata" / "load_mmap_demo_history.jsonl"


def _prime_physics_demo_fixture_history_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "testdata"
        / "load_mmap_prime_physics_demo_history.jsonl"
    )


def _demo_session_id(preset: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slugify_identifier(preset)}-{stamp}-{os.getpid()}"


def _demo_policy_family(record: dict | str | None) -> str:
    signature: str | None = None
    if isinstance(record, dict):
        signature = str(
            record.get("effective_policy_signature")
            or record.get("policy_signature")
            or record.get("effective_policy_mode")
            or ""
        )
    elif record is not None:
        signature = str(record)
    signature = signature or ""
    return "copy" if signature == "copy" or signature.startswith("copy") else "non-copy"


def _demo_policy_composite_score(record: dict) -> float:
    score = 0.0
    for metric_key, weight in _DEMO_POLICY_SCORE_WEIGHTS.items():
        value = record.get(metric_key)
        if isinstance(value, (int, float)):
            score += float(value) * float(weight)
    return round(score, 2)


def _classify_demo_choice_quality(
    *,
    auto_record: dict,
    best_seeded_record: dict,
) -> dict:
    auto_score = _demo_policy_composite_score(auto_record)
    best_score = _demo_policy_composite_score(best_seeded_record)
    tolerance = max(0.5, abs(best_score) * 0.05)
    same_family = _demo_policy_family(auto_record) == _demo_policy_family(best_seeded_record)
    auto_signature = str(
        auto_record.get("effective_policy_signature")
        or auto_record.get("policy_signature")
        or "unknown"
    )
    best_signature = str(
        best_seeded_record.get("effective_policy_signature")
        or best_seeded_record.get("policy_signature")
        or "unknown"
    )
    if auto_signature == best_signature:
        verdict = "match"
    elif same_family and abs(best_score - auto_score) <= tolerance:
        verdict = "near-match"
    else:
        verdict = "miss"
    return {
        "verdict": verdict,
        "auto_score": auto_score,
        "best_seeded_score": best_score,
        "score_gap": round(auto_score - best_score, 2),
        "tolerance": round(tolerance, 2),
        "same_family": same_family,
    }


def _demo_summary_base(
    *,
    demo_preset: str,
    history_source: str,
    targets: list[dict],
    steps: list[dict],
) -> dict:
    return {
        "demo_preset": demo_preset,
        "demo_session_id": _demo_session_id(demo_preset),
        "history_source": history_source,
        "targets": targets,
        "steps": steps,
        "verdict": {},
        "scorecard": {},
        "evidence": {},
    }


def _build_prime_physics_corpus_cases() -> list[dict]:
    return [
        {
            "case_id": "negative_results_persist",
            "title": "Negative Results Must Persist",
            "claim": (
                "Rejected stories should stay live in memory so the loop stops "
                "reopening them as if they were new."
            ),
            "history_bucket_key": "prime_negative_results_persist",
            "expected_route_id": "stability_audit",
            "loop_noticed": (
                "The loop keeps earlier refutations visible and asks for a "
                "stability audit before reopening the mechanism story."
            ),
            "scope_note": (
                "Negative evidence should remain first-class even when fresher "
                "language sounds more vivid."
            ),
            "refuted_stories": [
                "k* ~ sqrt(M)",
                "2p resonance",
                "large template-specific bonus",
            ],
            "refuted_route_id": "coverage_targeted_fallback_work",
        },
        {
            "case_id": "single_instance_scope",
            "title": "Single-Instance Asymmetry Stays Narrow",
            "claim": (
                "A verified canonical pair is not yet a general family theorem."
            ),
            "history_bucket_key": "prime_single_instance_scope",
            "expected_route_id": "broaden_the_matrix",
            "loop_noticed": (
                "The loop keeps the asymmetry local and asks for broader family "
                "coverage before generalizing."
            ),
            "scope_note": (
                "Single-instance wins should expand the matrix, not inflate the claim."
            ),
            "refuted_stories": [],
            "refuted_route_id": "coverage_targeted_fallback_work",
        },
        {
            "case_id": "formalization_scope_guardrail",
            "title": "Formalization Increases Trust, Not Claim Size",
            "claim": (
                "Lean and Agda proof surfaces strengthen local theorems without "
                "automatically proving the global density interpretation."
            ),
            "history_bucket_key": "prime_formalization_scope_guardrail",
            "expected_route_id": "stability_audit",
            "loop_noticed": (
                "The loop separates proof-surface trust from the larger public "
                "story and keeps the global claim scoped."
            ),
            "scope_note": (
                "Local rigor should increase trust exactly where it applies, "
                "not everywhere nearby."
            ),
            "refuted_stories": [
                "formalization volume alone proves the larger mechanism",
            ],
            "refuted_route_id": "coverage_targeted_fallback_work",
        },
    ]


def _sorted_route_outcome_stats(route_stats: list[dict] | dict[str, dict]) -> list[dict]:
    values = route_stats.values() if isinstance(route_stats, dict) else route_stats
    return sorted(
        values,
        key=lambda item: (
            -(
                item.get("success_rate")
                if item.get("success_rate") is not None
                else -1.0
            ),
            -int(item.get("scored_samples", 0) or 0),
            item["route_id"],
        ),
    )


def _summarize_prime_physics_corpus_case(case: dict, records: list[dict]) -> dict:
    bucket_records = [
        record
        for record in records
        if str(record.get("history_bucket_key") or "") == case["history_bucket_key"]
    ]
    route_stats = _sorted_route_outcome_stats(_route_outcome_stats(bucket_records))
    by_route = {entry["route_id"]: entry for entry in route_stats}
    expected_route = by_route.get(case["expected_route_id"])
    chosen_route = route_stats[0] if route_stats else None
    top_route_ids = [entry["route_id"] for entry in route_stats[:2]]
    if chosen_route is not None and chosen_route["route_id"] == case["expected_route_id"]:
        verdict = "pass"
    elif case["expected_route_id"] in top_route_ids:
        verdict = "mixed"
    else:
        verdict = "fail"
    chosen_summary = (
        _format_route_history_summary(chosen_route, scope="bucket")
        if chosen_route is not None
        else None
    )
    expected_summary = (
        _format_route_history_summary(expected_route, scope="bucket")
        if expected_route is not None
        else None
    )
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "claim": case["claim"],
        "loop_noticed": case["loop_noticed"],
        "scope_note": case["scope_note"],
        "verdict": verdict,
        "history_bucket_key": case["history_bucket_key"],
        "history_samples": len(bucket_records),
        "expected_route_id": case["expected_route_id"],
        "expected_route": _ROUTE_DISPLAY_BY_ID.get(
            case["expected_route_id"], case["expected_route_id"]
        ),
        "expected_route_history": expected_summary,
        "chosen_route_id": chosen_route.get("route_id") if chosen_route is not None else None,
        "chosen_route": chosen_route.get("route") if chosen_route is not None else None,
        "chosen_route_history": chosen_summary,
        "refuted_stories": list(case.get("refuted_stories") or []),
        "top_routes": [
            {
                "route_id": entry["route_id"],
                "route": entry["route"],
                "success_rate": entry.get("success_rate"),
                "samples": entry["samples"],
                "win": entry["win"],
                "loss": entry["loss"],
                "mixed": entry["mixed"],
                "inconclusive": entry["inconclusive"],
            }
            for entry in route_stats[:3]
        ],
    }


def _build_prime_physics_corpus_fixture_summary() -> dict:
    fixture_path = _prime_physics_demo_fixture_history_path()
    records = _load_history_jsonl(fixture_path)
    cases = [
        _summarize_prime_physics_corpus_case(case, records)
        for case in _build_prime_physics_corpus_cases()
    ]
    passed = sum(1 for case in cases if case["verdict"] == "pass")
    return {
        "fixture_path": str(fixture_path),
        "cases": cases,
        "pass_count": passed,
        "total_cases": len(cases),
        "match_rate": round(passed / len(cases), 3) if cases else 0.0,
    }


def _prime_physics_foreign_pressure_records() -> list[dict]:
    records = []
    for idx in range(1, 5):
        records.append(
            {
                "attempted_route": "Coverage-Targeted Fallback Work",
                "attempted_route_id": "coverage_targeted_fallback_work",
                "cache_mode": "session",
                "comparison": {"auto_outcome": "win"},
                "decode_mode": "none",
                "file": f"/demo/foreign_scope/coverage_bias_{idx}.md",
                "format": "external_corpus",
                "history_bucket": {
                    "cache_mode": "session",
                    "decode_mode": "none",
                    "format": "external_corpus",
                    "model_class": "prime_physics_claim",
                    "policy": "claim_scope",
                },
                "history_bucket_key": f"foreign_scope_pressure_{idx}",
                "model_class": "prime_physics_claim",
                "route_outcome": "win",
                "run_id": f"foreign-coverage-bias-{idx}",
                "timestamp": f"2026-04-{idx:02d}T12:00:00+00:00",
            }
        )
    return records


def _route_stats_by_id(records: list[dict]) -> dict[str, dict]:
    return {
        entry["route_id"]: entry
        for entry in _sorted_route_outcome_stats(_route_outcome_stats(records))
    }


def _history_summary_or_none(route_stats: dict | None, *, scope: str) -> str | None:
    if route_stats is None:
        return None
    return _format_route_history_summary(route_stats, scope=scope)


def _summarize_prime_physics_persistence_stage(
    *,
    case: dict,
    records: list[dict],
    stage_id: str,
    title: str,
    memory_mode: str,
    scope: str,
) -> dict:
    route_stats = _sorted_route_outcome_stats(_route_outcome_stats(records))
    by_route = {entry["route_id"]: entry for entry in route_stats}
    expected_route = by_route.get(case["expected_route_id"])
    refuted_route_id = str(case.get("refuted_route_id") or "")
    refuted_route = by_route.get(refuted_route_id)
    top_route_ids = [entry["route_id"] for entry in route_stats[:2]]

    if not route_stats:
        suppression_state = "unknown"
        suppression_reason = "No prior history exists yet, so nothing has been demoted."
    elif refuted_route is not None and refuted_route_id in top_route_ids:
        suppression_state = "resurfaced"
        suppression_reason = (
            "Unrelated wins keep the refuted route in the active candidate pool."
        )
    elif refuted_route is not None:
        suppression_state = "suppressed"
        suppression_reason = (
            "Local losses keep the refuted route out of the active candidate pool."
        )
    else:
        suppression_state = "unknown"
        suppression_reason = (
            "The refuted route has not accumulated enough comparable evidence yet."
        )

    return {
        "stage_id": stage_id,
        "title": title,
        "memory_mode": memory_mode,
        "scope": scope,
        "history_samples": len(records),
        "chosen_route_id": route_stats[0]["route_id"] if route_stats else None,
        "chosen_route": route_stats[0]["route"] if route_stats else None,
        "expected_route_id": case["expected_route_id"],
        "expected_route": _ROUTE_DISPLAY_BY_ID.get(
            case["expected_route_id"], case["expected_route_id"]
        ),
        "expected_route_history": _history_summary_or_none(
            expected_route, scope=scope if scope != "none" else "fresh"
        ),
        "refuted_route_id": refuted_route_id,
        "refuted_route": _ROUTE_DISPLAY_BY_ID.get(refuted_route_id, refuted_route_id),
        "refuted_route_history": _history_summary_or_none(
            refuted_route, scope=scope if scope != "none" else "fresh"
        ),
        "suppression_state": suppression_state,
        "suppression_reason": suppression_reason,
        "top_routes": [
            {
                "route_id": entry["route_id"],
                "route": entry["route"],
                "success_rate": entry.get("success_rate"),
                "samples": entry["samples"],
                "win": entry["win"],
                "loss": entry["loss"],
                "mixed": entry["mixed"],
                "inconclusive": entry["inconclusive"],
            }
            for entry in route_stats[:3]
        ],
    }


def _build_persistence_memory_demo_summary(args: argparse.Namespace) -> dict:
    del args
    fixture_path = _prime_physics_demo_fixture_history_path()
    local_records = _load_history_jsonl(fixture_path)
    foreign_records = _prime_physics_foreign_pressure_records()
    cases = []
    session_suppressed = 0
    persistent_scoped_suppressed = 0
    persistent_unscoped_resurfaced = 0

    for case in _build_prime_physics_corpus_cases():
        bucket_records = [
            record
            for record in local_records
            if str(record.get("history_bucket_key") or "") == case["history_bucket_key"]
        ]
        fresh = _summarize_prime_physics_persistence_stage(
            case=case,
            records=[],
            stage_id="fresh",
            title="Fresh Memory",
            memory_mode="fresh",
            scope="none",
        )
        session = _summarize_prime_physics_persistence_stage(
            case=case,
            records=bucket_records,
            stage_id="session",
            title="Session Memory",
            memory_mode="session",
            scope="bucket",
        )
        persistent_unscoped = _summarize_prime_physics_persistence_stage(
            case=case,
            records=[*bucket_records, *foreign_records],
            stage_id="persistent_unscoped",
            title="Persistent Memory (Unscoped)",
            memory_mode="persistent",
            scope="global",
        )
        persistent_scoped = _summarize_prime_physics_persistence_stage(
            case=case,
            records=bucket_records,
            stage_id="persistent_scoped",
            title="Persistent Memory (Scoped)",
            memory_mode="persistent",
            scope="bucket",
        )

        if session["suppression_state"] == "suppressed":
            session_suppressed += 1
        if persistent_scoped["suppression_state"] == "suppressed":
            persistent_scoped_suppressed += 1
        if persistent_unscoped["suppression_state"] == "resurfaced":
            persistent_unscoped_resurfaced += 1

        stages = [fresh, session, persistent_unscoped, persistent_scoped]
        case_verdict = (
            "pass"
            if session["suppression_state"] == "suppressed"
            and persistent_scoped["suppression_state"] == "suppressed"
            and persistent_unscoped["suppression_state"] == "resurfaced"
            else "fail"
        )
        cases.append(
            {
                "case_id": case["case_id"],
                "title": case["title"],
                "claim": case["claim"],
                "loop_noticed": case["loop_noticed"],
                "scope_note": case["scope_note"],
                "refuted_stories": list(case.get("refuted_stories") or []),
                "verdict": case_verdict,
                "stages": stages,
            }
        )

    total_cases = len(cases)
    summary = _demo_summary_base(
        demo_preset="persistence-memory",
        history_source="fixture",
        targets=[
            {"case_id": case["case_id"], "title": case["title"]}
            for case in _build_prime_physics_corpus_cases()
        ],
        steps=[
            {"name": "load_prime_fixture_history", "status": "ok", "fixture": str(fixture_path)},
            {
                "name": "inject_foreign_pressure",
                "status": "ok",
                "records": len(foreign_records),
            },
            {
                "name": "simulate_memory_modes",
                "status": "ok",
                "case_count": total_cases,
            },
        ],
    )
    verdict_ok = (
        session_suppressed == total_cases
        and persistent_scoped_suppressed == total_cases
        and persistent_unscoped_resurfaced == total_cases
    )
    summary["verdict"] = {
        "status": "pass" if verdict_ok else "fail",
        "summary": (
            "Session memory suppresses locally refuted routes, unscoped persistence "
            "can let them resurface, and scoped persistence restores the guardrail."
        ),
    }
    summary["scorecard"] = {
        "cases": cases,
        "session_suppressed": session_suppressed,
        "persistent_scoped_suppressed": persistent_scoped_suppressed,
        "persistent_unscoped_resurfaced": persistent_unscoped_resurfaced,
        "total_cases": total_cases,
        "scoped_suppression_rate": round(
            persistent_scoped_suppressed / total_cases, 3
        )
        if total_cases
        else 0.0,
        "unscoped_resurfaced_rate": round(
            persistent_unscoped_resurfaced / total_cases, 3
        )
        if total_cases
        else 0.0,
    }
    summary["evidence"] = {
        "fixture_path": str(fixture_path),
        "foreign_pressure_records": len(foreign_records),
        "foreign_pressure_route_id": "coverage_targeted_fallback_work",
    }
    return summary


def _summarize_candidate_reasons(candidate: dict | None, *, limit: int = 3) -> list[str]:
    if not isinstance(candidate, dict):
        return []
    return [str(reason) for reason in (candidate.get("reasons") or [])[:limit]]


def _find_selected_auto_candidate(auto_policy_decision: dict | None) -> dict | None:
    if not isinstance(auto_policy_decision, dict):
        return None
    selected_name = str(auto_policy_decision.get("selected_candidate") or "")
    for candidate in auto_policy_decision.get("candidates") or []:
        if str(candidate.get("name") or "") == selected_name:
            return candidate
    candidates = auto_policy_decision.get("candidates") or []
    return candidates[0] if candidates else None


def _adaptation_seed_policy_cases() -> list[dict]:
    return [
        {
            "label": "copy",
            "requested_policy_mode": "copy",
            "policy_knobs": _build_policy_knobs(None),
        },
        {
            "label": "mapped",
            "requested_policy_mode": "mapped",
            "policy_knobs": _build_policy_knobs(None),
        },
        {
            "label": "hybrid_small_tensor_copy",
            "requested_policy_mode": "hybrid",
            "policy_knobs": _build_policy_knobs(65536),
        },
        {
            "label": "hybrid_hotset_willneed",
            "requested_policy_mode": "hybrid",
            "policy_knobs": _build_policy_knobs(
                None,
                mmap_prefetch_strategy="willneed",
                mmap_hotset_promotion_top_k=1,
            ),
        },
    ]


def _build_adaptation_demo_targets(
    *,
    file_paths: list[str],
    requested_format: str,
    discovered_formats: dict[str, str],
    discovered_details: dict[str, dict],
    temp_root: Path,
) -> tuple[list[dict], list[str]]:
    golden_targets = {title: target for title, target in _build_golden_matrix_targets(temp_root)}
    notes: list[str] = []
    external_catalog = _build_input_catalog(
        prior_history=[],
        file_paths=file_paths,
        requested_format=requested_format,
        discovered_formats=discovered_formats,
        discovered_details=discovered_details,
    )

    dense_target = _find_catalog_target(
        external_catalog, fmt="safetensors", model_class="dense_safetensors"
    )
    if dense_target is None:
        dense_target = golden_targets["Dense Safetensors"]
        if file_paths:
            notes.append("Falling back to the built-in dense safetensors fixture.")

    quantized_target = _find_catalog_target(
        external_catalog, fmt="gguf", model_class="quantized_gguf"
    )
    if quantized_target is None:
        quantized_target = golden_targets["Quantized GGUF"]
        if file_paths:
            notes.append("Falling back to the built-in quantized GGUF fixture.")

    misaligned_target = golden_targets["Misaligned Safetensors"]
    if file_paths:
        notes.append("The misaligned bucket uses the built-in hostile safetensors fixture.")

    return [
        {
            "bucket_id": "dense_warm",
            "title": "Dense Safetensors / warm",
            "cache_mode": "warm",
            "target": dense_target,
        },
        {
            "bucket_id": "dense_cold",
            "title": "Dense Safetensors / cold-best-effort",
            "cache_mode": "cold-best-effort",
            "target": dense_target,
        },
        {
            "bucket_id": "quantized_warm",
            "title": "Quantized GGUF / warm",
            "cache_mode": "warm",
            "target": quantized_target,
        },
        {
            "bucket_id": "misaligned_warm",
            "title": "Misaligned Safetensors / warm",
            "cache_mode": "warm",
            "target": misaligned_target,
        },
    ], notes


def _build_adaptation_demo_manifest(
    *,
    script_path: str,
    args: argparse.Namespace,
    history_json_path: str,
    buckets: list[dict],
) -> list[dict]:
    steps: list[dict] = []
    default_knobs = _build_policy_knobs(None)
    for bucket in buckets:
        target = dict(bucket["target"])
        target_fmt = str(target.get("format") or "unknown")
        prefer_synth_decode = target_fmt != "gguf"
        for seed_case in _adaptation_seed_policy_cases():
            case_id = f"{bucket['bucket_id']}::{seed_case['label']}"
            steps.append(
                {
                    "kind": "seed",
                    "bucket_id": bucket["bucket_id"],
                    "bucket_title": bucket["title"],
                    "case_id": case_id,
                    "case_label": seed_case["label"],
                    "target": target,
                    "command": _build_experiment_command(
                        script_path=script_path,
                        file_path=str(target.get("file") or ""),
                        fmt=target_fmt,
                        cache_mode=str(bucket["cache_mode"]),
                        args=args,
                        prefer_synth_decode=prefer_synth_decode,
                        policy_knobs=seed_case["policy_knobs"],
                        requested_policy_mode=seed_case["requested_policy_mode"],
                        history_json_path=history_json_path,
                        include_baseline_flags=False,
                        demo_parent="adaptation-ladder",
                        demo_case=case_id,
                    ),
                }
            )
        auto_case_id = f"{bucket['bucket_id']}::auto"
        steps.append(
            {
                "kind": "auto",
                "bucket_id": bucket["bucket_id"],
                "bucket_title": bucket["title"],
                "case_id": auto_case_id,
                "case_label": "auto",
                "target": target,
                "command": _build_experiment_command(
                    script_path=script_path,
                    file_path=str(target.get("file") or ""),
                    fmt=target_fmt,
                    cache_mode=str(bucket["cache_mode"]),
                    args=args,
                    prefer_synth_decode=prefer_synth_decode,
                    policy_knobs=default_knobs,
                    requested_policy_mode="auto",
                    history_json_path=history_json_path,
                    include_baseline_flags=False,
                    demo_parent="adaptation-ladder",
                    demo_case=auto_case_id,
                ),
            }
        )
    return steps


def _execute_demo_step(step: dict, *, history_path: Path | None) -> dict:
    before_records = _load_history_jsonl(history_path) if history_path is not None else []
    completed = subprocess.run(str(step["command"]), shell=True, check=False)
    after_records = _load_history_jsonl(history_path) if history_path is not None else []
    new_records = after_records[len(before_records) :]
    result = dict(step)
    result["returncode"] = int(completed.returncode)
    result["ok"] = completed.returncode == 0
    result["new_records"] = new_records
    result["record"] = new_records[-1] if new_records else None
    return result


def _summarize_adaptation_demo(
    *,
    history_source: str,
    buckets: list[dict],
    step_results: list[dict],
) -> dict:
    targets = [
        {
            "bucket_id": bucket["bucket_id"],
            "title": bucket["title"],
            "cache_mode": bucket["cache_mode"],
            "file": str(bucket["target"].get("file") or ""),
            "format": str(bucket["target"].get("format") or "unknown"),
            "model_class": str(bucket["target"].get("model_class") or "unknown"),
            "label": str(bucket["target"].get("label") or Path(str(bucket["target"].get("file") or "")).name),
        }
        for bucket in buckets
    ]
    step_summaries = [
        {
            "bucket_id": result["bucket_id"],
            "case_id": result["case_id"],
            "case_label": result["case_label"],
            "kind": result["kind"],
            "ok": bool(result["ok"]),
            "returncode": int(result["returncode"]),
            "command": str(result["command"]),
        }
        for result in step_results
    ]
    summary = _demo_summary_base(
        demo_preset="adaptation-ladder",
        history_source=history_source,
        targets=targets,
        steps=step_summaries,
    )
    bucket_results: list[dict] = []
    matched = 0
    misses = 0
    failed_steps = [result for result in step_results if not result["ok"]]
    for bucket in buckets:
        bucket_step_results = [
            result for result in step_results if result["bucket_id"] == bucket["bucket_id"]
        ]
        seeded_records = [
            result["record"]
            for result in bucket_step_results
            if result["kind"] == "seed" and isinstance(result.get("record"), dict)
        ]
        auto_record = next(
            (
                result.get("record")
                for result in bucket_step_results
                if result["kind"] == "auto" and isinstance(result.get("record"), dict)
            ),
            None,
        )
        seeded_ranked = sorted(
            seeded_records,
            key=lambda record: (
                -_demo_policy_composite_score(record),
                str(record.get("effective_policy_signature") or ""),
            ),
        )
        best_seeded = seeded_ranked[0] if seeded_ranked else None
        runner_up = seeded_ranked[1] if len(seeded_ranked) > 1 else None
        if auto_record is None or best_seeded is None:
            verdict_info = {
                "verdict": "miss",
                "auto_score": None,
                "best_seeded_score": None,
                "score_gap": None,
                "tolerance": None,
                "same_family": False,
            }
            verdict = "miss"
        else:
            verdict_info = _classify_demo_choice_quality(
                auto_record=auto_record,
                best_seeded_record=best_seeded,
            )
            verdict = verdict_info["verdict"]
        if verdict in {"match", "near-match"}:
            matched += 1
        else:
            misses += 1
        selected_candidate = _find_selected_auto_candidate(
            auto_record.get("auto_policy_decision") if isinstance(auto_record, dict) else None
        )
        probe_stats = (
            auto_record.get("auto_policy_probe_mmap_coverage", {})
            if isinstance(auto_record, dict)
            else {}
        ) or {}
        mapped_bytes = int(probe_stats.get("mapped_bytes") or 0)
        copied_bytes = int(probe_stats.get("copied_bytes") or 0)
        total_bytes = mapped_bytes + copied_bytes
        bucket_results.append(
            {
                "bucket_id": bucket["bucket_id"],
                "title": bucket["title"],
                "verdict": verdict,
                "chosen_policy": (
                    auto_record.get("effective_policy_signature")
                    if isinstance(auto_record, dict)
                    else None
                ),
                "winner_policy": (
                    best_seeded.get("effective_policy_signature")
                    if isinstance(best_seeded, dict)
                    else None
                ),
                "runner_up_policy": (
                    runner_up.get("effective_policy_signature")
                    if isinstance(runner_up, dict)
                    else None
                ),
                "auto_score": verdict_info["auto_score"],
                "best_seeded_score": verdict_info["best_seeded_score"],
                "score_gap": verdict_info["score_gap"],
                "cache_mode": bucket["cache_mode"],
                "history_support": (
                    int(selected_candidate.get("history_support", 0))
                    if isinstance(selected_candidate, dict)
                    else 0
                ),
                "mapped_ratio_pct": (
                    round(100.0 * mapped_bytes / total_bytes, 2) if total_bytes > 0 else None
                ),
                "fallback_mix": probe_stats.get("fallback_reason_bytes"),
                "candidate_reasons": _summarize_candidate_reasons(selected_candidate),
            }
        )
    total_buckets = len(bucket_results)
    match_rate = (matched / total_buckets) if total_buckets else 0.0
    if failed_steps:
        verdict_status = "fail"
        verdict_summary = f"{len(failed_steps)} demo step(s) failed."
    elif misses == 0:
        verdict_status = "pass"
        verdict_summary = "Auto policy matched or nearly matched every seeded bucket."
    elif matched > 0:
        verdict_status = "mixed"
        verdict_summary = (
            f"Auto policy matched or nearly matched {matched}/{total_buckets} bucket(s)."
        )
    else:
        verdict_status = "fail"
        verdict_summary = "Auto policy missed every seeded bucket."
    summary["verdict"] = {
        "status": verdict_status,
        "summary": verdict_summary,
    }
    summary["scorecard"] = {
        "bucket_match_rate": round(match_rate, 3),
        "matched_or_near": matched,
        "total_buckets": total_buckets,
        "bucket_results": bucket_results,
    }
    summary["evidence"] = {
        "failed_steps": [
            {
                "case_id": result["case_id"],
                "bucket_id": result["bucket_id"],
                "returncode": result["returncode"],
            }
            for result in failed_steps
        ],
    }
    return summary


def _analyze_demo_case(
    summary_payload: dict,
    *,
    prior_history: list[dict],
    file_path: str,
    fallback_priority_basis: str,
    decode_gate_pct: float,
    history_window: int,
    trend_min_samples: int,
    trend_mad_mult: float,
    baseline_run_id: str | None = None,
    baseline_git_head: str | None = None,
) -> dict:
    payload = dict(summary_payload)
    payload["history_bucket"] = _build_history_bucket(payload)
    explicit_bucket_key = str(summary_payload.get("history_bucket_key") or "").strip()
    payload["history_bucket_key"] = (
        explicit_bucket_key or _history_bucket_key(payload["history_bucket"])
    )
    if explicit_bucket_key:
        bucket_records = [
            record
            for record in prior_history
            if str(record.get("history_bucket_key") or "") == explicit_bucket_key
        ]
        file_records = [record for record in prior_history if record.get("file") == file_path]
        window = max(history_window, 1)
        trend_context = {
            "scope": "bucket",
            "records": bucket_records[-window:],
            "bucket": payload["history_bucket"],
            "bucket_key": explicit_bucket_key,
            "bucket_samples": len(bucket_records[-window:]),
            "file_samples": len(file_records[-window:]),
        }
    else:
        trend_context = _select_trend_history(
            prior_history,
            payload,
            file_path=file_path,
            history_window=history_window,
            trend_min_samples=trend_min_samples,
        )
    trend_regressions = _collect_trend_regressions(
        payload,
        list(trend_context.get("records") or []),
        trend_min_samples=trend_min_samples,
        trend_mad_mult=trend_mad_mult,
    )
    comparison = _build_comparison_summary(
        payload,
        prior_history=prior_history,
        file_path=file_path,
        trend_context=trend_context,
        baseline_run_id=baseline_run_id,
        baseline_git_head=baseline_git_head,
    )
    route_suggestions = _build_route_suggestions(
        payload,
        decode_enabled=payload.get("decode_mode") != "none",
        decode_gate_pct=decode_gate_pct,
        fallback_priority_basis=fallback_priority_basis,
        trend_regressions=trend_regressions,
    )
    route_suggestions = _calibrate_route_suggestions(
        route_suggestions,
        prior_history,
        bucket_key=payload["history_bucket_key"],
    )
    loss_metrics = []
    if comparison is not None:
        loss_metrics = [
            entry["label"]
            for entry in (comparison.get("metrics") or {}).values()
            if entry.get("status") == "loss"
        ]
    return {
        "summary_payload": payload,
        "trend_context": {
            key: value for key, value in trend_context.items() if key != "records"
        },
        "trend_regressions": trend_regressions,
        "comparison": comparison,
        "route_suggestions": route_suggestions,
        "top_suggestion": route_suggestions[0] if route_suggestions else None,
        "loss_metrics": loss_metrics,
    }


def _build_history_replay_demo_summary(args: argparse.Namespace) -> dict:
    fixture_path = _demo_fixture_history_path()
    records = _load_history_jsonl(fixture_path)
    report = _build_history_report(records, top_n=args.report_top)
    prime_physics = _build_prime_physics_corpus_fixture_summary()
    replay_payload = {
        "file": "/demo/replay_quantized.gguf",
        "format": "gguf",
        "model_class": "quantized_gguf",
        "cache_mode": "warm",
        "decode_mode": "synth",
        "history_bucket_key": "demo_quantized_warm",
        "effective_policy_signature": "mapped:default",
        "load_improve_pct_median": 9.0,
        "rss_reduce_pct_median": 10.0,
        "load_peak_memory_reduce_pct_median": 6.0,
        "total_peak_memory_reduce_pct_median": 5.0,
        "load_call_improve_pct_median": 8.0,
        "parse_improve_pct_median": 7.0,
        "tensor_setup_improve_pct_median": 6.0,
        "first_eval_improve_pct_median": 5.0,
        "decode_regression_pct_median": 5.5,
        "decode_first_token_regression_pct_median": 7.0,
        "mmap_coverage_mapped_ratio_pct": 52.0,
        "mmap_coverage_copied_bytes": 192 * 1024 * 1024,
        "mmap_coverage_fallback_tensors": 18,
        "mmap_coverage_fallback_reasons": {"quantized_conversion": 14},
        "mmap_coverage_fallback_reason_bytes": {
            "quantized_conversion": 160 * 1024 * 1024
        },
        "mmap_coverage_fallback_reason_source_bytes": {
            "quantized_conversion": 96 * 1024 * 1024
        },
    }
    comparable_case = _analyze_demo_case(
        replay_payload,
        prior_history=records,
        file_path=str(replay_payload["file"]),
        fallback_priority_basis=args.fallback_priority_basis,
        decode_gate_pct=args.decode_regression_gate_pct,
        history_window=max(args.history_window, 8),
        trend_min_samples=min(max(args.trend_min_samples, 2), 3),
        trend_mad_mult=args.trend_mad_mult,
    )
    explicit_case = _analyze_demo_case(
        replay_payload,
        prior_history=records,
        file_path=str(replay_payload["file"]),
        fallback_priority_basis=args.fallback_priority_basis,
        decode_gate_pct=args.decode_regression_gate_pct,
        history_window=max(args.history_window, 8),
        trend_min_samples=min(max(args.trend_min_samples, 2), 3),
        trend_mad_mult=args.trend_mad_mult,
        baseline_run_id="fixture-quantized-warm-copy-1",
    )
    targets = [
        {
            "bucket_key": bucket["bucket_key"],
            "samples": bucket["samples"],
            "latest_timestamp": bucket["latest_timestamp"],
        }
        for bucket in report["buckets"]
    ]
    steps = [
        {"name": "load_fixture_history", "status": "ok", "fixture": str(fixture_path)},
        {"name": "build_history_report", "status": "ok", "bucket_count": len(report["buckets"])},
        {
            "name": "check_comparison_basis",
            "status": "ok",
            "comparable_basis": (
                comparable_case["comparison"]["basis"]
                if comparable_case["comparison"] is not None
                else None
            ),
            "explicit_basis": (
                explicit_case["comparison"]["basis"]
                if explicit_case["comparison"] is not None
                else None
            ),
        },
    ]
    summary = _demo_summary_base(
        demo_preset="history-replay",
        history_source="fixture",
        targets=targets,
        steps=steps,
    )
    top_bucket = report["buckets"][0] if report["buckets"] else {}
    expected_comparable = (
        comparable_case["comparison"]["basis"] == "bucket_recent_median"
        if comparable_case["comparison"] is not None
        else False
    )
    expected_explicit = (
        explicit_case["comparison"]["basis"] == "baseline_run"
        if explicit_case["comparison"] is not None
        else False
    )
    corpus_ok = (
        prime_physics["pass_count"] == prime_physics["total_cases"]
        if prime_physics["total_cases"] > 0
        else True
    )
    summary["verdict"] = {
        "status": (
            "pass"
            if report["buckets"] and expected_comparable and expected_explicit and corpus_ok
            else "fail"
        ),
        "summary": (
            "Fixture replay exercised history reporting, calibrated suggestions, "
            "both comparison selection paths, and scoped external-corpus memory."
        ),
    }
    summary["scorecard"] = {
        "bucket_count": len(report["buckets"]),
        "route_hit_rate": [
            {
                "bucket_key": bucket["bucket_key"],
                "routes": bucket["route_hit_rate"],
            }
            for bucket in report["buckets"][: args.report_top]
        ],
        "comparison_basis_checks": {
            "comparable_history": (
                comparable_case["comparison"]["basis"]
                if comparable_case["comparison"] is not None
                else None
            ),
            "explicit_baseline": (
                explicit_case["comparison"]["basis"]
                if explicit_case["comparison"] is not None
                else None
            ),
        },
        "external_corpus_match_rate": prime_physics["match_rate"],
        "external_corpus_cases": prime_physics["cases"],
    }
    summary["evidence"] = {
        "fixture_path": str(fixture_path),
        "external_corpus_fixture_path": prime_physics["fixture_path"],
        "top_bucket": {
            "bucket_key": top_bucket.get("bucket_key"),
            "policy_winners": top_bucket.get("policy_winners"),
            "policy_losers": top_bucket.get("policy_losers"),
            "top_phase_regressions": top_bucket.get("top_phase_regressions"),
            "top_fallback_reasons_by_bytes": top_bucket.get("top_fallback_reasons_by_bytes"),
        },
        "calibrated_suggestion": comparable_case["top_suggestion"],
    }
    return summary


def _build_regression_forensics_cases() -> list[dict]:
    return [
        {
            "case_id": "parse_tensor_setup_regression",
            "title": "Parse + Tensor Setup Regression",
            "payload": {
                "file": "/demo/replay_dense.safetensors",
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "history_bucket_key": "demo_dense_warm",
                "effective_policy_signature": "mapped:default",
                "load_improve_pct_median": 17.0,
                "rss_reduce_pct_median": 25.0,
                "load_peak_memory_reduce_pct_median": 13.0,
                "total_peak_memory_reduce_pct_median": 10.0,
                "load_call_improve_pct_median": 14.0,
                "parse_improve_pct_median": 4.0,
                "tensor_setup_improve_pct_median": 5.0,
                "first_eval_improve_pct_median": 8.0,
                "decode_regression_pct_median": 1.4,
                "decode_first_token_regression_pct_median": 2.0,
                "mmap_coverage_mapped_ratio_pct": 98.0,
                "mmap_coverage_copied_bytes": 8 * 1024 * 1024,
                "mmap_coverage_fallback_tensors": 1,
                "mmap_coverage_fallback_reasons": {},
                "mmap_coverage_fallback_reason_bytes": {},
                "mmap_coverage_fallback_reason_source_bytes": {},
            },
            "expected_labels": {"parse improve", "tensor-setup improve"},
        },
        {
            "case_id": "first_token_regression",
            "title": "First-Token Regression",
            "payload": {
                "file": "/demo/replay_quantized.gguf",
                "format": "gguf",
                "model_class": "quantized_gguf",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "history_bucket_key": "demo_quantized_warm",
                "effective_policy_signature": "mapped:default",
                "load_improve_pct_median": 10.0,
                "rss_reduce_pct_median": 14.0,
                "load_peak_memory_reduce_pct_median": 8.0,
                "total_peak_memory_reduce_pct_median": 6.0,
                "load_call_improve_pct_median": 9.0,
                "parse_improve_pct_median": 8.0,
                "tensor_setup_improve_pct_median": 8.0,
                "first_eval_improve_pct_median": 7.0,
                "decode_regression_pct_median": 4.5,
                "decode_first_token_regression_pct_median": 11.5,
                "mmap_coverage_mapped_ratio_pct": 58.0,
                "mmap_coverage_copied_bytes": 220 * 1024 * 1024,
                "mmap_coverage_fallback_tensors": 20,
                "mmap_coverage_fallback_reasons": {"quantized_conversion": 16},
                "mmap_coverage_fallback_reason_bytes": {
                    "quantized_conversion": 180 * 1024 * 1024
                },
                "mmap_coverage_fallback_reason_source_bytes": {
                    "quantized_conversion": 112 * 1024 * 1024
                },
            },
            "expected_labels": {"first-token regression"},
        },
        {
            "case_id": "mixed_tradeoff",
            "title": "Mixed Load/RSS Tradeoff",
            "payload": {
                "file": "/demo/replay_dense.safetensors",
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "cache_mode": "cold-best-effort",
                "decode_mode": "synth",
                "history_bucket_key": "demo_dense_cold",
                "effective_policy_signature": "hybrid:prefetch=willneed",
                "load_improve_pct_median": 30.0,
                "rss_reduce_pct_median": 10.0,
                "load_peak_memory_reduce_pct_median": 7.0,
                "total_peak_memory_reduce_pct_median": 6.0,
                "load_call_improve_pct_median": 25.0,
                "parse_improve_pct_median": 12.0,
                "tensor_setup_improve_pct_median": 9.0,
                "first_eval_improve_pct_median": 8.0,
                "decode_regression_pct_median": 2.4,
                "decode_first_token_regression_pct_median": 3.5,
                "mmap_coverage_mapped_ratio_pct": 96.0,
                "mmap_coverage_copied_bytes": 24 * 1024 * 1024,
                "mmap_coverage_fallback_tensors": 2,
                "mmap_coverage_fallback_reasons": {},
                "mmap_coverage_fallback_reason_bytes": {},
                "mmap_coverage_fallback_reason_source_bytes": {},
            },
            "expected_outcome": "mixed",
        },
    ]


def _build_regression_forensics_demo_summary(args: argparse.Namespace) -> dict:
    fixture_path = _demo_fixture_history_path()
    records = _load_history_jsonl(fixture_path)
    prime_physics = _build_prime_physics_corpus_fixture_summary()
    cases = []
    for case in _build_regression_forensics_cases():
        analysis = _analyze_demo_case(
            case["payload"],
            prior_history=records,
            file_path=str(case["payload"]["file"]),
            fallback_priority_basis=args.fallback_priority_basis,
            decode_gate_pct=args.decode_regression_gate_pct,
            history_window=max(args.history_window, 8),
            trend_min_samples=min(max(args.trend_min_samples, 2), 3),
            trend_mad_mult=args.trend_mad_mult,
        )
        comparison = analysis["comparison"] or {}
        cases.append(
            {
                "case_id": case["case_id"],
                "title": case["title"],
                "comparison_basis": comparison.get("basis"),
                "outcome": comparison.get("auto_outcome"),
                "loss_metrics": analysis["loss_metrics"],
                "trend_regressions": analysis["trend_regressions"],
                "top_suggestion": analysis["top_suggestion"],
                "expected_labels": sorted(case.get("expected_labels", set())),
                "expected_outcome": case.get("expected_outcome"),
            }
        )
    basis_case = _analyze_demo_case(
        _build_regression_forensics_cases()[0]["payload"],
        prior_history=records,
        file_path="/demo/replay_dense.safetensors",
        fallback_priority_basis=args.fallback_priority_basis,
        decode_gate_pct=args.decode_regression_gate_pct,
        history_window=max(args.history_window, 8),
        trend_min_samples=min(max(args.trend_min_samples, 2), 3),
        trend_mad_mult=args.trend_mad_mult,
        baseline_run_id="fixture-dense-warm-mapped-1",
    )
    summary = _demo_summary_base(
        demo_preset="regression-forensics",
        history_source="fixture",
        targets=[{"case_id": case["case_id"], "title": case["title"]} for case in cases],
        steps=[
            {"name": "load_fixture_history", "status": "ok", "fixture": str(fixture_path)},
            {"name": "run_forensics_cases", "status": "ok", "case_count": len(cases)},
        ],
    )
    labels_ok = all(
        set(case["expected_labels"]).issubset(set(case["loss_metrics"]))
        if case["expected_labels"]
        else True
        for case in cases
    )
    outcomes_ok = all(
        case["expected_outcome"] == case["outcome"]
        if case["expected_outcome"] is not None
        else True
        for case in cases
    )
    suggestions_ok = all(case["top_suggestion"] is not None for case in cases)
    corpus_ok = (
        prime_physics["pass_count"] == prime_physics["total_cases"]
        if prime_physics["total_cases"] > 0
        else True
    )
    summary["verdict"] = {
        "status": (
            "pass" if labels_ok and outcomes_ok and suggestions_ok and corpus_ok else "fail"
        ),
        "summary": (
            "Replay forensics surfaced the intended regressions, follow-up routes, "
            "and scoped external-corpus claim discipline."
        ),
    }
    summary["scorecard"] = {
        "cases": cases,
        "comparison_basis_checks": {
            "comparable_history": cases[0]["comparison_basis"] if cases else None,
            "explicit_baseline": (
                basis_case["comparison"]["basis"]
                if basis_case["comparison"] is not None
                else None
            ),
        },
        "external_corpus_cases": prime_physics["cases"],
    }
    summary["evidence"] = {
        "fixture_path": str(fixture_path),
        "external_corpus_fixture_path": prime_physics["fixture_path"],
        "top_suggestions": [
            {
                "case_id": case["case_id"],
                "route_id": (
                    case["top_suggestion"].get("route_id")
                    if isinstance(case["top_suggestion"], dict)
                    else None
                ),
                "route": (
                    case["top_suggestion"].get("route")
                    if isinstance(case["top_suggestion"], dict)
                    else None
                ),
            }
            for case in cases
        ],
    }
    return summary


def _build_golden_guardrail_manifest(
    *,
    script_path: str,
    args: argparse.Namespace,
    history_json_path: str,
    temp_root: Path,
) -> tuple[list[dict], list[dict]]:
    matrix_cases, matrix_cache_modes = _compact_policy_matrix_cases(args)
    targets = []
    steps: list[dict] = []
    for title, target in _build_golden_matrix_targets(temp_root):
        target_fmt = str(target.get("format") or "unknown")
        prefer_synth_decode = target_fmt != "gguf"
        targets.append(
            {
                "title": title,
                "file": str(target.get("file") or ""),
                "format": target_fmt,
                "model_class": str(target.get("model_class") or "unknown"),
                "label": str(target.get("label") or Path(str(target.get("file") or "")).name),
            }
        )
        for cache_mode in matrix_cache_modes:
            for case in matrix_cases:
                case_id = f"{_slugify_identifier(title)}::{cache_mode}::{case['label']}"
                steps.append(
                    {
                        "kind": "guardrail",
                        "target_title": title,
                        "cache_mode": cache_mode,
                        "case_id": case_id,
                        "case_label": case["label"],
                        "command": _build_experiment_command(
                            script_path=script_path,
                            file_path=str(target.get("file") or ""),
                            fmt=target_fmt,
                            cache_mode=cache_mode,
                            args=args,
                            prefer_synth_decode=prefer_synth_decode,
                            policy_knobs=case["policy_knobs"],
                            requested_policy_mode="manual",
                            history_json_path=history_json_path,
                            include_baseline_flags=False,
                            demo_parent="golden-guardrail",
                            demo_case=case_id,
                        ),
                    }
                )
    return targets, steps


def _summarize_golden_guardrail_demo(
    *,
    history_source: str,
    targets: list[dict],
    step_results: list[dict],
) -> dict:
    summary = _demo_summary_base(
        demo_preset="golden-guardrail",
        history_source=history_source,
        targets=targets,
        steps=[
            {
                "target_title": result["target_title"],
                "cache_mode": result["cache_mode"],
                "case_id": result["case_id"],
                "case_label": result["case_label"],
                "ok": bool(result["ok"]),
                "returncode": int(result["returncode"]),
                "command": str(result["command"]),
            }
            for result in step_results
        ],
    )
    parity_cases = []
    fallback_reason_summary: dict[str, dict] = {}
    failed_cases: list[str] = []
    for target in targets:
        target_steps = [
            result for result in step_results if result["target_title"] == target["title"]
        ]
        all_ok = all(result["ok"] for result in target_steps)
        aggregated_fallback_bytes = Counter()
        for result in target_steps:
            record = result.get("record") or {}
            for reason, value in (record.get("mmap_coverage_fallback_reason_bytes") or {}).items():
                aggregated_fallback_bytes[str(reason)] += int(value)
            if not result["ok"]:
                failed_cases.append(result["case_id"])
        top_reason, top_bytes = _top_fallback_bytes_reason(dict(aggregated_fallback_bytes))
        parity_cases.append(
            {
                "title": target["title"],
                "parity_ok": all_ok,
                "case_count": len(target_steps),
                "failed_case_count": sum(1 for result in target_steps if not result["ok"]),
            }
        )
        fallback_reason_summary[target["title"]] = {
            "top_reason": top_reason,
            "bytes": top_bytes,
        }
    parity_ok = all(case["parity_ok"] for case in parity_cases)
    summary["verdict"] = {
        "status": "pass" if parity_ok else "fail",
        "summary": (
            "Adaptation claims are only trustworthy while the dense, quantized, "
            "and alignment-hostile golden checks stay green."
        ),
    }
    summary["scorecard"] = {
        "parity_ok": parity_ok,
        "cases": parity_cases,
    }
    summary["evidence"] = {
        "failed_cases": failed_cases,
        "fallback_reason_summary": fallback_reason_summary,
    }
    return summary


def _print_demo_summary_text(summary: dict) -> None:
    print(
        f"Demo: {summary['demo_preset']} "
        f"[{summary['verdict'].get('status', 'unknown')}]"
    )
    print(f"  {summary['verdict'].get('summary', '')}")
    print(f"  history source: {summary.get('history_source')}")
    if summary["demo_preset"] == "adaptation-ladder":
        print("  adaptation scorecard:")
        for bucket in summary["scorecard"].get("bucket_results", []):
            print(
                f"    {bucket['title']}: {bucket['verdict']} "
                f"chosen={bucket['chosen_policy']} winner={bucket['winner_policy']} "
                f"runner_up={bucket['runner_up_policy']}"
            )
            if bucket.get("candidate_reasons"):
                print(f"      why: {'; '.join(bucket['candidate_reasons'])}")
        print(
            "  bucket match rate: "
            f"{summary['scorecard'].get('matched_or_near', 0)}/"
            f"{summary['scorecard'].get('total_buckets', 0)}"
        )
    elif summary["demo_preset"] == "history-replay":
        print(
            "  comparison checks: "
            f"comparable={summary['scorecard']['comparison_basis_checks']['comparable_history']} "
            f"explicit={summary['scorecard']['comparison_basis_checks']['explicit_baseline']}"
        )
        calibrated = summary["evidence"].get("calibrated_suggestion") or {}
        if calibrated:
            print(
                "  calibrated suggestion: "
                f"{calibrated.get('route')} ({calibrated.get('route_id')})"
            )
        external_cases = summary["scorecard"].get("external_corpus_cases") or []
        if external_cases:
            print("  external corpus replay:")
            for case in external_cases:
                print(
                    f"    {case['title']}: {case['verdict']} "
                    f"chosen={case['chosen_route']} expected={case['expected_route']}"
                )
    elif summary["demo_preset"] == "regression-forensics":
        for case in summary["scorecard"].get("cases", []):
            print(
                f"  {case['title']}: outcome={case['outcome']} "
                f"noticed={', '.join(case['loss_metrics']) or 'none'}"
            )
            suggestion = case.get("top_suggestion") or {}
            if suggestion:
                print(
                    f"    next route: {suggestion.get('route')} "
                    f"({suggestion.get('route_id')})"
                )
        external_cases = summary["scorecard"].get("external_corpus_cases") or []
        if external_cases:
            print("  external corpus cases:")
            for case in external_cases:
                print(f"    {case['title']}: {case['loop_noticed']}")
                print(
                    f"      next route: {case['chosen_route']} "
                    f"({case['chosen_route_id']})"
                )
    elif summary["demo_preset"] == "persistence-memory":
        print(
            "  suppression scorecard: "
            f"session={summary['scorecard'].get('session_suppressed', 0)}/"
            f"{summary['scorecard'].get('total_cases', 0)} "
            f"unscoped_resurfaced={summary['scorecard'].get('persistent_unscoped_resurfaced', 0)}/"
            f"{summary['scorecard'].get('total_cases', 0)} "
            f"scoped={summary['scorecard'].get('persistent_scoped_suppressed', 0)}/"
            f"{summary['scorecard'].get('total_cases', 0)}"
        )
        for case in summary["scorecard"].get("cases", []):
            print(f"  {case['title']}:")
            for stage in case.get("stages", []):
                print(
                    f"    {stage['title']}: {stage['suppression_state']} "
                    f"chosen={stage['chosen_route'] or 'none'} "
                    f"refuted={stage['refuted_route']}"
                )
    elif summary["demo_preset"] == "golden-guardrail":
        for case in summary["scorecard"].get("cases", []):
            print(
                f"  {case['title']}: parity_ok={case['parity_ok']} "
                f"failed_cases={case['failed_case_count']}"
            )
        fallback_reason_summary = summary["evidence"].get("fallback_reason_summary") or {}
        if fallback_reason_summary:
            print("  fallback reason sanity:")
            for title, entry in fallback_reason_summary.items():
                print(
                    f"    {title}: {entry.get('top_reason') or 'none'} "
                    f"({_format_mib(entry.get('bytes'))})"
                )


def _emit_demo_summary(summary: dict, *, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_demo_summary_text(summary)


def _validate_demo_preset_args(args: argparse.Namespace) -> None:
    if args.demo_preset is None:
        return
    if args.worker:
        raise ValueError("--demo-preset is not supported in worker mode")
    if args.report_history:
        raise ValueError("--demo-preset cannot be combined with --report-history")
    if args.policy_matrix is not None:
        raise ValueError("--demo-preset cannot be combined with --policy-matrix")
    if args.golden_matrix is not None:
        raise ValueError("--demo-preset cannot be combined with --golden-matrix")
    if args.demo_parent is not None or args.demo_case is not None:
        raise ValueError("--demo-preset cannot be combined with internal demo child flags")


def _run_demo_preset(
    *,
    args: argparse.Namespace,
    provided_flags: set[str],
    script_path: str,
) -> int:
    if args.demo_preset is None:
        return -1
    if (
        args.demo_preset in {"adaptation-ladder", "golden-guardrail"}
        and not args.decode_synth
        and args.decode_cmd is None
        and args.decode_llama_bench_binary is None
    ):
        args.decode_synth = True

    if args.demo_preset == "history-replay":
        summary = _build_history_replay_demo_summary(args)
        _emit_demo_summary(summary, output_format=args.demo_format)
        return 0 if summary["verdict"].get("status") == "pass" else 1

    if args.demo_preset == "regression-forensics":
        summary = _build_regression_forensics_demo_summary(args)
        _emit_demo_summary(summary, output_format=args.demo_format)
        return 0 if summary["verdict"].get("status") == "pass" else 1

    if args.demo_preset == "persistence-memory":
        summary = _build_persistence_memory_demo_summary(args)
        _emit_demo_summary(summary, output_format=args.demo_format)
        return 0 if summary["verdict"].get("status") == "pass" else 1

    file_paths, discovered_formats, discovered_details, discovered_candidates = _resolve_input_files(
        args
    )
    if discovered_candidates:
        print(
            "Discovery selected "
            f"{len(discovered_candidates)} model file(s) for the demo preset."
        )
        for candidate in discovered_candidates:
            suffix = f" [{candidate['model_id']}]" if candidate.get("model_id") else ""
            print(f"  discovered: {candidate['path']}{suffix}")

    history_was_explicit = "--history-json" in provided_flags
    history_source = "explicit" if history_was_explicit else "temp"

    with tempfile.TemporaryDirectory(prefix="load_mmap_demo_") as temp_dir:
        temp_root = Path(temp_dir)
        history_path = (
            Path(args.history_json).expanduser()
            if history_was_explicit
            else temp_root / "demo_history.jsonl"
        )
        if args.demo_preset == "adaptation-ladder":
            buckets, notes = _build_adaptation_demo_targets(
                file_paths=file_paths,
                requested_format=args.format,
                discovered_formats=discovered_formats,
                discovered_details=discovered_details,
                temp_root=temp_root,
            )
            if notes:
                print("Demo notes:")
                for note in notes:
                    print(f"  - {note}")
            steps = _build_adaptation_demo_manifest(
                script_path=script_path,
                args=args,
                history_json_path=str(history_path),
                buckets=buckets,
            )
            print(
                "Adaptation ladder enabled: "
                f"buckets={len(buckets)} steps={len(steps)} "
                f"history_source={history_source}"
            )
            step_results: list[dict] = []
            for index, step in enumerate(steps, start=1):
                print(f"\nDemo step [{index}/{len(steps)}]: {step['bucket_title']} / {step['case_label']}")
                step_results.append(_execute_demo_step(step, history_path=history_path))
            summary = _summarize_adaptation_demo(
                history_source=history_source,
                buckets=buckets,
                step_results=step_results,
            )
            _emit_demo_summary(summary, output_format=args.demo_format)
            return 0 if summary["verdict"].get("status") in {"pass", "mixed"} else 1

        targets, steps = _build_golden_guardrail_manifest(
            script_path=script_path,
            args=args,
            history_json_path=str(history_path),
            temp_root=temp_root,
        )
        print(
            "Golden guardrail enabled: "
            f"targets={len(targets)} steps={len(steps)} history_source={history_source}"
        )
        step_results = []
        for index, step in enumerate(steps, start=1):
            print(
                f"\nDemo step [{index}/{len(steps)}]: "
                f"{step['target_title']} / {step['cache_mode']} / {step['case_label']}"
            )
            step_results.append(_execute_demo_step(step, history_path=history_path))
        summary = _summarize_golden_guardrail_demo(
            history_source=history_source,
            targets=targets,
            step_results=step_results,
        )
        _emit_demo_summary(summary, output_format=args.demo_format)
        return 0 if summary["verdict"].get("status") == "pass" else 1


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
    mmap_small_tensor_copy_max_bytes: int | None,
    mmap_hotset_promotion_top_k: int | None,
    mmap_hotset_promotion_min_bytes: int | None,
    mmap_prefetch_strategy: str,
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
    if mmap_small_tensor_copy_max_bytes is not None:
        cmd += [
            "--mmap-small-tensor-copy-max-bytes",
            str(mmap_small_tensor_copy_max_bytes),
        ]
    if mmap_hotset_promotion_top_k is not None:
        cmd += [
            "--mmap-hotset-promotion-top-k",
            str(mmap_hotset_promotion_top_k),
        ]
    if mmap_hotset_promotion_min_bytes is not None:
        cmd += [
            "--mmap-hotset-promotion-min-bytes",
            str(mmap_hotset_promotion_min_bytes),
        ]
    if mmap_prefetch_strategy != "sequential":
        cmd += ["--mmap-prefetch-strategy", mmap_prefetch_strategy]
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
    if "mmap_stats" not in trial:
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
    mmap_small_tensor_copy_max_bytes: int | None,
    mmap_hotset_promotion_top_k: int | None,
    mmap_hotset_promotion_min_bytes: int | None,
    mmap_prefetch_strategy: str,
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
            mmap_small_tensor_copy_max_bytes=mmap_small_tensor_copy_max_bytes,
            mmap_hotset_promotion_top_k=mmap_hotset_promotion_top_k,
            mmap_hotset_promotion_min_bytes=mmap_hotset_promotion_min_bytes,
            mmap_prefetch_strategy=mmap_prefetch_strategy,
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
    policy_memory_map: bool,
    gguf_nvfp4_compat: bool,
    mmap_small_tensor_copy_max_bytes: int | None,
    mmap_hotset_promotion_top_k: int | None,
    mmap_hotset_promotion_min_bytes: int | None,
    mmap_prefetch_strategy: str,
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
        mode_order = (
            (False, policy_memory_map)
            if (run_idx % 2 == 0)
            else (policy_memory_map, False)
        )
        results: dict[bool, dict] = {}

        for memory_map in mode_order:
            mode_trials = _run_trials(
                script_path=script_path,
                path=path,
                fmt=fmt,
                memory_map=memory_map,
                gguf_nvfp4_compat=gguf_nvfp4_compat,
                mmap_small_tensor_copy_max_bytes=mmap_small_tensor_copy_max_bytes,
                mmap_hotset_promotion_top_k=mmap_hotset_promotion_top_k,
                mmap_hotset_promotion_min_bytes=mmap_hotset_promotion_min_bytes,
                mmap_prefetch_strategy=mmap_prefetch_strategy,
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
        "--mmap-small-tensor-copy-max-bytes",
        type=int,
        default=None,
        help=(
            "When memory_map is enabled, copy mapped-eligible tensors at or "
            "below this size instead of creating mapped views."
        ),
    )
    parser.add_argument(
        "--mmap-hotset-promotion-top-k",
        type=int,
        default=0,
        help=(
            "When memory_map is enabled, copy up to this many of the largest "
            "mapped-eligible tensors into owned buffers instead of returning "
            "mapped views."
        ),
    )
    parser.add_argument(
        "--mmap-hotset-promotion-min-bytes",
        type=int,
        default=None,
        help=(
            "Minimum tensor size for --mmap-hotset-promotion-top-k "
            "eligibility."
        ),
    )
    parser.add_argument(
        "--mmap-prefetch-strategy",
        choices=list(_ALLOWED_MMAP_PREFETCH_STRATEGIES),
        default="sequential",
        help=(
            "Prefetch hint for mapped file loads "
            "(default: sequential)."
        ),
    )
    parser.add_argument(
        "--policy-mode",
        choices=["manual", "auto", "copy", "mapped", "hybrid"],
        default="manual",
        help=(
            "Policy applied on the non-copy side of the benchmark. "
            "manual=use the requested mmap knobs, auto=choose from history and "
            "a mapped preflight probe, copy=force copy-only, "
            "mapped=force mapped default, hybrid=force the requested hybrid knobs."
        ),
    )
    parser.add_argument(
        "--sweep-mmap-small-tensor-copy-max-bytes",
        default=None,
        help=(
            "Comma-separated threshold sweep for "
            "--mmap-small-tensor-copy-max-bytes. Use 'none' for the default "
            "policy."
        ),
    )
    parser.add_argument(
        "--sweep-mmap-hotset-promotion-top-k",
        default=None,
        help=(
            "Comma-separated sweep for --mmap-hotset-promotion-top-k. "
            "Use 'none' or '0' to disable promotion."
        ),
    )
    parser.add_argument(
        "--sweep-mmap-hotset-promotion-min-bytes",
        default=None,
        help=(
            "Comma-separated sweep for --mmap-hotset-promotion-min-bytes. "
            "Use 'none' to remove the minimum-size gate."
        ),
    )
    parser.add_argument(
        "--sweep-mmap-prefetch-strategy",
        default=None,
        help=(
            "Comma-separated sweep for --mmap-prefetch-strategy "
            "(for example: sequential,willneed,none)."
        ),
    )
    parser.add_argument(
        "--policy-matrix",
        choices=["compact"],
        default=None,
        help=(
            "Run a preset dense-vs-quantized warm/cold policy matrix instead "
            "of a single benchmark pass."
        ),
    )
    parser.add_argument(
        "--policy-matrix-thresholds",
        default=None,
        help=(
            "Override the small-tensor threshold list used by "
            "--policy-matrix compact. Default: none,65536."
        ),
    )
    parser.add_argument(
        "--policy-matrix-prefetch-strategies",
        default=None,
        help=(
            "Override the prefetch strategy list used by "
            "--policy-matrix compact. Default: sequential,willneed."
        ),
    )
    parser.add_argument(
        "--policy-matrix-parent",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--policy-matrix-case",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--golden-matrix",
        choices=["compact"],
        default=None,
        help=(
            "Run a built-in correctness matrix over stable dense, quantized, and "
            "alignment-hostile fixtures."
        ),
    )
    parser.add_argument(
        "--golden-matrix-parent",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--golden-matrix-case",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--demo-preset",
        choices=list(_DEMO_PRESET_CHOICES),
        default=None,
        help=(
            "Run a benchmark-owned demo preset instead of a normal benchmark. "
            "history-replay=replay calibrated learning from canned history, "
            "adaptation-ladder=seed live buckets and test auto policy, "
            "golden-guardrail=wrap the correctness matrix with a demo summary, "
            "regression-forensics=replay curated regression investigations, "
            "persistence-memory=show fresh/session/scoped-memory suppression."
        ),
    )
    parser.add_argument(
        "--demo-format",
        choices=["text", "json"],
        default="text",
        help="Output format used by --demo-preset (default: text).",
    )
    parser.add_argument(
        "--demo-parent",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--demo-case",
        default=None,
        help=argparse.SUPPRESS,
    )
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
        "--report-history",
        action="store_true",
        help="Summarize benchmark history instead of running a benchmark.",
    )
    parser.add_argument(
        "--report-format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format used by --report-history (default: text).",
    )
    parser.add_argument(
        "--report-top",
        type=int,
        default=3,
        help="Top-N entries per history report section (default: 3).",
    )
    parser.add_argument(
        "--attempted-route",
        default=None,
        help=(
            "Route ID or display label being tested in this run, for example "
            "'hotset_promotion' or 'Hotset Promotion'."
        ),
    )
    parser.add_argument(
        "--route-outcome",
        choices=list(_OUTCOME_CHOICES),
        default="auto",
        help=(
            "Outcome classification for --attempted-route. "
            "Default: auto from comparison deltas."
        ),
    )
    parser.add_argument(
        "--baseline-run-id",
        default=None,
        help="Explicit history run_id to compare against.",
    )
    parser.add_argument(
        "--baseline-git-head",
        default=None,
        help="Explicit git SHA or short SHA prefix to compare against.",
    )
    parser.add_argument(
        "--route-suggestions",
        dest="route_suggestions",
        action="store_true",
        help=(
            "Print heuristic next-route suggestions from the current file "
            "summary (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-route-suggestions",
        dest="route_suggestions",
        action="store_false",
        help="Disable heuristic next-route suggestions.",
    )
    parser.add_argument(
        "--fallback-priority-basis",
        choices=["bytes", "count"],
        default="bytes",
        help=(
            "Fallback route ordering basis for suggestions "
            "(default: bytes = materialized destination bytes first, "
            "count breaks ties)."
        ),
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
    parser.set_defaults(
        reject_noisy_attempts=True,
        coverage_probe=True,
        route_suggestions=True,
    )

    raw_argv = sys.argv[1:]
    provided_flags = _provided_cli_flags(raw_argv)
    args = parser.parse_args()
    _apply_profile_defaults(args, provided_flags)
    script_path = os.path.abspath(__file__)
    _validate_demo_preset_args(args)
    args.route_outcome = _normalize_outcome_label(args.route_outcome)
    attempted_route_ref = _normalize_route_reference(args.attempted_route)
    if args.route_outcome != "auto" and attempted_route_ref is None:
        raise ValueError("--route-outcome requires --attempted-route")
    if args.baseline_run_id is not None and args.baseline_git_head is not None:
        raise ValueError(
            "--baseline-run-id and --baseline-git-head are mutually exclusive"
        )
    if args.report_top < 1:
        raise ValueError("--report-top must be >= 1")

    if args.mmap_small_tensor_copy_max_bytes is not None:
        if args.mmap_small_tensor_copy_max_bytes < 0:
            raise ValueError("--mmap-small-tensor-copy-max-bytes must be >= 0")
    args.mmap_hotset_promotion_top_k = _normalize_hotset_promotion_top_k(
        args.mmap_hotset_promotion_top_k
    )
    args.mmap_hotset_promotion_min_bytes = _normalize_hotset_promotion_min_bytes(
        args.mmap_hotset_promotion_min_bytes
    )
    args.mmap_prefetch_strategy = _normalize_prefetch_strategy(
        args.mmap_prefetch_strategy
    )
    threshold_sweep = _parse_optional_byte_threshold_list(
        args.sweep_mmap_small_tensor_copy_max_bytes
    )
    hotset_top_k_sweep = _parse_optional_nonnegative_int_list(
        args.sweep_mmap_hotset_promotion_top_k
    )
    hotset_min_bytes_sweep = _parse_hotset_promotion_min_bytes_list(
        args.sweep_mmap_hotset_promotion_min_bytes
    )
    prefetch_sweep = _parse_prefetch_strategy_list(args.sweep_mmap_prefetch_strategy)
    if threshold_sweep and args.mmap_small_tensor_copy_max_bytes is not None:
        raise ValueError(
            "--mmap-small-tensor-copy-max-bytes cannot be combined with "
            "--sweep-mmap-small-tensor-copy-max-bytes"
        )
    if hotset_top_k_sweep and args.mmap_hotset_promotion_top_k is not None:
        raise ValueError(
            "--mmap-hotset-promotion-top-k cannot be combined with "
            "--sweep-mmap-hotset-promotion-top-k"
        )
    if hotset_min_bytes_sweep and args.mmap_hotset_promotion_min_bytes is not None:
        raise ValueError(
            "--mmap-hotset-promotion-min-bytes cannot be combined with "
            "--sweep-mmap-hotset-promotion-min-bytes"
        )
    if prefetch_sweep and args.mmap_prefetch_strategy != "sequential":
        raise ValueError(
            "--mmap-prefetch-strategy cannot be combined with "
            "--sweep-mmap-prefetch-strategy"
        )
    if (
        args.mmap_hotset_promotion_min_bytes is not None
        and args.mmap_hotset_promotion_top_k is None
        and not hotset_top_k_sweep
    ):
        raise ValueError(
            "--mmap-hotset-promotion-min-bytes requires "
            "--mmap-hotset-promotion-top-k or its sweep variant"
        )
    if (
        hotset_min_bytes_sweep
        and args.mmap_hotset_promotion_top_k is None
        and not hotset_top_k_sweep
    ):
        raise ValueError(
            "--sweep-mmap-hotset-promotion-min-bytes requires "
            "--mmap-hotset-promotion-top-k or --sweep-mmap-hotset-promotion-top-k"
        )
    if args.policy_matrix is not None:
        if args.worker:
            raise ValueError("--policy-matrix is not supported in worker mode")
        if threshold_sweep or hotset_top_k_sweep or hotset_min_bytes_sweep or prefetch_sweep:
            raise ValueError(
                "--policy-matrix cannot be combined with explicit policy sweep flags"
            )
        if (
            args.mmap_small_tensor_copy_max_bytes is not None
            or args.mmap_prefetch_strategy != "sequential"
        ):
            raise ValueError(
                "--policy-matrix uses preset policy cases; override them with "
                "--policy-matrix-thresholds and --policy-matrix-prefetch-strategies"
            )
        if args.policy_matrix_parent is not None or args.policy_matrix_case is not None:
            raise ValueError(
                "--policy-matrix cannot be combined with internal matrix child flags"
            )
    if args.golden_matrix is not None:
        if args.worker:
            raise ValueError("--golden-matrix is not supported in worker mode")
        if args.policy_matrix is not None:
            raise ValueError("--golden-matrix cannot be combined with --policy-matrix")
        if args.golden_matrix_parent is not None or args.golden_matrix_case is not None:
            raise ValueError(
                "--golden-matrix cannot be combined with internal matrix child flags"
            )
    if args.policy_mode != "manual":
        if threshold_sweep or hotset_top_k_sweep or hotset_min_bytes_sweep or prefetch_sweep:
            raise ValueError("--policy-mode cannot be combined with policy sweeps")
        if args.policy_matrix is not None or args.golden_matrix is not None:
            raise ValueError(
                "--policy-mode cannot be combined with matrix presets"
            )
    requested_policy_knobs = _build_policy_knobs(
        args.mmap_small_tensor_copy_max_bytes,
        mmap_prefetch_strategy=args.mmap_prefetch_strategy,
        mmap_hotset_promotion_top_k=args.mmap_hotset_promotion_top_k,
        mmap_hotset_promotion_min_bytes=args.mmap_hotset_promotion_min_bytes,
    )
    requested_policy_signature = _policy_signature(requested_policy_knobs)
    if args.policy_mode == "copy" and requested_policy_signature != "default":
        raise ValueError("--policy-mode copy cannot be combined with mmap policy knobs")
    if args.policy_mode == "mapped" and requested_policy_signature != "default":
        raise ValueError(
            "--policy-mode mapped requires the default mmap knob set"
        )
    if args.policy_mode == "hybrid" and requested_policy_signature == "default":
        raise ValueError(
            "--policy-mode hybrid requires at least one non-default mmap policy knob"
        )
    if threshold_sweep or hotset_top_k_sweep or hotset_min_bytes_sweep or prefetch_sweep:
        if args.worker:
            raise ValueError("policy sweeps are not supported in worker mode")
        base_argv = raw_argv
        for option in [
            "--sweep-mmap-small-tensor-copy-max-bytes",
            "--sweep-mmap-hotset-promotion-top-k",
            "--sweep-mmap-hotset-promotion-min-bytes",
            "--sweep-mmap-prefetch-strategy",
            "--mmap-small-tensor-copy-max-bytes",
            "--mmap-hotset-promotion-top-k",
            "--mmap-hotset-promotion-min-bytes",
            "--mmap-prefetch-strategy",
        ]:
            base_argv = _remove_cli_option(base_argv, option, takes_value=True)
        threshold_values = (
            threshold_sweep
            if threshold_sweep
            else [args.mmap_small_tensor_copy_max_bytes]
        )
        hotset_top_k_values = (
            hotset_top_k_sweep
            if hotset_top_k_sweep
            else [args.mmap_hotset_promotion_top_k]
        )
        hotset_min_bytes_values = (
            hotset_min_bytes_sweep
            if hotset_min_bytes_sweep
            else [args.mmap_hotset_promotion_min_bytes]
        )
        prefetch_values = (
            prefetch_sweep if prefetch_sweep else [args.mmap_prefetch_strategy]
        )
        policy_cases: list[dict] = []
        seen_policy_signatures: set[str] = set()
        for prefetch_strategy in prefetch_values:
            for threshold in threshold_values:
                for hotset_top_k in hotset_top_k_values:
                    for hotset_min_bytes in hotset_min_bytes_values:
                        policy_knobs = _build_policy_knobs(
                            threshold,
                            mmap_prefetch_strategy=prefetch_strategy,
                            mmap_hotset_promotion_top_k=hotset_top_k,
                            mmap_hotset_promotion_min_bytes=hotset_min_bytes,
                        )
                        signature = _policy_signature(policy_knobs)
                        if signature in seen_policy_signatures:
                            continue
                        seen_policy_signatures.add(signature)
                        policy_cases.append(policy_knobs)
        total_policies = len(policy_cases)
        sweep_failed = False
        sweep_index = 0
        for policy_knobs in policy_cases:
            sweep_index += 1
            threshold = policy_knobs.get("mmap_small_tensor_copy_max_bytes")
            hotset_top_k = _normalize_hotset_promotion_top_k(
                policy_knobs.get("mmap_hotset_promotion_top_k")
            )
            hotset_min_bytes = _normalize_hotset_promotion_min_bytes(
                policy_knobs.get("mmap_hotset_promotion_min_bytes")
            )
            prefetch_strategy = str(
                policy_knobs.get("mmap_prefetch_strategy") or "sequential"
            )
            print(
                f"\nPolicy sweep [{sweep_index}/{total_policies}]: "
                f"{_policy_summary(policy_knobs)}"
            )
            cmd = [sys.executable, script_path, *base_argv]
            if threshold is not None:
                cmd.extend(
                    ["--mmap-small-tensor-copy-max-bytes", str(threshold)]
                )
            if hotset_top_k is not None:
                cmd.extend(["--mmap-hotset-promotion-top-k", str(hotset_top_k)])
            if hotset_min_bytes is not None:
                cmd.extend(
                    ["--mmap-hotset-promotion-min-bytes", str(hotset_min_bytes)]
                )
            if prefetch_strategy != "sequential":
                cmd.extend(["--mmap-prefetch-strategy", prefetch_strategy])
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                sweep_failed = True
        return 1 if sweep_failed else 0

    if args.worker:
        fmt = _resolve_format(args.file, args.format)
        result = _run_worker(
            args.file,
            fmt,
            args.memory_map == "1",
            args.gguf_nvfp4_compat,
            args.mmap_small_tensor_copy_max_bytes,
            args.mmap_hotset_promotion_top_k,
            args.mmap_hotset_promotion_min_bytes,
            args.mmap_prefetch_strategy,
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
    repo_root = Path(__file__).resolve().parents[2]
    environment_metadata = _collect_environment_metadata(repo_root)
    history_path = Path(args.history_json).expanduser() if args.history_json else None
    prior_history = _load_history_jsonl(history_path) if history_path else []
    if args.demo_preset is not None:
        return _run_demo_preset(
            args=args,
            provided_flags=provided_flags,
            script_path=script_path,
        )
    if args.report_history:
        report = _build_history_report(prior_history, top_n=args.report_top)
        if args.report_format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.report_format == "csv":
            print(_write_history_report_csv(report))
        else:
            _print_history_report(report)
        return 0
    if args.attempts > 1:
        print(
            "Consensus mode enabled: "
            f"attempts={args.attempts} min_pass={consensus_min_pass}"
        )
    if args.profile:
        print(f"Profile enabled: {args.profile}")
    if args.cache_mode != "inherit":
        print(f"Cache mode enabled: {args.cache_mode}")
    git_head = environment_metadata.get("git_head_short") or "unknown"
    host_chip = environment_metadata.get("host_chip") or "unknown"
    print(
        "Environment: "
        f"git={git_head} host={host_chip} "
        f"python={environment_metadata['python_version']}"
    )
    policy_knobs = requested_policy_knobs
    policy_signature = requested_policy_signature
    policy_summary_text = _policy_summary(policy_knobs)
    if args.policy_mode == "manual":
        print(f"Policy: {policy_summary_text}")
    elif args.policy_mode == "copy":
        print("Policy request: copy-only")
    elif args.policy_mode == "mapped":
        print("Policy request: mapped default")
    else:
        print(
            "Policy request: "
            f"{args.policy_mode} "
            f"({policy_summary_text if args.policy_mode == 'hybrid' else policy_signature})"
        )
    if args.policy_matrix_parent is not None:
        print(
            "Policy matrix: "
            f"{args.policy_matrix_parent} / {args.policy_matrix_case or 'unnamed case'}"
        )
    if args.golden_matrix_parent is not None:
        print(
            "Golden matrix: "
            f"{args.golden_matrix_parent} / {args.golden_matrix_case or 'unnamed case'}"
        )

    file_paths, discovered_formats, discovered_details, discovered_candidates = _resolve_input_files(
        args
    )
    if discovered_candidates:
        discover_roots = (
            [Path(root).expanduser().resolve() for root in args.discover_root]
            if args.discover_root
            else [Path.cwd()]
        )
        print(
            "Discovery selected "
            f"{len(discovered_candidates)} model file(s) from {len(discover_roots)} root(s)."
        )
        for candidate in discovered_candidates:
            suffix = f" [{candidate['model_id']}]" if candidate.get("model_id") else ""
            print(f"  discovered: {candidate['path']}{suffix}")

    if args.policy_matrix is not None:
        catalog = _build_input_catalog(
            prior_history=prior_history,
            file_paths=file_paths,
            requested_format=args.format,
            discovered_formats=discovered_formats,
            discovered_details=discovered_details,
        )
        selected_targets, matrix_notes = _select_matrix_targets(catalog)
        if not selected_targets:
            raise ValueError(
                "--policy-matrix compact could not find a dense safetensors or "
                "quantized GGUF target from the provided files, discovery roots, or history"
            )
        if matrix_notes:
            print("Policy matrix notes:")
            for note in matrix_notes:
                print(f"  - {note}")
        matrix_cases, matrix_cache_modes = _compact_policy_matrix_cases(args)
        total_cases = (
            len(selected_targets) * len(matrix_cases) * len(matrix_cache_modes)
        )
        print(
            "Policy matrix enabled: "
            f"{args.policy_matrix} targets={len(selected_targets)} "
            f"cache_modes={len(matrix_cache_modes)} policies={len(matrix_cases)} "
            f"total_cases={total_cases}"
        )
        matrix_failed = False
        matrix_index = 0
        for title, target in selected_targets:
            target_fmt = str(target.get("format") or "unknown")
            target_file = str(target.get("file") or "")
            prefer_synth_decode = target_fmt != "gguf"
            for cache_mode in matrix_cache_modes:
                for case in matrix_cases:
                    matrix_index += 1
                    case_label = f"{title} / {cache_mode} / {case['label']}"
                    print(f"\nMatrix case [{matrix_index}/{total_cases}]: {case_label}")
                    cmd = _build_experiment_command(
                        script_path=script_path,
                        file_path=target_file,
                        fmt=target_fmt,
                        cache_mode=cache_mode,
                        args=args,
                        prefer_synth_decode=prefer_synth_decode,
                        policy_knobs=case["policy_knobs"],
                        requested_policy_mode="manual",
                        policy_matrix_preset=args.policy_matrix,
                        policy_matrix_case=case_label,
                    )
                    completed = subprocess.run(cmd, shell=True, check=False)
                    if completed.returncode != 0:
                        matrix_failed = True
        return 1 if matrix_failed else 0

    if args.golden_matrix is not None:
        matrix_cases, matrix_cache_modes = _compact_policy_matrix_cases(args)
        with tempfile.TemporaryDirectory(prefix="load_mmap_golden_") as temp_dir:
            selected_targets = _build_golden_matrix_targets(Path(temp_dir))
            total_cases = (
                len(selected_targets) * len(matrix_cases) * len(matrix_cache_modes)
            )
            print(
                "Golden matrix enabled: "
                f"{args.golden_matrix} targets={len(selected_targets)} "
                f"cache_modes={len(matrix_cache_modes)} policies={len(matrix_cases)} "
                f"total_cases={total_cases}"
            )
            matrix_failed = False
            matrix_index = 0
            for title, target in selected_targets:
                target_fmt = str(target.get("format") or "unknown")
                target_file = str(target.get("file") or "")
                prefer_synth_decode = target_fmt != "gguf"
                for cache_mode in matrix_cache_modes:
                    for case in matrix_cases:
                        matrix_index += 1
                        case_label = f"{title} / {cache_mode} / {case['label']}"
                        print(
                            f"\nGolden case [{matrix_index}/{total_cases}]: {case_label}"
                        )
                        cmd = _build_experiment_command(
                            script_path=script_path,
                            file_path=target_file,
                            fmt=target_fmt,
                            cache_mode=cache_mode,
                            args=args,
                            prefer_synth_decode=prefer_synth_decode,
                            policy_knobs=case["policy_knobs"],
                            requested_policy_mode="manual",
                            golden_matrix_preset=args.golden_matrix,
                            golden_matrix_case=case_label,
                        )
                        completed = subprocess.run(cmd, shell=True, check=False)
                        if completed.returncode != 0:
                            matrix_failed = True
            return 1 if matrix_failed else 0

    if not file_paths:
        raise ValueError(
            "provide at least one file path or use --discover-models "
            "with --discover-root"
        )

    failed = False

    for file_path in file_paths:
        discovered_info = discovered_details.get(file_path)
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
        if decode_cmd_template:
            decode_mode = (
                "llama_bench" if args.decode_llama_bench_binary is not None else "custom"
            )
        elif args.decode_synth:
            decode_mode = "synth"
        else:
            decode_mode = "none"
        auto_policy_probe = None
        auto_policy_probe_stats = None
        model_metadata = _infer_model_metadata(
            file_path,
            fmt,
            discovered_info=discovered_info,
        )
        if args.policy_mode == "auto":
            auto_probe_decode_tokens = (
                min(max(args.decode_synth_tokens, 16), 32)
                if decode_mode == "synth"
                else 0
            )
            auto_policy_probe = _run_trials(
                script_path,
                file_path,
                fmt,
                True,
                args.gguf_nvfp4_compat,
                None,
                None,
                None,
                "sequential",
                1,
                0,
                args.debug_io,
                auto_probe_decode_tokens,
                min(args.decode_synth_max_elems, 1_000_000),
                1,
                args.cache_mode,
                1 if args.cache_mode == "steady-state" else 0,
                collect_mmap_stats=True,
            )[0]
            auto_policy_probe_stats = auto_policy_probe.get("mmap_stats")
            if auto_policy_probe_stats is not None:
                model_metadata = _infer_model_metadata(
                    file_path,
                    fmt,
                    fallback_reasons=auto_policy_probe_stats.get("fallback_reasons"),
                    discovered_info=discovered_info,
                )
            auto_policy_decision = _choose_auto_policy(
                file_path=file_path,
                fmt=fmt,
                model_metadata=model_metadata,
                cache_mode=args.cache_mode,
                decode_mode=decode_mode,
                prior_history=prior_history,
                probe_stats=auto_policy_probe_stats,
                probe_trial=auto_policy_probe,
            )
            effective_policy_memory_map = bool(
                auto_policy_decision["effective_memory_map"]
            )
            effective_policy_knobs = auto_policy_decision["effective_policy_knobs"]
            effective_policy_mode = str(auto_policy_decision["effective_policy_mode"])
            effective_policy_signature = str(
                auto_policy_decision["effective_policy_signature"]
            )
            effective_policy_summary = str(auto_policy_decision["effective_policy_summary"])
        elif args.policy_mode == "copy":
            auto_policy_decision = None
            effective_policy_memory_map = False
            effective_policy_knobs = _build_policy_knobs(None)
            effective_policy_mode = "copy"
            effective_policy_signature = "copy"
            effective_policy_summary = "copy-only policy"
        elif args.policy_mode == "mapped":
            auto_policy_decision = None
            effective_policy_memory_map = True
            effective_policy_knobs = _build_policy_knobs(None)
            effective_policy_mode = "mapped"
            effective_policy_signature = _effective_policy_signature(
                memory_map=True, policy_knobs=effective_policy_knobs
            )
            effective_policy_summary = _effective_policy_summary(
                memory_map=True, policy_knobs=effective_policy_knobs
            )
        else:
            auto_policy_decision = None
            effective_policy_memory_map = True
            effective_policy_knobs = policy_knobs
            effective_policy_mode = _classify_effective_policy_mode(
                True, effective_policy_knobs
            )
            effective_policy_signature = _effective_policy_signature(
                memory_map=True, policy_knobs=effective_policy_knobs
            )
            effective_policy_summary = _effective_policy_summary(
                memory_map=True, policy_knobs=effective_policy_knobs
            )
        effective_threshold = effective_policy_knobs.get(
            "mmap_small_tensor_copy_max_bytes"
        )
        effective_hotset_top_k = _normalize_hotset_promotion_top_k(
            effective_policy_knobs.get("mmap_hotset_promotion_top_k")
        )
        effective_hotset_min_bytes = _normalize_hotset_promotion_min_bytes(
            effective_policy_knobs.get("mmap_hotset_promotion_min_bytes")
        )
        effective_prefetch_strategy = str(
            effective_policy_knobs.get("mmap_prefetch_strategy") or "sequential"
        )
        load_improve_history: list[float] = []
        rss_reduce_history: list[float] = []
        decode_regression_history: list[float] = []
        decode_first_token_regression_history: list[float] = []
        copy_minor_fault_history: list[float] = []
        mmap_minor_fault_history: list[float] = []
        copy_major_fault_history: list[float] = []
        mmap_major_fault_history: list[float] = []
        copy_load_peak_memory_history: list[float] = []
        mmap_load_peak_memory_history: list[float] = []
        copy_total_peak_memory_history: list[float] = []
        mmap_total_peak_memory_history: list[float] = []
        load_call_improve_history: list[float] = []
        parse_improve_history: list[float] = []
        tensor_setup_improve_history: list[float] = []
        first_eval_improve_history: list[float] = []

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
            if total_attempts == 1:
                print(
                    "  effective policy "
                    f"mode={effective_policy_mode} signature={effective_policy_signature}"
                )
                print(f"    {effective_policy_summary}")
                if auto_policy_decision is not None:
                    print(f"    auto: {auto_policy_decision['summary']}")
                    probe_decode_summary = (
                        auto_policy_decision.get("probe_decode_summary") or {}
                    )
                    if probe_decode_summary:
                        print(
                            "    auto probe decode "
                            f"first_token={probe_decode_summary.get('decode_first_token_s'):.4f}s "
                            f"tok_s={probe_decode_summary.get('decode_tok_s'):.1f} "
                            f"first_token_faults={probe_decode_summary.get('first_token_faults')} "
                            f"steady_faults={probe_decode_summary.get('steady_faults')}"
                        )
                    for candidate in auto_policy_decision["candidates"][:3]:
                        print(
                            f"    candidate {candidate['label']}: "
                            f"score={candidate['score']:.2f} "
                            f"history={candidate['history_support']}"
                        )
                        for reason in candidate["reasons"][:3]:
                            print(f"      why: {reason}")

            attempt_meta = _collect_attempt_metadata()
            if args.show_trials:
                print(
                    "  host "
                    f"loadavg_1m={attempt_meta.get('loadavg_1m')} "
                    f"thermal={attempt_meta.get('thermal_pressure')} "
                    f"mem_pressure={attempt_meta.get('memory_pressure_state')}"
                )

            if args.interleave_modes and effective_policy_memory_map:
                copy_trials, mmap_trials = _run_interleaved_trials(
                    script_path=script_path,
                    path=file_path,
                    fmt=fmt,
                    policy_memory_map=effective_policy_memory_map,
                    gguf_nvfp4_compat=args.gguf_nvfp4_compat,
                    mmap_small_tensor_copy_max_bytes=effective_threshold,
                    mmap_hotset_promotion_top_k=effective_hotset_top_k,
                    mmap_hotset_promotion_min_bytes=effective_hotset_min_bytes,
                    mmap_prefetch_strategy=effective_prefetch_strategy,
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
                    args.mmap_small_tensor_copy_max_bytes,
                    args.mmap_hotset_promotion_top_k,
                    args.mmap_hotset_promotion_min_bytes,
                    args.mmap_prefetch_strategy,
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
                    effective_policy_memory_map,
                    args.gguf_nvfp4_compat,
                    effective_threshold,
                    effective_hotset_top_k,
                    effective_hotset_min_bytes,
                    effective_prefetch_strategy,
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
            copy_load_peak_memory = [
                x.get("load_peak_memory_bytes") for x in copy_trials
            ]
            mmap_load_peak_memory = [
                x.get("load_peak_memory_bytes") for x in mmap_trials
            ]
            copy_total_peak_memory = [
                x.get("total_peak_memory_bytes") for x in copy_trials
            ]
            mmap_total_peak_memory = [
                x.get("total_peak_memory_bytes") for x in mmap_trials
            ]
            copy_load_call = _optional_trial_metric(copy_trials, "load_call_s")
            mmap_load_call = _optional_trial_metric(mmap_trials, "load_call_s")
            copy_first_eval = _optional_trial_metric(copy_trials, "first_eval_s")
            mmap_first_eval = _optional_trial_metric(mmap_trials, "first_eval_s")
            copy_load_call_minor_faults = _optional_trial_metric(
                copy_trials, "load_call_minor_faults"
            )
            mmap_load_call_minor_faults = _optional_trial_metric(
                mmap_trials, "load_call_minor_faults"
            )
            copy_load_call_major_faults = _optional_trial_metric(
                copy_trials, "load_call_major_faults"
            )
            mmap_load_call_major_faults = _optional_trial_metric(
                mmap_trials, "load_call_major_faults"
            )
            copy_first_eval_minor_faults = _optional_trial_metric(
                copy_trials, "first_eval_minor_faults"
            )
            mmap_first_eval_minor_faults = _optional_trial_metric(
                mmap_trials, "first_eval_minor_faults"
            )
            copy_first_eval_major_faults = _optional_trial_metric(
                copy_trials, "first_eval_major_faults"
            )
            mmap_first_eval_major_faults = _optional_trial_metric(
                mmap_trials, "first_eval_major_faults"
            )
            copy_phase_open_map = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "open_map_seconds"
            )
            mmap_phase_open_map = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "open_map_seconds"
            )
            copy_phase_parse = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "parse_seconds"
            )
            mmap_phase_parse = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "parse_seconds"
            )
            copy_phase_tensor_setup = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "tensor_setup_seconds"
            )
            mmap_phase_tensor_setup = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "tensor_setup_seconds"
            )
            copy_phase_open_map_minor_faults = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "open_map_minor_faults"
            )
            mmap_phase_open_map_minor_faults = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "open_map_minor_faults"
            )
            copy_phase_open_map_major_faults = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "open_map_major_faults"
            )
            mmap_phase_open_map_major_faults = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "open_map_major_faults"
            )
            copy_phase_parse_minor_faults = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "parse_minor_faults"
            )
            mmap_phase_parse_minor_faults = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "parse_minor_faults"
            )
            copy_phase_parse_major_faults = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "parse_major_faults"
            )
            mmap_phase_parse_major_faults = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "parse_major_faults"
            )
            copy_phase_tensor_setup_minor_faults = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "tensor_setup_minor_faults"
            )
            mmap_phase_tensor_setup_minor_faults = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "tensor_setup_minor_faults"
            )
            copy_phase_tensor_setup_major_faults = _optional_nested_trial_metric(
                copy_trials, "load_phase_stats", "tensor_setup_major_faults"
            )
            mmap_phase_tensor_setup_major_faults = _optional_nested_trial_metric(
                mmap_trials, "load_phase_stats", "tensor_setup_major_faults"
            )

            copy_time_med = _median(copy_time)
            mmap_time_med = _median(mmap_time)
            copy_rss_med = _median(copy_rss)
            mmap_rss_med = _median(mmap_rss)
            copy_minor_faults_med = _median(copy_minor_faults)
            mmap_minor_faults_med = _median(mmap_minor_faults)
            copy_major_faults_med = _median(copy_major_faults)
            mmap_major_faults_med = _median(mmap_major_faults)
            copy_load_peak_memory_med = _median_optional(copy_load_peak_memory)
            mmap_load_peak_memory_med = _median_optional(mmap_load_peak_memory)
            copy_total_peak_memory_med = _median_optional(copy_total_peak_memory)
            mmap_total_peak_memory_med = _median_optional(mmap_total_peak_memory)
            copy_load_call_med = _median_optional(copy_load_call)
            mmap_load_call_med = _median_optional(mmap_load_call)
            copy_first_eval_med = _median_optional(copy_first_eval)
            mmap_first_eval_med = _median_optional(mmap_first_eval)
            copy_load_call_minor_faults_med = _median_optional(
                copy_load_call_minor_faults
            )
            mmap_load_call_minor_faults_med = _median_optional(
                mmap_load_call_minor_faults
            )
            copy_load_call_major_faults_med = _median_optional(
                copy_load_call_major_faults
            )
            mmap_load_call_major_faults_med = _median_optional(
                mmap_load_call_major_faults
            )
            copy_first_eval_minor_faults_med = _median_optional(
                copy_first_eval_minor_faults
            )
            mmap_first_eval_minor_faults_med = _median_optional(
                mmap_first_eval_minor_faults
            )
            copy_first_eval_major_faults_med = _median_optional(
                copy_first_eval_major_faults
            )
            mmap_first_eval_major_faults_med = _median_optional(
                mmap_first_eval_major_faults
            )
            copy_phase_open_map_med = _median_optional(copy_phase_open_map)
            mmap_phase_open_map_med = _median_optional(mmap_phase_open_map)
            copy_phase_parse_med = _median_optional(copy_phase_parse)
            mmap_phase_parse_med = _median_optional(mmap_phase_parse)
            copy_phase_tensor_setup_med = _median_optional(copy_phase_tensor_setup)
            mmap_phase_tensor_setup_med = _median_optional(mmap_phase_tensor_setup)
            copy_phase_open_map_minor_faults_med = _median_optional(
                copy_phase_open_map_minor_faults
            )
            mmap_phase_open_map_minor_faults_med = _median_optional(
                mmap_phase_open_map_minor_faults
            )
            copy_phase_open_map_major_faults_med = _median_optional(
                copy_phase_open_map_major_faults
            )
            mmap_phase_open_map_major_faults_med = _median_optional(
                mmap_phase_open_map_major_faults
            )
            copy_phase_parse_minor_faults_med = _median_optional(
                copy_phase_parse_minor_faults
            )
            mmap_phase_parse_minor_faults_med = _median_optional(
                mmap_phase_parse_minor_faults
            )
            copy_phase_parse_major_faults_med = _median_optional(
                copy_phase_parse_major_faults
            )
            mmap_phase_parse_major_faults_med = _median_optional(
                mmap_phase_parse_major_faults
            )
            copy_phase_tensor_setup_minor_faults_med = _median_optional(
                copy_phase_tensor_setup_minor_faults
            )
            mmap_phase_tensor_setup_minor_faults_med = _median_optional(
                mmap_phase_tensor_setup_minor_faults
            )
            copy_phase_tensor_setup_major_faults_med = _median_optional(
                copy_phase_tensor_setup_major_faults
            )
            mmap_phase_tensor_setup_major_faults_med = _median_optional(
                mmap_phase_tensor_setup_major_faults
            )

            load_improve_pct = _improvement_pct(copy_time_med, mmap_time_med)
            rss_reduce_pct = _improvement_pct(copy_rss_med, mmap_rss_med)
            load_call_improve_pct = (
                _improvement_pct(copy_load_call_med, mmap_load_call_med)
                if copy_load_call_med is not None and mmap_load_call_med is not None
                else None
            )
            parse_improve_pct = (
                _improvement_pct(copy_phase_parse_med, mmap_phase_parse_med)
                if copy_phase_parse_med is not None and mmap_phase_parse_med is not None
                else None
            )
            tensor_setup_improve_pct = (
                _improvement_pct(copy_phase_tensor_setup_med, mmap_phase_tensor_setup_med)
                if copy_phase_tensor_setup_med is not None
                and mmap_phase_tensor_setup_med is not None
                else None
            )
            first_eval_improve_pct = (
                _improvement_pct(copy_first_eval_med, mmap_first_eval_med)
                if copy_first_eval_med is not None and mmap_first_eval_med is not None
                else None
            )
            load_peak_memory_reduce_pct = (
                _improvement_pct(copy_load_peak_memory_med, mmap_load_peak_memory_med)
                if copy_load_peak_memory_med is not None
                and mmap_load_peak_memory_med is not None
                else None
            )
            total_peak_memory_reduce_pct = (
                _improvement_pct(copy_total_peak_memory_med, mmap_total_peak_memory_med)
                if copy_total_peak_memory_med is not None
                and mmap_total_peak_memory_med is not None
                else None
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
            if copy_load_call_med is not None or mmap_load_call_med is not None:
                print(
                    "  load phases copy "
                    f"open/map={_format_seconds(copy_phase_open_map_med)} "
                    f"parse={_format_seconds(copy_phase_parse_med)} "
                    f"tensor_setup={_format_seconds(copy_phase_tensor_setup_med)} "
                    f"first_eval={_format_seconds(copy_first_eval_med)}"
                )
                print(
                    "              mmap "
                    f"open/map={_format_seconds(mmap_phase_open_map_med)} "
                    f"parse={_format_seconds(mmap_phase_parse_med)} "
                    f"tensor_setup={_format_seconds(mmap_phase_tensor_setup_med)} "
                    f"first_eval={_format_seconds(mmap_first_eval_med)}"
                )
                print(
                    "  phase deltas improve% "
                    f"load_call={_format_pct(load_call_improve_pct)} "
                    f"parse={_format_pct(parse_improve_pct)} "
                    f"tensor_setup={_format_pct(tensor_setup_improve_pct)} "
                    f"first_eval={_format_pct(first_eval_improve_pct)}"
                )
                print(
                    "  phase faults copy "
                    f"open/map={_format_fault_pair(copy_phase_open_map_minor_faults_med, copy_phase_open_map_major_faults_med)} "
                    f"parse={_format_fault_pair(copy_phase_parse_minor_faults_med, copy_phase_parse_major_faults_med)} "
                    f"tensor_setup={_format_fault_pair(copy_phase_tensor_setup_minor_faults_med, copy_phase_tensor_setup_major_faults_med)} "
                    f"first_eval={_format_fault_pair(copy_first_eval_minor_faults_med, copy_first_eval_major_faults_med)}"
                )
                print(
                    "              mmap "
                    f"open/map={_format_fault_pair(mmap_phase_open_map_minor_faults_med, mmap_phase_open_map_major_faults_med)} "
                    f"parse={_format_fault_pair(mmap_phase_parse_minor_faults_med, mmap_phase_parse_major_faults_med)} "
                    f"tensor_setup={_format_fault_pair(mmap_phase_tensor_setup_minor_faults_med, mmap_phase_tensor_setup_major_faults_med)} "
                    f"first_eval={_format_fault_pair(mmap_first_eval_minor_faults_med, mmap_first_eval_major_faults_med)}"
                )
            if any(
                value is not None
                for value in (
                    copy_load_peak_memory_med,
                    mmap_load_peak_memory_med,
                    copy_total_peak_memory_med,
                    mmap_total_peak_memory_med,
                )
            ):
                print(
                    "  mlx peak memory load "
                    f"copy={_format_mib(copy_load_peak_memory_med)} "
                    f"mmap={_format_mib(mmap_load_peak_memory_med)}"
                )
                print(
                    "  mlx peak memory total "
                    f"copy={_format_mib(copy_total_peak_memory_med)} "
                    f"mmap={_format_mib(mmap_total_peak_memory_med)}"
                )
                print(
                    "  peak memory deltas improve% "
                    f"load={_format_pct(load_peak_memory_reduce_pct)} "
                    f"total={_format_pct(total_peak_memory_reduce_pct)}"
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
            decode_first_token_regression_pct = None
            decode_regression_ok = True
            decode_cv_ok = True
            decode_gate_effective_pct = args.decode_regression_gate_pct
            decode_aa_floor = 0.0
            copy_first_token_med = None
            mmap_first_token_med = None
            copy_decode_first_token_minor_faults_med = None
            copy_decode_first_token_major_faults_med = None
            mmap_decode_first_token_minor_faults_med = None
            mmap_decode_first_token_major_faults_med = None
            copy_decode_steady_minor_faults_med = None
            copy_decode_steady_major_faults_med = None
            mmap_decode_steady_minor_faults_med = None
            mmap_decode_steady_major_faults_med = None

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
                    mode=effective_policy_mode,
                    memory_map=effective_policy_memory_map,
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
                print("  decode first-token split unavailable for external decode command")
            elif args.decode_synth:
                copy_tok_s_trials = [x["decode_tok_s"] for x in copy_trials]
                mmap_tok_s_trials = [x["decode_tok_s"] for x in mmap_trials]
                copy_first_token_trials = _optional_trial_metric(
                    copy_trials, "decode_first_token_s"
                )
                mmap_first_token_trials = _optional_trial_metric(
                    mmap_trials, "decode_first_token_s"
                )
                copy_tok_s = _median(copy_tok_s_trials)
                mmap_tok_s = _median(mmap_tok_s_trials)
                copy_first_token_med = _median_optional(copy_first_token_trials)
                mmap_first_token_med = _median_optional(mmap_first_token_trials)
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
                if copy_first_token_med is not None and mmap_first_token_med is not None:
                    decode_first_token_regression_pct = _median_latency_regression_pct(
                        copy_values=[
                            float(v) for v in copy_first_token_trials if v is not None
                        ],
                        mmap_values=[
                            float(v) for v in mmap_first_token_trials if v is not None
                        ],
                        pairwise=args.interleave_modes,
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
                copy_decode_first_token_minor_faults_med = _median_optional(
                    _optional_trial_metric(copy_trials, "decode_first_token_minor_faults")
                )
                copy_decode_first_token_major_faults_med = _median_optional(
                    _optional_trial_metric(copy_trials, "decode_first_token_major_faults")
                )
                mmap_decode_first_token_minor_faults_med = _median_optional(
                    _optional_trial_metric(mmap_trials, "decode_first_token_minor_faults")
                )
                mmap_decode_first_token_major_faults_med = _median_optional(
                    _optional_trial_metric(mmap_trials, "decode_first_token_major_faults")
                )
                copy_decode_steady_minor_faults_med = _median_optional(
                    _optional_trial_metric(copy_trials, "decode_steady_minor_faults")
                )
                copy_decode_steady_major_faults_med = _median_optional(
                    _optional_trial_metric(copy_trials, "decode_steady_major_faults")
                )
                mmap_decode_steady_minor_faults_med = _median_optional(
                    _optional_trial_metric(mmap_trials, "decode_steady_minor_faults")
                )
                mmap_decode_steady_major_faults_med = _median_optional(
                    _optional_trial_metric(mmap_trials, "decode_steady_major_faults")
                )
                if args.show_trials:
                    print(
                        "  decode trials copy_tok_s="
                        f"{[round(v, 3) for v in copy_tok_s_trials]} "
                        "mmap_tok_s="
                        f"{[round(v, 3) for v in mmap_tok_s_trials]}"
                    )
                    print(
                        "  decode first-token ms copy="
                        f"{[round(1000.0 * float(v), 2) for v in copy_first_token_trials if v is not None]} "
                        "mmap="
                        f"{[round(1000.0 * float(v), 2) for v in mmap_first_token_trials if v is not None]}"
                    )
                print(
                    "  decode first-token "
                    f"copy={_format_ms(copy_first_token_med)} "
                    f"mmap={_format_ms(mmap_first_token_med)} "
                    f"regression={_format_pct(decode_first_token_regression_pct)}"
                )
                print(
                    "  decode first-token faults "
                    f"copy(min/maj)={_format_fault_pair(copy_decode_first_token_minor_faults_med, copy_decode_first_token_major_faults_med)} "
                    f"mmap(min/maj)={_format_fault_pair(mmap_decode_first_token_minor_faults_med, mmap_decode_first_token_major_faults_med)}"
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
                print(
                    "  decode steady faults "
                    f"copy(min/maj)={_format_fault_pair(copy_decode_steady_minor_faults_med, copy_decode_steady_major_faults_med)} "
                    f"mmap(min/maj)={_format_fault_pair(mmap_decode_steady_minor_faults_med, mmap_decode_steady_major_faults_med)}"
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
                if load_call_improve_pct is not None:
                    load_call_improve_history.append(load_call_improve_pct)
                if parse_improve_pct is not None:
                    parse_improve_history.append(parse_improve_pct)
                if tensor_setup_improve_pct is not None:
                    tensor_setup_improve_history.append(tensor_setup_improve_pct)
                if first_eval_improve_pct is not None:
                    first_eval_improve_history.append(first_eval_improve_pct)
                copy_minor_fault_history.append(copy_minor_faults_med)
                mmap_minor_fault_history.append(mmap_minor_faults_med)
                copy_major_fault_history.append(copy_major_faults_med)
                mmap_major_fault_history.append(mmap_major_faults_med)
                if copy_load_peak_memory_med is not None:
                    copy_load_peak_memory_history.append(copy_load_peak_memory_med)
                if mmap_load_peak_memory_med is not None:
                    mmap_load_peak_memory_history.append(mmap_load_peak_memory_med)
                if copy_total_peak_memory_med is not None:
                    copy_total_peak_memory_history.append(copy_total_peak_memory_med)
                if mmap_total_peak_memory_med is not None:
                    mmap_total_peak_memory_history.append(mmap_total_peak_memory_med)
                if decode_regression_pct is not None:
                    decode_regression_history.append(decode_regression_pct)
                if decode_first_token_regression_pct is not None:
                    decode_first_token_regression_history.append(
                        decode_first_token_regression_pct
                    )
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
                args.mmap_small_tensor_copy_max_bytes,
                args.mmap_hotset_promotion_top_k,
                args.mmap_hotset_promotion_min_bytes,
                args.mmap_prefetch_strategy,
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
        if args.coverage_probe and effective_policy_memory_map:
            probe = _run_trials(
                script_path,
                file_path,
                fmt,
                effective_policy_memory_map,
                args.gguf_nvfp4_compat,
                effective_threshold,
                effective_hotset_top_k,
                effective_hotset_min_bytes,
                effective_prefetch_strategy,
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
                if mmap_coverage["fallback_reason_bytes"]:
                    fallback_byte_summary = ", ".join(
                        f"{key}={_bytes_to_mib(value):.2f}MiB"
                        for key, value in sorted(
                            mmap_coverage["fallback_reason_bytes"].items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    )
                    print(
                        "  mmap fallback materialized bytes "
                        f"{fallback_byte_summary}"
                    )
                if mmap_coverage["fallback_reason_source_bytes"]:
                    fallback_source_summary = ", ".join(
                        f"{key}={_bytes_to_mib(value):.2f}MiB"
                        for key, value in sorted(
                            mmap_coverage["fallback_reason_source_bytes"].items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    )
                    print(f"  mmap fallback source bytes {fallback_source_summary}")
                fallback_priority_order = _format_fallback_priority_order(
                    mmap_coverage["fallback_reasons"],
                    mmap_coverage["fallback_reason_bytes"],
                    fallback_priority_basis=args.fallback_priority_basis,
                )
                if fallback_priority_order:
                    print(
                        "  mmap fallback priority "
                        f"{_fallback_priority_summary(args.fallback_priority_basis)}"
                    )
                    print(f"    order: {fallback_priority_order}")
                hotset_promoted_tensors = mmap_coverage.get("hotset_promoted_tensors") or {}
                if hotset_promoted_tensors:
                    print(
                        "  mmap hotset selection "
                        f"strategy={mmap_coverage.get('hotset_promotion_strategy') or 'unknown'}"
                    )
                    for name, reason in sorted(hotset_promoted_tensors.items()):
                        print(f"    {name}: {reason}")
            else:
                print("  mmap coverage unavailable (no MLX mmap stats emitted)")
        elif args.coverage_probe and auto_policy_probe_stats is not None:
            print(
                "  mmap coverage probe was used for auto-policy selection, "
                "but the chosen policy is copy-only."
            )

        model_metadata = _infer_model_metadata(
            file_path,
            fmt,
            fallback_reasons=(
                mmap_coverage["fallback_reasons"]
                if mmap_coverage is not None
                else (
                    auto_policy_probe_stats.get("fallback_reasons")
                    if auto_policy_probe_stats is not None
                    else None
                )
            ),
            discovered_info=discovered_info,
        )
        summary_payload = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": _make_run_id(
                file_path,
                git_head_short=environment_metadata.get("git_head_short"),
            ),
            "file": file_path,
            "label": model_metadata["label"],
            "format": model_metadata["format"],
            "weight_class": model_metadata["weight_class"],
            "model_class": model_metadata["model_class"],
            "source": model_metadata["source"],
            "model_id": model_metadata["model_id"],
            "size_bytes": model_metadata["size_bytes"],
            "phase": args.decode_phase,
            "decode_mode": decode_mode,
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
            "copy_load_peak_memory_bytes_median": _median(copy_load_peak_memory_history)
            if copy_load_peak_memory_history
            else None,
            "mmap_load_peak_memory_bytes_median": _median(mmap_load_peak_memory_history)
            if mmap_load_peak_memory_history
            else None,
            "copy_total_peak_memory_bytes_median": _median(copy_total_peak_memory_history)
            if copy_total_peak_memory_history
            else None,
            "mmap_total_peak_memory_bytes_median": _median(mmap_total_peak_memory_history)
            if mmap_total_peak_memory_history
            else None,
            "load_call_improve_pct_median": _median(load_call_improve_history)
            if load_call_improve_history
            else None,
            "parse_improve_pct_median": _median(parse_improve_history)
            if parse_improve_history
            else None,
            "tensor_setup_improve_pct_median": _median(tensor_setup_improve_history)
            if tensor_setup_improve_history
            else None,
            "first_eval_improve_pct_median": _median(first_eval_improve_history)
            if first_eval_improve_history
            else None,
            "load_peak_memory_reduce_pct_median": load_peak_memory_reduce_pct,
            "total_peak_memory_reduce_pct_median": total_peak_memory_reduce_pct,
            "decode_regression_pct_median": _median(decode_regression_history)
            if decode_regression_history
            else None,
            "decode_first_token_regression_pct_median": _median(
                decode_first_token_regression_history
            )
            if decode_first_token_regression_history
            else None,
            "phase_timing": {
                "copy": {
                    "load_call_s_median": copy_load_call_med,
                    "open_map_s_median": copy_phase_open_map_med,
                    "parse_s_median": copy_phase_parse_med,
                    "tensor_setup_s_median": copy_phase_tensor_setup_med,
                    "first_eval_s_median": copy_first_eval_med,
                },
                "mmap": {
                    "load_call_s_median": mmap_load_call_med,
                    "open_map_s_median": mmap_phase_open_map_med,
                    "parse_s_median": mmap_phase_parse_med,
                    "tensor_setup_s_median": mmap_phase_tensor_setup_med,
                    "first_eval_s_median": mmap_first_eval_med,
                },
            },
            "phase_faults": {
                "copy": {
                    "open_map_minor_faults_median": copy_phase_open_map_minor_faults_med,
                    "open_map_major_faults_median": copy_phase_open_map_major_faults_med,
                    "parse_minor_faults_median": copy_phase_parse_minor_faults_med,
                    "parse_major_faults_median": copy_phase_parse_major_faults_med,
                    "tensor_setup_minor_faults_median": copy_phase_tensor_setup_minor_faults_med,
                    "tensor_setup_major_faults_median": copy_phase_tensor_setup_major_faults_med,
                    "first_eval_minor_faults_median": copy_first_eval_minor_faults_med,
                    "first_eval_major_faults_median": copy_first_eval_major_faults_med,
                },
                "mmap": {
                    "open_map_minor_faults_median": mmap_phase_open_map_minor_faults_med,
                    "open_map_major_faults_median": mmap_phase_open_map_major_faults_med,
                    "parse_minor_faults_median": mmap_phase_parse_minor_faults_med,
                    "parse_major_faults_median": mmap_phase_parse_major_faults_med,
                    "tensor_setup_minor_faults_median": mmap_phase_tensor_setup_minor_faults_med,
                    "tensor_setup_major_faults_median": mmap_phase_tensor_setup_major_faults_med,
                    "first_eval_minor_faults_median": mmap_first_eval_minor_faults_med,
                    "first_eval_major_faults_median": mmap_first_eval_major_faults_med,
                },
            },
            "decode_phase_timing": {
                "copy": {
                    "first_token_s_median": copy_first_token_med,
                    "steady_tok_s_median": copy_tok_s if decode_enabled else None,
                },
                "mmap": {
                    "first_token_s_median": mmap_first_token_med,
                    "steady_tok_s_median": mmap_tok_s if decode_enabled else None,
                },
            },
            "decode_phase_faults": {
                "copy": {
                    "first_token_minor_faults_median": copy_decode_first_token_minor_faults_med,
                    "first_token_major_faults_median": copy_decode_first_token_major_faults_med,
                    "steady_minor_faults_median": copy_decode_steady_minor_faults_med,
                    "steady_major_faults_median": copy_decode_steady_major_faults_med,
                },
                "mmap": {
                    "first_token_minor_faults_median": mmap_decode_first_token_minor_faults_med,
                    "first_token_major_faults_median": mmap_decode_first_token_major_faults_med,
                    "steady_minor_faults_median": mmap_decode_steady_minor_faults_med,
                    "steady_major_faults_median": mmap_decode_steady_major_faults_med,
                },
            },
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
            "mmap_coverage_fallback_reason_bytes": mmap_coverage["fallback_reason_bytes"]
            if mmap_coverage is not None
            else None,
            "mmap_coverage_fallback_reason_source_bytes": mmap_coverage[
                "fallback_reason_source_bytes"
            ]
            if mmap_coverage is not None
            else None,
            "mmap_coverage_hotset_promoted_tensors": (
                mmap_coverage.get("hotset_promoted_tensors")
                if mmap_coverage is not None
                else None
            ),
            "mmap_coverage_hotset_promotion_strategy": (
                mmap_coverage.get("hotset_promotion_strategy")
                if mmap_coverage is not None
                else None
            ),
            "auto_policy_probe_mmap_coverage": auto_policy_probe_stats,
            "auto_policy_probe_decode": (
                auto_policy_decision.get("probe_decode_summary")
                if auto_policy_decision is not None
                else None
            ),
            "mmap_coverage": mmap_coverage,
            "requested_policy_mode": args.policy_mode,
            "policy_knobs": effective_policy_knobs,
            "policy_signature": effective_policy_signature,
            "policy_summary": effective_policy_summary,
            "requested_policy_knobs": policy_knobs,
            "requested_policy_signature": policy_signature,
            "requested_policy_summary": policy_summary_text,
            "effective_policy_mode": effective_policy_mode,
            "effective_memory_map": effective_policy_memory_map,
            "effective_policy_knobs": effective_policy_knobs,
            "effective_policy_signature": effective_policy_signature,
            "effective_policy_summary": effective_policy_summary,
            "auto_policy_decision": auto_policy_decision,
            "mmap_small_tensor_copy_max_bytes": effective_threshold,
            "mmap_hotset_promotion_top_k": effective_hotset_top_k,
            "mmap_hotset_promotion_min_bytes": effective_hotset_min_bytes,
            "mmap_prefetch_strategy": effective_prefetch_strategy,
            "policy_matrix_parent": args.policy_matrix_parent,
            "policy_matrix_case": args.policy_matrix_case,
            "golden_matrix_parent": args.golden_matrix_parent,
            "golden_matrix_case": args.golden_matrix_case,
            "demo_parent": args.demo_parent,
            "demo_case": args.demo_case,
            "fallback_priority_basis": args.fallback_priority_basis,
            "fallback_priority_summary": _fallback_priority_summary(
                args.fallback_priority_basis
            ),
            "comparison_request": {
                "baseline_run_id": args.baseline_run_id,
                "baseline_git_head": args.baseline_git_head,
            },
            "environment": environment_metadata,
            "code_state": _extract_code_state(environment_metadata),
        }
        summary_payload["history_bucket"] = _build_history_bucket(summary_payload)
        summary_payload["history_bucket_key"] = _history_bucket_key(
            summary_payload["history_bucket"]
        )

        trend_context = _select_trend_history(
            prior_history,
            summary_payload,
            file_path=file_path,
            history_window=args.history_window,
            trend_min_samples=args.trend_min_samples,
        )
        trend_context_summary = _describe_trend_context(trend_context)
        summary_payload["trend_context"] = {
            key: value for key, value in trend_context.items() if key != "records"
        }
        summary_payload["trend_context"]["summary"] = trend_context_summary
        recent = trend_context["records"]
        if history_path:
            if trend_context["scope"] == "bucket":
                print(
                    "  trend scope "
                    f"bucket samples={trend_context['bucket_samples']} "
                    f"key={trend_context['bucket_key']}"
                )
            else:
                print(
                    "  trend scope "
                    f"file samples={trend_context['file_samples']} "
                    f"(bucket_samples={trend_context['bucket_samples']})"
                )
            print(f"    why this bucket: {trend_context_summary}")
        trend_regressions = _collect_trend_regressions(
            summary_payload,
            recent,
            trend_min_samples=args.trend_min_samples,
            trend_mad_mult=args.trend_mad_mult,
        )

        if trend_regressions:
            print("Trend warnings:")
            for warning in trend_regressions:
                print(f"  - {warning}")
            if args.trend_hard_fail:
                file_ok = False
                print("Trend hard-fail enabled: failing file due to trend regression.")

        comparison = _build_comparison_summary(
            summary_payload,
            prior_history=prior_history,
            file_path=file_path,
            trend_context=trend_context,
            baseline_run_id=args.baseline_run_id,
            baseline_git_head=args.baseline_git_head,
        )
        summary_payload["comparison"] = comparison
        if comparison is not None:
            print(f"Comparison: {comparison['summary']}")
            for metric_key in [
                "load_improve_pct_median",
                "rss_reduce_pct_median",
                "load_peak_memory_reduce_pct_median",
                "total_peak_memory_reduce_pct_median",
                "load_call_improve_pct_median",
                "parse_improve_pct_median",
                "tensor_setup_improve_pct_median",
                "first_eval_improve_pct_median",
                "decode_regression_pct_median",
                "decode_first_token_regression_pct_median",
            ]:
                metric = (comparison.get("metrics") or {}).get(metric_key)
                if metric is None:
                    continue
                print(
                    f"  {metric['label']}: current={metric['current']:.2f} "
                    f"ref={metric['reference']:.2f} delta={metric['delta']:+.2f} "
                    f"status={metric['status']}"
                )

        route_suggestions = _build_route_suggestions(
            summary_payload,
            decode_enabled=decode_enabled,
            decode_gate_pct=args.decode_regression_gate_pct,
            fallback_priority_basis=args.fallback_priority_basis,
            trend_regressions=trend_regressions,
        )
        route_suggestions = _calibrate_route_suggestions(
            route_suggestions,
            prior_history,
            bucket_key=summary_payload["history_bucket_key"],
        )
        next_experiments, next_experiment_notes = _build_next_experiments(
            summary_payload,
            route_suggestions,
            prior_history=prior_history,
            script_path=script_path,
            args=args,
        )
        suggested_route_ids = [suggestion["route_id"] for suggestion in route_suggestions]
        attempted_route = (
            {
                "route_id": attempted_route_ref[0],
                "route": attempted_route_ref[1],
            }
            if attempted_route_ref is not None
            else None
        )
        route_outcome, route_outcome_source = _resolve_route_outcome(
            requested_outcome=args.route_outcome,
            comparison=comparison,
            file_ok=file_ok,
        )
        summary_payload["trend_regressions"] = trend_regressions
        summary_payload["route_suggestions"] = route_suggestions
        summary_payload["suggested_route_ids"] = suggested_route_ids
        summary_payload["next_experiments"] = next_experiments
        summary_payload["next_experiment_notes"] = next_experiment_notes
        summary_payload["attempted_route"] = (
            attempted_route["route"] if attempted_route is not None else None
        )
        summary_payload["attempted_route_id"] = (
            attempted_route["route_id"] if attempted_route is not None else None
        )
        summary_payload["attempted_route_was_suggested"] = (
            attempted_route is not None
            and attempted_route["route_id"] in suggested_route_ids
        )
        summary_payload["route_outcome"] = (
            route_outcome if attempted_route is not None else None
        )
        summary_payload["route_outcome_source"] = (
            route_outcome_source if attempted_route is not None else None
        )

        if args.route_suggestions:
            print("Suggested routes:")
            if summary_payload.get("mmap_coverage_fallback_reasons"):
                print(
                    "  fallback ordering: "
                    f"{summary_payload['fallback_priority_summary']}"
                )
            for suggestion in route_suggestions:
                print(
                    f"  {suggestion['route']} "
                    f"[{suggestion['confidence_label']} {suggestion['confidence']:.2f}]: "
                    f"{suggestion['why']}"
                )
                print(f"    confidence: {suggestion['confidence_why']}")
                if suggestion.get("history_summary"):
                    print(f"    history: {suggestion['history_summary']}")
                print(f"    try next: {suggestion['try_next']}")
            if next_experiment_notes:
                print("Experiment notes:")
                for note in next_experiment_notes:
                    print(f"  - {note}")
            if next_experiments:
                print("Next experiments:")
                for experiment in next_experiments:
                    print(f"  {experiment['title']}: {experiment['goal']}")
                    print(f"    cmd: {experiment['command']}")
        if attempted_route is not None:
            print(
                "Route outcome: "
                f"{attempted_route['route']} ({attempted_route['route_id']}) => "
                f"{route_outcome} [{route_outcome_source}]"
            )

        if history_path:
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
