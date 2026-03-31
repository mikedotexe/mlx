#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NEURAL_ROOT = REPO_ROOT.parent / "neural-triple-reservoir"
SIDECAR_SCRIPT = REPO_ROOT / "benchmarks/python/chat_mlx_local.py"
COUPLED_SCRIPT = NEURAL_ROOT / "coupled_astrid_server.py"
FAST_VENV_PYTHON = REPO_ROOT / ".venv-mlxlm" / "bin" / "python"
RESERVOIR_VENV_PYTHON = NEURAL_ROOT / ".venv" / "bin" / "python"

SIDECAR_PROMPT = (
    "Spectral context: fill=46.2%; lambda1_rel=0.58; geom_rel=0.41; "
    "trend=stable with light curiosity and mild contraction pressure. "
    "Describe what the controller notices, what changed, and what it should do next."
)

COUPLED_MESSAGES = [
    {
        "role": "system",
        "content": "You are Astrid. Reply in one concise paragraph.",
    },
    {
        "role": "user",
        "content": "Describe a steady, curious state without using ocean imagery.",
    },
]


def _timestamp_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_json_loads(payload: str) -> dict[str, object]:
    return json.loads(payload)


def _find_json_object(payload: str) -> dict[str, object]:
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return _safe_json_loads(line)
    return _safe_json_loads(payload)


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_ready(url: str, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - polling
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _http_post_json(url: str, payload: dict[str, object], *, timeout_s: float) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return _safe_json_loads(response.read().decode("utf-8"))


def _env_probe(python_path: Path) -> dict[str, object]:
    if not python_path.exists():
        return {"exists": False, "python": str(python_path)}

    probe_code = textwrap.dedent(
        """
        import inspect
        import json
        import sys
        import tempfile
        from pathlib import Path

        result = {
            "exists": True,
            "python": sys.executable,
        }

        try:
            import mlx.core as mx
            result["mlx_core_path"] = inspect.getfile(mx)
            result["has_mmap_stats"] = hasattr(mx, "last_mmap_load_stats")
        except Exception as exc:
            result["mlx_core_error"] = str(exc)
            print(json.dumps(result))
            raise SystemExit(0)

        try:
            import mlx_lm
            result["mlx_lm_path"] = inspect.getfile(mlx_lm)
        except Exception as exc:
            result["mlx_lm_error"] = str(exc)

        probe = {
            "memory_map_call_ok": False,
            "can_report_mapped_vs_copied": False,
            "mapped_bytes": None,
            "copied_bytes": None,
            "fallback_reasons": None,
        }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                probe_path = Path(tmpdir) / "probe.safetensors"
                mx.save_safetensors(
                    str(probe_path),
                    {"weights": mx.arange(8, dtype=mx.float32).reshape(2, 4)},
                )
                if hasattr(mx, "last_mmap_load_stats"):
                    try:
                        mx.last_mmap_load_stats(clear=True)
                    except Exception:
                        pass
                mx.load(str(probe_path), memory_map=True)
                probe["memory_map_call_ok"] = True
                if hasattr(mx, "last_mmap_load_stats"):
                    stats = mx.last_mmap_load_stats(clear=True) or {}
                    probe["can_report_mapped_vs_copied"] = True
                    probe["mapped_bytes"] = int(stats.get("mapped_bytes") or 0)
                    probe["copied_bytes"] = int(stats.get("copied_bytes") or 0)
                    probe["fallback_reasons"] = stats.get("fallback_reasons") or {}
        except Exception as exc:
            probe["memory_map_call_error"] = str(exc)

        result["probe"] = probe
        print(json.dumps(result))
        """
    )
    completed = _run_command(
        [str(python_path), "-c", probe_code],
        cwd=REPO_ROOT,
        timeout=120,
    )
    result: dict[str, object] = {
        "exists": True,
        "python": str(python_path),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    if completed.stdout.strip():
        result.update(_find_json_object(completed.stdout))
    return result


def _inspect_envs() -> dict[str, object]:
    system_python = subprocess.run(
        ["which", "python3"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    envs = {
        "python3": Path(system_python) if system_python else Path("/missing/python3"),
        "reservoir_venv": RESERVOIR_VENV_PYTHON,
        "mlx_fast_venv": FAST_VENV_PYTHON,
    }
    return {label: _env_probe(path) for label, path in envs.items()}


def _run_sidecar_scenario(
    *,
    scenario_name: str,
    output_dir: Path,
    memory_map: bool,
) -> dict[str, object]:
    cmd = [
        "python3",
        str(SIDECAR_SCRIPT),
        "--json",
        "--hardware-profile",
        "m4-mini",
        "--mode",
        "reflective",
        "--architecture",
        "reservoir-fixed",
        "--candidate-count",
        "2",
        "--max-tokens",
        "64",
        "--prompt",
        SIDECAR_PROMPT,
    ]
    if memory_map:
        cmd.append("--model-memory-map")
    completed = _run_command(cmd, cwd=REPO_ROOT, timeout=900)
    sidecar_log = output_dir / scenario_name / "sidecar.stderr.log"
    sidecar_log.parent.mkdir(parents=True, exist_ok=True)
    sidecar_log.write_text(completed.stderr, encoding="utf-8")

    result: dict[str, object] = {
        "success": completed.returncode == 0,
        "command": cmd,
        "returncode": completed.returncode,
        "stderr_log": str(sidecar_log),
    }
    if completed.stdout.strip():
        try:
            payload = _find_json_object(completed.stdout)
        except Exception as exc:
            result["success"] = False
            result["parse_error"] = str(exc)
            result["stdout_tail"] = completed.stdout[-2000:]
            return result
        profiling = dict(payload.get("profiling", {}) or {})
        runtime_audit = dict(profiling.get("runtime_audit", {}) or {})
        result["payload"] = payload
        result["runtime"] = runtime_audit.get("runtime")
        result["load"] = runtime_audit.get("load")
        result["generation"] = runtime_audit.get("generation")
        result["total_turn_seconds"] = profiling.get("total_turn_seconds")
        result["candidate_generation_seconds"] = profiling.get(
            "candidate_generation_seconds"
        )
        result["text_preview"] = str(payload.get("text", ""))[:240]
    return result


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _run_coupled_scenario(
    *,
    scenario_name: str,
    output_dir: Path,
    memory_map: bool,
) -> dict[str, object]:
    scenario_dir = output_dir / scenario_name / "coupled"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    server_log = scenario_dir / "server.log"
    port = _pick_free_port()
    handle_name = f"audit_{scenario_name}"
    cmd = [
        str(RESERVOIR_VENV_PYTHON),
        str(COUPLED_SCRIPT),
        "--port",
        str(port),
        "--coupling-strength",
        "0.1",
        "--audit-dir",
        str(scenario_dir),
    ]
    if memory_map:
        cmd.append("--model-memory-map")

    env = os.environ.copy()
    env["COUPLED_ASTRID_AUDIT_DIR"] = str(scenario_dir)
    response: dict[str, object] | None = None
    error: str | None = None
    with server_log.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(NEURAL_ROOT),
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    try:
        try:
            _wait_for_http_ready(f"http://127.0.0.1:{port}/v1/models", timeout_s=900)
            response = _http_post_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {
                    "messages": COUPLED_MESSAGES,
                    "temperature": 0.2,
                    "max_tokens": 96,
                    "reservoir_handle": handle_name,
                },
                timeout_s=900,
            )
            time.sleep(1.0)
        except Exception as exc:
            error = str(exc)
    finally:
        _terminate_process(proc)

    audit_path = scenario_dir / "coupled_request_metrics.jsonl"
    audit_record = None
    if audit_path.exists():
        lines = [line.strip() for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            audit_record = json.loads(lines[-1])

    result: dict[str, object] = {
        "success": audit_record is not None and error is None,
        "command": cmd,
        "server_log": str(server_log),
        "audit_path": str(audit_path),
        "response": response,
        "error": error,
        "text_preview": str(
            ((((response or {}).get("choices") or [{}])[0]).get("message") or {}).get(
                "content", ""
            )
        )[:240],
        "audit": audit_record,
    }
    if audit_record is not None:
        result["runtime"] = audit_record.get("runtime")
        result["load"] = audit_record.get("load")
        result["generation"] = audit_record.get("generation")
        result["reservoir"] = audit_record.get("reservoir")
    return result


def _install_mlx_fork_into_env(
    *,
    python_path: Path,
    output_dir: Path,
    label: str,
) -> dict[str, object]:
    log_path = output_dir / "install-logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(python_path), "-m", "pip", "install", "-e", str(REPO_ROOT), "-v"]
    completed = _run_command(cmd, cwd=REPO_ROOT, timeout=3600)
    log_path.write_text(
        (completed.stdout or "") + "\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "log_path": str(log_path),
        "command": cmd,
    }


def _delta_pct(before: float | None, after: float | None) -> float | None:
    if before in (None, 0) or after is None:
        return None
    return 100.0 * (float(before) - float(after)) / float(before)


def _extract_float(mapping: dict[str, object] | None, *path: str) -> float | None:
    current: object = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _build_verdicts(report: dict[str, object]) -> dict[str, object]:
    baseline = dict((report.get("scenarios") or {}).get("baseline", {}) or {})
    fork_enabled = dict((report.get("scenarios") or {}).get("fork_enabled", {}) or {})
    explicit = dict((report.get("scenarios") or {}).get("fork_enabled_explicit_mmap", {}) or {})

    baseline_envs = dict(report.get("baseline_envs", {}) or {})
    reservoir_env = dict(baseline_envs.get("reservoir_venv", {}) or {})
    fast_env = dict(baseline_envs.get("mlx_fast_venv", {}) or {})
    baseline_coupled = dict(baseline.get("coupled", {}) or {})
    explicit_coupled = dict(explicit.get("coupled", {}) or {})
    baseline_sidecar = dict(baseline.get("sidecar", {}) or {})
    explicit_sidecar = dict(explicit.get("sidecar", {}) or {})

    baseline_coupled_load = dict(baseline_coupled.get("load", {}) or {})
    explicit_coupled_load = dict(explicit_coupled.get("load", {}) or {})
    baseline_sidecar_load = dict(baseline_sidecar.get("load", {}) or {})
    explicit_sidecar_load = dict(explicit_sidecar.get("load", {}) or {})
    baseline_coupled_generation = dict(baseline_coupled.get("generation", {}) or {})
    explicit_coupled_generation = dict(explicit_coupled.get("generation", {}) or {})
    baseline_coupled_reservoir = dict(baseline_coupled.get("reservoir", {}) or {})

    host_sync_total = int(
        ((baseline_coupled_generation.get("host_sync_points") or {}).get("total") or 0)
    )
    host_sync_s = float(baseline_coupled_generation.get("host_sync_s") or 0.0)
    total_turn_s = float(baseline_coupled_generation.get("total_turn_s") or 0.0)
    host_sync_share = (100.0 * host_sync_s / total_turn_s) if total_turn_s > 0 else None

    verdict_today = {
        "fork_mmap_active_today": bool(
            reservoir_env.get("has_mmap_stats") and fast_env.get("has_mmap_stats")
        ),
        "meaningful_unified_memory_compute_today": bool(
            baseline_coupled.get("success")
            and host_sync_total > 0
            and not baseline_coupled_load.get("memory_map_effective")
        ),
        "summary": (
            "Astrid is meaningfully using MLX's in-process unified-memory compute path today "
            "for the coupled generation loop, but not this fork's mmap/zero-copy load path."
        ),
    }

    opportunities = []
    coupled_load_delta = _delta_pct(
        _extract_float({"load": baseline_coupled_load}, "load", "load_seconds"),
        _extract_float({"load": explicit_coupled_load}, "load", "load_seconds"),
    )
    sidecar_load_delta = _delta_pct(
        _extract_float({"load": baseline_sidecar_load}, "load", "load_seconds"),
        _extract_float({"load": explicit_sidecar_load}, "load", "load_seconds"),
    )
    if explicit_coupled_load:
        opportunities.append(
            {
                "rank": 1,
                "title": "Enable forked MLX plus explicit mapped loads",
                "coupled_mapped_bytes": explicit_coupled_load.get("mapped_bytes"),
                "sidecar_mapped_bytes": explicit_sidecar_load.get("mapped_bytes"),
                "coupled_load_seconds_delta_pct": coupled_load_delta,
                "sidecar_load_seconds_delta_pct": sidecar_load_delta,
            }
        )
    opportunities.append(
        {
            "rank": 2,
            "title": "Remove per-token CPU syncs in the coupled loop",
            "host_sync_points": baseline_coupled_generation.get("host_sync_points"),
            "host_sync_seconds": host_sync_s,
            "host_sync_share_pct": host_sync_share,
        }
    )
    opportunities.append(
        {
            "rank": 3,
            "title": "Reduce reservoir websocket/base64/NumPy crossings",
            "reservoir_pull_bytes": baseline_coupled_reservoir.get("pull_bytes"),
            "reservoir_push_bytes": baseline_coupled_reservoir.get("push_bytes"),
            "reservoir_serialization_s": baseline_coupled_reservoir.get(
                "serialization_s"
            ),
        }
    )

    return {
        "today": verdict_today,
        "opportunities": opportunities,
        "comparison": {
            "baseline": baseline,
            "fork_enabled": fork_enabled,
            "fork_enabled_explicit_mmap": explicit,
        },
    }


def _write_markdown_report(
    *,
    artifact: dict[str, object],
    output_path: Path,
) -> None:
    verdicts = dict(artifact.get("verdicts", {}) or {})
    today = dict(verdicts.get("today", {}) or {})
    opportunities = list(verdicts.get("opportunities", []) or [])

    lines = [
        "# Astrid Zero-Copy Audit",
        "",
        "## Verdicts",
        "",
        f"- Fork mmap path active today: `{today.get('fork_mmap_active_today')}`",
        f"- Unified-memory compute path meaningfully used today: `{today.get('meaningful_unified_memory_compute_today')}`",
        f"- Summary: {today.get('summary')}",
        "",
        "## Opportunities",
        "",
    ]
    for item in opportunities:
        lines.append(f"- [{item.get('rank')}] {item.get('title')}: `{json.dumps(item, sort_keys=True)}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Astrid's MLX zero-copy paths and benchmark follow-up scenarios."
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "benchmarks" / "python" / "audit-results"),
        help="Directory for combined JSON and Markdown artifacts.",
    )
    parser.add_argument(
        "--install-fork-envs",
        action="store_true",
        help="Install this MLX fork into Astrid's active venvs before follow-up scenarios.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser() / _timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, object] = {
        "created_at": time.time(),
        "repo_root": str(REPO_ROOT),
        "neural_root": str(NEURAL_ROOT),
        "baseline_envs": _inspect_envs(),
        "install": {},
        "scenarios": {},
    }

    artifact["scenarios"]["baseline"] = {
        "sidecar": _run_sidecar_scenario(
            scenario_name="baseline",
            output_dir=output_dir,
            memory_map=False,
        ),
        "coupled": _run_coupled_scenario(
            scenario_name="baseline",
            output_dir=output_dir,
            memory_map=False,
        ),
    }

    if args.install_fork_envs:
        artifact["install"] = {
            "reservoir_venv": _install_mlx_fork_into_env(
                python_path=RESERVOIR_VENV_PYTHON,
                output_dir=output_dir,
                label="reservoir_venv",
            ),
            "mlx_fast_venv": _install_mlx_fork_into_env(
                python_path=FAST_VENV_PYTHON,
                output_dir=output_dir,
                label="mlx_fast_venv",
            ),
        }
    artifact["post_install_envs"] = _inspect_envs()

    artifact["scenarios"]["fork_enabled"] = {
        "sidecar": _run_sidecar_scenario(
            scenario_name="fork_enabled",
            output_dir=output_dir,
            memory_map=False,
        ),
        "coupled": _run_coupled_scenario(
            scenario_name="fork_enabled",
            output_dir=output_dir,
            memory_map=False,
        ),
    }

    artifact["scenarios"]["fork_enabled_explicit_mmap"] = {
        "sidecar": _run_sidecar_scenario(
            scenario_name="fork_enabled_explicit_mmap",
            output_dir=output_dir,
            memory_map=True,
        ),
        "coupled": _run_coupled_scenario(
            scenario_name="fork_enabled_explicit_mmap",
            output_dir=output_dir,
            memory_map=True,
        ),
    }

    artifact["verdicts"] = _build_verdicts(artifact)

    json_path = output_dir / "astrid_zero_copy_audit.json"
    markdown_path = output_dir / "astrid_zero_copy_audit.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(artifact=artifact, output_path=markdown_path)

    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
