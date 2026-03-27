#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_executable_path(raw: str) -> str:
    return os.path.abspath(os.path.expanduser(raw))


def _default_model_path(repo_root: Path) -> str | None:
    candidate = (
        repo_root
        / ".local_models"
        / "tinyllama-1.1b-chat-mlx-4bit"
        / "model.safetensors"
    )
    return str(candidate) if candidate.exists() else None


def _quote_token(token: str) -> str:
    if token.startswith("{") and token.endswith("}"):
        return token
    return shlex.quote(token)


def _shell_join(tokens: list[str]) -> str:
    return " ".join(_quote_token(token) for token in tokens)


def _build_decode_command_template(
    *,
    decode_python: str,
    decode_helper: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None,
    trust_remote_code: bool,
    ignore_chat_template: bool,
    print_text: bool,
) -> str:
    tokens = [
        "/usr/bin/env",
        "PYTHONPATH=",
        decode_python,
        decode_helper,
        "--file",
        "{file}",
        "--prompt",
        prompt,
        "--max-tokens",
        str(max(max_tokens, 1)),
        "--temp",
        str(temperature),
        "--memory-map",
        "{memory_map}",
        "--format",
        "{format}",
    ]
    if system_prompt:
        tokens.extend(["--system-prompt", system_prompt])
    if trust_remote_code:
        tokens.append("--trust-remote-code")
    if ignore_chat_template:
        tokens.append("--ignore-chat-template")
    if print_text:
        tokens.append("--print-text")
    return _shell_join(tokens)


def _build_benchmark_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    repo_python = str(repo_root / "python")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repo_python
        if not current_pythonpath
        else repo_python + os.pathsep + current_pythonpath
    )
    return env


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    default_model = _default_model_path(repo_root)
    parser = argparse.ArgumentParser(
        description=(
            "Run the proven hybrid mmap benchmark path: repo MLX runtime for "
            "load_mmap_bench.py plus the fast mlx_lm venv for real prompt/decode signal."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=default_model,
        help=(
            "Model file to benchmark. Defaults to the local TinyLlama MLX model "
            "if present."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in five words.",
        help="Prompt passed to the decode guard (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for chat-template aware models.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Maximum generated tokens for the decode guard (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the decode guard (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        default="auto",
        help="Format forwarded to load_mmap_bench.py (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["inherit", "warm", "cold-best-effort", "steady-state"],
        default="warm",
        help="Worker cache mode (default: %(default)s).",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument(
        "--decode-runs",
        type=int,
        default=1,
        help="Decode-guard trial count (default: %(default)s).",
    )
    parser.add_argument(
        "--history-json",
        default="/tmp/hybrid_mlx_prompt_history.jsonl",
        help="History JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--benchmark-python",
        default=sys.executable,
        help="Python executable used to run load_mmap_bench.py (default: current interpreter).",
    )
    parser.add_argument(
        "--decode-python",
        default=str(repo_root / ".venv-mlxlm" / "bin" / "python"),
        help="Python executable used for the real prompt decode guard.",
    )
    parser.add_argument(
        "--decode-helper",
        default=str(Path(__file__).with_name("decode_guard_mlx_lm.py")),
        help="Path to the mlx_lm decode helper script.",
    )
    parser.add_argument(
        "--coverage-probe",
        action="store_true",
        help="Keep the mmap coverage probe enabled. Default is off for quicker prompt runs.",
    )
    parser.add_argument(
        "--print-text",
        action="store_true",
        help="Stream generated text from the decode guard to stderr.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to the decode guard.",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Pass --ignore-chat-template to the decode guard.",
    )
    parser.add_argument(
        "--decode-phase",
        choices=["soft", "hard"],
        default="soft",
        help="Decode gate phase for the benchmark (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command/env instead of executing it.",
    )
    parser.add_argument(
        "extra_benchmark_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to load_mmap_bench.py after '--'.",
    )
    args = parser.parse_args()
    if args.model is None:
        parser.error(
            "No model path was provided and the default local TinyLlama model was not found."
        )
    return args


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    benchmark_path = Path(__file__).with_name("load_mmap_bench.py")
    benchmark_python = _normalize_executable_path(args.benchmark_python)
    decode_python = _normalize_executable_path(args.decode_python)
    decode_helper = str(Path(args.decode_helper).expanduser().resolve())
    model_path = str(Path(args.model).expanduser().resolve())

    missing = [
        path
        for path in [decode_python, decode_helper, str(benchmark_path), model_path]
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))

    decode_cmd = _build_decode_command_template(
        decode_python=decode_python,
        decode_helper=decode_helper,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        trust_remote_code=args.trust_remote_code,
        ignore_chat_template=args.ignore_chat_template,
        print_text=args.print_text,
    )

    cmd = [
        benchmark_python,
        str(benchmark_path),
        model_path,
        "--format",
        args.format,
        "--runs",
        str(max(args.runs, 1)),
        "--warmup-runs",
        str(max(args.warmup_runs, 0)),
        "--attempts",
        str(max(args.attempts, 1)),
        "--cache-mode",
        args.cache_mode,
        "--history-json",
        args.history_json,
        "--decode-cmd",
        decode_cmd,
        "--decode-runs",
        str(max(args.decode_runs, 1)),
        "--decode-phase",
        args.decode_phase,
    ]
    if not args.coverage_probe:
        cmd.append("--no-coverage-probe")

    extra_args = list(args.extra_benchmark_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    env = _build_benchmark_env(repo_root)

    print("Hybrid benchmark launcher", flush=True)
    print(f"  benchmark python: {cmd[0]}", flush=True)
    print(f"  benchmark mlx path: {repo_root / 'python'}", flush=True)
    print(f"  decode python: {decode_python}", flush=True)
    print(f"  model: {model_path}", flush=True)
    print(f"  history: {args.history_json}", flush=True)
    print(f"  decode cmd: {decode_cmd}", flush=True)
    print(f"  command: {_shell_join(cmd)}", flush=True)

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
