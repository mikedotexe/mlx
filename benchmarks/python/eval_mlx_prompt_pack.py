#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_model_path(repo_root: Path) -> str | None:
    candidate = (
        repo_root
        / ".local_models"
        / "tinyllama-1.1b-chat-mlx-4bit"
        / "model.safetensors"
    )
    return str(candidate) if candidate.exists() else None


def _default_model_specs(repo_root: Path) -> list[dict[str, str]]:
    candidates = [
        {
            "label": "tinyllama_1.1b_chat_4bit",
            "path": (
                repo_root
                / ".local_models"
                / "tinyllama-1.1b-chat-mlx-4bit"
                / "model.safetensors"
            ),
        },
        {
            "label": "qwen2.5_1.5b_instruct_4bit",
            "path": (
                repo_root
                / ".local_models"
                / "qwen2.5-1.5b-instruct-mlx-4bit"
                / "model.safetensors"
            ),
        },
    ]
    return [
        {"label": spec["label"], "path": str(spec["path"])}
        for spec in candidates
        if Path(spec["path"]).exists()
    ]


def _default_pack_path() -> Path:
    return Path(__file__).resolve().parent / "testdata" / "mlx_prompt_eval_pack.json"


def _normalize_text(text: str) -> str:
    return str(text or "").strip()


def _word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", _normalize_text(text)) if token])


def _bullet_lines(text: str) -> list[str]:
    lines = []
    for line in _normalize_text(text).splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*", "\u2022")):
            lines.append(stripped[1:].strip())
    return lines


def _evaluate_expectation(output_text: str, expectation: dict) -> tuple[bool, list[dict]]:
    normalized = _normalize_text(output_text)
    normalized_lower = normalized.lower()
    checks: list[dict] = []

    if "equals" in expectation:
        expected = _normalize_text(str(expectation["equals"]))
        checks.append(
            {
                "name": "equals",
                "passed": normalized == expected,
                "detail": f"expected={expected!r} actual={normalized!r}",
            }
        )

    if "matches_regex" in expectation:
        pattern = str(expectation["matches_regex"])
        checks.append(
            {
                "name": "matches_regex",
                "passed": re.fullmatch(pattern, normalized) is not None,
                "detail": f"pattern={pattern!r} actual={normalized!r}",
            }
        )

    if "contains_any" in expectation:
        options = [str(item) for item in expectation["contains_any"]]
        passed = any(option.lower() in normalized_lower for option in options)
        checks.append(
            {
                "name": "contains_any",
                "passed": passed,
                "detail": f"options={options!r} actual={normalized!r}",
            }
        )

    if "contains_all" in expectation:
        required = [str(item) for item in expectation["contains_all"]]
        passed = all(option.lower() in normalized_lower for option in required)
        checks.append(
            {
                "name": "contains_all",
                "passed": passed,
                "detail": f"required={required!r} actual={normalized!r}",
            }
        )

    if "max_words" in expectation:
        max_words = int(expectation["max_words"])
        actual_words = _word_count(normalized)
        checks.append(
            {
                "name": "max_words",
                "passed": actual_words <= max_words,
                "detail": f"max={max_words} actual={actual_words}",
            }
        )

    if "word_count" in expectation:
        expected_words = int(expectation["word_count"])
        actual_words = _word_count(normalized)
        checks.append(
            {
                "name": "word_count",
                "passed": actual_words == expected_words,
                "detail": f"expected={expected_words} actual={actual_words}",
            }
        )

    if "bullet_count" in expectation:
        expected_bullets = int(expectation["bullet_count"])
        bullets = _bullet_lines(normalized)
        checks.append(
            {
                "name": "bullet_count",
                "passed": len(bullets) == expected_bullets,
                "detail": f"expected={expected_bullets} actual={len(bullets)}",
            }
        )

    if "max_words_per_bullet" in expectation:
        limit = int(expectation["max_words_per_bullet"])
        bullets = _bullet_lines(normalized)
        longest = max((_word_count(line) for line in bullets), default=0)
        checks.append(
            {
                "name": "max_words_per_bullet",
                "passed": bool(bullets) and longest <= limit,
                "detail": f"max={limit} actual={longest}",
            }
        )

    if "json_equals" in expectation:
        expected_json = expectation["json_equals"]
        actual_json = None
        parse_error = None
        try:
            actual_json = json.loads(normalized)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
        checks.append(
            {
                "name": "json_equals",
                "passed": actual_json == expected_json,
                "detail": (
                    f"expected={expected_json!r} actual={actual_json!r}"
                    if parse_error is None
                    else f"json parse failed: {parse_error}"
                ),
            }
        )

    overall = all(check["passed"] for check in checks) if checks else False
    return overall, checks


def _load_prompt_pack(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Prompt pack must be a JSON array: {path}")
    return payload


def _parse_model_spec(raw: str) -> dict[str, str]:
    label = None
    path = raw
    if "=" in raw:
        maybe_label, maybe_path = raw.split("=", 1)
        if maybe_label and maybe_path:
            label = maybe_label.strip()
            path = maybe_path.strip()
    resolved = str(Path(path).expanduser().resolve())
    if label is None:
        label = Path(resolved).resolve().parent.name.replace("-", "_")
    return {"label": label, "path": resolved}


def _resolve_models(args: argparse.Namespace, repo_root: Path) -> list[dict[str, str]]:
    if args.compare_default_models:
        models = _default_model_specs(repo_root)
        if not models:
            raise FileNotFoundError("No default local models were found under .local_models/")
        return models

    models: list[dict[str, str]] = []
    model_value = args.model or args.default_model
    if model_value:
        models.append(_parse_model_spec(model_value))
    for raw in args.compare_model or []:
        models.append(_parse_model_spec(raw))
    return models


def _filter_cases(cases: list[dict[str, Any]], *, case_ids: set[str], tags: set[str]) -> list[dict]:
    filtered = cases
    if case_ids:
        filtered = [case for case in filtered if case["id"] in case_ids]
    if tags:
        filtered = [
            case
            for case in filtered
            if tags.intersection(set(str(tag) for tag in case.get("tags", [])))
        ]
    return filtered


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for result in results:
        for tag in result.get("tags", []):
            bucket = summary.setdefault(
                str(tag),
                {"pass_count": 0, "total_cases": 0},
            )
            bucket["total_cases"] += 1
            bucket["pass_count"] += int(result.get("passed", False))
    for bucket in summary.values():
        total = int(bucket["total_cases"])
        bucket["pass_rate"] = (float(bucket["pass_count"]) / total) if total else 0.0
    return summary


def _run_prompt_case(
    *,
    decode_python: str,
    decode_helper: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    system_prompt: str | None,
) -> dict:
    cmd = [
        decode_python,
        decode_helper,
        "--file",
        model_path,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max(max_tokens, 1)),
        "--json",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        cmd,
        text=True,
        env=env,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "prompt case failed:\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    default_model = _default_model_path(repo_root)
    parser = argparse.ArgumentParser(
        description=(
            "Run a small fixed prompt-eval pack against a local MLX model using "
            "the fast mlx_lm venv path."
        )
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="MLX model file or directory. Defaults to the local TinyLlama model if present.",
    )
    parser.add_argument(
        "--pack",
        default=str(_default_pack_path()),
        help="JSON prompt pack path.",
    )
    parser.add_argument(
        "--decode-python",
        default=str(repo_root / ".venv-mlxlm" / "bin" / "python"),
        help="Python executable used for the fast mlx_lm decode path.",
    )
    parser.add_argument(
        "--decode-helper",
        default=str(Path(__file__).with_name("decode_guard_mlx_lm.py")),
        help="Path to the decode helper script.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case id to run. Repeat to keep multiple specific cases.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Case tag to run, such as general, primes, or esn. Repeat to keep multiple tags.",
    )
    parser.add_argument(
        "--compare-model",
        action="append",
        default=[],
        help=(
            "Additional model to compare. Accepts PATH or LABEL=PATH. "
            "Repeat to compare multiple models."
        ),
    )
    parser.add_argument(
        "--compare-default-models",
        action="store_true",
        help="Run against all detected default local models under .local_models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N selected cases.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the prompt pack cases and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List detected default local models and exit.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON results path.",
    )
    args = parser.parse_args()
    args.default_model = default_model
    if args.compare_default_models and (args.model is not None or args.compare_model):
        parser.error(
            "--compare-default-models cannot be combined with an explicit model "
            "or --compare-model."
        )
    if args.model is None and not args.list and not args.list_models and not args.compare_default_models:
        parser.error(
            "No model path was provided and the default local TinyLlama model was not found."
        )
    return args


def main() -> int:
    repo_root = _repo_root()
    args = _parse_args()
    pack_path = Path(args.pack).expanduser().resolve()
    cases = _load_prompt_pack(pack_path)
    if args.list_models:
        models = _default_model_specs(repo_root)
        if not models:
            print("No default local models detected.")
            return 0
        print("Detected default models:")
        for model in models:
            print(f"- {model['label']}: {model['path']}")
        return 0

    selected_ids = set(args.case or [])
    selected_tags = set(args.tag or [])
    cases = _filter_cases(cases, case_ids=selected_ids, tags=selected_tags)
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]

    if args.list:
        print(f"Prompt pack: {pack_path}")
        for case in cases:
            tags = ",".join(case.get("tags", []))
            tags_suffix = f" [{tags}]" if tags else ""
            print(f"- {case['id']}: {case['title']}{tags_suffix}")
        return 0

    decode_python = os.path.abspath(os.path.expanduser(args.decode_python))
    decode_helper = str(Path(args.decode_helper).expanduser().resolve())
    models = _resolve_models(args, repo_root)

    missing = [
        path
        for path in [pack_path, Path(decode_python), Path(decode_helper)]
        if not Path(path).exists()
    ]
    missing.extend(
        Path(model["path"])
        for model in models
        if not Path(model["path"]).exists()
    )
    if missing:
        raise FileNotFoundError(
            "Missing required path(s): " + ", ".join(str(path) for path in missing)
        )

    print(f"Prompt pack: {pack_path}")
    print(f"Decode python: {decode_python}")
    model_summaries = []
    for model in models:
        model_path = str(Path(model["path"]).expanduser().resolve())
        results = []
        passed = 0
        print(f"Model [{model['label']}]: {model_path}")
        for case in cases:
            payload = _run_prompt_case(
                decode_python=decode_python,
                decode_helper=decode_helper,
                model_path=model_path,
                prompt=case["prompt"],
                max_tokens=int(case.get("max_tokens", 32)),
                system_prompt=case.get("system_prompt"),
            )
            case_passed, checks = _evaluate_expectation(
                payload.get("text", ""), case.get("expect", {})
            )
            passed += int(case_passed)
            result = {
                "id": case["id"],
                "title": case["title"],
                "tags": list(case.get("tags", [])),
                "prompt": case["prompt"],
                "passed": case_passed,
                "checks": checks,
                "output_text": payload.get("text", ""),
                "metrics": {
                    "load_seconds": payload.get("load_seconds"),
                    "first_token_seconds": payload.get("first_token_seconds"),
                    "generate_seconds": payload.get("generate_seconds"),
                    "generated_tokens": payload.get("generated_tokens"),
                    "tok_per_second": payload.get("tok_per_second"),
                },
            }
            results.append(result)
            status = "pass" if case_passed else "fail"
            tok_s = payload.get("tok_per_second")
            first_token = payload.get("first_token_seconds")
            first_token_text = (
                f"{float(first_token):.3f}s"
                if isinstance(first_token, (int, float))
                else "n/a"
            )
            tok_s_text = (
                f"{float(tok_s):.2f}" if isinstance(tok_s, (int, float)) else "n/a"
            )
            print(
                f"[{status}] {case['id']}: "
                f"first_token={first_token_text} tok/s={tok_s_text}"
            )
            print(f"  output: {_normalize_text(payload.get('text', ''))!r}")
            for check in checks:
                marker = "ok" if check["passed"] else "x"
                print(f"  {marker} {check['name']}: {check['detail']}")

        model_summary = {
            "label": model["label"],
            "model": model_path,
            "pass_count": passed,
            "total_cases": len(results),
            "pass_rate": (passed / len(results)) if results else 0.0,
            "tag_summary": _summarize_results(results),
            "results": results,
        }
        model_summaries.append(model_summary)
        print(
            "Summary"
            f" [{model['label']}]: "
            f"{model_summary['pass_count']}/{model_summary['total_cases']} "
            f"({model_summary['pass_rate']:.0%})"
        )
        if model_summary["tag_summary"]:
            for tag, tag_metrics in sorted(model_summary["tag_summary"].items()):
                print(
                    f"  tag {tag}: "
                    f"{int(tag_metrics['pass_count'])}/{int(tag_metrics['total_cases'])} "
                    f"({float(tag_metrics['pass_rate']):.0%})"
                )

    summary = {
        "pack": str(pack_path),
        "models": model_summaries,
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON results: {output_path}")

    return 0 if all(
        model_summary["pass_count"] == model_summary["total_cases"]
        for model_summary in model_summaries
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
