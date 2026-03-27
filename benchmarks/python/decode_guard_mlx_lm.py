#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import load, sample_utils, stream_generate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real prompt/decode tok/s guard using mlx_lm against a local MLX model "
            "directory or an MLX model file inside that directory."
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to an MLX model directory or a file inside it.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for chat-template aware models.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens (default: 64).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature. Default 0.0 keeps decoding deterministic.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code in tokenizer config.",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt instead of tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--memory-map",
        default=None,
        help="Unused placeholder for benchmark template compatibility.",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Unused placeholder for benchmark template compatibility.",
    )
    parser.add_argument(
        "--print-text",
        action="store_true",
        help="Emit generated text to stderr while still printing tok/s to stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of a bare tok/s float.",
    )
    return parser.parse_args()


def _resolve_model_dir(file_arg: str) -> Path:
    path = Path(file_arg).expanduser().resolve()
    return path if path.is_dir() else path.parent


def _build_prompt(
    tokenizer,
    *,
    user_prompt: str,
    system_prompt: str | None,
    ignore_chat_template: bool,
) -> str:
    if ignore_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        if system_prompt:
            return f"{system_prompt}\n\n{user_prompt}"
        return user_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def main() -> int:
    args = _parse_args()
    model_dir = _resolve_model_dir(args.file)
    tokenizer_config = {"trust_remote_code": True} if args.trust_remote_code else None

    load_start = time.time()
    if tokenizer_config is not None:
        model, tokenizer = load(str(model_dir), tokenizer_config=tokenizer_config)
    else:
        model, tokenizer = load(str(model_dir))
    load_seconds = time.time() - load_start

    prompt = _build_prompt(
        tokenizer,
        user_prompt=args.prompt,
        system_prompt=args.system_prompt,
        ignore_chat_template=args.ignore_chat_template,
    )

    sampler = sample_utils.make_sampler(temp=args.temp)
    generate_start = time.time()
    first_token_seconds = None
    token_count = 0
    text_chunks = []

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max(args.max_tokens, 1),
        sampler=sampler,
    ):
        now = time.time()
        if first_token_seconds is None:
            first_token_seconds = now - generate_start
        token_count += 1
        if response.text:
            text_chunks.append(response.text)
            if args.print_text:
                print(response.text, end="", file=sys.stderr, flush=True)

    generate_seconds = time.time() - generate_start
    if args.print_text and text_chunks:
        print("", file=sys.stderr)

    tok_per_second = (
        float(token_count) / generate_seconds if generate_seconds > 0.0 else 0.0
    )
    payload = {
        "model_dir": str(model_dir),
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "first_token_seconds": first_token_seconds,
        "generated_tokens": token_count,
        "tok_per_second": tok_per_second,
        "prompt_chars": len(prompt),
        "text": "".join(text_chunks),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{tok_per_second:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
