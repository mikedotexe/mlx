#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_EXTENSIONS = {".gguf", ".safetensors"}
EXCLUDE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
EXCLUDE_NAME_SUBSTRINGS = {
    "vocab",
    "tokenizer",
    "tokenizers",
    "merges",
    "sentencepiece",
    "special_tokens",
    "ggml-vocab",
}
SAFETENSORS_SHARD_PATTERN = re.compile(r"-\d{5}-of-\d{5}\.safetensors$")


@dataclass
class Candidate:
    path: str
    size_bytes: int
    extension: str
    parent_dir: str
    format: str | None = None
    source: str = "filesystem"
    model_id: str | None = None

    @property
    def size_mib(self) -> float:
        return self.size_bytes / (1024.0 * 1024.0)


def _iter_files(roots: list[Path]):
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue

        for path in root.rglob("*"):
            if path.is_dir() and path.name in EXCLUDE_DIR_NAMES:
                continue
            if path.is_file():
                if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                    continue
                yield path


def _is_candidate(path: Path, min_size_bytes: int) -> bool:
    if path.suffix.lower() not in MODEL_EXTENSIONS:
        return False
    name_l = path.name.lower()
    if any(token in name_l for token in EXCLUDE_NAME_SUBSTRINGS):
        return False
    if path.suffix.lower() == ".safetensors" and SAFETENSORS_SHARD_PATTERN.search(name_l):
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < min_size_bytes:
        return False
    return True


def _iter_ollama_candidates(min_size_bytes: int):
    manifest_root = (
        Path.home()
        / ".ollama"
        / "models"
        / "manifests"
        / "registry.ollama.ai"
        / "library"
    )
    blob_root = Path.home() / ".ollama" / "models" / "blobs"
    if not manifest_root.exists() or not blob_root.exists():
        return

    for manifest in manifest_root.glob("*/*"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        model_name = f"{manifest.parent.name}:{manifest.name}"
        for layer in payload.get("layers", []):
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue
            digest = str(layer.get("digest", ""))
            if not digest.startswith("sha256:"):
                continue
            blob = blob_root / f"sha256-{digest.split(':', maxsplit=1)[1]}"
            try:
                size = blob.stat().st_size
            except OSError:
                continue
            if size < min_size_bytes:
                continue
            try:
                with blob.open("rb") as f:
                    magic = f.read(4)
            except OSError:
                continue
            if magic != b"GGUF":
                continue

            yield Candidate(
                path=str(blob.resolve()),
                size_bytes=size,
                extension=".gguf",
                parent_dir=str(manifest.parent.resolve()),
                format="gguf",
                source="ollama",
                model_id=model_name,
            )


def find_candidates(
    roots: list[Path],
    min_size_bytes: int,
    one_per_dir: bool,
    include_ollama_blobs: bool = False,
    model_id_regex: str | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in _iter_files(roots):
        if not _is_candidate(path, min_size_bytes):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        candidates.append(
            Candidate(
                path=str(path.resolve()),
                size_bytes=size,
                extension=path.suffix.lower(),
                parent_dir=str(path.parent.resolve()),
                format=path.suffix.lower().lstrip("."),
            )
        )

    if include_ollama_blobs:
        candidates.extend(_iter_ollama_candidates(min_size_bytes))

    if model_id_regex:
        pattern = re.compile(model_id_regex)
        candidates = [
            c for c in candidates if c.model_id is not None and pattern.search(c.model_id)
        ]

    if one_per_dir:
        best_by_dir: dict[str, Candidate] = {}
        for c in candidates:
            existing = best_by_dir.get(c.parent_dir)
            if existing is None or c.size_bytes > existing.size_bytes:
                best_by_dir[c.parent_dir] = c
        candidates = list(best_by_dir.values())

    candidates.sort(key=lambda c: c.size_bytes, reverse=True)
    return candidates


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Find benchmark-worthy local model checkpoints "
            "(filters out vocab/tokenizer artifacts and tiny files)."
        )
    )
    p.add_argument(
        "roots",
        nargs="*",
        default=["."],
        help="Directories/files to scan recursively. Default: current directory.",
    )
    p.add_argument(
        "--min-size-mib",
        type=float,
        default=256.0,
        help="Minimum file size in MiB (default: 256).",
    )
    p.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of candidates to print (default: 20).",
    )
    p.add_argument(
        "--all-per-dir",
        action="store_true",
        help="Do not collapse to the largest candidate per directory.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text output.",
    )
    p.add_argument(
        "--paths-only",
        action="store_true",
        help="Emit only absolute candidate paths (one per line).",
    )
    p.add_argument(
        "--include-ollama-blobs",
        action="store_true",
        help=(
            "Also inspect ~/.ollama manifests and include GGUF model blobs "
            "as benchmark candidates."
        ),
    )
    p.add_argument(
        "--model-id-regex",
        default=None,
        help=(
            "Regex filter applied to discovered model IDs. "
            "Primarily useful with --include-ollama-blobs."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    roots = [Path(r).expanduser().resolve() for r in args.roots]
    min_size_bytes = int(args.min_size_mib * 1024 * 1024)

    candidates = find_candidates(
        roots=roots,
        min_size_bytes=min_size_bytes,
        one_per_dir=not args.all_per_dir,
        include_ollama_blobs=args.include_ollama_blobs,
        model_id_regex=args.model_id_regex,
    )
    candidates = candidates[: max(args.max_results, 0)]

    if args.paths_only:
        for c in candidates:
            print(c.path)
        return 0

    if args.json:
        payload = [
            {
                **asdict(c),
                "size_mib": round(c.size_mib, 2),
            }
            for c in candidates
        ]
        print(json.dumps(payload, indent=2))
        return 0

    if not candidates:
        print("No benchmark-worthy model files found.")
        return 0

    print("Curated benchmark candidates:")
    for i, c in enumerate(candidates, start=1):
        suffix = f"  [{c.model_id}]" if c.model_id else ""
        print(f"{i:2d}. {c.size_mib:9.2f} MiB  {c.path}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
