#!/usr/bin/env python3
"""
Scaffold external prose/code into a reviewable hypothesis ledger.

This is intentionally lightweight. It does not try to prove or summarize the
source corpus perfectly. It extracts candidate claims, hypotheses, and
verification entrypoints so they can be curated into replay/demo artifacts.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


_HYPOTHESIS_RE = re.compile(r"\b[hH]ypothesis\b[: ]")
_VERIFIED_RE = re.compile(r"\b[Vv]erified\b|\b[Pp]roved\b|\b[Rr]efuted\b")
_ENTRYPOINT_RE = re.compile(r"`([^`]+)`")
_CLAIM_RE = re.compile(r"\b[Cc]laim\b[: ]|\b[Cc]onjecture\b[: ]")
_SCOPE_RE = re.compile(
    r"\bnot core math\b|\bnot a number-theoretic result\b|\bnot .* proof\b|"
    r"\bopen generalization\b|\bopen mechanism\b|\bnot .* theorem\b",
    re.IGNORECASE,
)
_RUST_ITEM_RE = re.compile(
    r"^\s*(?:pub\s+)?(?P<kind>const|struct|enum|trait|type|fn)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
_LEAN_DECL_RE = re.compile(
    r"^\s*(?P<kind>theorem|lemma|def|example|structure|class|inductive)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_']*)"
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _unique_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        ordered.append(tag)
    return ordered


def _classify_candidate_tags(text: str, *, extra_tags: list[str] | None = None) -> list[str]:
    tags = list(extra_tags or [])
    if _HYPOTHESIS_RE.search(text) or _CLAIM_RE.search(text):
        tags.append("hypothesis")
    if _VERIFIED_RE.search(text) or re.search(r"\bimplemented\b", text, re.IGNORECASE):
        tags.append("verification")
    if _SCOPE_RE.search(text):
        tags.append("scope")
    return _unique_tags(tags)


def _extract_markdown_candidates(path: Path, text: str) -> list[dict]:
    entries: list[dict] = []
    heading = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            continue
        if line.startswith("- ") or line.startswith("* "):
            body = line[2:].strip()
            entrypoints = _ENTRYPOINT_RE.findall(body)
            entries.append(
                {
                    "kind": "bullet",
                    "heading": heading,
                    "text": body,
                    "tags": _classify_candidate_tags(body),
                    "entrypoints": entrypoints,
                }
            )
        elif "|" in line and heading:
            if line.startswith("|") and not set(line.replace("|", "").strip()) <= {"-", ":"}:
                entries.append(
                    {
                        "kind": "table_row",
                        "heading": heading,
                        "text": line,
                        "tags": ["table"],
                        "entrypoints": _ENTRYPOINT_RE.findall(line),
                    }
                )
    return entries


def _extract_python_candidates(path: Path, text: str) -> list[dict]:
    entries: list[dict] = []
    try:
        module = ast.parse(text, filename=str(path))
    except SyntaxError:
        return entries

    module_doc = ast.get_docstring(module)
    if module_doc:
        for line in module_doc.splitlines():
            item = line.strip()
            if not item:
                continue
            if _HYPOTHESIS_RE.search(item):
                entries.append(
                    {
                        "kind": "module_doc",
                        "heading": "module docstring",
                        "text": item,
                        "tags": _classify_candidate_tags(item),
                        "entrypoints": [],
                    }
                )

    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            tags: list[str] = []
            name = node.name
            text_items = [name]
            doc = ast.get_docstring(node)
            if doc:
                text_items.extend(line.strip() for line in doc.splitlines() if line.strip())
            merged = " | ".join(text_items)
            if name.startswith("test_"):
                tags.append("test")
            tags = _classify_candidate_tags(merged, extra_tags=tags)
            if tags:
                entries.append(
                    {
                        "kind": "function",
                        "heading": name,
                        "text": merged,
                        "tags": tags,
                        "entrypoints": [],
                    }
                )
    return entries


def _extract_rust_candidates(path: Path, text: str) -> list[dict]:
    entries: list[dict] = []
    lines = text.splitlines()
    i = 0
    pending_test = False

    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("#[test]"):
            pending_test = True
            i += 1
            continue

        if stripped.startswith("//!") or stripped.startswith("///"):
            prefix = stripped[:3]
            doc_lines: list[str] = []
            while i < len(lines):
                current = lines[i].lstrip()
                if not current.startswith(prefix):
                    break
                doc_lines.append(current[3:].strip())
                i += 1
            doc_text = " ".join(part for part in doc_lines if part).strip()
            entrypoints = _ENTRYPOINT_RE.findall(doc_text)
            if prefix == "//!":
                entries.append(
                    {
                        "kind": "module_doc",
                        "heading": "module docs",
                        "text": doc_text,
                        "tags": _classify_candidate_tags(doc_text, extra_tags=["module"]),
                        "entrypoints": entrypoints,
                    }
                )
                pending_test = False
                continue

            j = i
            local_test = pending_test
            while j < len(lines):
                probe = lines[j].strip()
                if not probe:
                    j += 1
                    continue
                if probe.startswith("#["):
                    if probe == "#[test]":
                        local_test = True
                    j += 1
                    continue
                break
            item_match = _RUST_ITEM_RE.match(lines[j]) if j < len(lines) else None
            heading = "doc block"
            kind = "item_doc"
            if item_match is not None:
                heading = item_match.group("name")
                kind = item_match.group("kind")
                entrypoints = [heading, *entrypoints]
                i = j + 1
            tags = _classify_candidate_tags(
                doc_text,
                extra_tags=["test"] if local_test else None,
            )
            entries.append(
                {
                    "kind": kind,
                    "heading": heading,
                    "text": doc_text,
                    "tags": tags,
                    "entrypoints": _unique_tags(entrypoints),
                }
            )
            pending_test = False
            continue

        if pending_test:
            item_match = _RUST_ITEM_RE.match(lines[i])
            if item_match is not None and item_match.group("kind") == "fn":
                name = item_match.group("name")
                entries.append(
                    {
                        "kind": "test",
                        "heading": name,
                        "text": name,
                        "tags": ["test"],
                        "entrypoints": [name],
                    }
                )
                pending_test = False
                i += 1
                continue
            if stripped and not stripped.startswith("#["):
                pending_test = False

        i += 1
    return entries


def _extract_lean_candidates(path: Path, text: str) -> list[dict]:
    entries: list[dict] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("/--"):
            doc_parts: list[str] = []
            current = stripped[3:].strip()
            while True:
                if "-/" in current:
                    doc_parts.append(current.split("-/", maxsplit=1)[0].strip())
                    i += 1
                    break
                if current:
                    doc_parts.append(current)
                i += 1
                if i >= len(lines):
                    break
                current = lines[i].strip()

            doc_text = " ".join(part for part in doc_parts if part).strip()
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            decl_match = _LEAN_DECL_RE.match(lines[j]) if j < len(lines) else None
            heading = "doc block"
            kind = "doc_block"
            entrypoints = _ENTRYPOINT_RE.findall(doc_text)
            extra_tags: list[str] = []
            if decl_match is not None:
                kind = decl_match.group("kind")
                heading = decl_match.group("name")
                entrypoints = [heading, *entrypoints]
                if kind in {"theorem", "lemma", "example"}:
                    extra_tags.append("verification")
                elif kind == "def":
                    extra_tags.append("definition")
                i = j + 1
            entries.append(
                {
                    "kind": kind,
                    "heading": heading,
                    "text": doc_text,
                    "tags": _classify_candidate_tags(doc_text, extra_tags=extra_tags),
                    "entrypoints": _unique_tags(entrypoints),
                }
            )
            continue

        decl_match = _LEAN_DECL_RE.match(lines[i])
        if decl_match is not None:
            kind = decl_match.group("kind")
            name = decl_match.group("name")
            extra_tags = []
            if kind in {"theorem", "lemma", "example"}:
                extra_tags.append("verification")
            elif kind == "def":
                extra_tags.append("definition")
            entries.append(
                {
                    "kind": kind,
                    "heading": name,
                    "text": stripped,
                    "tags": _unique_tags(extra_tags),
                    "entrypoints": [name],
                }
            )
        i += 1
    return entries


def _extract_source_candidates(path: Path) -> list[dict]:
    text = _read_text(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _extract_markdown_candidates(path, text)
    if suffix == ".py":
        return _extract_python_candidates(path, text)
    if suffix == ".rs":
        return _extract_rust_candidates(path, text)
    if suffix == ".lean":
        return _extract_lean_candidates(path, text)
    return []


def _infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "prose"
    if suffix == ".py":
        return "python"
    if suffix == ".rs":
        return "rust"
    if suffix == ".lean":
        return "lean"
    return "other"


def build_hypothesis_ledger(paths: list[Path], *, corpus_label: str) -> dict:
    sources = []
    total_candidates = 0
    for path in paths:
        candidates = _extract_source_candidates(path)
        total_candidates += len(candidates)
        sources.append(
            {
                "path": str(path),
                "source_type": _infer_source_type(path),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
    return {
        "corpus_label": corpus_label,
        "source_count": len(paths),
        "candidate_count": total_candidates,
        "sources": sources,
    }


def _ledger_to_markdown(ledger: dict) -> str:
    lines = [
        f"# Hypothesis Scaffold: {ledger['corpus_label']}",
        "",
        f"- Sources: {ledger['source_count']}",
        f"- Candidate fragments: {ledger['candidate_count']}",
        "",
    ]
    for source in ledger["sources"]:
        lines.append(f"## {source['path']}")
        lines.append("")
        lines.append(f"- Source type: {source['source_type']}")
        lines.append(f"- Candidate fragments: {source['candidate_count']}")
        lines.append("")
        for entry in source["candidates"][:12]:
            tag_text = ", ".join(entry["tags"]) if entry["tags"] else "none"
            lines.append(f"- `{entry['kind']}` [{tag_text}] {entry['heading']}: {entry['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold external prose/code into a reviewable hypothesis ledger."
    )
    parser.add_argument("paths", nargs="+", help="Files to ingest.")
    parser.add_argument(
        "--corpus-label",
        default="external_corpus",
        help="Human label for the corpus (default: external_corpus).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json).",
    )
    args = parser.parse_args()

    paths = [Path(raw).expanduser().resolve() for raw in args.paths]
    ledger = build_hypothesis_ledger(paths, corpus_label=args.corpus_label)
    if args.format == "markdown":
        print(_ledger_to_markdown(ledger), end="")
    else:
        print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
