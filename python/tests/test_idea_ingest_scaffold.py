# Copyright © 2026 Apple Inc.

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PYTHON = REPO_ROOT / "benchmarks" / "python"
if str(BENCHMARKS_PYTHON) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PYTHON))

import idea_ingest_scaffold as ingest


class TestIdeaIngestScaffold(unittest.TestCase):
    def test_extract_markdown_candidates_finds_bullets_and_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "claims.md"
            path.write_text(
                "\n".join(
                    [
                        "# Verified Contributions",
                        "- Hypothesis: residual lift is classical",
                        "- Verified by `cargo test`",
                        "",
                        "| Hypothesis | Result |",
                        "|------------|--------|",
                        "| `2p resonance` | Refuted |",
                    ]
                ),
                encoding="utf-8",
            )
            entries = ingest._extract_source_candidates(path)

        self.assertGreaterEqual(len(entries), 3)
        self.assertEqual(entries[0]["kind"], "bullet")
        self.assertIn("hypothesis", entries[0]["tags"])
        self.assertEqual(entries[1]["entrypoints"], ["cargo test"])
        self.assertEqual(entries[2]["kind"], "table_row")

    def test_extract_python_candidates_finds_hypothesis_docstrings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "analysis.py"
            path.write_text(
                "\n".join(
                    [
                        '"""Hypothesis: better seeds increase density."""',
                        "",
                        "def test_seed_quality():",
                        '    """Hypothesis: significant effect expected."""',
                        "    return 1",
                    ]
                ),
                encoding="utf-8",
            )
            entries = ingest._extract_source_candidates(path)

        self.assertEqual(entries[0]["kind"], "module_doc")
        self.assertIn("hypothesis", entries[0]["tags"])
        self.assertEqual(entries[1]["heading"], "test_seed_quality")
        self.assertIn("test", entries[1]["tags"])

    def test_build_hypothesis_ledger_collects_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            md_path = Path(temp_dir) / "claims.md"
            py_path = Path(temp_dir) / "analysis.py"
            md_path.write_text("# Notes\n- Verified result\n", encoding="utf-8")
            py_path.write_text('"""Hypothesis: x."""\n', encoding="utf-8")

            ledger = ingest.build_hypothesis_ledger(
                [md_path, py_path], corpus_label="demo_corpus"
            )

        self.assertEqual(ledger["corpus_label"], "demo_corpus")
        self.assertEqual(ledger["source_count"], 2)
        self.assertEqual(len(ledger["sources"]), 2)
        self.assertGreaterEqual(ledger["candidate_count"], 2)

    def test_extract_rust_candidates_finds_module_scope_and_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "analysis.rs"
            path.write_text(
                "\n".join(
                    [
                        "//! Claim: the metaphor is not core math.",
                        "//! See [`crate::gravity`] for the actual analysis.",
                        "",
                        "/// Verified helper for narrow bookkeeping",
                        "pub const WINDOW: usize = 32;",
                        "",
                        "#[test]",
                        "fn test_preserves_negative_result_memory() {}",
                    ]
                ),
                encoding="utf-8",
            )
            entries = ingest._extract_source_candidates(path)

        self.assertEqual(entries[0]["kind"], "module_doc")
        self.assertIn("hypothesis", entries[0]["tags"])
        self.assertIn("scope", entries[0]["tags"])
        self.assertEqual(entries[0]["entrypoints"], ["crate::gravity"])
        self.assertEqual(entries[1]["heading"], "WINDOW")
        self.assertIn("verification", entries[1]["tags"])
        self.assertEqual(entries[2]["kind"], "test")
        self.assertEqual(entries[2]["heading"], "test_preserves_negative_result_memory")

    def test_extract_lean_candidates_finds_doc_blocks_and_theorems(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Proofs.lean"
            path.write_text(
                "\n".join(
                    [
                        "/-- Claim: local certificates are proved, global mechanism remains open. -/",
                        "theorem local_certificate : True := by",
                        "  trivial",
                        "",
                        "def witnessValue : Nat := 3",
                    ]
                ),
                encoding="utf-8",
            )
            entries = ingest._extract_source_candidates(path)

        self.assertEqual(entries[0]["kind"], "theorem")
        self.assertEqual(entries[0]["heading"], "local_certificate")
        self.assertIn("hypothesis", entries[0]["tags"])
        self.assertIn("verification", entries[0]["tags"])
        self.assertEqual(entries[0]["entrypoints"], ["local_certificate"])
        self.assertEqual(entries[1]["kind"], "def")
        self.assertEqual(entries[1]["heading"], "witnessValue")
        self.assertIn("definition", entries[1]["tags"])


if __name__ == "__main__":
    unittest.main()
