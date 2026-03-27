# Copyright © 2026 Apple Inc.

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PYTHON = REPO_ROOT / "benchmarks" / "python"
if str(BENCHMARKS_PYTHON) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PYTHON))

import eval_mlx_prompt_pack as prompt_eval


class TestEvalMlxPromptPack(unittest.TestCase):
    def test_parse_model_spec_accepts_label_prefix(self):
        parsed = prompt_eval._parse_model_spec("qwen=/tmp/model.safetensors")

        self.assertEqual(parsed["label"], "qwen")
        self.assertTrue(parsed["path"].endswith("/tmp/model.safetensors"))

    def test_filter_cases_respects_tags(self):
        cases = [
            {"id": "general_case", "tags": ["general"]},
            {"id": "prime_case", "tags": ["primes"]},
        ]

        filtered = prompt_eval._filter_cases(cases, case_ids=set(), tags={"primes"})

        self.assertEqual([case["id"] for case in filtered], ["prime_case"])

    def test_evaluate_expectation_handles_exact_match(self):
        passed, checks = prompt_eval._evaluate_expectation(
            "45",
            {"equals": "45"},
        )

        self.assertTrue(passed)
        self.assertEqual(checks[0]["name"], "equals")

    def test_evaluate_expectation_handles_bullets_and_word_limits(self):
        passed, checks = prompt_eval._evaluate_expectation(
            "- red fruit\n- crisp snack\n- pie filling",
            {
                "bullet_count": 3,
                "max_words_per_bullet": 2,
            },
        )

        self.assertTrue(passed)
        self.assertEqual(len(checks), 2)

    def test_evaluate_expectation_handles_json_equality(self):
        passed, checks = prompt_eval._evaluate_expectation(
            '{"animal":"cat","sound":"meow"}',
            {"json_equals": {"animal": "cat", "sound": "meow"}},
        )

        self.assertTrue(passed)
        self.assertEqual(checks[0]["name"], "json_equals")

    def test_load_prompt_pack_returns_cases(self):
        pack = prompt_eval._load_prompt_pack(
            REPO_ROOT / "benchmarks" / "python" / "testdata" / "mlx_prompt_eval_pack.json"
        )

        self.assertGreaterEqual(len(pack), 10)
        self.assertEqual(pack[0]["id"], "arithmetic_digits")
        self.assertIn("tags", pack[0])

    def test_summarize_results_groups_by_tag(self):
        summary = prompt_eval._summarize_results(
            [
                {"passed": True, "tags": ["general", "esn"]},
                {"passed": False, "tags": ["esn"]},
            ]
        )

        self.assertEqual(summary["general"]["pass_count"], 1)
        self.assertEqual(summary["general"]["total_cases"], 1)
        self.assertEqual(summary["esn"]["pass_count"], 1)
        self.assertEqual(summary["esn"]["total_cases"], 2)


if __name__ == "__main__":
    unittest.main()
