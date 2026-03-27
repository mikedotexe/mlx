# Copyright © 2026 Apple Inc.

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PYTHON = REPO_ROOT / "benchmarks" / "python"
if str(BENCHMARKS_PYTHON) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PYTHON))

import run_hybrid_mlx_prompt_bench as hybrid


class TestRunHybridMlxPromptBench(unittest.TestCase):
    def test_shell_join_leaves_placeholders_unquoted(self):
        command = hybrid._shell_join(
            [
                "/tmp/python",
                "/tmp/helper.py",
                "--file",
                "{file}",
                "--prompt",
                "Say hello in five words.",
            ]
        )

        self.assertIn("{file}", command)
        self.assertIn("'Say hello in five words.'", command)

    def test_build_decode_command_template_includes_benchmark_placeholders(self):
        command = hybrid._build_decode_command_template(
            decode_python="/tmp/python",
            decode_helper="/tmp/helper.py",
            prompt="Say hello in five words.",
            max_tokens=8,
            temperature=0.0,
            system_prompt=None,
            trust_remote_code=False,
            ignore_chat_template=False,
            print_text=False,
        )

        self.assertTrue(command.startswith("/usr/bin/env PYTHONPATH= /tmp/python"))
        self.assertIn("--file {file}", command)
        self.assertIn("--memory-map {memory_map}", command)
        self.assertIn("--format {format}", command)
        self.assertIn("--max-tokens 8", command)

    def test_build_benchmark_env_prepends_repo_python(self):
        env = hybrid._build_benchmark_env(REPO_ROOT)

        self.assertTrue(env["PYTHONPATH"].startswith(str(REPO_ROOT / "python")))


if __name__ == "__main__":
    unittest.main()
