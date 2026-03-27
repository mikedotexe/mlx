# Copyright © 2026 Apple Inc.

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PYTHON = REPO_ROOT / "benchmarks" / "python"
if str(BENCHMARKS_PYTHON) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PYTHON))

import unified_memory_zero_copy_bench as bench


class TestUnifiedMemoryZeroCopyBench(unittest.TestCase):
    def test_summarize_ns(self):
        summary = bench._summarize_ns([5, 9, 11])

        self.assertEqual(summary["median_ns"], 9)
        self.assertEqual(summary["min_ns"], 5)
        self.assertEqual(summary["max_ns"], 11)
        self.assertEqual(summary["mean_ns"], 8)

    def test_build_child_env_for_wheel_clears_pythonpath(self):
        env = bench._build_child_env(lane="wheel")

        self.assertIn("PYTHONPATH", env)
        self.assertEqual(env["PYTHONPATH"], "")

    def test_build_child_env_for_repo_prepends_pythonpath(self):
        env = bench._build_child_env(lane="repo_mmap", repo_pythonpath="/tmp/repo-python")

        self.assertTrue(env["PYTHONPATH"].startswith("/tmp/repo-python"))

    def test_build_child_command_includes_model_file_when_present(self):
        command = bench._build_child_command(
            python_executable="/tmp/python",
            child_mode="repo_mmap",
            output_format="json",
            warmup=2,
            runs=3,
            model_file="/tmp/model.safetensors",
        )

        self.assertIn("--child-mode", command)
        self.assertIn("repo_mmap", command)
        self.assertIn("--model-file", command)
        self.assertIn("/tmp/model.safetensors", command)

    def test_format_text_handles_skipped_lanes(self):
        payload = {
            "meta": {
                "warmup": 5,
                "runs": 30,
                "wheel_python": None,
                "repo_pythonpath": None,
                "model_file": None,
            },
            "results": {
                "wheel": {"available": False, "reason": "missing wheel"},
                "repo_mmap_probe": {"available": False, "reason": "missing repo"},
            },
        }

        text = bench._format_text(payload)

        self.assertIn("missing wheel", text)
        self.assertIn("missing repo", text)

    def test_format_text_handles_populated_payload(self):
        payload = {
            "meta": {
                "warmup": 1,
                "runs": 2,
                "wheel_python": "/tmp/wheel-python",
                "repo_pythonpath": "/tmp/repo-python",
                "model_file": "/tmp/model.safetensors",
            },
            "results": {
                "wheel": {
                    "available": True,
                    "metal_available": True,
                    "default_device": "Device(gpu, 0)",
                    "device_name": "Apple Test GPU",
                    "cases": {
                        case_id: {
                            "median_ns": 10,
                            "min_ns": 8,
                            "max_ns": 12,
                            "mean_ns": 10,
                        }
                        for case_id in bench._CASE_ORDER
                    },
                    "semantics": {
                        "np_view_owndata": False,
                        "np_view_writeable": True,
                        "np_view_mutation_reflects": True,
                    },
                },
                "repo_mmap_probe": {
                    "available": True,
                    "load_ns": {
                        "median_ns": 100,
                        "min_ns": 90,
                        "max_ns": 110,
                        "mean_ns": 100,
                    },
                    "mmap_stats": {
                        "mapped_bytes": 123,
                        "copied_bytes": 456,
                        "fallback_tensors": 7,
                        "fallback_reasons": {"make_buffer_failed": 7},
                    },
                },
            },
        }

        text = bench._format_text(payload)

        self.assertIn("Apple Test GPU", text)
        self.assertIn("docs_gpu_cpu", text)
        self.assertIn("mapped_bytes=123", text)

    def test_main_json_skips_missing_lanes(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = bench.main(["--format", "json"])

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["results"]["wheel"]["available"])
        self.assertFalse(payload["results"]["repo_mmap_probe"]["available"])


if __name__ == "__main__":
    unittest.main()
