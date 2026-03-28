# Copyright © 2026 Apple Inc.

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


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

    def test_parse_args_accepts_repacked_comparison_flags(self):
        args = bench._parse_args(
            [
                "--compare-repacked",
                "--repacked-model-file",
                "/tmp/model.aligned.safetensors",
            ]
        )

        self.assertTrue(args.compare_repacked)
        self.assertEqual(args.repacked_model_file, "/tmp/model.aligned.safetensors")

    def test_parse_args_rejects_repacked_model_without_comparison_flag(self):
        with self.assertRaises(SystemExit):
            bench._parse_args(
                ["--repacked-model-file", "/tmp/model.aligned.safetensors"]
            )

    def test_format_text_handles_skipped_lanes(self):
        payload = {
            "meta": {
                "warmup": 5,
                "runs": 30,
                "wheel_python": None,
                "repo_pythonpath": None,
                "model_file": None,
                "compare_repacked": False,
                "repacked_model_file": None,
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
                "compare_repacked": False,
                "repacked_model_file": None,
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

    def test_format_text_handles_repacked_comparison_payload(self):
        payload = {
            "meta": {
                "warmup": 1,
                "runs": 2,
                "wheel_python": None,
                "repo_pythonpath": "/tmp/repo-python",
                "model_file": "/tmp/model.safetensors",
                "compare_repacked": True,
                "repacked_model_file": None,
            },
            "results": {
                "wheel": {"available": False, "reason": "missing wheel"},
                "repo_mmap_probe": {
                    "available": True,
                    "load_ns": {
                        "median_ns": 200,
                        "min_ns": 180,
                        "max_ns": 220,
                        "mean_ns": 200,
                    },
                    "mmap_stats": {
                        "mapped_bytes": 0,
                        "copied_bytes": 2048,
                        "fallback_tensors": 2,
                        "fallback_reasons": {"misaligned_offset": 2},
                    },
                },
                "repo_mmap_probe_repacked": {
                    "available": True,
                    "load_ns": {
                        "median_ns": 150,
                        "min_ns": 140,
                        "max_ns": 170,
                        "mean_ns": 153,
                    },
                    "mmap_stats": {
                        "mapped_bytes": 2048,
                        "copied_bytes": 0,
                        "fallback_tensors": 0,
                        "fallback_reasons": {},
                    },
                },
                "repack_summary": {
                    "total_padding_bytes_added": 4096,
                    "payload_start_before": 57,
                    "payload_start_after": 82,
                    "expected_to_eliminate_misaligned_offset": True,
                },
                "repo_mmap_probe_delta": {
                    "available": True,
                    "mapped_bytes_delta": 2048,
                    "copied_bytes_delta": -2048,
                    "fallback_tensors_delta": -2,
                    "load_median_ns_delta": -50,
                    "improved_mapping": True,
                    "fallback_reason_change_summary": {
                        "before": {"misaligned_offset": 2},
                        "after": {},
                        "delta": {"misaligned_offset": -2},
                        "removed": ["misaligned_offset"],
                        "added": [],
                    },
                },
            },
        }

        text = bench._format_text(payload)

        self.assertIn("Repo mmap probe (repacked):", text)
        self.assertIn("Compare: mapped_bytes_delta=2048", text)
        self.assertIn("fallback_reason_delta={'misaligned_offset': -2}", text)

    def test_build_repo_mmap_probe_delta(self):
        delta = bench._build_repo_mmap_probe_delta(
            {
                "available": True,
                "load_ns": {"median_ns": 200},
                "mmap_stats": {
                    "mapped_bytes": 0,
                    "copied_bytes": 100,
                    "fallback_tensors": 1,
                    "fallback_reasons": {"misaligned_offset": 1},
                },
            },
            {
                "available": True,
                "load_ns": {"median_ns": 150},
                "mmap_stats": {
                    "mapped_bytes": 100,
                    "copied_bytes": 0,
                    "fallback_tensors": 0,
                    "fallback_reasons": {},
                },
            },
        )

        self.assertTrue(delta["available"])
        self.assertEqual(delta["mapped_bytes_delta"], 100)
        self.assertEqual(delta["copied_bytes_delta"], -100)
        self.assertEqual(delta["fallback_tensors_delta"], -1)
        self.assertEqual(delta["load_median_ns_delta"], -50)
        self.assertEqual(
            delta["fallback_reason_change_summary"]["delta"],
            {"misaligned_offset": -1},
        )

    def test_main_json_skips_missing_lanes(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = bench.main(["--format", "json"])

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["results"]["wheel"]["available"])
        self.assertFalse(payload["results"]["repo_mmap_probe"]["available"])

    def test_main_json_can_auto_create_and_compare_repacked_probe(self):
        original_probe = {
            "available": True,
            "load_ns": {
                "median_ns": 200,
                "min_ns": 190,
                "max_ns": 210,
                "mean_ns": 200,
            },
            "mmap_stats": {
                "mapped_bytes": 0,
                "copied_bytes": 2048,
                "fallback_tensors": 2,
                "fallback_reasons": {"misaligned_offset": 2},
            },
        }
        repacked_probe = {
            "available": True,
            "load_ns": {
                "median_ns": 120,
                "min_ns": 110,
                "max_ns": 130,
                "mean_ns": 120,
            },
            "mmap_stats": {
                "mapped_bytes": 2048,
                "copied_bytes": 0,
                "fallback_tensors": 0,
                "fallback_reasons": {},
            },
        }

        def fake_repack(model_file, *, output_path, base_alignment=None, force=False):
            self.assertEqual(model_file, "/tmp/model.safetensors")
            self.assertTrue(str(output_path).endswith(".aligned.safetensors"))
            return {
                "input_file": model_file,
                "output_file": str(output_path),
                "input_size": 100,
                "output_size": 4096,
                "tensor_count": 2,
                "total_padding_bytes_added": 3996,
                "base_alignment": 4096,
                "layout_iterations": 2,
                "payload_start_before": 57,
                "payload_start_after": 100,
                "alignment_before": {},
                "alignment_after": {},
                "expected_to_eliminate_misaligned_offset": True,
            }

        stdout = io.StringIO()
        with mock.patch.object(
            bench, "_run_child", side_effect=[original_probe, repacked_probe]
        ) as run_child_mock, mock.patch.object(
            bench, "repack_safetensors_file", side_effect=fake_repack
        ) as repack_mock, contextlib.redirect_stdout(stdout):
            rc = bench.main(
                [
                    "--format",
                    "json",
                    "--repo-pythonpath",
                    "/tmp/repo-python",
                    "--model-file",
                    "/tmp/model.safetensors",
                    "--compare-repacked",
                    "--warmup",
                    "1",
                    "--runs",
                    "2",
                ]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(run_child_mock.call_count, 2)
        repack_mock.assert_called_once()
        payload = json.loads(stdout.getvalue())
        self.assertIn("repo_mmap_probe_repacked", payload["results"])
        self.assertIn("repack_summary", payload["results"])
        self.assertIn("repo_mmap_probe_delta", payload["results"])
        self.assertTrue(payload["results"]["repo_mmap_probe_delta"]["improved_mapping"])

    def test_main_json_uses_explicit_repacked_model_file(self):
        original_probe = {
            "available": True,
            "load_ns": {
                "median_ns": 200,
                "min_ns": 190,
                "max_ns": 210,
                "mean_ns": 200,
            },
            "mmap_stats": {
                "mapped_bytes": 0,
                "copied_bytes": 2048,
                "fallback_tensors": 2,
                "fallback_reasons": {"misaligned_offset": 2},
            },
        }
        repacked_probe = {
            "available": True,
            "load_ns": {
                "median_ns": 150,
                "min_ns": 140,
                "max_ns": 170,
                "mean_ns": 150,
            },
            "mmap_stats": {
                "mapped_bytes": 2048,
                "copied_bytes": 0,
                "fallback_tensors": 0,
                "fallback_reasons": {},
            },
        }

        stdout = io.StringIO()
        with mock.patch.object(
            bench, "_run_child", side_effect=[original_probe, repacked_probe]
        ) as run_child_mock, mock.patch.object(
            bench,
            "summarize_repacked_pair",
            return_value={
                "input_file": "/tmp/model.safetensors",
                "output_file": "/tmp/model.aligned.safetensors",
                "input_size": 100,
                "output_size": 4096,
                "tensor_count": 2,
                "total_padding_bytes_added": 3996,
                "base_alignment": 4096,
                "layout_iterations": 2,
                "payload_start_before": 57,
                "payload_start_after": 100,
                "alignment_before": {},
                "alignment_after": {},
                "expected_to_eliminate_misaligned_offset": True,
            },
        ) as summarize_mock, mock.patch.object(
            bench, "repack_safetensors_file"
        ) as repack_mock, contextlib.redirect_stdout(stdout):
            rc = bench.main(
                [
                    "--format",
                    "json",
                    "--repo-pythonpath",
                    "/tmp/repo-python",
                    "--model-file",
                    "/tmp/model.safetensors",
                    "--compare-repacked",
                    "--repacked-model-file",
                    "/tmp/model.aligned.safetensors",
                ]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(run_child_mock.call_count, 2)
        summarize_mock.assert_called_once_with(
            "/tmp/model.safetensors", "/tmp/model.aligned.safetensors"
        )
        repack_mock.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            payload["results"]["repack_summary"]["output_file"],
            "/tmp/model.aligned.safetensors",
        )

    def test_main_json_reports_repack_failure_without_crashing(self):
        original_probe = {
            "available": True,
            "load_ns": {
                "median_ns": 200,
                "min_ns": 190,
                "max_ns": 210,
                "mean_ns": 200,
            },
            "mmap_stats": {
                "mapped_bytes": 0,
                "copied_bytes": 2048,
                "fallback_tensors": 2,
                "fallback_reasons": {"misaligned_offset": 2},
            },
        }

        stdout = io.StringIO()
        with mock.patch.object(
            bench, "_run_child", return_value=original_probe
        ) as run_child_mock, mock.patch.object(
            bench, "repack_safetensors_file", side_effect=ValueError("bad file")
        ), contextlib.redirect_stdout(stdout):
            rc = bench.main(
                [
                    "--format",
                    "json",
                    "--repo-pythonpath",
                    "/tmp/repo-python",
                    "--model-file",
                    "/tmp/model.safetensors",
                    "--compare-repacked",
                ]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(run_child_mock.call_count, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["results"]["repo_mmap_probe_repacked"]["available"])
        self.assertIn(
            "repack comparison failed",
            payload["results"]["repo_mmap_probe_repacked"]["reason"],
        )
        self.assertFalse(payload["results"]["repo_mmap_probe_delta"]["available"])


if __name__ == "__main__":
    unittest.main()
