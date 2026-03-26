# Copyright © 2026 Apple Inc.

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PYTHON = REPO_ROOT / "benchmarks" / "python"
if str(BENCHMARKS_PYTHON) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PYTHON))

import load_mmap_bench as bench


def _demo_args(**overrides):
    values = {
        "mmap_small_tensor_copy_max_bytes": None,
        "mmap_prefetch_strategy": "sequential",
        "mmap_hotset_promotion_top_k": None,
        "mmap_hotset_promotion_min_bytes": None,
        "policy_mode": "manual",
        "history_json": str(bench._demo_fixture_history_path()),
        "min_pass": None,
        "max_invalid_attempts": None,
        "gguf_nvfp4_compat": False,
        "baseline_run_id": None,
        "baseline_git_head": None,
        "fallback_priority_basis": "bytes",
        "reject_noisy_attempts": True,
        "debug_io": False,
        "show_trials": False,
        "history_window": 20,
        "trend_min_samples": 2,
        "trend_mad_mult": 3.0,
        "trend_hard_fail": False,
        "decode_phase": "soft",
        "decode_cv_gate_pct": None,
        "decode_regression_gate_pct": 3.0,
        "decode_aa_noise_floor": False,
        "decode_aa_multiplier": 1.0,
        "decode_aa_margin_pct": 0.5,
        "decode_llama_bench_binary": None,
        "decode_llama_bench_prompt_tokens": 16,
        "decode_llama_bench_gen_tokens": 64,
        "decode_llama_bench_repetitions": 1,
        "decode_llama_bench_gpu_layers": 99,
        "decode_llama_bench_depth": 0,
        "decode_llama_bench_threads": None,
        "decode_llama_bench_keep_warmup": False,
        "decode_cmd": None,
        "decode_runs": 3,
        "decode_synth_tokens": 256,
        "decode_synth_max_elems": 4_000_000,
        "decode_synth_repeats": 3,
        "runs": 1,
        "warmup_runs": 0,
        "attempts": 1,
        "report_top": 2,
        "format": "auto",
        "files": [],
        "discover_models": False,
        "discover_root": [],
        "discover_min_size_mib": 256.0,
        "discover_max_results": 20,
        "discover_all_per_dir": False,
        "discover_ollama_blobs": False,
        "discover_model_id_regex": None,
        "worker": False,
        "report_history": False,
        "policy_matrix": None,
        "golden_matrix": None,
        "demo_preset": None,
        "demo_parent": None,
        "demo_case": None,
        "demo_format": "text",
        "decode_synth": True,
    }
    values.update(overrides)
    return Namespace(**values)


class TestLoadMmapBench(unittest.TestCase):
    def test_median_optional_ignores_missing_values(self):
        self.assertEqual(bench._median_optional([None, 2, 4, None]), 3.0)

    def test_format_mib_preserves_missing_values(self):
        self.assertEqual(bench._format_mib(None), "n/a")
        self.assertEqual(bench._format_mib(2 * 1024 * 1024), "2.00MiB")

    def test_median_latency_regression_pct_uses_latency_direction(self):
        self.assertAlmostEqual(
            bench._median_latency_regression_pct([0.010, 0.012], [0.015, 0.018], False),
            50.0,
        )

    def test_format_fault_pair_handles_missing_values(self):
        self.assertEqual(bench._format_fault_pair(None, None), "(n/a)")
        self.assertEqual(bench._format_fault_pair(3, 1), "(3/1)")

    def test_parse_optional_byte_threshold_list(self):
        self.assertEqual(
            bench._parse_optional_byte_threshold_list("none,4096,65536,4096"),
            [None, 4096, 65536],
        )

    def test_parse_prefetch_strategy_list(self):
        self.assertEqual(
            bench._parse_prefetch_strategy_list(
                "sequential,willneed,none,willneed"
            ),
            ["sequential", "willneed", "none"],
        )

    def test_parse_optional_nonnegative_int_list(self):
        self.assertEqual(
            bench._parse_optional_nonnegative_int_list("none,1,4,0,1"),
            [None, 1, 4],
        )

    def test_history_bucket_includes_policy_signature(self):
        bucket = bench._build_history_bucket(
            {
                "format": "gguf",
                "model_class": "quantized_gguf",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "policy_signature": "prefetch=willneed,small_tensor_copy_lte=65536B",
            }
        )

        self.assertEqual(
            bucket["policy"], "prefetch=willneed,small_tensor_copy_lte=65536B"
        )

    def test_history_bucket_prefers_effective_policy_signature(self):
        bucket = bench._build_history_bucket(
            {
                "format": "gguf",
                "model_class": "quantized_gguf",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "policy_signature": "default",
                "effective_policy_signature": "copy",
            }
        )

        self.assertEqual(bucket["policy"], "copy")

    def test_policy_summary_combines_prefetch_and_threshold(self):
        summary = bench._policy_summary(
            bench._build_policy_knobs(
                65536,
                mmap_prefetch_strategy="willneed",
            )
        )

        self.assertEqual(
            summary,
            "mmap prefetch willneed; copy mapped-eligible tensors at or below 64KiB",
        )

    def test_policy_summary_includes_hotset_promotion(self):
        summary = bench._policy_summary(
            bench._build_policy_knobs(
                None,
                mmap_hotset_promotion_top_k=2,
                mmap_hotset_promotion_min_bytes=32 * 1024 * 1024,
            )
        )

        self.assertEqual(
            summary,
            "promote top 2 mapped-eligible tensors at or above 32MiB into owned buffers",
        )

    def test_effective_policy_signature_distinguishes_policy_families(self):
        self.assertEqual(
            bench._effective_policy_signature(
                memory_map=False, policy_knobs=bench._build_policy_knobs(None)
            ),
            "copy",
        )
        self.assertEqual(
            bench._effective_policy_signature(
                memory_map=True, policy_knobs=bench._build_policy_knobs(None)
            ),
            "mapped:default",
        )
        self.assertEqual(
            bench._effective_policy_signature(
                memory_map=True, policy_knobs=bench._build_policy_knobs(65536)
            ),
            "hybrid:small_tensor_copy_lte=65536B",
        )

    def test_compact_policy_matrix_cases(self):
        cases, cache_modes = bench._compact_policy_matrix_cases(
            Namespace(
                policy_matrix_thresholds=None,
                policy_matrix_prefetch_strategies=None,
                mmap_hotset_promotion_top_k=None,
                mmap_hotset_promotion_min_bytes=None,
            )
        )

        self.assertEqual(cache_modes, ["warm", "cold-best-effort"])
        self.assertEqual(
            [case["label"] for case in cases],
            [
                "default",
                "small_tensor_copy_lte=65536B",
                "prefetch=willneed",
                "prefetch=willneed,small_tensor_copy_lte=65536B",
            ],
        )

    def test_select_matrix_targets_prefers_dense_and_quantized(self):
        targets, notes = bench._select_matrix_targets(
            [
                {
                    "file": "/tmp/a.safetensors",
                    "format": "safetensors",
                    "model_class": "dense_safetensors",
                },
                {
                    "file": "/tmp/b.gguf",
                    "format": "gguf",
                    "model_class": "quantized_gguf",
                },
            ]
        )

        self.assertEqual([title for title, _ in targets], ["Dense Safetensors", "Quantized GGUF"])
        self.assertEqual(notes, [])

    def test_build_policy_knobs_ignores_hotset_min_without_top_k(self):
        knobs = bench._build_policy_knobs(
            None,
            mmap_hotset_promotion_top_k=None,
            mmap_hotset_promotion_min_bytes=32 * 1024 * 1024,
        )

        self.assertIsNone(knobs["mmap_hotset_promotion_top_k"])
        self.assertIsNone(knobs["mmap_hotset_promotion_min_bytes"])

    def test_parse_mmap_debug_stats_with_materialized_and_source_bytes(self):
        stderr = "\n".join(
            [
                "worker noise",
                (
                    "[io mmap] gguf file=/tmp/model.gguf "
                    "mapped_bytes=1024 copied_bytes=2048 fallback_tensors=3 "
                    "fallback_reasons={quantized_conversion:2,misaligned_offset:1} "
                    "fallback_reason_bytes={quantized_conversion:1536,misaligned_offset:512} "
                    "fallback_reason_source_bytes={quantized_conversion:768,misaligned_offset:512}"
                ),
            ]
        )

        stats = bench._parse_mmap_debug_stats(stderr)

        self.assertIsNotNone(stats)
        self.assertEqual(stats["tag"], "gguf")
        self.assertEqual(stats["mapped_bytes"], 1024)
        self.assertEqual(stats["copied_bytes"], 2048)
        self.assertEqual(stats["fallback_tensors"], 3)
        self.assertEqual(
            stats["fallback_reasons"],
            {"quantized_conversion": 2, "misaligned_offset": 1},
        )
        self.assertEqual(
            stats["fallback_reason_bytes"],
            {"quantized_conversion": 1536, "misaligned_offset": 512},
        )
        self.assertEqual(
            stats["fallback_reason_source_bytes"],
            {"quantized_conversion": 768, "misaligned_offset": 512},
        )

    def test_parse_mmap_debug_stats_without_optional_byte_maps(self):
        stats = bench._parse_mmap_debug_stats(
            "[io mmap] safetensors file=/tmp/model.safetensors "
            "mapped_bytes=4096 copied_bytes=0 fallback_tensors=0"
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats["fallback_reasons"], {})
        self.assertEqual(stats["fallback_reason_bytes"], {})
        self.assertEqual(stats["fallback_reason_source_bytes"], {})

    def test_build_route_suggestions_defaults_to_bytes_first_fallback_order(self):
        summary_payload = {
            "cache_mode": "warm",
            "mmap_coverage_mapped_ratio_pct": 90.0,
            "mmap_coverage_copied_bytes": 80 * 1024 * 1024,
            "mmap_coverage_fallback_tensors": 8,
            "mmap_coverage_fallback_reasons": {
                "quantized_conversion": 6,
                "dtype_conversion": 2,
            },
            "mmap_coverage_fallback_reason_bytes": {
                "quantized_conversion": 16 * 1024 * 1024,
                "dtype_conversion": 64 * 1024 * 1024,
            },
            "mmap_coverage_fallback_reason_source_bytes": {
                "quantized_conversion": 8 * 1024 * 1024,
                "dtype_conversion": 32 * 1024 * 1024,
            },
        }

        suggestions = bench._build_route_suggestions(
            summary_payload,
            decode_enabled=False,
            decode_gate_pct=5.0,
            fallback_priority_basis="bytes",
            trend_regressions=[],
        )

        self.assertGreaterEqual(len(suggestions), 2)
        self.assertEqual(suggestions[0]["route"], "Lazy Dtype Conversion")
        self.assertEqual(suggestions[0]["route_id"], "lazy_dtype_conversion")
        self.assertEqual(suggestions[1]["route"], "Quantized Direct Map")
        self.assertEqual(suggestions[1]["route_id"], "quantized_direct_map")

    def test_build_route_suggestions_can_switch_to_count_first_order(self):
        summary_payload = {
            "cache_mode": "warm",
            "mmap_coverage_mapped_ratio_pct": 90.0,
            "mmap_coverage_copied_bytes": 80 * 1024 * 1024,
            "mmap_coverage_fallback_tensors": 8,
            "mmap_coverage_fallback_reasons": {
                "quantized_conversion": 6,
                "dtype_conversion": 2,
            },
            "mmap_coverage_fallback_reason_bytes": {
                "quantized_conversion": 16 * 1024 * 1024,
                "dtype_conversion": 64 * 1024 * 1024,
            },
            "mmap_coverage_fallback_reason_source_bytes": {
                "quantized_conversion": 8 * 1024 * 1024,
                "dtype_conversion": 32 * 1024 * 1024,
            },
        }

        suggestions = bench._build_route_suggestions(
            summary_payload,
            decode_enabled=False,
            decode_gate_pct=5.0,
            fallback_priority_basis="count",
            trend_regressions=[],
        )

        self.assertGreaterEqual(len(suggestions), 2)
        self.assertEqual(suggestions[0]["route"], "Quantized Direct Map")
        self.assertEqual(suggestions[1]["route"], "Lazy Dtype Conversion")

    def test_normalize_route_reference_accepts_display_and_id(self):
        self.assertEqual(
            bench._normalize_route_reference("Quantized Direct Map"),
            ("quantized_direct_map", "Quantized Direct Map"),
        )
        self.assertEqual(
            bench._normalize_route_reference("quantized_direct_map"),
            ("quantized_direct_map", "Quantized Direct Map"),
        )

    def test_build_comparison_summary_prefers_explicit_baseline_run(self):
        summary_payload = {
            "load_improve_pct_median": 20.0,
            "rss_reduce_pct_median": 25.0,
            "load_peak_memory_reduce_pct_median": 15.0,
            "total_peak_memory_reduce_pct_median": 10.0,
            "load_call_improve_pct_median": 12.0,
            "parse_improve_pct_median": 8.0,
            "tensor_setup_improve_pct_median": 7.0,
            "first_eval_improve_pct_median": 6.0,
            "decode_regression_pct_median": 2.0,
            "decode_first_token_regression_pct_median": 4.0,
            "format": "gguf",
            "model_class": "quantized_gguf",
            "cache_mode": "warm",
            "decode_mode": "synth",
            "policy_signature": "default",
        }
        baseline = {
            "run_id": "baseline-1",
            "file": "/tmp/model.gguf",
            "load_improve_pct_median": 10.0,
            "rss_reduce_pct_median": 20.0,
            "load_peak_memory_reduce_pct_median": 12.0,
            "total_peak_memory_reduce_pct_median": 8.0,
            "load_call_improve_pct_median": 9.0,
            "parse_improve_pct_median": 4.0,
            "tensor_setup_improve_pct_median": 5.0,
            "first_eval_improve_pct_median": 3.0,
            "decode_regression_pct_median": 5.0,
            "decode_first_token_regression_pct_median": 6.0,
            "environment": {"git_head_short": "abc1234"},
        }

        comparison = bench._build_comparison_summary(
            summary_payload,
            prior_history=[baseline],
            file_path="/tmp/model.gguf",
            trend_context={
                "scope": "bucket",
                "records": [],
                "bucket_key": "ignored",
                "bucket": bench._build_history_bucket(summary_payload),
            },
            baseline_run_id="baseline-1",
            baseline_git_head=None,
        )

        self.assertIsNotNone(comparison)
        self.assertEqual(comparison["basis"], "baseline_run")
        self.assertEqual(comparison["reference_run_id"], "baseline-1")
        self.assertAlmostEqual(
            comparison["metrics"]["load_improve_pct_median"]["delta"], 10.0
        )
        self.assertEqual(comparison["metrics"]["decode_regression_pct_median"]["status"], "win")

    def test_calibrate_route_suggestions_uses_bucket_history(self):
        suggestions = [
            {
                "route_id": "hotset_promotion",
                "route": "Hotset Promotion",
                "priority": 90,
                "confidence": 0.6,
                "confidence_label": "low",
                "confidence_why": "test",
            }
        ]
        prior_history = [
            {
                "history_bucket_key": "bucket-a",
                "attempted_route_id": "hotset_promotion",
                "route_outcome": "win",
            },
            {
                "history_bucket_key": "bucket-a",
                "attempted_route_id": "hotset_promotion",
                "route_outcome": "mixed",
            },
        ]

        calibrated = bench._calibrate_route_suggestions(
            suggestions, prior_history, bucket_key="bucket-a"
        )

        self.assertEqual(calibrated[0]["history_scope"], "bucket")
        self.assertIn("hit_rate", calibrated[0]["history_summary"])
        self.assertGreater(calibrated[0]["confidence"], 0.6)

    def test_build_history_report_aggregates_bucket_routes_fallbacks_and_policies(self):
        records = [
            {
                "history_bucket_key": "bucket-a",
                "history_bucket": {
                    "format": "gguf",
                    "model_class": "quantized_gguf",
                    "cache_mode": "warm",
                    "decode_mode": "synth",
                    "policy": "default",
                },
                "timestamp": "2026-03-26T00:00:00+00:00",
                "attempted_route_id": "quantized_direct_map",
                "attempted_route": "Quantized Direct Map",
                "route_outcome": "win",
                "policy_signature": "default",
                "mmap_coverage_fallback_reason_bytes": {
                    "quantized_conversion": 32 * 1024 * 1024
                },
                "comparison": {
                    "metrics": {
                        "parse_improve_pct_median": {
                            "status": "loss",
                            "delta": -4.0,
                        }
                    },
                    "auto_outcome": "win",
                },
            },
            {
                "history_bucket_key": "bucket-a",
                "history_bucket": {
                    "format": "gguf",
                    "model_class": "quantized_gguf",
                    "cache_mode": "warm",
                    "decode_mode": "synth",
                    "policy": "prefetch=willneed",
                },
                "timestamp": "2026-03-27T00:00:00+00:00",
                "attempted_route_id": "quantized_direct_map",
                "attempted_route": "Quantized Direct Map",
                "route_outcome": "loss",
                "policy_signature": "prefetch=willneed",
                "mmap_coverage_fallback_reason_bytes": {
                    "quantized_conversion": 8 * 1024 * 1024
                },
                "comparison": {
                    "metrics": {
                        "decode_regression_pct_median": {
                            "status": "loss",
                            "delta": 6.0,
                        }
                    },
                    "auto_outcome": "loss",
                },
            },
        ]

        report = bench._build_history_report(records, top_n=2)

        self.assertEqual(report["records"], 2)
        bucket = report["buckets"][0]
        self.assertEqual(bucket["bucket_key"], "bucket-a")
        self.assertEqual(
            bucket["route_hit_rate"][0]["route_id"], "quantized_direct_map"
        )
        self.assertEqual(
            bucket["top_fallback_reasons_by_bytes"][0]["reason"],
            "quantized_conversion",
        )
        self.assertEqual(bucket["policy_winners"][0]["policy"], "default")

    def test_choose_auto_policy_prefers_copy_for_quantized_low_coverage(self):
        decision = bench._choose_auto_policy(
            file_path="/tmp/model.gguf",
            fmt="gguf",
            model_metadata={
                "format": "gguf",
                "model_class": "quantized_gguf",
                "weight_class": "quantized",
            },
            cache_mode="warm",
            decode_mode="synth",
            prior_history=[],
            probe_stats={
                "mapped_bytes": 32 * 1024 * 1024,
                "copied_bytes": 256 * 1024 * 1024,
                "fallback_tensors": 32,
                "fallback_reason_bytes": {
                    "quantized_conversion": 224 * 1024 * 1024
                },
            },
        )

        self.assertEqual(decision["effective_policy_mode"], "copy")
        self.assertFalse(decision["effective_memory_map"])

    def test_choose_auto_policy_prefers_bucket_history_winner(self):
        prior_history = [
            {
                "file": "/tmp/model.safetensors",
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "effective_policy_signature": "hybrid:small_tensor_copy_lte=65536B",
                "effective_policy_mode": "hybrid",
                "route_outcome": "win",
                "load_improve_pct_median": 18.0,
                "rss_reduce_pct_median": 24.0,
                "decode_regression_pct_median": 1.0,
            },
            {
                "file": "/tmp/model.safetensors",
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "effective_policy_signature": "mapped:default",
                "effective_policy_mode": "mapped",
                "route_outcome": "loss",
                "load_improve_pct_median": 3.0,
                "rss_reduce_pct_median": 5.0,
                "decode_regression_pct_median": 9.0,
            },
        ]

        decision = bench._choose_auto_policy(
            file_path="/tmp/model.safetensors",
            fmt="safetensors",
            model_metadata={
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "weight_class": "dense",
            },
            cache_mode="warm",
            decode_mode="synth",
            prior_history=prior_history,
            probe_stats={
                "mapped_bytes": 512 * 1024 * 1024,
                "copied_bytes": 64 * 1024 * 1024,
                "fallback_tensors": 4,
                "fallback_reason_bytes": {},
            },
        )

        self.assertEqual(
            decision["effective_policy_signature"],
            "hybrid:small_tensor_copy_lte=65536B",
        )

    def test_choose_auto_policy_prefers_hotset_when_probe_decode_is_first_touch_heavy(self):
        decision = bench._choose_auto_policy(
            file_path="/tmp/model.safetensors",
            fmt="safetensors",
            model_metadata={
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "weight_class": "dense",
            },
            cache_mode="warm",
            decode_mode="synth",
            prior_history=[],
            probe_stats={
                "mapped_bytes": 512 * 1024 * 1024,
                "copied_bytes": 8 * 1024 * 1024,
                "fallback_tensors": 1,
                "fallback_reason_bytes": {},
            },
            probe_trial={
                "load_call_s": 0.40,
                "first_eval_s": 0.20,
                "decode_first_token_s": 0.32,
                "decode_tok_s": 1400.0,
                "decode_first_token_minor_faults": 9,
                "decode_first_token_major_faults": 1,
                "decode_steady_minor_faults": 1,
                "decode_steady_major_faults": 0,
            },
        )

        self.assertEqual(decision["selected_candidate"], "hybrid_hotset")
        self.assertEqual(
            decision["probe_decode_summary"]["first_token_faults"],
            10,
        )

    def test_choose_auto_policy_uses_first_token_history_to_avoid_mapped_default(self):
        prior_history = [
            {
                "file": "/tmp/model.safetensors",
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "effective_policy_signature": "mapped:default",
                "effective_policy_mode": "mapped",
                "route_outcome": "mixed",
                "load_improve_pct_median": 20.0,
                "rss_reduce_pct_median": 24.0,
                "decode_regression_pct_median": 1.0,
                "decode_first_token_regression_pct_median": 15.0,
            },
            {
                "file": "/tmp/model.safetensors",
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "cache_mode": "warm",
                "decode_mode": "synth",
                "effective_policy_signature": "hybrid:hotset_topk=2,hotset_min_bytes=33554432B",
                "effective_policy_mode": "hybrid",
                "route_outcome": "mixed",
                "load_improve_pct_median": 18.0,
                "rss_reduce_pct_median": 22.0,
                "decode_regression_pct_median": 1.5,
                "decode_first_token_regression_pct_median": 2.0,
            },
        ]

        decision = bench._choose_auto_policy(
            file_path="/tmp/model.safetensors",
            fmt="safetensors",
            model_metadata={
                "format": "safetensors",
                "model_class": "dense_safetensors",
                "weight_class": "dense",
            },
            cache_mode="warm",
            decode_mode="synth",
            prior_history=prior_history,
            probe_stats={
                "mapped_bytes": 512 * 1024 * 1024,
                "copied_bytes": 16 * 1024 * 1024,
                "fallback_tensors": 2,
                "fallback_reason_bytes": {},
            },
        )

        self.assertEqual(
            decision["effective_policy_signature"],
            "hybrid:hotset_topk=2,hotset_min_bytes=33554432B",
        )

    def test_build_golden_matrix_targets_creates_expected_fixtures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            targets = bench._build_golden_matrix_targets(Path(temp_dir))

            self.assertEqual(
                [title for title, _ in targets],
                [
                    "Dense Safetensors",
                    "Quantized GGUF",
                    "Misaligned Safetensors",
                ],
            )
            for _, target in targets:
                self.assertTrue(Path(target["file"]).exists())

    def test_validate_demo_preset_args_rejects_conflicting_modes(self):
        args = _demo_args(demo_preset="history-replay", report_history=True)

        with self.assertRaisesRegex(
            ValueError, "--demo-preset cannot be combined with --report-history"
        ):
            bench._validate_demo_preset_args(args)

    def test_build_adaptation_demo_targets_uses_canonical_buckets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            buckets, notes = bench._build_adaptation_demo_targets(
                file_paths=[],
                requested_format="auto",
                discovered_formats={},
                discovered_details={},
                temp_root=Path(temp_dir),
            )

        self.assertEqual(
            [bucket["bucket_id"] for bucket in buckets],
            ["dense_warm", "dense_cold", "quantized_warm", "misaligned_warm"],
        )
        self.assertEqual(notes, [])

    def test_build_adaptation_demo_manifest_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            buckets, _ = bench._build_adaptation_demo_targets(
                file_paths=[],
                requested_format="auto",
                discovered_formats={},
                discovered_details={},
                temp_root=Path(temp_dir),
            )
            manifest = bench._build_adaptation_demo_manifest(
                script_path="/tmp/load_mmap_bench.py",
                args=_demo_args(),
                history_json_path="/tmp/demo_history.jsonl",
                buckets=buckets,
            )

        self.assertEqual(len(manifest), 20)
        self.assertEqual(manifest[0]["case_id"], "dense_warm::copy")
        self.assertEqual(manifest[-1]["case_id"], "misaligned_warm::auto")

    def test_build_history_replay_demo_summary_uses_fixture_history(self):
        summary = bench._build_history_replay_demo_summary(_demo_args())

        self.assertEqual(summary["demo_preset"], "history-replay")
        self.assertEqual(summary["history_source"], "fixture")
        self.assertEqual(
            summary["scorecard"]["comparison_basis_checks"]["comparable_history"],
            "bucket_recent_median",
        )
        self.assertEqual(
            summary["scorecard"]["comparison_basis_checks"]["explicit_baseline"],
            "baseline_run",
        )
        self.assertEqual(
            set(summary.keys()),
            {
                "demo_preset",
                "demo_session_id",
                "history_source",
                "targets",
                "steps",
                "verdict",
                "scorecard",
                "evidence",
            },
        )
        self.assertEqual(len(summary["scorecard"]["external_corpus_cases"]), 3)

    def test_build_prime_physics_corpus_fixture_summary_prefers_scoped_routes(self):
        summary = bench._build_prime_physics_corpus_fixture_summary()
        cases = {case["case_id"]: case for case in summary["cases"]}

        self.assertEqual(summary["pass_count"], 3)
        self.assertEqual(summary["match_rate"], 1.0)
        self.assertEqual(
            cases["negative_results_persist"]["chosen_route_id"], "stability_audit"
        )
        self.assertEqual(
            cases["single_instance_scope"]["chosen_route_id"], "broaden_the_matrix"
        )
        self.assertEqual(
            cases["formalization_scope_guardrail"]["chosen_route_id"], "stability_audit"
        )

    def test_build_regression_forensics_demo_summary_surfaces_expected_labels(self):
        summary = bench._build_regression_forensics_demo_summary(_demo_args())
        cases = {case["case_id"]: case for case in summary["scorecard"]["cases"]}

        self.assertIn(
            "parse improve",
            cases["parse_tensor_setup_regression"]["loss_metrics"],
        )
        self.assertIn(
            "tensor-setup improve",
            cases["parse_tensor_setup_regression"]["loss_metrics"],
        )
        self.assertIn(
            "first-token regression",
            cases["first_token_regression"]["loss_metrics"],
        )
        self.assertEqual(cases["mixed_tradeoff"]["outcome"], "mixed")
        self.assertEqual(len(summary["scorecard"]["external_corpus_cases"]), 3)

    def test_build_persistence_memory_demo_summary_shows_scope_guardrail(self):
        summary = bench._build_persistence_memory_demo_summary(_demo_args())
        cases = {case["case_id"]: case for case in summary["scorecard"]["cases"]}
        negative = {
            stage["stage_id"]: stage for stage in cases["negative_results_persist"]["stages"]
        }

        self.assertEqual(summary["demo_preset"], "persistence-memory")
        self.assertEqual(summary["history_source"], "fixture")
        self.assertEqual(summary["verdict"]["status"], "pass")
        self.assertEqual(
            negative["fresh"]["suppression_state"],
            "unknown",
        )
        self.assertEqual(
            negative["session"]["suppression_state"],
            "suppressed",
        )
        self.assertEqual(
            negative["persistent_unscoped"]["suppression_state"],
            "resurfaced",
        )
        self.assertEqual(
            negative["persistent_scoped"]["suppression_state"],
            "suppressed",
        )
        self.assertEqual(summary["scorecard"]["session_suppressed"], 3)
        self.assertEqual(summary["scorecard"]["persistent_unscoped_resurfaced"], 3)
        self.assertEqual(summary["scorecard"]["persistent_scoped_suppressed"], 3)

    def test_summarize_adaptation_demo_prefers_copy_for_quantized_and_misaligned(self):
        buckets = [
            {
                "bucket_id": "dense_warm",
                "title": "Dense Safetensors / warm",
                "cache_mode": "warm",
                "target": {
                    "file": "/tmp/dense.safetensors",
                    "format": "safetensors",
                    "model_class": "dense_safetensors",
                    "label": "dense",
                },
            },
            {
                "bucket_id": "quantized_warm",
                "title": "Quantized GGUF / warm",
                "cache_mode": "warm",
                "target": {
                    "file": "/tmp/quant.gguf",
                    "format": "gguf",
                    "model_class": "quantized_gguf",
                    "label": "quant",
                },
            },
            {
                "bucket_id": "misaligned_warm",
                "title": "Misaligned Safetensors / warm",
                "cache_mode": "warm",
                "target": {
                    "file": "/tmp/misaligned.safetensors",
                    "format": "safetensors",
                    "model_class": "dense_safetensors",
                    "label": "misaligned",
                },
            },
        ]
        dense_mapped = {
            "effective_policy_signature": "mapped:default",
            "load_improve_pct_median": 18.0,
            "rss_reduce_pct_median": 24.0,
            "decode_regression_pct_median": 1.0,
        }
        dense_copy = {
            "effective_policy_signature": "copy",
            "load_improve_pct_median": 5.0,
            "rss_reduce_pct_median": 1.0,
            "decode_regression_pct_median": 0.2,
        }
        quant_copy = {
            "effective_policy_signature": "copy",
            "load_improve_pct_median": 8.0,
            "rss_reduce_pct_median": 6.0,
            "decode_regression_pct_median": 0.6,
        }
        quant_mapped = {
            "effective_policy_signature": "mapped:default",
            "load_improve_pct_median": 12.0,
            "rss_reduce_pct_median": 15.0,
            "decode_regression_pct_median": 6.5,
        }
        misaligned_copy = {
            "effective_policy_signature": "copy",
            "load_improve_pct_median": 1.0,
            "rss_reduce_pct_median": 0.0,
            "decode_regression_pct_median": 0.1,
        }
        misaligned_mapped = {
            "effective_policy_signature": "mapped:default",
            "load_improve_pct_median": -1.0,
            "rss_reduce_pct_median": 3.0,
            "decode_regression_pct_median": 0.3,
        }

        step_results = [
            {
                "bucket_id": "dense_warm",
                "case_id": "dense_warm::copy",
                "case_label": "copy",
                "kind": "seed",
                "ok": True,
                "returncode": 0,
                "command": "seed",
                "record": dense_copy,
            },
            {
                "bucket_id": "dense_warm",
                "case_id": "dense_warm::mapped",
                "case_label": "mapped",
                "kind": "seed",
                "ok": True,
                "returncode": 0,
                "command": "seed",
                "record": dense_mapped,
            },
            {
                "bucket_id": "dense_warm",
                "case_id": "dense_warm::auto",
                "case_label": "auto",
                "kind": "auto",
                "ok": True,
                "returncode": 0,
                "command": "auto",
                "record": {
                    **dense_mapped,
                    "effective_policy_signature": "hybrid:small_tensor_copy_lte=65536B",
                    "auto_policy_decision": {
                        "selected_candidate": "hybrid_small",
                        "candidates": [
                            {
                                "name": "hybrid_small",
                                "history_support": 2,
                                "reasons": ["dense warm bucket likes a non-copy policy"],
                            }
                        ],
                    },
                    "auto_policy_probe_mmap_coverage": {
                        "mapped_ratio_pct": 94.0,
                        "fallback_reason_bytes": {},
                    },
                },
            },
            {
                "bucket_id": "quantized_warm",
                "case_id": "quantized_warm::copy",
                "case_label": "copy",
                "kind": "seed",
                "ok": True,
                "returncode": 0,
                "command": "seed",
                "record": quant_copy,
            },
            {
                "bucket_id": "quantized_warm",
                "case_id": "quantized_warm::mapped",
                "case_label": "mapped",
                "kind": "seed",
                "ok": True,
                "returncode": 0,
                "command": "seed",
                "record": quant_mapped,
            },
            {
                "bucket_id": "quantized_warm",
                "case_id": "quantized_warm::auto",
                "case_label": "auto",
                "kind": "auto",
                "ok": True,
                "returncode": 0,
                "command": "auto",
                "record": {
                    **quant_copy,
                    "auto_policy_decision": {
                        "selected_candidate": "copy_only",
                        "candidates": [
                            {
                                "name": "copy_only",
                                "history_support": 3,
                                "reasons": ["quantized coverage is weak"],
                            }
                        ],
                    },
                    "auto_policy_probe_mmap_coverage": {
                        "mapped_ratio_pct": 35.0,
                        "fallback_reason_bytes": {"quantized_conversion": 123},
                    },
                },
            },
            {
                "bucket_id": "misaligned_warm",
                "case_id": "misaligned_warm::copy",
                "case_label": "copy",
                "kind": "seed",
                "ok": True,
                "returncode": 0,
                "command": "seed",
                "record": misaligned_copy,
            },
            {
                "bucket_id": "misaligned_warm",
                "case_id": "misaligned_warm::mapped",
                "case_label": "mapped",
                "kind": "seed",
                "ok": True,
                "returncode": 0,
                "command": "seed",
                "record": misaligned_mapped,
            },
            {
                "bucket_id": "misaligned_warm",
                "case_id": "misaligned_warm::auto",
                "case_label": "auto",
                "kind": "auto",
                "ok": True,
                "returncode": 0,
                "command": "auto",
                "record": {
                    **misaligned_copy,
                    "auto_policy_decision": {
                        "selected_candidate": "copy_only",
                        "candidates": [
                            {
                                "name": "copy_only",
                                "history_support": 2,
                                "reasons": ["misaligned mapping looks hostile"],
                            }
                        ],
                    },
                    "auto_policy_probe_mmap_coverage": {
                        "mapped_ratio_pct": 0.0,
                        "fallback_reason_bytes": {"misaligned_offset": 456},
                    },
                },
            },
        ]

        summary = bench._summarize_adaptation_demo(
            history_source="temp",
            buckets=buckets,
            step_results=step_results,
        )
        bucket_results = {
            bucket["bucket_id"]: bucket
            for bucket in summary["scorecard"]["bucket_results"]
        }

        self.assertNotEqual(bucket_results["dense_warm"]["chosen_policy"], "copy")
        self.assertEqual(bucket_results["quantized_warm"]["chosen_policy"], "copy")
        self.assertEqual(bucket_results["misaligned_warm"]["chosen_policy"], "copy")
        self.assertEqual(
            set(summary.keys()),
            {
                "demo_preset",
                "demo_session_id",
                "history_source",
                "targets",
                "steps",
                "verdict",
                "scorecard",
                "evidence",
            },
        )


if __name__ == "__main__":
    unittest.main()
