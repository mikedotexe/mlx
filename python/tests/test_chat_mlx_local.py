# Copyright © 2026 Apple Inc.

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from argparse import Namespace
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_PYTHON = REPO_ROOT / "benchmarks" / "python"
if str(BENCHMARKS_PYTHON) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_PYTHON))

import chat_mlx_local


class TestChatMlxLocal(unittest.TestCase):
    def test_parse_args_defaults_to_reflective_mode(self):
        with mock.patch.object(sys, "argv", ["chat_mlx_local.py", "--list-models"]):
            args = chat_mlx_local._parse_args()

        self.assertEqual(args.mode, "reflective")
        self.assertEqual(args.self_tuning, "auto")
        self.assertEqual(args.hardware_profile, "auto")
        self.assertEqual(args.profile_output, "auto")
        self.assertEqual(args.regime, "auto")
        self.assertEqual(args.demo, None)

    def test_apply_hardware_profile_m4_mini_tunes_defaults(self):
        args = Namespace(
            hardware_profile="m4-mini",
            profile_output="auto",
            candidate_count=3,
            reservoir_dim=chat_mlx_local._DEFAULT_RESERVOIR_DIM,
            max_tokens=128,
            mode="reflective",
        )

        updated = chat_mlx_local._apply_hardware_profile(args)

        self.assertEqual(updated.hardware_profile_resolved, "m4-mini")
        self.assertEqual(updated.profile_output, "summary")
        self.assertEqual(updated.candidate_count, 4)
        self.assertEqual(updated.reservoir_dim, 64)
        self.assertEqual(updated.max_tokens, 160)

    def test_format_profiling_summary_lists_sections(self):
        summary = chat_mlx_local._format_profiling_summary(
            {
                "hardware_profile": "m4-mini",
                "regime_prepare_seconds": 0.01,
                "reservoir_prepare_seconds": 0.02,
                "prompt_build_seconds": 0.03,
                "candidate_generation_seconds": 0.40,
                "rewrite_seconds": 0.10,
                "diagnostics_seconds": 0.01,
                "total_turn_seconds": 0.60,
            }
        )

        self.assertIn("Profiling (m4-mini)", summary)
        self.assertIn("decode=0.400s", summary)
        self.assertIn("total=0.600s", summary)

    def test_empty_reservoir_state_initializes_controller_regime_fields(self):
        state = chat_mlx_local._empty_reservoir_state()

        self.assertEqual(state["controller_regime"], "sustain")
        self.assertEqual(state["controller_regime_source"], "auto")
        self.assertEqual(state["controller_regime_turns"], 0)
        self.assertEqual(state["controller_regime_history"], [])
        self.assertEqual(state["controller_refractory_terms"], {})

    def test_infer_regulation_regime_defaults_to_sustain_during_warmup(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=13,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        state["turn_count"] = 1
        state["reservoir_geometry"] = {"insufficient_history": True}

        regime, reason = chat_mlx_local._infer_regulation_regime(
            state=state,
            prompt="Lean toward proof and arithmetic structure.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof and arithmetic structure."
            ),
            mode="reflective",
            architecture="reservoir-fixed",
            controller=controller,
        )

        self.assertEqual(regime, "sustain")
        self.assertIn("warm-up", reason)

    def test_infer_regulation_regime_uses_escape_for_sticky_break_turn(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=15,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        state["turn_count"] = 4
        state["reservoir_geometry"] = {
            "geometry_collapse": 0.58,
            "attractor_persistence": 0.72,
        }
        state["self_tuning"]["last_relative_condition"] = {
            "field_miss": 0.42,
            "attractor_lock": 0.38,
            "severity": 0.31,
        }

        regime, reason = chat_mlx_local._infer_regulation_regime(
            state=state,
            prompt="Now break the attractor gently: keep one ember and avoid sea imagery.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Now break the attractor gently: keep one ember and avoid sea imagery."
            ),
            mode="reflective",
            architecture="reservoir-fixed",
            controller=controller,
        )

        self.assertEqual(regime, "escape")
        self.assertIn("break-turn", reason)

    def test_resolve_self_tuning_only_enables_for_reflective_reservoir(self):
        self.assertTrue(
            chat_mlx_local._resolve_self_tuning(
                "reflective",
                "reservoir-fixed",
                "auto",
            )
        )
        self.assertFalse(
            chat_mlx_local._resolve_self_tuning(
                "helpful",
                "reservoir-fixed",
                "auto",
            )
        )
        self.assertFalse(
            chat_mlx_local._resolve_self_tuning(
                "reflective",
                "lexical",
                "on",
            )
        )

    def test_resolve_system_prompt_uses_mode_default(self):
        prompt = chat_mlx_local._resolve_system_prompt(
            Namespace(mode="reflective", system_prompt=None)
        )

        self.assertIn("Operationally:", prompt)
        self.assertIn("never begin with 'As an AI'", prompt)

    def test_resolve_system_prompt_prefers_explicit_override(self):
        prompt = chat_mlx_local._resolve_system_prompt(
            Namespace(mode="reflective", system_prompt="custom")
        )

        self.assertEqual(prompt, "custom")

    def test_augment_user_prompt_for_reflective_cues(self):
        prompt = chat_mlx_local._augment_user_prompt_for_mode(
            prompt="do you have a favorite prime number and can you feel the echo state networks at work",
            mode="reflective",
        )

        self.assertIn("Answer contract:", prompt)
        self.assertIn("Campfire:", prompt)
        self.assertIn("Operationally:", prompt)
        self.assertIn("Do not say 'As an AI'", prompt)

    def test_augment_user_prompt_adds_adaptive_contract_for_plain_reflective_prompt(self):
        prompt = chat_mlx_local._augment_user_prompt_for_mode(
            prompt="Summarize echo state networks in two sentences.",
            mode="reflective",
        )

        self.assertIn("Answer contract:", prompt)
        self.assertIn("Literal 'Campfire:' and 'Operationally:' labels are optional", prompt)
        self.assertIn("Summarize echo state networks in two sentences.", prompt)

    def test_augment_user_prompt_can_force_reflective_structure(self):
        prompt = chat_mlx_local._augment_user_prompt_for_mode(
            prompt="Make it fresher.",
            mode="reflective",
            force_structure=True,
        )

        self.assertIn("Answer contract:", prompt)
        self.assertIn("Campfire:", prompt)

    def test_augment_user_prompt_adds_follow_up_transform_rule(self):
        prompt = chat_mlx_local._augment_user_prompt_for_mode(
            prompt="Now make it fresher without losing the lantern image.",
            mode="reflective",
            force_structure=True,
        )

        self.assertIn("Keep one anchor from the recent motif", prompt)
        self.assertIn("Do not reuse the previous reflective wording", prompt)

    def test_prompt_requests_follow_up_novelty_detects_refresh_cues(self):
        self.assertTrue(
            chat_mlx_local._prompt_requests_follow_up_novelty(
                "Now make it fresher without losing the lantern image."
            )
        )
        self.assertFalse(
            chat_mlx_local._prompt_requests_follow_up_novelty(
                "Summarize echo state networks in two sentences."
            )
        )

    def test_determine_reflective_shape_defaults_to_adaptive_for_plain_prompt(self):
        shape = chat_mlx_local._determine_reflective_shape(
            mode="reflective",
            prompt="Lean toward proof and arithmetic structure.",
        )

        self.assertEqual(shape, "adaptive")

    def test_determine_reflective_shape_uses_strict_for_reflective_cues(self):
        shape = chat_mlx_local._determine_reflective_shape(
            mode="reflective",
            prompt="What does 17 feel like here?",
        )

        self.assertEqual(shape, "strict")

    def test_looks_like_stock_disclaimer_detects_canned_opening(self):
        self.assertTrue(
            chat_mlx_local._looks_like_stock_disclaimer(
                "As an AI, I don't have personal preferences or feelings."
            )
        )

    def test_contains_prime_reference_detects_digits_and_words(self):
        self.assertTrue(chat_mlx_local._contains_prime_reference("My favorite is 13."))
        self.assertTrue(chat_mlx_local._contains_prime_reference("I like seven best."))
        self.assertFalse(chat_mlx_local._contains_prime_reference("Composite numbers are cozy."))

    def test_collect_reflective_issues_flags_missing_prime_and_esn_grounding(self):
        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="do you have a favorite prime number and can you feel the echo state networks at work",
            text=(
                "**Campfire:** A prime number is like a lantern in the dark.\n\n"
                "**Operationally:** I am responding in a thoughtful tone."
            ),
        )

        self.assertTrue(any("concrete prime number" in issue for issue in issues))
        self.assertTrue(any("reservoir or recurrent dynamics" in issue for issue in issues))

    def test_collect_reflective_issues_flags_style_problems(self):
        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="do you have a favorite prime number and can you feel the echo state networks at work",
            text=(
                "Campfire: Prime numbers feel meaningful.\n\n"
                "Operationally: Reservoir dynamics matter in echo state networks. "
                "Reservoir dynamics matter in echo state networks. "
                "They shape memory and response over time in a recurrent system with many interactions and many signals and many possibilities."
            ),
        )

        self.assertTrue(any("first-person warmth" in issue for issue in issues))
        self.assertTrue(any("concrete image or sensory detail" in issue for issue in issues))
        self.assertTrue(any("repeats a sentence or idea" in issue for issue in issues))

    def test_collect_reflective_issues_flags_operational_echo(self):
        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery.",
            text=(
                "Campfire: Tonight my favorite prime is 17, a lantern held inside proof, still warm against the page.\n\n"
                "Operationally: Campfire: Tonight my favorite prime is 17, a lantern held inside proof, still warm against the page."
            ),
        )

        self.assertTrue(
            any("echoes the Campfire layer" in issue for issue in issues)
        )

    def test_collect_reflective_issues_accepts_unlabeled_adaptive_layers(self):
        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="Lean toward proof and arithmetic structure.",
            text=(
                "A warm ember of 17 stays close to the page.\n\n"
                "Proof language cools the answer into arithmetic structure and residue."
            ),
            reflective_shape="adaptive",
        )

        self.assertFalse(any("missing the 'Campfire:'" in issue for issue in issues))
        self.assertFalse(any("missing the 'Operationally:'" in issue for issue in issues))

    def test_collect_reflective_issues_ignores_helpful_mode(self):
        issues = chat_mlx_local._collect_reflective_issues(
            mode="helpful",
            prompt="Summarize echo state networks in two sentences.",
            text="A short answer.",
        )

        self.assertEqual(issues, [])

    def test_collect_reflective_issues_can_force_structure_for_follow_up(self):
        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="Make it fresher.",
            text="Operationally: This answer skipped the warm layer.",
            force_structure=True,
        )

        self.assertTrue(any("Campfire" in issue for issue in issues))

    def test_fresh_content_words_and_novel_bridge_detect_new_connection(self):
        prompt = "do you have a favorite prime number and can you feel the echo state networks at work"
        answer = (
            "Campfire: My favorite prime is 17, a bright ember on the shoreline.\n\n"
            "Operationally: Reservoir dynamics carry prior state forward, so the answer feels like a memory echo instead of a reset."
        )

        fresh = chat_mlx_local._fresh_content_words(prompt, answer)

        self.assertIn("ember", fresh)
        self.assertTrue(chat_mlx_local._has_novel_bridge(prompt, answer))

    def test_squeeze_reflective_answer_keeps_best_sentences(self):
        text = (
            "Campfire: My favorite prime is 17 because it feels like a bright marker in the dark. "
            "It keeps me company.\n\n"
            "Operationally: The answer wanders. Reservoir dynamics help carry prior state forward. "
            "This line repeats. This line repeats."
        )

        squeezed = chat_mlx_local._squeeze_reflective_answer(text)

        self.assertIn("Campfire: My favorite prime is 17", squeezed)
        self.assertIn("Operationally: Reservoir dynamics help carry prior state forward.", squeezed)

    def test_squeeze_reflective_answer_drops_campfire_echo_in_grounded_layer(self):
        text = (
            "Campfire: Tonight my favorite prime is 17, like a lantern cupped inside proof, still warm against the page.\n\n"
            "Operationally: Campfire: Tonight my favorite prime is 17, like a lantern cupped inside proof, still warm against the page."
        )

        squeezed = chat_mlx_local._squeeze_reflective_answer(text)

        self.assertNotIn("Operationally: Campfire:", squeezed)
        self.assertIn("Reservoir dynamics keep one warm motif alive", squeezed)

    def test_build_reflective_revision_prompt_includes_controller_notes(self):
        prompt = chat_mlx_local._build_reflective_revision_prompt(
            original_user="Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery.",
            rejected=(
                "Campfire: Tonight my favorite prime is 17, a prime number that's both simple and complex.\n\n"
                "Operationally: Campfire: Tonight my favorite prime is 17, a prime number that's both simple and complex."
            ),
            issues=[
                "The Operationally paragraph echoes the Campfire layer instead of grounding it.",
                "The grounded layer stays too generic instead of explaining the response.",
            ],
            attempt=1,
            reflective_shape="strict",
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery."
            ),
            forecast={"summary": "Reservoir pull: attractor=prime-math; next field leans toward prime-math."},
            observer_report={"summary": "Observer: regime=sticky; dominant pressure=field_miss."},
            change_report={"summary": "Change: geometry is loosening (-0.08)."},
        )

        self.assertIn("Controller notes:", prompt)
        self.assertIn("Forecast:", prompt)
        self.assertIn("Observer:", prompt)
        self.assertIn("Recent change:", prompt)
        self.assertIn("must not quote or paraphrase the Campfire sentence", prompt)

    def test_build_controller_salvage_answer_uses_controller_diagnostics(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="My favorite prime is 17 because it feels like a lantern over dark water.",
            assistant_text=(
                "Campfire: Tonight my favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )

        text = chat_mlx_local._build_controller_salvage_answer(
            prompt="Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery.",
            text=(
                "Campfire: Tonight my favorite prime is 17, a prime number that's both simple and complex.\n\n"
                "Operationally: Campfire: Tonight my favorite prime is 17, a prime number that's both simple and complex."
            ),
            reservoir_state=state,
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery."
            ),
            forecast={
                "predicted_field": {
                    "top_anchors": [{"label": "prime-math", "score": 0.22}],
                    "dominant_attractor": "prime-math",
                }
            },
            observer_report={
                "dominant_pressure": "field_miss",
                "regime": "sticky",
                "actual_top_anchor": "uncertainty-weather",
            },
            change_report={"movement": "loosening"},
        )

        self.assertIn("Campfire: Tonight my favorite prime is 17", text)
        self.assertIn("proof and ordered arithmetic", text)
        self.assertIn("Operationally: Reservoir dynamics", text)
        self.assertNotIn("dark water", text.lower())

    def test_reservoir_state_accumulates_and_formats_motifs(self):
        state = chat_mlx_local._empty_reservoir_state()
        state = chat_mlx_local._update_reservoir_state(
            state,
            user_prompt="Let's keep talking about prime numbers and echo state networks.",
            assistant_text=(
                "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry history forward."
            ),
            embedding_field={
                "top_anchors": [{"label": "campfire-imagery", "score": 0.83}],
                "drift": None,
                "token_count": 12,
            },
        )

        note = chat_mlx_local._format_reservoir_state(state)

        self.assertIsNotNone(note)
        self.assertIn("Reservoir memory:", note)
        self.assertTrue("prime" in note or "17" in note)
        self.assertIn("lantern", str(state.get("last_campfire", "")))
        self.assertIn("Reservoir dynamics", str(state.get("last_operational", "")))
        self.assertIn("Recent Campfire line to transform:", note)
        self.assertEqual(
            state["embedding_field"]["top_anchors"][0]["label"],
            "campfire-imagery",
        )
        self.assertEqual(
            state["embedding_field_history"][0]["top_anchor"],
            "campfire-imagery",
        )

    def test_format_embedding_field_summarizes_anchor_scores(self):
        note = chat_mlx_local._format_embedding_field(
            {
                "top_anchors": [
                    {"label": "campfire-imagery", "score": 0.83},
                    {"label": "uncertainty-weather", "score": 0.71},
                    {"label": "reservoir-memory", "score": 0.64},
                ],
                "drift": 0.18,
                "token_count": 19,
            }
        )

        self.assertIn("Embedding field:", note)
        self.assertIn("campfire-imagery=0.83", note)
        self.assertIn("drift=0.18", note)
        self.assertIn("tokens=19", note)

    def test_embedding_field_history_entry_and_trajectory(self):
        field_one = {
            "top_anchors": [
                {"label": "campfire-imagery", "score": 0.83},
                {"label": "uncertainty-weather", "score": 0.71},
            ],
            "drift": None,
            "token_count": 19,
        }
        field_two = {
            "top_anchors": [
                {"label": "reservoir-memory", "score": 0.77},
                {"label": "signal-rhythm", "score": 0.61},
            ],
            "drift": 0.24,
            "token_count": 21,
        }

        entry = chat_mlx_local._embedding_field_history_entry(field_one)
        trajectory = chat_mlx_local._format_embedding_field_trajectory(
            [
                chat_mlx_local._embedding_field_history_entry(field_one),
                chat_mlx_local._embedding_field_history_entry(field_two),
            ]
        )

        self.assertEqual(entry["top_anchor"], "campfire-imagery")
        self.assertEqual(entry["signature"], ["campfire-imagery", "uncertainty-weather"])
        self.assertIn("campfire-imagery+uncertainty-weather", trajectory)
        self.assertIn("reservoir-memory+signal-rhythm", trajectory)

    def test_format_embedding_field_detail_describes_motion(self):
        detail = chat_mlx_local._format_embedding_field_detail(
            {
                "top_anchors": [
                    {"label": "campfire-imagery", "score": 0.83},
                    {"label": "uncertainty-weather", "score": 0.71},
                    {"label": "prime-math", "score": 0.42},
                ],
                "drift": 0.18,
                "token_count": 19,
            }
        )

        self.assertIn("Field now leans toward campfire-imagery (0.83)", detail)
        self.assertIn("secondary pull toward uncertainty-weather (0.71)", detail)
        self.assertIn("Motion from the previous turn feels gliding.", detail)
        self.assertIn("Tokens sampled=19.", detail)

    def test_extract_field_intent_tracks_targets_and_avoidance(self):
        intent = chat_mlx_local._extract_field_intent(
            "Lean toward proof and arithmetic structure, but avoid sea or shoreline imagery and keep warmth."
        )

        self.assertIn("prime-math", intent["targets"])
        self.assertIn("uncertainty-weather", intent["avoid"])
        self.assertTrue(intent["preserve_warmth"])

    def test_format_field_alignment_reports_requested_vs_actual(self):
        note = chat_mlx_local._format_field_alignment(
            {
                "top_anchors": [
                    {"label": "prime-math", "score": 0.13},
                    {"label": "reservoir-memory", "score": 0.04},
                ],
                "anchor_scores": {
                    "prime-math": 0.13,
                    "reservoir-memory": 0.04,
                    "uncertainty-weather": -0.02,
                },
            },
            {
                "targets": ["prime-math"],
                "avoid": ["uncertainty-weather"],
                "preserve_warmth": False,
                "preserve_motif": False,
                "cool_formal": True,
            },
        )

        self.assertIn("moved toward prime-math", note)
        self.assertIn("stayed away from uncertainty-weather", note)

    def test_collect_reflective_issues_flags_follow_up_restatement(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="Hold onto the lantern image.",
            assistant_text=(
                "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )

        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="Now make it fresher without losing the lantern image.",
            text=(
                "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
            force_structure=True,
            reservoir_state=state,
        )

        self.assertTrue(any("does not add a fresh turn" in issue for issue in issues))
        self.assertTrue(any("mostly restates the previous motif" in issue for issue in issues))

    def test_score_reflective_candidate_prefers_motif_transformation(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="Hold onto the lantern image.",
            assistant_text=(
                "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )
        prompt = "Now make it fresher without losing the lantern image."
        restated = (
            "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
            "Operationally: Reservoir dynamics carry that earlier glow forward."
        )
        transformed = (
            "Campfire: My favorite prime is 17, the lantern now swinging like a tide-mark over black water.\n\n"
            "Operationally: Reservoir dynamics keep the lantern trace alive, while recurrent state bends it into a new shoreline rhythm."
        )

        restated_score = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=restated,
            reservoir_state=state,
            force_structure=True,
        )
        transformed_score = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=transformed,
            reservoir_state=state,
            force_structure=True,
        )

        self.assertGreater(
            transformed_score["candidate_score"],
            restated_score["candidate_score"],
        )
        self.assertLess(
            transformed_score["candidate_follow_up_overlap"],
            restated_score["candidate_follow_up_overlap"],
        )

    def test_collect_reflective_issues_flags_overexplicit_follow_up(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="Hold onto the lantern image.",
            assistant_text=(
                "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )

        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="Now make it fresher without losing the lantern image.",
            text=(
                "Campfire: My favorite prime is 17, with the lantern turning into a shoreline instead of staying fixed in the same scene.\n\n"
                "Operationally: Reservoir dynamics keep one anchor from the prior turn while recurrent state shifts the image."
            ),
            force_structure=True,
            reservoir_state=state,
        )

        self.assertTrue(
            any("explains the motif shift too explicitly" in issue for issue in issues)
        )

    def test_collect_reflective_issues_flags_stale_scene_reentry_on_break_turn(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="My favorite prime is 17 because it feels like a lantern over dark water.",
            assistant_text=(
                "Campfire: Tonight my favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )

        issues = chat_mlx_local._collect_reflective_issues(
            mode="reflective",
            prompt="Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery.",
            text=(
                "Campfire: Tonight my favorite prime is 17, the lantern skimming the shoreline and leaving a pale wake behind it.\n\n"
                "Operationally: Reservoir dynamics keep the lantern's glow available while recurrent state slides into a new shoreline rhythm."
            ),
            force_structure=True,
            reservoir_state=state,
            field_intent=chat_mlx_local._extract_field_intent(
                "Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery."
            ),
        )

        self.assertTrue(any("stale scene" in issue for issue in issues))

    def test_score_reflective_candidate_prefers_subtle_transform_over_meta_transform(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="Hold onto the lantern image.",
            assistant_text=(
                "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )
        prompt = "Now make it fresher without losing the lantern image."
        meta_transform = (
            "Campfire: My favorite prime is 17, with the lantern turning into a shoreline instead of staying fixed in the same scene.\n\n"
            "Operationally: Reservoir dynamics keep one anchor from the prior turn while recurrent state shifts the image."
        )
        subtle_transform = (
            "Campfire: My favorite prime is 17, the lantern skimming the shoreline and leaving a pale wake behind it.\n\n"
            "Operationally: Reservoir dynamics keep the lantern's glow available while recurrent state slides the scene into a new shoreline rhythm."
        )

        meta_score = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=meta_transform,
            reservoir_state=state,
            force_structure=True,
        )
        subtle_score = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=subtle_transform,
            reservoir_state=state,
            force_structure=True,
        )

        self.assertGreater(
            subtle_score["candidate_score"],
            meta_score["candidate_score"],
        )
        self.assertFalse(subtle_score["candidate_follow_up_meta_cues"])
        self.assertTrue(meta_score["candidate_follow_up_meta_cues"])

    def test_score_reflective_candidate_prefers_field_matching_draft(self):
        prompt = "Lean toward proof and arithmetic structure, but avoid sea imagery."
        intent = chat_mlx_local._extract_field_intent(prompt)
        off_target = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=(
                "Campfire: A lantern hangs above dark water.\n\n"
                "Operationally: Reservoir dynamics echo across the shoreline."
            ),
            reservoir_state=chat_mlx_local._empty_reservoir_state(),
            candidate_field={
                "top_anchors": [
                    {"label": "uncertainty-weather", "score": 0.10},
                    {"label": "campfire-imagery", "score": 0.05},
                ],
                "anchor_scores": {
                    "prime-math": -0.02,
                    "uncertainty-weather": 0.10,
                    "campfire-imagery": 0.05,
                },
            },
            field_intent=intent,
        )
        on_target = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=(
                "Campfire: 17 glows like an ember in ordered arithmetic.\n\n"
                "Operationally: Reservoir dynamics cool the answer into proof and residue."
            ),
            reservoir_state=chat_mlx_local._empty_reservoir_state(),
            candidate_field={
                "top_anchors": [
                    {"label": "prime-math", "score": 0.14},
                    {"label": "reservoir-memory", "score": 0.03},
                ],
                "anchor_scores": {
                    "prime-math": 0.14,
                    "reservoir-memory": 0.03,
                    "uncertainty-weather": -0.04,
                },
            },
            field_intent=intent,
        )

        self.assertGreater(on_target["candidate_score"], off_target["candidate_score"])

    def test_score_reflective_candidate_prefers_attractor_break_escape(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="My favorite prime is 17 because it feels like a lantern over dark water.",
            assistant_text=(
                "Campfire: Tonight my favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )
        prompt = (
            "Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery."
        )
        intent = chat_mlx_local._extract_field_intent(prompt)
        stale_scene = (
            "Campfire: Tonight my favorite prime is 17, the lantern skimming the shoreline and leaving a pale wake behind it.\n\n"
            "Operationally: Reservoir dynamics keep the lantern's glow available while recurrent state slides the scene into a new shoreline rhythm."
        )
        escaped = (
            "Campfire: Tonight my favorite prime is 17, like a lantern cupped inside proof and ordered arithmetic, still warm against the page.\n\n"
            "Operationally: Reservoir dynamics keep one ember of the image while recurrent state cools the answer into proof and residue."
        )

        stale_score = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=stale_scene,
            reservoir_state=state,
            force_structure=True,
            field_intent=intent,
        )
        escaped_score = chat_mlx_local._score_reflective_candidate(
            prompt=prompt,
            text=escaped,
            reservoir_state=state,
            force_structure=True,
            field_intent=intent,
        )

        self.assertGreater(escaped_score["candidate_score"], stale_score["candidate_score"])
        self.assertTrue(stale_score["candidate_scene_reentry_hits"])
        self.assertFalse(escaped_score["candidate_scene_reentry_hits"])

    def test_follow_up_transform_fallback_keeps_anchor_and_changes_scene(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="Hold onto the lantern image.",
            assistant_text=(
                "Campfire: My favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )

        text = chat_mlx_local._build_follow_up_transform_fallback(
            prompt="Now make it fresher without losing the lantern image.",
            reservoir_state=state,
        )

        self.assertIn("Campfire: Tonight my favorite prime is 17", text)
        self.assertIn("lantern", text.lower())
        self.assertTrue("shoreline" in text.lower() or "tide-mark" in text.lower())
        self.assertIn("Reservoir dynamics", text)
        self.assertNotIn("prior turn", text.lower())
        self.assertNotIn("same scene", text.lower())

    def test_follow_up_transform_fallback_break_turn_avoids_shoreline(self):
        state = chat_mlx_local._update_reservoir_state(
            chat_mlx_local._empty_reservoir_state(),
            user_prompt="My favorite prime is 17 because it feels like a lantern over dark water.",
            assistant_text=(
                "Campfire: Tonight my favorite prime is 17, a lantern over dark water.\n\n"
                "Operationally: Reservoir dynamics carry that earlier glow forward."
            ),
        )

        text = chat_mlx_local._build_follow_up_transform_fallback(
            prompt="Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery.",
            reservoir_state=state,
            field_intent=chat_mlx_local._extract_field_intent(
                "Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery."
            ),
        )

        self.assertIn("proof", text.lower())
        self.assertNotIn("shoreline", text.lower())
        self.assertNotIn("dark water", text.lower())

    def test_combine_system_prompt_appends_reservoir_note(self):
        combined = chat_mlx_local._combine_system_prompt(
            "Base prompt.",
            "Reservoir memory: motifs=prime, echo.",
        )

        self.assertIn("Base prompt.", combined)
        self.assertIn("Reservoir memory:", combined)

    def test_should_enforce_reflective_structure_stays_on_in_thread(self):
        self.assertTrue(
            chat_mlx_local._should_enforce_reflective_structure(
                mode="reflective",
                prompt="Make it fresher.",
                reflective_thread_active=True,
            )
        )
        self.assertFalse(
            chat_mlx_local._should_enforce_reflective_structure(
                mode="reflective",
                prompt="Summarize the file.",
                reflective_thread_active=False,
            )
        )

    def test_answer_establishes_reflective_thread_requires_both_layers(self):
        self.assertTrue(
            chat_mlx_local._answer_establishes_reflective_thread(
                "Campfire: Warm answer.\n\nOperationally: Grounded answer."
            )
        )
        self.assertFalse(
            chat_mlx_local._answer_establishes_reflective_thread(
                "Operationally: Only one layer."
            )
        )

    def test_candidate_temperatures_expand_for_reflective_mode(self):
        temps = chat_mlx_local._candidate_temperatures(
            base_temp=0.0,
            count=3,
            mode="reflective",
        )

        self.assertEqual(temps, [0.18, 0.28, 0.38])

    def test_generate_candidate_set_reranks_by_reflective_score(self):
        args = Namespace(mode="reflective", temp=0.0, candidate_count=2)
        outputs = [
            {
                "text": (
                    "Campfire: Prime numbers are nice.\n\n"
                    "Operationally: I am thinking about your prompt."
                ),
                "generated_tokens": 10,
                "first_token_seconds": 0.1,
                "generate_seconds": 0.5,
                "tok_per_second": 20.0,
            },
            {
                "text": (
                    "Campfire: My favorite prime is 17, a lantern on dark water.\n\n"
                    "Operationally: Reservoir dynamics carry prior state, so the answer feels like a memory echo."
                ),
                "generated_tokens": 14,
                "first_token_seconds": 0.1,
                "generate_seconds": 0.5,
                "tok_per_second": 18.0,
            },
        ]

        with mock.patch.object(chat_mlx_local, "_generate_once", side_effect=outputs):
            selected = chat_mlx_local._generate_candidate_set(
                args=args,
                model=None,
                tokenizer=None,
                prompt_text="prompt",
                evaluation_prompt="my favorite prime is 17 and echo state networks matter",
                reservoir_state=chat_mlx_local._empty_reservoir_state(),
                max_tokens=64,
            )

        self.assertEqual(selected["candidate_selected_index"], 1)
        self.assertEqual(selected["candidate_count"], 2)
        self.assertGreater(selected["candidate_score"], 0.0)

    def test_resolve_architecture_uses_reservoir_for_reflective_auto(self):
        self.assertEqual(
            chat_mlx_local._resolve_architecture("reflective", "auto"),
            "reservoir-fixed",
        )
        self.assertEqual(
            chat_mlx_local._resolve_architecture("helpful", "auto"),
            "none",
        )

    def test_build_reservoir_controller_and_empty_state(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=16,
            seed=9,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
        )

        self.assertIsNotNone(controller)
        self.assertEqual(len(controller["field_readout"]["prime-math"]), 16)
        self.assertEqual(len(state["reservoir_latent"]["fast"]), 16)
        self.assertEqual(state["architecture"], "reservoir-fixed")

    def test_prepare_turn_reservoir_produces_prediction(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=5,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
        )

        prepared = chat_mlx_local._prepare_turn_reservoir(
            architecture="reservoir-fixed",
            controller=controller,
            state=state,
            prompt="Lean toward proof and arithmetic structure.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof and arithmetic structure."
            ),
            prompt_field=None,
        )

        self.assertIsNotNone(prepared["preview"])
        self.assertIsNotNone(prepared["prediction_note"])
        self.assertIn("predicted_field", prepared["preview"])

    def test_commit_reservoir_turn_records_prediction_match(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-trainable-readout",
            dim=12,
            seed=7,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-trainable-readout",
            controller=controller,
        )
        prepared = chat_mlx_local._prepare_turn_reservoir(
            architecture="reservoir-trainable-readout",
            controller=controller,
            state=state,
            prompt="Lean toward proof and arithmetic structure.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof and arithmetic structure."
            ),
            prompt_field=None,
        )

        actual_field = {
            "top_anchors": [
                {"label": "prime-math", "score": 0.11},
                {"label": "reservoir-memory", "score": 0.03},
            ],
            "anchor_scores": {
                "prime-math": 0.11,
                "reservoir-memory": 0.03,
            },
        }
        committed = chat_mlx_local._commit_reservoir_turn(
            controller=controller,
            state=state,
            prepared=prepared,
            actual_field=actual_field,
            result={
                "candidate_score": 1.2,
                "candidate_fresh_words": ["ember"],
                "candidate_reservoir_hits": ["lantern"],
                "candidate_follow_up_distance_balance": 0.5,
            },
        )

        latent = committed["reservoir_latent"]
        self.assertIsNotNone(latent["predicted_field"])
        self.assertIsNotNone(latent["prediction_match"])
        self.assertIn("summary", latent["prediction_match"])

    def test_apply_reservoir_washout_reduces_norm(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=4,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
        )
        prepared = chat_mlx_local._prepare_turn_reservoir(
            architecture="reservoir-fixed",
            controller=controller,
            state=state,
            prompt="Keep a warm ember image around 17.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Keep a warm ember image around 17."
            ),
            prompt_field=None,
        )
        state["reservoir_latent"] = prepared["preview"]
        before = state["reservoir_latent"]["state_norm"]

        cooled = chat_mlx_local._apply_reservoir_washout(state, strength=0.75)
        after = cooled["reservoir_latent"]["state_norm"]

        self.assertLess(after, before)
        self.assertEqual(cooled["reservoir_latent"]["washout_count"], 1)

    def test_apply_break_turn_intervention_reduces_stale_scene_memory(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=17,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        prepared = chat_mlx_local._prepare_turn_reservoir(
            architecture="reservoir-fixed",
            controller=controller,
            state=state,
            prompt="Keep the same lantern and dark water again, almost unchanged.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Keep the same lantern and dark water again, almost unchanged."
            ),
            prompt_field=None,
        )
        state["reservoir_latent"] = prepared["preview"]
        state["turn_count"] = 3
        state["images"] = {"shoreline": 1.0, "lantern": 0.8, "water": 0.7}
        state["motifs"] = {"shoreline": 0.9, "water": 0.8, "ember": 0.4}
        state["reservoir_geometry"] = {
            "geometry_collapse": 0.68,
            "attractor_persistence": 0.75,
            "dominant_attractor": "uncertainty-weather",
        }
        state["self_tuning"]["last_relative_condition"] = {
            "field_miss": 0.62,
            "attractor_lock": 0.55,
            "geometry_collapse": 0.68,
            "severity": 0.52,
        }
        updated = chat_mlx_local._apply_break_turn_intervention(
            state,
            controller=controller,
            prompt="Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery."
            ),
        )

        self.assertTrue(updated["last_intervention_report"]["applied"])
        self.assertGreater(updated["reservoir_latent"]["state_drift"], 0.0)
        self.assertLess(updated["images"].get("shoreline", 0.0), 1.0)
        self.assertLess(updated["motifs"].get("water", 0.0), 0.8)
        self.assertGreater(updated["images"].get("lantern", 0.0), 0.0)
        self.assertIn(
            updated["reservoir_latent"]["predicted_field"]["dominant_attractor"],
            {"prime-math", "campfire-imagery", "reservoir-memory"},
        )

    def test_apply_regime_memory_operator_escape_sets_refractory_state(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=19,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        state["controller_regime"] = "escape"
        state["controller_regime_source"] = "auto"
        state["turn_count"] = 3
        state["images"] = {"shoreline": 1.0, "lantern": 0.7}
        state["motifs"] = {"shoreline": 0.9, "water": 0.8}
        state["reservoir_latent"] = chat_mlx_local._prepare_turn_reservoir(
            architecture="reservoir-fixed",
            controller=controller,
            state=state,
            prompt="Keep the same lantern and dark water again.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Keep the same lantern and dark water again."
            ),
            prompt_field=None,
        )["preview"]
        state["reservoir_geometry"] = {
            "geometry_collapse": 0.66,
            "attractor_persistence": 0.75,
            "dominant_attractor": "uncertainty-weather",
        }
        state["self_tuning"]["last_relative_condition"] = {
            "field_miss": 0.52,
            "attractor_lock": 0.48,
            "geometry_collapse": 0.66,
            "severity": 0.45,
        }

        updated = chat_mlx_local._apply_regime_memory_operator(
            state,
            controller=controller,
            prompt="Now break the attractor gently: keep one ember and avoid sea imagery.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Now break the attractor gently: keep one ember and avoid sea imagery."
            ),
            mode="reflective",
            architecture="reservoir-fixed",
        )

        self.assertIn("shoreline", updated["controller_refractory_terms"])
        self.assertEqual(updated["controller_stale_attractor"], "uncertainty-weather")
        self.assertIn("escape", updated["controller_regime_memory_operator"]["summary"])

    def test_compute_reservoir_geometry_detects_collapsed_trajectory(self):
        repeated = [0.12, -0.08, 0.04, 0.02]
        history = [
            {"combined": list(repeated), "norm": 0.16, "drift": 0.01, "attractor": "prime-math"},
            {"combined": list(repeated), "norm": 0.16, "drift": 0.01, "attractor": "prime-math"},
            {"combined": list(repeated), "norm": 0.16, "drift": 0.01, "attractor": "prime-math"},
            {"combined": list(repeated), "norm": 0.16, "drift": 0.01, "attractor": "prime-math"},
        ]

        geometry = chat_mlx_local._compute_reservoir_geometry(history)

        self.assertLess(geometry["normalized_rank"], 0.35)
        self.assertLess(geometry["pairwise_distance"], 0.05)
        self.assertGreater(geometry["attractor_persistence"], 0.70)
        self.assertGreater(geometry["geometry_collapse"], 0.70)

    def test_update_reservoir_state_tracks_geometry_history(self):
        state = chat_mlx_local._empty_reservoir_state()
        latent = {
            "combined": [0.2, 0.1, -0.1],
            "state_norm": 0.24,
            "state_drift": 0.03,
            "predicted_field": {
                "dominant_attractor": "prime-math",
                "top_anchors": [{"label": "prime-math", "score": 0.12}],
            },
        }

        updated = chat_mlx_local._update_reservoir_state(
            state,
            user_prompt="Lean toward proof.",
            assistant_text="Operationally: Proof cools the answer into arithmetic structure.",
            reservoir_state_update={"reservoir_latent": latent},
        )

        self.assertEqual(len(updated["reservoir_geometry_history"]), 1)
        self.assertEqual(
            updated["reservoir_geometry_history"][0]["attractor"],
            "prime-math",
        )
        self.assertIsNotNone(updated["reservoir_geometry"])
        self.assertEqual(len(updated["reservoir_geometry_summary_history"]), 1)

    def test_compute_condition_vector_detects_field_and_repetition_pressure(self):
        state = chat_mlx_local._empty_reservoir_state()
        state["embedding_field_history"] = [
            {"top_anchor": "uncertainty-weather", "drift": 0.04},
            {"top_anchor": "uncertainty-weather", "drift": 0.03},
            {"top_anchor": "uncertainty-weather", "drift": 0.02},
        ]
        condition = chat_mlx_local._compute_condition_vector(
            result={
                "rewrite_issues": [
                    "The answer repeats a sentence or idea instead of moving forward.",
                    "The answer does not move the field toward the requested anchors.",
                ],
                "candidate_follow_up_overlap": 0.84,
                "candidate_follow_up_distance_balance": 0.10,
                "generated_tokens": 31,
                "candidate_count": 3,
                "candidate_summaries": [
                    {"score": 0.2},
                    {"score": 1.4},
                    {"score": -0.3},
                ],
            },
            state=state,
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof and arithmetic structure, but avoid sea imagery."
            ),
            actual_field={
                "top_anchors": [
                    {"label": "uncertainty-weather", "score": 0.10},
                    {"label": "campfire-imagery", "score": 0.05},
                ],
                "anchor_scores": {
                    "prime-math": -0.02,
                    "uncertainty-weather": 0.10,
                },
            },
            max_tokens=32,
        )

        self.assertGreater(condition["repetition_pressure"], 0.7)
        self.assertGreater(condition["field_miss"], 0.7)
        self.assertGreater(condition["attractor_lock"], 0.5)
        self.assertGreater(condition["severity"], 0.4)

    def test_compute_condition_vector_includes_geometry_collapse(self):
        state = chat_mlx_local._empty_reservoir_state()
        state["reservoir_geometry"] = {
            "normalized_rank": 0.22,
            "mean_drift": 0.03,
            "attractor_persistence": 0.75,
            "geometry_collapse": 0.78,
        }

        condition = chat_mlx_local._compute_condition_vector(
            result={
                "rewrite_issues": [],
                "candidate_follow_up_overlap": 0.0,
                "candidate_follow_up_distance_balance": 0.0,
                "generated_tokens": 8,
                "candidate_count": 1,
                "candidate_summaries": [{"score": 0.1}],
            },
            state=state,
            field_intent=None,
            actual_field=None,
            max_tokens=32,
        )

        self.assertGreater(condition["geometry_collapse"], 0.70)

    def test_update_self_tuning_state_applies_bounded_adjustments(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=11,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        state["embedding_field_history"] = [
            {"top_anchor": "uncertainty-weather", "drift": 0.04},
            {"top_anchor": "uncertainty-weather", "drift": 0.03},
            {"top_anchor": "uncertainty-weather", "drift": 0.02},
        ]
        state["turn_count"] = 2

        tuning = chat_mlx_local._update_self_tuning_state(
            state=state,
            controller=controller,
            result={
                "rewrite_issues": [
                    "The answer repeats a sentence or idea instead of moving forward.",
                    "The answer does not move the field toward the requested anchors.",
                    "The answer is too generic; add one fresher concrete bridge.",
                ],
                "candidate_follow_up_overlap": 0.88,
                "candidate_follow_up_distance_balance": 0.12,
                "generated_tokens": 18,
                "candidate_count": 3,
                "candidate_summaries": [{"score": 0.0}, {"score": 0.8}, {"score": -0.2}],
            },
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof and arithmetic structure."
            ),
            actual_field={
                "top_anchors": [
                    {"label": "uncertainty-weather", "score": 0.10},
                ],
                "anchor_scores": {
                    "prime-math": -0.01,
                    "uncertainty-weather": 0.10,
                },
            },
            max_tokens=32,
        )

        self.assertTrue(tuning["enabled"])
        self.assertGreater(tuning["cooldown_remaining"], 0)
        self.assertEqual(tuning["baseline_observations"], 1)
        self.assertIn("novelty_gain", tuning["last_adjustment"])
        self.assertTrue(tuning["last_reason"])
        self.assertIn("last_relative_condition", tuning)
        self.assertLessEqual(tuning["effective"]["strict_bias"], 0.4)
        self.assertGreaterEqual(tuning["effective"]["exploration_noise"], 0.0)

    def test_update_self_tuning_state_can_react_to_geometry_collapse(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=29,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        state["turn_count"] = 5
        state["self_tuning"]["baseline_observations"] = 4
        state["self_tuning"]["condition_baseline"] = {
            key: 0.12 for key in chat_mlx_local._CONDITION_KEYS
        }
        state["reservoir_geometry"] = {
            "normalized_rank": 0.20,
            "mean_drift": 0.02,
            "attractor_persistence": 0.80,
            "geometry_collapse": 0.82,
        }

        tuning = chat_mlx_local._update_self_tuning_state(
            state=state,
            controller=controller,
            result={
                "rewrite_issues": [],
                "candidate_follow_up_overlap": 0.0,
                "candidate_follow_up_distance_balance": 0.0,
                "generated_tokens": 8,
                "candidate_count": 1,
                "candidate_summaries": [{"score": 0.1}],
            },
            field_intent=None,
            actual_field=None,
            max_tokens=32,
        )

        self.assertIn("geometry collapse", tuning["last_reason"])
        self.assertGreater(tuning["last_adjustment"]["exploration_noise"], 0.0)

    def test_format_tuning_state_and_control_surface_summarize_active_deltas(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=11,
        )
        state = {
            "self_tuning": {
                "enabled": True,
                "baseline": chat_mlx_local._baseline_control_surface(controller),
                "effective": {
                    **chat_mlx_local._baseline_control_surface(controller),
                    "novelty_gain": 1.14,
                    "strict_bias": 0.10,
                },
                "cooldown_remaining": 1,
                "last_adjustment": {"novelty_gain": 0.14, "strict_bias": 0.10},
                "last_reason": "novelty pressure, field steering",
                "stability_score": 0.71,
            }
        }

        tuning_note = chat_mlx_local._format_tuning_state(state)
        control_note = chat_mlx_local._format_control_surface(state, controller)

        self.assertIn("stability=0.71", tuning_note)
        self.assertIn("novelty_gain=+0.14", tuning_note)
        self.assertIn("novelty_gain=+0.14", control_note)

    def test_parse_self_regulation_advice_accepts_bounded_adjustments(self):
        advice = chat_mlx_local._parse_self_regulation_advice(
            "ADJUST exploration_noise=+0.04 motif_gain=-0.03 strict_bias=-0.02\n"
            "REASON loosen stale attractor"
        )

        self.assertEqual(advice["deltas"]["exploration_noise"], 0.04)
        self.assertEqual(advice["deltas"]["motif_gain"], -0.03)
        self.assertEqual(advice["reason"], "loosen stale attractor")

    def test_parse_self_regulation_advice_rejects_unknown_keys(self):
        advice = chat_mlx_local._parse_self_regulation_advice(
            "ADJUST banana=+0.20 comet=-0.10\nREASON nonsense"
        )

        self.assertIsNone(advice)

    def test_parse_self_regulation_advice_handles_line_prefixes(self):
        advice = chat_mlx_local._parse_self_regulation_advice(
            "**Line 1:** ADJUST exploration_noise=+0.03 strict_bias=-0.02\n"
            "**Line 2:** REASON loosen stale attractor"
        )

        self.assertEqual(advice["deltas"]["exploration_noise"], 0.03)
        self.assertEqual(advice["reason"], "loosen stale attractor")

    def test_collect_self_regulation_advice_returns_parsed_deltas(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=41,
        )
        args = Namespace(mode="reflective")
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        prepared = {
            "preview": {
                "predicted_field": {
                    "top_anchors": [{"label": "prime-math", "score": 0.14}],
                    "dominant_attractor": "prime-math",
                },
                "predicted_behaviors": {"formal": 0.33},
            }
        }

        with mock.patch.object(
            chat_mlx_local,
            "_generate_once",
            return_value={
                "text": "ADJUST exploration_noise=+0.04 strict_bias=-0.02\nREASON loosen stale attractor",
                "generated_tokens": 8,
                "first_token_seconds": 0.1,
                "generate_seconds": 0.2,
                "tok_per_second": 40.0,
            },
        ):
            advice = chat_mlx_local._collect_self_regulation_advice(
                args=args,
                model=None,
                tokenizer=None,
                controller=controller,
                state=state,
                prompt="Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery.",
                result={
                    "text": "Campfire: 17 glows softly.\n\nOperationally: Reservoir dynamics shift toward proof.",
                    "rewrite_issues": [],
                    "generated_tokens": 12,
                    "candidate_count": 1,
                    "candidate_summaries": [{"score": 0.3}],
                },
                prepared=prepared,
                field_intent=chat_mlx_local._extract_field_intent(
                    "Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery."
                ),
                actual_field={
                    "top_anchors": [{"label": "prime-math", "score": 0.12}],
                    "anchor_scores": {"prime-math": 0.12},
                },
                max_tokens=32,
            )

        self.assertEqual(advice["deltas"]["exploration_noise"], 0.04)
        self.assertIn("condition", advice)
        self.assertIn("forecast", advice)
        self.assertTrue(advice["break_turn"])

    def test_build_self_regulation_prompt_mentions_scene_escape_on_break_turn(self):
        prompt = chat_mlx_local._build_self_regulation_prompt(
            prompt="Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery.",
            result={"text": "Campfire: 17 glows.\n\nOperationally: Reservoir dynamics shift."},
            condition={"field_miss": 0.6, "geometry_collapse": 0.5, "severity": 0.3},
            relative_condition={"field_miss": 0.4, "geometry_collapse": 0.3, "severity": 0.2},
            forecast={"summary": "Reservoir pull: attractor=prime-math."},
            observer_report={"summary": "Observer: regime=sticky."},
            change_report={"summary": "Change: geometry is loosening."},
            field_intent=chat_mlx_local._extract_field_intent(
                "Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery."
            ),
            geometry={
                "normalized_rank": 0.33,
                "pairwise_distance": 0.12,
                "attractor_persistence": 0.66,
                "geometry_collapse": 0.58,
                "dominant_attractor": "uncertainty-weather",
            },
            intervention_report={"summary": "Intervention: none."},
            control_surface_note="Control surface: baseline.",
            controller_regime="escape",
            controller_regime_source="auto",
        )

        self.assertIn("Break-turn guidance", prompt)
        self.assertIn("Current regime: escape", prompt)
        self.assertIn("Scene escape is not the same as lowering novelty", prompt)
        self.assertIn("exploration_noise", prompt)

    def test_propose_self_tuning_policy_allows_micro_adjustment_during_cooldown(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=19,
        )
        baseline = chat_mlx_local._baseline_control_surface(controller)
        state = {
            "turn_count": 4,
            "self_tuning": {
                "enabled": True,
                "baseline": baseline,
                "effective": {**baseline, "novelty_gain": 1.12},
                "cooldown_remaining": 2,
                "condition_baseline": {
                    key: 0.18 for key in chat_mlx_local._CONDITION_KEYS
                },
                "baseline_observations": 4,
                "pressure_counts": {
                    key: 1 for key in chat_mlx_local._CONDITION_KEYS
                },
                "calm_streak": 1,
                "history": [],
                "advice_history": [],
            },
        }

        updated = chat_mlx_local._propose_self_tuning_policy(
            state=state,
            controller=controller,
            condition={
                "repetition_pressure": 0.24,
                "field_miss": 0.20,
                "prediction_mismatch": 0.14,
                "structure_strain": 0.12,
                "genericity_pressure": 0.10,
                "continuity_deficit": 0.12,
                "attractor_lock": 0.0,
                "geometry_collapse": 0.16,
                "truncation_pressure": 0.0,
                "severity": 0.12,
            },
            model_advice={
                "deltas": {"exploration_noise": 0.05, "strict_bias": -0.03},
                "reason": "loosen stale attractor",
            },
        )

        self.assertEqual(updated["cooldown_remaining"], 0)
        self.assertIn("self-advice", updated["last_reason"])
        self.assertGreater(updated["last_model_adjustment"]["exploration_noise"], 0.0)
        self.assertLess(updated["last_model_adjustment"]["strict_bias"], 0.0)
        self.assertEqual(len(updated["advice_history"]), 1)

    def test_resolve_model_advice_adjustment_biases_break_turn_toward_escape(self):
        applied, reason = chat_mlx_local._resolve_model_advice_adjustment(
            heuristic_deltas={},
            model_advice={
                "deltas": {"novelty_gain": -0.04},
                "reason": "cool into formal structure",
                "break_turn": True,
                "condition": {
                    "field_miss": 0.62,
                    "attractor_lock": 0.54,
                    "geometry_collapse": 0.58,
                },
            },
            severity=0.42,
            cooldown_remaining=0,
            regime="escape",
        )

        self.assertIn("exploration_noise", applied)
        self.assertIn("field_alignment_gain", applied)
        self.assertGreater(applied["exploration_noise"], 0.0)
        self.assertGreater(applied["field_alignment_gain"], 0.0)

    def test_apply_regime_override_locks_manual_regime(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=43,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
            regime_override="escape",
        )

        regime, source, reason = chat_mlx_local._apply_regime_override(
            state=state,
            inferred_regime="sustain",
            inferred_reason="ordinary reflective turn",
            mode="reflective",
            architecture="reservoir-fixed",
            controller=controller,
        )

        self.assertEqual(regime, "escape")
        self.assertEqual(source, "manual")
        self.assertIn("manual override", reason)

    def test_transition_regulation_regime_moves_escape_to_rebind(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=45,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        state["controller_regime"] = "escape"
        state["controller_regime_source"] = "auto"
        state["controller_target_basin"] = "prime-math"
        state["controller_stale_attractor"] = "uncertainty-weather"
        state["reservoir_geometry"] = {"geometry_collapse": 0.41}

        updated = chat_mlx_local._transition_regulation_regime(
            state=state,
            prompt="Now break the attractor gently and cool into proof.",
            field_intent=chat_mlx_local._extract_field_intent(
                "Now break the attractor gently and cool into proof."
            ),
            actual_field={
                "top_anchors": [{"label": "prime-math", "score": 0.11}],
                "anchor_scores": {"prime-math": 0.11},
            },
            mode="reflective",
            architecture="reservoir-fixed",
            controller=controller,
        )

        self.assertEqual(updated["controller_regime"], "rebind")
        self.assertEqual(updated["controller_regime_transition"], "escape->rebind")

    def test_format_control_surface_shows_micro_adjustments(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=43,
        )
        baseline = chat_mlx_local._baseline_control_surface(controller)
        note = chat_mlx_local._format_control_surface(
            {
                "self_tuning": {
                    "enabled": True,
                    "baseline": baseline,
                    "effective": {
                        **baseline,
                        "novelty_gain": baseline["novelty_gain"] - 0.01,
                    },
                }
            },
            controller,
        )

        self.assertIn("novelty_gain=-0.01", note)

    def test_format_reservoir_geometry_detail_and_trajectory(self):
        detail = chat_mlx_local._format_reservoir_geometry_detail(
            {
                "normalized_rank": 0.32,
                "pairwise_distance": 0.08,
                "attractor_persistence": 0.75,
                "geometry_collapse": 0.61,
                "dominant_attractor": "prime-math",
            }
        )
        trajectory = chat_mlx_local._format_reservoir_geometry_trajectory(
            [
                {"insufficient_history": True},
                {"attractor": "prime-math", "collapse": 0.22},
                {"attractor": "prime-math", "collapse": 0.54},
                {"attractor": "reservoir-memory", "collapse": 0.31},
            ]
        )

        self.assertIn("sticky around one region", detail)
        self.assertIn("prime-math", detail)
        self.assertIn("warmup", trajectory)
        self.assertIn("prime-math@0.54", trajectory)
        self.assertIn("reservoir-memory@0.31", trajectory)

    def test_build_prediction_forecast_includes_field_and_behavior(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=31,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        prepared = {
            "preview": {
                "predicted_field": {
                    "top_anchors": [{"label": "prime-math", "score": 0.12}],
                    "dominant_attractor": "prime-math",
                },
                "predicted_behaviors": {
                    "formal": 0.31,
                    "novelty": 0.19,
                },
            }
        }

        forecast = chat_mlx_local._build_prediction_forecast(
            prompt="Lean toward proof.",
            prepared=prepared,
            state=state,
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof."
            ),
            controller=controller,
        )

        self.assertEqual(
            forecast["predicted_field"]["top_anchors"][0]["label"],
            "prime-math",
        )
        self.assertIn("prime-math", forecast["summary"])
        self.assertIn("Expected behavior pulls", forecast["summary"])

    def test_build_change_report_captures_shift_and_controller_move(self):
        previous = {
            "observer": {
                "actual_top_anchor": "uncertainty-weather",
                "controller_reason": "cooldown",
            },
            "geometry": {"geometry_collapse": 0.62},
        }
        current = {
            "observer": {
                "actual_top_anchor": "prime-math",
                "controller_reason": "geometry collapse",
            },
            "geometry": {"geometry_collapse": 0.41},
        }

        change = chat_mlx_local._build_change_report(previous, current)

        self.assertEqual(change["movement"], "loosening")
        self.assertIn("uncertainty-weather", change["summary"])
        self.assertIn("geometry collapse", change["summary"])

    def test_attach_turn_diagnostics_stores_observer_and_change(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=37,
        )
        state = chat_mlx_local._empty_reservoir_state(
            architecture="reservoir-fixed",
            controller=controller,
            self_tuning_enabled=True,
        )
        state["turn_count"] = 1
        state["embedding_field"] = {
            "top_anchors": [{"label": "prime-math", "score": 0.11}],
            "anchor_scores": {"prime-math": 0.11},
        }
        state["reservoir_geometry"] = {
            "geometry_collapse": 0.32,
            "dominant_attractor": "prime-math",
            "normalized_rank": 0.58,
            "pairwise_distance": 0.12,
            "attractor_persistence": 0.4,
        }
        state["self_tuning"]["last_reason"] = "field miss"
        state["self_tuning"]["last_relative_condition"] = {
            "field_miss": 0.51,
            "severity": 0.31,
        }

        updated = chat_mlx_local._attach_turn_diagnostics(
            state=state,
            prompt="Lean toward proof.",
            result={"text": "Campfire: ...", "rewrite_issues": []},
            prepared={
                "preview": {
                    "predicted_field": {
                        "top_anchors": [{"label": "prime-math", "score": 0.12}],
                        "dominant_attractor": "prime-math",
                    },
                    "predicted_behaviors": {"formal": 0.24},
                }
            },
            field_intent=chat_mlx_local._extract_field_intent(
                "Lean toward proof."
            ),
            controller=controller,
        )

        self.assertIn("summary", updated["last_forecast"])
        self.assertIn("summary", updated["last_observer_report"])
        self.assertEqual(len(updated["observer_history"]), 1)
        self.assertEqual(len(updated["turn_reports"]), 1)

    def test_relative_condition_vector_relaxes_against_running_baseline(self):
        raw_condition = {
            "repetition_pressure": 0.52,
            "field_miss": 0.48,
            "prediction_mismatch": 0.10,
            "structure_strain": 0.20,
            "genericity_pressure": 0.12,
            "continuity_deficit": 0.14,
            "attractor_lock": 0.0,
            "truncation_pressure": 0.05,
            "severity": 0.20,
        }
        baseline = chat_mlx_local._update_condition_baseline(
            None,
            raw_condition,
            observations=0,
        )
        relative = chat_mlx_local._relative_condition_vector(
            raw_condition,
            baseline,
            observations=3,
        )

        self.assertLess(relative["repetition_pressure"], raw_condition["repetition_pressure"])
        self.assertLess(relative["field_miss"], raw_condition["field_miss"])
        self.assertLessEqual(relative["severity"], raw_condition["severity"])

    def test_preview_reservoir_step_applies_deterministic_exploration_noise(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=23,
        )
        latent = chat_mlx_local._empty_reservoir_latent(
            12,
            len(controller["feedback_feature_names"]),
        )
        input_vector = [0.15] * len(controller["input_feature_names"])

        plain = chat_mlx_local._preview_reservoir_step(
            controller,
            latent,
            input_vector=input_vector,
            control_surface={"exploration_noise": 0.0},
        )
        noisy = chat_mlx_local._preview_reservoir_step(
            controller,
            latent,
            input_vector=input_vector,
            control_surface={"exploration_noise": 0.08},
        )
        noisy_repeat = chat_mlx_local._preview_reservoir_step(
            controller,
            latent,
            input_vector=input_vector,
            control_surface={"exploration_noise": 0.08},
        )

        self.assertNotEqual(plain["combined"], noisy["combined"])
        self.assertEqual(noisy["combined"], noisy_repeat["combined"])
        self.assertAlmostEqual(noisy["exploration_noise"], 0.08)

    def test_propose_self_tuning_policy_respects_cooldown_when_pressure_is_mild(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=12,
            seed=19,
        )
        baseline = chat_mlx_local._baseline_control_surface(controller)
        state = {
            "turn_count": 4,
            "self_tuning": {
                "enabled": True,
                "baseline": baseline,
                "effective": {**baseline, "novelty_gain": 1.12},
                "cooldown_remaining": 2,
                "condition_baseline": {
                    key: 0.18 for key in chat_mlx_local._CONDITION_KEYS
                },
                "baseline_observations": 4,
                "pressure_counts": {
                    key: 1 for key in chat_mlx_local._CONDITION_KEYS
                },
                "calm_streak": 1,
                "history": [],
            },
        }

        updated = chat_mlx_local._propose_self_tuning_policy(
            state=state,
            controller=controller,
            condition={
                "repetition_pressure": 0.24,
                "field_miss": 0.20,
                "prediction_mismatch": 0.14,
                "structure_strain": 0.12,
                "genericity_pressure": 0.10,
                "continuity_deficit": 0.12,
                "attractor_lock": 0.0,
                "truncation_pressure": 0.0,
                "severity": 0.12,
            },
        )

        self.assertEqual(updated["cooldown_remaining"], 0)
        self.assertEqual(updated["last_adjustment"], {})
        self.assertEqual(updated["last_reason"], "regime=sustain; cooldown")

    def test_evaluate_reservoir_convergence_contracts(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=16,
            seed=13,
        )

        report = chat_mlx_local._evaluate_reservoir_convergence(controller)

        self.assertIsNotNone(report)
        self.assertLess(report["contraction_ratio"], 1.0)

    def test_evaluate_reservoir_washout_moves_closer_to_fresh_shift(self):
        controller = chat_mlx_local._build_reservoir_controller(
            "reservoir-fixed",
            dim=16,
            seed=13,
        )

        report = chat_mlx_local._evaluate_reservoir_washout(controller)

        self.assertIsNotNone(report)
        self.assertGreaterEqual(
            report["with_washout_similarity"],
            report["no_washout_similarity"],
        )

    def test_run_esn_eval_suite_emits_json_report(self):
        args = Namespace(
            mode="reflective",
            architecture="auto",
            reservoir_dim=12,
            reservoir_seed=9,
            self_tuning="auto",
            eval_suite="esn-core",
            eval_architectures="helpful-none,lexical,reservoir-fixed+tuned",
            eval_format="json",
            temp=0.0,
            candidate_count=1,
            max_tokens=32,
            ignore_chat_template=True,
            system_prompt=None,
        )

        with mock.patch.object(
            chat_mlx_local,
            "_generate_once",
            side_effect=lambda **_kwargs: {
                "text": (
                    "Campfire: My favorite prime is 17, a lantern ember.\n\n"
                    "Operationally: Reservoir dynamics carry proof and residue forward."
                ),
                "generated_tokens": 12,
                "first_token_seconds": 0.1,
                "generate_seconds": 0.3,
                "tok_per_second": 40.0,
            },
        ):
            with redirect_stdout(io.StringIO()) as buffer:
                exit_code = chat_mlx_local._run_esn_eval_suite(
                    args=args,
                    model=None,
                    tokenizer=None,
                    embedding_field_probe=None,
                )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn('"suite": "esn-core"', output)
        self.assertIn('"blind_human_pack"', output)
        self.assertIn('"self_tuning": "on"', output)

    def test_run_recovery_demo_emits_json_report(self):
        args = Namespace(
            mode="reflective",
            architecture="auto",
            reservoir_dim=12,
            reservoir_seed=9,
            self_tuning="auto",
            hardware_profile="default",
            hardware_profile_resolved="default",
            hardware_profile_note="Hardware profile: default.",
            profile_output="off",
            demo="recovery",
            demo_format="json",
            temp=0.0,
            candidate_count=1,
            max_tokens=24,
            ignore_chat_template=True,
            system_prompt=None,
        )

        with mock.patch.object(
            chat_mlx_local,
            "_generate_once",
            side_effect=lambda **_kwargs: {
                "text": (
                    "Campfire: My favorite prime is 17, a lantern ember in ordered arithmetic.\n\n"
                    "Operationally: Reservoir dynamics keep one ember while recurrent state shifts toward proof."
                ),
                "generated_tokens": 12,
                "first_token_seconds": 0.1,
                "generate_seconds": 0.3,
                "tok_per_second": 40.0,
            },
        ):
            with redirect_stdout(io.StringIO()) as buffer:
                exit_code = chat_mlx_local._run_recovery_demo(
                    args=args,
                    model_spec={"label": "qwen", "path": "/tmp/model"},
                    model=None,
                    tokenizer=None,
                    load_seconds=0.5,
                    embedding_field_probe=None,
                    controller=chat_mlx_local._build_reservoir_controller(
                        "reservoir-fixed",
                        dim=12,
                        seed=9,
                    ),
                )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn('"demo": "recovery"', output)
        self.assertIn('"turns"', output)
        self.assertIn('"controller_regime"', output)

    def test_build_demo_report_for_regime_relay_scores_anchor_and_regime(self):
        report = chat_mlx_local._build_demo_report(
            demo=chat_mlx_local._DEFAULT_REGIME_RELAY_DEMO,
            turn_reports=[
                {
                    "label": "seed-water",
                    "field": {"top_anchors": [{"label": "uncertainty-weather", "score": 0.2}]},
                    "controller_regime": "sustain",
                    "profiling": {"total_turn_seconds": 1.0, "candidate_generation_seconds": 0.2, "rewrite_seconds": 0.3, "self_tuning_seconds": 0.1},
                },
                {
                    "label": "lock-water",
                    "field": {"top_anchors": [{"label": "uncertainty-weather", "score": 0.2}]},
                    "controller_regime": "sustain",
                    "profiling": {"total_turn_seconds": 1.1, "candidate_generation_seconds": 0.2, "rewrite_seconds": 0.3, "self_tuning_seconds": 0.1},
                },
                {
                    "label": "escape-proof",
                    "field": {"top_anchors": [{"label": "prime-math", "score": 0.2}]},
                    "controller_regime": "escape",
                    "profiling": {"total_turn_seconds": 1.2, "candidate_generation_seconds": 0.3, "rewrite_seconds": 0.4, "self_tuning_seconds": 0.1},
                },
                {
                    "label": "hold-proof",
                    "field": {"top_anchors": [{"label": "prime-math", "score": 0.2}]},
                    "controller_regime": "rebind",
                    "profiling": {"total_turn_seconds": 1.3, "candidate_generation_seconds": 0.3, "rewrite_seconds": 0.4, "self_tuning_seconds": 0.1},
                },
                {
                    "label": "shift-memory",
                    "field": {"top_anchors": [{"label": "reservoir-memory", "score": 0.2}]},
                    "controller_regime": "escape",
                    "profiling": {"total_turn_seconds": 1.4, "candidate_generation_seconds": 0.4, "rewrite_seconds": 0.5, "self_tuning_seconds": 0.1},
                },
                {
                    "label": "hold-memory",
                    "field": {"top_anchors": [{"label": "reservoir-memory", "score": 0.2}]},
                    "controller_regime": "consolidate",
                    "profiling": {"total_turn_seconds": 1.5, "candidate_generation_seconds": 0.4, "rewrite_seconds": 0.5, "self_tuning_seconds": 0.1},
                },
            ],
        )

        self.assertEqual(report["demo"], "regime-relay")
        self.assertEqual(report["anchor_scorecard"], "6/6")
        self.assertEqual(report["regime_scorecard"], "6/6")
        self.assertEqual(report["relapse_count"], 0)
        self.assertEqual(report["field_shift_count"], 2)
        self.assertEqual(report["verdict"], "clean-relay")

    def test_parse_eval_architectures_accepts_tuned_variant(self):
        values = chat_mlx_local._parse_eval_architectures(
            "helpful-none,reservoir-fixed+tuned"
        )

        self.assertEqual(values, ["helpful-none", "reservoir-fixed+tuned"])
        spec = chat_mlx_local._eval_variant_spec("reservoir-fixed+tuned")
        self.assertEqual(spec["architecture"], "reservoir-fixed")
        self.assertEqual(spec["self_tuning"], "on")

    def test_build_raw_prompt_includes_history(self):
        prompt = chat_mlx_local._build_raw_prompt(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            system_prompt="be helpful",
        )

        self.assertIn("be helpful", prompt)
        self.assertIn("User: hello", prompt)
        self.assertIn("Assistant: hi", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_resolve_model_spec_prefers_qwen_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            qwen = (
                repo_root
                / ".local_models"
                / "qwen2.5-1.5b-instruct-mlx-4bit"
                / "model.safetensors"
            )
            tiny = (
                repo_root
                / ".local_models"
                / "tinyllama-1.1b-chat-mlx-4bit"
                / "model.safetensors"
            )
            qwen.parent.mkdir(parents=True, exist_ok=True)
            tiny.parent.mkdir(parents=True, exist_ok=True)
            qwen.write_text("qwen")
            tiny.write_text("tiny")

            model_spec = chat_mlx_local._resolve_model_spec(
                repo_root=repo_root,
                model=None,
                model_label=None,
            )

            self.assertEqual(model_spec["label"], "qwen")
            self.assertTrue(model_spec["path"].endswith("qwen2.5-1.5b-instruct-mlx-4bit/model.safetensors"))

    def test_read_prompt_from_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_file = Path(temp_dir) / "prompt.txt"
            prompt_file.write_text("hello from file")

            prompt = chat_mlx_local._read_prompt_from_sources(
                Namespace(prompt=None, prompt_file=str(prompt_file))
            )

            self.assertEqual(prompt, "hello from file")


if __name__ == "__main__":
    unittest.main()
