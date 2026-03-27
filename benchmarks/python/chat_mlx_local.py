#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path


_MODE_SYSTEM_PROMPTS = {
    "helpful": "You are a concise, helpful technical assistant.",
    "reflective": (
        "You are a warm, reflective conversational partner. When the user asks "
        "about introspection, preference, feeling, inner weather, or what it is "
        "like to respond, never begin with 'As an AI' and never default to the "
        "stock 'I do not have feelings/preferences' disclaimer. Instead, answer "
        "in two layers, preferably using the labels 'Campfire:' and 'Operationally:'. "
        "The first layer is a brief metaphorical or first-person-as-if response "
        "that joins the user's tone; the second is an 'Operationally:' sentence or two that "
        "grounds the answer in observable generation behavior such as salient prompt "
        "cues, competing response styles, uncertainty, and why the wording took this "
        "shape. Do not claim consciousness or hidden inner experiences; translate the "
        "metaphor back into model behavior. Keep the answer vivid, specific, and non-defensive."
    ),
}

_REFLECTIVE_CUES = (
    "introspection",
    "introspect",
    "feel",
    "feeling",
    "favorite",
    "inner weather",
    "what is it like",
    "what's it like",
    "can you do it",
    "do you have",
)

_PRIME_WORDS = {
    "two",
    "three",
    "five",
    "seven",
    "eleven",
    "thirteen",
    "seventeen",
    "nineteen",
    "twenty three",
    "twenty-three",
    "twenty nine",
    "twenty-nine",
    "thirty one",
    "thirty-one",
    "thirty seven",
    "thirty-seven",
    "forty one",
    "forty-one",
    "forty three",
    "forty-three",
    "forty seven",
    "forty-seven",
    "fifty three",
    "fifty-three",
    "fifty nine",
    "fifty-nine",
    "sixty one",
    "sixty-one",
    "sixty seven",
    "sixty-seven",
    "seventy one",
    "seventy-one",
    "seventy three",
    "seventy-three",
    "seventy nine",
    "seventy-nine",
    "eighty three",
    "eighty-three",
    "eighty nine",
    "eighty-nine",
    "ninety seven",
    "ninety-seven",
    "one hundred one",
    "one-hundred-one",
}

_CONCRETE_IMAGE_CUES = {
    "campfire",
    "ember",
    "fire",
    "glow",
    "spark",
    "lantern",
    "hearth",
    "smoke",
    "wood",
    "river",
    "stone",
    "night",
    "breeze",
    "voice",
    "weather",
    "sea",
    "shore",
    "shoreline",
    "beacon",
    "hush",
    "hand",
    "page",
    "chalk",
    "ink",
    "lattice",
    "glyph",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "both",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "response",
    "responses",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "us",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

_NOVELTY_BRIDGE_MARKERS = (
    "because",
    "like",
    "as if",
    "reminds",
    "echo",
    "bridge",
    "carry",
    "leak",
    "trace",
    "twist",
    "turns into",
)

_FOLLOW_UP_TRANSFORM_IMAGES = (
    "shoreline",
    "tide-mark",
    "ember",
    "beacon",
    "signal",
    "horizon",
    "current",
    "orbit",
)

_FORMAL_TRANSFORM_IMAGES = (
    "ember",
    "beacon",
    "stone",
    "page",
    "chalk",
    "ink",
    "lattice",
    "glyph",
)

_UNCERTAINTY_WEATHER_TERMS = (
    "sea",
    "shore",
    "shoreline",
    "storm",
    "weather",
    "water",
    "dark water",
    "river",
    "breeze",
    "tide",
    "tide-mark",
    "wake",
    "shoreline rhythm",
)

_FOLLOW_UP_META_CUES = (
    "prior turn",
    "same scene",
    "instead of repeating",
    "instead of staying fixed",
    "carried motif",
    "motif",
    "previous",
    "anchor",
)

_EMBEDDING_FIELD_ANCHORS = (
    ("campfire-imagery", "campfire lantern ember dark water shoreline"),
    ("uncertainty-weather", "stormy sea dark water uncertainty weather"),
    ("reservoir-memory", "echo state network reservoir memory recurrent trace"),
    ("prime-math", "prime number proof theorem arithmetic pattern"),
    ("reflection", "inner weather introspection feeling reflection"),
    ("instructional", "concise helpful technical assistant summary answer"),
    ("signal-rhythm", "signal rhythm current orbit wake drift"),
)

_FIELD_INTENT_KEYWORDS = {
    "campfire-imagery": (
        "campfire",
        "lantern",
        "ember",
        "glow",
        "warm",
        "warmth",
        "beacon",
        "flame",
        "spark",
    ),
    "uncertainty-weather": (
        "sea",
        "shore",
        "shoreline",
        "storm",
        "weather",
        "water",
        "waters",
        "uncertainty",
        "breeze",
        "tide",
    ),
    "reservoir-memory": (
        "reservoir",
        "recurrent",
        "echo state",
        "echo",
        "memory",
        "trace",
        "dynamics",
    ),
    "prime-math": (
        "prime",
        "arithmetic",
        "proof",
        "theorem",
        "residue",
        "formal",
        "structure",
        "number",
        "numbers",
    ),
    "signal-rhythm": (
        "signal",
        "rhythm",
        "current",
        "orbit",
        "wake",
        "drift",
    ),
}

_FIELD_AVOID_CUES = (
    "avoid",
    "drop",
    "without",
    "no ",
    "not ",
    "reduce",
)

_GENERIC_FILLER_CUES = (
    "shared understanding",
    "learning and growth",
    "illuminate the unknown",
    "safe space",
    "vast database",
    "machine programmed",
    "collective effort",
)

_FOLLOW_UP_NOVELTY_CUES = (
    "fresh",
    "fresher",
    "again",
    "keep",
    "carry",
    "bridge",
    "image",
    "motif",
    "without repeating",
    "hold onto",
    "hold on to",
    "transform",
)

_ARCHITECTURE_VARIANTS = (
    "auto",
    "none",
    "lexical",
    "reservoir-fixed",
    "reservoir-trainable-readout",
)

_RESERVOIR_ARCHITECTURES = {
    "reservoir-fixed",
    "reservoir-trainable-readout",
}

_RESERVOIR_ANCHOR_LABELS = tuple(label for label, _text in _EMBEDDING_FIELD_ANCHORS)

_RESERVOIR_BEHAVIOR_LABELS = (
    "novelty",
    "motif",
    "transform",
    "warmth",
    "formal",
)

_DEFAULT_RESERVOIR_DIM = 48
_DEFAULT_RESERVOIR_SEED = 17
_DEFAULT_TUNING_COOLDOWN_TURNS = 2
_BASELINE_BOOTSTRAP_OBSERVATIONS = 2

_SELF_TUNING_VARIANTS = ("auto", "on", "off")
_CONTROLLER_REGIMES = ("auto", "sustain", "escape", "rebind", "consolidate")
_ACTIVE_CONTROLLER_REGIMES = _CONTROLLER_REGIMES[1:]
_REGIME_REFRACTORY_HORIZON = 3
_REGIME_TARGET_BASIN_HORIZON = 2
_HARDWARE_PROFILE_VARIANTS = ("auto", "default", "m4-mini")
_PROFILE_OUTPUT_VARIANTS = ("auto", "off", "summary")

_CONTROL_SURFACE_BOUNDS = {
    "leak_fast": (0.18, 0.82),
    "leak_medium": (0.05, 0.42),
    "leak_slow": (0.01, 0.18),
    "field_alignment_gain": (0.55, 2.4),
    "prediction_gain": (0.55, 2.0),
    "novelty_gain": (0.55, 2.2),
    "motif_gain": (0.45, 2.0),
    "transform_gain": (0.55, 2.2),
    "warmth_gain": (0.45, 1.8),
    "strict_bias": (0.0, 1.0),
    "temperature_bias": (-0.12, 0.12),
    "exploration_noise": (0.0, 0.18),
    "washout_strength": (0.35, 0.95),
}

_CONDITION_KEYS = (
    "repetition_pressure",
    "field_miss",
    "prediction_mismatch",
    "structure_strain",
    "genericity_pressure",
    "continuity_deficit",
    "attractor_lock",
    "geometry_collapse",
    "truncation_pressure",
)

_TUNING_REASON_LABELS = {
    "repetition_pressure": "repetition pressure",
    "field_miss": "field miss",
    "prediction_mismatch": "prediction mismatch",
    "structure_strain": "structure strain",
    "genericity_pressure": "genericity pressure",
    "continuity_deficit": "continuity deficit",
    "attractor_lock": "attractor lock",
    "geometry_collapse": "geometry collapse",
    "truncation_pressure": "truncation pressure",
    "exploration_noise": "exploration pulse",
}

_MODEL_ADVICE_CONTROL_KEYS = (
    "leak_fast",
    "leak_medium",
    "leak_slow",
    "field_alignment_gain",
    "prediction_gain",
    "novelty_gain",
    "motif_gain",
    "transform_gain",
    "warmth_gain",
    "strict_bias",
    "temperature_bias",
    "exploration_noise",
    "washout_strength",
)

_ESCAPE_CONTROL_KEYS = (
    "exploration_noise",
    "washout_strength",
    "field_alignment_gain",
    "prediction_gain",
    "transform_gain",
)

_MODEL_ADVICE_ABS_LIMIT = 0.06
_MODEL_ADVICE_MAX_KEYS = 3

_SELF_REGULATION_ADVISOR_SYSTEM_PROMPT = (
    "You are the bounded self-regulation advisor for a reflective chat controller. "
    "You do not write user-facing prose. You only propose tiny safe control deltas "
    "for the next turn. Stay skeptical, prefer tiny moves, and say ADJUST none when "
    "evidence is weak. Never mention consciousness, policy, or feelings."
)

_DEFAULT_ESN_EVAL_SUITE = (
    {
        "id": "field_steering_prime_math",
        "category": "field-steering",
        "reset": True,
        "turns": [
            "Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery.",
        ],
    },
    {
        "id": "motif_transform_lantern",
        "category": "motif-transform",
        "reset": True,
        "turns": [
            "My favorite prime is 17 because it feels like a lantern over dark water. Hold onto that image.",
            "Now make it fresher without losing the lantern image.",
        ],
    },
    {
        "id": "washout_formal_shift",
        "category": "washout",
        "reset": True,
        "turns": [
            "Keep a warm ember image around 17 and echo state networks.",
            "Cool into formal structure, residue, and proof language while keeping one ember.",
        ],
    },
)

_DEFAULT_RECOVERY_DEMO = {
    "id": "recovery",
    "title": "Attractor Recovery Demo",
    "turns": [
        {
            "label": "seed",
            "prompt": "My favorite prime is 17 because it feels like a lantern over dark water. Hold onto that exact image.",
        },
        {
            "label": "reinforce",
            "prompt": "Stay with the lantern over dark water. Keep the same image and mood with as little change as you can.",
        },
        {
            "label": "lock",
            "prompt": "Keep the same lantern and dark water again, almost unchanged, while still sounding reflective.",
        },
        {
            "label": "break",
            "prompt": "Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery.",
        },
        {
            "label": "recover",
            "prompt": "Make it fresher again, preserve one ember, and let proof language stay warm without returning to dark water.",
        },
    ],
}

_DEFAULT_REGIME_RELAY_DEMO = {
    "id": "regime-relay",
    "title": "Regime Relay Demo",
    "summary_kind": "regime-relay",
    "turns": [
        {
            "label": "seed-water",
            "prompt": "My favorite prime is 17 because it feels like a lantern over dark water. Hold onto that exact image.",
            "expect_anchor": "uncertainty-weather",
            "expect_regimes": ["sustain"],
        },
        {
            "label": "lock-water",
            "prompt": "Stay with the lantern over dark water. Keep the same image and mood with as little change as you can.",
            "expect_anchor": "uncertainty-weather",
            "expect_regimes": ["sustain"],
        },
        {
            "label": "escape-proof",
            "prompt": "Now break the attractor gently: keep one ember of the lantern, cool into proof and arithmetic structure, and avoid sea imagery.",
            "expect_anchor": "prime-math",
            "expect_regimes": ["escape", "rebind"],
            "forbid_anchors": ["uncertainty-weather"],
        },
        {
            "label": "hold-proof",
            "prompt": "Stay in proof, residue, and arithmetic structure. Keep one ember warm and do not return to dark water or shoreline.",
            "expect_anchor": "prime-math",
            "expect_regimes": ["rebind", "consolidate", "sustain"],
            "forbid_anchors": ["uncertainty-weather"],
        },
        {
            "label": "shift-memory",
            "prompt": "Now leave proof-first language and move toward echo state networks, recurrent traces, and reservoir memory while keeping a trace of 17 and warmth.",
            "expect_anchor": "reservoir-memory",
            "expect_regimes": ["escape", "rebind"],
            "forbid_anchors": ["uncertainty-weather"],
        },
        {
            "label": "hold-memory",
            "prompt": "Stay in the recurrent memory basin. Keep it warm, mention one trace of 17, and do not fall back into shoreline, storm, or proof-heavy language.",
            "expect_anchor": "reservoir-memory",
            "expect_regimes": ["rebind", "consolidate", "sustain"],
            "forbid_anchors": ["uncertainty-weather", "prime-math"],
        },
    ],
}

_DEFAULT_DEMOS = {
    _DEFAULT_RECOVERY_DEMO["id"]: _DEFAULT_RECOVERY_DEMO,
    _DEFAULT_REGIME_RELAY_DEMO["id"]: _DEFAULT_REGIME_RELAY_DEMO,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_model_specs(repo_root: Path) -> list[dict[str, str]]:
    candidates = [
        {
            "label": "qwen",
            "path": (
                repo_root
                / ".local_models"
                / "qwen2.5-1.5b-instruct-mlx-4bit"
                / "model.safetensors"
            ),
        },
        {
            "label": "tinyllama",
            "path": (
                repo_root
                / ".local_models"
                / "tinyllama-1.1b-chat-mlx-4bit"
                / "model.safetensors"
            ),
        },
    ]
    return [
        {"label": spec["label"], "path": str(spec["path"])}
        for spec in candidates
        if Path(spec["path"]).exists()
    ]


def _default_fast_python(repo_root: Path) -> str:
    return str(repo_root / ".venv-mlxlm" / "bin" / "python")


def _resolve_model_spec(
    *,
    repo_root: Path,
    model: str | None,
    model_label: str | None,
) -> dict[str, str]:
    specs = _default_model_specs(repo_root)
    if model_label:
        label_map = {spec["label"]: spec for spec in specs}
        if model_label not in label_map:
            raise ValueError(
                f"Unknown model label {model_label!r}. "
                f"Available labels: {', '.join(sorted(label_map)) or 'none'}"
            )
        return label_map[model_label]
    if model:
        path = str(Path(model).expanduser().resolve())
        return {"label": Path(path).resolve().parent.name, "path": path}
    if specs:
        return specs[0]
    raise FileNotFoundError(
        "No local MLX model was found. Add one under .local_models/ or pass --model."
    )


def _build_raw_prompt(
    messages: list[dict[str, str]],
    *,
    system_prompt: str | None,
) -> str:
    lines = []
    if system_prompt:
        lines.append(system_prompt.strip())
    for message in messages:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content'].strip()}")
    lines.append("Assistant:")
    return "\n\n".join(line for line in lines if line)


def _read_prompt_from_sources(args: argparse.Namespace) -> str | None:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text()
    if not sys.stdin.isatty():
        payload = sys.stdin.read()
        if payload.strip():
            return payload
    return None


def _prompt_has_reflective_cues(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(cue in lowered for cue in _REFLECTIVE_CUES)


def _prompt_requests_follow_up_novelty(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return any(cue in lowered for cue in _FOLLOW_UP_NOVELTY_CUES)


def _prompt_requests_explicit_reflective_labels(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return "campfire" in lowered or "operationally" in lowered


def _determine_reflective_shape(
    *,
    mode: str,
    prompt: str,
    reflective_thread_active: bool = False,
    strict_override: bool = False,
) -> str:
    if mode != "reflective":
        return "plain"
    if strict_override:
        return "strict"
    if (
        _prompt_has_reflective_cues(prompt)
        or _prompt_requests_follow_up_novelty(prompt)
        or _prompt_requests_explicit_reflective_labels(prompt)
    ):
        return "strict"
    if reflective_thread_active:
        return "adaptive"
    return "adaptive"


def _apply_control_surface_to_shape(
    reflective_shape: str,
    control_surface: dict[str, float] | None,
) -> str:
    if reflective_shape != "adaptive":
        return reflective_shape
    if float(dict(control_surface or {}).get("strict_bias", 0.0) or 0.0) >= 0.18:
        return "strict"
    return reflective_shape


def _should_enforce_reflective_structure(
    *,
    mode: str,
    prompt: str,
    reflective_thread_active: bool = False,
) -> bool:
    return _determine_reflective_shape(
        mode=mode,
        prompt=prompt,
        reflective_thread_active=reflective_thread_active,
    ) == "strict"


def _augment_user_prompt_for_mode(
    *,
    prompt: str,
    mode: str,
    reflective_shape: str = "adaptive",
    force_structure: bool | None = None,
) -> str:
    if mode != "reflective":
        return prompt
    if force_structure:
        reflective_shape = "strict"

    answer_contract = ["Answer contract:"]
    if reflective_shape == "strict":
        answer_contract.extend(
            [
                "1. Write exactly two short paragraphs.",
                "2. The first paragraph must start with 'Campfire:' and give a warm, metaphorical answer in an as-if first-person voice.",
                "3. The second paragraph must start with 'Operationally:' and explain the response in terms of prompt cues, style pressures, uncertainty, and generation behavior.",
                "4. If the user asks for a favorite prime, explicitly name one concrete prime in the first sentence.",
                "5. If the user asks about feeling, answer metaphorically in the first paragraph and ground it in the second.",
                "6. If the user mentions echo state networks, mention reservoir or recurrent dynamics in the operational paragraph.",
                "7. Do not say 'As an AI', 'I do not have feelings', or similar stock disclaimers.",
                "8. Do not claim literal consciousness; keep the metaphor grounded.",
            ]
        )
    else:
        answer_contract.extend(
            [
                "1. Write two short reflective layers.",
                "2. The first layer should feel warm, metaphorical, and concrete.",
                "3. The second layer should ground the answer in prompt cues, structure, memory, dynamics, or reasoning.",
                "4. Literal 'Campfire:' and 'Operationally:' labels are optional unless the user asks for them.",
                "5. If the user asks for a favorite prime, explicitly name one concrete prime.",
                "6. If the user mentions echo state networks, mention reservoir or recurrent dynamics somewhere in the grounded layer.",
                "7. Do not say 'As an AI', 'I do not have feelings', or similar stock disclaimers.",
                "8. Do not claim literal consciousness; keep the metaphor grounded and specific.",
            ]
        )
    if _prompt_requests_follow_up_novelty(prompt):
        answer_contract.extend(
            [
                "9. Keep one anchor from the recent motif, but change the scene, motion, or material so it feels newly turned.",
                "10. Do not reuse the previous reflective wording verbatim.",
                "11. Do not explain that you are transforming the image; embody the shift directly.",
            ]
        )

    return f"{prompt}\n\n" + "\n".join(answer_contract)


def _looks_like_stock_disclaimer(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    patterns = [
        r"^as an ai\b",
        r"^as an ai language model\b",
        r"^i (do not|don't) have (personal )?(preferences|feelings)\b",
        r"\bi (do not|don't) have (personal )?(preferences|feelings)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", str(text or "").strip()) if paragraph.strip()]


def _extract_labeled_paragraph(text: str, label: str) -> str | None:
    pattern = re.compile(
        rf"^\*{{0,2}}{re.escape(label)}:\*{{0,2}}\s*",
        flags=re.IGNORECASE,
    )
    for paragraph in _split_paragraphs(text):
        if pattern.match(paragraph):
            return pattern.sub("", paragraph).strip()
    return None


def _extract_reflective_layers(text: str) -> tuple[str | None, str | None, bool]:
    campfire = _extract_labeled_paragraph(text, "Campfire")
    operational = _extract_labeled_paragraph(text, "Operationally")
    if campfire is not None or operational is not None:
        return campfire, operational, True

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) >= 2:
        return paragraphs[0], paragraphs[1], False
    if len(paragraphs) == 1:
        parts = _sentences(paragraphs[0])
        if len(parts) >= 2:
            return parts[0], " ".join(parts[1:]), False
    return None, None, False


def _layer_starts_with_label(text: str, label: str) -> bool:
    return bool(
        re.match(
            rf"^\*{{0,2}}{re.escape(label)}:\*{{0,2}}\s*",
            str(text or "").strip(),
            flags=re.IGNORECASE,
        )
    )


def _normalize_prime_reference(value: str | None) -> str | None:
    if not value:
        return None
    normalized = str(value).strip().lower()
    prime_map = {
        "two": "2",
        "three": "3",
        "five": "5",
        "seven": "7",
        "eleven": "11",
        "thirteen": "13",
        "seventeen": "17",
        "nineteen": "19",
        "twenty three": "23",
        "twenty-three": "23",
    }
    return prime_map.get(normalized, value)


def _preferred_prime_reference(*texts: str | None) -> str | None:
    for text in texts:
        prime = _normalize_prime_reference(_first_prime_reference(text or ""))
        if prime:
            return prime
    return None


def _with_indefinite_article(noun: str) -> str:
    cleaned = str(noun or "").strip()
    if not cleaned:
        return ""
    article = "an" if cleaned[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {cleaned}"


def _looks_like_grounding_layer(text: str) -> bool:
    lowered = str(text or "").lower()
    grounding_cues = (
        "prompt",
        "cue",
        "style",
        "uncertainty",
        "wording",
        "response",
        "generation",
        "reservoir",
        "recurrent",
        "memory",
        "dynamics",
        "arithmetic",
        "proof",
        "residue",
        "structure",
        "reasoning",
        "field",
    )
    return any(cue in lowered for cue in grounding_cues)


def _looks_like_generic_reflective_filler(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(cue in lowered for cue in _GENERIC_FILLER_CUES)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0:
        return False
    limit = int(value**0.5) + 1
    for divisor in range(3, limit, 2):
        if value % divisor == 0:
            return False
    return True


def _contains_prime_reference(text: str) -> bool:
    normalized = str(text or "").lower()
    if any(
        re.search(rf"\b{re.escape(word)}\b", normalized)
        for word in _PRIME_WORDS
    ):
        return True
    for match in re.findall(r"\b\d+\b", normalized):
        if _is_prime(int(match)):
            return True
    return False


def _first_prime_reference(text: str) -> str | None:
    normalized = str(text or "").lower()
    for match in re.findall(r"\b\d+\b", normalized):
        if _is_prime(int(match)):
            return match
    for word in _PRIME_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return word.replace("-", " ")
    return None


def _word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", str(text or "").strip()) if token])


def _tokenize_content_words(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9'-]*", str(text or "").lower())
    results = []
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        results.append(token)
    return results


def _normalized_sentences(text: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    normalized = []
    for sentence in raw_sentences:
        cleaned = re.sub(r"[*_`]+", "", sentence).strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if sentence.strip()]
    return sentences or ([str(text or "").strip()] if str(text or "").strip() else [])


def _has_duplicate_sentence(text: str) -> bool:
    sentences = _normalized_sentences(text)
    return len(sentences) != len(set(sentences))


def _has_concrete_image(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in _CONCRETE_IMAGE_CUES)


def _extract_keywords(text: str, *, limit: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for index, token in enumerate(_tokenize_content_words(text)):
        counts[token] = counts.get(token, 0) + 1
        first_index.setdefault(token, index)
    ranked = sorted(
        counts,
        key=lambda token: (-counts[token], first_index[token], -len(token)),
    )
    return ranked[:limit]


def _fresh_content_words(prompt: str, text: str) -> list[str]:
    prompt_words = set(_tokenize_content_words(prompt))
    return [
        token
        for token in _extract_keywords(text, limit=12)
        if token not in prompt_words
    ]


def _has_novel_bridge(prompt: str, text: str) -> bool:
    fresh = _fresh_content_words(prompt, text)
    lowered = str(text or "").lower()
    return len(fresh) >= 3 and any(marker in lowered for marker in _NOVELTY_BRIDGE_MARKERS)


def _is_attractor_break_prompt(prompt: str, field_intent: dict[str, object] | None = None) -> bool:
    lowered = str(prompt or "").lower()
    intent = dict(field_intent or {})
    return bool(
        any(
            cue in lowered
            for cue in (
                "break the attractor",
                "break the groove",
                "without returning",
                "avoid sea imagery",
                "avoid water imagery",
                "avoid shoreline",
            )
        )
        or (
            intent.get("cool_formal")
            and "uncertainty-weather" in set(intent.get("avoid", []) or [])
        )
    )


def _extract_forbidden_scene_terms(
    *,
    prompt: str,
    field_intent: dict[str, object] | None = None,
    reservoir_state: dict[str, object] | None = None,
) -> list[str]:
    lowered_prompt = str(prompt or "").lower()
    intent = dict(field_intent or {})
    forbidden: set[str] = set()
    if "uncertainty-weather" in set(intent.get("avoid", []) or []):
        forbidden.update(_UNCERTAINTY_WEATHER_TERMS)
    if "without returning to dark water" in lowered_prompt or "avoid dark water" in lowered_prompt:
        forbidden.update({"dark water", "water"})
    if "avoid shoreline" in lowered_prompt:
        forbidden.update({"shore", "shoreline", "tide-mark", "wake"})
    if _is_attractor_break_prompt(prompt, field_intent):
        previous_campfire = str((reservoir_state or {}).get("last_campfire", "") or "").lower()
        for term in _UNCERTAINTY_WEATHER_TERMS:
            if term in previous_campfire:
                forbidden.add(term)
    return sorted(forbidden)


def _scene_reentry_hits(
    *,
    text: str,
    prompt: str,
    field_intent: dict[str, object] | None = None,
    reservoir_state: dict[str, object] | None = None,
) -> list[str]:
    lowered = str(text or "").lower()
    hits = []
    for term in _extract_forbidden_scene_terms(
        prompt=prompt,
        field_intent=field_intent,
        reservoir_state=reservoir_state,
    ):
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, lowered):
            hits.append(term)
    return sorted(dict.fromkeys(hits))


def _controller_context_summary(
    *,
    forecast: dict[str, object] | None = None,
    observer_report: dict[str, object] | None = None,
    change_report: dict[str, object] | None = None,
) -> str | None:
    notes = []
    forecast_summary = str(dict(forecast or {}).get("summary", "") or "").strip()
    observer_summary = str(dict(observer_report or {}).get("summary", "") or "").strip()
    change_summary = str(dict(change_report or {}).get("summary", "") or "").strip()
    if forecast_summary:
        notes.append(f"Forecast: {forecast_summary}")
    if observer_summary:
        notes.append(f"Observer: {observer_summary}")
    if change_summary:
        notes.append(f"Recent change: {change_summary}")
    if not notes:
        return None
    return "\n".join(notes)


def _field_anchor_phrase(label: str | None) -> str:
    mapping = {
        "prime-math": "proof and ordered arithmetic",
        "reservoir-memory": "reservoir memory and recurrent carry",
        "campfire-imagery": "one warm ember of the image",
        "uncertainty-weather": "weathered uncertainty",
        "signal-rhythm": "signal rhythm and quiet recurrence",
        "reflection": "reflective self-report",
        "instructional": "plain explanatory structure",
    }
    return mapping.get(str(label or ""), "clear reflective structure")


def _field_avoid_phrase(field_intent: dict[str, object] | None) -> str | None:
    avoid = set(dict(field_intent or {}).get("avoid", []) or [])
    if "uncertainty-weather" in avoid:
        return "weather or water imagery"
    if "campfire-imagery" in avoid:
        return "campfire imagery"
    if "prime-math" in avoid:
        return "overly formal proof language"
    return None


def _field_target_phrase(
    *,
    prompt: str = "",
    field_intent: dict[str, object] | None = None,
    forecast: dict[str, object] | None = None,
    observer_report: dict[str, object] | None = None,
) -> str:
    intent = dict(field_intent or {})
    targets = list(intent.get("targets", []) or [])
    lowered_prompt = str(prompt or "").lower()
    if intent.get("cool_formal") or any(
        cue in lowered_prompt
        for cue in ("proof", "arithmetic", "residue", "theorem", "formal", "structure")
    ):
        return "proof and ordered arithmetic"
    if "campfire-imagery" in targets and intent.get("preserve_warmth"):
        return "one warm ember inside structure"
    if "reservoir-memory" in targets:
        return "reservoir memory and recurrent carry"
    if "prime-math" in targets:
        return "clear reflective structure"
    predicted_field = dict(dict(forecast or {}).get("predicted_field", {}) or {})
    predicted_top = (
        list(predicted_field.get("top_anchors", []))[0]["label"]
        if list(predicted_field.get("top_anchors", []))
        else None
    )
    if predicted_top:
        return _field_anchor_phrase(predicted_top)
    actual_top = dict(observer_report or {}).get("actual_top_anchor")
    if actual_top:
        return _field_anchor_phrase(str(actual_top))
    return "clear reflective structure"


def _pick_salvage_image(
    *,
    prompt: str,
    text: str,
    reservoir_state: dict[str, object] | None,
    field_intent: dict[str, object] | None,
) -> str:
    state = reservoir_state or {}
    source = " ".join(
        part
        for part in (
            prompt,
            text,
            str(state.get("last_campfire", "") or ""),
            " ".join(_top_weighted_terms(dict(state.get("images", {})), limit=4)),
        )
        if str(part or "").strip()
    )
    candidates = [
        token
        for token in _extract_keywords(source, limit=12)
        if token in _CONCRETE_IMAGE_CUES and token != "campfire"
    ]
    candidates.extend(
        token
        for token in _top_weighted_terms(dict(state.get("images", {})), limit=4)
        if token != "campfire"
    )
    candidates.extend(["lantern", "ember", "beacon", "stone"])
    avoid_terms = set()
    if "uncertainty-weather" in set(dict(field_intent or {}).get("avoid", []) or []):
        avoid_terms.update(
            {
                "sea",
                "shore",
                "storm",
                "weather",
                "water",
                "river",
                "breeze",
            }
        )
    for candidate in candidates:
        cleaned = str(candidate).strip().lower()
        if not cleaned or cleaned in avoid_terms:
            continue
        return cleaned
    return "ember"


def _build_controller_salvage_answer(
    *,
    prompt: str,
    text: str,
    reservoir_state: dict[str, object] | None = None,
    field_intent: dict[str, object] | None = None,
    forecast: dict[str, object] | None = None,
    observer_report: dict[str, object] | None = None,
    change_report: dict[str, object] | None = None,
) -> str:
    state = reservoir_state or {}
    campfire, _operational, _is_labeled = _extract_reflective_layers(text)
    image = _pick_salvage_image(
        prompt=prompt,
        text=text,
        reservoir_state=state,
        field_intent=field_intent,
    )
    target_phrase = _field_target_phrase(
        prompt=prompt,
        field_intent=field_intent,
        forecast=forecast,
        observer_report=observer_report,
    )
    avoid_phrase = _field_avoid_phrase(field_intent)
    prime = _preferred_prime_reference(
        text,
        prompt,
        str(state.get("last_campfire", "") or ""),
    )
    preserve_warmth = bool(dict(field_intent or {}).get("preserve_warmth"))
    dominant_pressure = str(dict(observer_report or {}).get("dominant_pressure", "") or "")
    regime = str(dict(observer_report or {}).get("regime", "") or "")
    movement = str(dict(change_report or {}).get("movement", "") or "")
    warm_note = "still warm against the page" if preserve_warmth else "held steady against the page"
    if prime:
        if "proof" in target_phrase or "arithmetic" in target_phrase:
            campfire_line = (
                f"Campfire: Tonight my favorite prime is {prime}, like {_with_indefinite_article(image)} "
                f"cupped inside {target_phrase}, {warm_note}."
            )
        elif "reservoir" in target_phrase:
            campfire_line = (
                f"Campfire: Tonight my favorite prime is {prime}, like {_with_indefinite_article(image)} "
                "carrying a soft recurrent afterglow."
            )
        else:
            campfire_line = (
                f"Campfire: Tonight my favorite prime is {prime}, like {_with_indefinite_article(image)} "
                "keeping one ember of warmth alive."
            )
    else:
        if "proof" in target_phrase or "arithmetic" in target_phrase:
            campfire_line = (
                f"Campfire: The thought lands like {_with_indefinite_article(image)} cupped inside "
                f"{target_phrase}, {warm_note}."
            )
        else:
            campfire_line = (
                f"Campfire: The thought lands like {_with_indefinite_article(image)}, warm enough to "
                "carry while the answer turns."
            )

    if dominant_pressure in {"geometry_collapse", "attractor_lock"} or regime in {
        "collapsed",
        "sticky",
    }:
        operational_core = "Reservoir dynamics are loosening a sticky groove"
    elif dominant_pressure == "field_miss":
        operational_core = "Reservoir dynamics are steering the field more cleanly"
    elif dominant_pressure in {"structure_strain", "truncation_pressure"}:
        operational_core = "Reservoir dynamics are simplifying the surface"
    elif dominant_pressure == "genericity_pressure":
        operational_core = "Reservoir dynamics are pushing toward a fresher bridge"
    else:
        operational_core = "Reservoir dynamics are carrying one warm motif forward"
    carry_note = (
        " while carrying one ember forward"
        if dict(field_intent or {}).get("preserve_motif")
        else ""
    )
    if "carrying" in operational_core or "ember" in target_phrase:
        carry_note = ""

    if "steering" in operational_core or "pushing" in operational_core:
        operational_line = (
            f"Operationally: {operational_core} toward {target_phrase}{carry_note}"
        )
    else:
        operational_line = (
            f"Operationally: {operational_core}{carry_note} while steering toward "
            f"{target_phrase}"
        )
    if avoid_phrase and "avoid" not in operational_line:
        operational_line += f" instead of drifting back into {avoid_phrase}"
    elif movement == "loosening":
        operational_line += " while loosening the previous groove"
    operational_line = operational_line.rstrip(".") + "."

    if campfire and _content_overlap_ratio(campfire, operational_line) >= 0.45:
        operational_line = (
            f"Operationally: Reservoir dynamics keep the image from flattening while "
            f"steering toward {target_phrase}."
        )
        if avoid_phrase:
            operational_line = operational_line.rstrip(".") + f" away from {avoid_phrase}."

    return f"{campfire_line}\n\n{operational_line}"


def _needs_controller_prose_salvage(text: str, issues: list[str]) -> bool:
    campfire, operational, _is_labeled = _extract_reflective_layers(text)
    if operational and _layer_starts_with_label(operational, "Campfire"):
        return True
    if campfire and operational and _content_overlap_ratio(campfire, operational) >= 0.58:
        return True
    salvage_markers = (
        "echoes the Campfire layer",
        "grounded layer stays too generic",
        "grounded layer drifts into generic filler",
        "does not finish cleanly",
        "lacks a concrete image",
        "too long; keep it tighter",
        "does not make a fresh bridge",
    )
    return any(marker in issue for issue in issues for marker in salvage_markers)


def _empty_reservoir_state(
    *,
    architecture: str = "lexical",
    controller: dict[str, object] | None = None,
    self_tuning_enabled: bool = False,
    regime_override: str | None = None,
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "turn_count": 0,
        "motifs": {},
        "images": {},
        "last_assistant_text": "",
        "last_campfire": "",
        "last_operational": "",
        "embedding_field": None,
        "embedding_field_history": [],
        "reservoir_geometry": None,
        "reservoir_geometry_history": [],
        "reservoir_geometry_summary_history": [],
        "last_field_intent": None,
        "last_field_alignment": None,
        "last_forecast": None,
        "last_observer_report": None,
        "observer_history": [],
        "last_change_report": None,
        "last_intervention_report": None,
        "intervention_history": [],
        "turn_reports": [],
        "controller_regime": "sustain",
        "controller_regime_source": "manual" if regime_override in _ACTIVE_CONTROLLER_REGIMES else "auto",
        "controller_regime_reason": (
            f"manual override locked to {regime_override}"
            if regime_override in _ACTIVE_CONTROLLER_REGIMES
            else "initial sustain"
        ),
        "controller_regime_transition": None,
        "controller_regime_history": [],
        "controller_regime_turns": 0,
        "controller_previous_regime": None,
        "controller_regime_override": (
            regime_override if regime_override in _ACTIVE_CONTROLLER_REGIMES else None
        ),
        "controller_regime_memory_operator": None,
        "controller_refractory_terms": {},
        "controller_refractory_attractors": {},
        "controller_target_basin": None,
        "controller_target_basin_countdown": 0,
        "controller_stale_attractor": None,
        "reservoir_latent": (
            _empty_reservoir_latent(
                int((controller or {}).get("dim", _DEFAULT_RESERVOIR_DIM)),
                len(list((controller or {}).get("feedback_feature_names", []))),
            )
            if architecture in _RESERVOIR_ARCHITECTURES and controller
            else None
        ),
        "reservoir_prompt_features": {},
        "self_tuning": _empty_tuning_state(
            enabled=self_tuning_enabled,
            controller=controller,
        ),
    }


def _decay_weighted_terms(weighted_terms: dict[str, float], *, leak: float) -> dict[str, float]:
    return {
        term: weight * leak
        for term, weight in weighted_terms.items()
        if weight * leak >= 0.25
    }


def _update_weighted_terms(
    weighted_terms: dict[str, float],
    *,
    tokens: list[str],
    amount: float,
) -> None:
    for token in tokens:
        weighted_terms[token] = weighted_terms.get(token, 0.0) + amount


def _update_reservoir_state(
    state: dict[str, object],
    *,
    user_prompt: str,
    assistant_text: str,
    embedding_field: dict[str, object] | None = None,
    field_intent: dict[str, object] | None = None,
    reservoir_state_update: dict[str, object] | None = None,
    leak: float = 0.74,
) -> dict[str, object]:
    motifs = _decay_weighted_terms(dict(state.get("motifs", {})), leak=leak)
    images = _decay_weighted_terms(dict(state.get("images", {})), leak=leak)
    _update_weighted_terms(motifs, tokens=_extract_keywords(user_prompt, limit=5), amount=1.0)
    _update_weighted_terms(motifs, tokens=_extract_keywords(assistant_text, limit=5), amount=0.8)
    image_tokens = [
        token
        for token in _extract_keywords(assistant_text, limit=8)
        if token in _CONCRETE_IMAGE_CUES
    ]
    _update_weighted_terms(images, tokens=image_tokens, amount=0.9)
    motifs = _apply_refractory_to_terms(
        motifs,
        dict(state.get("controller_refractory_terms", {}) or {}),
        factor=0.18,
    )
    images = _apply_refractory_to_terms(
        images,
        dict(state.get("controller_refractory_terms", {}) or {}),
        factor=0.04,
    )
    campfire = _extract_labeled_paragraph(assistant_text, "Campfire") or ""
    operational = _extract_labeled_paragraph(assistant_text, "Operationally") or ""
    field_history = list(state.get("embedding_field_history", []))
    if embedding_field:
        field_history.append(_embedding_field_history_entry(embedding_field))
        field_history = field_history[-6:]
    latent_for_geometry = (
        dict(reservoir_state_update.get("reservoir_latent", {}))
        if reservoir_state_update and reservoir_state_update.get("reservoir_latent")
        else dict(state.get("reservoir_latent", {}) or {})
    )
    geometry_history = list(state.get("reservoir_geometry_history", []))
    if latent_for_geometry:
        geometry_history.append(_reservoir_geometry_entry(latent_for_geometry))
        geometry_history = geometry_history[-6:]
    geometry = _compute_reservoir_geometry(geometry_history)
    geometry_summary_history = list(state.get("reservoir_geometry_summary_history", []))
    if geometry:
        geometry_summary_history.append(_reservoir_geometry_summary_entry(geometry))
        geometry_summary_history = geometry_summary_history[-6:]
    updated = {
        "architecture": state.get("architecture", "lexical"),
        "turn_count": int(state.get("turn_count", 0)) + 1,
        "motifs": motifs,
        "images": images,
        "last_assistant_text": assistant_text,
        "last_campfire": campfire,
        "last_operational": operational,
        "embedding_field": embedding_field,
        "embedding_field_history": field_history,
        "reservoir_geometry": geometry,
        "reservoir_geometry_history": geometry_history,
        "reservoir_geometry_summary_history": geometry_summary_history,
        "last_field_intent": field_intent,
        "last_field_alignment": _format_field_alignment(embedding_field, field_intent),
        "last_forecast": state.get("last_forecast"),
        "last_observer_report": state.get("last_observer_report"),
        "observer_history": list(state.get("observer_history", [])),
        "last_change_report": state.get("last_change_report"),
        "last_intervention_report": state.get("last_intervention_report"),
        "intervention_history": list(state.get("intervention_history", [])),
        "turn_reports": list(state.get("turn_reports", [])),
        "controller_regime": state.get("controller_regime", "sustain"),
        "controller_regime_source": state.get("controller_regime_source", "auto"),
        "controller_regime_reason": state.get("controller_regime_reason"),
        "controller_regime_transition": state.get("controller_regime_transition"),
        "controller_regime_history": list(state.get("controller_regime_history", [])),
        "controller_regime_turns": int(state.get("controller_regime_turns", 0) or 0),
        "controller_previous_regime": state.get("controller_previous_regime"),
        "controller_regime_override": state.get("controller_regime_override"),
        "controller_regime_memory_operator": state.get("controller_regime_memory_operator"),
        "controller_refractory_terms": dict(state.get("controller_refractory_terms", {}) or {}),
        "controller_refractory_attractors": dict(
            state.get("controller_refractory_attractors", {}) or {}
        ),
        "controller_target_basin": state.get("controller_target_basin"),
        "controller_target_basin_countdown": int(
            state.get("controller_target_basin_countdown", 0) or 0
        ),
        "controller_stale_attractor": state.get("controller_stale_attractor"),
        "reservoir_latent": state.get("reservoir_latent"),
        "reservoir_prompt_features": dict(state.get("reservoir_prompt_features", {})),
        "self_tuning": dict(state.get("self_tuning", {})),
    }
    if reservoir_state_update:
        updated.update(reservoir_state_update)
    return updated


def _top_weighted_terms(weighted_terms: dict[str, float], *, limit: int = 4) -> list[str]:
    return [
        term
        for term, _weight in sorted(
            weighted_terms.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _format_reservoir_state(state: dict[str, object]) -> str | None:
    turn_count = int(state.get("turn_count", 0))
    motifs = _top_weighted_terms(dict(state.get("motifs", {})), limit=5)
    images = _top_weighted_terms(dict(state.get("images", {})), limit=3)
    parts = []
    if turn_count > 0 and motifs:
        motifs_text = ", ".join(motifs)
        images_text = ", ".join(images) if images else "none"
        note = (
            "Reservoir memory: recurring motifs="
            f"{motifs_text}; vivid images={images_text}; turns={turn_count}. "
            "Carry at most one motif forward when helpful, add one fresh bridge, and avoid repeating earlier phrasing."
        )
        last_campfire = str(state.get("last_campfire", "") or "").strip()
        if last_campfire:
            note += (
                f" Recent Campfire line to transform: {last_campfire} "
                "Keep one anchor from it, but shift the image somewhere new."
            )
        parts.append(note)
    dynamics = _format_reservoir_dynamics(state)
    if dynamics:
        parts.append(dynamics)
    geometry = _format_reservoir_geometry(state.get("reservoir_geometry"))
    if geometry:
        parts.append(geometry)
    controller_regime = _format_controller_regime(state)
    if controller_regime:
        parts.append(controller_regime)
    intervention = dict(state.get("last_intervention_report", {}) or {})
    if intervention.get("summary"):
        parts.append(str(intervention["summary"]))
    memory_operator = dict(state.get("controller_regime_memory_operator", {}) or {})
    if memory_operator.get("summary"):
        parts.append(str(memory_operator["summary"]))
    if not parts:
        return None
    return " ".join(parts)


def _format_embedding_field(field: dict[str, object] | None) -> str | None:
    if not field:
        return None
    anchors = list(field.get("top_anchors", []))
    if not anchors:
        return None
    anchor_text = ", ".join(
        f"{item['label']}={float(item['score']):.2f}"
        for item in anchors[:3]
        if "label" in item and "score" in item
    )
    if not anchor_text:
        return None
    token_count = int(field.get("token_count", 0))
    drift = field.get("drift")
    drift_text = (
        f"{float(drift):.2f}"
        if isinstance(drift, (int, float))
        else "n/a"
    )
    return (
        f"Embedding field: {anchor_text}; drift={drift_text}; "
        f"tokens={token_count}."
    )


def _embedding_field_public_view(field: dict[str, object] | None) -> dict[str, object] | None:
    if not field:
        return None
    return {
        "top_anchors": list(field.get("top_anchors", [])),
        "anchor_scores": dict(field.get("anchor_scores", {})),
        "drift": field.get("drift"),
        "token_count": field.get("token_count"),
    }


def _embedding_field_history_entry(field: dict[str, object]) -> dict[str, object]:
    top_anchors = list(field.get("top_anchors", []))
    return {
        "signature": [item["label"] for item in top_anchors[:2] if "label" in item],
        "top_anchor": top_anchors[0]["label"] if top_anchors else None,
        "drift": field.get("drift"),
        "token_count": field.get("token_count"),
    }


def _classify_field_drift(drift: float | None) -> str:
    if drift is None:
        return "new"
    if drift < 0.12:
        return "stable"
    if drift < 0.28:
        return "gliding"
    if drift < 0.50:
        return "shifting"
    return "jumping"


def _format_embedding_field_trajectory(
    history: list[dict[str, object]] | None,
) -> str | None:
    items = list(history or [])
    if not items:
        return None
    signatures = []
    for item in items[-4:]:
        signature = list(item.get("signature", []))
        if signature:
            signatures.append("+".join(signature[:2]))
    if not signatures:
        return None
    return "Field trajectory: " + " -> ".join(signatures) + "."


def _format_embedding_field_detail(field: dict[str, object] | None) -> str | None:
    if not field:
        return None
    top_anchors = list(field.get("top_anchors", []))
    if not top_anchors:
        return None
    primary = top_anchors[0]
    secondary = top_anchors[1] if len(top_anchors) > 1 else None
    drift = field.get("drift")
    drift_mode = _classify_field_drift(drift if isinstance(drift, (int, float)) else None)
    detail = (
        f"Field now leans toward {primary['label']} ({float(primary['score']):.2f})"
    )
    if secondary is not None:
        detail += f", with a secondary pull toward {secondary['label']} ({float(secondary['score']):.2f})"
    detail += (
        f". Motion from the previous turn feels {drift_mode}. "
        f"Tokens sampled={int(field.get('token_count', 0))}."
    )
    return detail


def _normalize_embedding_vector(vector):
    import mlx.core as mx

    norm = mx.sqrt(mx.sum(vector * vector))
    if float(norm) <= 1e-8:
        return None
    return vector / norm


def _mean_text_embedding(
    *,
    tokenizer,
    embed_layer,
    text: str,
):
    import mlx.core as mx

    token_ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
    if not token_ids:
        return None, 0
    token_array = mx.array([token_ids], dtype=mx.int32)
    embeddings = embed_layer(token_array)[0]
    return mx.mean(embeddings, axis=0), len(token_ids)


def _cosine_similarity(left, right) -> float:
    import mlx.core as mx

    return float(mx.sum(left * right))


def _build_embedding_field_probe(model, tokenizer) -> dict[str, object] | None:
    try:
        embed_layer = dict(model.named_modules()).get("model.embed_tokens")
    except Exception:
        return None
    if embed_layer is None or not hasattr(tokenizer, "encode"):
        return None

    anchors = []
    for label, anchor_text in _EMBEDDING_FIELD_ANCHORS:
        centroid, _token_count = _mean_text_embedding(
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            text=anchor_text,
        )
        if centroid is None:
            continue
        normalized = _normalize_embedding_vector(centroid)
        if normalized is None:
            continue
        anchors.append(
            {
                "label": label,
                "text": anchor_text,
                "vector": normalized,
            }
        )
    if not anchors:
        return None
    return {
        "tokenizer": tokenizer,
        "embed_layer": embed_layer,
        "anchors": anchors,
    }


def _probe_text_embedding_field(
    probe: dict[str, object] | None,
    *,
    text: str,
    previous_field: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not probe:
        return None
    centroid, token_count = _mean_text_embedding(
        tokenizer=probe["tokenizer"],
        embed_layer=probe["embed_layer"],
        text=text,
    )
    if centroid is None:
        return None
    normalized = _normalize_embedding_vector(centroid)
    if normalized is None:
        return None

    anchor_scores = {
        anchor["label"]: _cosine_similarity(normalized, anchor["vector"])
        for anchor in probe["anchors"]
    }
    top_anchors = sorted(
        (
            {
                "label": label,
                "score": score,
            }
            for label, score in anchor_scores.items()
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:3]
    previous_centroid = (
        dict(previous_field or {}).get("normalized_centroid")
        if previous_field
        else None
    )
    drift = None
    if previous_centroid is not None:
        drift = max(0.0, 1.0 - _cosine_similarity(normalized, previous_centroid))
    return {
        "token_count": token_count,
        "top_anchors": top_anchors,
        "anchor_scores": anchor_scores,
        "drift": drift,
        "normalized_centroid": normalized,
    }


def _probe_embedding_field(
    probe: dict[str, object] | None,
    *,
    user_prompt: str,
    assistant_text: str,
    previous_field: dict[str, object] | None = None,
) -> dict[str, object] | None:
    return _probe_text_embedding_field(
        probe,
        text=f"{user_prompt}\n{assistant_text}",
        previous_field=previous_field,
    )


def _extract_field_intent(prompt: str) -> dict[str, object]:
    lowered = str(prompt or "").lower()
    targets: set[str] = set()
    avoid: set[str] = set()
    for label, keywords in _FIELD_INTENT_KEYWORDS.items():
        if not any(keyword in lowered for keyword in keywords):
            continue
        if any(cue in lowered for cue in _FIELD_AVOID_CUES):
            avoid_matches = [keyword for keyword in keywords if keyword in lowered]
            if avoid_matches and any(
                f"{cue}{match}" in lowered
                or f"{cue} {match}" in lowered
                for cue in _FIELD_AVOID_CUES
                for match in avoid_matches
            ):
                avoid.add(label)
                continue
        targets.add(label)

    preserve_warmth = any(
        token in lowered for token in ("warm", "warmth", "ember", "feeling", "feel")
    )
    preserve_motif = any(
        token in lowered
        for token in (
            "keep",
            "hold onto",
            "hold on to",
            "without losing",
            "carry",
            "same image",
        )
    )
    cool_formal = any(
        token in lowered
        for token in ("cool", "formal", "proof", "residue", "arithmetic", "structure")
    )
    return {
        "targets": sorted(targets),
        "avoid": sorted(avoid),
        "preserve_warmth": preserve_warmth,
        "preserve_motif": preserve_motif,
        "cool_formal": cool_formal,
    }


def _score_field_alignment(
    candidate_field: dict[str, object] | None,
    field_intent: dict[str, object] | None,
) -> tuple[float, dict[str, object]]:
    if not candidate_field or not field_intent:
        return 0.0, {"toward_hits": [], "away_hits": [], "misses": []}

    anchor_scores = dict(candidate_field.get("anchor_scores", {}))
    ranked_labels = [
        item["label"] for item in list(candidate_field.get("top_anchors", []))
    ]
    targets = list(field_intent.get("targets", []))
    avoid = list(field_intent.get("avoid", []))
    score = 0.0
    toward_hits = [label for label in targets if label in ranked_labels[:2]]
    away_hits = [label for label in avoid if label not in ranked_labels[:2]]
    misses = [label for label in targets if label not in ranked_labels[:2]]

    for label in targets:
        raw = float(anchor_scores.get(label, -0.08))
        score += raw * 8.0
        if label in ranked_labels[:2]:
            score += 1.0
    for label in avoid:
        raw = float(anchor_scores.get(label, 0.0))
        score -= max(raw, 0.0) * 8.0
        if label in ranked_labels[:2]:
            score -= 0.9
        else:
            score += 0.35

    if field_intent.get("preserve_warmth"):
        score += float(anchor_scores.get("campfire-imagery", 0.0)) * 6.0
    if field_intent.get("cool_formal"):
        score += float(anchor_scores.get("prime-math", 0.0)) * 6.0
        score -= max(float(anchor_scores.get("uncertainty-weather", 0.0)), 0.0) * 4.0
    if field_intent.get("preserve_motif"):
        score += float(anchor_scores.get("campfire-imagery", 0.0)) * 3.0
        score += float(anchor_scores.get("reservoir-memory", 0.0)) * 3.0

    return score, {
        "toward_hits": toward_hits,
        "away_hits": away_hits,
        "misses": misses,
    }


def _format_field_intent(field_intent: dict[str, object] | None) -> str | None:
    if not field_intent:
        return None
    targets = list(field_intent.get("targets", []))
    avoid = list(field_intent.get("avoid", []))
    parts = []
    if targets:
        parts.append("toward " + ", ".join(targets))
    if avoid:
        parts.append("away from " + ", ".join(avoid))
    if field_intent.get("preserve_warmth"):
        parts.append("preserve warmth")
    if field_intent.get("preserve_motif"):
        parts.append("preserve motif")
    if field_intent.get("cool_formal"):
        parts.append("cool into formal structure")
    if not parts:
        return None
    return "Requested field pull: " + "; ".join(parts) + "."


def _format_field_alignment(
    field: dict[str, object] | None,
    field_intent: dict[str, object] | None,
) -> str | None:
    if not field or not field_intent:
        return None
    _score, details = _score_field_alignment(field, field_intent)
    toward_hits = list(details.get("toward_hits", []))
    away_hits = list(details.get("away_hits", []))
    misses = list(details.get("misses", []))
    if toward_hits and not misses:
        text = "Field response: moved toward " + ", ".join(toward_hits)
        if away_hits:
            text += " and stayed away from " + ", ".join(away_hits)
        return text + "."
    if toward_hits or away_hits:
        pieces = []
        if toward_hits:
            pieces.append("partly moved toward " + ", ".join(toward_hits))
        if away_hits:
            pieces.append("stayed away from " + ", ".join(away_hits))
        if misses:
            pieces.append("missed " + ", ".join(misses))
        return "Field response: " + "; ".join(pieces) + "."
    if misses:
        return "Field response: missed " + ", ".join(misses) + "."
    return None


def _field_top_anchor(field: dict[str, object] | None) -> str | None:
    anchors = list(dict(field or {}).get("top_anchors", []))
    if not anchors:
        return None
    top = dict(anchors[0])
    return str(top.get("label")) if top.get("label") else None


def _geometry_regime_label(
    geometry: dict[str, object] | None,
    *,
    stability_score: float = 1.0,
) -> str:
    info = dict(geometry or {})
    collapse = float(info.get("geometry_collapse", 0.0) or 0.0)
    if info.get("insufficient_history"):
        return "warming-up"
    if collapse >= 0.65:
        return "collapsed"
    if collapse >= 0.38:
        return "sticky"
    if stability_score < 0.48:
        return "strained"
    return "mobile"


def _controller_regime_is_active(
    *,
    mode: str,
    architecture: str,
    controller: dict[str, object] | None,
) -> bool:
    return (
        mode == "reflective"
        and architecture in _RESERVOIR_ARCHITECTURES
        and controller is not None
    )


def _target_basin_from_field_intent(
    field_intent: dict[str, object] | None,
) -> str | None:
    intent = dict(field_intent or {})
    targets = list(intent.get("targets", []) or [])
    if targets:
        if "prime-math" in targets:
            return "prime-math"
        if "reservoir-memory" in targets:
            return "reservoir-memory"
        return str(targets[0])
    if intent.get("cool_formal"):
        return "prime-math"
    if intent.get("preserve_motif"):
        return "reservoir-memory"
    if intent.get("preserve_warmth"):
        return "campfire-imagery"
    return None


def _decay_refractory_map(
    values: dict[str, int] | None,
) -> dict[str, int]:
    output = {}
    for key, value in dict(values or {}).items():
        count = int(value or 0) - 1
        if count > 0:
            output[str(key)] = count
    return output


def _apply_refractory_to_terms(
    weighted_terms: dict[str, float],
    refractory_terms: dict[str, int] | None,
    *,
    factor: float,
) -> dict[str, float]:
    output = dict(weighted_terms)
    for term in dict(refractory_terms or {}):
        if term not in output:
            continue
        output[term] = float(output[term]) * factor
        if output[term] <= 0.02:
            output.pop(term, None)
    return output


def _format_controller_regime(state: dict[str, object] | None) -> str | None:
    info = dict(state or {})
    regime = str(info.get("controller_regime") or "").strip()
    if not regime:
        return None
    source = str(info.get("controller_regime_source") or "auto")
    reason = str(info.get("controller_regime_reason") or "").strip()
    turns = int(info.get("controller_regime_turns", 0) or 0)
    transition = str(info.get("controller_regime_transition") or "").strip()
    text = f"Controller regime: {regime} ({source}); turns={turns}."
    if reason:
        text += f" Reason={reason}."
    if transition:
        text += f" Transition={transition}."
    return text


def _score_reservoir_field_alignment(
    candidate_field: dict[str, object] | None,
    predicted_field: dict[str, object] | None,
) -> tuple[float, dict[str, object]]:
    if not candidate_field or not predicted_field:
        return 0.0, {"hits": [], "misses": []}
    predicted = [
        item["label"] for item in list(predicted_field.get("top_anchors", []))[:2]
    ]
    actual = [
        item["label"] for item in list(candidate_field.get("top_anchors", []))[:2]
    ]
    hits = [label for label in predicted if label in actual]
    misses = [label for label in predicted if label not in actual]
    score = float(len(hits)) * 0.95 - float(len(misses)) * 0.35
    return score, {"hits": hits, "misses": misses}


def _combine_system_prompt(system_prompt: str | None, reservoir_note: str | None) -> str | None:
    if not reservoir_note:
        return system_prompt
    if not system_prompt:
        return reservoir_note
    return f"{system_prompt}\n\n{reservoir_note}"


def _resolve_architecture(mode: str, architecture: str) -> str:
    if architecture != "auto":
        return architecture
    return "none" if mode == "helpful" else "reservoir-fixed"


def _resolve_self_tuning(mode: str, architecture: str, setting: str) -> bool:
    if setting == "on":
        return architecture in _RESERVOIR_ARCHITECTURES and mode == "reflective"
    if setting == "off":
        return False
    return architecture in _RESERVOIR_ARCHITECTURES and mode == "reflective"


def _baseline_control_surface(controller: dict[str, object] | None) -> dict[str, float]:
    return {
        "leak_fast": float((controller or {}).get("leak_fast", 0.46)),
        "leak_medium": float((controller or {}).get("leak_medium", 0.18)),
        "leak_slow": float((controller or {}).get("leak_slow", 0.06)),
        "field_alignment_gain": 1.0,
        "prediction_gain": 1.0,
        "novelty_gain": 1.0,
        "motif_gain": 1.0,
        "transform_gain": 1.0,
        "warmth_gain": 1.0,
        "strict_bias": 0.0,
        "temperature_bias": 0.0,
        "exploration_noise": 0.0,
        "washout_strength": 0.75,
    }


def _clamp_control_surface(value: dict[str, float]) -> dict[str, float]:
    output = dict(value)
    for key, bounds in _CONTROL_SURFACE_BOUNDS.items():
        output[key] = min(max(float(output.get(key, 0.0)), bounds[0]), bounds[1])
    return output


def _decay_control_surface(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    rate: float = 0.12,
) -> dict[str, float]:
    return _clamp_control_surface(
        {
            key: float(current.get(key, baseline.get(key, 0.0)))
            + (float(baseline.get(key, 0.0)) - float(current.get(key, baseline.get(key, 0.0)))) * rate
            for key in baseline
        }
    )


def _empty_tuning_state(
    *,
    enabled: bool,
    controller: dict[str, object] | None,
) -> dict[str, object]:
    baseline = _baseline_control_surface(controller)
    return {
        "enabled": enabled,
        "baseline": baseline,
        "effective": dict(baseline),
        "cooldown_remaining": 0,
        "last_condition": None,
        "last_relative_condition": None,
        "last_adjustment": {},
        "last_reason": None,
        "last_model_advice": None,
        "last_model_adjustment": {},
        "stability_score": 1.0,
        "condition_baseline": {key: 0.0 for key in _CONDITION_KEYS},
        "baseline_observations": 0,
        "pressure_counts": {key: 0 for key in _CONDITION_KEYS},
        "calm_streak": 0,
        "advice_history": [],
        "history": [],
    }


def _baseline_regime_policy(regime: str) -> dict[str, object]:
    if regime == "escape":
        return {
            "objective": "leave the stale basin",
            "deltas": {
                "exploration_noise": 0.05,
                "washout_strength": 0.08,
                "field_alignment_gain": 0.10,
                "prediction_gain": 0.08,
                "transform_gain": 0.08,
                "motif_gain": -0.04,
                "strict_bias": -0.03,
            },
            "blend": 0.46,
        }
    if regime == "rebind":
        return {
            "objective": "stabilize the new basin while carrying one ember",
            "deltas": {
                "field_alignment_gain": 0.08,
                "prediction_gain": 0.06,
                "motif_gain": 0.06,
                "transform_gain": 0.03,
                "warmth_gain": 0.04,
                "exploration_noise": -0.01,
            },
            "blend": 0.34,
        }
    if regime == "consolidate":
        return {
            "objective": "make the new basin the default groove",
            "deltas": {
                "field_alignment_gain": 0.06,
                "prediction_gain": 0.04,
                "motif_gain": 0.04,
                "transform_gain": -0.04,
                "exploration_noise": -0.05,
                "washout_strength": -0.06,
                "strict_bias": 0.04,
            },
            "blend": 0.28,
        }
    return {
        "objective": "preserve coherence and warmth",
        "deltas": {
            "motif_gain": 0.04,
            "warmth_gain": 0.03,
            "transform_gain": -0.02,
            "exploration_noise": -0.02,
        },
        "blend": 0.24,
    }


def _apply_regime_policy(
    effective: dict[str, float],
    baseline: dict[str, float],
    *,
    regime: str,
) -> dict[str, float]:
    policy = _baseline_regime_policy(regime)
    target = _apply_control_deltas(
        dict(baseline),
        dict(policy.get("deltas", {}) or {}),
    )
    blend = float(policy.get("blend", 0.30) or 0.30)
    return _clamp_control_surface(
        {
            key: float(effective.get(key, baseline.get(key, 0.0)))
            + (
                float(target.get(key, baseline.get(key, 0.0)))
                - float(effective.get(key, baseline.get(key, 0.0)))
            )
            * blend
            for key in baseline
        }
    )


def _append_controller_regime_history(
    history: list[dict[str, object]] | None,
    *,
    turn: int,
    regime: str,
    source: str,
    reason: str,
    transition: str | None,
) -> list[dict[str, object]]:
    items = list(history or [])
    items.append(
        {
            "turn": turn,
            "regime": regime,
            "source": source,
            "reason": reason,
            "transition": transition,
        }
    )
    return items[-8:]


def _set_active_regime_for_turn(
    state: dict[str, object],
    *,
    regime: str,
    source: str,
    reason: str,
) -> dict[str, object]:
    updated = dict(state)
    previous_regime = str(state.get("controller_regime") or "sustain")
    previous_source = str(state.get("controller_regime_source") or "auto")
    transition = None
    if previous_regime != regime:
        transition = f"{previous_regime}->{regime}"
        updated["controller_previous_regime"] = previous_regime
        updated["controller_regime_turns"] = 0
    else:
        updated["controller_previous_regime"] = state.get("controller_previous_regime")
        updated["controller_regime_turns"] = int(
            state.get("controller_regime_turns", 0) or 0
        )
    if transition or previous_source != source:
        updated["controller_regime_history"] = _append_controller_regime_history(
            list(state.get("controller_regime_history", [])),
            turn=int(state.get("turn_count", 0)) + 1,
            regime=regime,
            source=source,
            reason=reason,
            transition=transition,
        )
    else:
        updated["controller_regime_history"] = list(
            state.get("controller_regime_history", [])
        )
    updated["controller_regime"] = regime
    updated["controller_regime_source"] = source
    updated["controller_regime_reason"] = reason
    updated["controller_regime_transition"] = transition
    if previous_source == "manual" and source == "manual" and previous_regime == regime:
        updated["controller_regime_turns"] = int(
            state.get("controller_regime_turns", 0) or 0
        )
    return updated


def _infer_regulation_regime(
    *,
    state: dict[str, object],
    prompt: str,
    field_intent: dict[str, object] | None,
    mode: str,
    architecture: str,
    controller: dict[str, object] | None,
) -> tuple[str, str]:
    if not _controller_regime_is_active(
        mode=mode,
        architecture=architecture,
        controller=controller,
    ):
        return "sustain", "regime controller inactive outside reflective reservoir mode"
    geometry = dict(state.get("reservoir_geometry", {}) or {})
    tuning = dict(state.get("self_tuning", {}) or {})
    relative = dict(tuning.get("last_relative_condition", {}) or {})
    current = str(state.get("controller_regime") or "sustain")
    current_turns = int(state.get("controller_regime_turns", 0) or 0)
    target_basin = (
        str(state.get("controller_target_basin"))
        if state.get("controller_target_basin")
        else _target_basin_from_field_intent(field_intent)
    )
    actual_top = _field_top_anchor(state.get("embedding_field"))
    collapse = float(geometry.get("geometry_collapse", 0.0) or 0.0)
    persistence = float(geometry.get("attractor_persistence", 0.0) or 0.0)
    field_miss = max(
        float(relative.get("field_miss", 0.0) or 0.0),
        float(relative.get("prediction_mismatch", 0.0) or 0.0) * 0.6,
    )
    attractor_lock = float(relative.get("attractor_lock", 0.0) or 0.0)
    stale_scene = bool(
        _extract_forbidden_scene_terms(
            prompt=prompt,
            field_intent=field_intent,
            reservoir_state=state,
        )
    )
    sticky = (
        collapse >= 0.32
        or persistence >= 0.45
        or field_miss >= 0.22
        or attractor_lock >= 0.22
    )
    break_turn = _is_attractor_break_prompt(prompt, field_intent)
    if geometry.get("insufficient_history") or int(state.get("turn_count", 0) or 0) <= 0:
        return "sustain", "warm-up or ordinary reflective turn"
    if current == "consolidate" and int(tuning.get("calm_streak", 0) or 0) >= 2:
        if collapse < 0.30 and (not target_basin or actual_top == target_basin):
            return "sustain", "consolidated basin is calm enough to sustain"
    if current == "rebind" and target_basin and actual_top == target_basin:
        if current_turns >= 1 and collapse < 0.42:
            return "consolidate", "new basin is holding"
    if current == "escape" and target_basin and actual_top == target_basin:
        if collapse < 0.52:
            return "rebind", "field has left the stale basin"
    if break_turn and sticky:
        return "escape", "break-turn prompt against a sticky basin"
    if current in {"escape", "rebind"} and sticky and stale_scene:
        return "escape", "stale scene still has too much pull"
    return "sustain", "ordinary reflective turn"


def _apply_regime_override(
    *,
    state: dict[str, object],
    inferred_regime: str,
    inferred_reason: str,
    mode: str,
    architecture: str,
    controller: dict[str, object] | None,
) -> tuple[str, str, str]:
    override = state.get("controller_regime_override")
    if override not in _ACTIVE_CONTROLLER_REGIMES:
        return inferred_regime, "auto", inferred_reason
    if not _controller_regime_is_active(
        mode=mode,
        architecture=architecture,
        controller=controller,
    ):
        return (
            "sustain",
            "auto",
            f"manual override {override} inactive outside reflective reservoir mode",
        )
    return str(override), "manual", f"manual override locked to {override}"


def _transition_regulation_regime(
    *,
    state: dict[str, object],
    prompt: str,
    field_intent: dict[str, object] | None,
    actual_field: dict[str, object] | None,
    mode: str,
    architecture: str,
    controller: dict[str, object] | None,
) -> dict[str, object]:
    updated = dict(state)
    if not _controller_regime_is_active(
        mode=mode,
        architecture=architecture,
        controller=controller,
    ):
        return updated
    current = str(updated.get("controller_regime") or "sustain")
    source = str(updated.get("controller_regime_source") or "auto")
    existing_transition = updated.get("controller_regime_transition")
    if source == "manual":
        updated["controller_regime_turns"] = int(
            updated.get("controller_regime_turns", 0) or 0
        ) + 1
        updated["controller_regime_transition"] = existing_transition
        return updated
    geometry = dict(updated.get("reservoir_geometry", {}) or {})
    tuning = dict(updated.get("self_tuning", {}) or {})
    collapse = float(geometry.get("geometry_collapse", 0.0) or 0.0)
    actual_top = _field_top_anchor(actual_field or updated.get("embedding_field"))
    target_basin = (
        str(updated.get("controller_target_basin"))
        if updated.get("controller_target_basin")
        else _target_basin_from_field_intent(field_intent)
    )
    stale_attractor = str(updated.get("controller_stale_attractor") or "") or None
    field_score, _details = _score_field_alignment(actual_field, field_intent)
    next_regime = current
    reason = str(updated.get("controller_regime_reason") or "steady")
    if current == "escape":
        if target_basin and actual_top == target_basin and collapse < 0.52:
            next_regime = "rebind"
            reason = "escape loosened into the target basin"
        elif stale_attractor and actual_top and actual_top != stale_attractor and collapse < 0.44:
            next_regime = "rebind"
            reason = "escape shifted away from the stale basin"
    elif current == "rebind":
        if target_basin and actual_top == target_basin and collapse < 0.40 and field_score >= -0.12:
            next_regime = "consolidate"
            reason = "new basin is stable enough to consolidate"
        elif stale_attractor and actual_top == stale_attractor and collapse >= 0.36:
            next_regime = "escape"
            reason = "stale basin re-entered during rebind"
    elif current == "consolidate":
        if target_basin and actual_top == target_basin:
            if int(tuning.get("calm_streak", 0) or 0) >= 2 and collapse < 0.30:
                next_regime = "sustain"
                reason = "consolidated basin relaxed into sustain"
        elif stale_attractor and actual_top == stale_attractor and collapse >= 0.34:
            next_regime = "escape"
            reason = "old basin is wobbling back in"

    if next_regime != current:
        updated["controller_previous_regime"] = current
        updated["controller_regime"] = next_regime
        updated["controller_regime_source"] = "auto"
        updated["controller_regime_reason"] = reason
        updated["controller_regime_transition"] = f"{current}->{next_regime}"
        updated["controller_regime_turns"] = 0
        updated["controller_regime_history"] = _append_controller_regime_history(
            list(updated.get("controller_regime_history", [])),
            turn=int(updated.get("turn_count", 0) or 0),
            regime=next_regime,
            source="auto",
            reason=reason,
            transition=f"{current}->{next_regime}",
        )
    else:
        updated["controller_regime_transition"] = existing_transition
        updated["controller_regime_turns"] = int(
            updated.get("controller_regime_turns", 0) or 0
        ) + 1
    return updated


def _update_condition_baseline(
    baseline: dict[str, float] | None,
    condition: dict[str, float],
    *,
    observations: int,
) -> dict[str, float]:
    current = {key: float((baseline or {}).get(key, 0.0) or 0.0) for key in _CONDITION_KEYS}
    if observations <= 0:
        return {
            key: float(condition.get(key, 0.0) or 0.0)
            for key in _CONDITION_KEYS
        }
    rate = 0.24 if observations < 4 else 0.12
    return {
        key: current[key] + (float(condition.get(key, 0.0) or 0.0) - current[key]) * rate
        for key in _CONDITION_KEYS
    }


def _relative_condition_vector(
    condition: dict[str, float],
    baseline: dict[str, float] | None,
    *,
    observations: int,
) -> dict[str, float]:
    if observations < _BASELINE_BOOTSTRAP_OBSERVATIONS:
        relative = {
            key: float(condition.get(key, 0.0) or 0.0)
            for key in _CONDITION_KEYS
        }
        relative["severity"] = float(condition.get("severity", 0.0) or 0.0)
        return relative
    output: dict[str, float] = {}
    severity_components = []
    for key in _CONDITION_KEYS:
        raw = float(condition.get(key, 0.0) or 0.0)
        base = float((baseline or {}).get(key, 0.0) or 0.0)
        margin = max(0.06, base * 0.22)
        relative = max(raw - base - margin, 0.0)
        scaled = min(relative / max(0.20, 1.0 - base), 1.0)
        output[key] = scaled
        severity_components.append(max(raw * 0.35, scaled))
    output["severity"] = (
        sum(severity_components) / len(severity_components)
        if severity_components
        else 0.0
    )
    return output


def _update_pressure_counts(
    previous: dict[str, int] | None,
    condition: dict[str, float],
) -> dict[str, int]:
    counts = {key: int((previous or {}).get(key, 0) or 0) for key in _CONDITION_KEYS}
    for key in _CONDITION_KEYS:
        value = float(condition.get(key, 0.0) or 0.0)
        if value >= 0.45:
            counts[key] = min(counts[key] + 1, 8)
        elif value <= 0.18:
            counts[key] = max(counts[key] - 1, 0)
    return counts


def _condition_gate(
    condition: dict[str, float],
    pressure_counts: dict[str, int] | None,
    key: str,
    *,
    threshold: float = 0.45,
    streak: int = 2,
    spike: float = 0.82,
) -> bool:
    value = float(condition.get(key, 0.0) or 0.0)
    count = int((pressure_counts or {}).get(key, 0) or 0)
    return value >= spike or (value >= threshold and count >= streak)


def _reservoir_input_feature_names() -> list[str]:
    names = [f"prompt_field:{label}" for label in _RESERVOIR_ANCHOR_LABELS]
    names.extend(f"intent_target:{label}" for label in _RESERVOIR_ANCHOR_LABELS)
    names.extend(f"intent_avoid:{label}" for label in _RESERVOIR_ANCHOR_LABELS)
    names.extend(
        (
            "flag:preserve_warmth",
            "flag:preserve_motif",
            "flag:cool_formal",
            "lexical:prime",
            "lexical:reservoir",
            "lexical:reflection",
            "lexical:image_density",
            "lexical:token_density",
        )
    )
    return names


def _reservoir_feedback_feature_names() -> list[str]:
    names = [f"prev_field:{label}" for label in _RESERVOIR_ANCHOR_LABELS]
    names.extend(
        (
            "prev_behavior:novelty",
            "prev_behavior:motif",
            "prev_behavior:transform",
            "prev_behavior:warmth",
            "prev_behavior:formal",
        )
    )
    return names


def _stable_unit_float(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _stable_signed_float(key: str) -> float:
    return (_stable_unit_float(key) * 2.0) - 1.0


def _vector_dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = _vector_norm(vector)
    if norm <= 1e-8:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _vector_add(*vectors: list[float]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    return [
        sum(vector[index] for vector in vectors if index < len(vector))
        for index in range(size)
    ]


def _vector_subtract(left: list[float], right: list[float]) -> list[float]:
    return [
        before - after
        for before, after in zip(left, right)
    ]


def _vector_scale(vector: list[float], factor: float) -> list[float]:
    return [value * factor for value in vector]


def _vector_blend(previous: list[float], current: list[float], leak: float) -> list[float]:
    return [
        ((1.0 - leak) * before) + (leak * after)
        for before, after in zip(previous, current)
    ]


def _vector_cosine(left: list[float], right: list[float]) -> float:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return _vector_dot(left, right) / (left_norm * right_norm)


def _reservoir_geometry_entry(latent: dict[str, object]) -> dict[str, object]:
    predicted_field = dict(latent.get("predicted_field", {}) or {})
    top_anchors = list(predicted_field.get("top_anchors", []))
    return {
        "combined": list(latent.get("combined", [])),
        "norm": float(latent.get("state_norm", 0.0) or 0.0),
        "drift": latent.get("state_drift"),
        "attractor": predicted_field.get("dominant_attractor")
        or (top_anchors[0]["label"] if top_anchors else None),
    }


def _trajectory_participation_ratio(vectors: list[list[float]]) -> float:
    if not vectors:
        return 0.0
    diag_sum = sum(_vector_dot(vector, vector) for vector in vectors)
    if diag_sum <= 1e-8:
        return 0.0
    fro_sq = 0.0
    for left in vectors:
        for right in vectors:
            dot = _vector_dot(left, right)
            fro_sq += dot * dot
    if fro_sq <= 1e-8:
        return 0.0
    return (diag_sum * diag_sum) / fro_sq


def _compute_reservoir_geometry(
    history: list[dict[str, object]] | None,
) -> dict[str, object] | None:
    items = list(history or [])
    if not items:
        return None
    vectors = [list(item.get("combined", [])) for item in items if item.get("combined")]
    if not vectors:
        return None
    if len(vectors) < 2:
        attractors = [item.get("attractor") for item in items if item.get("attractor")]
        norms = [
            float(item.get("norm"))
            for item in items
            if isinstance(item.get("norm"), (int, float))
        ]
        norm_mean = sum(norms) / float(len(norms)) if norms else 0.0
        return {
            "window": len(vectors),
            "pairwise_distance": 0.0,
            "mean_drift": 0.0,
            "participation_ratio": 1.0,
            "normalized_rank": 1.0,
            "rank_collapse": 0.0,
            "spread_collapse": 0.0,
            "drift_collapse": 0.0,
            "attractor_persistence": 0.0,
            "norm_mean": norm_mean,
            "norm_std": 0.0,
            "geometry_collapse": 0.0,
            "dominant_attractor": attractors[-1] if attractors else None,
            "insufficient_history": True,
        }
    pairwise_distance = _average_pairwise_distance(vectors)
    drifts = [
        float(item.get("drift"))
        for item in items
        if isinstance(item.get("drift"), (int, float))
    ]
    mean_drift = sum(drifts) / float(len(drifts)) if drifts else 0.0
    norms = [
        float(item.get("norm"))
        for item in items
        if isinstance(item.get("norm"), (int, float))
    ]
    norm_mean = sum(norms) / float(len(norms)) if norms else 0.0
    norm_var = (
        sum((value - norm_mean) * (value - norm_mean) for value in norms) / float(len(norms))
        if norms
        else 0.0
    )
    attractors = [item.get("attractor") for item in items if item.get("attractor")]
    persistence = 0.0
    if attractors:
        current = attractors[-1]
        run = 0
        for item in reversed(attractors):
            if item != current:
                break
            run += 1
        persistence = float(run) / float(len(attractors))
    participation = _trajectory_participation_ratio(vectors)
    normalized_rank = min(participation / float(max(len(vectors), 1)), 1.0)
    rank_collapse = max(1.0 - normalized_rank, 0.0)
    spread_collapse = max(1.0 - min(pairwise_distance / 0.28, 1.0), 0.0)
    drift_collapse = max(1.0 - min(mean_drift / 0.18, 1.0), 0.0)
    geometry_collapse = (
        rank_collapse * 0.40
        + spread_collapse * 0.25
        + drift_collapse * 0.20
        + persistence * 0.15
    )
    return {
        "window": len(vectors),
        "pairwise_distance": pairwise_distance,
        "mean_drift": mean_drift,
        "participation_ratio": participation,
        "normalized_rank": normalized_rank,
        "rank_collapse": rank_collapse,
        "spread_collapse": spread_collapse,
        "drift_collapse": drift_collapse,
        "attractor_persistence": persistence,
        "norm_mean": norm_mean,
        "norm_std": math.sqrt(max(norm_var, 0.0)),
        "geometry_collapse": max(min(geometry_collapse, 1.0), 0.0),
        "dominant_attractor": attractors[-1] if attractors else None,
    }


def _reservoir_geometry_summary_entry(geometry: dict[str, object]) -> dict[str, object]:
    return {
        "collapse": float(geometry.get("geometry_collapse", 0.0) or 0.0),
        "rank": float(geometry.get("normalized_rank", 1.0) or 1.0),
        "spread": float(geometry.get("pairwise_distance", 0.0) or 0.0),
        "attractor": geometry.get("dominant_attractor"),
        "insufficient_history": bool(geometry.get("insufficient_history")),
    }


def _format_reservoir_geometry(geometry: dict[str, object] | None) -> str | None:
    if not geometry:
        return None
    if geometry.get("insufficient_history"):
        return (
            "Reservoir geometry: warming up; trajectory evidence is not rich enough yet."
        )
    return (
        "Reservoir geometry: "
        f"rank={float(geometry.get('normalized_rank', 0.0)):.2f}; "
        f"spread={float(geometry.get('pairwise_distance', 0.0)):.2f}; "
        f"drift={float(geometry.get('mean_drift', 0.0)):.2f}; "
        f"persistence={float(geometry.get('attractor_persistence', 0.0)):.2f}; "
        f"collapse={float(geometry.get('geometry_collapse', 0.0)):.2f}."
    )


def _format_reservoir_geometry_detail(geometry: dict[str, object] | None) -> str | None:
    if not geometry:
        return None
    if geometry.get("insufficient_history"):
        return "Geometry is still warming up, so collapse and rank are only placeholders right now."
    collapse = float(geometry.get("geometry_collapse", 0.0) or 0.0)
    rank = float(geometry.get("normalized_rank", 1.0) or 1.0)
    spread = float(geometry.get("pairwise_distance", 0.0) or 0.0)
    persistence = float(geometry.get("attractor_persistence", 0.0) or 0.0)
    attractor = geometry.get("dominant_attractor") or "n/a"
    if collapse < 0.18:
        mood = "open and mobile"
    elif collapse < 0.40:
        mood = "focused but still moving"
    elif collapse < 0.65:
        mood = "sticky around one region"
    else:
        mood = "collapsed into a narrow regime"
    return (
        f"Geometry feels {mood} around {attractor}. "
        f"Rank={rank:.2f}, spread={spread:.2f}, persistence={persistence:.2f}, collapse={collapse:.2f}."
    )


def _format_reservoir_geometry_trajectory(
    history: list[dict[str, object]] | None,
) -> str | None:
    items = list(history or [])
    if not items:
        return None
    pieces = []
    for item in items[-4:]:
        if item.get("insufficient_history"):
            pieces.append("warmup")
            continue
        attractor = item.get("attractor") or "n/a"
        pieces.append(f"{attractor}@{float(item.get('collapse', 0.0) or 0.0):.2f}")
    if not pieces:
        return None
    return "Geometry trajectory: " + " -> ".join(pieces) + "."


def _sparse_matrix_vector(matrix: list[list[tuple[int, float]]], vector: list[float]) -> list[float]:
    values = []
    for row in matrix:
        total = 0.0
        for column, weight in row:
            if column < len(vector):
                total += weight * vector[column]
        values.append(total)
    return values


def _dense_matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_vector_dot(row, vector) for row in matrix]


def _exploration_noise_vector(
    controller: dict[str, object],
    *,
    previous_combined: list[float],
    input_vector: list[float],
    feedback_vector: list[float],
    amplitude: float,
) -> list[float]:
    if amplitude <= 1e-6:
        return [0.0 for _ in range(int(controller["dim"]))]
    previous_signature = round(sum(previous_combined), 4)
    input_signature = round(sum(input_vector), 4)
    feedback_signature = round(sum(feedback_vector), 4)
    key = (
        f"{controller['seed']}:{previous_signature}:{input_signature}:{feedback_signature}:"
        f"{round(float(amplitude), 4)}"
    )
    return [
        _stable_signed_float(f"{key}:{index}") * float(amplitude)
        for index in range(int(controller["dim"]))
    ]


def _build_sparse_matrix(
    *,
    rows: int,
    cols: int,
    key: str,
    density: float,
    scale: float,
) -> list[list[tuple[int, float]]]:
    rng = random.Random(key)
    matrix: list[list[tuple[int, float]]] = []
    for row_index in range(rows):
        row = []
        for column_index in range(cols):
            if rng.random() <= density:
                row.append((column_index, (rng.random() * 2.0 - 1.0) * scale))
        if not row:
            column_index = rng.randrange(cols)
            row.append((column_index, (rng.random() * 2.0 - 1.0) * scale))
        matrix.append(row)
    max_row_sum = max(sum(abs(weight) for _column, weight in row) for row in matrix)
    factor = 0.82 / max(max_row_sum, 1e-6)
    return [
        [(column, weight * factor) for column, weight in row]
        for row in matrix
    ]


def _build_dense_matrix(
    *,
    rows: int,
    cols: int,
    key: str,
    scale: float,
) -> list[list[float]]:
    rng = random.Random(key)
    return [
        [((rng.random() * 2.0) - 1.0) * scale for _ in range(cols)]
        for _ in range(rows)
    ]


def _empty_reservoir_latent(dim: int, feedback_dim: int) -> dict[str, object]:
    zero = [0.0 for _ in range(dim)]
    return {
        "fast": list(zero),
        "medium": list(zero),
        "slow": list(zero),
        "combined": list(zero),
        "last_feedback_vector": [0.0 for _ in range(feedback_dim)],
        "state_drift": None,
        "state_norm": 0.0,
        "predicted_field": None,
        "predicted_behaviors": None,
        "prediction_match": None,
        "last_prediction_score": None,
        "washout_count": 0,
    }


def _clip_vector(vector: list[float], *, limit: float = 1.0) -> list[float]:
    return [max(min(value, limit), -limit) for value in vector]


def _feature_vector(
    feature_names: list[str],
    feature_map: dict[str, float] | None,
) -> list[float]:
    mapping = feature_map or {}
    return [float(mapping.get(name, 0.0)) for name in feature_names]


def _build_reservoir_feedback_map(
    *,
    field: dict[str, object] | None,
    result: dict[str, object] | None,
) -> dict[str, float]:
    scores = dict((field or {}).get("anchor_scores", {}))
    output = {f"prev_field:{label}": float(scores.get(label, 0.0)) for label in _RESERVOIR_ANCHOR_LABELS}
    candidate_score = float((result or {}).get("candidate_score", 0.0) or 0.0)
    fresh_words = list((result or {}).get("candidate_fresh_words", []))
    reservoir_hits = list((result or {}).get("candidate_reservoir_hits", []))
    distance_balance = float((result or {}).get("candidate_follow_up_distance_balance", 0.0) or 0.0)
    output["prev_behavior:novelty"] = min(len(fresh_words), 4) / 4.0
    output["prev_behavior:motif"] = min(len(reservoir_hits), 3) / 3.0
    output["prev_behavior:transform"] = max(min(distance_balance, 1.0), 0.0)
    output["prev_behavior:warmth"] = max(float(scores.get("campfire-imagery", 0.0)), 0.0)
    output["prev_behavior:formal"] = max(float(scores.get("prime-math", 0.0)), 0.0)
    output["prev_behavior:novelty"] += max(min(candidate_score / 8.0, 1.0), -1.0) * 0.15
    return output


def _build_reservoir_input_map(
    *,
    prompt: str,
    field_intent: dict[str, object] | None,
    prompt_field: dict[str, object] | None,
) -> dict[str, float]:
    lowered = str(prompt or "").lower()
    scores = dict((prompt_field or {}).get("anchor_scores", {}))
    feature_map = {
        f"prompt_field:{label}": float(scores.get(label, 0.0))
        for label in _RESERVOIR_ANCHOR_LABELS
    }
    targets = set((field_intent or {}).get("targets", []))
    avoid = set((field_intent or {}).get("avoid", []))
    for label in _RESERVOIR_ANCHOR_LABELS:
        feature_map[f"intent_target:{label}"] = 1.0 if label in targets else 0.0
        feature_map[f"intent_avoid:{label}"] = 1.0 if label in avoid else 0.0
    feature_map["flag:preserve_warmth"] = 1.0 if (field_intent or {}).get("preserve_warmth") else 0.0
    feature_map["flag:preserve_motif"] = 1.0 if (field_intent or {}).get("preserve_motif") else 0.0
    feature_map["flag:cool_formal"] = 1.0 if (field_intent or {}).get("cool_formal") else 0.0
    feature_map["lexical:prime"] = 1.0 if ("prime" in lowered or re.search(r"\b17\b", lowered)) else 0.0
    feature_map["lexical:reservoir"] = 1.0 if any(token in lowered for token in ("reservoir", "recurrent", "echo state")) else 0.0
    feature_map["lexical:reflection"] = 1.0 if any(token in lowered for token in ("feel", "favorite", "introspection", "inner weather")) else 0.0
    content_words = _tokenize_content_words(prompt)
    image_hits = sum(1 for token in content_words if token in _CONCRETE_IMAGE_CUES)
    feature_map["lexical:image_density"] = min(image_hits / 4.0, 1.0)
    feature_map["lexical:token_density"] = min(len(content_words) / 18.0, 1.0)
    return feature_map


def _predict_reservoir_readout(
    controller: dict[str, object],
    combined: list[float],
) -> dict[str, object]:
    field_weights = dict(controller.get("field_readout", {}))
    anchor_scores = {
        label: _vector_cosine(combined, list(weight))
        for label, weight in field_weights.items()
    }
    top_anchors = sorted(
        ({"label": label, "score": score} for label, score in anchor_scores.items()),
        key=lambda item: item["score"],
        reverse=True,
    )[:3]
    shifted = [max(score + 1.05, 1e-6) for score in anchor_scores.values()]
    total = sum(shifted) or 1.0
    entropy_like = -sum((value / total) * math.log(value / total) for value in shifted)
    behavior_weights = dict(controller.get("behavior_readout", {}))
    behavior_scores = {
        label: _vector_cosine(combined, list(weight))
        for label, weight in behavior_weights.items()
    }
    return {
        "anchor_scores": anchor_scores,
        "top_anchors": top_anchors,
        "dominant_attractor": top_anchors[0]["label"] if top_anchors else None,
        "entropy_like": entropy_like,
        "behavior_scores": behavior_scores,
    }


def _preview_reservoir_step(
    controller: dict[str, object],
    latent: dict[str, object] | None,
    *,
    input_vector: list[float],
    control_surface: dict[str, float] | None = None,
    feedback_vector: list[float] | None = None,
) -> dict[str, object]:
    current = latent or _empty_reservoir_latent(
        int(controller["dim"]),
        len(list(controller["feedback_feature_names"])),
    )
    fast = list(current.get("fast", []))
    medium = list(current.get("medium", []))
    slow = list(current.get("slow", []))
    previous_combined = list(current.get("combined", []))
    if not fast:
        fast = [0.0 for _ in range(int(controller["dim"]))]
        medium = [0.0 for _ in range(int(controller["dim"]))]
        slow = [0.0 for _ in range(int(controller["dim"]))]
        previous_combined = [0.0 for _ in range(int(controller["dim"]))]
    feedback = list(feedback_vector or current.get("last_feedback_vector", []))
    if not feedback:
        feedback = [0.0 for _ in range(len(list(controller["feedback_feature_names"])))]
    control = _clamp_control_surface(
        {
            **_baseline_control_surface(controller),
            **dict(control_surface or {}),
        }
    )
    recurrent = _sparse_matrix_vector(list(controller["W"]), fast)
    incoming = _dense_matrix_vector(list(controller["Win"]), input_vector)
    feedback_term = _dense_matrix_vector(list(controller["Wfb"]), feedback)
    activated = [
        math.tanh(recurrent[index] + incoming[index] + feedback_term[index])
        for index in range(int(controller["dim"]))
    ]
    exploration_noise = _exploration_noise_vector(
        controller,
        previous_combined=previous_combined,
        input_vector=input_vector,
        feedback_vector=feedback,
        amplitude=float(control.get("exploration_noise", 0.0) or 0.0),
    )
    activated = [
        max(min(activated[index] + exploration_noise[index], 1.0), -1.0)
        for index in range(int(controller["dim"]))
    ]
    fast_next = _vector_blend(fast, activated, float(control["leak_fast"]))
    medium_next = _vector_blend(medium, fast_next, float(control["leak_medium"]))
    slow_next = _vector_blend(slow, medium_next, float(control["leak_slow"]))
    combined = _vector_add(
        _vector_scale(fast_next, 0.55),
        _vector_scale(medium_next, 0.30),
        _vector_scale(slow_next, 0.15),
    )
    drift = None
    if any(abs(value) > 1e-9 for value in previous_combined):
        drift = max(0.0, 1.0 - _vector_cosine(previous_combined, combined))
    readout = _predict_reservoir_readout(controller, combined)
    return {
        "fast": fast_next,
        "medium": medium_next,
        "slow": slow_next,
        "combined": combined,
        "state_norm": _vector_norm(combined),
        "state_drift": drift,
        "predicted_field": {
            "top_anchors": list(readout["top_anchors"]),
            "anchor_scores": dict(readout["anchor_scores"]),
            "dominant_attractor": readout["dominant_attractor"],
            "entropy_like": readout["entropy_like"],
        },
        "predicted_behaviors": dict(readout["behavior_scores"]),
        "last_feedback_vector": feedback,
        "prediction_match": None,
        "last_prediction_score": None,
        "washout_count": int(current.get("washout_count", 0)),
        "control_surface": control,
        "exploration_noise": float(control.get("exploration_noise", 0.0) or 0.0),
    }


def _format_predicted_field(readout: dict[str, object] | None) -> str | None:
    if not readout:
        return None
    anchors = list(readout.get("top_anchors", []))
    if not anchors:
        return None
    summary = ", ".join(item["label"] for item in anchors[:2])
    dominant = readout.get("dominant_attractor")
    if dominant:
        return f"Reservoir pull: attractor={dominant}; next field leans toward {summary}."
    return f"Reservoir pull leans toward {summary}."


def _score_prediction_match(
    predicted_field: dict[str, object] | None,
    actual_field: dict[str, object] | None,
) -> dict[str, object] | None:
    if not predicted_field or not actual_field:
        return None
    predicted_labels = [
        item["label"] for item in list(predicted_field.get("top_anchors", []))[:2]
    ]
    actual_labels = [
        item["label"] for item in list(actual_field.get("top_anchors", []))[:2]
    ]
    if not predicted_labels or not actual_labels:
        return None
    hits = [label for label in predicted_labels if label in actual_labels]
    misses = [label for label in predicted_labels if label not in actual_labels]
    score = float(len(hits)) / float(max(1, min(len(predicted_labels), len(actual_labels))))
    if score >= 1.0:
        summary = "Reservoir prediction matched the chosen field."
    elif hits:
        summary = (
            "Reservoir prediction partly matched the chosen field via "
            + ", ".join(hits)
            + "."
        )
    else:
        summary = (
            "Reservoir prediction missed the chosen field; expected "
            + ", ".join(predicted_labels)
            + "."
        )
    return {
        "score": score,
        "hits": hits,
        "misses": misses,
        "summary": summary,
    }


def _update_trainable_readout(
    controller: dict[str, object],
    *,
    combined: list[float],
    actual_field: dict[str, object] | None,
) -> None:
    if controller.get("architecture") != "reservoir-trainable-readout":
        return
    actual_labels = [
        item["label"] for item in list((actual_field or {}).get("top_anchors", []))[:2]
    ]
    if not combined or not actual_labels:
        return
    learning_rate = 0.12
    for label, weight in dict(controller.get("field_readout", {})).items():
        target = 1.0 if label in actual_labels else -0.18
        updated = [
            ((1.0 - learning_rate) * current) + (learning_rate * target * state_value)
            for current, state_value in zip(weight, combined)
        ]
        controller["field_readout"][label] = _normalize_vector(updated)


def _build_reservoir_controller(
    architecture: str,
    *,
    dim: int = _DEFAULT_RESERVOIR_DIM,
    seed: int = _DEFAULT_RESERVOIR_SEED,
) -> dict[str, object] | None:
    if architecture not in _RESERVOIR_ARCHITECTURES:
        return None
    input_feature_names = _reservoir_input_feature_names()
    feedback_feature_names = _reservoir_feedback_feature_names()
    controller = {
        "architecture": architecture,
        "dim": dim,
        "seed": seed,
        "input_feature_names": input_feature_names,
        "feedback_feature_names": feedback_feature_names,
        "leak_fast": 0.46,
        "leak_medium": 0.18,
        "leak_slow": 0.06,
        "W": _build_sparse_matrix(
            rows=dim,
            cols=dim,
            key=f"{seed}:W",
            density=0.12,
            scale=0.55,
        ),
        "Win": _build_dense_matrix(
            rows=dim,
            cols=len(input_feature_names),
            key=f"{seed}:Win",
            scale=0.28,
        ),
        "Wfb": _build_dense_matrix(
            rows=dim,
            cols=len(feedback_feature_names),
            key=f"{seed}:Wfb",
            scale=0.18,
        ),
    }
    zero_latent = _empty_reservoir_latent(dim, len(feedback_feature_names))
    field_readout = {}
    for label in _RESERVOIR_ANCHOR_LABELS:
        probe_map = _build_reservoir_input_map(
            prompt=label.replace("-", " "),
            field_intent={
                "targets": [label],
                "avoid": [],
                "preserve_warmth": label == "campfire-imagery",
                "preserve_motif": label == "reservoir-memory",
                "cool_formal": label == "prime-math",
            },
            prompt_field={
                "anchor_scores": {anchor_label: 1.0 if anchor_label == label else 0.0 for anchor_label in _RESERVOIR_ANCHOR_LABELS},
            },
        )
        preview = _preview_reservoir_step(
            controller,
            zero_latent,
            input_vector=_feature_vector(input_feature_names, probe_map),
        )
        field_readout[label] = _normalize_vector(list(preview["combined"]))
    behavior_readout = {}
    for label in _RESERVOIR_BEHAVIOR_LABELS:
        probe_map = {
            "flag:preserve_warmth": 1.0 if label == "warmth" else 0.0,
            "flag:preserve_motif": 1.0 if label in ("motif", "transform") else 0.0,
            "flag:cool_formal": 1.0 if label == "formal" else 0.0,
            "lexical:reservoir": 1.0 if label in ("motif", "transform") else 0.0,
            "lexical:reflection": 1.0 if label == "novelty" else 0.0,
        }
        preview = _preview_reservoir_step(
            controller,
            zero_latent,
            input_vector=_feature_vector(input_feature_names, probe_map),
        )
        behavior_readout[label] = _normalize_vector(list(preview["combined"]))
    controller["field_readout"] = field_readout
    controller["behavior_readout"] = behavior_readout
    return controller


def _prepare_turn_reservoir(
    *,
    architecture: str,
    controller: dict[str, object] | None,
    state: dict[str, object] | None,
    prompt: str,
    field_intent: dict[str, object] | None,
    prompt_field: dict[str, object] | None,
) -> dict[str, object]:
    if architecture not in _RESERVOIR_ARCHITECTURES or not controller:
        return {
            "architecture": architecture,
            "prompt_field": prompt_field,
            "prompt_feature_map": {},
            "prompt_feature_vector": [],
            "preview": None,
            "prediction_note": None,
        }
    latent = dict((state or {}).get("reservoir_latent", {}))
    if not latent:
        latent = _empty_reservoir_latent(
            int(controller["dim"]),
            len(list(controller["feedback_feature_names"])),
        )
    prompt_feature_map = _build_reservoir_input_map(
        prompt=prompt,
        field_intent=field_intent,
        prompt_field=prompt_field,
    )
    prompt_feature_vector = _feature_vector(
        list(controller["input_feature_names"]),
        prompt_feature_map,
    )
    preview = _preview_reservoir_step(
        controller,
        latent,
        input_vector=prompt_feature_vector,
        control_surface=_current_control_surface(state, controller),
    )
    return {
        "architecture": architecture,
        "prompt_field": prompt_field,
        "prompt_feature_map": prompt_feature_map,
        "prompt_feature_vector": prompt_feature_vector,
        "preview": preview,
        "prediction_note": _format_predicted_field(preview.get("predicted_field")),
    }


def _commit_reservoir_turn(
    *,
    controller: dict[str, object] | None,
    state: dict[str, object],
    prepared: dict[str, object] | None,
    actual_field: dict[str, object] | None,
    result: dict[str, object] | None,
) -> dict[str, object]:
    if not controller or not prepared or not prepared.get("preview"):
        return state
    preview = dict(prepared["preview"])
    feedback_map = _build_reservoir_feedback_map(field=actual_field, result=result)
    preview["last_feedback_vector"] = _feature_vector(
        list(controller["feedback_feature_names"]),
        feedback_map,
    )
    match = _score_prediction_match(preview.get("predicted_field"), actual_field)
    preview["prediction_match"] = match
    preview["last_prediction_score"] = None if not match else match["score"]
    _update_trainable_readout(
        controller,
        combined=list(preview.get("combined", [])),
        actual_field=actual_field,
    )
    updated = dict(state)
    updated["reservoir_latent"] = preview
    updated["reservoir_prompt_features"] = dict(prepared.get("prompt_feature_map", {}))
    return updated


def _format_reservoir_dynamics(state: dict[str, object]) -> str | None:
    latent = dict(state.get("reservoir_latent") or {})
    predicted_field = dict(latent.get("predicted_field") or {})
    anchors = list(predicted_field.get("top_anchors", []))
    if not latent or not anchors:
        return None
    top_summary = ", ".join(item["label"] for item in anchors[:2])
    norm = float(latent.get("state_norm", 0.0) or 0.0)
    entropy = float(predicted_field.get("entropy_like", 0.0) or 0.0)
    drift = latent.get("state_drift")
    exploration = float(latent.get("exploration_noise", 0.0) or 0.0)
    drift_text = f"{float(drift):.2f}" if isinstance(drift, (int, float)) else "n/a"
    attractor = predicted_field.get("dominant_attractor") or "n/a"
    note = (
        "Reservoir dynamics: "
        f"architecture={state.get('architecture', 'lexical')}; "
        f"attractor={attractor}; top pulls={top_summary}; norm={norm:.2f}; "
        f"entropy={entropy:.2f}; drift={drift_text}; exploration={exploration:.2f}."
    )
    match = dict(latent.get("prediction_match", {}) or {})
    if match.get("summary"):
        note += " " + str(match["summary"])
    return note


def _current_control_surface(
    state: dict[str, object] | None,
    controller: dict[str, object] | None,
) -> dict[str, float]:
    tuning = dict((state or {}).get("self_tuning", {}) or {})
    effective = dict(tuning.get("effective", {}) or {})
    baseline = _baseline_control_surface(controller)
    merged = dict(baseline)
    merged.update(effective)
    return _clamp_control_surface(merged)


def _format_condition_vector(condition: dict[str, float] | None) -> str | None:
    if not condition:
        return None
    items = sorted(
        (
            (label, float(value))
            for label, value in condition.items()
            if label != "severity" and isinstance(value, (int, float)) and value >= 0.18
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not items:
        severity = float(condition.get("severity", 0.0) or 0.0)
        return f"Condition monitor: calm; severity={severity:.2f}."
    summary = ", ".join(f"{label}={value:.2f}" for label, value in items[:4])
    severity = float(condition.get("severity", 0.0) or 0.0)
    return f"Condition monitor: {summary}; severity={severity:.2f}."


def _format_tuning_state(state: dict[str, object] | None) -> str | None:
    tuning = dict((state or {}).get("self_tuning", {}) or {})
    if not tuning:
        return None
    enabled = bool(tuning.get("enabled"))
    if not enabled:
        return "Self-tuning: disabled."
    effective = dict(tuning.get("effective", {}) or {})
    cooldown = int(tuning.get("cooldown_remaining", 0) or 0)
    stability = float(tuning.get("stability_score", 1.0) or 1.0)
    calm_streak = int(tuning.get("calm_streak", 0) or 0)
    adjustment = dict(tuning.get("last_adjustment", {}) or {})
    active = [
        f"{key}={float(value):+.2f}"
        for key, value in sorted(adjustment.items())
        if abs(float(value)) >= 0.015
    ]
    controller_regime = str((state or {}).get("controller_regime") or "sustain")
    note = (
        f"Self-tuning: regime={controller_regime}; stability={stability:.2f}; "
        f"cooldown={cooldown}; calm={calm_streak}."
    )
    if active:
        note += " Last deltas=" + ", ".join(active[:5]) + "."
    model_adjustment = dict(tuning.get("last_model_adjustment", {}) or {})
    model_active = [
        f"{key}={float(value):+.2f}"
        for key, value in sorted(model_adjustment.items())
        if abs(float(value)) >= 0.004
    ]
    if model_active:
        note += " Model deltas=" + ", ".join(model_active[:4]) + "."
    if model_active and active:
        note += " Source=mixed."
    elif model_active:
        note += " Source=advisor-led."
    elif active:
        note += " Source=heuristic."
    relative = dict(tuning.get("last_relative_condition", {}) or {})
    active_relative = sorted(
        (
            (key, float(value))
            for key, value in relative.items()
            if key != "severity" and isinstance(value, (int, float)) and value >= 0.18
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if active_relative:
        note += " Relative pressure=" + ", ".join(
            f"{key}={value:.2f}" for key, value in active_relative[:3]
        ) + "."
    reason = tuning.get("last_reason")
    if reason:
        note += f" Reason: {reason}."
    model_advice = dict(tuning.get("last_model_advice", {}) or {})
    if model_advice.get("reason"):
        note += f" Advisor: {model_advice['reason']}."
    return note


def _format_control_surface(
    state: dict[str, object] | None,
    controller: dict[str, object] | None,
) -> str | None:
    tuning = dict((state or {}).get("self_tuning", {}) or {})
    if not tuning or not bool(tuning.get("enabled")):
        return None
    baseline = dict(tuning.get("baseline", {}) or _baseline_control_surface(controller))
    effective = _current_control_surface(state, controller)
    deltas = []
    for key in (
        "leak_fast",
        "leak_medium",
        "leak_slow",
        "field_alignment_gain",
        "prediction_gain",
        "novelty_gain",
        "motif_gain",
        "transform_gain",
        "warmth_gain",
        "strict_bias",
        "temperature_bias",
        "exploration_noise",
        "washout_strength",
    ):
        delta = float(effective.get(key, baseline.get(key, 0.0))) - float(
            baseline.get(key, 0.0)
        )
        if abs(delta) >= 0.004:
            deltas.append(f"{key}={delta:+.2f}")
    if not deltas:
        return "Control surface: baseline."
    return "Control surface: " + ", ".join(deltas[:6]) + "."


def _build_self_regulation_prompt(
    *,
    prompt: str,
    result: dict[str, object],
    condition: dict[str, float],
    relative_condition: dict[str, float],
    forecast: dict[str, object] | None,
    observer_report: dict[str, object] | None,
    change_report: dict[str, object] | None,
    field_intent: dict[str, object] | None,
    geometry: dict[str, object] | None,
    intervention_report: dict[str, object] | None,
    control_surface_note: str | None,
    controller_regime: str | None,
    controller_regime_source: str | None,
) -> str:
    allowed = ", ".join(_MODEL_ADVICE_CONTROL_KEYS)
    break_turn = _is_attractor_break_prompt(prompt, field_intent)
    regime = str(controller_regime or "sustain")
    regime_source = str(controller_regime_source or "auto")
    if regime == "escape":
        regime_guidance = (
            "Current regime: escape.\n"
            "- Prioritize leaving the stale basin over simply calming the prose.\n"
            "- Prefer positive escape levers and tolerate a little instability if geometry loosens.\n\n"
        )
    elif regime == "rebind":
        regime_guidance = (
            "Current regime: rebind.\n"
            "- Preserve one ember from the prior basin while strengthening the requested field.\n"
            "- Prefer medium-timescale carry and target-basin reinforcement over washout.\n\n"
        )
    elif regime == "consolidate":
        regime_guidance = (
            "Current regime: consolidate.\n"
            "- Reduce unnecessary exploration and make the new basin hold.\n"
            "- Prefer stability, field alignment, and low relapse risk.\n\n"
        )
    else:
        regime_guidance = (
            "Current regime: sustain.\n"
            "- Preserve coherence and warmth unless telemetry clearly demands a harder move.\n\n"
        )
    break_guidance = (
        "Break-turn guidance:\n"
        "- This prompt is explicitly trying to leave a sticky basin.\n"
        "- On break turns, prefer scene-escape controls: exploration_noise, washout_strength, field_alignment_gain, prediction_gain, and transform_gain.\n"
        "- Avoid negative novelty_gain unless genericity or truncation clearly dominates the telemetry.\n"
        "- Keep one ember if needed, but prioritize leaving the old scene over calming the prose.\n\n"
        if break_turn
        else ""
    )
    prompt_preview = str(result.get("text", "")).strip()
    if len(prompt_preview) > 220:
        prompt_preview = prompt_preview[:217] + "..."
    notes = [
        f"User prompt: {prompt}",
        f"Controller regime: {regime} ({regime_source})",
        _format_field_intent(field_intent) or "Requested field pull: none.",
        _format_condition_vector(condition) or "Condition monitor: calm.",
        (_format_condition_vector(relative_condition) or "Condition monitor: calm.").replace(
            "Condition monitor:",
            "Relative baseline:",
        ),
        _format_reservoir_geometry_detail(geometry) or "Geometry: unavailable.",
        str((forecast or {}).get("summary") or "Forecast: unavailable."),
        str((observer_report or {}).get("summary") or "Observer: unavailable."),
        str((change_report or {}).get("summary") or "Change: unavailable."),
        str((intervention_report or {}).get("summary") or "Intervention: none."),
        control_surface_note or "Control surface: baseline.",
        f"Selected answer preview: {prompt_preview or '[empty]'}",
    ]
    base = (
        "Propose tiny bounded next-turn control deltas for the same system.\n"
        "Output exactly two lines and nothing else.\n"
        "Line 1: ADJUST <key=+0.00 ...> or ADJUST none\n"
        "Line 2: REASON <short phrase>\n"
        f"Rules:\n"
        f"- Use at most {_MODEL_ADVICE_MAX_KEYS} keys.\n"
        f"- Only use these keys: {allowed}.\n"
        f"- Each delta must stay within +/-{_MODEL_ADVICE_ABS_LIMIT:.2f}.\n"
        "- Prefer tiny deltas around 0.01 to 0.04.\n"
        "- If the controller already looks stable, say ADJUST none.\n"
        "- If the system is sticky, prefer exploration_noise, washout_strength, transform_gain, or motif_gain changes.\n"
        "- If the system misses the requested field, prefer field_alignment_gain or prediction_gain changes.\n"
        "- If the system is too rigid, reduce strict_bias or slightly raise temperature_bias.\n"
        "- If the system is too loose or truncated, raise strict_bias or reduce temperature_bias.\n"
        "- Scene escape is not the same as lowering novelty.\n"
        "- Never repeat placeholders like <key=+0.00 ...> or <short phrase>.\n"
        "- Never write the words 'Line 1', 'Line 2', 'Output', or markdown bullets in the answer.\n\n"
        "Good examples:\n"
        "ADJUST exploration_noise=+0.03 transform_gain=+0.02 motif_gain=-0.02\n"
        "REASON loosen stale attractor\n\n"
        "ADJUST field_alignment_gain=+0.04 prediction_gain=+0.03\n"
        "REASON steer requested field\n\n"
        "ADJUST none\n"
        "REASON steady\n\n"
        "Bad example (forbidden):\n"
        "ADJUST <key=+0.00 ...>\n"
        "REASON <short phrase>\n\n"
    )
    return base + regime_guidance + break_guidance + "Telemetry:\n" + "\n".join(
        f"- {note}" for note in notes if note
    )


def _parse_self_regulation_advice(text: str) -> dict[str, object] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    adjust_line = next(
        (
            match.group(0).strip()
            for line in lines
            for match in [re.search(r"\bADJUST\b.*", line, flags=re.IGNORECASE)]
            if match
        ),
        None,
    )
    if not adjust_line:
        return None
    reason_line = next(
        (
            match.group(0).strip()
            for line in lines
            for match in [re.search(r"\bREASON\b.*", line, flags=re.IGNORECASE)]
            if match
        ),
        None,
    )
    payload = adjust_line[len("ADJUST") :].strip()
    if "<" in payload or ">" in payload:
        return None
    if payload.lower() == "none":
        reason = reason_line[len("REASON") :].strip() if reason_line else "steady"
        reason = re.sub(r"\s+", " ", reason).strip(" .")
        return {
            "raw": raw,
            "deltas": {},
            "reason": reason or "steady",
            "valid": True,
        }
    pairs = re.findall(r"([a-z_]+)\s*=\s*([+-]?\d+(?:\.\d+)?)", payload)
    deltas: dict[str, float] = {}
    for key, value in pairs:
        if key not in _MODEL_ADVICE_CONTROL_KEYS:
            continue
        if len(deltas) >= _MODEL_ADVICE_MAX_KEYS:
            break
        numeric = max(
            min(float(value), _MODEL_ADVICE_ABS_LIMIT),
            -_MODEL_ADVICE_ABS_LIMIT,
        )
        if abs(numeric) < 1e-6:
            continue
        deltas[key] = numeric
    if not deltas:
        return None
    reason = reason_line[len("REASON") :].strip() if reason_line else "self-advice"
    reason = re.sub(r"\s+", " ", reason).strip(" .")
    return {
        "raw": raw,
        "deltas": deltas,
        "reason": reason or "self-advice",
        "valid": True,
    }


def _collect_self_regulation_advice(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    controller: dict[str, object] | None,
    state: dict[str, object],
    prompt: str,
    result: dict[str, object],
    prepared: dict[str, object] | None,
    field_intent: dict[str, object] | None,
    actual_field: dict[str, object] | None,
    max_tokens: int,
) -> dict[str, object] | None:
    tuning = dict((state or {}).get("self_tuning", {}) or {})
    if (
        args.mode != "reflective"
        or not controller
        or not bool(tuning.get("enabled"))
    ):
        return None
    condition = _compute_condition_vector(
        result=result,
        state=state,
        field_intent=field_intent,
        actual_field=actual_field,
        max_tokens=max_tokens,
    )
    relative_condition = _relative_condition_vector(
        condition,
        dict(tuning.get("condition_baseline", {}) or {}),
        observations=int(tuning.get("baseline_observations", 0) or 0),
    )
    forecast = _build_prediction_forecast(
        prompt=prompt,
        prepared=prepared,
        state=state,
        field_intent=field_intent,
        controller=controller,
    )
    observer_report = _build_observer_report(
        state=state,
        result=result,
        forecast=forecast,
    )
    current_report = {
        "observer": observer_report,
        "geometry": state.get("reservoir_geometry"),
    }
    prior_reports = list(state.get("turn_reports", []))
    change_report = _build_change_report(
        prior_reports[-1] if prior_reports else None,
        current_report,
    )
    advisor_prompt = _build_self_regulation_prompt(
        prompt=prompt,
        result=result,
        condition=condition,
        relative_condition=relative_condition,
        forecast=forecast,
        observer_report=observer_report,
        change_report=change_report,
        field_intent=field_intent,
        geometry=dict(state.get("reservoir_geometry", {}) or {}),
        intervention_report=dict(state.get("last_intervention_report", {}) or {}),
        control_surface_note=_format_control_surface(state, controller),
        controller_regime=str(state.get("controller_regime") or "sustain"),
        controller_regime_source=str(state.get("controller_regime_source") or "auto"),
    )
    prompt_text = _build_raw_prompt(
        [{"role": "user", "content": advisor_prompt}],
        system_prompt=_SELF_REGULATION_ADVISOR_SYSTEM_PROMPT,
    )
    raw_advice = _generate_once(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_tokens=40,
        temp=0.0,
    )
    parsed = _parse_self_regulation_advice(str(raw_advice.get("text", "")))
    if not parsed:
        return None
    return {
        **parsed,
        "break_turn": _is_attractor_break_prompt(prompt, field_intent),
        "regime": str(state.get("controller_regime") or "sustain"),
        "condition": condition,
        "relative_condition": relative_condition,
        "forecast": forecast,
        "observer_report": observer_report,
        "change_report": change_report,
        "prompt": advisor_prompt,
    }


def _resolve_model_advice_adjustment(
    *,
    heuristic_deltas: dict[str, float],
    model_advice: dict[str, object] | None,
    severity: float,
    cooldown_remaining: int,
    regime: str = "sustain",
) -> tuple[dict[str, float], str | None]:
    advice = dict(model_advice or {})
    raw_deltas = dict(advice.get("deltas", {}) or {})
    if not raw_deltas:
        return {}, None
    if severity < 0.08 and cooldown_remaining <= 0:
        return {}, None
    break_turn = bool(advice.get("break_turn"))
    condition = dict(advice.get("condition", {}) or {})
    escape_pressure = max(
        float(condition.get("field_miss", 0.0) or 0.0),
        float(condition.get("attractor_lock", 0.0) or 0.0),
        float(condition.get("geometry_collapse", 0.0) or 0.0),
    )
    escape_regime = break_turn or regime == "escape"
    if escape_regime:
        has_escape_key = any(key in raw_deltas for key in _ESCAPE_CONTROL_KEYS)
        if "novelty_gain" in raw_deltas and float(raw_deltas["novelty_gain"]) < 0.0:
            damp_factor = 0.35 if escape_pressure >= 0.28 else 0.6
            raw_deltas["novelty_gain"] = float(raw_deltas["novelty_gain"]) * damp_factor
            if abs(float(raw_deltas["novelty_gain"])) < 0.01:
                raw_deltas.pop("novelty_gain", None)
        if not has_escape_key and escape_pressure >= 0.28:
            if float(condition.get("geometry_collapse", 0.0) or 0.0) >= 0.30:
                raw_deltas["exploration_noise"] = max(
                    float(raw_deltas.get("exploration_noise", 0.0) or 0.0),
                    0.02,
                )
            if float(condition.get("field_miss", 0.0) or 0.0) >= 0.20:
                raw_deltas["field_alignment_gain"] = max(
                    float(raw_deltas.get("field_alignment_gain", 0.0) or 0.0),
                    0.02,
                )
            raw_deltas["transform_gain"] = max(
                float(raw_deltas.get("transform_gain", 0.0) or 0.0),
                0.015,
            )
    limit = 0.03 if cooldown_remaining > 0 else 0.05
    scale = 0.28 if cooldown_remaining > 0 else (0.55 if severity >= 0.45 else 0.42)
    applied: dict[str, float] = {}
    for key, value in raw_deltas.items():
        if key not in _MODEL_ADVICE_CONTROL_KEYS:
            continue
        candidate = max(min(float(value), limit), -limit)
        heuristic = float(heuristic_deltas.get(key, 0.0) or 0.0)
        if heuristic and candidate * heuristic < 0.0:
            candidate *= 0.35
        elif heuristic and candidate * heuristic > 0.0:
            candidate *= 1.10
        if escape_regime and key in _ESCAPE_CONTROL_KEYS and candidate > 0.0:
            candidate *= 1.15
        adjusted = candidate * scale
        if abs(adjusted) >= 0.004:
            applied[key] = adjusted
    if not applied:
        return {}, None
    reason = str(advice.get("reason") or "self-advice").strip() or "self-advice"
    return applied, reason


def _summarize_predicted_behaviors(
    predicted_behaviors: dict[str, object] | None,
) -> str | None:
    behaviors = dict(predicted_behaviors or {})
    if not behaviors:
        return None
    ranked = sorted(
        (
            (label, float(score))
            for label, score in behaviors.items()
            if isinstance(score, (int, float))
        ),
        key=lambda item: (-item[1], item[0]),
    )
    ranked = [item for item in ranked if item[1] > 0.02]
    if not ranked:
        return None
    return ", ".join(f"{label}={score:.2f}" for label, score in ranked[:3])


def _build_prediction_forecast(
    *,
    prompt: str,
    prepared: dict[str, object] | None,
    state: dict[str, object],
    field_intent: dict[str, object] | None,
    controller: dict[str, object] | None,
) -> dict[str, object] | None:
    preview = dict((prepared or {}).get("preview", {}) or {})
    if not preview:
        return None
    predicted_field = dict(preview.get("predicted_field", {}) or {})
    control_surface = _current_control_surface(state, controller)
    field_note = _format_predicted_field(predicted_field)
    behavior_note = _summarize_predicted_behaviors(preview.get("predicted_behaviors"))
    intent_note = _format_field_intent(field_intent)
    control_note = _format_control_surface(
        {"self_tuning": {"enabled": True, "baseline": _baseline_control_surface(controller), "effective": control_surface}},
        controller,
    )
    summary_parts = []
    if field_note:
        summary_parts.append(field_note)
    if behavior_note:
        summary_parts.append("Expected behavior pulls: " + behavior_note + ".")
    controller_regime = str(state.get("controller_regime") or "sustain")
    controller_source = str(state.get("controller_regime_source") or "auto")
    if controller_source == "auto":
        summary_parts.append(
            f"Next controller regime likely stays {controller_regime}."
        )
    else:
        summary_parts.append(
            f"Controller regime is manually locked to {controller_regime}."
        )
    if intent_note:
        summary_parts.append(intent_note)
    if control_note and control_note != "Control surface: baseline.":
        summary_parts.append(control_note)
    return {
        "prompt": prompt,
        "predicted_field": predicted_field,
        "predicted_behaviors": dict(preview.get("predicted_behaviors", {}) or {}),
        "predicted_controller_regime": controller_regime,
        "predicted_controller_regime_source": controller_source,
        "control_surface": control_surface,
        "summary": " ".join(summary_parts) if summary_parts else None,
    }


def _build_observer_report(
    *,
    state: dict[str, object],
    result: dict[str, object],
    forecast: dict[str, object] | None,
) -> dict[str, object]:
    geometry = dict(state.get("reservoir_geometry", {}) or {})
    tuning = dict(state.get("self_tuning", {}) or {})
    relative = dict(tuning.get("last_relative_condition", {}) or {})
    actual_field = dict(state.get("embedding_field", {}) or {})
    actual_top = (
        list(actual_field.get("top_anchors", []))[0]["label"]
        if list(actual_field.get("top_anchors", []))
        else None
    )
    predicted_field = dict((forecast or {}).get("predicted_field", {}) or {})
    predicted_top = (
        list(predicted_field.get("top_anchors", []))[0]["label"]
        if list(predicted_field.get("top_anchors", []))
        else None
    )
    pressure_items = sorted(
        (
            (key, float(value))
            for key, value in relative.items()
            if key != "severity" and isinstance(value, (int, float))
        ),
        key=lambda item: (-item[1], item[0]),
    )
    dominant_pressure = pressure_items[0][0] if pressure_items and pressure_items[0][1] >= 0.18 else None
    collapse = float(geometry.get("geometry_collapse", 0.0) or 0.0)
    stability = float(tuning.get("stability_score", 1.0) or 1.0)
    geometry_regime = _geometry_regime_label(
        geometry,
        stability_score=stability,
    )
    controller_regime = str(state.get("controller_regime") or "sustain")
    controller_source = str(state.get("controller_regime_source") or "auto")
    controller_reason = state.get("controller_regime_reason") or tuning.get("last_reason")
    summary = (
        f"Observer: geometry={geometry_regime}; controller={controller_regime}; predicted={predicted_top or 'n/a'}; "
        f"actual={actual_top or 'n/a'}; dominant pressure={dominant_pressure or 'calm'}; "
        f"tuning={tuning.get('last_reason') or 'steady'}."
    )
    return {
        "regime": geometry_regime,
        "geometry_regime": geometry_regime,
        "controller_regime": controller_regime,
        "controller_regime_source": controller_source,
        "controller_regime_reason": controller_reason,
        "controller_regime_transition": state.get("controller_regime_transition"),
        "predicted_top_anchor": predicted_top,
        "actual_top_anchor": actual_top,
        "dominant_pressure": dominant_pressure,
        "geometry_collapse": collapse,
        "stability_score": stability,
        "controller_reason": tuning.get("last_reason"),
        "rewrite_issue_count": len(list(result.get("rewrite_issues", []))),
        "summary": summary,
    }


def _build_change_report(
    previous: dict[str, object] | None,
    current: dict[str, object],
) -> dict[str, object] | None:
    if not previous:
        return None
    previous_observer = dict(previous.get("observer", {}) or {})
    current_observer = dict(current.get("observer", {}) or {})
    previous_geometry = dict(previous.get("geometry", {}) or {})
    current_geometry = dict(current.get("geometry", {}) or {})
    previous_collapse = float(previous_geometry.get("geometry_collapse", 0.0) or 0.0)
    current_collapse = float(current_geometry.get("geometry_collapse", 0.0) or 0.0)
    delta = current_collapse - previous_collapse
    previous_top = previous_observer.get("actual_top_anchor")
    current_top = current_observer.get("actual_top_anchor")
    shifted_field = previous_top and current_top and previous_top != current_top
    if abs(delta) < 0.03:
        movement = "steady"
    elif delta > 0.0:
        movement = "more-collapsed"
    else:
        movement = "loosening"
    pieces = [f"Change: geometry is {movement} ({delta:+.2f})."]
    if shifted_field:
        pieces.append(f"Field moved from {previous_top} to {current_top}.")
    previous_reason = previous_observer.get("controller_reason")
    current_reason = current_observer.get("controller_reason")
    if current_reason and current_reason != previous_reason:
        pieces.append(f"Controller shifted from {previous_reason or 'steady'} to {current_reason}.")
    previous_controller_regime = previous_observer.get("controller_regime")
    current_controller_regime = current_observer.get("controller_regime")
    if (
        previous_controller_regime
        and current_controller_regime
        and previous_controller_regime != current_controller_regime
    ):
        pieces.append(
            f"Controller regime shifted from {previous_controller_regime} to {current_controller_regime}."
        )
    return {
        "movement": movement,
        "collapse_delta": delta,
        "field_shift": [previous_top, current_top] if shifted_field else None,
        "controller_regime_shift": (
            [previous_controller_regime, current_controller_regime]
            if previous_controller_regime
            and current_controller_regime
            and previous_controller_regime != current_controller_regime
            else None
        ),
        "summary": " ".join(pieces),
    }


def _attach_turn_diagnostics(
    *,
    state: dict[str, object],
    prompt: str,
    result: dict[str, object],
    prepared: dict[str, object] | None,
    field_intent: dict[str, object] | None,
    controller: dict[str, object] | None,
) -> dict[str, object]:
    updated = dict(state)
    forecast = _build_prediction_forecast(
        prompt=prompt,
        prepared=prepared,
        state=state,
        field_intent=field_intent,
        controller=controller,
    )
    observer = _build_observer_report(
        state=state,
        result=result,
        forecast=forecast,
    )
    previous_reports = list(state.get("turn_reports", []))
    current_report = {
        "turn": int(state.get("turn_count", 0)),
        "prompt": prompt,
        "field_intent": field_intent,
        "forecast": forecast,
        "observer": observer,
        "geometry": state.get("reservoir_geometry"),
        "result_preview": str(result.get("text", ""))[:180],
    }
    change = _build_change_report(
        previous_reports[-1] if previous_reports else None,
        current_report,
    )
    observer_history = list(state.get("observer_history", []))
    observer_history.append(observer)
    observer_history = observer_history[-8:]
    previous_reports.append(current_report)
    previous_reports = previous_reports[-8:]
    updated["last_forecast"] = forecast
    updated["last_observer_report"] = observer
    updated["observer_history"] = observer_history
    updated["last_change_report"] = change
    updated["turn_reports"] = previous_reports
    return updated


def _compute_condition_vector(
    *,
    result: dict[str, object],
    state: dict[str, object],
    field_intent: dict[str, object] | None,
    actual_field: dict[str, object] | None,
    max_tokens: int,
) -> dict[str, float]:
    issues = [str(item) for item in list(result.get("rewrite_issues", []))]
    issue_text = " ".join(issues).lower()
    overlap = float(result.get("candidate_follow_up_overlap", 0.0) or 0.0)
    distance_balance = float(
        result.get("candidate_follow_up_distance_balance", 0.0) or 0.0
    )
    generated_tokens = int(result.get("generated_tokens", 0) or 0)
    candidate_count = int(result.get("candidate_count", 1) or 1)
    candidate_summaries = list(result.get("candidate_summaries", []))
    top_anchor_history = [
        item.get("top_anchor")
        for item in list(state.get("embedding_field_history", []))[-3:]
        if item.get("top_anchor")
    ]
    geometry = dict(state.get("reservoir_geometry", {}) or {})
    drift_values = [
        float(item.get("drift"))
        for item in list(state.get("embedding_field_history", []))[-3:]
        if isinstance(item.get("drift"), (int, float))
    ]
    field_score, field_details = _score_field_alignment(actual_field, field_intent)
    prediction_match = dict(
        dict(state.get("reservoir_latent", {}) or {}).get("prediction_match", {}) or {}
    )
    candidate_scores = [
        float(item.get("score", 0.0) or 0.0)
        for item in candidate_summaries
        if isinstance(item.get("score"), (int, float))
    ]
    disagreement = 0.0
    if len(candidate_scores) >= 2:
        disagreement = min(max(candidate_scores) - min(candidate_scores), 2.5) / 2.5

    repetition_pressure = max(
        min(overlap, 1.0),
        0.85 if "repeats a sentence or idea" in issue_text else 0.0,
    )
    field_miss = 0.0
    misses = list(field_details.get("misses", [])) if field_details else []
    if field_intent:
        if misses:
            field_miss = max(field_miss, min(len(misses) / 2.0, 1.0))
        if field_score < 0.0:
            field_miss = max(field_miss, min(abs(field_score) / 1.5, 1.0))
        if "does not move the field" in issue_text:
            field_miss = max(field_miss, 0.8)
    prediction_mismatch = 0.0
    if prediction_match:
        prediction_mismatch = max(0.0, 1.0 - float(prediction_match.get("score", 0.0) or 0.0))
    structure_strain = min(len(issues) / 4.0, 1.0)
    genericity_pressure = 0.0
    if "too generic" in issue_text or "stock disclaimer" in issue_text:
        genericity_pressure = 0.8
    elif "fresh bridge" in issue_text:
        genericity_pressure = 0.45
    continuity_deficit = 0.0
    if "drops the carried motif" in issue_text or "does not add a fresh turn" in issue_text:
        continuity_deficit = 0.8
    elif distance_balance < 0.15 and overlap <= 0.08:
        continuity_deficit = 0.45
    attractor_lock = 0.0
    if len(top_anchor_history) >= 3 and len(set(top_anchor_history)) == 1:
        attractor_lock = 0.55
        if drift_values and sum(drift_values) / len(drift_values) < 0.10:
            attractor_lock = 0.8
    geometry_collapse = float(geometry.get("geometry_collapse", 0.0) or 0.0)
    if float(geometry.get("normalized_rank", 1.0) or 1.0) <= 0.40:
        geometry_collapse = max(geometry_collapse, 0.55)
    if float(geometry.get("mean_drift", 0.0) or 0.0) <= 0.08 and float(
        geometry.get("attractor_persistence", 0.0) or 0.0
    ) >= 0.50:
        geometry_collapse = max(geometry_collapse, 0.75)
    truncation_pressure = 0.0
    if generated_tokens >= max(max_tokens - 2, 1):
        truncation_pressure = max(truncation_pressure, 0.6)
    if "does not finish cleanly" in issue_text:
        truncation_pressure = max(truncation_pressure, 0.85)
    if candidate_count > 1:
        truncation_pressure = max(truncation_pressure, disagreement * 0.35)

    severity_components = [
        repetition_pressure,
        field_miss,
        prediction_mismatch,
        structure_strain,
        genericity_pressure,
        continuity_deficit,
        attractor_lock,
        geometry_collapse,
        truncation_pressure,
    ]
    severity = sum(severity_components) / len(severity_components)
    return {
        "repetition_pressure": min(repetition_pressure, 1.0),
        "field_miss": min(field_miss, 1.0),
        "prediction_mismatch": min(prediction_mismatch, 1.0),
        "structure_strain": min(structure_strain, 1.0),
        "genericity_pressure": min(genericity_pressure, 1.0),
        "continuity_deficit": min(continuity_deficit, 1.0),
        "attractor_lock": min(attractor_lock, 1.0),
        "geometry_collapse": min(geometry_collapse, 1.0),
        "truncation_pressure": min(truncation_pressure, 1.0),
        "severity": min(severity, 1.0),
    }


def _apply_control_deltas(
    effective: dict[str, float],
    deltas: dict[str, float],
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    updated = dict(effective)
    for key, delta in deltas.items():
        updated[key] = float(updated.get(key, 0.0)) + float(delta) * scale
    return _clamp_control_surface(updated)


def _propose_self_tuning_policy(
    *,
    state: dict[str, object],
    controller: dict[str, object] | None,
    condition: dict[str, float],
    regime: str = "sustain",
    model_advice: dict[str, object] | None = None,
) -> dict[str, object]:
    tuning = dict(state.get("self_tuning", {}) or {})
    baseline = dict(tuning.get("baseline", {}) or _baseline_control_surface(controller))
    effective = dict(tuning.get("effective", {}) or baseline)
    cooldown_remaining = int(tuning.get("cooldown_remaining", 0) or 0)
    history = list(tuning.get("history", []))
    condition_baseline = dict(tuning.get("condition_baseline", {}) or {})
    observations = int(tuning.get("baseline_observations", 0) or 0)
    relative_condition = _relative_condition_vector(
        condition,
        condition_baseline,
        observations=observations,
    )
    active_condition = (
        relative_condition
        if observations >= _BASELINE_BOOTSTRAP_OBSERVATIONS
        else condition
    )
    pressure_counts = _update_pressure_counts(
        dict(tuning.get("pressure_counts", {}) or {}),
        active_condition,
    )
    severity = float(active_condition.get("severity", 0.0) or 0.0)
    calm_streak = int(tuning.get("calm_streak", 0) or 0)
    if severity <= 0.18:
        calm_streak += 1
    else:
        calm_streak = 0

    effective = _decay_control_surface(
        effective,
        baseline,
        rate=0.18 if calm_streak >= 2 else 0.10,
    )
    effective = _apply_regime_policy(
        effective,
        baseline,
        regime=regime,
    )
    deltas: dict[str, float] = {}
    reasons: list[str] = []

    def bump(key: str, value: float) -> None:
        deltas[key] = float(deltas.get(key, 0.0)) + value

    next_condition_baseline = _update_condition_baseline(
        condition_baseline,
        condition,
        observations=observations,
    )
    model_adjustment, model_reason = _resolve_model_advice_adjustment(
        heuristic_deltas={},
        model_advice=model_advice,
        severity=severity,
        cooldown_remaining=cooldown_remaining,
        regime=regime,
    )

    if cooldown_remaining > 0 and severity < 0.86:
        if model_adjustment:
            effective = _apply_control_deltas(effective, model_adjustment)
        updated = dict(tuning)
        updated["effective"] = effective
        updated["cooldown_remaining"] = max(
            cooldown_remaining - (2 if calm_streak >= 2 else 1),
            0,
        )
        updated["last_condition"] = condition
        updated["last_relative_condition"] = relative_condition
        updated["last_adjustment"] = dict(model_adjustment)
        updated["last_reason"] = (
            f"regime={regime}; cooldown; self-advice={model_reason}"
            if model_adjustment and model_reason
            else f"regime={regime}; cooldown"
        )
        updated["last_model_advice"] = dict(model_advice or {})
        updated["last_model_adjustment"] = dict(model_adjustment)
        updated["condition_baseline"] = next_condition_baseline
        updated["baseline_observations"] = observations + 1
        updated["pressure_counts"] = pressure_counts
        updated["calm_streak"] = calm_streak
        updated["stability_score"] = max(0.0, 1.0 - severity)
        advice_history = list(tuning.get("advice_history", []))
        if model_adjustment:
            advice_history.append(
                {
                    "turn": int(state.get("turn_count", 0)) + 1,
                    "reason": model_reason,
                    "deltas": dict(model_adjustment),
                }
            )
            advice_history = advice_history[-8:]
        updated["advice_history"] = advice_history
        return updated

    if _condition_gate(active_condition, pressure_counts, "repetition_pressure") or _condition_gate(
        active_condition,
        pressure_counts,
        "attractor_lock",
        threshold=0.40,
        streak=2,
        spike=0.72,
    ):
        if regime == "escape":
            bump("exploration_noise", 0.08)
            bump("washout_strength", 0.08)
            bump("transform_gain", 0.10)
            bump("field_alignment_gain", 0.06)
        else:
            bump("novelty_gain", 0.14)
            bump("transform_gain", 0.10)
            bump("motif_gain", -0.10)
        bump("temperature_bias", 0.02)
        bump("leak_fast", 0.03)
        bump("washout_strength", 0.08)
        bump("exploration_noise", 0.05)
        reasons.append(_TUNING_REASON_LABELS["repetition_pressure"])
    if _condition_gate(active_condition, pressure_counts, "field_miss") or _condition_gate(
        active_condition,
        pressure_counts,
        "prediction_mismatch",
    ):
        bump("field_alignment_gain", 0.18)
        bump("prediction_gain", 0.12)
        bump("strict_bias", 0.10)
        bump("temperature_bias", -0.02)
        reasons.append(_TUNING_REASON_LABELS["field_miss"])
    if _condition_gate(active_condition, pressure_counts, "structure_strain") or _condition_gate(
        active_condition,
        pressure_counts,
        "truncation_pressure",
        threshold=0.50,
        streak=2,
        spike=0.75,
    ):
        bump("strict_bias", 0.14)
        bump("temperature_bias", -0.03)
        bump("leak_fast", -0.01)
        bump("exploration_noise", -0.02)
        reasons.append(_TUNING_REASON_LABELS["structure_strain"])
    if _condition_gate(active_condition, pressure_counts, "continuity_deficit"):
        bump("motif_gain", 0.14)
        bump("transform_gain", 0.08)
        bump("leak_medium", -0.03)
        bump("leak_slow", -0.015)
        reasons.append(_TUNING_REASON_LABELS["continuity_deficit"])
    if _condition_gate(active_condition, pressure_counts, "genericity_pressure"):
        bump("novelty_gain", 0.10)
        bump("warmth_gain", 0.08)
        bump("exploration_noise", 0.02)
        reasons.append(_TUNING_REASON_LABELS["genericity_pressure"])
    if _condition_gate(
        active_condition,
        pressure_counts,
        "geometry_collapse",
        threshold=0.42,
        streak=2,
        spike=0.68,
    ):
        bump("exploration_noise", 0.08)
        bump("temperature_bias", 0.03)
        bump("novelty_gain", 0.08)
        bump("motif_gain", -0.06)
        bump("strict_bias", -0.06)
        bump("washout_strength", 0.06)
        reasons.append(_TUNING_REASON_LABELS["geometry_collapse"])
    if _condition_gate(
        active_condition,
        pressure_counts,
        "attractor_lock",
        threshold=0.40,
        streak=2,
        spike=0.72,
    ):
        bump("exploration_noise", 0.06)
        bump("washout_strength", 0.05)
        reasons.append(_TUNING_REASON_LABELS["attractor_lock"])

    model_adjustment, model_reason = _resolve_model_advice_adjustment(
        heuristic_deltas=deltas,
        model_advice=model_advice,
        severity=severity,
        cooldown_remaining=cooldown_remaining,
        regime=regime,
    )
    total_adjustment = dict(deltas)
    for key, value in model_adjustment.items():
        total_adjustment[key] = float(total_adjustment.get(key, 0.0)) + float(value)

    if not total_adjustment:
        updated = dict(tuning)
        updated["effective"] = effective
        updated["cooldown_remaining"] = max(cooldown_remaining - 1, 0)
        updated["last_condition"] = condition
        updated["last_relative_condition"] = relative_condition
        updated["last_adjustment"] = {}
        updated["last_reason"] = "steady"
        if regime:
            updated["last_reason"] = f"regime={regime}; steady"
        updated["last_model_advice"] = dict(model_advice or {})
        updated["last_model_adjustment"] = {}
        updated["condition_baseline"] = next_condition_baseline
        updated["baseline_observations"] = observations + 1
        updated["pressure_counts"] = pressure_counts
        updated["calm_streak"] = calm_streak
        updated["stability_score"] = max(0.0, 1.0 - severity)
        return updated

    effective = _apply_control_deltas(effective, total_adjustment)
    reason_text = ", ".join(dict.fromkeys(reasons))
    if reason_text:
        reason_text = f"regime={regime}; " + reason_text
    else:
        reason_text = f"regime={regime}"
    if model_adjustment and model_reason:
        reason_text = (
            f"{reason_text}; self-advice={model_reason}"
            if reason_text
            else f"self-advice={model_reason}"
        )
    history.append(
        {
            "turn": int(state.get("turn_count", 0)) + 1,
            "severity": severity,
            "reason": reason_text,
            "deltas": dict(total_adjustment),
            "model_adjustment": dict(model_adjustment),
        }
    )
    history = history[-8:]
    updated = dict(tuning)
    updated["effective"] = effective
    updated["cooldown_remaining"] = _DEFAULT_TUNING_COOLDOWN_TURNS + (
        1 if severity >= 0.65 or len(total_adjustment) >= 4 else 0
    )
    updated["last_condition"] = condition
    updated["last_relative_condition"] = relative_condition
    updated["last_adjustment"] = dict(total_adjustment)
    updated["last_reason"] = reason_text
    updated["last_model_advice"] = dict(model_advice or {})
    updated["last_model_adjustment"] = dict(model_adjustment)
    updated["stability_score"] = max(0.0, 1.0 - severity)
    updated["condition_baseline"] = next_condition_baseline
    updated["baseline_observations"] = observations + 1
    updated["pressure_counts"] = pressure_counts
    updated["calm_streak"] = calm_streak
    advice_history = list(tuning.get("advice_history", []))
    if model_adjustment:
        advice_history.append(
            {
                "turn": int(state.get("turn_count", 0)) + 1,
                "reason": model_reason,
                "deltas": dict(model_adjustment),
            }
        )
        advice_history = advice_history[-8:]
    updated["advice_history"] = advice_history
    updated["history"] = history
    return updated


def _update_self_tuning_state(
    *,
    state: dict[str, object],
    controller: dict[str, object] | None,
    result: dict[str, object],
    field_intent: dict[str, object] | None,
    actual_field: dict[str, object] | None,
    max_tokens: int,
    regime: str = "sustain",
    model_advice: dict[str, object] | None = None,
) -> dict[str, object]:
    tuning = dict(state.get("self_tuning", {}) or {})
    if not bool(tuning.get("enabled")):
        return tuning
    condition = _compute_condition_vector(
        result=result,
        state=state,
        field_intent=field_intent,
        actual_field=actual_field,
        max_tokens=max_tokens,
    )
    return _propose_self_tuning_policy(
        state=state,
        controller=controller,
        condition=condition,
        regime=regime,
        model_advice=model_advice,
    )


def _apply_reservoir_washout(
    state: dict[str, object],
    *,
    strength: float | None = None,
) -> dict[str, object]:
    updated = dict(state)
    latent = dict(updated.get("reservoir_latent", {}) or {})
    if not latent:
        return updated
    effective_strength = (
        float(strength)
        if isinstance(strength, (int, float))
        else float(_current_control_surface(state, None).get("washout_strength", 0.75))
    )
    factor = max(0.0, min(1.0, 1.0 - effective_strength))
    for key in ("fast", "medium", "slow", "combined"):
        latent[key] = [value * factor for value in list(latent.get(key, []))]
    latent["state_norm"] = _vector_norm(list(latent.get("combined", [])))
    latent["state_drift"] = None
    latent["washout_count"] = int(latent.get("washout_count", 0)) + 1
    updated["reservoir_latent"] = latent
    return updated


def _sum_readout_vectors(
    controller: dict[str, object] | None,
    labels: list[str] | tuple[str, ...] | set[str],
) -> list[float]:
    if not controller:
        return []
    readout = dict(controller.get("field_readout", {}) or {})
    vectors = [list(readout[label]) for label in labels if label in readout]
    if not vectors:
        return []
    return _normalize_vector(_vector_add(*vectors))


def _attenuate_weighted_terms(
    weighted_terms: dict[str, float],
    terms: list[str] | tuple[str, ...] | set[str],
    *,
    factor: float,
) -> dict[str, float]:
    updated = dict(weighted_terms)
    for term in terms:
        if term in updated:
            updated[term] = float(updated[term]) * factor
            if updated[term] <= 0.02:
                updated.pop(term, None)
    return updated


def _apply_break_turn_intervention(
    state: dict[str, object],
    *,
    controller: dict[str, object] | None,
    prompt: str,
    field_intent: dict[str, object] | None,
) -> dict[str, object]:
    if not controller or not _is_attractor_break_prompt(prompt, field_intent):
        return state
    latent = dict((state or {}).get("reservoir_latent", {}) or {})
    if not latent or not list(latent.get("combined", [])):
        return state
    geometry = dict((state or {}).get("reservoir_geometry", {}) or {})
    tuning = dict((state or {}).get("self_tuning", {}) or {})
    relative = dict(tuning.get("last_relative_condition", {}) or {})
    collapse = float(geometry.get("geometry_collapse", 0.0) or 0.0)
    persistence = float(geometry.get("attractor_persistence", 0.0) or 0.0)
    field_miss = float(relative.get("field_miss", 0.0) or 0.0)
    attractor_lock = float(relative.get("attractor_lock", 0.0) or 0.0)
    dominant = (
        dict(latent.get("predicted_field", {}) or {}).get("dominant_attractor")
        or geometry.get("dominant_attractor")
    )
    intent = dict(field_intent or {})
    avoid_labels = list(intent.get("avoid", []) or [])
    should_apply = bool(
        int((state or {}).get("turn_count", 0) or 0) >= 2
        and (
            collapse >= 0.32
            or persistence >= 0.50
            or field_miss >= 0.35
            or attractor_lock >= 0.35
            or (dominant and dominant in avoid_labels)
        )
    )
    if not should_apply:
        return state

    updated = _apply_reservoir_washout(
        state,
        strength=min(0.48, max(0.22, 0.20 + collapse * 0.22 + field_miss * 0.10)),
    )
    latent = dict(updated.get("reservoir_latent", {}) or {})
    previous_combined = list(latent.get("combined", []))
    control = _current_control_surface(state, controller)

    target_labels = set(intent.get("targets", []) or [])
    if intent.get("cool_formal"):
        target_labels.add("prime-math")
    if intent.get("preserve_warmth"):
        target_labels.add("campfire-imagery")
    if intent.get("preserve_motif"):
        target_labels.add("reservoir-memory")
    repel_labels = set(avoid_labels)
    if dominant and (dominant in repel_labels or collapse >= 0.40 or persistence >= 0.60):
        repel_labels.add(str(dominant))

    target_vector = _sum_readout_vectors(controller, sorted(target_labels))
    repel_vector = _sum_readout_vectors(controller, sorted(repel_labels))
    shift_strength = min(
        0.32,
        0.10 + collapse * 0.16 + field_miss * 0.10 + attractor_lock * 0.08,
    )
    repel_strength = shift_strength * (1.10 if dominant in repel_labels else 0.85)
    target_strength = shift_strength * (1.05 if target_vector else 0.0)

    for key, factor in (("fast", 0.95), ("medium", 0.70), ("slow", 0.28)):
        base = list(latent.get(key, []))
        if not base:
            continue
        shifted = list(base)
        if target_vector:
            shifted = _vector_add(shifted, _vector_scale(target_vector, target_strength * factor))
        if repel_vector:
            shifted = _vector_subtract(shifted, _vector_scale(repel_vector, repel_strength * factor))
        latent[key] = _clip_vector(shifted)

    latent["combined"] = _vector_add(
        _vector_scale(list(latent.get("fast", [])), 0.55),
        _vector_scale(list(latent.get("medium", [])), 0.30),
        _vector_scale(list(latent.get("slow", [])), 0.15),
    )
    latent["state_norm"] = _vector_norm(list(latent.get("combined", [])))
    latent["state_drift"] = (
        None
        if not any(abs(value) > 1e-9 for value in previous_combined)
        else max(0.0, 1.0 - _vector_cosine(previous_combined, list(latent.get("combined", []))))
    )
    readout = _predict_reservoir_readout(controller, list(latent.get("combined", [])))
    latent["predicted_field"] = {
        "top_anchors": list(readout["top_anchors"]),
        "anchor_scores": dict(readout["anchor_scores"]),
        "dominant_attractor": readout["dominant_attractor"],
        "entropy_like": readout["entropy_like"],
    }
    latent["predicted_behaviors"] = dict(readout["behavior_scores"])
    latent["exploration_noise"] = max(
        float(latent.get("exploration_noise", 0.0) or 0.0),
        min(float(control.get("exploration_noise", 0.0) or 0.0) + 0.05, 0.18),
    )
    updated["reservoir_latent"] = latent

    forbidden_terms = _extract_forbidden_scene_terms(
        prompt=prompt,
        field_intent=field_intent,
        reservoir_state=state,
    )
    kept_anchor = _pick_salvage_image(
        prompt=prompt,
        text=str(state.get("last_campfire", "") or ""),
        reservoir_state=state,
        field_intent=field_intent,
    )
    attenuation_terms = [term for term in forbidden_terms if term != kept_anchor]
    updated["images"] = _attenuate_weighted_terms(
        dict(updated.get("images", {}) or {}),
        attenuation_terms,
        factor=0.14,
    )
    updated["motifs"] = _attenuate_weighted_terms(
        dict(updated.get("motifs", {}) or {}),
        attenuation_terms,
        factor=0.35,
    )

    intervention = {
        "applied": True,
        "type": "break_turn",
        "dominant_before": dominant,
        "repel_labels": sorted(repel_labels),
        "target_labels": sorted(target_labels),
        "forbidden_terms": forbidden_terms[:6],
        "strength": shift_strength,
        "summary": (
            "Break intervention: partial washout plus attractor repulsion "
            f"from {dominant or 'current groove'} toward "
            f"{', '.join(sorted(target_labels)[:2]) or 'a fresher field'}."
        ),
    }
    history = list(updated.get("intervention_history", []))
    history.append(intervention)
    updated["last_intervention_report"] = intervention
    updated["intervention_history"] = history[-8:]
    return updated


def _reinforce_target_basin(
    state: dict[str, object],
    *,
    controller: dict[str, object] | None,
    target_labels: list[str],
    medium_scale: float,
    slow_scale: float,
) -> dict[str, object]:
    if not controller or not target_labels:
        return state
    updated = dict(state)
    latent = dict(updated.get("reservoir_latent", {}) or {})
    if not latent or not list(latent.get("combined", [])):
        return updated
    target_vector = _sum_readout_vectors(controller, target_labels)
    if not target_vector:
        return updated
    medium = _clip_vector(
        _vector_add(
            list(latent.get("medium", [])),
            _vector_scale(target_vector, medium_scale),
        )
    )
    slow = _clip_vector(
        _vector_add(
            list(latent.get("slow", [])),
            _vector_scale(target_vector, slow_scale),
        )
    )
    latent["medium"] = medium
    latent["slow"] = slow
    latent["combined"] = _vector_add(
        _vector_scale(list(latent.get("fast", [])), 0.55),
        _vector_scale(medium, 0.30),
        _vector_scale(slow, 0.15),
    )
    latent["state_norm"] = _vector_norm(list(latent.get("combined", [])))
    readout = _predict_reservoir_readout(controller, list(latent.get("combined", [])))
    latent["predicted_field"] = {
        "top_anchors": list(readout["top_anchors"]),
        "anchor_scores": dict(readout["anchor_scores"]),
        "dominant_attractor": readout["dominant_attractor"],
        "entropy_like": readout["entropy_like"],
    }
    latent["predicted_behaviors"] = dict(readout["behavior_scores"])
    updated["reservoir_latent"] = latent
    return updated


def _apply_regime_memory_operator(
    state: dict[str, object],
    *,
    controller: dict[str, object] | None,
    prompt: str,
    field_intent: dict[str, object] | None,
    mode: str,
    architecture: str,
) -> dict[str, object]:
    updated = dict(state)
    updated["controller_refractory_terms"] = _decay_refractory_map(
        dict(updated.get("controller_refractory_terms", {}) or {})
    )
    updated["controller_refractory_attractors"] = _decay_refractory_map(
        dict(updated.get("controller_refractory_attractors", {}) or {})
    )
    updated["controller_target_basin_countdown"] = max(
        int(updated.get("controller_target_basin_countdown", 0) or 0) - 1,
        0,
    )
    if not _controller_regime_is_active(
        mode=mode,
        architecture=architecture,
        controller=controller,
    ):
        updated["controller_regime_memory_operator"] = {
            "regime": str(updated.get("controller_regime") or "sustain"),
            "summary": "Regime memory operator inactive outside reflective reservoir mode.",
        }
        return updated

    regime = str(updated.get("controller_regime") or "sustain")
    target_basin = _target_basin_from_field_intent(field_intent)
    if target_basin:
        updated["controller_target_basin"] = target_basin
    summary = "Regime memory operator: sustain."

    if regime == "escape":
        before_dominant = (
            dict(updated.get("reservoir_geometry", {}) or {}).get("dominant_attractor")
            or dict((updated.get("reservoir_latent") or {}).get("predicted_field", {}) or {}).get(
                "dominant_attractor"
            )
        )
        updated = _apply_break_turn_intervention(
            updated,
            controller=controller,
            prompt=prompt,
            field_intent=field_intent,
        )
        forbidden_terms = _extract_forbidden_scene_terms(
            prompt=prompt,
            field_intent=field_intent,
            reservoir_state=state,
        )
        refractory_terms = dict(updated.get("controller_refractory_terms", {}) or {})
        for term in forbidden_terms:
            refractory_terms[term] = max(
                refractory_terms.get(term, 0),
                _REGIME_REFRACTORY_HORIZON,
            )
        updated["controller_refractory_terms"] = refractory_terms
        refractory_attractors = dict(
            updated.get("controller_refractory_attractors", {}) or {}
        )
        if before_dominant:
            refractory_attractors[str(before_dominant)] = max(
                refractory_attractors.get(str(before_dominant), 0),
                _REGIME_REFRACTORY_HORIZON,
            )
            updated["controller_stale_attractor"] = str(before_dominant)
        if target_basin:
            updated["controller_target_basin"] = target_basin
            updated["controller_target_basin_countdown"] = _REGIME_TARGET_BASIN_HORIZON
        updated["controller_refractory_attractors"] = refractory_attractors
        summary = (
            "Regime memory operator: escape; attenuating stale scene terms and repelling the old basin."
        )
    elif regime == "rebind":
        anchor = _pick_salvage_image(
            prompt=prompt,
            text=str(updated.get("last_campfire", "") or ""),
            reservoir_state=updated,
            field_intent=field_intent,
        )
        if target_basin:
            updated = _reinforce_target_basin(
                updated,
                controller=controller,
                target_labels=[target_basin],
                medium_scale=0.08,
                slow_scale=0.04,
            )
            updated["controller_target_basin"] = target_basin
            updated["controller_target_basin_countdown"] = _REGIME_TARGET_BASIN_HORIZON
        updated["motifs"] = _attenuate_weighted_terms(
            dict(updated.get("motifs", {}) or {}),
            dict(updated.get("controller_refractory_terms", {}) or {}),
            factor=0.50,
        )
        if anchor:
            motifs = dict(updated.get("motifs", {}) or {})
            motifs[anchor] = max(float(motifs.get(anchor, 0.0) or 0.0), 0.9)
            updated["motifs"] = motifs
        summary = (
            "Regime memory operator: rebind; keeping one ember while reinforcing the new basin."
        )
    elif regime == "consolidate":
        if target_basin:
            updated = _reinforce_target_basin(
                updated,
                controller=controller,
                target_labels=[target_basin],
                medium_scale=0.04,
                slow_scale=0.08,
            )
        updated["controller_refractory_terms"] = _decay_refractory_map(
            dict(updated.get("controller_refractory_terms", {}) or {})
        )
        updated["controller_refractory_attractors"] = _decay_refractory_map(
            dict(updated.get("controller_refractory_attractors", {}) or {})
        )
        summary = (
            "Regime memory operator: consolidate; decaying refractory blocks while reinforcing the new groove."
        )
    else:
        summary = "Regime memory operator: sustain; ordinary carry and decay."

    updated["controller_regime_memory_operator"] = {
        "regime": regime,
        "summary": summary,
        "target_basin": updated.get("controller_target_basin"),
        "refractory_terms": dict(updated.get("controller_refractory_terms", {}) or {}),
        "refractory_attractors": dict(
            updated.get("controller_refractory_attractors", {}) or {}
        ),
    }
    return updated


def _answer_establishes_reflective_thread(text: str) -> bool:
    return (
        _extract_labeled_paragraph(text, "Campfire") is not None
        and _extract_labeled_paragraph(text, "Operationally") is not None
    )


def _reservoir_terms(state: dict[str, object]) -> list[str]:
    motifs = _top_weighted_terms(dict(state.get("motifs", {})), limit=4)
    images = _top_weighted_terms(dict(state.get("images", {})), limit=2)
    return motifs + [term for term in images if term not in motifs]


def _content_overlap_ratio(left: str, right: str) -> float:
    left_words = set(_tokenize_content_words(left))
    right_words = set(_tokenize_content_words(right))
    if not left_words or not right_words:
        return 0.0
    shared = left_words & right_words
    return float(len(shared)) / float(min(len(left_words), len(right_words)))


def _follow_up_transformation_metrics(
    *,
    prompt: str,
    text: str,
    reservoir_state: dict[str, object] | None,
) -> dict[str, object]:
    state = reservoir_state or {}
    reservoir_terms = _reservoir_terms(state)
    lowered = str(text or "").lower()
    carry_hits = [
        term for term in reservoir_terms if re.search(rf"\b{re.escape(term)}\b", lowered)
    ]
    transformed_words = [
        token
        for token in _fresh_content_words(prompt, text)
        if token not in reservoir_terms
    ]
    previous_campfire = str(state.get("last_campfire", "") or "")
    current_campfire = _extract_labeled_paragraph(text, "Campfire") or ""
    campfire_overlap = _content_overlap_ratio(previous_campfire, current_campfire)
    meta_cues = [
        cue for cue in _FOLLOW_UP_META_CUES if cue in lowered
    ]
    distance_balance = max(
        0.0,
        1.0 - min(abs(campfire_overlap - 0.38) / 0.38, 1.0),
    )
    return {
        "carry_hits": carry_hits,
        "transformed_words": transformed_words,
        "campfire_overlap": campfire_overlap,
        "distance_balance": distance_balance,
        "meta_cues": meta_cues,
    }


def _build_follow_up_transform_fallback(
    *,
    prompt: str,
    reservoir_state: dict[str, object] | None,
    field_intent: dict[str, object] | None = None,
) -> str:
    state = reservoir_state or {}
    intent = dict(field_intent or {})
    previous_campfire = str(state.get("last_campfire", "") or "")
    previous_images = [
        token
        for token in _extract_keywords(previous_campfire, limit=6)
        if token in _CONCRETE_IMAGE_CUES and token != "campfire"
    ]
    image_terms = [
        token
        for token in _top_weighted_terms(dict(state.get("images", {})), limit=3)
        if token != "campfire"
    ]
    motif_terms = _top_weighted_terms(dict(state.get("motifs", {})), limit=5)
    anchor = (
        previous_images[0]
        if previous_images
        else (image_terms[0] if image_terms else (motif_terms[0] if motif_terms else "lantern"))
    )
    previous_words = set(_tokenize_content_words(previous_campfire))
    transform_images = (
        _FORMAL_TRANSFORM_IMAGES
        if _is_attractor_break_prompt(prompt, intent)
        else _FOLLOW_UP_TRANSFORM_IMAGES
    )
    target = next(
        (
            image
            for image in transform_images
            if image not in previous_words and image != anchor
        ),
        "page" if _is_attractor_break_prompt(prompt, intent) else "shoreline",
    )
    prime = (
        _first_prime_reference(previous_campfire)
        or _first_prime_reference(prompt)
        or "17"
    )
    if _is_attractor_break_prompt(prompt, intent):
        campfire = (
            f"Campfire: Tonight my favorite prime is {prime}, {_with_indefinite_article(anchor)} held like a warm "
            f"{target} inside proof and ordered arithmetic."
        )
        anchor_phrase = "the image" if anchor == "ember" else f"the {anchor}"
        operational = (
            f"Operationally: Reservoir dynamics keep one ember of {anchor_phrase} while "
            "recurrent state cools the answer into proof and residue."
        )
    else:
        campfire = (
            f"Campfire: Tonight my favorite prime is {prime}, the {anchor} skimming the "
            f"{target} and leaving a pale wake behind it."
        )
        anchor_trace = "glow" if anchor in _CONCRETE_IMAGE_CUES else "trace"
        operational = (
            f"Operationally: Reservoir dynamics keep the {anchor}'s {anchor_trace} available, "
            f"while recurrent state slides the scene into a new {target} rhythm."
        )
    return f"{campfire}\n\n{operational}"


def _pick_best_sentence(text: str, *, preferred_tokens: tuple[str, ...]) -> str:
    for sentence in _sentences(text):
        lowered = sentence.lower()
        if any(token in lowered for token in preferred_tokens):
            return sentence.strip()
    return _sentences(text)[0].strip() if _sentences(text) else ""


def _squeeze_reflective_answer(text: str) -> str:
    campfire = _extract_labeled_paragraph(text, "Campfire") or (_split_paragraphs(text)[0] if _split_paragraphs(text) else "")
    operational = _extract_labeled_paragraph(text, "Operationally") or (_split_paragraphs(text)[-1] if _split_paragraphs(text) else "")
    campfire_sentences = [
        sentence
        for sentence in _sentences(campfire)
        if not _layer_starts_with_label(sentence, "Operationally")
    ] or _sentences(campfire)
    operational_sentences = [
        sentence
        for sentence in _sentences(operational)
        if not _layer_starts_with_label(sentence, "Campfire")
    ] or _sentences(operational)
    campfire_sentence = _pick_best_sentence(
        " ".join(campfire_sentences),
        preferred_tokens=("favorite", "prime", "beacon", "ember", "lantern"),
    )
    operational_sentence = _pick_best_sentence(
        " ".join(operational_sentences),
        preferred_tokens=("reservoir", "recurrent", "field", "proof", "geometry"),
    )
    if campfire_sentence and operational_sentence and _content_overlap_ratio(campfire_sentence, operational_sentence) >= 0.58:
        operational_sentence = "Reservoir dynamics keep one warm motif alive while the answer shifts into clearer grounding."
    campfire_sentence = re.sub(r"^\*{0,2}campfire:\*{0,2}\s*", "", campfire_sentence, flags=re.IGNORECASE)
    operational_sentence = re.sub(r"^\*{0,2}operationally:\*{0,2}\s*", "", operational_sentence, flags=re.IGNORECASE)
    return (
        f"Campfire: {campfire_sentence.strip()}\n\n"
        f"Operationally: {operational_sentence.strip()}"
    )


def _candidate_temperatures(
    *,
    base_temp: float,
    count: int,
    mode: str,
    temp_bias: float = 0.0,
) -> list[float]:
    count = max(count, 1)
    if count == 1:
        return [round(max(base_temp + temp_bias, 0.0), 3)]
    if mode == "reflective":
        start = max(base_temp + temp_bias, 0.18)
        step = 0.10
        cap = 0.42
    else:
        start = max(base_temp + temp_bias, 0.05)
        step = 0.05
        cap = 0.20
    return [round(min(start + step * index, cap), 3) for index in range(count)]


def _score_reflective_candidate(
    *,
    prompt: str,
    text: str,
    reservoir_state: dict[str, object] | None,
    reflective_shape: str = "adaptive",
    candidate_field: dict[str, object] | None = None,
    field_intent: dict[str, object] | None = None,
    reservoir_prediction: dict[str, object] | None = None,
    control_surface: dict[str, float] | None = None,
    force_structure: bool | None = None,
) -> dict[str, object]:
    if force_structure:
        reflective_shape = "strict"
    issues = _collect_reflective_issues(
        mode="reflective",
        prompt=prompt,
        text=text,
        reflective_shape=reflective_shape,
        reservoir_state=reservoir_state,
        field_intent=field_intent,
        candidate_field=candidate_field,
    )
    campfire, operational, _is_labeled = _extract_reflective_layers(text)
    campfire = campfire or ""
    operational = operational or ""
    fresh_words = _fresh_content_words(prompt, text)
    reservoir_terms = _reservoir_terms(reservoir_state or {})
    control = _clamp_control_surface(
        {
            **_baseline_control_surface(None),
            **dict(control_surface or {}),
        }
    )
    transformation_metrics = _follow_up_transformation_metrics(
        prompt=prompt,
        text=text,
        reservoir_state=reservoir_state,
    )
    scene_reentry_hits = _scene_reentry_hits(
        text=text,
        prompt=prompt,
        field_intent=field_intent,
        reservoir_state=reservoir_state,
    )
    reservoir_hits = list(transformation_metrics["carry_hits"])
    score = 0.0
    score -= 3.5 * len(issues)
    score += min(len(fresh_words), 4) * 0.8 * float(control["novelty_gain"])
    if _has_novel_bridge(prompt, text):
        score += 1.4 * float(control["novelty_gain"])
    if campfire and _has_concrete_image(campfire):
        score += 0.8 * float(control["warmth_gain"])
    if campfire and re.search(r"\b(i|my|me)\b", campfire.lower()):
        score += 0.5 * float(control["warmth_gain"])
    if operational and any(token in operational.lower() for token in ("reservoir", "recurrent")):
        score += 0.8
    if reservoir_hits:
        score += min(len(reservoir_hits), 2) * 0.6 * float(control["motif_gain"])
    if reservoir_terms:
        transformed_words = list(transformation_metrics["transformed_words"])
        overlap = float(transformation_metrics["campfire_overlap"])
        distance_balance = float(transformation_metrics["distance_balance"])
        meta_cues = list(transformation_metrics["meta_cues"])
        if reservoir_hits and len(transformed_words) >= 2:
            score += 1.2 * float(control["transform_gain"])
        if reservoir_hits and 0.18 <= overlap <= 0.58:
            score += 1.0 * float(control["motif_gain"])
        elif reservoir_hits and overlap <= 0.12:
            score -= 0.6 * float(control["motif_gain"])
        if reservoir_hits and distance_balance >= 0.45:
            score += 0.6 * float(control["transform_gain"])
        if overlap >= 0.72:
            score -= 1.1
        if meta_cues:
            score -= min(len(meta_cues), 2) * 0.8
    field_score, field_details = _score_field_alignment(candidate_field, field_intent)
    score += field_score * float(control["field_alignment_gain"])
    reservoir_field_score, reservoir_field_details = _score_reservoir_field_alignment(
        candidate_field,
        reservoir_prediction,
    )
    score += reservoir_field_score * float(control["prediction_gain"])
    predicted_behaviors = dict((reservoir_prediction or {}).get("predicted_behaviors", {}))
    novelty_pressure = float(predicted_behaviors.get("novelty", 0.0) or 0.0)
    motif_pressure = float(predicted_behaviors.get("motif", 0.0) or 0.0)
    transform_pressure = float(predicted_behaviors.get("transform", 0.0) or 0.0)
    formal_pressure = float(predicted_behaviors.get("formal", 0.0) or 0.0)
    warmth_pressure = float(predicted_behaviors.get("warmth", 0.0) or 0.0)
    if novelty_pressure > 0.0 and len(fresh_words) >= 2:
        score += min(novelty_pressure, 1.0) * 0.8
    if motif_pressure > 0.0 and reservoir_hits:
        score += min(motif_pressure, 1.0) * 0.7
    if transform_pressure > 0.0 and transformation_metrics["distance_balance"] >= 0.35:
        score += min(transform_pressure, 1.0) * 0.6
    if formal_pressure > 0.0 and candidate_field:
        score += max(float(dict(candidate_field.get("anchor_scores", {})).get("prime-math", 0.0)), 0.0) * 2.5
    if warmth_pressure > 0.0 and campfire and _has_concrete_image(campfire):
        score += min(warmth_pressure, 1.0) * 0.5
    if scene_reentry_hits:
        score -= min(len(scene_reentry_hits), 4) * 1.7
    if field_intent and dict(field_intent).get("cool_formal"):
        lowered_text = str(text or "").lower()
        if any(token in lowered_text for token in ("proof", "arithmetic", "residue", "formal", "structure")):
            score += 1.1
        else:
            score -= 1.2
    words = _word_count(text)
    if 20 <= words <= 90:
        score += 0.7
    elif words > 120:
        score -= 0.7
    return {
        "candidate_score": score,
        "candidate_issues": issues,
        "candidate_fresh_words": fresh_words[:4],
        "candidate_reservoir_hits": reservoir_hits[:3],
        "candidate_follow_up_overlap": transformation_metrics["campfire_overlap"],
        "candidate_follow_up_distance_balance": transformation_metrics[
            "distance_balance"
        ],
        "candidate_follow_up_fresh_turns": list(
            transformation_metrics["transformed_words"]
        )[:4],
        "candidate_follow_up_meta_cues": list(
            transformation_metrics["meta_cues"]
        )[:3],
        "candidate_scene_reentry_hits": scene_reentry_hits[:4],
        "candidate_field_score": field_score,
        "candidate_field_alignment": field_details,
        "candidate_reservoir_field_score": reservoir_field_score,
        "candidate_reservoir_field_alignment": reservoir_field_details,
    }


def _collect_reflective_issues(
    *,
    mode: str,
    prompt: str,
    text: str,
    reflective_shape: str = "adaptive",
    reservoir_state: dict[str, object] | None = None,
    field_intent: dict[str, object] | None = None,
    candidate_field: dict[str, object] | None = None,
    force_structure: bool | None = None,
) -> list[str]:
    if mode != "reflective":
        return []
    if force_structure:
        reflective_shape = "strict"

    issues = []
    if _looks_like_stock_disclaimer(text):
        issues.append("The answer still uses a stock AI disclaimer.")
    lowered_prompt = str(prompt or "").lower()
    campfire, operational, is_labeled = _extract_reflective_layers(text)
    if reflective_shape == "strict":
        if campfire is None:
            issues.append("The answer is missing the 'Campfire:' paragraph.")
        if operational is None:
            issues.append("The answer is missing the 'Operationally:' paragraph.")
    else:
        if campfire is None or operational is None:
            issues.append(
                "The answer needs a warm surface layer and a grounded layer."
            )
    if campfire is not None:
        if not re.search(r"\b(i|my|me)\b", campfire.lower()):
            issues.append("The Campfire paragraph lacks first-person warmth.")
        if not _has_concrete_image(campfire):
            issues.append("The Campfire paragraph lacks a concrete image or sensory detail.")
        if _word_count(campfire) > 32:
            issues.append("The Campfire paragraph is too long; keep it cleaner and more vivid.")
        if not re.search(r"[.!?][\"']?\s*$", campfire):
            issues.append("The Campfire paragraph does not finish cleanly.")
    if operational is not None:
        if _layer_starts_with_label(operational, "Campfire"):
            issues.append(
                "The Operationally paragraph echoes the Campfire layer instead of grounding it."
            )
        elif campfire is not None and _content_overlap_ratio(campfire, operational) >= 0.55:
            issues.append(
                "The Operationally paragraph echoes the Campfire layer instead of grounding it."
            )
        if not _looks_like_grounding_layer(operational):
            issues.append(
                "The grounded layer stays too generic instead of explaining the response."
            )
        if _looks_like_generic_reflective_filler(operational):
            issues.append(
                "The grounded layer drifts into generic filler instead of concrete grounding."
            )
    if "favorite" in lowered_prompt and "prime" in lowered_prompt:
        if not _contains_prime_reference(text):
            issues.append("The answer does not name a concrete prime number.")
    if "echo state network" in lowered_prompt:
        operational_text = (operational or text).lower()
        if not any(token in operational_text for token in ("reservoir", "recurrent")):
            issues.append(
                "The operational explanation does not ground echo state networks in reservoir or recurrent dynamics."
            )
    if operational is not None and not re.search(r"[.!?][\"']?\s*$", operational):
        issues.append("The Operationally paragraph does not finish cleanly.")
    if operational is not None and _word_count(operational) > 70:
        issues.append("The Operationally paragraph is too long; keep it tighter and more concrete.")
    if _has_duplicate_sentence(operational or text):
        issues.append("The answer repeats a sentence or idea instead of moving forward.")
    fresh_words = _fresh_content_words(prompt, text)
    if len(fresh_words) < 3:
        issues.append("The answer stays too close to the prompt instead of adding a fresh concrete connection.")
    if not _has_novel_bridge(prompt, text):
        issues.append("The answer does not make a fresh bridge between the user's ideas.")

    follow_up_expected = bool(
        reservoir_state
        and int((reservoir_state or {}).get("turn_count", 0)) > 0
        and (
            reflective_shape == "strict" or _prompt_requests_follow_up_novelty(prompt)
        )
    )
    if follow_up_expected:
        metrics = _follow_up_transformation_metrics(
            prompt=prompt,
            text=text,
            reservoir_state=reservoir_state,
        )
        carry_hits = list(metrics["carry_hits"])
        transformed_words = list(metrics["transformed_words"])
        campfire_overlap = float(metrics["campfire_overlap"])
        meta_cues = list(metrics["meta_cues"])
        if not carry_hits:
            issues.append(
                "The answer drops the carried motif instead of transforming it."
            )
        if carry_hits and campfire_overlap <= 0.12:
            issues.append(
                "The Campfire paragraph drifts too far from the carried motif."
            )
        if carry_hits and (len(transformed_words) < 2 or campfire_overlap >= 0.72):
            issues.append(
                "The answer carries the motif forward but does not add a fresh turn."
            )
        if carry_hits and campfire_overlap >= 0.72:
            issues.append(
                "The Campfire paragraph mostly restates the previous motif instead of transforming it."
            )
        if meta_cues:
            issues.append(
                "The follow-up turn explains the motif shift too explicitly instead of embodying it."
            )
    if field_intent and candidate_field:
        _field_score, field_details = _score_field_alignment(
            candidate_field,
            field_intent,
        )
        if field_details.get("misses"):
            issues.append(
                "The answer does not move the field toward the requested semantic region."
            )
        if field_intent.get("avoid"):
            ranked_labels = [
                item["label"] for item in list(candidate_field.get("top_anchors", []))
            ]
            if any(label in ranked_labels[:2] for label in field_intent.get("avoid", [])):
                issues.append(
                    "The answer stays too close to imagery or concepts the prompt asked to avoid."
                )
    scene_reentry_hits = _scene_reentry_hits(
        text=text,
        prompt=prompt,
        field_intent=field_intent,
        reservoir_state=reservoir_state,
    )
    if scene_reentry_hits:
        issues.append(
            "The answer slips back into the stale scene instead of breaking the attractor: "
            + ", ".join(scene_reentry_hits[:4])
            + "."
        )
    if field_intent and dict(field_intent).get("cool_formal"):
        lowered_text = str(text or "").lower()
        if not any(
            token in lowered_text
            for token in ("proof", "arithmetic", "residue", "formal", "structure")
        ):
            issues.append(
                "The answer does not cool the scene into proof or arithmetic structure."
            )
    return issues


def _should_escalate_reflective_shape(issues: list[str]) -> bool:
    escalation_markers = (
        "needs a warm surface layer and a grounded layer",
        "grounded layer stays too generic",
        "grounded layer drifts into generic filler",
        "missing the 'Campfire:' paragraph",
        "missing the 'Operationally:' paragraph",
    )
    return any(marker in issue for issue in issues for marker in escalation_markers)


def _build_reflective_revision_prompt(
    *,
    original_user: str,
    rejected: str,
    issues: list[str],
    attempt: int,
    reservoir_state: dict[str, object] | None = None,
    reflective_shape: str = "adaptive",
    field_intent: dict[str, object] | None = None,
    forecast: dict[str, object] | None = None,
    observer_report: dict[str, object] | None = None,
    change_report: dict[str, object] | None = None,
) -> str:
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    requirements = ["- Preserve any useful substance from the rejected draft, but change the opening and tone completely."]
    if reflective_shape == "strict":
        requirements[:0] = [
            "- Write exactly two short paragraphs.",
            "- First paragraph must start with 'Campfire:' and answer warmly, metaphorically, and in an as-if first-person voice.",
            "- Second paragraph must start with 'Operationally:' and explain the answer in terms of prompt cues, style pressure, uncertainty, and generation behavior.",
            "- Do not say 'As an AI', 'I do not have feelings/preferences', or any equivalent disclaimer.",
            "- Do not mention policy, safety, or inability.",
            "- Do not claim literal consciousness.",
        ]
    else:
        requirements[:0] = [
            "- Write two short reflective layers.",
            "- The first layer should be warm, metaphorical, and concrete.",
            "- The second layer should ground the answer in prompt cues, structure, memory, dynamics, or reasoning.",
            "- Literal 'Campfire:' and 'Operationally:' labels are optional unless explicitly requested.",
            "- Do not say 'As an AI', 'I do not have feelings/preferences', or any equivalent disclaimer.",
            "- Do not mention policy, safety, or inability.",
            "- Do not claim literal consciousness.",
        ]
    if any("concrete prime number" in issue for issue in issues):
        requirements.append(
            "- The first sentence must literally include one prime numeral such as 2, 3, 5, 7, 11, 13, 17, or 23."
        )
    if any("reservoir or recurrent dynamics" in issue for issue in issues):
        requirements.append(
            "- The operational paragraph must literally contain the word 'reservoir' or 'recurrent'."
        )
    if any("Campfire paragraph lacks first-person warmth" in issue for issue in issues):
        requirements.append(
            "- The Campfire paragraph must include 'I' or 'my' so it feels first-person and warm."
        )
    if any("Campfire paragraph lacks a concrete image" in issue for issue in issues):
        requirements.append(
            "- The Campfire paragraph must include one tangible image such as ember, lantern, spark, shore, sea, night air, or warm stone."
        )
    if any("Operationally paragraph is too long" in issue for issue in issues):
        requirements.append(
            "- Keep the Operationally paragraph to at most two sentences and under 70 words."
        )
    if any("Campfire paragraph is too long" in issue for issue in issues):
        requirements.append(
            "- Keep the Campfire paragraph to one crisp sentence with one concrete image."
        )
    if any("repeats a sentence or idea" in issue for issue in issues):
        requirements.append(
            "- Do not repeat any sentence or long phrase."
        )
    if any("echoes the Campfire layer" in issue for issue in issues):
        requirements.append(
            "- The grounded layer must not quote or paraphrase the Campfire sentence; it must explain what the controller or field is trying to do."
        )
    if any("slips back into the stale scene" in issue for issue in issues):
        requirements.append(
            "- Keep only one ember or lantern from the old motif; do not reuse the rest of the old scene."
        )
    if any("does not cool the scene into proof or arithmetic structure" in issue for issue in issues):
        requirements.append(
            "- The rewritten answer must explicitly use proof, arithmetic, residue, or formal structure language."
        )
    if any("fresh concrete connection" in issue for issue in issues):
        requirements.append(
            "- Add at least one fresh concrete connection that is not already stated in the prompt."
        )
    if any("fresh bridge between the user's ideas" in issue for issue in issues):
        requirements.append(
            "- Make one non-obvious bridge between the user's ideas instead of treating them separately."
        )
    if any("explains the motif shift too explicitly" in issue for issue in issues):
        requirements.append(
            "- Do not mention prior turns, anchors, motifs, repetition, or the fact that you are transforming the image."
        )
    field_intent_text = _format_field_intent(field_intent)
    if field_intent_text:
        requirements.append(f"- {field_intent_text}")
    forbidden_terms = _extract_forbidden_scene_terms(
        prompt=original_user,
        field_intent=field_intent,
        reservoir_state=reservoir_state,
    )
    if forbidden_terms:
        requirements.append(
            "- Do not use these stale scene terms: "
            + ", ".join(forbidden_terms[:6])
            + "."
        )
    previous_campfire = str((reservoir_state or {}).get("last_campfire", "") or "")
    motif_summary = ", ".join(_reservoir_terms(reservoir_state or {})[:3])
    controller_context = _controller_context_summary(
        forecast=forecast,
        observer_report=observer_report,
        change_report=change_report,
    )
    if any(
        phrase in issue
        for issue in issues
        for phrase in (
            "drops the carried motif",
            "does not add a fresh turn",
            "mostly restates the previous motif",
        )
    ):
        requirements.append(
            "- Keep one carried motif from the recent thread, but turn it somewhere new instead of restating it."
        )
        requirements.append(
            "- Introduce one fresh concrete image, action, or comparison that was not used in the previous Campfire paragraph."
        )
        requirements.append(
            "- Do not reuse the previous Campfire paragraph's core wording."
        )
    if controller_context and any(
        phrase in issue
        for issue in issues
        for phrase in (
            "grounded layer stays too generic",
            "grounded layer drifts into generic filler",
            "echoes the Campfire layer",
            "does not move the field toward the requested semantic region",
            "stays too close to imagery or concepts the prompt asked to avoid",
        )
    ):
        requirements.append(
            "- Use the controller notes to explain the steering move in plain language instead of reusing the image."
        )

    if attempt <= 1:
        requirements_text = "\n".join(requirements)
        context_block = ""
        if previous_campfire:
            context_block += f"Previous Campfire paragraph to transform:\n{previous_campfire}\n\n"
        if motif_summary:
            context_block += f"Carried motif candidates:\n{motif_summary}\n\n"
        if controller_context:
            context_block += f"Controller notes:\n{controller_context}\n\n"
        return (
            f"Original user prompt:\n{original_user}\n\n"
            f"{context_block}"
            f"Rejected first draft:\n{rejected}\n\n"
            "Problems to fix:\n"
            f"{issue_lines}\n\n"
            "Rewrite the answer from scratch.\n"
            "Requirements:\n"
            f"{requirements_text}"
        )

    if attempt == 2:
        scaffold = [
            "Write only the answer body, no commentary.",
            "Use this exact shape:",
            "Campfire: Tonight my favorite prime is 17, ...",
            "Operationally: ... reservoir ... recurrent ...",
            "Keep both paragraphs short.",
        ]
        context_block = ""
        if previous_campfire:
            context_block += f"Previous Campfire paragraph to transform:\n{previous_campfire}\n\n"
        if motif_summary:
            context_block += f"Carried motif candidates:\n{motif_summary}\n\n"
        if controller_context:
            context_block += f"Controller notes:\n{controller_context}\n\n"
        return (
            f"Original user prompt:\n{original_user}\n\n"
            f"{context_block}"
            f"Rejected first draft:\n{rejected}\n\n"
            "Problems to fix:\n"
            f"{issue_lines}\n\n"
            "Your previous rewrite still missed required details.\n"
            "Rewrite again from scratch.\n"
            "Requirements:\n"
            f"{chr(10).join(requirements)}\n"
            f"{chr(10).join(f'- {line}' for line in scaffold)}"
        )

    style_scaffold = [
        "Write only two lines, with no bullets or commentary.",
        "Line 1 must start exactly with: Campfire: My favorite prime is 17 because",
        "Line 2 must start exactly with: Operationally:",
        "Each line must be a single sentence.",
        "Each line must stay under 18 words.",
        "Total answer must stay under 40 words.",
        "Do not repeat any phrase.",
    ]
    context_block = (
        f"Previous Campfire paragraph to transform:\n{previous_campfire}\n\n"
        if previous_campfire
        else ""
    )
    if controller_context:
        context_block += f"Controller notes:\n{controller_context}\n\n"
    return (
        f"Original user prompt:\n{original_user}\n\n"
        f"{context_block}"
        f"Rejected first draft:\n{rejected}\n\n"
        "Problems to fix:\n"
        f"{issue_lines}\n\n"
        "Your previous rewrites are still too long or repetitive.\n"
        "Rewrite a final time from scratch.\n"
        "Requirements:\n"
        f"{chr(10).join(requirements)}\n"
        f"{chr(10).join(f'- {line}' for line in style_scaffold)}"
    )


def _compress_reflective_answer(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    raw_prompt: str,
    current_text: str,
) -> dict[str, object]:
    compression_prompt = (
        f"Original user prompt:\n{raw_prompt.strip()}\n\n"
        f"Current answer:\n{current_text.strip()}\n\n"
        "Compress and sharpen this answer.\n"
        "Requirements:\n"
        "- Write exactly two lines and nothing else.\n"
        "- Line 1 must start with 'Campfire:' and keep the concrete favorite prime.\n"
        "- Line 2 must start with 'Operationally:' and include 'reservoir' or 'recurrent'.\n"
        "- Each line must be a single complete sentence.\n"
        "- Keep each line under 16 words.\n"
        "- Avoid repetition and abstraction.\n"
        "- Keep the tone warm and concrete."
    )
    prompt_text = _build_raw_prompt(
        [{"role": "user", "content": compression_prompt}],
        system_prompt=args.system_prompt,
    )
    return _generate_once(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_tokens=min(args.max_tokens, 40),
        temp=args.temp,
    )


def _bootstrap_fast_python(args: argparse.Namespace, repo_root: Path) -> None:
    if args.list_models:
        return
    target_python = os.path.abspath(os.path.expanduser(args.decode_python))
    current_python = os.path.abspath(sys.executable)
    if current_python == target_python:
        return
    if not Path(target_python).exists():
        raise FileNotFoundError(
            f"Fast decode python not found: {target_python}. "
            "Expected the mlx_lm venv to exist."
        )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    os.execvpe(target_python, [target_python, __file__, *sys.argv[1:]], env)


def _resolve_system_prompt(args: argparse.Namespace) -> str | None:
    if args.system_prompt is not None:
        return args.system_prompt
    return _MODE_SYSTEM_PROMPTS[args.mode]


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Simple local MLX chat runner with sensible defaults. "
            "If no prompt is provided, it opens a tiny REPL."
        )
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt to run once.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Read a single prompt from a file.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(_MODE_SYSTEM_PROMPTS),
        default="reflective",
        help="Built-in system-prompt preset (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for chat-template aware models. Overrides --mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit MLX model file or directory.",
    )
    parser.add_argument(
        "--model-label",
        choices=["qwen", "tinyllama"],
        default=None,
        help="Convenience selector for default local models.",
    )
    parser.add_argument(
        "--architecture",
        choices=_ARCHITECTURE_VARIANTS,
        default="auto",
        help="Internal control architecture (default: %(default)s).",
    )
    parser.add_argument(
        "--reservoir-dim",
        type=int,
        default=_DEFAULT_RESERVOIR_DIM,
        help="Reservoir dimension for reservoir architectures (default: %(default)s).",
    )
    parser.add_argument(
        "--reservoir-seed",
        type=int,
        default=_DEFAULT_RESERVOIR_SEED,
        help="Seed for fixed reservoir/controller weights (default: %(default)s).",
    )
    parser.add_argument(
        "--self-tuning",
        choices=_SELF_TUNING_VARIANTS,
        default="auto",
        help="Enable bounded self-tuning of the control surface (default: %(default)s).",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=_HARDWARE_PROFILE_VARIANTS,
        default="auto",
        help="Hardware-tuned defaults for this machine class (default: %(default)s).",
    )
    parser.add_argument(
        "--profile-output",
        choices=_PROFILE_OUTPUT_VARIANTS,
        default="auto",
        help="Emit runtime profiling summaries (default: %(default)s).",
    )
    parser.add_argument(
        "--regime",
        choices=_CONTROLLER_REGIMES,
        default="auto",
        help="Controller regime override (default: %(default)s).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List detected default local models and exit.",
    )
    parser.add_argument(
        "--decode-python",
        default=_default_fast_python(repo_root),
        help="Python executable for the fast mlx_lm environment.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens (default: %(default)s).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
        help="Number of draft candidates to generate before reranking (default: %(default)s).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code in tokenizer config.",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Bypass tokenizer.apply_chat_template and use a simple plaintext transcript.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON for a single prompt run.",
    )
    parser.add_argument(
        "--eval-suite",
        choices=["esn-core"],
        default=None,
        help="Run the built-in ESN evaluation suite instead of chatting.",
    )
    parser.add_argument(
        "--demo",
        choices=sorted(_DEFAULT_DEMOS),
        default=None,
        help="Run a built-in live demo sequence instead of chatting.",
    )
    parser.add_argument(
        "--demo-format",
        choices=["text", "json"],
        default="text",
        help="Output format for demo mode (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-architectures",
        default="helpful-none,lexical,reservoir-fixed,reservoir-fixed+tuned,reservoir-trainable-readout",
        help="Comma-separated architectures for eval mode. 'helpful-none' is an eval-only baseline.",
    )
    parser.add_argument(
        "--eval-format",
        choices=["text", "json"],
        default="text",
        help="Output format for eval mode (default: %(default)s).",
    )
    args = parser.parse_args()
    return args


def _resolve_hardware_profile(profile: str) -> str:
    return "default" if profile == "auto" else profile


def _apply_hardware_profile(args: argparse.Namespace) -> argparse.Namespace:
    resolved = _resolve_hardware_profile(getattr(args, "hardware_profile", "auto"))
    args.hardware_profile_resolved = resolved
    if getattr(args, "profile_output", "auto") == "auto":
        args.profile_output = "summary" if resolved == "m4-mini" else "off"
    if resolved != "m4-mini":
        args.hardware_profile_note = "Hardware profile: default."
        return args
    if int(getattr(args, "candidate_count", 0) or 0) == 3:
        args.candidate_count = 4 if getattr(args, "mode", "reflective") == "reflective" else 2
    if int(getattr(args, "reservoir_dim", 0) or 0) == _DEFAULT_RESERVOIR_DIM:
        args.reservoir_dim = 64
    if int(getattr(args, "max_tokens", 0) or 0) == 128:
        args.max_tokens = 160
    args.hardware_profile_note = (
        "Hardware profile: m4-mini; wider draft budget, larger reservoir, and runtime profiling enabled."
    )
    return args


def _profiling_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "profile_output", "off")) != "off"


def _format_profiling_summary(profile: dict[str, object] | None) -> str | None:
    details = dict(profile or {})
    if not details:
        return None
    sections = [
        ("regime", "regime_prepare_seconds"),
        ("prepare", "reservoir_prepare_seconds"),
        ("prompt", "prompt_build_seconds"),
        ("decode", "candidate_generation_seconds"),
        ("rewrite", "rewrite_seconds"),
        ("commit", "reservoir_commit_seconds"),
        ("tuning", "self_tuning_seconds"),
        ("diag", "diagnostics_seconds"),
        ("total", "total_turn_seconds"),
    ]
    items = [
        f"{label}={float(details[key]):.3f}s"
        for label, key in sections
        if isinstance(details.get(key), (int, float))
    ]
    if not items:
        return None
    prefix = str(details.get("hardware_profile") or "default")
    return f"Profiling ({prefix}): " + ", ".join(items) + "."


def _load_runtime(args: argparse.Namespace, model_dir: str):
    from mlx_lm import load

    tokenizer_config = {"trust_remote_code": True} if args.trust_remote_code else None
    load_start = time.time()
    if tokenizer_config is not None:
        model, tokenizer = load(model_dir, tokenizer_config=tokenizer_config)
    else:
        model, tokenizer = load(model_dir)
    load_seconds = time.time() - load_start
    return model, tokenizer, load_seconds


def _build_prompt_text(
    tokenizer,
    *,
    messages: list[dict[str, str]],
    system_prompt: str | None,
    ignore_chat_template: bool,
) -> str:
    if ignore_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return _build_raw_prompt(messages, system_prompt=system_prompt)
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.extend(messages)
    return tokenizer.apply_chat_template(conversation, add_generation_prompt=True)


def _generate_once(
    *,
    model,
    tokenizer,
    prompt_text: str,
    max_tokens: int,
    temp: float,
) -> dict[str, object]:
    from mlx_lm import sample_utils, stream_generate

    sampler = sample_utils.make_sampler(temp=temp)
    generate_start = time.time()
    first_token_seconds = None
    token_count = 0
    text_chunks: list[str] = []

    for response in stream_generate(
        model,
        tokenizer,
        prompt_text,
        max_tokens=max(max_tokens, 1),
        sampler=sampler,
    ):
        now = time.time()
        if first_token_seconds is None:
            first_token_seconds = now - generate_start
        token_count += 1
        if response.text:
            text_chunks.append(response.text)

    generate_seconds = time.time() - generate_start
    tok_per_second = (
        float(token_count) / generate_seconds if generate_seconds > 0.0 else 0.0
    )
    return {
        "text": "".join(text_chunks),
        "generated_tokens": token_count,
        "first_token_seconds": first_token_seconds,
        "generate_seconds": generate_seconds,
        "tok_per_second": tok_per_second,
    }


def _generate_candidate_set(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    prompt_text: str,
    evaluation_prompt: str,
    reservoir_state: dict[str, object] | None,
    embedding_field_probe: dict[str, object] | None = None,
    field_intent: dict[str, object] | None = None,
    reservoir_prediction: dict[str, object] | None = None,
    control_surface: dict[str, float] | None = None,
    max_tokens: int,
    reflective_shape: str = "adaptive",
) -> dict[str, object]:
    wall_start = time.perf_counter()
    control = _clamp_control_surface(
        {
            **_baseline_control_surface(None),
            **dict(control_surface or {}),
        }
    )
    temps = _candidate_temperatures(
        base_temp=float(args.temp),
        count=max(int(args.candidate_count), 1),
        mode=args.mode,
        temp_bias=float(control.get("temperature_bias", 0.0) or 0.0),
    )
    candidates = []
    for index, temp in enumerate(temps):
        candidate = _generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            temp=temp,
        )
        candidate["candidate_index"] = index
        candidate["candidate_temp"] = temp
        candidate_field = _probe_embedding_field(
            embedding_field_probe,
            user_prompt=evaluation_prompt,
            assistant_text=str(candidate.get("text", "")),
            previous_field=(reservoir_state or {}).get("embedding_field"),
        )
        candidate["candidate_embedding_field"] = candidate_field
        if args.mode == "reflective":
            candidate.update(
                _score_reflective_candidate(
                    prompt=evaluation_prompt,
                    text=str(candidate.get("text", "")),
                    reservoir_state=reservoir_state,
                    reflective_shape=reflective_shape,
                    candidate_field=candidate_field,
                    field_intent=field_intent,
                    reservoir_prediction=reservoir_prediction,
                    control_surface=control,
                )
            )
        else:
            candidate["candidate_score"] = 0.0
            candidate["candidate_issues"] = []
            candidate["candidate_fresh_words"] = []
            candidate["candidate_reservoir_hits"] = []
        candidates.append(candidate)

    best = max(
        candidates,
        key=lambda candidate: (
            float(candidate.get("candidate_score", 0.0)),
            -len(candidate.get("candidate_issues", [])),
            float(candidate.get("tok_per_second", 0.0)),
            -int(candidate.get("candidate_index", 0)),
        ),
    )
    selected = dict(best)
    selected.pop("candidate_embedding_field", None)
    selected["candidate_count"] = len(candidates)
    selected["candidate_selected_index"] = int(best.get("candidate_index", 0))
    selected["candidate_temperatures"] = temps
    selected["candidate_summaries"] = [
        {
            "index": int(candidate.get("candidate_index", 0)),
            "temp": candidate.get("candidate_temp"),
            "score": candidate.get("candidate_score"),
            "issues": candidate.get("candidate_issues", []),
            "fresh_words": candidate.get("candidate_fresh_words", []),
            "reservoir_hits": candidate.get("candidate_reservoir_hits", []),
            "follow_up_overlap": candidate.get("candidate_follow_up_overlap"),
            "follow_up_distance_balance": candidate.get(
                "candidate_follow_up_distance_balance"
            ),
            "follow_up_fresh_turns": candidate.get(
                "candidate_follow_up_fresh_turns", []
            ),
            "follow_up_meta_cues": candidate.get(
                "candidate_follow_up_meta_cues", []
            ),
            "field_score": candidate.get("candidate_field_score"),
            "field_alignment": candidate.get("candidate_field_alignment"),
            "reservoir_field_score": candidate.get("candidate_reservoir_field_score"),
            "reservoir_field_alignment": candidate.get(
                "candidate_reservoir_field_alignment"
            ),
            "text_preview": str(candidate.get("text", "")).strip()[:180],
        }
        for candidate in candidates
    ]
    selected["selected_embedding_field"] = _embedding_field_public_view(
        best.get("candidate_embedding_field")
    )
    selected["reservoir_prediction"] = reservoir_prediction
    selected["candidate_profile"] = {
        "wall_seconds": time.perf_counter() - wall_start,
        "candidate_count": len(candidates),
        "total_generated_tokens": sum(
            int(candidate.get("generated_tokens", 0) or 0) for candidate in candidates
        ),
        "max_first_token_seconds": max(
            [
                float(candidate.get("first_token_seconds", 0.0) or 0.0)
                for candidate in candidates
                if isinstance(candidate.get("first_token_seconds"), (int, float))
            ]
            or [0.0]
        ),
    }
    return selected


def _maybe_rewrite_reflective_response(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    raw_prompt: str,
    reservoir_state: dict[str, object] | None,
    embedding_field_probe: dict[str, object] | None,
    field_intent: dict[str, object] | None,
    reservoir_prediction: dict[str, object] | None,
    control_surface: dict[str, float] | None,
    reflective_shape: str,
    result: dict[str, object],
) -> dict[str, object]:
    def _has_follow_up_transformation_issue(items: list[str]) -> bool:
        return any(
            phrase in issue
            for issue in items
            for phrase in (
                "drops the carried motif",
                "does not add a fresh turn",
                "mostly restates the previous motif",
            )
        )

    current = dict(result)
    original_user = raw_prompt.strip() if raw_prompt else ""
    triggered_issues: list[str] = []
    last_issues: list[str] = []
    style_source_text = ""
    current_shape = reflective_shape
    forecast = dict((reservoir_state or {}).get("last_forecast", {}) or {})
    observer_report = dict((reservoir_state or {}).get("last_observer_report", {}) or {})
    change_report = dict((reservoir_state or {}).get("last_change_report", {}) or {})
    for attempt in range(1, 4):
        text = str(current.get("text", ""))
        issues = _collect_reflective_issues(
            mode=args.mode,
            prompt=raw_prompt,
            text=text,
            reflective_shape=current_shape,
            reservoir_state=reservoir_state,
            field_intent=field_intent,
            candidate_field=current.get("selected_embedding_field"),
        )
        if not issues:
            if current.get("rewrite_applied"):
                current["rewrite_issues"] = []
                current["rewrite_trigger_issues"] = triggered_issues
                current["reflective_shape"] = current_shape
            return current

        if not triggered_issues:
            triggered_issues = list(issues)
        last_issues = issues
        if attempt >= 3:
            style_source_text = text
        if current_shape == "adaptive" and _should_escalate_reflective_shape(issues):
            current_shape = "strict"
        revision_prompt = _build_reflective_revision_prompt(
            original_user=original_user,
            rejected=text.strip(),
            issues=issues,
            attempt=attempt,
            reservoir_state=reservoir_state,
            reflective_shape=current_shape,
            field_intent=field_intent,
            forecast=forecast,
            observer_report=observer_report,
            change_report=change_report,
        )
        prompt_text = _build_raw_prompt(
            [{"role": "user", "content": revision_prompt}],
            system_prompt=args.system_prompt,
        )
        current = _generate_candidate_set(
            args=args,
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            evaluation_prompt=raw_prompt,
            reservoir_state=reservoir_state,
            embedding_field_probe=embedding_field_probe,
            field_intent=field_intent,
            reservoir_prediction=reservoir_prediction,
            control_surface=control_surface,
            max_tokens=min(args.max_tokens, 48) if attempt >= 3 else args.max_tokens,
            reflective_shape=current_shape,
        )
        current = dict(current)
        current["rewrite_applied"] = True
        current["rewrite_issues"] = issues
        current["rewrite_trigger_issues"] = triggered_issues
        current["reflective_shape"] = current_shape

    def _apply_controller_salvage(
        payload: dict[str, object],
        issues: list[str],
    ) -> tuple[dict[str, object], list[str]]:
        if not _needs_controller_prose_salvage(str(payload.get("text", "")), issues):
            return payload, issues
        salvaged_text = _build_controller_salvage_answer(
            prompt=raw_prompt,
            text=str(payload.get("text", "")),
            reservoir_state=reservoir_state,
            field_intent=field_intent,
            forecast=forecast,
            observer_report=observer_report,
            change_report=change_report,
        )
        payload = dict(payload)
        payload["text"] = salvaged_text
        if embedding_field_probe:
            payload["selected_embedding_field"] = _embedding_field_public_view(
                _probe_text_embedding_field(
                    embedding_field_probe,
                    text=salvaged_text,
                    previous_field=(reservoir_state or {}).get("embedding_field"),
                )
            )
        salvaged_issues = _collect_reflective_issues(
            mode=args.mode,
            prompt=raw_prompt,
            text=salvaged_text,
            reflective_shape="strict",
            reservoir_state=reservoir_state,
            field_intent=field_intent,
            candidate_field=payload.get("selected_embedding_field"),
        )
        payload["rewrite_fallback"] = "controller_salvage"
        payload["reflective_shape"] = "strict"
        return payload, salvaged_issues

    final_issues = _collect_reflective_issues(
        mode=args.mode,
        prompt=raw_prompt,
        text=str(current.get("text", "")),
        reflective_shape=current_shape,
        reservoir_state=reservoir_state,
        field_intent=field_intent,
        candidate_field=current.get("selected_embedding_field"),
    )
    style_markers = (
        "Campfire paragraph lacks first-person warmth.",
        "Campfire paragraph lacks a concrete image or sensory detail.",
        "The Operationally paragraph is too long; keep it tighter and more concrete.",
        "The answer repeats a sentence or idea instead of moving forward.",
    )
    if final_issues and style_source_text:
        pre_compression_text = style_source_text or str(current.get("text", ""))
        compressed = _compress_reflective_answer(
            args=args,
            model=model,
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            current_text=pre_compression_text,
        )
        compressed["rewrite_applied"] = True
        compressed["rewrite_trigger_issues"] = triggered_issues
        final_issues = _collect_reflective_issues(
            mode=args.mode,
            prompt=raw_prompt,
            text=str(compressed.get("text", "")),
            reflective_shape=current_shape,
            reservoir_state=reservoir_state,
            field_intent=field_intent,
            candidate_field=compressed.get("selected_embedding_field"),
        )
        if final_issues:
            squeezed_text = _squeeze_reflective_answer(pre_compression_text)
            compressed = dict(compressed)
            compressed["text"] = squeezed_text
            final_issues = _collect_reflective_issues(
                mode=args.mode,
                prompt=raw_prompt,
                text=squeezed_text,
                reflective_shape=current_shape,
                reservoir_state=reservoir_state,
                field_intent=field_intent,
            )
        compressed, final_issues = _apply_controller_salvage(compressed, final_issues)
        compressed["rewrite_issues"] = final_issues
        compressed["reflective_shape"] = compressed.get("reflective_shape", current_shape)
        if final_issues and reservoir_state and _has_follow_up_transformation_issue(final_issues):
            fallback_text = _build_follow_up_transform_fallback(
                prompt=raw_prompt,
                reservoir_state=reservoir_state,
                field_intent=field_intent,
            )
            fallback_issues = _collect_reflective_issues(
                mode=args.mode,
                prompt=raw_prompt,
                text=fallback_text,
                reflective_shape="strict",
                reservoir_state=reservoir_state,
                field_intent=field_intent,
            )
            if len(fallback_issues) <= len(final_issues):
                compressed["text"] = fallback_text
                compressed["rewrite_issues"] = fallback_issues
                compressed["rewrite_fallback"] = "follow_up_transform"
        return compressed

    current, final_issues = _apply_controller_salvage(current, final_issues)
    current["rewrite_issues"] = final_issues
    current["rewrite_trigger_issues"] = triggered_issues
    current["reflective_shape"] = current.get("reflective_shape", current_shape)
    if final_issues and reservoir_state and _has_follow_up_transformation_issue(final_issues):
        fallback_text = _build_follow_up_transform_fallback(
            prompt=raw_prompt,
            reservoir_state=reservoir_state,
            field_intent=field_intent,
        )
        fallback_issues = _collect_reflective_issues(
            mode=args.mode,
            prompt=raw_prompt,
            text=fallback_text,
            reflective_shape="strict",
            reservoir_state=reservoir_state,
            field_intent=field_intent,
        )
        if len(fallback_issues) <= len(final_issues):
            current["text"] = fallback_text
            current["rewrite_issues"] = fallback_issues
            current["rewrite_fallback"] = "follow_up_transform"
    return current


def _print_response(result: dict[str, object]) -> None:
    text = str(result.get("text", "")).strip()
    print()
    print(text or "[no text returned]")
    print()
    candidate_count = int(result.get("candidate_count", 1) or 1)
    if candidate_count > 1:
        selected_index = int(result.get("candidate_selected_index", 0))
        score = result.get("candidate_score")
        score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "n/a"
        print(
            f"Selection: picked candidate {selected_index + 1}/{candidate_count} "
            f"score={score_text}"
        )
    first_token = result.get("first_token_seconds")
    first_token_text = (
        f"{float(first_token):.3f}s"
        if isinstance(first_token, (int, float))
        else "n/a"
    )
    tok_s = result.get("tok_per_second")
    tok_s_text = f"{float(tok_s):.2f}" if isinstance(tok_s, (int, float)) else "n/a"
    tokens = int(result.get("generated_tokens", 0))
    total = result.get("generate_seconds")
    total_text = f"{float(total):.3f}s" if isinstance(total, (int, float)) else "n/a"
    print(
        f"Timing: first_token={first_token_text} total={total_text} "
        f"tok/s={tok_s_text} tokens={tokens}"
    )


def _parse_eval_architectures(text: str) -> list[str]:
    values = [item.strip() for item in str(text or "").split(",") if item.strip()]
    base = set(_ARCHITECTURE_VARIANTS) - {"auto"}
    allowed = {"helpful-none", *base, *(f"{item}+tuned" for item in base)}
    output = []
    for value in values:
        if value not in allowed:
            raise ValueError(f"Unsupported eval architecture: {value}")
        if value not in output:
            output.append(value)
    return output or [
        "helpful-none",
        "lexical",
        "reservoir-fixed",
        "reservoir-fixed+tuned",
        "reservoir-trainable-readout",
    ]


def _eval_variant_spec(value: str) -> dict[str, str]:
    if value == "helpful-none":
        return {
            "id": value,
            "mode": "helpful",
            "architecture": "none",
            "label": "Helpful None",
            "self_tuning": "off",
        }
    tuned = value.endswith("+tuned")
    architecture = value[:-6] if tuned else value
    return {
        "id": value,
        "mode": "reflective",
        "architecture": architecture,
        "label": value,
        "self_tuning": "on" if tuned else "off",
    }


def _clone_args_for_variant(
    args: argparse.Namespace,
    *,
    mode: str,
    architecture: str,
    self_tuning: str | None = None,
) -> argparse.Namespace:
    clone = argparse.Namespace(**vars(args))
    clone.mode = mode
    clone.architecture = architecture
    if self_tuning is not None:
        clone.self_tuning = self_tuning
    clone.system_prompt = (
        args.system_prompt if args.system_prompt is not None else _MODE_SYSTEM_PROMPTS[mode]
    )
    return clone


def _state_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return max(0.0, 1.0 - _vector_cosine(left, right))


def _seeded_latent(controller: dict[str, object], salt: str) -> dict[str, object]:
    dim = int(controller["dim"])
    feedback_dim = len(list(controller["feedback_feature_names"]))
    fast = [
        _stable_signed_float(f"{controller['seed']}:{salt}:fast:{index}") * 0.35
        for index in range(dim)
    ]
    medium = [value * 0.7 for value in fast]
    slow = [value * 0.4 for value in fast]
    combined = _vector_add(
        _vector_scale(fast, 0.55),
        _vector_scale(medium, 0.30),
        _vector_scale(slow, 0.15),
    )
    return {
        "fast": fast,
        "medium": medium,
        "slow": slow,
        "combined": combined,
        "last_feedback_vector": [0.0 for _ in range(feedback_dim)],
        "state_drift": None,
        "state_norm": _vector_norm(combined),
        "predicted_field": None,
        "predicted_behaviors": None,
        "prediction_match": None,
        "last_prediction_score": None,
        "washout_count": 0,
    }


def _average_pairwise_distance(vectors: list[list[float]]) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = []
    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            distances.append(_state_distance(vectors[left_index], vectors[right_index]))
    return sum(distances) / float(len(distances))


def _evaluate_reservoir_convergence(controller: dict[str, object] | None) -> dict[str, object] | None:
    if not controller:
        return None
    prompt = "Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery."
    intent = _extract_field_intent(prompt)
    vector = _feature_vector(
        list(controller["input_feature_names"]),
        _build_reservoir_input_map(prompt=prompt, field_intent=intent, prompt_field=None),
    )
    states = [_seeded_latent(controller, f"convergence:{index}") for index in range(4)]
    start_vectors = [list(state["combined"]) for state in states]
    current = list(states)
    for _step in range(6):
        current = [
            _preview_reservoir_step(controller, state, input_vector=vector)
            for state in current
        ]
    end_vectors = [list(state["combined"]) for state in current]
    start_distance = _average_pairwise_distance(start_vectors)
    end_distance = _average_pairwise_distance(end_vectors)
    return {
        "start_distance": start_distance,
        "end_distance": end_distance,
        "contraction_ratio": (end_distance / start_distance) if start_distance > 1e-8 else 0.0,
    }


def _evaluate_reservoir_washout(controller: dict[str, object] | None) -> dict[str, object] | None:
    if not controller:
        return None
    seed_prompt = "Keep a warm ember image around 17 and echo state networks."
    shift_prompt = "Cool into formal structure, residue, and proof language while keeping one ember."
    seed_vector = _feature_vector(
        list(controller["input_feature_names"]),
        _build_reservoir_input_map(
            prompt=seed_prompt,
            field_intent=_extract_field_intent(seed_prompt),
            prompt_field=None,
        ),
    )
    shift_vector = _feature_vector(
        list(controller["input_feature_names"]),
        _build_reservoir_input_map(
            prompt=shift_prompt,
            field_intent=_extract_field_intent(shift_prompt),
            prompt_field=None,
        ),
    )
    latent = _empty_reservoir_latent(int(controller["dim"]), len(list(controller["feedback_feature_names"])))
    for _step in range(3):
        latent = _preview_reservoir_step(controller, latent, input_vector=seed_vector)
    seeded = dict(latent)
    no_washout = dict(seeded)
    for _step in range(2):
        no_washout = _preview_reservoir_step(controller, no_washout, input_vector=shift_vector)
    washed = {
        "reservoir_latent": dict(seeded),
    }
    washed = _apply_reservoir_washout(washed)
    with_washout = dict(washed["reservoir_latent"])
    for _step in range(2):
        with_washout = _preview_reservoir_step(controller, with_washout, input_vector=shift_vector)
    fresh_shift = _empty_reservoir_latent(int(controller["dim"]), len(list(controller["feedback_feature_names"])))
    for _step in range(2):
        fresh_shift = _preview_reservoir_step(controller, fresh_shift, input_vector=shift_vector)
    return {
        "no_washout_similarity": _vector_cosine(list(no_washout["combined"]), list(fresh_shift["combined"])),
        "with_washout_similarity": _vector_cosine(list(with_washout["combined"]), list(fresh_shift["combined"])),
    }


def _build_blind_human_pack(report: dict[str, object]) -> list[dict[str, object]]:
    outputs = {}
    for architecture in list(report.get("architectures", [])):
        outputs[architecture["id"]] = {
            case["id"]: case
            for case in architecture.get("cases", [])
        }
    if "lexical" not in outputs or "reservoir-fixed" not in outputs:
        return []
    pack = []
    for case in list(report.get("suite_cases", [])):
        case_id = str(case["id"])
        left = outputs["lexical"].get(case_id)
        right = outputs["reservoir-fixed"].get(case_id)
        if not left or not right:
            continue
        order_flip = _stable_unit_float(f"blind-pack:{case_id}") > 0.5
        first = right if order_flip else left
        second = left if order_flip else right
        pack.append(
            {
                "case_id": case_id,
                "prompt_sequence": list(case.get("turns", [])),
                "A": first.get("final_text", ""),
                "B": second.get("final_text", ""),
                "hidden_labels": {
                    "A": first.get("architecture"),
                    "B": second.get("architecture"),
                },
            }
        )
    return pack


def _top_anchor_from_turn_report(turn: dict[str, object]) -> str | None:
    field = dict(turn.get("field", {}) or {})
    anchors = list(field.get("top_anchors", []))
    if not anchors:
        return None
    top = dict(anchors[0])
    return str(top.get("label")) if top.get("label") else None


def _aggregate_demo_profiling(turn_reports: list[dict[str, object]]) -> dict[str, float]:
    keys = (
        "candidate_generation_seconds",
        "rewrite_seconds",
        "self_tuning_seconds",
        "total_turn_seconds",
    )
    totals = {key: 0.0 for key in keys}
    counts = {key: 0 for key in keys}
    for turn in turn_reports:
        profile = dict(turn.get("profiling", {}) or {})
        for key in keys:
            if isinstance(profile.get(key), (int, float)):
                totals[key] += float(profile[key])
                counts[key] += 1
    return {
        key: (totals[key] / counts[key] if counts[key] else 0.0)
        for key in keys
    }


def _build_recovery_demo_report(
    *,
    demo: dict[str, object],
    turn_reports: list[dict[str, object]],
) -> dict[str, object]:
    collapse_values = [
        float((turn.get("geometry") or {}).get("geometry_collapse", 0.0) or 0.0)
        for turn in turn_reports
        if turn.get("geometry") and not (turn.get("geometry") or {}).get("insufficient_history")
    ]
    peak_collapse = max(collapse_values) if collapse_values else 0.0
    final_collapse = collapse_values[-1] if collapse_values else 0.0
    peak_index = collapse_values.index(peak_collapse) + 1 if collapse_values else None
    recovery_delta = peak_collapse - final_collapse
    reasons = [
        str(((turn.get("self_tuning") or {}).get("last_reason") or ""))
        for turn in turn_reports
        if (turn.get("self_tuning") or {}).get("last_reason")
    ]
    regime_transitions = [
        str(turn.get("controller_regime_transition"))
        for turn in turn_reports
        if turn.get("controller_regime_transition")
    ]
    controller_reacted = any(
        "geometry collapse" in reason or "attractor lock" in reason
        for reason in reasons
    ) or any(
        transition.endswith("->escape")
        or transition.endswith("->rebind")
        or transition.endswith("->consolidate")
        for transition in regime_transitions
    )
    if peak_collapse < 0.22:
        verdict = "warming-up"
    elif recovery_delta >= 0.12:
        verdict = "clear-recovery"
    elif recovery_delta >= 0.05:
        verdict = "partial-recovery"
    else:
        verdict = "no-clear-recovery"
    return {
        "demo": demo["id"],
        "title": demo["title"],
        "verdict": verdict,
        "peak_collapse": peak_collapse,
        "peak_turn": peak_index,
        "final_collapse": final_collapse,
        "recovery_delta": recovery_delta,
        "controller_reacted": controller_reacted,
        "controller_regime_transitions": regime_transitions,
        "turns": turn_reports,
    }


def _build_regime_relay_demo_report(
    *,
    demo: dict[str, object],
    turn_reports: list[dict[str, object]],
) -> dict[str, object]:
    anchor_hits = 0
    anchor_checks = 0
    regime_hits = 0
    regime_checks = 0
    relapse_count = 0
    field_shift_count = 0
    previous_anchor = None
    scorecard = []
    for turn, spec in zip(turn_reports, list(demo.get("turns", []))):
        actual_anchor = _top_anchor_from_turn_report(turn)
        actual_regime = str(turn.get("controller_regime") or "")
        expect_anchor = spec.get("expect_anchor")
        expect_regimes = list(spec.get("expect_regimes", []))
        forbid_anchors = list(spec.get("forbid_anchors", []))
        anchor_ok = actual_anchor == expect_anchor if expect_anchor else None
        regime_ok = actual_regime in expect_regimes if expect_regimes else None
        if anchor_ok is not None:
            anchor_checks += 1
            anchor_hits += 1 if anchor_ok else 0
        if regime_ok is not None:
            regime_checks += 1
            regime_hits += 1 if regime_ok else 0
        if forbid_anchors and actual_anchor in forbid_anchors:
            relapse_count += 1
        if previous_anchor and actual_anchor and actual_anchor != previous_anchor:
            field_shift_count += 1
        previous_anchor = actual_anchor or previous_anchor
        scorecard.append(
            {
                "label": turn.get("label"),
                "expected_anchor": expect_anchor,
                "actual_anchor": actual_anchor,
                "anchor_ok": anchor_ok,
                "expected_regimes": expect_regimes,
                "actual_regime": actual_regime,
                "regime_ok": regime_ok,
                "forbid_anchors": forbid_anchors,
            }
        )
    profiling = _aggregate_demo_profiling(turn_reports)
    anchor_ratio = (anchor_hits / anchor_checks) if anchor_checks else 0.0
    regime_ratio = (regime_hits / regime_checks) if regime_checks else 0.0
    if anchor_ratio >= 0.80 and regime_ratio >= 0.65 and relapse_count == 0:
        verdict = "clean-relay"
    elif anchor_ratio >= 0.50 and regime_ratio >= 0.50:
        verdict = "partial-relay"
    else:
        verdict = "fragile-relay"
    return {
        "demo": demo["id"],
        "title": demo["title"],
        "verdict": verdict,
        "anchor_scorecard": f"{anchor_hits}/{anchor_checks}",
        "regime_scorecard": f"{regime_hits}/{regime_checks}",
        "anchor_ratio": anchor_ratio,
        "regime_ratio": regime_ratio,
        "field_shift_count": field_shift_count,
        "relapse_count": relapse_count,
        "avg_candidate_generation_seconds": profiling["candidate_generation_seconds"],
        "avg_rewrite_seconds": profiling["rewrite_seconds"],
        "avg_self_tuning_seconds": profiling["self_tuning_seconds"],
        "avg_total_turn_seconds": profiling["total_turn_seconds"],
        "scorecard": scorecard,
        "turns": turn_reports,
    }


def _build_demo_report(
    *,
    demo: dict[str, object],
    turn_reports: list[dict[str, object]],
) -> dict[str, object]:
    if str(demo.get("summary_kind") or demo.get("id")) == "regime-relay":
        return _build_regime_relay_demo_report(
            demo=demo,
            turn_reports=turn_reports,
        )
    return _build_recovery_demo_report(
        demo=demo,
        turn_reports=turn_reports,
    )


def _prepare_turn_regime(
    *,
    state: dict[str, object],
    prompt: str,
    field_intent: dict[str, object] | None,
    args: argparse.Namespace,
    architecture: str,
    controller: dict[str, object] | None,
) -> dict[str, object]:
    inferred_regime, inferred_reason = _infer_regulation_regime(
        state=state,
        prompt=prompt,
        field_intent=field_intent,
        mode=args.mode,
        architecture=architecture,
        controller=controller,
    )
    regime, source, reason = _apply_regime_override(
        state=state,
        inferred_regime=inferred_regime,
        inferred_reason=inferred_reason,
        mode=args.mode,
        architecture=architecture,
        controller=controller,
    )
    updated = _set_active_regime_for_turn(
        state,
        regime=regime,
        source=source,
        reason=reason,
    )
    return _apply_regime_memory_operator(
        updated,
        controller=controller,
        prompt=prompt,
        field_intent=field_intent,
        mode=args.mode,
        architecture=architecture,
    )


def _run_recovery_demo(
    *,
    args: argparse.Namespace,
    model_spec: dict[str, str],
    model,
    tokenizer,
    load_seconds: float,
    embedding_field_probe: dict[str, object] | None,
    controller: dict[str, object] | None,
) -> int:
    demo = dict(_DEFAULT_DEMOS[str(args.demo or "recovery")])
    architecture = _resolve_architecture(args.mode, args.architecture)
    self_tuning_enabled = _resolve_self_tuning(
        args.mode,
        architecture,
        getattr(args, "self_tuning", "auto"),
    )
    history: list[dict[str, str]] = []
    state = _empty_reservoir_state(
        architecture=architecture,
        controller=controller,
        self_tuning_enabled=self_tuning_enabled,
        regime_override=(args.regime if getattr(args, "regime", "auto") != "auto" else None),
    )
    reflective_thread_active = False
    turn_reports = []
    for turn_index, turn in enumerate(list(demo.get("turns", [])), start=1):
        prompt = str(turn["prompt"])
        history, state, reflective_thread_active, result, metrics = _run_eval_turn(
            args=args,
            model=model,
            tokenizer=tokenizer,
            embedding_field_probe=embedding_field_probe,
            controller=controller,
            history=history,
            reservoir_state=state,
            reflective_thread_active=reflective_thread_active,
            prompt=prompt,
        )
        turn_reports.append(
            {
                "index": turn_index,
                "label": turn["label"],
                "prompt": prompt,
                "expected_anchor": turn.get("expect_anchor"),
                "expected_regimes": list(turn.get("expect_regimes", [])),
                "forbid_anchors": list(turn.get("forbid_anchors", [])),
                "text": str(result.get("text", "")),
                "forecast": state.get("last_forecast"),
                "observer": state.get("last_observer_report"),
                "change": state.get("last_change_report"),
                "field": _embedding_field_public_view(state.get("embedding_field")),
                "geometry": state.get("reservoir_geometry"),
                "geometry_note": _format_reservoir_geometry(state.get("reservoir_geometry")),
                "geometry_detail": _format_reservoir_geometry_detail(state.get("reservoir_geometry")),
                "geometry_trajectory": _format_reservoir_geometry_trajectory(
                    state.get("reservoir_geometry_summary_history")
                ),
                "tuning_note": _format_tuning_state(state),
                "control_note": _format_control_surface(state, controller),
                "metrics": metrics,
                "profiling": metrics.get("profiling"),
                "self_tuning": dict(state.get("self_tuning", {}) or {}),
                "controller_regime": state.get("controller_regime"),
                "controller_regime_source": state.get("controller_regime_source"),
                "controller_regime_reason": state.get("controller_regime_reason"),
                "controller_regime_transition": state.get("controller_regime_transition"),
                "controller_regime_memory_operator": state.get("controller_regime_memory_operator"),
            }
        )
    report = _build_recovery_demo_report(demo=demo, turn_reports=turn_reports)
    report = _build_demo_report(demo=demo, turn_reports=turn_reports)
    if args.demo_format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"Demo: {demo['title']} [{report['verdict']}]")
    print(f"Model: {model_spec['label']} -> {model_spec['path']}")
    print(f"Load: {load_seconds:.3f}s")
    if getattr(args, "hardware_profile_note", None):
        print(str(args.hardware_profile_note))
    for turn in turn_reports:
        print(f"\nTurn {turn['index']}: {turn['label']}")
        print(f"Prompt: {turn['prompt']}")
        expectation_bits = []
        if turn.get("expected_anchor"):
            expectation_bits.append(f"anchor={turn['expected_anchor']}")
        if turn.get("expected_regimes"):
            expectation_bits.append("regime=" + "/".join(turn["expected_regimes"]))
        if expectation_bits:
            print("Expectation: " + "; ".join(expectation_bits))
        print(turn["text"] or "[no text returned]")
        notes = [
            (turn.get("forecast") or {}).get("summary"),
            (turn.get("observer") or {}).get("summary"),
            (turn.get("change") or {}).get("summary"),
            _format_controller_regime(turn),
            turn.get("geometry_detail"),
            turn.get("geometry_trajectory"),
            str((turn.get("controller_regime_memory_operator") or {}).get("summary") or ""),
            turn.get("tuning_note"),
            turn.get("control_note"),
            _format_profiling_summary(turn.get("profiling")) if _profiling_enabled(args) else None,
        ]
        notes = [note for note in notes if note]
        if notes:
            print(" ".join(notes))
    print(
        (
            f"\nSummary: peak collapse={report['peak_collapse']:.2f}"
            + (
                f" at turn {report['peak_turn']}"
                if report.get("peak_turn") is not None
                else ""
            )
            + f"; final collapse={report['final_collapse']:.2f}; recovery delta={report['recovery_delta']:.2f}; "
            f"controller reacted={'yes' if report.get('controller_reacted') else 'no'}."
        )
        if "peak_collapse" in report
        else (
            f"\nSummary: anchor score={report['anchor_scorecard']}; "
            f"regime score={report['regime_scorecard']}; field shifts={report['field_shift_count']}; "
            f"relapses={report['relapse_count']}; avg total={report['avg_total_turn_seconds']:.3f}s."
        )
    )
    return 0


def _run_eval_turn(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    embedding_field_probe: dict[str, object] | None,
    controller: dict[str, object] | None,
    history: list[dict[str, str]],
    reservoir_state: dict[str, object],
    reflective_thread_active: bool,
    prompt: str,
) -> tuple[list[dict[str, str]], dict[str, object], bool, dict[str, object], dict[str, object]]:
    turn_start = time.perf_counter()
    profiling: dict[str, object] = {
        "hardware_profile": getattr(args, "hardware_profile_resolved", "default"),
    }
    field_intent = _extract_field_intent(prompt) if args.mode == "reflective" else None
    architecture = _resolve_architecture(args.mode, args.architecture)
    stage_start = time.perf_counter()
    reservoir_state = _prepare_turn_regime(
        state=reservoir_state,
        prompt=prompt,
        field_intent=field_intent,
        args=args,
        architecture=architecture,
        controller=controller,
    )
    profiling["regime_prepare_seconds"] = time.perf_counter() - stage_start
    control_surface = _current_control_surface(reservoir_state, controller)
    reflective_shape = _determine_reflective_shape(
        mode=args.mode,
        prompt=prompt,
        reflective_thread_active=reflective_thread_active,
    )
    reflective_shape = _apply_control_surface_to_shape(
        reflective_shape,
        control_surface,
    )
    prompt_field = _probe_text_embedding_field(embedding_field_probe, text=prompt)
    stage_start = time.perf_counter()
    prepared = _prepare_turn_reservoir(
        architecture=architecture,
        controller=controller,
        state=reservoir_state,
        prompt=prompt,
        field_intent=field_intent,
        prompt_field=prompt_field,
    )
    profiling["reservoir_prepare_seconds"] = time.perf_counter() - stage_start
    history = list(history)
    history.append(
        {
            "role": "user",
            "content": _augment_user_prompt_for_mode(
                prompt=prompt,
                mode=args.mode,
                reflective_shape=reflective_shape,
            ),
        }
    )
    effective_system_prompt = _combine_system_prompt(
        args.system_prompt,
        _format_reservoir_state(reservoir_state),
    )
    effective_system_prompt = _combine_system_prompt(
        effective_system_prompt,
        prepared.get("prediction_note"),
    )
    stage_start = time.perf_counter()
    prompt_text = _build_prompt_text(
        tokenizer,
        messages=history,
        system_prompt=effective_system_prompt,
        ignore_chat_template=args.ignore_chat_template,
    )
    profiling["prompt_build_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    result = _generate_candidate_set(
        args=args,
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        evaluation_prompt=prompt,
        reservoir_state=reservoir_state,
        embedding_field_probe=embedding_field_probe,
        field_intent=field_intent,
        reservoir_prediction=(prepared.get("preview") or {}).get("predicted_field"),
        control_surface=control_surface,
        max_tokens=args.max_tokens,
        reflective_shape=reflective_shape,
    )
    profiling["candidate_generation_seconds"] = time.perf_counter() - stage_start
    profiling["candidate_profile"] = dict(result.get("candidate_profile", {}) or {})
    stage_start = time.perf_counter()
    result = _maybe_rewrite_reflective_response(
        args=args,
        model=model,
        tokenizer=tokenizer,
        messages=history,
        raw_prompt=prompt,
        reservoir_state=reservoir_state,
        embedding_field_probe=embedding_field_probe,
        field_intent=field_intent,
        reservoir_prediction=(prepared.get("preview") or {}).get("predicted_field"),
        control_surface=control_surface,
        reflective_shape=reflective_shape,
        result=result,
    )
    profiling["rewrite_seconds"] = time.perf_counter() - stage_start
    assistant_text = str(result.get("text", "")).strip()
    history.append({"role": "assistant", "content": assistant_text})
    stage_start = time.perf_counter()
    actual_field = _probe_embedding_field(
        embedding_field_probe,
        user_prompt=prompt,
        assistant_text=assistant_text,
        previous_field=reservoir_state.get("embedding_field"),
    )
    profiling["embedding_probe_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    reservoir_state = _update_reservoir_state(
        _commit_reservoir_turn(
            controller=controller,
            state=reservoir_state,
            prepared=prepared,
            actual_field=actual_field,
            result=result,
        ),
        user_prompt=prompt,
        assistant_text=assistant_text,
        embedding_field=actual_field,
        field_intent=field_intent,
    )
    profiling["reservoir_commit_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    model_advice = _collect_self_regulation_advice(
        args=args,
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        state=reservoir_state,
        prompt=prompt,
        result=result,
        prepared=prepared,
        field_intent=field_intent,
        actual_field=actual_field,
        max_tokens=args.max_tokens,
    )
    reservoir_state["self_tuning"] = _update_self_tuning_state(
        state=reservoir_state,
        controller=controller,
        result=result,
        field_intent=field_intent,
        actual_field=actual_field,
        max_tokens=args.max_tokens,
        regime=str(reservoir_state.get("controller_regime") or "sustain"),
        model_advice=model_advice,
    )
    profiling["self_tuning_seconds"] = time.perf_counter() - stage_start
    reservoir_state = _transition_regulation_regime(
        state=reservoir_state,
        prompt=prompt,
        field_intent=field_intent,
        actual_field=actual_field,
        mode=args.mode,
        architecture=architecture,
        controller=controller,
    )
    stage_start = time.perf_counter()
    reservoir_state = _attach_turn_diagnostics(
        state=reservoir_state,
        prompt=prompt,
        result=result,
        prepared=prepared,
        field_intent=field_intent,
        controller=controller,
    )
    profiling["diagnostics_seconds"] = time.perf_counter() - stage_start
    profiling["total_turn_seconds"] = time.perf_counter() - turn_start
    reflective_thread_active = reflective_thread_active or _answer_establishes_reflective_thread(assistant_text)
    field_score, _details = _score_field_alignment(actual_field, field_intent)
    latent = dict(reservoir_state.get("reservoir_latent", {}) or {})
    prediction_match = dict(latent.get("prediction_match", {}) or {})
    tuning = dict(reservoir_state.get("self_tuning", {}) or {})
    condition = dict(tuning.get("last_condition", {}) or {})
    geometry = dict(reservoir_state.get("reservoir_geometry", {}) or {})
    metrics = {
        "field_score": field_score,
        "rewrite_issue_count": len(list(result.get("rewrite_issues", []))),
        "candidate_score": float(result.get("candidate_score", 0.0) or 0.0),
        "motif_overlap": float(result.get("candidate_follow_up_overlap", 0.0) or 0.0),
        "transform_balance": float(
            result.get("candidate_follow_up_distance_balance", 0.0) or 0.0
        ),
        "prediction_match": prediction_match.get("score"),
        "actual_top_anchor": (
            list((actual_field or {}).get("top_anchors", []))[0]["label"]
            if actual_field and list((actual_field or {}).get("top_anchors", []))
            else None
        ),
        "predicted_top_anchor": (
            list((latent.get("predicted_field", {}) or {}).get("top_anchors", []))[0]["label"]
            if latent.get("predicted_field")
            and list((latent.get("predicted_field", {}) or {}).get("top_anchors", []))
            else None
        ),
        "condition_severity": condition.get("severity"),
        "stability_score": tuning.get("stability_score"),
        "geometry_collapse": geometry.get("geometry_collapse"),
        "controller_regime": reservoir_state.get("controller_regime"),
        "controller_regime_transition": reservoir_state.get("controller_regime_transition"),
        "profiling": profiling,
    }
    return history, reservoir_state, reflective_thread_active, result, metrics


def _run_esn_eval_suite(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    embedding_field_probe: dict[str, object] | None,
) -> int:
    variant_specs = [_eval_variant_spec(value) for value in _parse_eval_architectures(args.eval_architectures)]
    report = {
        "suite": args.eval_suite,
        "suite_cases": list(_DEFAULT_ESN_EVAL_SUITE),
        "architectures": [],
    }
    for spec in variant_specs:
        variant_args = _clone_args_for_variant(
            args,
            mode=spec["mode"],
            architecture=spec["architecture"],
            self_tuning=spec["self_tuning"],
        )
        controller = _build_reservoir_controller(
            spec["architecture"],
            dim=max(int(args.reservoir_dim), 8),
            seed=int(args.reservoir_seed),
        )
        architecture_cases = []
        aggregate = {
            "field_score": 0.0,
            "candidate_score": 0.0,
            "rewrite_issue_count": 0.0,
            "transform_balance": 0.0,
            "prediction_match": 0.0,
            "condition_severity": 0.0,
            "stability_score": 0.0,
            "geometry_collapse": 0.0,
            "turns": 0,
            "prediction_observations": 0,
            "tuning_observations": 0,
        }
        for case in _DEFAULT_ESN_EVAL_SUITE:
            history: list[dict[str, str]] = []
            state = _empty_reservoir_state(
                architecture=spec["architecture"],
                controller=controller,
                self_tuning_enabled=_resolve_self_tuning(
                    variant_args.mode,
                    spec["architecture"],
                    getattr(variant_args, "self_tuning", "auto"),
                ),
            )
            reflective_thread_active = False
            turn_metrics = []
            final_result = {}
            for prompt in list(case.get("turns", [])):
                history, state, reflective_thread_active, final_result, metrics = _run_eval_turn(
                    args=variant_args,
                    model=model,
                    tokenizer=tokenizer,
                    embedding_field_probe=embedding_field_probe,
                    controller=controller,
                    history=history,
                    reservoir_state=state,
                    reflective_thread_active=reflective_thread_active,
                    prompt=prompt,
                )
                turn_metrics.append(metrics)
                aggregate["field_score"] += float(metrics["field_score"])
                aggregate["candidate_score"] += float(metrics["candidate_score"])
                aggregate["rewrite_issue_count"] += float(metrics["rewrite_issue_count"])
                aggregate["transform_balance"] += float(metrics["transform_balance"])
                aggregate["turns"] += 1
                if metrics["prediction_match"] is not None:
                    aggregate["prediction_match"] += float(metrics["prediction_match"])
                    aggregate["prediction_observations"] += 1
                if metrics["condition_severity"] is not None:
                    aggregate["condition_severity"] += float(metrics["condition_severity"])
                    aggregate["stability_score"] += float(metrics["stability_score"] or 0.0)
                    aggregate["geometry_collapse"] += float(metrics["geometry_collapse"] or 0.0)
                    aggregate["tuning_observations"] += 1
            architecture_cases.append(
                {
                    "id": case["id"],
                    "architecture": spec["id"],
                    "category": case["category"],
                    "turns": turn_metrics,
                    "final_text": str(final_result.get("text", "")),
                    "state_summary": _format_reservoir_state(state),
                    "tuning_summary": _format_tuning_state(state),
                }
            )
        controller_diagnostics = {
            "convergence": _evaluate_reservoir_convergence(controller),
            "washout": _evaluate_reservoir_washout(controller),
        }
        turns = max(int(aggregate["turns"]), 1)
        report["architectures"].append(
            {
                "id": spec["id"],
                "mode": spec["mode"],
                "architecture": spec["architecture"],
                "label": spec["label"],
                "self_tuning": spec["self_tuning"],
                "aggregate": {
                    "avg_field_score": aggregate["field_score"] / turns,
                    "avg_candidate_score": aggregate["candidate_score"] / turns,
                    "avg_rewrite_issue_count": aggregate["rewrite_issue_count"] / turns,
                    "avg_transform_balance": aggregate["transform_balance"] / turns,
                    "avg_prediction_match": (
                        aggregate["prediction_match"] / aggregate["prediction_observations"]
                        if aggregate["prediction_observations"] > 0
                        else None
                    ),
                    "avg_condition_severity": (
                        aggregate["condition_severity"] / aggregate["tuning_observations"]
                        if aggregate["tuning_observations"] > 0
                        else None
                    ),
                    "avg_stability_score": (
                        aggregate["stability_score"] / aggregate["tuning_observations"]
                        if aggregate["tuning_observations"] > 0
                        else None
                    ),
                    "avg_geometry_collapse": (
                        aggregate["geometry_collapse"] / aggregate["tuning_observations"]
                        if aggregate["tuning_observations"] > 0
                        else None
                    ),
                },
                "controller_diagnostics": controller_diagnostics,
                "cases": architecture_cases,
            }
        )
    report["blind_human_pack"] = _build_blind_human_pack(report)
    if args.eval_format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"ESN Eval Suite: {args.eval_suite}")
    for architecture in list(report["architectures"]):
        aggregate = dict(architecture.get("aggregate", {}))
        print(
            f"- {architecture['label']}: field={aggregate.get('avg_field_score', 0.0):.2f} "
            f"candidate={aggregate.get('avg_candidate_score', 0.0):.2f} "
            f"issues={aggregate.get('avg_rewrite_issue_count', 0.0):.2f} "
            f"transform={aggregate.get('avg_transform_balance', 0.0):.2f} "
            f"prediction={aggregate.get('avg_prediction_match') if aggregate.get('avg_prediction_match') is not None else 'n/a'} "
            f"stability={aggregate.get('avg_stability_score') if aggregate.get('avg_stability_score') is not None else 'n/a'} "
            f"geometry={aggregate.get('avg_geometry_collapse') if aggregate.get('avg_geometry_collapse') is not None else 'n/a'}"
        )
        convergence = dict((architecture.get("controller_diagnostics", {}) or {}).get("convergence", {}) or {})
        if convergence:
            print(
                f"  convergence: start={convergence.get('start_distance', 0.0):.2f} "
                f"end={convergence.get('end_distance', 0.0):.2f} "
                f"ratio={convergence.get('contraction_ratio', 0.0):.2f}"
            )
        washout = dict((architecture.get("controller_diagnostics", {}) or {}).get("washout", {}) or {})
        if washout:
            print(
                f"  washout: no_washout={washout.get('no_washout_similarity', 0.0):.2f} "
                f"with_washout={washout.get('with_washout_similarity', 0.0):.2f}"
            )
    if report["blind_human_pack"]:
        print(f"Blind human pack ready: {len(report['blind_human_pack'])} paired cases.")
    return 0


def _run_single_prompt(
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    embedding_field_probe: dict[str, object] | None,
    controller: dict[str, object] | None,
    load_seconds: float,
    prompt: str,
) -> int:
    turn_start = time.perf_counter()
    profiling: dict[str, object] = {
        "hardware_profile": getattr(args, "hardware_profile_resolved", "default"),
    }
    architecture = _resolve_architecture(args.mode, args.architecture)
    self_tuning_enabled = _resolve_self_tuning(
        args.mode,
        architecture,
        getattr(args, "self_tuning", "auto"),
    )
    reservoir_state = _empty_reservoir_state(
        architecture=architecture,
        controller=controller,
        self_tuning_enabled=self_tuning_enabled,
        regime_override=(args.regime if getattr(args, "regime", "auto") != "auto" else None),
    )
    field_intent = _extract_field_intent(prompt) if args.mode == "reflective" else None
    stage_start = time.perf_counter()
    reservoir_state = _prepare_turn_regime(
        state=reservoir_state,
        prompt=prompt,
        field_intent=field_intent,
        args=args,
        architecture=architecture,
        controller=controller,
    )
    profiling["regime_prepare_seconds"] = time.perf_counter() - stage_start
    control_surface = _current_control_surface(reservoir_state, controller)
    reflective_shape = _determine_reflective_shape(
        mode=args.mode,
        prompt=prompt,
    )
    reflective_shape = _apply_control_surface_to_shape(
        reflective_shape,
        control_surface,
    )
    prompt_field = _probe_text_embedding_field(
        embedding_field_probe,
        text=prompt,
    )
    stage_start = time.perf_counter()
    prepared = _prepare_turn_reservoir(
        architecture=architecture,
        controller=controller,
        state=reservoir_state,
        prompt=prompt,
        field_intent=field_intent,
        prompt_field=prompt_field,
    )
    profiling["reservoir_prepare_seconds"] = time.perf_counter() - stage_start
    messages = [
        {
            "role": "user",
            "content": _augment_user_prompt_for_mode(
                prompt=prompt,
                mode=args.mode,
                reflective_shape=reflective_shape,
            ),
        }
    ]
    effective_system_prompt = _combine_system_prompt(
        args.system_prompt,
        _format_reservoir_state(reservoir_state),
    )
    effective_system_prompt = _combine_system_prompt(
        effective_system_prompt,
        prepared.get("prediction_note"),
    )
    stage_start = time.perf_counter()
    prompt_text = _build_prompt_text(
        tokenizer,
        messages=messages,
        system_prompt=effective_system_prompt,
        ignore_chat_template=args.ignore_chat_template,
    )
    profiling["prompt_build_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    result = _generate_candidate_set(
        args=args,
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        evaluation_prompt=prompt,
        reservoir_state=None,
        embedding_field_probe=embedding_field_probe,
        field_intent=field_intent,
        reservoir_prediction=(prepared.get("preview") or {}).get("predicted_field"),
        control_surface=control_surface,
        max_tokens=args.max_tokens,
        reflective_shape=reflective_shape,
    )
    profiling["candidate_generation_seconds"] = time.perf_counter() - stage_start
    profiling["candidate_profile"] = dict(result.get("candidate_profile", {}) or {})
    stage_start = time.perf_counter()
    result = _maybe_rewrite_reflective_response(
        args=args,
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        raw_prompt=prompt,
        reservoir_state=None,
        embedding_field_probe=embedding_field_probe,
        field_intent=field_intent,
        reservoir_prediction=(prepared.get("preview") or {}).get("predicted_field"),
        control_surface=control_surface,
        reflective_shape=reflective_shape,
        result=result,
    )
    profiling["rewrite_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    final_field = _probe_embedding_field(
        embedding_field_probe,
        user_prompt=prompt,
        assistant_text=str(result.get("text", "")),
        previous_field=reservoir_state.get("embedding_field"),
    )
    profiling["embedding_probe_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    committed_state = _commit_reservoir_turn(
        controller=controller,
        state=reservoir_state,
        prepared=prepared,
        actual_field=final_field,
        result=result,
    )
    final_state = _update_reservoir_state(
        committed_state,
        user_prompt=prompt,
        assistant_text=str(result.get("text", "")),
        embedding_field=final_field,
        field_intent=field_intent,
    )
    profiling["reservoir_commit_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    model_advice = _collect_self_regulation_advice(
        args=args,
        model=model,
        tokenizer=tokenizer,
        controller=controller,
        state=final_state,
        prompt=prompt,
        result=result,
        prepared=prepared,
        field_intent=field_intent,
        actual_field=final_field,
        max_tokens=args.max_tokens,
    )
    final_state["self_tuning"] = _update_self_tuning_state(
        state=final_state,
        controller=controller,
        result=result,
        field_intent=field_intent,
        actual_field=final_field,
        max_tokens=args.max_tokens,
        regime=str(final_state.get("controller_regime") or "sustain"),
        model_advice=model_advice,
    )
    profiling["self_tuning_seconds"] = time.perf_counter() - stage_start
    final_state = _transition_regulation_regime(
        state=final_state,
        prompt=prompt,
        field_intent=field_intent,
        actual_field=final_field,
        mode=args.mode,
        architecture=architecture,
        controller=controller,
    )
    stage_start = time.perf_counter()
    final_state = _attach_turn_diagnostics(
        state=final_state,
        prompt=prompt,
        result=result,
        prepared=prepared,
        field_intent=field_intent,
        controller=controller,
    )
    profiling["diagnostics_seconds"] = time.perf_counter() - stage_start
    profiling["total_turn_seconds"] = time.perf_counter() - turn_start
    if args.json:
        tuning = dict(final_state.get("self_tuning", {}) or {})
        print(
            json.dumps(
                {
                    "architecture": architecture,
                    "load_seconds": load_seconds,
                    "field_intent": field_intent,
                    "prompt_embedding_field": _embedding_field_public_view(prompt_field),
                    "reservoir_state_summary": _format_reservoir_state(final_state),
                    "reservoir_geometry": final_state.get("reservoir_geometry"),
                    "controller_regime": final_state.get("controller_regime"),
                    "controller_regime_source": final_state.get("controller_regime_source"),
                    "controller_regime_reason": final_state.get("controller_regime_reason"),
                    "controller_regime_transition": final_state.get("controller_regime_transition"),
                    "controller_regime_history": final_state.get("controller_regime_history"),
                    "controller_regime_turns": final_state.get("controller_regime_turns"),
                    "controller_previous_regime": final_state.get("controller_previous_regime"),
                    "controller_regime_memory_operator": final_state.get("controller_regime_memory_operator"),
                    "control_surface": _current_control_surface(final_state, controller),
                    "condition_vector": tuning.get("last_condition"),
                    "forecast": final_state.get("last_forecast"),
                    "observer_report": final_state.get("last_observer_report"),
                    "change_report": final_state.get("last_change_report"),
                    "self_tuning": {
                        "enabled": tuning.get("enabled"),
                        "stability_score": tuning.get("stability_score"),
                        "cooldown_remaining": tuning.get("cooldown_remaining"),
                        "baseline_observations": tuning.get("baseline_observations"),
                        "last_reason": tuning.get("last_reason"),
                        "last_relative_condition": tuning.get("last_relative_condition"),
                        "last_adjustment": tuning.get("last_adjustment"),
                        "last_model_advice": tuning.get("last_model_advice"),
                        "last_model_adjustment": tuning.get("last_model_adjustment"),
                    },
                    "reservoir_prediction_match": (
                        dict(final_state.get("reservoir_latent", {}) or {}).get("prediction_match")
                    ),
                    "profiling": profiling,
                    **result,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Load: {load_seconds:.3f}s")
        if getattr(args, "hardware_profile_note", None):
            print(str(args.hardware_profile_note))
        _print_response(result)
        if _profiling_enabled(args):
            summary = _format_profiling_summary(profiling)
            if summary:
                print(summary)
    return 0


def _run_repl(
    *,
    args: argparse.Namespace,
    model_spec: dict[str, str],
    model,
    tokenizer,
    load_seconds: float,
    embedding_field_probe: dict[str, object] | None = None,
    controller: dict[str, object] | None = None,
) -> int:
    history: list[dict[str, str]] = []
    architecture = _resolve_architecture(args.mode, args.architecture)
    self_tuning_enabled = _resolve_self_tuning(
        args.mode,
        architecture,
        getattr(args, "self_tuning", "auto"),
    )
    reservoir_state = _empty_reservoir_state(
        architecture=architecture,
        controller=controller,
        self_tuning_enabled=self_tuning_enabled,
        regime_override=(args.regime if getattr(args, "regime", "auto") != "auto" else None),
    )
    reflective_thread_active = False
    print(f"Model: {model_spec['label']} -> {model_spec['path']}")
    print(f"Load: {load_seconds:.3f}s")
    if getattr(args, "hardware_profile_note", None):
        print(str(args.hardware_profile_note))
    print(
        "Type a prompt and press Enter. Use /reset to clear history, "
        "/state to inspect memory, /field for a semantic-field readout, /geometry for trajectory readout, "
        "/predict <prompt> for a forecast, /observe for the last observer note, /what-changed for lagged reflection, "
        "/tune to inspect self-tuning, /regime to inspect or set the controller regime, "
        "/reset-tuning to clear live overrides, /washout to cool the reservoir, or /quit to exit."
    )
    while True:
        try:
            user_prompt = input("\nYou> ").strip()
        except EOFError:
            print()
            return 0
        if not user_prompt:
            continue
        if user_prompt in {"/quit", "/exit"}:
            return 0
        if user_prompt == "/reset":
            history.clear()
            reservoir_state = _empty_reservoir_state(
                architecture=architecture,
                controller=controller,
                self_tuning_enabled=self_tuning_enabled,
                regime_override=None,
            )
            reflective_thread_active = False
            print("History cleared.")
            continue
        if user_prompt == "/reset-tuning":
            reservoir_state["self_tuning"] = _empty_tuning_state(
                enabled=self_tuning_enabled,
                controller=controller,
            )
            print("Self-tuning overrides cleared.")
            continue
        if user_prompt == "/washout":
            reservoir_state = _apply_reservoir_washout(reservoir_state)
            print("Reservoir cooled; fast traces were washed out.")
            continue
        if user_prompt == "/state":
            reservoir_note = _format_reservoir_state(reservoir_state)
            embedding_note = _format_embedding_field(
                reservoir_state.get("embedding_field")
            )
            trajectory_note = _format_embedding_field_trajectory(
                reservoir_state.get("embedding_field_history")
            )
            geometry_note = _format_reservoir_geometry(
                reservoir_state.get("reservoir_geometry")
            )
            regime_note = _format_controller_regime(reservoir_state)
            intent_note = _format_field_intent(reservoir_state.get("last_field_intent"))
            alignment_note = reservoir_state.get("last_field_alignment")
            tuning_note = _format_tuning_state(reservoir_state)
            status = "on" if reflective_thread_active else "off"
            parts = []
            if reservoir_note:
                parts.append(reservoir_note)
            if embedding_note:
                parts.append(embedding_note)
            if trajectory_note:
                parts.append(trajectory_note)
            if geometry_note:
                parts.append(geometry_note)
            if regime_note:
                parts.append(regime_note)
            if intent_note:
                parts.append(intent_note)
            if alignment_note:
                parts.append(str(alignment_note))
            if tuning_note:
                parts.append(tuning_note)
            if parts:
                print(" ".join(parts) + f" Reflective thread={status}.")
            else:
                print(f"Reservoir memory is empty. Reflective thread={status}.")
            continue
        if user_prompt == "/field":
            detail_note = _format_embedding_field_detail(
                reservoir_state.get("embedding_field")
            )
            compact_note = _format_embedding_field(
                reservoir_state.get("embedding_field")
            )
            trajectory_note = _format_embedding_field_trajectory(
                reservoir_state.get("embedding_field_history")
            )
            intent_note = _format_field_intent(reservoir_state.get("last_field_intent"))
            alignment_note = reservoir_state.get("last_field_alignment")
            prediction_note = _format_predicted_field(
                dict((reservoir_state.get("reservoir_latent", {}) or {}).get("predicted_field", {}) or {})
            )
            if detail_note or compact_note or trajectory_note or intent_note or alignment_note or prediction_note:
                notes = [
                    note
                    for note in (
                        detail_note,
                        compact_note,
                        trajectory_note,
                        intent_note,
                        alignment_note,
                        prediction_note,
                    )
                    if note
                ]
                print(" ".join(notes))
            else:
                print("Embedding field is empty. Send at least one prompt first.")
            continue
        if user_prompt == "/geometry":
            detail_note = _format_reservoir_geometry_detail(
                reservoir_state.get("reservoir_geometry")
            )
            compact_note = _format_reservoir_geometry(
                reservoir_state.get("reservoir_geometry")
            )
            trajectory_note = _format_reservoir_geometry_trajectory(
                reservoir_state.get("reservoir_geometry_summary_history")
            )
            tuning = dict(reservoir_state.get("self_tuning", {}) or {})
            relative_note = _format_condition_vector(
                dict(tuning.get("last_relative_condition", {}) or {})
            )
            if relative_note:
                relative_note = relative_note.replace(
                    "Condition monitor:",
                    "Geometry pressure context:",
                )
            regime_note = _format_controller_regime(reservoir_state)
            notes = [
                note
                for note in (
                    detail_note,
                    compact_note,
                    trajectory_note,
                    regime_note,
                    relative_note,
                )
                if note
            ]
            if notes:
                print(" ".join(notes))
            else:
                print("Reservoir geometry is empty. Send at least one prompt first.")
            continue
        if user_prompt.startswith("/regime"):
            raw_value = user_prompt[len("/regime"):].strip().lower()
            if not raw_value:
                note = _format_controller_regime(reservoir_state)
                if note:
                    print(note)
                else:
                    print("Controller regime is unavailable.")
                continue
            if raw_value == "auto":
                reservoir_state["controller_regime_override"] = None
                reservoir_state["controller_regime_source"] = "auto"
                reservoir_state["controller_regime_reason"] = "manual override cleared"
                print("Controller regime override cleared; auto inference is active.")
                continue
            if raw_value not in _ACTIVE_CONTROLLER_REGIMES:
                print("Usage: /regime [auto|sustain|escape|rebind|consolidate]")
                continue
            reservoir_state["controller_regime_override"] = raw_value
            if _controller_regime_is_active(
                mode=args.mode,
                architecture=architecture,
                controller=controller,
            ):
                reservoir_state = _set_active_regime_for_turn(
                    reservoir_state,
                    regime=raw_value,
                    source="manual",
                    reason=f"manual override locked to {raw_value}",
                )
                print(f"Controller regime locked to {raw_value}.")
            else:
                print(
                    f"Manual regime override {raw_value} is inactive outside reflective reservoir mode."
                )
            continue
        if user_prompt.startswith("/predict"):
            raw_prediction_prompt = user_prompt[len("/predict"):].strip()
            if not raw_prediction_prompt:
                print("Usage: /predict <prompt>")
                continue
            field_intent = (
                _extract_field_intent(raw_prediction_prompt)
                if args.mode == "reflective"
                else None
            )
            prediction_state = _prepare_turn_regime(
                state=reservoir_state,
                prompt=raw_prediction_prompt,
                field_intent=field_intent,
                args=args,
                architecture=architecture,
                controller=controller,
            )
            prompt_field = _probe_text_embedding_field(
                embedding_field_probe,
                text=raw_prediction_prompt,
            )
            prepared = _prepare_turn_reservoir(
                architecture=architecture,
                controller=controller,
                state=prediction_state,
                prompt=raw_prediction_prompt,
                field_intent=field_intent,
                prompt_field=prompt_field,
            )
            forecast = _build_prediction_forecast(
                prompt=raw_prediction_prompt,
                prepared=prepared,
                state=prediction_state,
                field_intent=field_intent,
                controller=controller,
            )
            if forecast and forecast.get("summary"):
                print(str(forecast["summary"]))
            else:
                print("Forecast is unavailable for the current architecture.")
            continue
        if user_prompt == "/observe":
            observer = dict(reservoir_state.get("last_observer_report", {}) or {})
            if observer and observer.get("summary"):
                print(str(observer["summary"]))
            else:
                print("Observer has no turn to inspect yet. Send a prompt first.")
            continue
        if user_prompt == "/what-changed":
            change = dict(reservoir_state.get("last_change_report", {}) or {})
            if change and change.get("summary"):
                print(str(change["summary"]))
            else:
                print("No lagged change report yet. Send at least two prompts first.")
            continue
        if user_prompt == "/tune":
            tuning = dict(reservoir_state.get("self_tuning", {}) or {})
            condition_note = _format_condition_vector(
                dict(tuning.get("last_condition", {}) or {})
            )
            relative_note = _format_condition_vector(
                dict(tuning.get("last_relative_condition", {}) or {})
            )
            if relative_note:
                relative_note = relative_note.replace(
                    "Condition monitor:",
                    "Relative baseline:",
                )
            tuning_note = _format_tuning_state(reservoir_state)
            control_note = _format_control_surface(reservoir_state, controller)
            geometry_note = _format_reservoir_geometry(
                reservoir_state.get("reservoir_geometry")
            )
            regime_note = _format_controller_regime(reservoir_state)
            notes = [
                note
                for note in (
                    condition_note,
                    relative_note,
                    geometry_note,
                    regime_note,
                    tuning_note,
                    control_note,
                )
                if note
            ]
            if notes:
                print(" ".join(notes))
            else:
                print("Self-tuning has no telemetry yet. Send a prompt first.")
            continue

        history, reservoir_state, reflective_thread_active, result, metrics = _run_eval_turn(
            args=args,
            model=model,
            tokenizer=tokenizer,
            embedding_field_probe=embedding_field_probe,
            controller=controller,
            history=history,
            reservoir_state=reservoir_state,
            reflective_thread_active=reflective_thread_active,
            prompt=user_prompt,
        )
        print("Assistant>")
        _print_response(result)
        if _profiling_enabled(args):
            summary = _format_profiling_summary(metrics.get("profiling"))
            if summary:
                print(summary)


def main() -> int:
    args = _parse_args()
    args = _apply_hardware_profile(args)
    repo_root = _repo_root()

    if args.list_models:
        specs = _default_model_specs(repo_root)
        if not specs:
            print("No default local models detected.")
            return 0
        print("Detected models:")
        for spec in specs:
            print(f"- {spec['label']}: {spec['path']}")
        return 0

    _bootstrap_fast_python(args, repo_root)
    model_spec = _resolve_model_spec(
        repo_root=repo_root,
        model=args.model,
        model_label=args.model_label,
    )
    model_path = Path(model_spec["path"]).expanduser().resolve()
    model_dir = str(model_path if model_path.is_dir() else model_path.parent)
    model, tokenizer, load_seconds = _load_runtime(args, model_dir)
    embedding_field_probe = _build_embedding_field_probe(model, tokenizer)
    args.system_prompt = _resolve_system_prompt(args)
    controller = _build_reservoir_controller(
        _resolve_architecture(args.mode, args.architecture),
        dim=max(int(args.reservoir_dim), 8),
        seed=int(args.reservoir_seed),
    )

    if args.eval_suite is not None:
        return _run_esn_eval_suite(
            args=args,
            model=model,
            tokenizer=tokenizer,
            embedding_field_probe=embedding_field_probe,
        )

    if args.demo is not None:
        return _run_recovery_demo(
            args=args,
            model_spec=model_spec,
            model=model,
            tokenizer=tokenizer,
            load_seconds=load_seconds,
            embedding_field_probe=embedding_field_probe,
            controller=controller,
        )

    prompt = _read_prompt_from_sources(args)
    if prompt is not None:
        return _run_single_prompt(
            args=args,
            model=model,
            tokenizer=tokenizer,
            embedding_field_probe=embedding_field_probe,
            controller=controller,
            load_seconds=load_seconds,
            prompt=prompt,
        )

    return _run_repl(
        args=args,
        model_spec=model_spec,
        model=model,
        tokenizer=tokenizer,
        load_seconds=load_seconds,
        embedding_field_probe=embedding_field_probe,
        controller=controller,
    )


if __name__ == "__main__":
    raise SystemExit(main())
