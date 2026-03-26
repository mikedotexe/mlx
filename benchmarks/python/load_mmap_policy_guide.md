# Load mmap Policy Guide

This guide is the practical entry point for the mmap loading work.

Use it when you want to:

- tune `mx.load(...)` without reading the loader internals,
- understand what `--policy-mode auto` is doing,
- run evidence-first demos and reports instead of arguing from intuition.

## What The System Does

There are three layers now:

- Runtime policy surface: `mx.load(...)` exposes mapped, copy, and hybrid loading knobs.
- Diagnostics surface: `mx.last_mmap_load_stats()` and `mx.last_load_phase_stats()` explain what happened.
- Research loop: `load_mmap_bench.py` can sweep policies, choose an auto policy, record outcomes, and replay what it has learned.

The benchmark story is not "mmap always wins." The benchmark story is "the right policy depends on format, cache temperature, decode behavior, and the local history bucket."

## `mx.load(...)` Policy Knobs

All mmap policy knobs are no-ops unless `memory_map=True`.

```python
import mlx.core as mx

weights = mx.load(
    "/absolute/path/to/model.safetensors",
    memory_map=True,
    mmap_small_tensor_copy_max_bytes=65536,
    mmap_hotset_promotion_top_k=2,
    mmap_hotset_promotion_min_bytes=32 * 1024 * 1024,
    mmap_prefetch_strategy="sequential",
)
```

### `memory_map`

- `False`: copy-only load path.
- `True`: opt into mapped loading for path-based `.safetensors` and `.gguf` files.

Backend note:

- On builds without a backend that supports external no-copy buffer wrapping, the mmap loader path can still run but may fall back to copies with `make_buffer_failed` instead of producing direct mapped views.
- In practice, that means a CPU-only local build is still useful for loader, diagnostics, and benchmark-orchestration validation, but it is not enough to claim that mapped views are materially working end to end.

### `mmap_small_tensor_copy_max_bytes`

- Copies mapped-eligible tensors at or below this size instead of creating mapped views.
- Good for testing whether small tensors create more overhead than benefit.
- Typical experimental value: `65536`.

### `mmap_hotset_promotion_top_k`

- Copies a small number of mapped-eligible tensors into owned buffers instead of leaving them mapped.
- Intended for decode-sensitive hybrid policies.
- The loader now uses name-aware hotset ranking, so embeddings, final norms, and output heads are favored before raw size ties.

### `mmap_hotset_promotion_min_bytes`

- Byte floor for hotset promotion eligibility.
- Use it when you want promotion to stay focused on genuinely large tensors instead of scattering across medium-sized weights.
- Typical experimental value: `32 * 1024 * 1024`.

### `mmap_prefetch_strategy`

- `sequential`: the default mapped hint.
- `willneed`: a stronger cold-start hint for cases where first-touch faults dominate.
- `none`: disable explicit prefetch hinting.

## Programmatic Diagnostics

The easiest way to inspect one load is:

```python
import json
import mlx.core as mx

mx.load(
    "/absolute/path/to/model.gguf",
    memory_map=True,
    mmap_hotset_promotion_top_k=1,
    mmap_prefetch_strategy="willneed",
)

print(json.dumps(mx.last_mmap_load_stats(), indent=2, sort_keys=True))
print(json.dumps(mx.last_load_phase_stats(), indent=2, sort_keys=True))
```

### `mx.last_mmap_load_stats()`

Use this to answer:

- how many bytes were mapped vs copied,
- which fallback reasons dominated,
- which tensors were hotset-promoted,
- which promotion strategy was active.

Key fields:

- `mapped_bytes`
- `copied_bytes`
- `fallback_tensors`
- `fallback_reasons`
- `fallback_reason_bytes`
- `fallback_reason_source_bytes`
- `hotset_promoted_tensors`
- `hotset_promotion_strategy`

### `mx.last_load_phase_stats()`

Use this when you need to localize a regression instead of only proving that one exists.

Key fields:

- `open_map_seconds`
- `parse_seconds`
- `tensor_setup_seconds`
- `open_map_minor_faults`
- `open_map_major_faults`
- `parse_minor_faults`
- `parse_major_faults`
- `tensor_setup_minor_faults`
- `tensor_setup_major_faults`

If you prefer stderr logging during ad hoc experiments, set `MLX_DEBUG_IO_MEMORY_MAP=1`, but the structured APIs are the better long-term path.

## Benchmark Policy Modes

The benchmark compares copy mode against an effective non-copy policy.

### `--policy-mode manual`

- Use the exact mmap knobs you requested.
- Best for controlled experiments and sweeps.

### `--policy-mode copy`

- Force copy-only on the non-copy side as a sanity check or baseline override.

### `--policy-mode mapped`

- Force mapped default with no hybrid knobs.
- Useful when you want to ask "does direct mapping win here at all?"

### `--policy-mode hybrid`

- Force a non-default hybrid policy.
- Use this when you already know which knobs you want to test.

### `--policy-mode auto`

- Choose the effective policy from comparable history plus a mapped preflight probe.
- This is the right mode when the question is "what should the system do by default here?"

## Decode-Aware Auto Policy

The auto chooser is now decode-aware. In plain language, it no longer treats a fast loader as a good policy if the decode path looks fragile.

It currently uses four evidence sources:

1. Comparable history bucket.
   The chooser looks at prior runs with the same broad context: format, model class, cache mode, decode mode, and effective policy family.
2. Same-file support.
   If the exact file has been seen before, that evidence gets extra weight.
3. Mapped preflight probe.
   The benchmark runs one lightweight mapped probe to measure mapped ratio, fallback tensors, copied bytes, and fallback-byte mix.
4. Tiny synth decode probe.
   This happens when the benchmark is already running with `--decode-synth`. The auto chooser uses a small preflight decode to estimate first-token cost and first-touch fault pressure before committing to a policy.

### What It Penalizes

- low mapped coverage,
- high quantized-conversion fallback share,
- high first-token regression in comparable history,
- high first-token faults in the preflight probe,
- mapped default in warm decode contexts when there is no direct decode probe yet.

### What It Favors

- dense safetensors with high mapped coverage,
- cold-start mapped or `willneed` variants when startup faults dominate,
- hybrid hotset policies when mapped coverage is already high but first-token cost is clearly the problem,
- conservative copy policies when quantized fallback bytes or hostile alignment make aggressive mapping hard to justify.

### Important Scope Note

The auto chooser only gets a live decode preflight today when you use `--decode-synth`. If you use `--decode-cmd`, the benchmark still measures decode during the main run, but the auto preflight itself stays loader-first plus history-driven.

## Backend Reality Check

Two different claims are easy to blur together, so keep them separate:

- `The benchmark loop works`: the benchmark runs, records policy decisions, writes history, replays suggestions, and prints diagnostics.
- `Mapped views are truly active`: the runtime is actually creating direct mapped views instead of reporting fallback reasons like `make_buffer_failed`.

If your local build was produced with `MLX_BUILD_METAL=OFF`, you can still validate the first claim. You should not use that build alone to claim the second one.

## Evidence-First Recipes

Set a couple of shell helpers first:

```bash
export BENCH=/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py
export HIST=/tmp/load_mmap_history_$(date +%Y%m%d_%H%M%S).jsonl
export PYTHONPATH=/Users/mikepurvis/other/mlx/python${PYTHONPATH:+:$PYTHONPATH}
```

If you are running from an installed wheel instead of the source tree, you do not need the `PYTHONPATH` export.

### 1. Ask Auto Policy What It Would Do For One Model

```bash
python "$BENCH" \
  /absolute/path/to/model.safetensors \
  --policy-mode auto \
  --cache-mode warm \
  --decode-synth \
  --history-json "$HIST"
```

Use this when you want the benchmark to pick a default instead of hand-tuning one.

Look for:

- `effective policy mode=... signature=...`,
- the `auto:` summary line,
- the `candidate ...` lines,
- `auto probe decode` when `--decode-synth` is enabled.

### 2. Sweep One Hybrid Idea Instead Of Arguing About It

```bash
python "$BENCH" \
  /absolute/path/to/model.gguf \
  --cache-mode warm \
  --decode-synth \
  --sweep-mmap-small-tensor-copy-max-bytes none,4096,65536 \
  --sweep-mmap-hotset-promotion-top-k none,1,2 \
  --sweep-mmap-prefetch-strategy sequential,willneed
```

Use this when the question is "is there any signal here?" rather than "what should the default be?"

### 3. Run The Dense-vs-Quantized Warm/Cold Matrix

```bash
python "$BENCH" \
  /absolute/path/to/dense_model.safetensors \
  /absolute/path/to/quantized_model.gguf \
  --policy-matrix compact \
  --decode-synth \
  --history-json "$HIST"
```

Use this when you want a compact performance matrix instead of one-file anecdotes.

### 4. Wrap Correctness Before Trusting The Story

```bash
python "$BENCH" \
  --golden-matrix compact \
  --history-json "$HIST"
```

Use this when you need the dense parity, quantized parity, and alignment-hostile checks without the broader demo wrapper.

If you want the same guardrail in demo form, run:

```bash
python "$BENCH" \
  --demo-preset golden-guardrail \
  --history-json "$HIST"
```

### 5. Replay The Learning Loop Deterministically

```bash
python "$BENCH" --demo-preset history-replay
```

Use this when you want proof that the reporting and route-learning loop works without depending on the current machine state.

### 6. Replay Curated Regression Diagnosis

```bash
python "$BENCH" --demo-preset regression-forensics
```

Use this when you want examples of the loop noticing parse regressions, first-token regressions, and mixed outcomes.

### 7. Audit Auto Policy Against Seeded Alternatives

```bash
python "$BENCH" \
  --demo-preset adaptation-ladder \
  --history-json "$HIST" \
  /absolute/path/to/dense_model.safetensors \
  /absolute/path/to/quantized_model.gguf \
  --decode-synth
```

Use this when you want evidence that auto policy is choosing well, not just choosing confidently.

If this is running on a CPU-only local build, expect the demo to be informative before it is flattering. The adaptation ladder may fail honestly because the dense safetensors buckets cannot create direct mapped views and therefore collapse toward `make_buffer_failed` coverage.

The adaptation ladder reports:

- chosen policy,
- winner policy,
- runner-up policy,
- `match`, `near-match`, or `miss`,
- bucket match rate,
- candidate reasons and evidence.

### 8. Report What The System Has Learned

```bash
python "$BENCH" \
  --history-json "$HIST" \
  --report-history \
  --report-format text \
  --report-top 5
```

Use this when you want bucket winners, route hit-rate, fallback bytes, and phase regressions instead of raw JSONL.

### 9. Test A Specific Route And Record The Outcome

```bash
python "$BENCH" \
  /absolute/path/to/model.gguf \
  --policy-mode hybrid \
  --mmap-hotset-promotion-top-k 1 \
  --mmap-prefetch-strategy willneed \
  --attempted-route hotset_promotion \
  --route-outcome auto \
  --baseline-git-head abc1234 \
  --decode-synth \
  --history-json "$HIST"
```

Use this when you want the run to become part of the learning loop rather than a one-off experiment.

If you omit `--baseline-run-id` and `--baseline-git-head`, the benchmark falls back to the prior comparable history bucket.

### 10. Prove Scoped Memory Beats Immortal Memory

```bash
python "$BENCH" --demo-preset persistence-memory
```

Use this when you want to show that refuted stories can stay suppressed in session or scoped persistent memory instead of reappearing because of unrelated wins elsewhere.

### 11. Read The Machine-Usable History Fields

Treat these JSONL keys as the branch-stable evidence surface for now:

- `run_id`
- `attempted_route`
- `route_outcome`
- `effective_policy_mode`
- `effective_policy_signature`
- `policy_signature`
- `policy_summary`
- `history_bucket`
- `comparison`
- `auto_policy_decision`
- `auto_policy_probe_decode`
- `mmap_coverage`
- `phase_timing`
- `decode_phase_timing`
- `route_suggestions`

## Which Command To Run When You Want Evidence

| If the question is... | Run this |
| --- | --- |
| "What would the system choose by default here?" | `--policy-mode auto --decode-synth` |
| "Is there any signal in this knob family?" | policy sweeps |
| "How do dense and quantized buckets differ?" | `--policy-matrix compact` |
| "What has this history file actually learned?" | `--report-history` |
| "Can I prove the loop works without live model noise?" | `--demo-preset history-replay` |
| "Can it localize different regression shapes?" | `--demo-preset regression-forensics` |
| "Is auto policy actually choosing well?" | `--demo-preset adaptation-ladder` |
| "Are the claims still correct?" | `--golden-matrix compact` or `--demo-preset golden-guardrail` |
| "Does scoped memory suppress refuted stories?" | `--demo-preset persistence-memory` |

## Related Demo Scripts

For longer narrated walkthroughs, see:

- `load_mmap_demo_show_script.md`
- `load_mmap_demo_intake_show_script.md`

## Practical Heuristic

When in doubt:

1. start with `--policy-mode auto --decode-synth`,
2. inspect `--report-history`,
3. use `adaptation-ladder` before claiming the chooser is smart,
4. use `golden-guardrail` before claiming the chooser is trustworthy.
