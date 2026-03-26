# Load mmap Roadmap

This document tracks the benchmark and loader work around mmap-based model
loading. The goal is to turn `load_mmap_bench.py` from a one-off benchmark into
an experiment system that can suggest, test, and learn which efficiency routes
are worth pursuing.

## Working Rules

- Keep each tranche shippable on its own.
- Prefer instrumentation before policy changes.
- Treat "wrong but informative" experiments as wins.
- Do not mark a tranche complete until the new signal is visible in the bench.

## Tranche 0: Foundation

- [x] Add route suggestions from benchmark summary metrics.
- [x] Add route confidence and confidence rationale.
- [x] Add bucketed trend comparisons with a printed "why this bucket" summary.
- [x] Generate runnable next-experiment commands.
- [x] Record environment metadata in JSONL.
- [x] Make the CLI usable without importing `mlx.core` for `--help`.

Exit criteria:
- The bench can explain what it thinks is happening and what to run next.

## Tranche 1: Coverage Attribution

- [x] Track fallback cost by bytes, not just tensor count.
- [x] Surface per-reason fallback bytes in Python benchmark output.
- [x] Thread byte-weighted fallback information into route confidence.
- [x] Add targeted tests or fixtures for parsing mmap debug stats.
- [x] Distinguish source-byte cost vs destination-byte cost where conversion changes size.
- [x] Decide whether count-first or byte-first ranking should drive the default suggestion order.

Exit criteria:
- We can answer "which fallback reason is most expensive" without manual forensics.
- The bench explains that fallback ordering is `bytes-first` by default, with count as a tie-breaker.

## Tranche 2: Experiment Control Surface

- [x] Add internal toggles for small-tensor copy thresholds.
- [x] Add internal toggles for hotset promotion / rematerialization thresholds.
- [x] Add internal toggles for prefetch strategies or staged first-touch.
- [x] Teach the benchmark to sweep those toggles automatically.
- [x] Record the active policy knobs in JSONL.
- [x] Add a compact "policy matrix" preset for warm/cold dense-vs-quantized runs.

Exit criteria:
- The benchmark can test real policy variants, not just observe the default runtime.

## Tranche 3: Structured Diagnostics

- [x] Expose mmap load stats programmatically instead of relying only on stderr parsing.
- [x] Teach the benchmark to prefer the structured stats path and keep stderr as fallback.
- [x] Add a tiny runtime/API test for structured mmap stats once the Python extension is rebuilt in CI.
- [x] Record allocator or device memory alongside RSS so "wins" can be checked for cost shifting.

Exit criteria:
- Benchmarks, tests, and applications can read the same mmap diagnostics without scraping logs.

## Tranche 4: Phase Timing

- [x] Split load timing into header parse, mapping/view creation, first eval, and decode phases.
- [x] Record first-token vs steady-state decode separately.
- [x] Attribute page-fault spikes to the phase where they happen.
- [x] Add phase-specific regressions to trend checks.
- [x] Print phase deltas in the experiment summary.

Exit criteria:
- A regression can be localized to startup, first-touch, or steady state.

## Tranche 5: Correctness and Golden Coverage

- [x] Add mapped-vs-copy parity fixtures for dense safetensors.
- [x] Add mapped-vs-copy parity fixtures for quantized GGUF.
- [x] Include at least one alignment-hostile checkpoint fixture or synthetic case.
- [x] Curate a stable dense safetensors set.
- [x] Curate a stable quantized GGUF set.
- [x] Add a standard "golden matrix" command that covers the core cases.

Exit criteria:
- Hybrid policies can be validated for correctness and are no longer benchmarked only on convenience files.

## Tranche 6: Smarter Policy Selection

- [x] Make hotset promotion name-aware instead of bytes-only.
- [x] Add an auto policy mode that can choose between copy, mapped, and hybrid knobs.
- [x] Teach auto policy to use format, cache mode, mapped ratio, and fallback mix.
- [x] Surface the chosen auto policy in diagnostics and history.
- [x] Revisit whether hotset promotion should be decode-mode aware.

Exit criteria:
- The runtime can make a decent first policy choice without manual knob sweeping.

## Tranche 7: Learning Loop and Reporting

- [x] Record which route was actually attempted after a suggestion.
- [x] Record the code state or commit that carried the change.
- [x] Backtest route confidence against historical outcomes.
- [x] Track route hit-rate by bucket.
- [x] Build a small report over `load_mmap_history.jsonl`.
- [x] Show route hit-rate, fallback mix, and top regressions per bucket.
- [x] Add compare-to-baseline-commit mode.
- [x] Add export-friendly JSON/CSV summaries for external analysis.

Exit criteria:
- Confidence becomes calibrated by outcomes rather than hand-tuned heuristics, and recent history is easy to scan.

## Tranche 8: Docs and User-Facing Examples

- [x] Document the mmap policy knobs on `mx.load`.
- [x] Document `mx.last_mmap_load_stats()` with a short example.
- [x] Add one or two benchmark recipes for dense-vs-quantized warm/cold comparisons.
- [x] Decide whether part of the benchmark guidance should graduate into docs.

Guide:
- `load_mmap_policy_guide.md` is now the practical entry point for policy knobs, diagnostics, decode-aware auto behavior, and evidence-first demo/report commands.
- `load_mmap_branch_checklist.md` records the current pre-branch build, smoke, test, demo, and schema gates, including the CPU-only fallback workflow when the Metal toolchain is unavailable.

Exit criteria:
- Someone new to this work can use the knobs and diagnostics without reading the benchmark source.

## Tranche 9: External Corpus Intake and Claim Replay

- [x] Add a lightweight ingest scaffold for prose and code.
- [x] Extend the scaffold to extract candidate hypotheses from Rust and Lean.
- [x] Curate an external-corpus hypothesis ledger for `prime-physics-engine`.
- [x] Encode at least 3 prime-physics claim-scope stories as deterministic replay/forensics fixtures.
- [x] Surface those cases inside the existing replay/forensics demo presets.
- [x] Add a persistence-focused demo that shows session memory suppressing already-refuted stories over time.

Exit criteria:
- External prose/code corpora can be turned into reviewable claim fixtures, and the demo loop can prove it remembers scope and negative evidence.

## Near-Term Order

1. Consider whether claim-ingest and persistence demos should graduate into a more generic research-demo recipe.
2. Decide whether the new guide should be linked from broader MLX docs once the surface settles further.
3. Expand the guide with a discovered-model intake recipe after a few more live runs on real corpora.
