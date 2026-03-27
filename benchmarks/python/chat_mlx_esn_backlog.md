# Chat MLX ESN Backlog

This document tracks the reflective chat and controller work in
[`chat_mlx_local.py`](/Users/mikepurvis/other/mlx/benchmarks/python/chat_mlx_local.py).
The goal is to move from a prompt-shaped reflective assistant toward a
measurable, recurrent, self-regulating system whose state is doing real work.

## Working Rules

- Keep each tranche shippable on its own.
- Prefer instrumentation before adding more expressive behavior.
- Treat controller quality and prose quality as separate axes.
- Do not call something "ESN-like" unless the recurrent state is both measured
  and behaviorally consequential.
- Keep the visible interface mostly fixed while swapping internals.

## Current State

What is already in place:

- Reflective and helpful chat modes.
- Candidate generation, reranking, and rewrite passes.
- Lexical motif carry and follow-up novelty pressure.
- Input-embedding field probe with `/field` and `/state`.
- Reservoir architectures:
  - `none`
  - `lexical`
  - `reservoir-fixed`
  - `reservoir-trainable-readout`
- Multi-timescale reservoir state with leaky recurrence.
- Reservoir-side prediction of field and behavior pulls.
- Bounded self-tuning with cooldown and live control-surface overrides.
- Relative-baseline monitoring and attractor-recovery exploration pulses.
- Geometry-aware trajectory monitoring with collapse detection.
- Eval suite and blind-comparison pack scaffolding.

Current honest limitation:

- The controller is ahead of the prose. The system is increasingly good at
  noticing and regulating state, but the generated language is still uneven on
  hard reflective prompts.

## Proof Bar

We should treat the work as genuinely novel only if we can show all three:

1. The recurrent state predicts or explains behavior better than transcript-only
   features.
2. The controller can steer, recover, and forget in measurable ways.
3. The user-facing shell stays mostly fixed while the recurrent controller
   materially improves continuity, transformation, or regulation.

## Tranche 0: Reflective Shell

- [x] Add reflective/helpful modes.
- [x] Add reflective issue collection and rewrite passes.
- [x] Add candidate generation and reranking.
- [x] Add motif carry and follow-up transformation pressure.
- [x] Add simple REPL commands for state inspection.

Exit criteria:
- The assistant can stay in a reflective mode across turns and self-correct
  obvious canned disclaimers or dropped structure.

## Tranche 1: Real Reservoir Controller

- [x] Add explicit multi-timescale reservoir state.
- [x] Add fixed random recurrent/input/feedback weights.
- [x] Add field and behavior readout heads.
- [x] Use reservoir state to shape candidate scoring.
- [x] Add washout and convergence diagnostics.

Exit criteria:
- The chat loop is not just carrying motifs lexically; it has a real external
  recurrent state with readouts and measurable dynamics.

## Tranche 2: Bounded Self-Tuning

- [x] Add a condition monitor from observable failure signals.
- [x] Add a bounded self-tuning policy with clipped deltas.
- [x] Add cooldown and decay back toward baseline.
- [x] Surface tuning state in JSON and REPL.
- [x] Add `/tune` and `/reset-tuning`.

Exit criteria:
- The controller can adjust a small interpretable control surface without
  becoming unstable or opaque.

## Tranche 3: Relative Baselines, Hysteresis, and Recovery

- [x] Measure pressure relative to a running baseline instead of raw values
  alone.
- [x] Add pressure streaks and calm streaks.
- [x] Refine cooldown so recovery can be gradual instead of twitchy.
- [x] Add bounded exploration noise for sticky attractors.
- [x] Add geometry-aware collapse pressure as a controller input.

Exit criteria:
- The system can recover from repetitive or sticky regimes by loosening and
  reorienting, not only by forcing stricter output structure.

## Tranche 4: Recovery Demo and Readability

- [x] Add a dedicated multi-turn recovery demo that induces attractor lock and
  shows recovery.
- [x] Add a `/geometry` REPL command with a compact explanation of the current
  reservoir trajectory regime.
- [x] Add a short trajectory line for geometry, similar to field trajectory.
- [x] Add a clearer human-readable summary of controller intent vs outcome.

Exit criteria:
- A collaborator can watch the system get stuck, react, and recover without
  reading raw JSON.

Current note:
- The instrumentation and demo path are now present, but the first live smoke
  still landed in `no-clear-recovery`. That is useful signal, not failure
  theater: the system can now show the controller reacting even when the prose
  layer and recovery quality are still lagging.

## Tranche 5: Prose Quality Under Control

- [ ] Make rewrite quality controller-aware instead of structure-aware only.
- [ ] Separate "controller succeeded" from "response reads well".
- [ ] Add a style-quality critic that respects current controller targets.
- [ ] Reduce truncation and repeated-label failure modes in reflective rewrites.
- [ ] Add a small eval slice focused only on prose quality under stable control.

Exit criteria:
- The system can be both well-regulated and pleasant to read on the same turn
  often enough to matter.

## Tranche 6: More Faithful Spectral Self-Measurement

- [ ] Add an EWMA covariance over recent reservoir states.
- [ ] Add a lightweight top-eigenvalue or anisotropy estimate.
- [ ] Distinguish low-rank collapse from merely calm stable dynamics.
- [ ] Feed spectral pressure into self-tuning alongside current geometry
  signals.
- [ ] Surface the new measurement in `/geometry`, `/tune`, and JSON.

Exit criteria:
- Geometry collapse is no longer a heuristic blend only; it has an explicit
  spectral component closer to the `mikeconsciouness` notion of self-reference.

## Tranche 7: Richer Control Surface

- [ ] Let self-tuning change candidate count temporarily.
- [ ] Add better exploration pulse decay.
- [ ] Add bounded automatic washout after sustained attractor lock.
- [ ] Split baselines by conversational regime if needed.
- [ ] Add explicit controller prediction before generation and compare it to
  actual movement after generation.

Exit criteria:
- The controller can not only notice regime problems, but choose a more
  appropriate search or recovery mode for a few turns.

## Tranche 8: Stronger Evaluation

- [ ] Extend the eval suite with longer multi-turn cases.
- [ ] Add explicit attractor-lock induction cases.
- [ ] Add forced regime-switch and washout cases.
- [ ] Score controller quality separately from prose quality.
- [ ] Run paired human judgments from the blind pack.

Exit criteria:
- We can defend the claim that the recurrent controller matters, rather than
  relying on vivid examples alone.

## Tranche 9: Multi-Voice and Higher-Level Readouts

- [ ] Explore multiple readout styles drawing from the same reservoir.
- [ ] Try warm/metaphoric, formal/analytic, and geometry-aware readouts sharing
  one evolving state.
- [ ] Measure whether shared-state multi-voice output improves continuity or
  self-reference.
- [ ] Keep this behind an experimental flag until the proof bar is clearer.

Exit criteria:
- We have evidence that multiple "voices" can read from the same dynamical
  state in a meaningful way instead of merely rephrasing each other.

## Near-Term Order

1. Build the recovery demo and `/geometry`.
2. Improve prose quality under the existing controller.
3. Add the first spectral self-measurement pass.
4. Expand the eval suite with attractor-lock and recovery cases.
5. Run real paired human judgments on controller variants.

## Commands We Should Keep Using

Live chat:

```bash
python /Users/mikepurvis/other/mlx/benchmarks/python/chat_mlx_local.py
```

Inspect a single turn:

```bash
python /Users/mikepurvis/other/mlx/benchmarks/python/chat_mlx_local.py --json --prompt "Lean toward proof and arithmetic structure, keep warmth, and avoid sea imagery."
```

Compare controller variants:

```bash
python /Users/mikepurvis/other/mlx/benchmarks/python/chat_mlx_local.py --eval-suite esn-core --eval-architectures lexical,reservoir-fixed+tuned --eval-format text
```

## Notes To Future Us

- If a tranche improves telemetry but not prose, that is still progress.
- If a tranche improves prose but hides controller behavior, that is not enough.
- The base model is still the language engine. The novelty claim lives in the
  recurrent control layer, its measurements, and its ability to regulate itself
  in bounded ways.
