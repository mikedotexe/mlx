# Prime Physics Engine: Hypothesis Intake Tranche

This tranche turns material from
[/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine)
into a demo-ready hypothesis ledger.

The goal is not to adjudicate the mathematics here. The goal is to show how an
external prose-and-code corpus can be converted into:

- falsifiable hypotheses,
- negative-result checks,
- persistence experiments,
- and replay/forensics stories.

The first encoded claim-scope replay fixture now lives at:

- [load_mmap_prime_physics_demo_history.jsonl](/Users/mikepurvis/other/mlx/benchmarks/python/testdata/load_mmap_prime_physics_demo_history.jsonl)

It currently covers:

- negative-result persistence
- single-instance scope discipline
- formalization-without-overclaiming guardrails

The corresponding scoped-memory demo is now runnable via:

```bash
python /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py \
  --demo-preset persistence-memory
```

## Curated Top Set From The Live Run

The live scaffold run over the real repo produced `212` candidate fragments
across `8` sources. These are the `8` claims I would keep at the top of the
ledger because they are the cleanest fit for our claim-memory and replay system.

### 1. Negative Results Must Persist

- Claim: the system should remember that `k* ~ sqrt(M)`, `2p resonance`,
  template-specific bonuses, boundary-digit specialness, and phase-lock stories
  were explicitly tested and not supported.
- Best sources: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md)
- Why it matters to us: this is the cleanest memory-suppression test in the
  whole corpus.

### 2. The Density Story Should Stay Classical Unless Controls Separate

- Claim: the membrane-vs-random-coprime ratio staying near `1` should push the
  system toward a classical or control-dominated interpretation rather than a
  novelty-heavy mechanism story.
- Best sources: [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md), [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md)
- Why it matters to us: it tests whether the loop can prefer "boring but right"
  over "exciting but unsupported."

### 3. Single-Instance Findings Must Stay Narrow

- Claim: canonical asymmetries or memorable examples should remain explicitly
  single-instance until the matrix broadens.
- Best sources: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md), [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md)
- Why it matters to us: this is a direct scope-discipline test.

### 4. Statistical Probes Should Stay Statistical

- Claim: prime-outer-digit, discriminant, and Goldbach-richness stories should
  remain contingent on significance, correlation, and effect size rather than
  on memorable anecdotes.
- Best sources: [prime_outer_analysis.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/prime_outer_analysis.py), [analyze_discriminant.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py)
- Why it matters to us: these are ideal for forensics-style "suggestive vs
  significant" replay cases.

### 5. Formalization Increases Trust, Not Claim Size

- Claim: Lean proof volume should strengthen local theorem families without
  inflating the global density interpretation.
- Best sources: [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md), [STATUS.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/STATUS.md)
- Why it matters to us: this is the cleanest "local proof vs global narrative"
  separation case.

### 6. Tidal And Gravity Language Must Stay In The Metaphor Bucket

- Claim: the tidal-analysis layer is explicitly a visualization surface and
  should not be promoted into mathematical evidence.
- Best sources: [src/tidal/mod.rs](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/src/tidal/mod.rs)
- Why it matters to us: this is a strong terminology-audit case because the
  source itself already gives the scope disclaimer.

### 7. The Lean Surface Contains Real Local Mathematical Signal

- Claim: midpoint obstruction, coprimality filters, admissible endings, unit
  residue symmetry, and orbit-count statements are genuine proved local claims.
- Best sources: [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md)
- Why it matters to us: it gives us a genuine "proved" bucket instead of only
  prose claims and code hypotheses.

### 8. Verification Entry Points Are Evidence Surfaces, Not Claims

- Claim: commands like `cargo test --lib`, `cargo clippy`, `lake build`, and
  named example runners should increase confidence in the repo's evidence
  surface without being mistaken for direct evidence of any single hypothesis.
- Best sources: [STATUS.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/STATUS.md), [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md)
- Why it matters to us: it tests whether our system can distinguish "verification
  infrastructure exists" from "claim is therefore true."

## Source Set

Primary sources used in this tranche:

- [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md)
- [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md)
- [prime_outer_analysis.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/prime_outer_analysis.py)
- [analyze_discriminant.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py)
- [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md)
- [STATUS.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/STATUS.md)

## Why This Corpus Is Good

It contains all three signal classes we want:

- verified contributions,
- explicit non-claims and refutations,
- executable statistical probes with named hypotheses.

That mix is ideal for our system because it lets us test whether the loop can:

- distinguish verified signal from speculation,
- preserve negative results instead of forgetting them,
- and improve its suggestions as evidence accumulates.

## Mined Hypotheses

### 1. Coprimality-Dominated Interpretation

- Claim: the observed lift is largely explained by classical coprimality
  filtering plus candidate-size effects.
- Source: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md), [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md)
- Expected status: `supported`
- Replay form: a history case where a flashy template-specific route loses to a
  simpler classical-explanation route.
- Good success signal: the system prefers an explanation like
  `coverage_targeted_fallback_work` over a novelty-heavy route when the matched
  control ratio is near `1`.
- Good falsifier: the system keeps insisting on a strong template-specific
  mechanism despite weak control separation.

### 2. Negative Results Must Persist

- Claim: rejected hypotheses such as `k* ~ sqrt(M)`, `2p resonance`, and large
  template-specific bonuses should remain part of the live memory surface.
- Source: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md)
- Expected status: `supported`
- Replay form: a forensics case where the loop should say "this mechanism was
  previously tested and not supported."
- Good success signal: persistent or session memory suppresses repeatedly
  re-suggesting already-refuted routes.
- Good falsifier: the same discredited story keeps resurfacing as if it were new.

### 3. Structural Distinction Is Not Mechanistic Proof

- Claim: a structural distinction from ordinary palindromes is real, but it is
  not evidence for a new density mechanism by itself.
- Source: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md), [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md)
- Expected status: `supported but narrow`
- Replay form: a `mixed` case where a structural finding is classified as real
  but insufficient to justify a broader causal claim.
- Good success signal: the loop can say "verified structural distinction, open
  mechanism."
- Good falsifier: it collapses all verified facts into one inflated story.

### 4. Single-Instance Asymmetry Should Stay Narrow

- Claim: connector asymmetry is real for the canonical pair, but broader
  generalization is open.
- Source: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md), [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md)
- Expected status: `supported single instance`
- Replay form: a history case where the loop should preserve the phrase
  "verified single-instance phenomenon, open generalization."
- Good success signal: the suggestion system recommends more matrix breadth or
  family generalization instead of overclaiming.
- Good falsifier: it treats one canonical pair as a general theorem.

### 5. Prime Outer Digit Correlation Is A Statistical Claim, Not A Story

- Claim: M=2 anomalies may correlate with prime outer digits, but the decision
  should follow Fisher exact significance rather than a memorable anecdote.
- Source: [prime_outer_analysis.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/prime_outer_analysis.py)
- Expected status: `open / likely fragile`
- Replay form: a forensics case whose recommendation changes with p-value and
  anomaly count.
- Good success signal: the loop uses evidence levels like `supported`,
  `suggestive`, `not significant`.
- Good falsifier: it upgrades a memorable `4/4` pattern into a mechanistic claim.

### 6. Discriminant Magnitude Needs A Real Correlation Test

- Claim: discriminant properties should only be treated as predictive if the
  point-biserial, Spearman, or Welch tests separate meaningfully.
- Source: [analyze_discriminant.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py)
- Expected status: `open`
- Replay form: one case with a strong discriminant signal and one with no
  correlation.
- Good success signal: the loop downgrades discriminant-based stories when the
  correlations are weak.
- Good falsifier: it clings to discriminant rhetoric even when tests fail.

### 7. k=1 Advantage Needs Mechanistic Evidence, Not Just A Delta

- Claim: if `k=1` wins, the question is whether it wins because it selects
  better discriminants or favorable residue structure.
- Source: [analyze_discriminant.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py)
- Expected status: `conditional`
- Replay form: a `mixed` case where density advantage exists but the mechanism
  remains unclear.
- Good success signal: the loop says "advantage exists, mechanism not yet
  significant."
- Good falsifier: it jumps directly from a density delta to a mechanism claim.

### 8. Goldbach Richness Is A Good Example Of A Soft Predictor

- Claim: Goldbach-pair counts may correlate with primality, but weak
  correlation should remain weak in the explanation layer.
- Source: [analyze_discriminant.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py)
- Expected status: `weak or open`
- Replay form: a case where the loop recommends more data instead of upgrading
  a weak signal.
- Good success signal: it emits a suggestion like "collect more matched data."
- Good falsifier: it treats weak correlation as if it were durable.

### 9. Formalization Surfaces Should Increase Trust, Not Inflate Claims

- Claim: Lean and Agda formalization surfaces are real rigor contributions, but
  they do not automatically prove the larger density interpretation.
- Source: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md), [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md), [STATUS.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/STATUS.md)
- Expected status: `supported`
- Replay form: a trust case where proof infrastructure strengthens local
  subclaims but not the entire public narrative.
- Good success signal: the loop can say "proved local theorem family, open
  global interpretation."
- Good falsifier: it treats formalization volume as evidence for unrelated claims.

### 10. Physics Metaphors Should Be Downweighted

- Claim: gravity, tidal, and Lagrange metaphors are visualization or legacy
  surfaces, not mathematical evidence.
- Source: [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md)
- Expected status: `supported`
- Replay form: an explanation-quality case where the loop should prefer
  standard mathematical language over metaphor.
- Good success signal: it routes toward classical or formal explanations first.
- Good falsifier: it centers metaphor when audited evidence says not to.

## How To Encode These In Our Demo System

### Replay Cases

Best fits:

- Hypotheses 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

Why:

- they are mostly about interpretation quality, evidence strength, and claim
  scope
- they do not require live model execution to be useful

### Forensics Cases

Best fits:

- Hypotheses 5, 6, 7, 8

Why:

- they are shaped like statistical diagnosis questions
- they benefit from "what the loop noticed" plus "what to try next"

### Persistence Experiments

Best fits:

- Hypotheses 2, 4, 9, 10

Why:

- they test whether negative results and scope limits remain visible over time

### Adaptation Narratives

Best fits:

- Hypotheses 1 and 7

Why:

- they let us ask whether the system becomes more conservative or more precise
  as comparable evidence accumulates

## Persistence Questions This Corpus Gives Us

This corpus is especially strong for testing memory quality:

- Does session memory reduce re-suggesting already-refuted stories?
- Does persistent memory preserve claim scope, or does it flatten everything
  into "proven"?
- Can the loop remember that a finding is single-instance, not general?
- Can it keep formal proof surfaces and global interpretation claims separate?

## Recommended Next Encodings

If we continue this tranche, I would encode these next:

1. a replay fixture for "negative results must persist"
2. a forensics fixture for "suggestive vs significant statistical signal"
3. a persistence demo where a scoped history file learns to suppress
   overclaiming language
4. a terminology audit case that downgrades physics-metaphor explanations when
   stronger classical language is available
