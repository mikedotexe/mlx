# Prime Physics Engine: Live Ingest Report

This file records the live corpus-ingest run against
[/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine).

The goal is not to judge the mathematics. The goal is to preserve what the
ingest and replay surfaces actually extracted from the real repository.

## Run Summary

- Corpus label: `prime_physics_engine`
- Sources scanned: `8`
- Candidate fragments extracted: `212`
- Replay check: `history-replay` passed
- Persistence check: `persistence-memory` passed

Commands used:

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python python /Users/mikepurvis/other/mlx/benchmarks/python/idea_ingest_scaffold.py \
  --corpus-label prime_physics_engine \
  --format markdown \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/STATUS.md \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/prime_outer_analysis.py \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/src/tidal/mod.rs \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/PrimeArithmetic.lean
```

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python python /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py --demo-preset history-replay
```

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python python /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py --demo-preset persistence-memory
```

## Candidate Counts By Source

- [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md): `25`
- [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md): `14`
- [STATUS.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/STATUS.md): `10`
- [prime_outer_analysis.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/prime_outer_analysis.py): `1`
- [analyze_discriminant.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py): `5`
- [src/tidal/mod.rs](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/src/tidal/mod.rs): `49`
- [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md): `108`
- [PrimeArithmetic.lean](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/PrimeArithmetic.lean): `0`

## Highest-Signal Extracts

### Refuted Or Narrowed Claims

Pulled most clearly from [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md):

- `k* ~ sqrt(M)` scaling law: refuted
- `2p` resonance: refuted in tested cases
- large template-specific bonus: not detected
- boundary-digit specialness after coprimality matching: not detected
- phase-lock harmonic story: not detected

### Verified Or Operational Evidence

Pulled from [EVIDENCE.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md) and [STATUS.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/STATUS.md):

- membrane vs random-coprime structure ratio near `1.020 +/- 0.053`
- interpretation explicitly says "not statistically distinguishable from `1.0`"
- concrete verification entrypoints exist, especially `cargo run --example membrane_vs_random`
- repo-health evidence is explicit: `cargo test --lib`, `cargo clippy --lib -- -D warnings`, `lake build`

### Statistical Probe Surfaces

Pulled from [prime_outer_analysis.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/prime_outer_analysis.py) and [analyze_discriminant.py](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/analyze_discriminant.py):

- prime outer-digit anomaly hypothesis
- perfect-square discriminant hypothesis
- discriminant/primality correlation
- `k=0` vs `k=1` discriminant shift
- Goldbach-richness hypothesis

### Formal And Scope Surfaces

Pulled from [src/tidal/mod.rs](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/src/tidal/mod.rs) and [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md):

- the Rust tidal layer explicitly says it is a visualization metaphor and "not core math"
- the Lean theorem index contains many `proved` local statements about midpoint obstruction, coprimality, admissible residues, unit-residue symmetry, and orbit structure

## Replay And Memory Results

### `history-replay`

Observed summary:

- passed
- exercised both comparison modes
- calibrated suggestion was `Quantized Direct Map`
- all `3/3` external-corpus replay cases matched the expected route choice

The external-corpus replay cases were:

- negative results must persist
- single-instance asymmetry stays narrow
- formalization increases trust, not claim size

### `persistence-memory`

Observed summary:

- passed
- session suppression score: `3/3`
- unscoped resurfacing score: `3/3`
- scoped suppression score: `3/3`

Interpretation:

- the current system can keep negative-result memory alive,
- can show that unscoped persistence is dangerous,
- and can restore the guardrail once memory is bucketed again.

## What This Run Proved About Our System

- The scaffold is useful on a real prose-plus-code-plus-formal corpus, not just synthetic inputs.
- The replay layer can keep refuted ideas alive instead of letting them vanish under later positive evidence.
- The persistence demo can distinguish session memory from unscoped persistent memory.
- The system is especially strong at three things:
  - preserving negative results
  - preserving claim scope
  - separating local formal proof from larger narrative claims

## What This Run Did Not Prove

- It did not adjudicate whether the prime-physics claims are correct in the large.
- It did not automatically convert every extracted fragment into a high-quality hypothesis.
- It did not find direct Lean fragments in [PrimeArithmetic.lean](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/PrimeArithmetic.lean); the useful formal signal came from [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md).

## Related Artifacts

- [load_mmap_prime_physics_hypotheses.md](/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_prime_physics_hypotheses.md)
- [load_mmap_prime_physics_formal_signal_map.md](/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_prime_physics_formal_signal_map.md)
- [load_mmap_external_idea_ingest.md](/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_external_idea_ingest.md)
- [load_mmap_demo_history.jsonl](/Users/mikepurvis/other/mlx/benchmarks/python/testdata/load_mmap_demo_history.jsonl)
- [load_mmap_prime_physics_demo_history.jsonl](/Users/mikepurvis/other/mlx/benchmarks/python/testdata/load_mmap_prime_physics_demo_history.jsonl)
