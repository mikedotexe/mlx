# Prime Physics Engine: Formal Surface Signal Map

This file is the tighter Rust-and-Lean-only pass over the prime-physics corpus.

Sources used:

- [src/tidal/mod.rs](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/src/tidal/mod.rs)
- [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md)
- [PrimeArithmetic.lean](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/PrimeArithmetic.lean)

Observed ingest summary:

- sources: `3`
- candidate fragments: `157`
- Rust `tidal` module fragments: `49`
- Lean theorem-index fragments: `108`
- direct fragments from `PrimeArithmetic.lean`: `0`

## Separation Result

The Rust-and-Lean-only pass gives a very clear result:

- `proved`: strong and abundant
- `metaphor-only`: explicit and easy to identify
- `refuted`: not directly carried by the Rust/Lean surface itself

That last point matters. The refutations in this corpus live primarily in
prose, especially [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md), not in the formal surface.

## Proved Statements

Theorem families that should be treated as genuine local mathematical signal:

- midpoint obstruction / fixed-point exclusion in paired residue families
- modular mirror symmetry with only trivial fixed residues
- balanced reflected residue buckets and constructive pairing
- balanced bucket counts yielding certificate-style midpoint exclusion
- generated residue-list and position-list artifacts feeding certificate shells
- small finite windows certified end to end
- non-coprime boundaries fail
- admissible endings are exactly the units
- unit residues are closed under complement / negation
- admissible residues split into two-element symmetry orbits

Best source:

- [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md)

Operational reading for our system:

- these should increase trust in local residue and coprimality subclaims,
- they should not be upgraded into broad density-mechanism claims without other evidence.

## Refuted Statements

Direct Rust/Lean result:

- none explicitly surfaced as refutations

Interpretation:

- the formal surface is not where this corpus stores its negative-result memory
- if we ingest only Rust and Lean, we will overrepresent proof and underrepresent refutation

Practical consequence for our system:

- do not treat Lean/Rust-only ingest as a sufficient corpus for claim-memory demos
- always pair it with prose sources when negative-result persistence matters

Closest adjacent contrast:

- the formal surface proves narrow local structure
- the prose surface carries the "tested and not supported" statements

## Metaphor-Only Or Non-Core Statements

The Rust tidal module makes this unusually explicit.

Statements that should remain in the metaphor bucket:

- tidal analysis is a "simulation / visualization" layer
- the "optimal tidal strength" observation is an empirical metaphor result, not a number-theoretic theorem
- readers should defer to `crate::gravity` and `crate::hzlib` for the actual mathematical analysis tools

Best source:

- [src/tidal/mod.rs](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/src/tidal/mod.rs)

Operational reading for our system:

- tidal, gravity, Roche-limit, and related language should be downweighted in explanation mode
- this module is a scope disclaimer as much as it is a code surface

## What `PrimeArithmetic.lean` Means Here

The ingest scaffold extracted `0` direct fragments from
[PrimeArithmetic.lean](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/PrimeArithmetic.lean).

That does not mean the Lean package is empty. It means:

- the useful formal signal in this tighter pass comes from the curated theorem
  inventory in [THEOREM_INDEX.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/lean-proofs/THEOREM_INDEX.md)
- the top-level Lean aggregator file is not itself the right extraction target
  for hypothesis fragments

## Recommended Policy For This Corpus

When ingesting this repo, the safest separation is:

- `proved`: theorem-index rows with `Status = proved`
- `metaphor_only`: tidal/gravity visualization language from the Rust surface
- `refuted`: prose-only negative-result rows from [NOVELTY.md](/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md)

That three-way split is the right one for our demo system because it keeps:

- local proof,
- explicit refutation,
- and scoped metaphor

from getting flattened into one undifferentiated "support" bucket.
