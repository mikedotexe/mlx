# External Idea Ingest Workflow

This workflow shows how to feed external prose and code into the mmap demo and
research loop without pretending the benchmark directly understands arbitrary
theory text.

The pattern is:

1. ingest source material into a reviewable hypothesis scaffold,
2. curate the best claims into a hypothesis ledger,
3. map each claim to a demo shape,
4. decide what persistence level should remember it.

## What This Is For

Use this when you have:

- prose with claims or interpretations,
- code that names hypotheses or tests mechanisms,
- formal material that sharpens what is proved versus open,
- or a mixed research repo you want to pressure-test with our loop.

The scaffold currently has first-class extraction for:

- `Markdown` and `txt`
- `Python`
- `Rust` doc comments and `#[test]` surfaces
- `Lean` doc blocks plus theorem/lemma/definition declarations

## Step 1: Choose A Small, High-Signal Source Set

Good source sets usually include:

- one prose claim file
- one evidence or status file
- one code file that contains executable tests or statistical logic

For the prime-physics-engine example:

```bash
python /Users/mikepurvis/other/mlx/benchmarks/python/idea_ingest_scaffold.py \
  --corpus-label prime_physics_engine \
  --format markdown \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/NOVELTY.md \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/EVIDENCE.md \
  /Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/primes/prime-physics-engine/prime_outer_analysis.py
```

This does not produce final truth. It produces a reviewable scaffold.

## Step 2: Curate Into A Hypothesis Ledger

Take the scaffold output and rewrite it into a short ledger with:

- the claim
- expected status
- source path
- success signal
- falsifier
- best demo form

Example ledger:

- [load_mmap_prime_physics_hypotheses.md](/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_prime_physics_hypotheses.md)

The act of curation matters. It is where we compress "interesting text" into
"testable claim."

## Step 3: Map Each Claim To A Demo Shape

Use this rule of thumb:

- `replay`
  Best for claim scope, negative-result memory, winners/losers, and evidence-quality checks.
- `forensics`
  Best for statistical ambiguity, mixed outcomes, and "what should we test next?"
- `adaptation`
  Best for policy or explanation changes across buckets or repeated runs.
- `guardrail`
  Best for correctness, narrow proofs, and "do not overclaim" checks.
- `persistence`
  Best for questions about forgetting, overfitting, or scope drift over time.

## Step 4: Choose A Memory Mode

Not every idea should persist the same way.

- `fresh`
  Use when you want a first-contact read without prior bias.
- `session`
  Use when you want semi-persistent learning over a work session or a review meeting.
- `persistent`
  Use only when the history is scoped by host, branch, model lane, or corpus family.

Rule:

If you are not sure whether something deserves persistent memory, start with
session memory.

## Step 5: Ask Four Questions Of The Claim

Every ingested claim should answer these:

1. What would make this look true?
2. What would make this look false?
3. What weaker interpretation would still survive?
4. Should the system remember this as `supported`, `refuted`, `single-instance`,
   `open`, or `unclear`?

This is the minimum needed to keep the loop from collapsing into hype.

## Good Claim Shapes

These work especially well:

- "This effect disappears under matched controls."
- "This phenomenon is real, but only in one canonical case."
- "This predictor is suggestive, not significant."
- "This structural distinction is real but not mechanistic proof."
- "This proof surface supports a narrow subclaim, not the whole narrative."

## Weak Claim Shapes

These usually need rewriting before ingest:

- "This feels deep."
- "This probably explains everything."
- "This visualization suggests a mechanism."
- "This seems special because it is memorable."

## Suggested Directory Convention

When a corpus turns out to be valuable, keep three artifacts together:

- scaffold output from [idea_ingest_scaffold.py](/Users/mikepurvis/other/mlx/benchmarks/python/idea_ingest_scaffold.py)
- curated hypothesis ledger
- any replay or forensics fixture derived from it

For the prime-physics corpus, the first encoded replay fixture now lives at:

- [load_mmap_prime_physics_demo_history.jsonl](/Users/mikepurvis/other/mlx/benchmarks/python/testdata/load_mmap_prime_physics_demo_history.jsonl)

And the scoped-memory demo surface now lives behind:

```bash
python /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py \
  --demo-preset persistence-memory
```

That keeps the provenance of the experiment legible.

## Recommended First Tranche For A New Corpus

If you are starting from scratch, do this:

1. scaffold 3 source files
2. curate 5-10 hypotheses
3. encode 2 replay cases
4. encode 1 forensics case
5. decide whether session memory improves behavior before allowing persistence

## Prime Physics Example

For the current example corpus, start here:

- [load_mmap_prime_physics_hypotheses.md](/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_prime_physics_hypotheses.md)

The most promising next replay and persistence cases are:

- negative results must persist
- single-instance findings must not become general claims
- formal proof surfaces should increase trust locally without inflating global interpretation
- metaphor-heavy explanations should be downweighted when classical language is available
