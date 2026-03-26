# Load mmap Demo Show Script: Model Intake Concierge

This is a second 15-19 minute engineer-facing demo for the mmap experiment
system. It is deliberately different from the "desk mode vs travel mode"
story.

The story here is:

1. unfamiliar models arrive in a folder,
2. the system has to make a sane first policy choice,
3. it should improve as it sees more similar models,
4. and that learning should be scoped, inspectable, and reversible.

## Theme

"Model Intake Concierge" for a bring-your-own-model app.

The product question is not "is mmap faster?" The product question is:

"When a user drops random checkpoints into a folder, can we pick a reasonable
default load policy, explain it, and get better over time without turning the
history file into a mysterious oracle?"

## What Makes This Different

- The first show script is about one machine changing operating context.
- This one is about many unknown artifacts arriving over time.
- The first novelty is adaptation by bucket.
- This novelty is adaptation plus scoped memory.

## Setup

Use one terminal.

```bash
export BENCH=/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py
export DROP=/absolute/path/to/a/model_drop_folder
export FRESH=/tmp/load_mmap_intake_fresh_$(date +%Y%m%d_%H%M%S).jsonl
export SESSION=/tmp/load_mmap_intake_session.jsonl
export PERSIST=~/.cache/mlx/load_mmap_intake_$(hostname -s).jsonl
```

Optional sanity check:

```bash
python - <<'PY'
import mlx.core as mx
print("mlx ok", hasattr(mx, "load"))
PY
```

If `mlx.core` is not importable, skip the live intake and guardrail sections.
You can still tell most of the story using the replay demos from the main show
script.

## Persistence Ladder

We are going to toy with three levels of memory:

- `fresh`: a unique history file for a single run; good for proving first-contact behavior.
- `session`: a reused history file for one work session or one day; this is the sweet spot for semi-persistence.
- `persistent`: a host-scoped longer-lived history file; useful, but only if we keep it scoped and inspectable.

Important framing line:

"I do not want one global immortal benchmark memory. I want scoped memory with an obvious reset button."

## Minute 0-2: Frame The Intake Problem

Say:

"Imagine a local app that lets people drag in checkpoints from Hugging Face, Ollama blobs, hand-quantized GGUFs, and half-odd synthetic artifacts from experiments."

"The intake layer needs to make a decent default choice before anyone has hand-tuned a knob."

No command yet.

## Minute 2-5: First Contact With No Memory

Run:

```bash
rm -f "$FRESH"
python "$BENCH" \
  --discover-models \
  --discover-root "$DROP" \
  --discover-max-results 4 \
  --policy-mode auto \
  --history-json "$FRESH" \
  --decode-synth \
  --runs 1 \
  --warmup-runs 0 \
  --attempts 1
```

Say:

"This is the concierge meeting unfamiliar models for the first time. There is no prior memory except the built-in heuristics and whatever the mapped preflight probe sees."

"What I want here is not perfection. I want conservative, explainable choices."

Point out:

- the discovered files
- the printed effective policy mode and signature
- any auto-policy candidate reasons
- whether dense-looking artifacts go non-copy and quantized or hostile ones stay conservative

Success condition:

- the system chooses different policies across discovered files
- the explanations reference mapped ratio, fallback mix, or history support

## Minute 5-7: Read The Fresh Intake Memory

Run:

```bash
python "$BENCH" \
  --history-json "$FRESH" \
  --report-history \
  --report-format text \
  --report-top 3
```

Say:

"Even the first contact run leaves behind something useful. The question is whether it is readable enough that a teammate would trust it."

Point out:

- bucket separation
- policy winners and losers
- fallback reasons by bytes
- whether any route hit-rate is already emerging

## Minute 7-11: Semi-Persistent Memory

Run twice against the same session history:

```bash
python "$BENCH" \
  --discover-models \
  --discover-root "$DROP" \
  --discover-max-results 4 \
  --policy-mode auto \
  --history-json "$SESSION" \
  --decode-synth \
  --runs 1 \
  --warmup-runs 0 \
  --attempts 1
```

```bash
python "$BENCH" \
  --discover-models \
  --discover-root "$DROP" \
  --discover-max-results 4 \
  --policy-mode auto \
  --history-json "$SESSION" \
  --decode-synth \
  --runs 1 \
  --warmup-runs 0 \
  --attempts 1
```

Then report:

```bash
python "$BENCH" \
  --history-json "$SESSION" \
  --report-history \
  --report-format text \
  --report-top 4
```

Say:

"This is the semi-persistent version. Same session, same machine, same intake problem, but now the concierge has memory."

"What I want to see is not blind repetition. I want stronger bucket evidence and more legible winners and losers."

Point out:

- stronger policy winner/loser separation
- route hit-rate by bucket
- whether similar discovered models start collapsing into the same policy family

Success condition:

- the second run produces richer bucket evidence than the first
- the history report looks more like a learned intake ledger than raw benchmark debris

## Minute 11-14: Audit The Concierge With Adaptation Ladder

Run:

```bash
python "$BENCH" \
  --demo-preset adaptation-ladder \
  --history-json "$SESSION" \
  --discover-models \
  --discover-root "$DROP" \
  --decode-synth
```

Say:

"Now I want to test whether the session memory is helping the auto chooser in a principled way."

"The ladder seeds known candidate policies per bucket and then asks auto policy to pick. This is the audit of the concierge."

Point out:

- the dense warm and dense cold buckets
- the quantized warm bucket
- the misaligned hostile bucket
- the `match`, `near-match`, or `miss` result
- the chosen policy versus seeded winner

Success condition:

- at least one dense bucket chooses a non-copy policy
- quantized or hostile input chooses copy or a cautious hybrid
- the system can explain why

## Minute 14-16: Persistent Memory, Carefully Scoped

Run:

```bash
python "$BENCH" \
  --discover-models \
  --discover-root "$DROP" \
  --discover-max-results 4 \
  --policy-mode auto \
  --history-json "$PERSIST" \
  --decode-synth \
  --runs 1 \
  --warmup-runs 0 \
  --attempts 1
```

Then:

```bash
python "$BENCH" \
  --history-json "$PERSIST" \
  --report-history \
  --report-format text \
  --report-top 4
```

Say:

"This is the long-lived version, but notice the scoping: one file per host or deployment context, not one universal global memory."

"Persistent memory is powerful, but it only stays trustworthy if it is segmented, inspectable, and easy to throw away."

Point out:

- that the same reporting tools work on the persistent file
- that you can compare it to session behavior
- that the file is just JSONL, not hidden magic

## Minute 16-18: Guardrail The Intake Story

Run:

```bash
python "$BENCH" \
  --demo-preset golden-guardrail \
  --history-json "$SESSION"
```

Say:

"The concierge only gets to keep its authority if correctness remains green. If parity breaks, the learned default no longer matters."

Point out:

- dense parity
- quantized parity
- alignment-hostile case
- fallback reason sanity

## Minute 18-19: Close

Say:

"This demo is about operational memory, not just raw performance."

"The system can make a first-contact guess, improve during a session, accumulate scoped long-lived evidence, and still expose every part of that memory in ordinary reports."

"That is what makes it feel like a real intake concierge instead of a benchmark script with extra flags."

## Suggested Talking Points On Persistence

Use these if the audience gets curious:

- Fresh history proves the heuristic floor.
- Session history is my favorite: it gives learning without too much fossilized bias.
- Persistent history is useful only when segmented by host, branch, model family, or deployment lane.
- If the persistent file starts feeling magical, it is already too persistent.
- The best persistence is boring to inspect and easy to delete.

## Good Questions To Invite

- Should persistent history be segmented by `host`, `branch`, or `model family`?
- How much history is enough before the auto chooser should trust it more than the probe?
- When should we deliberately ignore history because the environment changed too much?
- What should graduate from benchmark memory into runtime defaults?

## Backup Plan If The Live Runtime Is Unavailable

Use the replay path plus the persistence framing:

```bash
python "$BENCH" --demo-preset history-replay
python "$BENCH" --demo-preset regression-forensics --demo-format text
```

Say:

"I cannot show the live intake path on this machine right now, but I can still show the core behaviors we would rely on: replayable history, explainable diagnosis, and auditable suggestions."

## Presenter Notes

- Keep saying "scoped memory" instead of just "persistence."
- Emphasize that semi-persistence is often more valuable than permanent memory.
- If the audience worries about overfitting, agree with them and use that to justify host- or lane-scoped histories.
- If the intake run makes a visible mistake, use it as a strength: the whole point is that the mistake becomes reportable and improvable.
