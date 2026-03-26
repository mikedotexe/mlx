# Load mmap Demo Show Script

This is a 15-19 minute engineer-facing demo for the mmap experiment system.
The story is not "mmap is always better." The story is that the loop can:

1. replay what it has learned,
2. localize regressions,
3. adapt its policy choice to context,
4. keep itself honest with correctness guardrails.

## Theme

"Desk mode vs travel mode" for a local copilot.

- Desk mode: warm cache, repeated interactions, dense weights, where mapped or hybrid policies can shine.
- Travel mode: colder starts, tighter memory, quantized GGUF, where copy or a conservative hybrid may be the safer choice.

The novelty is that the benchmark does not just report numbers. It explains why the choice changes, records outcomes, and can defend that behavior with history and guardrails.

## Setup

Use one terminal for the demo.

```bash
export BENCH=/Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py
export HIST=/tmp/load_mmap_show_history_$(date +%Y%m%d_%H%M%S).jsonl
export DENSE=/absolute/path/to/your/dense_model.safetensors
export QUANT=/absolute/path/to/your/quantized_model.gguf
```

Optional quick sanity check:

```bash
python - <<'PY'
import mlx.core as mx
print("mlx ok", hasattr(mx, "load"))
PY
```

If `mlx.core` is not importable, skip the live adaptation and golden sections and run only the replay sections. Those are still valuable and deterministic.

## Minute 0-2: Frame The Problem

Say:

"We are not trying to prove a single optimization trick. We are trying to prove that this loop can learn which loading policy is right for a given bucket, explain its reasoning, and stay grounded in correctness."

"The two key questions are: does it adapt when the context changes, and can we trust it when it adapts?"

No command yet. This is the framing.

## Minute 2-5: Deterministic Replay

Run:

```bash
python "$BENCH" --demo-preset history-replay
```

Say:

"First I want a zero-drama proof that the learning loop exists independent of this machine's current thermal state."

"This replay is canned history. It shows that the system can summarize bucket winners, losers, route hit-rate, and choose the right comparison basis."

Point out:

- `comparison checks: comparable=bucket_recent_median explicit=baseline_run`
- the calibrated route suggestion
- that this did not require a live model load

Success condition:

- the replay passes
- it shows both comparison modes
- it surfaces a concrete calibrated suggestion

## Minute 5-8: Regression Forensics

Run:

```bash
python "$BENCH" --demo-preset regression-forensics --demo-format text
```

Say:

"Now we move from summary to diagnosis. I want to see if the loop can notice different failure shapes and suggest the next route instead of just saying 'performance changed.'"

"These cases are curated: parse or tensor setup regression, first-token regression, and a mixed load-vs-memory tradeoff."

Point out:

- the case that notices parse and tensor-setup issues
- the case that notices first-token regression
- the mixed tradeoff case where the outcome is not a simplistic win/loss
- the suggested next route per case

Success condition:

- each case calls out a different problem shape
- each case suggests a plausible next route

## Minute 8-13: Live Adaptation Ladder

Run:

```bash
python "$BENCH" \
  --demo-preset adaptation-ladder \
  --history-json "$HIST" \
  "$DENSE" \
  "$QUANT" \
  --decode-synth
```

If you do not have convenient model paths, use discovery instead:

```bash
python "$BENCH" \
  --demo-preset adaptation-ladder \
  --history-json "$HIST" \
  --discover-models \
  --discover-root /absolute/path/to/models \
  --decode-synth
```

Say:

"This is the live part. The demo seeds a few fixed policy candidates per bucket, then lets auto policy choose after it has real local evidence."

"The buckets are intentionally different: dense warm, dense cold, quantized warm, and misaligned warm."

"What I care about is not that one policy wins everywhere. I care that the chosen policy changes for understandable reasons."

Point out while it runs:

- dense warm should usually lean toward `mapped` or a non-copy hybrid
- dense cold may favor mapped or prefetch-friendly behavior
- quantized warm should often lean toward `copy` or cautious hybrid behavior
- misaligned warm should be hostile to aggressive mapping

After the summary appears, point out:

- `chosen policy`
- `winner policy`
- `runner_up policy`
- the `match`, `near-match`, or `miss` verdict
- the candidate reasons
- the overall bucket match rate

Success condition:

- at least one dense bucket picks a non-copy policy
- quantized or misaligned chooses `copy` or a conservative non-copy policy
- the explanation mentions history support, fallback mix, or mapped ratio

## Minute 13-15: Read The Learned History

Run:

```bash
python "$BENCH" \
  --history-json "$HIST" \
  --report-history \
  --report-format text \
  --report-top 3
```

Say:

"Now I want to see whether the run left behind something reusable. This is the important difference between a benchmark and a research loop."

"The system should now be able to tell us which policies are winning per bucket, which fallback reasons dominate, and where regressions cluster."

Point out:

- route hit-rate by bucket
- policy winners and losers
- fallback reasons by bytes
- top phase regressions

Success condition:

- the report clearly separates buckets
- it shows policy winners and route hit-rate, not just raw trial data

## Minute 15-17: Guardrail The Claim

Run:

```bash
python "$BENCH" \
  --demo-preset golden-guardrail \
  --history-json "$HIST"
```

Say:

"This is the honesty check. If the clever choices are not correct, the rest of the story does not matter."

"The guardrail wraps dense parity, quantized parity, and the alignment-hostile case in a demo summary instead of burying them in a test suite."

Point out:

- per-case parity status
- failed case count, if any
- fallback reason sanity summary

Success condition:

- guardrail passes
- the output makes it obvious that adaptation claims are only valid while this stays green

## Minute 17-19: Close

Say:

"What is special here is not a single optimization. It is that the system can replay what it learned, diagnose what changed, adapt its policy to the bucket, and then defend that choice with a correctness backstop."

"That means we can use it both as a research tool and as a trust-building tool for future runtime policy work."

"The next frontier after this is longitudinal learning across commits and more decode-aware auto policy."

## Backup Plan If Live MLX Is Unavailable

Use this shorter 8-10 minute version:

```bash
python "$BENCH" --demo-preset history-replay
python "$BENCH" --demo-preset regression-forensics --demo-format text
python "$BENCH" --demo-preset history-replay --demo-format json
```

Say:

"I can still prove the learning and explanation loop today. The only part I am skipping is the live execution path, which depends on a local MLX runtime."

## Presenter Notes

- Keep coming back to "adaptation quality, not one universal winner."
- Prefer showing one or two lines from each section, not reading the full output.
- If the adaptation ladder has a miss, use it. A visible miss with good evidence is often more convincing than a perfect canned win.
- If the audience is more systems-oriented, emphasize fallback bytes and first-token regression.
- If the audience is more product-oriented, emphasize desk mode vs travel mode and "trustworthy default policy."
