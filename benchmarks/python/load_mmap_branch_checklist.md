# Load mmap Branch Checklist

This checklist is the pre-branch stabilization record for the mmap research stack.

It is intentionally practical:

- how to rebuild a current local `mlx.core`,
- which tests and demos were treated as the branch gates,
- what a CPU-only local build proves,
- which history/report fields are being treated as stable for this branch.

## 1. Build A Current Local Runtime

If the machine has a working Metal toolchain, use the normal build:

```bash
python setup.py build_ext --inplace
```

If the machine is missing `metallib`, use the CPU-only fallback build:

```bash
CMAKE_ARGS='-DMLX_BUILD_METAL=OFF' python setup.py build_ext --inplace
```

This is enough to validate the Python bindings, loader diagnostics, and benchmark orchestration. It is not enough to claim that direct mapped views are active on dense safetensors.

## 2. Import Smoke Gate

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python python - <<'PY'
import mlx.core as mx
print('load', hasattr(mx, 'load'))
print('last_mmap_load_stats', hasattr(mx, 'last_mmap_load_stats'))
print('last_load_phase_stats', hasattr(mx, 'last_load_phase_stats'))
PY
```

Expected:

- `load True`
- `last_mmap_load_stats True`
- `last_load_phase_stats True`

## 3. Python Sanity Gates

```bash
python -m py_compile \
  /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py \
  /Users/mikepurvis/other/mlx/benchmarks/python/idea_ingest_scaffold.py \
  /Users/mikepurvis/other/mlx/python/tests/test_load_mmap_bench.py \
  /Users/mikepurvis/other/mlx/python/tests/test_idea_ingest_scaffold.py \
  /Users/mikepurvis/other/mlx/python/tests/test_load.py
```

```bash
python -m unittest discover \
  -s /Users/mikepurvis/other/mlx/python/tests \
  -p 'test_load_mmap_bench.py'
```

```bash
python -m unittest \
  /Users/mikepurvis/other/mlx/python/tests/test_idea_ingest_scaffold.py
```

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python:/Users/mikepurvis/other/mlx/python/tests \
  python -m unittest /Users/mikepurvis/other/mlx/python/tests/test_load.py
```

Current local expectation:

- `test_load.py` passes on the rebuilt runtime, with one backend-aware skip on CPU-only builds where dense safetensors cannot create direct mapped views.

## 4. Relevant C++ Load Gates

Build a clean CPU-only test tree if the default tree is tied to unavailable Metal tools:

```bash
cmake -S /Users/mikepurvis/other/mlx \
  -B /Users/mikepurvis/other/mlx/build_stabilize_cpu \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF
```

```bash
cmake --build /Users/mikepurvis/other/mlx/build_stabilize_cpu --target tests -j10
```

Run the load-focused doctest cases:

```bash
/Users/mikepurvis/other/mlx/build_stabilize_cpu/tests/tests \
  --test-case='*memory_map*,*gguf*,*save_safetensors*'
```

Notes:

- The full CPU-only C++ suite may still have unrelated failures outside the loader path.
- For branch stabilization, the relevant load/mmap/gguf/safetensors doctests are the gating signal.

## 5. Live Benchmark And Demo Gates

### Single-file auto policy smoke

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python python /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py \
  /tmp/load_mmap_dense_probe.safetensors \
  --policy-mode auto \
  --cache-mode warm \
  --decode-synth \
  --decode-synth-tokens 16 \
  --decode-synth-max-elems 100000 \
  --decode-synth-repeats 1 \
  --runs 1 \
  --warmup-runs 0 \
  --attempts 1 \
  --history-json /tmp/load_mmap_stabilize_history.jsonl
```

Required signal:

- the run reaches the live benchmark path,
- it prints `effective policy mode=... signature=...`,
- it prints the `auto:` summary and candidate reasons.

### Adaptation ladder

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python python /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py \
  --demo-preset adaptation-ladder \
  --demo-format text \
  --history-json /tmp/load_mmap_stabilize_history.jsonl \
  --decode-synth \
  --decode-synth-tokens 16 \
  --decode-synth-max-elems 100000 \
  --decode-synth-repeats 1 \
  --runs 1 \
  --warmup-runs 0 \
  --attempts 1
```

### Golden guardrail

```bash
PYTHONPATH=/Users/mikepurvis/other/mlx/python python /Users/mikepurvis/other/mlx/benchmarks/python/load_mmap_bench.py \
  --demo-preset golden-guardrail \
  --demo-format text \
  --history-json /tmp/load_mmap_stabilize_history.jsonl \
  --runs 1 \
  --warmup-runs 0 \
  --attempts 1
```

Interpretation:

- On a full mapped-view-capable runtime, these should be used as true branch-confidence checks.
- On the current CPU-only local build, these are still valuable because they prove the demo/reporting paths are alive and honest.
- If the output is dominated by `make_buffer_failed`, treat that as a backend limitation, not as proof that the demo layer is broken.

## 6. Stable JSONL / Report Surface

Treat these keys as the stable branch-facing evidence surface:

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

## 7. Branch Packaging

When the branch is cut, keep one feature branch but shape it into three logical commits:

1. runtime, bindings, runtime tests
2. benchmark learning loop, auto policy, demos, report/history
3. docs, show scripts, ingest scaffold, testdata
