# Unified Memory and Zero-Copy Guide

This guide explains the two related but different stories behind the MLX
Apple-silicon memory model:

1. `Unified memory execution`
2. `Mapped file loading`

They are both important, but they are not the same thing.

## Unified Memory

On Apple silicon, MLX arrays live in a unified memory architecture where the
CPU and GPU share the same memory pool.

The important behavioral consequence is:

- arrays are not "moved to the GPU" as a mandatory programming step
- instead, MLX decides which device executes an operation when you run that
  operation

That means one array can participate in CPU work and GPU work without the user
manually shuttling it between separate device heaps.

## Zero-Copy Host Views

When you convert an MLX array to NumPy with:

```python
np.array(arr, copy=False)
```

that is the zero-copy host-view path. The resulting NumPy array does not own
its memory and can expose the same underlying storage.

By contrast:

```python
np.array(arr)
```

is a copy.

The local exploratory measurements that motivated this guide showed this very
clearly on Apple silicon:

- `np.array(arr, copy=False)` was on the order of hundreds of nanoseconds
- `np.array(arr)` was on the order of hundreds of microseconds

The exact numbers are machine- and runtime-specific, but the qualitative
behavior is the key point.

## NumPy to MLX Is Still a Copy

Current MLX conversion from NumPy to MLX should be treated as a copy path.

That is visible in:

- [`python/src/convert.cpp`](../../python/src/convert.cpp)

where the NumPy conversion helper explicitly says:

- `Make a copy of the numpy buffer`

and it is also visible in local timing probes where `mx.array(np_array)` is
materially slower than the zero-copy NumPy host view.

Do not describe `mx.array(np_array)` as zero-copy unless the implementation
changes.

## Mapped Loading Is A Different Story

The `memory_map=True` loader path is not "NumPy zero-copy" and not merely
"unified memory." It is a file-to-runtime storage story.

With mapped loading, MLX can attempt to wrap file-backed tensor pages directly
instead of allocating and copying tensor payloads into new buffers.

That path is exposed through:

- `mx.load(..., memory_map=True)`
- `mx.last_mmap_load_stats()`

and instrumented by:

- [`benchmarks/python/load_mmap_bench.py`](load_mmap_bench.py)

It depends on runtime/backend capability. On a CPU-only local repo build, a
real safetensors probe may still fall back completely with reasons like
`make_buffer_failed`, while a Metal-capable runtime is the lane where the
shared-buffer story can actually be realized.

On Metal-enabled repo builds, the mapped-loading tests also depend on a working
Apple Metal toolchain. In practice that means both `metal` and `metallib` must
be discoverable by `xcrun` at configure time so the runtime can build and load
`mlx.metallib` instead of failing later during `ctest`.

## What The Benchmark Covers

[`benchmarks/python/unified_memory_zero_copy_bench.py`](unified_memory_zero_copy_bench.py)
keeps these lanes separate on purpose:

- `wheel` lane:
  measures unified-memory execution and zero-copy host views in a Metal-enabled
  MLX runtime
- `repo_mmap_probe` lane:
  measures mapped-vs-copied bytes and fallback reasons in the repo runtime

The benchmark reports nanosecond timings and can emit either human-readable
text or JSON.

## Example

```bash
python benchmarks/python/unified_memory_zero_copy_bench.py \
  --wheel-python /path/to/metal-enabled/python \
  --repo-pythonpath /path/to/mlx/python \
  --model-file /path/to/model.safetensors \
  --format text
```

This should be read as:

- "how cheap is host-view creation?"
- "how do mixed CPU/GPU execution paths behave under unified memory?"
- "did the repo runtime actually map bytes, or did it fall back to copying?"

Those are related questions, but not interchangeable ones.
