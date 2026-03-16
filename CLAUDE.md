# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is MLX

MLX is an array framework for machine learning on Apple Silicon (and CUDA GPUs on Linux), developed by Apple ML research. It provides C++20, Python, C, and Swift APIs. Version is defined in `mlx/version.h` (currently 0.31.1).

## Build Commands

### Python package (development install)

```bash
pip install -e ".[dev]" -v           # Standard dev install
DEBUG=1 pip install -e . -v          # Debug build
```

Pass extra CMake options via `CMAKE_ARGS`, e.g.:
```bash
CMAKE_ARGS="-DMLX_BUILD_CUDA=ON" pip install -e . -v
CMAKE_ARGS="-DMLX_METAL_JIT=ON" pip install -e . -v
```

### C++ only build

```bash
cmake . -B build
cmake --build build -j$(sysctl -n hw.ncpu)    # macOS
cmake --build build -j$(nproc)                  # Linux
```

Key CMake options: `MLX_BUILD_METAL` (ON by default on macOS), `MLX_BUILD_CUDA` (OFF), `MLX_BUILD_CPU` (ON), `MLX_BUILD_TESTS` (ON), `MLX_BUILD_BENCHMARKS` (OFF), `MLX_METAL_JIT` (OFF), `MLX_METAL_DEBUG` (OFF).

### Running Tests

**Python tests:**
```bash
python -m unittest discover python/tests -v             # All tests
DEVICE=cpu python -m unittest discover python/tests -v   # CPU only
DEVICE=gpu python -m unittest discover python/tests -v   # GPU only
python -m unittest python/tests/test_ops.py -v           # Single file
```

**C++ tests** (uses doctest):
```bash
./build/tests/tests                          # All C++ tests
./build/tests/tests -tc="test name"          # Single test case
./build/tests/tests -sf="*ops_tests*"        # Filter by source file
```

GPU tests on macOS require: `METAL_DEVICE_WRAPPER_TYPE=1 METAL_DEBUG_ERROR_MODE=0`

### Formatting

Pre-commit hooks enforce `clang-format` (C++), `black` (Python), `isort --profile=black` (Python imports), and `cmake-format` (CMake). Install with:
```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files    # Check everything
clang-format -i file.cpp      # Format single C++ file
black file.py                 # Format single Python file
```

C++ style: 2-space indent, 80-column limit, braces attach, no bin-packing. See `.clang-format`.

## Architecture

### Core Abstractions (all in `mlx::core` namespace)

- **`array`** (`mlx/array.h`): The fundamental type. A node in a lazy computation graph with shape, dtype, and strides. Data (`MTLBuffer`-backed on macOS) is only materialized on `eval()`.

- **`Primitive`** (`mlx/primitives.h`): Abstract base class for all operations. Each primitive implements `eval_cpu()`, `eval_gpu()`, and optionally `jvp()`, `vjp()`, `vmap()`, and `output_shapes()`. Macros like `DEFINE_GRADS()`, `DEFINE_VMAP()`, `DEFINE_NAME()` reduce boilerplate.

- **`Stream`/`Device`** (`mlx/stream.h`, `mlx/device.h`): Operations are dispatched to streams on `cpu` or `gpu` devices. The `StreamOrDevice` alias used throughout the API defaults to the default device's default stream.

- **`Scheduler`** (`mlx/scheduler.h`): Manages per-stream worker threads. CPU streams get dedicated threads; GPU streams use the Metal/CUDA command queue.

### Operation Flow

1. **Op functions** (`mlx/ops.h`, `mlx/ops.cpp`): Create `array` nodes with associated `Primitive` objects. No computation happens yet.
2. **`eval()`** (`mlx/transforms.h`): Triggers materialization. A DFS pass resolves dependencies, then a BFS pass allocates buffers and dispatches to backends.
3. **Function transforms** (`mlx/transforms.h`): `vjp()`, `jvp()`, `vmap()`, `compile()` operate on the graph structure.
4. **`mx.fast.*`** (`mlx/fast.h`): Fused operations (RMS norm, layer norm, RoPE, SDPA, custom kernels) with hand-tuned GPU implementations.

### Backend Structure

```
mlx/backend/
  common/     # Shared utilities (broadcasting, slicing, buffer cache, reduce)
  cpu/        # CPU implementations using Accelerate/BLAS, SIMD helpers in simd/
  gpu/        # Shared GPU logic (copy, slicing, eval interface)
  metal/      # Metal backend: allocator, kernel dispatch, JIT compilation
    kernels/  # .metal shader files and headers
    jit/      # JIT kernel generation
  cuda/       # CUDA backend: cuBLAS/cuDNN integration, CUTLASS, worker threads
  no_cpu/     # Stubs when CPU backend disabled
  no_gpu/     # Stubs when GPU backend disabled
```

Each primitive's `eval_cpu()` lives in `backend/cpu/`, `eval_gpu()` is split between `backend/metal/` and `backend/cuda/` with shared logic in `backend/gpu/`.

### Python Layer

- **`python/src/`**: nanobind bindings. Each `.cpp` file mirrors a C++ module (ops, transforms, fft, linalg, etc.). Builds `mlx.core` extension module.
- **`python/mlx/`**: Pure Python. `nn/` (neural network modules following PyTorch conventions), `optimizers/`, `utils.py`.
- **`python/tests/`**: Python unittest tests. Use `DEVICE` env var to select backend.

### Key Design Patterns

- **Lazy evaluation**: All ops return unevaluated graph nodes. Call `mx.eval()` to materialize.
- **Unified memory**: On Apple Silicon, all buffers are `MTLResourceStorageModeShared`. No explicit CPU<->GPU transfers.
- **Pool allocator**: `MetalAllocator` (`backend/metal/allocator.cpp`) maintains a `BufferCache` to recycle `MTLBuffer` objects.
- **Composable transforms**: `grad`, `vmap`, `compile` are higher-order functions that transform computation graphs.
- **StreamOrDevice pattern**: Nearly every op takes an optional `StreamOrDevice s = {}` trailing parameter for device/stream targeting.

### Unified Memory Architecture

MLX's central design advantage on Apple Silicon is zero-copy CPU/GPU memory sharing. This section covers the implementation details.

#### Metal Allocator: How Zero-Copy Works

`MetalAllocator` (`backend/metal/allocator.cpp`) allocates all buffers with `MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked`. This maps the same physical DRAM pages into both CPU and GPU virtual address spaces. CPU reads data via `buffer.contents()` (raw pointer); GPU accesses through Metal buffer bindings. No `didModifyRange:` or explicit cache flushes are needed—Apple Silicon's hardware coherence handles everything.

`make_buffer(void* ptr, size_t size)` wraps existing CPU memory as an MTLBuffer via `newBufferWithBytesNoCopy` without copying (the zero-copy path for external data).

On Apple Silicon, the System Level Cache (SLC, 16–48 MiB) enables efficient GPU→CPU handoffs: buffers ≤4 MB get direct GPU→SLC→CPU transfer; larger buffers (≥32 MB) stage through DRAM first then refill SLC on CPU access. This was empirically verified by the Rust+Metal toolkit at `/Users/mikepurvis/Library/CloudStorage/Dropbox/Kairos/gpu-cpu-shared-memory` (see its `CLAUDE.md` or `RESEARCH_LANDSCAPE.md` for SLC measurement methodology and cache coherence details).

#### Buffer Pool and Recycling

`BufferCache` (`backend/common/buffer_cache.h`): LRU pool backed by `std::multimap<size_t, BufferHolder*>` indexed by buffer size. A cached buffer is reused if `cached_size < min(2 * requested, requested + 2 * page_size)`. All sizes are page-aligned (16384 bytes on Apple Silicon).

`ResidencySet` (`backend/metal/resident.cpp`): on macOS 15+, pins hot buffers in GPU-accessible memory to reduce paging overhead.

#### Command Buffer Batching

The Metal device (`backend/metal/device.cpp`) batches multiple kernel dispatches into one `MTLCommandBuffer`. A commit is triggered when `buffer_ops > max_ops_per_buffer_` (20–50 depending on GPU tier) or `buffer_sizes > max_mb_per_buffer_` (40–50 MB). Override via `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` env vars.

`CommandEncoder` tracks input/output buffer sets per dispatch. If an input buffer was a previous output (`prev_outputs_`), it sets `needs_barrier_` and inserts `memoryBarrier(BarrierScopeBuffers)` before the next dispatch (RAW hazard detection).

#### Lazy Evaluation and Memory Pressure

`eval()` (`mlx/transforms.cpp`): DFS dependency resolution → BFS buffer allocation and dispatch. When `active_memory > memory_limit` and tasks are in flight, the evaluator calls `scheduler::wait_for_one()` in a loop before allocating more buffers.

Memory API (`mlx/memory.h`): `set_memory_limit()`, `set_cache_limit()`, `set_wired_limit()`, `clear_cache()`.

#### File Loading (Not mmap)

MLX does **not** mmap model files. Safetensors tensor offsets are arbitrary byte positions, not page-aligned, making `newBufferWithBytesNoCopy` impractical per-tensor. Instead, the `Load` primitive (`backend/common/load.cpp`) dispatches async `pread()` calls (via `ParallelFileReader` in `mlx/io/load.cpp`) into Metal-allocated buffers on eval. GGUF has 32-byte alignment—closer to zero-copy viable, but MLX still uses read-into-buffer.

#### Online Resources

**MLX documentation:**
- [Unified Memory guide](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) — official MLX docs on the unified memory model
- [Metal memory API (Python)](https://ml-explore.github.io/mlx/build/html/python/metal.html) — `set_memory_limit`, `set_cache_limit`, etc.

**WWDC sessions:**
- [Get started with MLX for Apple silicon (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/315/) — covers unified memory, lazy computation, function transforms
- [Explore large language models on Apple silicon with MLX (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/298/) — inference and fine-tuning on Mac
- [Harness Apple GPUs with Metal (WWDC20)](https://developer.apple.com/videos/play/wwdc2020/10602/) — foundational UMA concepts, storage modes
- [Metal Compute on MacBook Pro (Tech Talk)](https://developer.apple.com/videos/play/tech-talks/10580/) — Apple GPU compute architecture

**Apple Metal documentation:**
- [Choosing a resource storage mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus) — why `StorageModeShared` is the right choice on Apple Silicon
- [MTLStorageMode.shared](https://developer.apple.com/documentation/metal/mtlstoragemode/shared) — API reference
- [Metal Best Practices: Resource Options](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/ResourceOptions.html) — when to use shared vs managed vs private
- [Simplifying GPU resource management with residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets) — `MTLResidencySet` API that MLX uses on macOS 15+

**Apple Silicon cache architecture:**
- [EXAM: Exploiting Exclusive SLC in Apple M-Series SoCs (arXiv:2504.13385)](https://arxiv.org/abs/2504.13385) — academic paper reverse-engineering SLC inclusiveness policy (exclusive to CPU, inclusive to GPU)
- [iGPU Cache Setups Compared, Including M1 (Chips and Cheese)](https://chipsandcheese.com/p/igpu-cache-setups-compared-including-m1) — independent SLC size/latency measurements (16 MB M1, 24 MB M1 Pro, 48 MB M1 Max)
- [Exploring LLMs with MLX and the M5 GPU (Apple ML Research)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) — M5 Neural Accelerators, memory bandwidth (153 GB/s), TensorOps integration

**Community / third-party:**
- [metal-usm (Philip Turner)](https://github.com/philipturner/metal-usm) — accessing CPU pointers from inside the Apple GPU; explores USM performance characteristics
- [metal-benchmarks (Philip Turner)](https://github.com/philipturner/metal-benchmarks) — Apple GPU microarchitecture benchmarks
- [Benchmarking On-Device ML on Apple Silicon with MLX (arXiv:2510.18921)](https://arxiv.org/abs/2510.18921) — systematic MLX performance benchmarks across M-series chips
- [Safetensors format (Hugging Face)](https://huggingface.co/docs/safetensors/en/index) — format spec; explains why tensor offsets aren't page-aligned
- [Speeding up Model Loading with fastsafetensors (arXiv:2505.23072)](https://arxiv.org/html/2505.23072v1) — GPU-direct loading approaches and alignment challenges

### Adding a New Primitive

1. Declare the primitive class in `mlx/primitives.h` inheriting from `Primitive` (or `UnaryPrimitive`).
2. Implement `eval_cpu()` in `mlx/backend/cpu/` and `eval_gpu()` in `mlx/backend/metal/` and/or `mlx/backend/cuda/`.
3. Add the op function in `mlx/ops.h` / `mlx/ops.cpp`.
4. Implement `jvp()`/`vjp()` in `mlx/primitives.cpp` for autodiff support.
5. Add Python bindings in `python/src/ops.cpp`.
6. Add Metal kernel in `mlx/backend/metal/kernels/` if needed.

### Distributed

`mlx/distributed/` supports MPI, NCCL, JACCL (Apple), and a ring-based backend. Python entry points: `mlx.launch` and `mlx.distributed_config`.

### Serialization

`mlx/io/` supports safetensors and GGUF formats. Loading is lazy — file headers are parsed first, data is read into Metal buffers on eval.

## CI

GitHub Actions runs on push to `main` and PRs: lint check (pre-commit), Linux CPU (x86_64 + aarch64), CUDA (12.6 + 12.9), macOS (SDK 14/15/26), Windows CPU, sanitizers (ASan, UBSan), and documentation build.
