# Edge LLM Runtime

A lightweight CPU transformer inference runtime implemented in modern C++.

The project is intended to explore the components behind edge-oriented
large-language-model inference, including tensor operations, transformer
layers, KV caching, quantization, benchmarking, and memory optimization.

## Current Status

The runtime now supports uncached causal attention and KV-cached attention
through separate prefill and single-token decode paths. Prefill projects and
caches the prompt's rotated keys and values, while decode projects only the
new token and attends it against all cached history. Cached outputs are tested
for numerical agreement with full-sequence uncached attention.

## Planned Features

- Contiguous tensor storage and shape validation
- Matrix multiplication and element-wise operations
- RMSNorm and softmax
- Token embeddings and linear layers
- Rotary positional embeddings
- Causal self-attention
- Feed-forward and transformer blocks
- Autoregressive token generation
- KV caching with prefill and decode paths
- INT8 weight quantization
- FP32 and INT8 performance benchmarks
- Linux CPU profiling and buffer reuse

## Requirements

- C++20-compatible compiler
- CMake 3.20 or newer

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

## Reference Benchmark Results

The following results were collected using an Apple ARM64 system with Apple
Clang 21 and a Release build. The benchmark uses deterministic synthetic
weights and a compact two-block transformer configuration; the results are
intended to demonstrate runtime behavior rather than production-model
performance.

### FP32 versus weight-only INT8 linear execution

| Metric | FP32 | INT8 |
|---|---:|---:|
| Weight storage | 262,144 bytes | 66,560 bytes |
| Mean latency | 2.4085 ms | 1.9846 ms |
| Throughput | 13,286 rows/sec | 16,124 rows/sec |

Per-row INT8 quantization reduced weight storage by approximately 74.6%,
providing 3.94x compression. For this benchmark configuration, INT8 execution
was 1.21x faster than FP32. The maximum absolute output error was 0.0217 and
the mean absolute error was 0.0042.

### Uncached versus KV-cached generation

| Metric | Uncached | KV cached |
|---|---:|---:|
| Mean generation latency | 3.6798 ms | 0.5762 ms |
| Generated tokens/sec | 2,174 | 13,883 |
| Relative speedup | 1.00x | 6.39x |

The generation benchmark used a 16-token prompt, generated 8 new tokens, and
executed a two-block transformer with model dimension 32. Cached and uncached
execution generated identical token sequences. Isolated cached decode latency
averaged 0.0260 ms per token.

The complete machine-readable output is available in
`results/benchmark_results_macos_arm64.txt`.
