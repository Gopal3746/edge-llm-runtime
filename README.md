# Edge LLM Runtime

A lightweight CPU transformer inference runtime implemented in modern C++.

The project is intended to explore the components behind edge-oriented
large-language-model inference, including tensor operations, transformer
layers, KV caching, quantization, benchmarking, and memory optimization.

## Current Status

The runtime provides an end-to-end decoder-only transformer inference path
with greedy generation. It now also includes fixed-capacity per-layer KV-cache
storage that preallocates contiguous key and value buffers, supports validated
multi-token appends, and resets logical state without reallocating memory.
Cached attention integration is the next milestone.

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
