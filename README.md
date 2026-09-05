# Edge LLM Runtime

A lightweight CPU transformer inference runtime implemented in modern C++.

The project is intended to explore the components behind edge-oriented
large-language-model inference, including tensor operations, transformer
layers, KV caching, quantization, benchmarking, and memory optimization.

## Current Status

The runtime currently provides a CMake-based C++20 foundation, an owning FP32
tensor type with contiguous storage and checked indexing, core tensor
arithmetic, numerically stable softmax, and RMSNorm applied over the final
tensor dimension.

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