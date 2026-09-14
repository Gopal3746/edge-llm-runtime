# Linux CPU Profiling

The runtime includes two Linux profiling paths:

1. A portable GitHub Actions workflow using GNU `time` and Valgrind Callgrind.
2. A local Linux script using `perf stat` and `perf record`.

## Profiling build

Profiling uses `RelWithDebInfo`, which retains debugging symbols while enabling
compiler optimization.

```bash
cmake -S . -B out/profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DEDGE_LLM_BUILD_TESTS=ON \
  -DEDGE_LLM_BUILD_BENCHMARKS=ON \
  -DEDGE_LLM_WARNINGS_AS_ERRORS=ON

cmake --build out/profile
