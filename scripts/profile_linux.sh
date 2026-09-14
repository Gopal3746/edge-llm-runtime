#!/usr/bin/env bash

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This profiling script requires Linux."
    exit 1
fi

if ! command -v perf >/dev/null 2>&1; then
    echo "perf is required but was not found."
    echo "On Ubuntu, install it with:"
    echo "  sudo apt-get install linux-tools-common linux-tools-generic"
    exit 1
fi

project_root="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)"

build_directory="${project_root}/out/profile"
result_directory="${project_root}/results/profiling"

mkdir -p "${result_directory}"

cmake \
    -S "${project_root}" \
    -B "${build_directory}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DEDGE_LLM_BUILD_TESTS=ON \
    -DEDGE_LLM_BUILD_BENCHMARKS=ON \
    -DEDGE_LLM_WARNINGS_AS_ERRORS=ON

cmake \
    --build "${build_directory}" \
    --parallel

ctest \
    --test-dir "${build_directory}" \
    --output-on-failure

perf stat \
    -d \
    -r 5 \
    -o "${result_directory}/perf_stat.txt" \
    "${build_directory}/edge_llm_benchmark"

perf record \
    -g \
    -o "${result_directory}/perf.data" \
    "${build_directory}/edge_llm_benchmark"

perf report \
    --stdio \
    --sort=dso,symbol \
    -i "${result_directory}/perf.data" \
    > "${result_directory}/perf_report.txt"

echo
echo "Profiling complete:"
echo "  ${result_directory}/perf_stat.txt"
echo "  ${result_directory}/perf_report.txt"
