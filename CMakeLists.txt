cmake_minimum_required(VERSION 3.20)

project(
    edge_llm_runtime
    VERSION 0.1.0
    DESCRIPTION "Lightweight CPU transformer inference runtime"
    LANGUAGES CXX
)

option(
    EDGE_LLM_BUILD_TESTS
    "Build Edge LLM Runtime tests"
    ON
)

option(
    EDGE_LLM_BUILD_BENCHMARKS
    "Build Edge LLM Runtime benchmarks"
    ON
)

set(
    CMAKE_CXX_EXTENSIONS
    OFF
)

add_library(
    edge_llm_options
    INTERFACE
)

target_compile_features(
    edge_llm_options
    INTERFACE
    cxx_std_20
)

if(MSVC)
    target_compile_options(
        edge_llm_options
        INTERFACE
        /W4
    )
else()
    target_compile_options(
        edge_llm_options
        INTERFACE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_library(
    edge_llm
    src/attention.cpp
    src/feed_forward.cpp
    src/kv_cache.cpp
    src/layers.cpp
    src/model.cpp
    src/nn.cpp
    src/ops.cpp
    src/quantization.cpp
    src/rope.cpp
    src/tensor.cpp
    src/transformer.cpp
)

target_link_libraries(
    edge_llm
    PUBLIC
    edge_llm_options
)

target_include_directories(
    edge_llm
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(
    edge_llm_cli
    apps/main.cpp
)

target_link_libraries(
    edge_llm_cli
    PRIVATE
    edge_llm
)

if(EDGE_LLM_BUILD_BENCHMARKS)
    add_executable(
        edge_llm_benchmark
        benchmarks/benchmark.cpp
    )

    target_link_libraries(
        edge_llm_benchmark
        PRIVATE
        edge_llm
    )
endif()

include(CTest)

if(BUILD_TESTING AND EDGE_LLM_BUILD_TESTS)
    function(edge_llm_add_test target_name source_file)
        add_executable(
            ${target_name}
            ${source_file}
        )

        target_link_libraries(
            ${target_name}
            PRIVATE
            edge_llm
        )

        add_test(
            NAME ${target_name}
            COMMAND ${target_name}
        )
    endfunction()

    edge_llm_add_test(
        edge_llm_smoke_test
        tests/smoke_test.cpp
    )

    edge_llm_add_test(
        edge_llm_tensor_test
        tests/tensor_test.cpp
    )

    edge_llm_add_test(
        edge_llm_ops_test
        tests/ops_test.cpp
    )

    edge_llm_add_test(
        edge_llm_nn_test
        tests/nn_test.cpp
    )

    edge_llm_add_test(
        edge_llm_layers_test
        tests/layers_test.cpp
    )

    edge_llm_add_test(
        edge_llm_rope_test
        tests/rope_test.cpp
    )

    edge_llm_add_test(
        edge_llm_attention_test
        tests/attention_test.cpp
    )

    edge_llm_add_test(
        edge_llm_transformer_test
        tests/transformer_test.cpp
    )

    edge_llm_add_test(
        edge_llm_model_test
        tests/model_test.cpp
    )

    edge_llm_add_test(
        edge_llm_kv_cache_test
        tests/kv_cache_test.cpp
    )

    edge_llm_add_test(
        edge_llm_cached_attention_test
        tests/cached_attention_test.cpp
    )

    edge_llm_add_test(
        edge_llm_cached_transformer_test
        tests/cached_transformer_test.cpp
    )

    edge_llm_add_test(
        edge_llm_cached_model_test
        tests/cached_model_test.cpp
    )

    edge_llm_add_test(
        edge_llm_quantization_test
        tests/quantization_test.cpp
    )
endif()
