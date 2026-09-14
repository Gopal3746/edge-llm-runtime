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

option(
    EDGE_LLM_WARNINGS_AS_ERRORS
    "Treat compiler warnings as errors"
    OFF
)

option(
    EDGE_LLM_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF
)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Shared compiler options
# -----------------------------------------------------------------------------

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

if(EDGE_LLM_WARNINGS_AS_ERRORS)
    if(MSVC)
        target_compile_options(
            edge_llm_options
            INTERFACE
            /WX
        )
    else()
        target_compile_options(
            edge_llm_options
            INTERFACE
            -Werror
        )
    endif()
endif()

if(EDGE_LLM_ENABLE_SANITIZERS)
    if(MSVC)
        message(
            FATAL_ERROR
            "EDGE_LLM_ENABLE_SANITIZERS is not supported with MSVC"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(
            edge_llm_options
            INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )

        target_link_options(
            edge_llm_options
            INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
    else()
        message(
            FATAL_ERROR
            "Sanitizers require Clang, AppleClang, or GCC"
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# Runtime library
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Command-line application
# -----------------------------------------------------------------------------

add_executable(
    edge_llm_cli
    apps/main.cpp
)

target_link_libraries(
    edge_llm_cli
    PRIVATE
    edge_llm
)

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

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

    add_executable(
        edge_llm_buffer_reuse_benchmark
        benchmarks/buffer_reuse_benchmark.cpp
    )

    target_link_libraries(
        edge_llm_buffer_reuse_benchmark
        PRIVATE
        edge_llm
    )
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

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
