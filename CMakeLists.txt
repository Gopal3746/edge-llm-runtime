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
    src/layers.cpp
    src/nn.cpp
    src/ops.cpp
    src/rope.cpp
    src/tensor.cpp
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

include(CTest)

if(BUILD_TESTING AND EDGE_LLM_BUILD_TESTS)
    add_executable(
        edge_llm_smoke_test
        tests/smoke_test.cpp
    )

    target_link_libraries(
        edge_llm_smoke_test
        PRIVATE
        edge_llm
    )

    add_test(
        NAME edge_llm_smoke_test
        COMMAND edge_llm_smoke_test
    )

    add_executable(
        edge_llm_tensor_test
        tests/tensor_test.cpp
    )

    target_link_libraries(
        edge_llm_tensor_test
        PRIVATE
        edge_llm
    )

    add_test(
        NAME edge_llm_tensor_test
        COMMAND edge_llm_tensor_test
    )

    add_executable(
        edge_llm_ops_test
        tests/ops_test.cpp
    )

    target_link_libraries(
        edge_llm_ops_test
        PRIVATE
        edge_llm
    )

    add_test(
        NAME edge_llm_ops_test
        COMMAND edge_llm_ops_test
    )

    add_executable(
        edge_llm_nn_test
        tests/nn_test.cpp
    )

    target_link_libraries(
        edge_llm_nn_test
        PRIVATE
        edge_llm
    )

    add_test(
        NAME edge_llm_nn_test
        COMMAND edge_llm_nn_test
    )

        add_executable(
        edge_llm_layers_test
        tests/layers_test.cpp
    )

    target_link_libraries(
        edge_llm_layers_test
        PRIVATE
        edge_llm
    )

    add_test(
        NAME edge_llm_layers_test
        COMMAND edge_llm_layers_test
    )

        add_executable(
        edge_llm_rope_test
        tests/rope_test.cpp
    )

    target_link_libraries(
        edge_llm_rope_test
        PRIVATE
        edge_llm
    )

    add_test(
        NAME edge_llm_rope_test
        COMMAND edge_llm_rope_test
    )
        add_executable(
        edge_llm_attention_test
        tests/attention_test.cpp
    )

    target_link_libraries(
        edge_llm_attention_test
        PRIVATE
        edge_llm
    )

    add_test(
        NAME edge_llm_attention_test
        COMMAND edge_llm_attention_test
    )
endif()