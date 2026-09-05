cmake_minimum_required(VERSION 3.20)

project(
    edge_llm_runtime
    VERSION 0.1.0
    DESCRIPTION "Lightweight CPU transformer inference runtime"
    LANGUAGES CXX
)

option(EDGE_LLM_BUILD_TESTS "Build Edge LLM Runtime tests" ON)

add_library(edge_llm_options INTERFACE)

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

add_executable(
    edge_llm_cli
    apps/main.cpp
)

target_link_libraries(
    edge_llm_cli
    PRIVATE
    edge_llm_options
)

target_include_directories(
    edge_llm_cli
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
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
        edge_llm_options
    )

    target_include_directories(
        edge_llm_smoke_test
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME edge_llm_smoke_test
        COMMAND edge_llm_smoke_test
    )
endif()