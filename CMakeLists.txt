cmake_minimum_required(VERSION 3.20)

project(autograd-tensor-library
    LANGUAGES CXX
    VERSION 1.0.0
    DESCRIPTION "C++ Production-Grade Autograd Tensor Library"
    HOMEPAGE_URL ""
)

# Set C++ standard properties
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project directories
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Options
option(BUILD_BENCHMARKS "Build benchmark targets" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers in Debug builds" ON)
option(WITH_AVX2 "Enable AVX2 optimized kernels" OFF)
option(BUILD_TESTING "Build testing targets" ON)  # Now enabled as we have tests
option(BUILD_EXAMPLES "Build example targets" OFF)

# Compiler warnings and sanitizers
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Base warning flags as a list
    set(BASE_WARNING_FLAGS -Wall -Wextra -Wpedantic)

    # Sanitizer flags for Debug builds
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        if(ENABLE_SANITIZERS)
            set(SANITIZER_FLAGS -fsanitize=address,undefined)
            # Add -Werror when sanitizers are enabled in Debug
            list(APPEND COMPILE_FLAGS ${BASE_WARNING_FLAGS} ${SANITIZER_FLAGS} -Werror)
        else()
            set(COMPILE_FLAGS ${BASE_WARNING_FLAGS})
        endif()
    else()
        # Release/RelWithDebInfo builds - just warnings, no -Werror
        set(COMPILE_FLAGS ${BASE_WARNING_FLAGS})
    endif()
else()
    # For other compilers, just use basic flags
    set(COMPILE_FLAGS -Wall -Wextra -Wpedantic)
endif()

# Set compiler flags globally
add_compile_options(${COMPILE_FLAGS})

# AVX2 definition
if(WITH_AVX2)
    add_compile_definitions(WITH_AVX2)
    message(STATUS "AVX2 optimizations enabled")
else()
    message(STATUS "AVX2 optimizations disabled (scalar reference only)")
endif()

# Library target - sources will be added as we implement them
add_library(autograd
    src/storage.cpp
    src/tensor_impl.cpp
    src/tensor.cpp
    src/operations.cpp
    src/autograd_engine.cpp
    src/function_nodes.cpp
)

target_include_directories(autograd
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# Create a comprehensive test target with gtest
if(BUILD_TESTING)
    # Find GTest
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    add_executable(autograd_tests
        tests/test_storage.cpp
        tests/test_tensor_impl.cpp
        tests/test_tensor.cpp
        tests/hardcore_test.cpp
    )

    target_include_directories(autograd_tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
    )

    target_link_libraries(autograd_tests
        PRIVATE
            autograd
            ${GTEST_LIBRARIES}
            gtest_main
            pthread
    )

    enable_testing()
    add_test(NAME autograd_tests COMMAND autograd_tests)
endif()

# Benchmark target placeholder
if(BUILD_BENCHMARKS)
    message(STATUS "Benchmark targets will be implemented in later phases")
endif()

# Examples target placeholder
if(BUILD_EXAMPLES)
    message(STATUS "Example targets will be implemented in later phases")
endif()

# Install rules
install(TARGETS autograd
        EXPORT autogradTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT autogradTargets
        FILE autogradTargets.cmake
        NAMESPACE autograd::
        DESTINATION lib/cmake/autograd)

# Configure a header file for version information
configure_file(
    "${PROJECT_SOURCE_DIR}/include/autograd/version.h.in"
    "${PROJECT_BINARY_DIR}/include/autograd/version.h"
)

target_include_directories(autograd
    PRIVATE
        ${PROJECT_BINARY_DIR}/include
)

# Print build configuration summary
message(STATUS "Build configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Sanitizers: ${ENABLE_SANITIZERS}")
message(STATUS "  AVX2: ${WITH_AVX2}")
message(STATUS "  Testing: ${BUILD_TESTING}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")