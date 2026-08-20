# C++ Production-Grade Autograd Tensor Library

A high-performance, CPU-optimized N-dimensional tensor library with reverse-mode automatic differentiation (autograd) implemented from scratch in modern C++20.

## Overview

This library is designed to train a Multi-Layer Perceptron (MLP) on MNIST with >95% accuracy without linking to any third-party linear algebra frameworks (Eigen, BLAS, PyTorch). Performance is achieved through optional cache-aware memory tiling and SIMD vectorization, with a fallback scalar reference implementation guaranteeing correctness on any platform.

## Features

- N-dimensional tensors (up to 8 dimensions)
- Reverse-mode automatic differentiation
- Broadcasting support (forward and backward)
- Elementwise operations (Add, Sub, Mul, Div, Pow)
- Unary operations (Neg, Exp, Log, Sqrt, Abs)
- Activation functions (ReLU, Sigmoid, Tanh)
- Matrix multiplication (MatMul) with scalar reference and optional AVX2-optimized kernels
- Reductions (Sum, Mean, Max, Min)
- Normalization (LogSoftmax)
- Zero-copy views (slice, narrow, reshape)
- Contiguous memory layout with explicit strides and offsets
- 64-byte aligned memory allocations for SIMD readiness
- Comprehensive test suite with Google Test
- Performance benchmarks with Google Benchmark
- Memory safety with AddressSanitizer and UndefinedBehaviorSanitizer
- Deterministic builds and reproducibility

## Building

### Prerequisites

- Ubuntu 22.04 or 24.04 (Linux x86_64)
- GCC 13+ or Clang 16+
- CMake 3.20+
- Git

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd autograd-tensor-library

# Create and enter build directory
mkdir build && cd build

# Configure with Debug build (includes sanitizers)
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON

# Build the library and tests
cmake --build .

# Run tests
ctest

# Run benchmarks (if enabled)
./autograd_benchmarks
```

### Build Options

- `-DCMAKE_BUILD_TYPE=Debug`: Build with debug symbols and sanitizers (default for development)
- `-DCMAKE_BUILD_TYPE=Release`: Build with optimizations for performance
- `-DCMAKE_BUILD_TYPE=RelWithDebInfo`: Build with optimizations and debug symbols (for profiling)
- `-DENABLE_SANITIZERS=ON`: Enable AddressSanitizer and UndefinedBehaviorSanitizer (recommended for Debug)
- `-DWITH_AVX2=ON`: Enable AVX2 optimized kernels (requires AVX2-capable CPU)
- `-DBUILD_BENCHMARKS=ON`: Build benchmark targets

## Documentation

- [Product Requirements Document](prd.md)
- [Design Document](design.md)
- [Implementation Phases](phases.md)
- [Development Rules](rules.md)

## Testing

The library uses Google Test for unit testing. Tests cover:
- Tensor core functionality (Storage, TensorImpl, Tensor)
- Views and indexing (slice, narrow, reshape, contiguous)
- Forward operations (elementwise, unary, activations, MatMul, reductions)
- Autograd engine (gradient checks, DAG traversal, shared intermediates)
- Memory safety (leaks, buffer overflows)
- End-to-end validation (MNIST training)

## Continuous Integration

CI pipeline runs on Ubuntu 22.04 and 24.04 with GCC 13 and Clang 16, including:
- Build with `-Wall -Wextra -Wpedantic -Werror`
- Tests with Google Test
- Sanitizer checks (AddressSanitizer, UndefinedBehaviorSanitizer)
- Valgrind memcheck for memory leaks
- Optional benchmark runs

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project follows best practices for modern C++ development, including:
- RAII resource management
- Smart pointers for ownership
- Value-semantic tensor handles
- Explicit autograd graph ownership model
- Separation of concerns between tensor metadata, storage, and operations# autograd-
