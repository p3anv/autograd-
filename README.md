# AutoGrad++

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/autograd-tensor-library/actions)
[![codecov](https://img.shields.io/codecov/c/github/yourusername/autograd-tensor-library)](https://codecov.io/gh/yourusername/autograd-tensor-library)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://yourusername.github.io/autograd-tensor-library)

A high-performance, CPU-optimized N-dimensional tensor library with reverse-mode automatic differentiation (autograd) implemented from scratch in modern C++20.

---

## 🚀 Overview

AutoGrad++ is designed to train a Multi-Layer Perceptron (MLP) on MNIST with >95% accuracy **without** linking to any third‑party linear algebra frameworks (Eigen, BLAS, PyTorch, TensorFlow). Performance is achieved through optional cache‑aware memory tiling and SIMD vectorization, with a fallback scalar reference implementation guaranteeing correctness on any platform.

---

## ✨ Features

- **N‑dimensional tensors** (up to 8 dimensions)
- **Reverse‑mode automatic differentiation**
- **Broadcasting support** (forward and backward)
- **Elementwise operations**: Add, Sub, Mul, Div, Pow
- **Unary operations**: Neg, Exp, Log, Sqrt, Abs
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax
- **Matrix multiplication** (MatMul) with scalar reference and optional AVX2‑optimized kernels
- **Reductions**: Sum, Mean, Max, Min
- **Normalization**: LogSoftmax
- **Zero‑copy views**: slice, narrow, reshape
- **Contiguous memory layout** with explicit strides and offsets
- **64‑byte aligned memory allocations** for SIMD readiness
- **Comprehensive test suite** with Google Test
- **Performance benchmarks** with Google Benchmark
- **Memory safety** with AddressSanitizer and UndefinedBehaviorSanitizer
- **Deterministic builds** and reproducibility

---

## 📦 Installation & Build

### Prerequisites

- Ubuntu 22.04 or 24.04 (Linux x86_64) – other Linux distributions work similarly
- GCC 13+ **or** Clang 16+
- CMake 3.20+
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/autograd-tensor-library.git
cd autograd-tensor-library

# Create and enter build directory
mkdir -p build && cd build

# Configure (Debug build with sanitizers recommended for development)
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON

# Build library and tests
cmake --build .

# Run the test suite
ctest --output-on-failure

# Run benchmarks (if enabled)
./autograd_benchmarks
```

### Build Options

| Option                              | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `-DCMAKE_BUILD_TYPE=Debug`          | Debug symbols + sanitizers (default for development)                       |
| `-DCMAKE_BUILD_TYPE=Release`        | Optimized for performance                                                   |
| `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | Optimized with debug symbols (suitable for profiling)                      |
| `-DENABLE_SANITIZERS=ON`            | Enable AddressSanitizer & UndefinedBehaviorSanitizer (highly recommended)  |
| `-DWITH_AVX2=ON`                    | Enable AVX2‑optimized kernels (requires AVX2‑capable CPU)                 |
| `-DBUILD_BENCHMARKS=ON`             | Build benchmark targets                                                     |
| `-DBUILD_EXAMPLES=ON`               | Build example programs (if any)                                             |

---

## 📖 Documentation

- [Product Requirements Document](prd.md)
- [Design Document](design.md)
- [Implementation Phases](phases.md)
- [Development Rules](rules.md)
- [API Reference](https://yourusername.github.io/autograd-tensor-library) *(generated with Doxygen)*

---

## 🧪 Testing

The library uses **Google Test** for unit testing. Tests cover:

- Tensor core functionality (Storage, TensorImpl, Tensor)
- Views and indexing (slice, narrow, reshape, contiguous)
- Forward operations (elementwise, unary, activations, MatMul, reductions)
- Autograd engine (gradient checks, DAG traversal, shared intermediates)
- Memory safety (leaks, buffer overflows, use‑after‑free)
- End‑to‑end validation (MNIST training >95% accuracy)

Run tests with:

```bash
cd build && ctest --output-on-failure
```

---

## 🔧 Continuous Integration

CI pipelines run on Ubuntu 22.04 and 24.04 with GCC 13 and Clang 16, including:

- Build with `-Wall -Wextra -Wpedantic -Werror`
- Test execution with Google Test
- Sanitizer checks (AddressSanitizer, UndefinedBehaviorSanitizer)
- Valgrind memcheck for memory leaks
- Optional benchmark runs
- Coverage reporting with `gcovr` + Codecov

---

## 📚 Usage Example

Here’s a minimal example that builds a simple 2‑layer network and runs a forward pass:

```cpp
#include "autograd/tensor.hpp"
#include "autograd/operations.hpp"
#include <iostream>

int main() {
    using namespace autograd;

    // Create a batch of 4 samples, each with 3 features
    Tensor x({4, 3}, true); // requires_grad = true
    x.randn();              // random initialization

    // First linear layer: 3 -> 10
    Tensor W1({10, 3}, true);
    Tensor b1({10}, true);
    W1.randn()*0.1;
    b1.fill(0.0f);

    Tensor z1 = add(matmul(x, W1.transpose()), b1);
    Tensor a1 = relu(z1);    // activation

    // Second linear layer: 10 -> 1 (binary classification)
    Tensor W2({1, 10}, true);
    Tensor b2({1}, true);
    W2.randn()*0.1;
    b2.fill(0.0f);

    Tensor logits = add(matmul(a1, W2.transpose()), b2);
    Tensor probs = sigmoid(logits);

    std::cout << "Probabilities:\n" << probs << std::endl;

    // Dummy loss: binary cross‑entropy
    Tensor target({4, 1});
    target.fill(1.0f); // all ones
    Tensor loss = binary_cross_entropy(probs, target);
    loss.backward();    // compute gradients

    std::cout << "Loss: " << loss.item() << std::endl;
    std::cout << "Gradient norm of W1: " << W1.grad()->norm().item() << std::endl;

    return 0;
}
```

Compile with:

```bash
g++ -std=c++20 example.cpp -I./include -L./build -lautograd -o example -pthread
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by PyTorch’s autograd system and NumPy’s broadcasting rules.
- Built with modern C++20 best practices: RAII, smart pointers, value‑semantic tensor handles.
- Thanks to the open‑source community for Google Test, Google Benchmark, and Catch2.
- Special thanks to JetBrains for providing CLion licenses for open‑source projects.

---

**Happy tensor hacking!** 🚀