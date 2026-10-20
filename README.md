# AutoGrad++

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![Tests](https://img.shields.io/badge/tests-128%20passing-brightgreen)](#-testing)
[![Memory Safe](https://img.shields.io/badge/valgrind-clean-brightgreen)](#-memory-safety)
[![AVX2](https://img.shields.io/badge/SIMD-AVX2%2FFMA-orange)](#-simd-optimization)

A high-performance, CPU-optimized N-dimensional tensor library with reverse-mode automatic differentiation (autograd) implemented from scratch in modern C++20. Trains an MLP on MNIST to **96.66% accuracy** without any third-party ML frameworks.

---

## 🚀 Overview

AutoGrad++ provides a PyTorch-like tensor API with full autograd capabilities, built entirely from scratch:

- **Zero dependencies** — no Eigen, BLAS, PyTorch, or TensorFlow
- **SIMD accelerated** — AVX2/FMA kernels with runtime CPU dispatch
- **Memory safe** — RAII, 64-byte aligned allocations, Valgrind/ASan/UBSan clean
- **Production ready** — 128 tests passing, MNIST validated at >95% accuracy

---

## ✨ Features

### Tensor Core
- N-dimensional tensors (up to 8 dimensions)
- Zero-copy views: `slice()`, `narrow()`, `reshape()`, `transpose()`
- Contiguous memory layout with explicit strides and offsets
- 64-byte aligned memory allocations for SIMD readiness
- Broadcasting following NumPy/PyTorch semantics

### Operations
| Category | Operations |
|----------|------------|
| Elementwise Binary | `add`, `sub`, `mul`, `div`, `pow` (with broadcasting) |
| Elementwise Unary | `neg`, `exp`, `log`, `sqrt`, `abs` |
| Activations | `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax` |
| Reductions | `sum`, `mean`, `max`, `min` (with dimension parameter) |
| Linear Algebra | `matmul` (rank-2, contiguous-only) |

### Autograd Engine
- Reverse-mode automatic differentiation with computational graph
- Dynamic DAG construction during forward pass
- Topological sort + gradient accumulation in backward pass
- Proper broadcasting support in backward pass
- Gradient accumulation in leaf tensors

### SIMD Optimization (AVX2/FMA)
| Operation | Implementation |
|-----------|----------------|
| Elementwise | 8-wide vectorized + broadcast variants |
| Unary (exp, log, sqrt) | Polynomial approximations (1e-4 rel. error) |
| Activations | Vectorized exp-based formulas |
| Reductions | Horizontal hadd tree |
| Softmax/LogSoftmax | Fused multi-pass (max → exp/sum → normalize) |
| MatMul | Blocked (MC=64, NC=64, KC=256) + 8×8 FMA microkernel |

---

## 📦 Installation & Build

### Prerequisites
- **OS**: Ubuntu 22.04/24.04 (Linux x86_64)
- **Compiler**: GCC 13+ or Clang 16+
- **Tools**: CMake 3.20+, Git

### Quick Start
```bash
# Clone
git clone https://github.com/yourusername/autograd-tensor-library.git
cd autograd-tensor-library

# Build (Debug with sanitizers)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Run MNIST training (requires BUILD_EXAMPLES=ON)
cmake .. -DBUILD_EXAMPLES=ON
cmake --build . -j$(nproc)
./mnist_train
```

### Build Options
| Option | Description | Default |
|--------|-------------|---------|
| `-DCMAKE_BUILD_TYPE=Debug` | Debug symbols + sanitizers | ✅ Dev |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized for performance | |
| `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | Optimized with debug symbols | |
| `-DENABLE_SANITIZERS=ON` | AddressSanitizer + UBSan | ON (Debug) |
| `-DWITH_AVX2=ON` | Enable AVX2-optimized kernels | OFF |
| `-DBUILD_BENCHMARKS=ON` | Build benchmark targets | OFF |
| `-DBUILD_EXAMPLES=ON` | Build example programs | OFF |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AutoGrad++                              │
├─────────────────────────────────────────────────────────────┤
│  User API: Tensor, Operations, Autograd                     │
├─────────────────────────────────────────────────────────────┤
│  Tensor Core                                                │
│  ├── Storage (aligned raw memory)                           │
│  ├── TensorImpl (shape, strides, offset, autograd metadata) │
│  └── Tensor (value-semantic handle, shared_ptr<TensorImpl>) │
├─────────────────────────────────────────────────────────────┤
│  Views & Indexing (zero-copy)                               │
│  ├── slice(), narrow(), reshape(), transpose()              │
│  └── contiguous() for SIMD fast-path                        │
├─────────────────────────────────────────────────────────────┤
│  Operations                                                 │
│  ├── Elementwise (broadcasting)                             │
│  ├── Unary & Activations                                    │
│  ├── Reductions (dim-aware)                                 │
│  └── MatMul (blocked + SIMD)                                │
├─────────────────────────────────────────────────────────────┤
│  Autograd Engine                                            │
│  ├── FunctionNode hierarchy (AddBackward, MatMulBackward…) │
│  ├── AutogradEngine (DAG + topological sort)                │
│  └── Tensor::backward() → gradient accumulation             │
├─────────────────────────────────────────────────────────────┤
│  SIMD Layer                                                 │
│  ├── cpu_capabilities (CPUID runtime detection)             │
│  ├── dispatch (ScalarTag/SseTag/Avx2Tag)                    │
│  ├── kernels_scalar.cpp (reference)                         │
│  └── kernels_avx2.cpp (AVX2+FMA optimized)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Usage Example

```cpp
#include "autograd/tensor.hpp"
#include "autograd/operations.hpp"
#include <iostream>

int main() {
    using namespace autograd;

    // Create a batch of 4 samples, each with 3 features
    Tensor x({4, 3}, true);  // requires_grad = true
    x.randn();               // random initialization

    // First linear layer: 3 -> 10
    Tensor W1({10, 3}, true);
    Tensor b1({10}, true);
    W1.randn() * 0.1f;
    b1.fill(0.0f);

    Tensor z1 = add(matmul(x, W1.transpose()), b1);
    Tensor a1 = relu(z1);    // activation

    // Second linear layer: 10 -> 1 (binary classification)
    Tensor W2({1, 10}, true);
    Tensor b2({1}, true);
    W2.randn() * 0.1f;
    b2.fill(0.0f);

    Tensor logits = add(matmul(a1, W2.transpose()), b2);
    Tensor probs = sigmoid(logits);

    std::cout << "Probabilities:\n" << probs << std::endl;

    // Binary cross-entropy loss
    Tensor target({4, 1});
    target.fill(1.0f);
    Tensor loss = binary_cross_entropy(probs, target);
    loss.backward();  // compute gradients

    std::cout << "Loss: " << loss.item() << std::endl;
    std::cout << "Gradient norm of W1: " << W1.grad()->norm().item() << std::endl;

    return 0;
}
```

**Compile:**
```bash
g++ -std=c++20 example.cpp -I./include -L./build -lautograd -o example -pthread
```

---

## 🧪 Testing

The library uses **Google Test** with comprehensive coverage:

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| `StorageTest` | 4 | Memory management |
| `TensorImplTest` | 14 | Views, strides, indexing |
| `TensorTest` | 32 | Operations, broadcasting |
| `AutogradTest` | 21 | Gradient correctness, DAG |
| `HardcoreTest` | 58 | Mathematical properties, edge cases |
| **Total** | **128** | **100% passing** |

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run specific test suites
./autograd_tests
./hardcore_tests

# Memory leak check
cd ../build_valgrind && valgrind --leak-check=full ./autograd_tests
```

### Numerical Validation
All AVX2 kernels validated against scalar reference:
- **Exact operations** (add, mul): 0 tolerance
- **Approximate** (exp, log, sigmoid, tanh): ≤ 1e-4 relative error
- **MatMul** (various M,N,K): ≤ 1e-4 relative error

---

## 🔬 Memory Safety

| Tool | Status |
|------|--------|
| **AddressSanitizer** | ✅ Clean (no buffer overflows, use-after-free) |
| **UndefinedBehaviorSanitizer** | ✅ Clean |
| **Valgrind Memcheck** | ✅ Zero leaks (core library) |
| **ThreadSanitizer** | ⏳ Not yet tested |

---

## 📊 Benchmarks (Enable with `-DBUILD_BENCHMARKS=ON`)

```bash
cmake .. -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release -DWITH_AVX2=ON
cmake --build . -j$(nproc)
./autograd_benchmarks
```

**Expected Performance (AVX2 vs Scalar):**

| Operation | Speedup | Notes |
|-----------|---------|-------|
| Elementwise add/mul | 4-8× | 8 floats/YMM register |
| Reductions (sum) | 4-6× | Horizontal hadd tree |
| Softmax | 3-5× | Vectorized max/exp/sum |
| MatMul (≥256³) | 10-20× | Blocked + 8×8 FMA microkernel |
| MatMul (<64³) | 2-4× | Overhead dominates |

---

## 📁 Project Structure

```
autograd-tensor-library/
├── include/autograd/
│   ├── tensor.hpp           # Public API
│   ├── operations.hpp       # Forward operations
│   ├── autograd_engine.hpp  # Backward engine
│   ├── function_nodes.hpp   # FunctionNode hierarchy
│   ├── internal.hpp         # Internal helpers
│   ├── cpu_capabilities.hpp # CPUID detection
│   ├── dispatch.hpp         # SIMD tag dispatch
│   ├── kernels.hpp          # Scalar kernel declarations
│   └── kernels_avx2.hpp     # AVX2 kernel declarations
├── src/
│   ├── tensor.cpp           # Tensor implementation
│   ├── tensor_impl.cpp      # TensorImpl (views, strides)
│   ├── storage.cpp          # Aligned memory management
│   ├── operations.cpp       # All forward ops + SIMD dispatch
│   ├── autograd_engine.cpp  # DAG construction, topo sort
│   ├── function_nodes.cpp   # Backward implementations
│   └── simd/
│       ├── cpu_capabilities.cpp
│       ├── kernels_scalar.cpp    # Reference implementations
│       └── kernels_avx2.cpp      # AVX2+FMA optimized (600+ lines)
├── tests/
│   ├── test_storage.cpp
│   ├── test_tensor_impl.cpp
│   ├── test_tensor.cpp
│   ├── test_autograd.cpp
│   └── hardcore_test.cpp        # Comprehensive validation
├── benchmarks/                  # Google Benchmark (optional)
├── examples/                    # Usage examples (optional)
│   ├── mnist_train.cpp
│   └── mnist_model.cpp
├── CMakeLists.txt
└── LICENSE
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code of conduct
- Development workflow
- Coding standards (C++20, RAII, value semantics)
- Testing requirements
- Pull request process

---

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **PyTorch** – autograd system design inspiration
- **NumPy** – broadcasting rules
- **Google Test / Benchmark** – testing infrastructure
- **JetBrains** – CLion licenses for open-source development
- **C++ Community** – modern C++20 best practices

---

**Happy tensor hacking!** 🚀

*AutoGrad++ – From scalar to SIMD, all in C++20.*