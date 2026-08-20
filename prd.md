# PRD: C++ Production-Grade Autograd Tensor Library

**Version:** 1.3 (Final – Semantics Locked)  
**Date:** 2026-08-17  
**Environment:** Ubuntu 22.04 / 24.04 (Linux x86_64)  
**Compiler:** GCC 13+ or Clang 16+

---

## 1. Vision Statement

Build a high-performance, CPU-optimized N-dimensional tensor library with reverse-mode automatic differentiation (autograd) from scratch in modern C++20. 

The library shall be capable of training a Multi-Layer Perceptron (MLP) on MNIST with >95% accuracy **without** linking to any third-party linear algebra frameworks (Eigen, BLAS, PyTorch). Performance will be achieved through optional cache-aware memory tiling and SIMD vectorization, with a fallback scalar reference implementation guaranteeing correctness on any platform.

---

## 2. Core Functional Requirements (The "MVP")

### 2.1 Memory & Tensor Core
- [ ] **Data Types**: Primary dtype is `float32`. `float64` support is optional but encouraged.
- [ ] **Custom Storage**: Tensor storage must provide at least **64-byte alignment** for allocations used by optimized kernels. The implementation must use an appropriate RAII allocator to guarantee this (e.g., `std::aligned_alloc` with a custom deleter, or `posix_memalign`).
- [ ] **N-Dimensional Tensor**: Support for shapes up to 8 dimensions (e.g., `{batch, channels, height, width}`).
- [ ] **Memory Layout**: Row-major (C-style) contiguous layout.
- [ ] **Strides & Offsets**: Tensors must store explicit stride arrays and an offset into the storage to support views.
- [ ] **Zero-Sized & Scalar Tensors**: Support empty shapes `{}` (scalars) and tensors with zero elements (e.g., `{0, 5}`).
- [ ] **Zero-Copy Views**: `slice()` and `narrow()` create new Tensor objects sharing the underlying storage (with modified shapes/strides/offsets).
- [ ] **Reshape Semantics (Unambiguous)**:
    - `reshape()` returns a zero-copy view **only when** the requested shape is compatible with the tensor's current stride layout.
    - If the tensor is non-contiguous and the requested shape is incompatible, `reshape()` **shall fail** (throw a clear exception) rather than silently allocating.
    - The `contiguous()` method explicitly creates a contiguous copy of the tensor's data.
- [ ] **No Silent Copies**:
    - Operations shall not silently materialize or copy non-contiguous tensors unless explicitly documented.
    - APIs that require contiguous storage (e.g., optimized `MatMul` kernels) must either call `contiguous()` on the input explicitly or fail with a clear error stating the stride requirements.
- [ ] **Max Tensor Size**: The library must safely handle at least `2^31 - 1` elements (within system memory limits).

### 2.2 Mathematical Operations (Forward Pass)
- [ ] **Elementwise**: `Add`, `Sub`, `Mul`, `Div`, `Pow` (scalar and tensor variants).
- [ ] **Unary Ops**: `Neg`, `Exp`, `Log`, `Sqrt`, `Abs`.
- [ ] **Activation Functions**: `ReLU`, `Sigmoid`, `Tanh` (with numerically stable implementations).
- [ ] **Matrix Multiplication (`MatMul`)**: 
    - **Scalar Reference**: A naive triple-loop implementation (for correctness).
    - **Optimized Kernels (Optional)**: Tiled implementations using AVX2/AVX-512 intrinsics that can be enabled at compile-time.
- [ ] **Reductions**: `Sum`, `Mean`, `Max`, `Min` across specified axes (with proper broadcasting behavior).
- [ ] **Normalization**: `LogSoftmax` (numerically stable).

### 2.3 Broadcasting (Forward & Backward)
- [ ] **Forward**: Full support for PyTorch/NumPy-style broadcasting rules for binary ops (e.g., `{4, 3}` + `{3}` -> `{4, 3}`).
- [ ] **Backward**: The gradient for a broadcasted operation must be correctly **reduced (summed)** along the broadcasted axes to match the original input's shape.

### 2.4 Automatic Differentiation (Autograd)
- [ ] **Graph Construction (Explicit Ownership Model)**:
    - The computation graph is built through explicit tensor-to-node ownership.
    - **Tensor** holds its autograd node via `std::shared_ptr<FunctionNode>`.
    - **Ownership Rule (CRITICAL)**: **(FIX #1)**  
      A `FunctionNode` owns **only** the tensors required to execute its `backward()` operation (e.g., saved input tensors for `ReLU`, the weight matrix for `MatMul`).  
      Graph traversal references that **do not** require ownership (e.g., child→parent backlinks, references to non-saved intermediate tensors) must use `std::weak_ptr`.  
      The implementation must **not** create ownership cycles.
- [ ] **`backward()` Signature Contract (Unambiguous)**: **(FIX #2)**
    - Signature: `virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;`
    - **Fixed contract**: The returned gradient vector contains **exactly one gradient per parent tensor** in the strict parent order stored internally by the node.
    - **Non-differentiable parents** (e.g., scalar constants, integer indices): the corresponding entry in the returned vector shall be an **empty/zero tensor**, which the autograd engine will safely ignore.
- [ ] **Single-Threaded Backward**: The backward pass is **single-threaded** for v1. Gradients are accumulated into normal `float` buffers (no `std::atomic` overhead).
- [ ] **Detach Semantics**: `detach()` creates a new tensor that shares the same storage but has `requires_grad=false` and **nulls out its `grad_fn` pointer**, severing the graph link. The old graph remains intact for other paths.
- [ ] **Graph Lifetime**: Backward graphs are owned by the tensors themselves. When a tensor goes out of scope, its `grad_fn` (and subsequent nodes) are destructed automatically. Calling `backward()` does *not* clear the graph by default (allows multiple backward calls if desired).

### 2.5 Validation & Testing
- [ ] **Unit Tests**: Every public API method must have a corresponding test in `tests/` using Google Test.
- [ ] **Reference Implementation Independence**: **(FIX #3)**
    - Reference (scalar) implementations must be **simple, unoptimized, and independently trusted**.
    - They **must not** share the optimized kernel's indexing, tiling, or loop ordering logic.
    - This isolation ensures that a bug in the optimized kernel's indexing helper does not get silently masked by the reference implementation agreeing on the same wrong result.
- [ ] **Optimized Kernel Validation**:
    - Every optimized kernel (SIMD, tiled, parallel) must be tested against the independent reference implementation over deterministic and randomized inputs before any performance metrics are collected.
- [ ] **Gradient Checking (Per-Op)**: For every differentiable operation, the autograd engine must pass a gradient check comparing analytic gradients to finite differences.
    - Tolerances selected per dtype and validated against representative operations. Initial baseline for `float32`: `rtol = 1e-3`, `atol = 1e-5`. These will be tightened empirically only if proven stable.
- [ ] **Gradient Checking (Small Network)**: After individual ops are validated, a 2-layer MLP (e.g., `784 -> 128 -> 10`) must pass gradient checking for a full forward/backward pass.
- [ ] **End-to-End Training (MNIST)**:
    - **Architecture**: MLP: 784 (input) -> 256 (ReLU) -> 128 (ReLU) -> 10 (LogSoftmax).
    - **Loss**: Negative Log-Likelihood (NLL) or Cross-Entropy.
    - **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate `0.01`.
    - **Hyperparameters**: Batch size `64`, Epochs `10`.
    - **Initialization**: Kaiming Uniform (He) for weight layers. Zero for biases.
    - **Normalization**: Input pixels scaled from `[0, 255]` to `[0.0, 1.0]`.
    - **Dataset**: Load MNIST via a custom C++ binary parser reading the standard `.idx` format (no external `torchvision`).
    - **Acceptance Criteria**: Achieve **>95%** test accuracy after 10 epochs.

---

## 3. Non-Functional Requirements

### 3.1 Performance
- **Scalar Reference**: A naive, unoptimized implementation must exist for every op for validation purposes.
- **Optimized Kernels**: Where significant speedups are proven (e.g., MatMul), compile-time flags (`-DWITH_AVX2=ON`) must enable faster kernels.
- **Parallelism**: The system is not required to use `std::execution::par`. The implementation may use raw `std::thread`, OpenMP, or manual loop unrolling to achieve parallelism, **provided benchmarks prove the parallel version outperforms the serial version** for sizes > 10,000 elements.
- **Benchmark Target (MatMul)**:
    - A dedicated `benchmark_matmul` target will run the **Scalar Reference** against the **AVX2 Optimized** implementation on dimensions `{512, 1024, 2048}`.
    - The target is to *demonstrate a measurable speedup* (aiming for >4x), but it is **not** a hard CI pass/fail gate—it is a reporting metric.

### 3.2 Code Quality & Error Handling
- **C++20 Standard**: Strict. 
- **Zero Raw `new`/`delete`**: Use smart pointers (`unique_ptr`, `shared_ptr`) with custom deleters for aligned memory.
- **Error Handling (C++20 Native)**:
    - Use straightforward C++ exceptions for API errors:
        - `throw std::invalid_argument(...)` for dimension mismatches and invalid inputs.
        - `throw std::out_of_range(...)` for out-of-bounds indexing.
        - `std::bad_alloc` for allocation failure.
- **Mutable Semantics**: **Strictly ban in-place tensor mutation for v1.** (e.g., `tensor.add_(5)` is illegal). All operations must return new tensors. This dramatically simplifies autograd correctness.

### 3.3 Thread-Safety Contract (Explicit)
- Independent tensor objects and independent computation graphs may be used concurrently from different threads.
- Concurrent access to the **same mutable tensor/storage** or the **same computation graph** is unsupported and results in undefined behavior.
- v1 autograd execution is strictly single-threaded.

### 3.4 Determinism & Reproducibility
- Deterministic random number generation is supported by passing an explicit `RNG` object (e.g., `std::mt19937`) to initialization functions.
- The property is: **Same seed + same graph + same inputs → identical weights and deterministic outputs**.
- The library **shall not** rely on a hidden global RNG.

### 3.5 Build & Tooling
- **CMake**: Modern CMake (target-based, `FetchContent` for dependencies).
- **Sanitizers**: Debug builds must compile with `-fsanitize=address,undefined`.
- **Profiling**: Must be compatible with Linux `perf` and `valgrind --tool=cachegrind`.

---

## 4. Explicit Non-Goals (Out of Scope for v1)

- ❌ No GPU or CUDA support.
- ❌ No multi-GPU, distributed training, or `torch.distributed`.
- ❌ No ONNX, PyTorch, or TensorFlow model importer/exporter.
- ❌ No Windows or macOS support (Ubuntu/Linux only).
- ❌ No Adam/other optimizers (only SGD is required for validation).
- ❌ No complex convolution ops (Conv2d is out of scope).
- ❌ No C++23 features.
- ❌ No in-place tensor operations.
- ❌ No hidden global state (RNGs are explicit).

---

## 5. Success Criteria (Definition of "Done")

The project is considered **complete** and successful when:

### 5.1 Correctness (Sanitizers & Valgrind)
1.  **Compilation**: Zero warnings with `-Wall -Wextra -Wpedantic`.
2.  **Memory**: Valgrind reports `0 bytes definitely lost` for the MNIST training run (build with `-O0 -g` for this specific test).
3.  **Sanitizers**: AddressSanitizer and UndefinedBehaviorSanitizer pass all unit tests and the MNIST training run without errors.

### 5.2 Performance (Benchmarks)
4.  **Benchmark Suite**: `google/benchmark` runs successfully.
5.  **MatMul Comparison**: The benchmark target outputs a clear comparison table (Scalar vs. AVX2) showing the speedup factor for the specified dimensions.

### 5.3 Autograd Validation
6.  **Per-Op Gradient Check**: All atomic operations pass the finite-difference test.
7.  **Small Network Gradient Check**: A 2-layer MLP passes gradient checking on a single random batch.

### 5.4 Business Logic
8.  **MNIST Accuracy**: The training script prints `Test Accuracy: > 95%` after 10 epochs using only the library's code.

---

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Broadcasting Backward Semantics** | Gradients are wrong sizes for broadcasted inputs | Implement comprehensive test suite covering all broadcasting scenarios (same shape, singleton broadcasting, multi-dimensional broadcasting) with gradient validation. Use property-based testing to generate random shapes and verify gradient correctness against finite differences. Require 100% pass rate on broadcasting tests before advancing to Phase 5. |
| **SIMD Tiling Complexity** | Phase 4 takes > 2 weeks | Implement and thoroughly test scalar reference MatMul using independent validation against BLAS-like reference implementations. Decouple AVX2 optimization behind feature flag (-DWITH_AVX2=ON) with runtime dispatch. Require scalar MatMul to pass all correctness tests (including gradient checks) before enabling AVX2 optimization work. |
| **Graph Ownership Cycles** | Memory leaks or dangling references | Implement strict ownership semantics using shared_ptr for owned relationships and weak_ptr for non-owned backlinks. Add runtime cycle detection in debug builds using smart pointer ownership tracking. Require zero memory leaks in Valgrind for all autograd tests and add leak detection to CI pipeline. |
| **Reshape Contiguity Confusion** | Silent data corruption | Implement reshape() with explicit contiguity check that throws std::invalid_argument for incompatible views. Add comprehensive test suite verifying both successful zero-copy reshapes and proper exception throwing for incompatible cases. Include stress tests with random shapes and sequences of view operations. |

---

## 7. Dependencies (External Libraries)

| Library | Purpose | Integration |
| :--- | :--- | :--- |
| **Google Test** | Unit Testing | `FetchContent` |
| **Google Benchmark** | Performance Measurement | `FetchContent` |
| **zlib (Optional)** | (Future) Serialization compression | System package (`libz-dev`) |

*Note: Error handling relies on standard C++ exceptions. No `tl::expected`, Boost, or Outcome libraries are used.*

---

*Approved by: [Your Name]*  
*Date: 2026-08-17*