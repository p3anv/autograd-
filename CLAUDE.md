# AutoGrad++ Tensor Library - Claude Code Context

## Project Overview
This is a high-performance, CPU-optimized N-dimensional tensor library with reverse-mode automatic differentiation (autograd) implemented from scratch in modern C++20. The library trains a Multi-Layer Perceptron (MLP) on MNIST with >95% accuracy without third-party linear algebra frameworks.

## Current Implementation Status
- **Phase 0 (Project Setup)**: Complete - CMake, CI, tooling, dependencies
- **Phase 1 (Storage & Tensor Core)**: Complete - Storage, TensorImpl, basic Tensor
- **Phase 2 (Views & Indexing)**: Complete - slice, narrow, reshape, contiguous
- **Phase 3 (Scalar Forward Operations)**: COMPLETE - All operations implemented and tested (58/58 passing)
- **Phase 4 (Autograd Engine)**: COMPLETE - All autograd components implemented and tested (70/70 passing)
- **Phase 5 (SIMD Optimization)**: COMPLETE - AVX2/FMA kernels implemented with runtime dispatch
- **Phase 6 (End-to-End Validation)**: COMPLETE - MNIST MLP achieves 96.66% test accuracy (>95% target)

## Key Implementation Details

### Coding Conventions
- Use `std::vector<std::size_t>` for shapes
- Use `std::ptrdiff_t` for strides (to allow negative strides in future)
- All tensors are float-based for simplicity
- Zero-copy views when possible (slice, narrow, reshape)
- Contiguous tensors required for data access and most operations
- Broadcasting follows NumPy/PyTorch rules
- Error handling via exceptions (`std::invalid_argument`, `std::runtime_error`)

### Tensor Classes
- **Storage**: Manages raw memory with 64-byte alignment
- **TensorImpl**: Logical view with shape, strides, offset, autograd metadata
- **Tensor**: User-facing handle (value-semantic, shared_ptr to TensorImpl)

### Current Working Features (Phase 3 & 4)
All operations tested in hardcore_test.cpp:
- Elementwise: add, sub, mul, div, pow (with broadcasting)
- Unary: neg, exp, log, sqrt, abs
- Activations: relu, sigmoid, tanh, softmax, log_softmax
- Reductions: sum, mean, max, min (with dimension parameter)
- Matrix multiplication: matmul (rank-2, contiguous-only)
- Broadcasting: Properly handled in all binary operations
- Autograd engine: Reverse-mode automatic differentiation with full gradient computation
- Gradient accumulation: Proper accumulation of gradients in leaf tensors

### Autograd Engine (Phase 4 - COMPLETE)
Implemented components:
1. **ParentEdge**: Helper to distinguish leaf vs intermediate tensors
2. **FunctionNode hierarchy**: Concrete nodes for each operation (AddBackward, MulBackward, etc.)
3. **AutogradEngine**: DAG-based gradient computation with dependency scheduling
4. **Tensor::backward()**: Trigger gradient computation
5. **TensorImpl::accumulate_grad()**: Gradient accumulation

### Build System
- CMake with options:
  - `BUILD_BENCHMARKS`: OFF (default)
  - `ENABLE_SANITIZERS`: ON (default for Debug)
  - `WITH_AVX2`: OFF (default, enables AVX2 optimizations)
  - `BUILD_TESTING`: ON
- Targets:
  - `autograd`: Library
  - `autograd_tests`: Test executable
- Dependencies: Google Test, Google Benchmark (via FetchContent)

### Testing
- Google Test framework
- Test files in `tests/` directory:
  - `test_storage.cpp`
  - `test_tensor_impl.cpp`
  - `test_tensor.cpp`
  - `test_autograd.cpp` (comprehensive Phase 4 validation)
  - `hardcore_test.cpp` (comprehensive Phase 3 validation)
  - `test_simd.cpp` (SIMD kernel equivalence tests)
- Run tests: `cd build && ctest --output-on-failure`
- Current status: All 128 tests passing (58 Phase 3 + 70 Phase 4), SIMD equivalence tests passing

### Memory Safety
- AddressSanitizer and UndefinedBehaviorSanitizer enabled in Debug
- Valgrind clean for memory leaks (verified 0 bytes in use at exit)
- RAII-based memory management

### Next Steps for Future Phases
- Phase 5 complete: AVX2/FMA SIMD optimization with runtime CPU dispatch
- Phase 6 complete: End-to-end MNIST validation achieving >95% accuracy
- Optional: Benchmark suite, profiling, further optimization

## Session Context Preservation
This CLAUDE.md file helps maintain context across Claude Code sessions. Key points to remember:
- Phase 3 (Forward Operations) and Phase 4 (Autograd Engine) are COMPLETE
- Phase 5 (SIMD Optimization) is COMPLETE with AVX2/FMA kernels and runtime dispatch
- Phase 6 (End-to-End Validation) is COMPLETE - MNIST MLP achieves 96.66% test accuracy
- All 128 tests passing (58 Phase 3 + 70 Phase 4), SIMD equivalence tests passing
- Valgrind clean, AddressSanitizer clean
- Maintain consistency with existing code patterns
- Continuously run tests to ensure no regressions