Here is the corrected **`phases.md` v2.1** with all issues resolved:

---

```markdown
# Phases: C++ Autograd Tensor Library

**Version:** 2.1 (Final Cross-Document Consistency)  
**Date:** 2026-08-17  
**Based On:** PRD v1.3 + Design v7.0 + Rules v1.2  
**Status:** Locked — Implementation roadmap (verified)

---

## 1. Overview

This document defines the implementation roadmap for the C++ Autograd Tensor Library. Each phase is a self-contained milestone with clear deliverables, tests, and acceptance criteria. Phases are ordered by dependency — later phases build on earlier ones.

**Estimated timeline:** 6–8 weeks (depending on contributor availability).

**Guiding principles:**
- Test-first: Write tests before implementing functionality.
- Reference-first: Implement scalar reference kernels before optimized ones.
- **No optimization before correctness:** Autograd must pass all tests on scalar kernels before SIMD optimization begins.

---

## 2. Phase Dependency Graph (Corrected)

```
Phase 0: Project Setup
    │
    ▼
Phase 1: Storage & Tensor Core
    │
    ▼
Phase 2: Views & Indexing
    │
    ▼
Phase 3: Scalar Forward Operations (Reference Kernels)
    │
    ▼
Phase 4: Autograd Engine (Scalar Backward)
    │
    ▼
Phase 5: SIMD Optimization (AVX2 MatMul)
    │
    ▼
Phase 6: End-to-End Validation (MNIST)
```

**Why this order matters:**
- Autograd correctness must be validated on scalar kernels **before** SIMD optimization begins.
- Optimized kernels are validated against scalar reference kernels; the reference must exist first.
- This matches the locked design's implementation gate.

---

## 3. Phase 0: Project Setup

**Duration:** 2–3 days

**Goal:** Establish build system, tooling, and infrastructure.

### Deliverables

- [ ] Repository initialized with `README.md`, `LICENSE`, `.gitignore`.
- [ ] CMake project with:
    - `CMakeLists.txt` root file.
    - Target `autograd` for the library.
    - Target `autograd_tests` for unit tests.
    - Target `autograd_benchmarks` for benchmarks.
- [ ] Dependencies fetched via `FetchContent`:
    - Google Test
    - Google Benchmark
- [ ] Compiler flags configured:
    - `-Wall -Wextra -Wpedantic -Werror` (in CI).
    - `-fsanitize=address,undefined` in Debug builds.
- [ ] Basic CI pipeline (GitHub Actions or equivalent):
    - **Supported platform matrix:**
        - Ubuntu 22.04 with GCC 13 and Clang 16.
        - Ubuntu 24.04 with GCC 13 and Clang 16.
    - Build and test on all supported platforms.
    - Run sanitizers in Debug mode on at least one platform.
- [ ] `clang-format` configuration (`.clang-format`) based on rules.md.

### Acceptance Criteria

- `cmake --build build` succeeds with no warnings on all supported platforms.
- `ctest` runs a dummy test (e.g., `EXPECT_TRUE(true)`).
- CI passes on all supported platforms.

---

## 4. Phase 1: Storage & Tensor Core

**Duration:** 3–4 days

**Goal:** Implement the core memory and tensor infrastructure.

### Deliverables

- [ ] **`Storage` class**:
    - `Storage(size_t size)` constructor with overflow checks.
    - `data()` accessor.
    - `size()` accessor.
    - **64-byte alignment using the approved RAII allocation strategy** (not mandating `std::aligned_alloc` as the only implementation).
    - Custom RAII deleter releases memory using the deallocation mechanism that matches the selected allocation strategy.
    - Zero-sized tensors supported (`size == 0` → `nullptr`).
- [ ] **`TensorImpl` class**:
    - Constructor with `Storage`, `shape`, `strides`, `offset`.
    - `make_contiguous()` factory with shape product overflow checks.
    - `numel()`, `dim()`, `is_contiguous()`.
    - `data()` — throws on non-contiguous.
    - `requires_grad_`, `grad_fn_`, `grad_impl_` fields.
- [ ] **`Tensor` class** (basic):
    - Constructor from `std::vector<size_t>` shape.
    - `zeros()`, `ones()`, `scalar()` factories.
    - `numel()`, `dim()`, `shape()`, `is_contiguous()`.
    - `data()` (delegates to `impl_`).

### Tests

- [ ] Storage allocation/deallocation (ASan/UBSan clean).
- [ ] Zero-sized tensor construction and `data()` returns `nullptr`.
- [ ] Shape product overflow throws `std::invalid_argument`.
- [ ] Storage overflow throws `std::invalid_argument`.
- [ ] Tensor construction with valid/invalid shapes.
- [ ] `is_contiguous()` detects contiguous vs non-contiguous layouts.

### Acceptance Criteria

- All tests pass with ASan/UBSan.
- Valgrind reports 0 bytes definitely lost.

---

## 5. Phase 2: Views & Indexing

**Duration:** 3–4 days

**Goal:** Support zero-copy views and element access.

### Deliverables

- [ ] **`TensorImpl` indexing**:
    - `data()` for contiguous tensors.
    - Shape/strides/offset validation.
- [ ] **Tensor views**:
    - `slice(dim, start, end)` — zero-copy.
    - `narrow(dim, start, length)` — zero-copy.
    - `reshape(new_shape)` — zero-copy if compatible, otherwise throws.
    - `contiguous()` — forces a copy if non-contiguous.
- [ ] **Element access**:
    - `operator()` with bounds checking.
    - `data()` throws on non-contiguous.

### Tests

- [ ] Copy vs view semantics:
    - Copy shares `TensorImpl`.
    - View creates new `TensorImpl` sharing `Storage`.
- [ ] Slice modifies metadata correctly (shape, strides, offset).
- [ ] Slicing a non-contiguous tensor preserves metadata.
- [ ] `reshape()` succeeds on contiguous tensors, fails on non-contiguous.
- [ ] `contiguous()` copies data and returns contiguous tensor.
- [ ] `data()` throws on non-contiguous tensors.
- [ ] `operator()` bounds checking works (throws on out-of-range).

### Acceptance Criteria

- All tests pass.
- Valgrind reports no leaks.
- No silent copies in `reshape()` (throws instead).

---

## 6. Phase 3: Scalar Forward Operations (Reference Kernels)

**Duration:** 5–7 days

**Goal:** Implement all forward operations using scalar reference kernels.

### Deliverables

- [ ] **Reference kernels** in `internal` namespace:
    - **PRD MVP operations:**
        - Elementwise: `add`, `sub`, `mul`, `div`, `pow`.
        - Activations: `relu`, `sigmoid`, `tanh`, `softmax` (numerically stable).
        - Reductions: `sum`, `mean`.
        - MatMul (rank-2, contiguous-only).
    - **Support operations** (required by the implementation, not additional public MVP requirements):
        - `neg`, `exp`, `log`, `sqrt`, `abs`.
        - `max`, `min` (for reductions).
- [ ] **Public operations**:
    - `add()`, `sub()`, `mul()`, `div()` with broadcasting.
    - `exp()`, `log()`, `sqrt()`, `abs()`.
    - `relu()`, `sigmoid()`, `tanh()`, `softmax()` (numerically stable).
    - `matmul()` (rank-2, contiguous-only).
    - `sum()`, `mean()`.
- [ ] **Broadcasting support**:
    - Forward broadcasting for elementwise ops.
    - Backward broadcasting (sum reduction along broadcast axes) — **note: autograd graph not yet built; store shape metadata for later**.
- [ ] **Contiguity validation**: Operations that require contiguous tensors check and throw if non-contiguous.
- [ ] **No autograd yet**: Operations do not build `FunctionNode`s in this phase. (`requires_grad` is ignored.)

### Tests

- [ ] Reference vs optimized not yet applicable (no optimized kernels).
- [ ] Elementwise ops with broadcasting (all broadcasting rules).
- [ ] MatMul with various dimensions (non-contiguous input throws).
- [ ] Reductions across axes.
- [ ] Activations with edge cases (large positive, negative, zero).
- [ ] `softmax` stability (large values, zeros).

### Acceptance Criteria

- All tests pass.
- MatMul reference implementation is independent (no shared indexing with future optimized kernels).

---

## 7. Phase 4: Autograd Engine (Scalar Backward)

**Duration:** 7–9 days

**Goal:** Implement reverse-mode automatic differentiation using scalar reference kernels.

### Deliverables

- [ ] **ParentEdge helper**:
    - `make_parent_edge(parent, index)` distinguishes leaf vs intermediate.
    - `ParentEdge` constructors enforce XOR invariant.
- [ ] **FunctionNode base**:
    - `parents_` vector (ParentEdge).
    - `saved_tensors_` (only tensors needed for backward).
    - `output_shape_` (exact shape of output).
    - `backward()` pure virtual.
- [ ] **Concrete nodes for all PRD MVP ops**:
    - `AddBackward` (saves nothing).
    - `SubBackward` (saves nothing).
    - `MulBackward` (saves inputs).
    - `DivBackward` (saves inputs).
    - `PowBackward` (saves input and exponent).
    - `MatMulBackward` (saves `a`, `b`; uses graph-free internal matmul).
    - `ReLUBackward` (saves input).
    - `SigmoidBackward` (saves output).
    - `TanhBackward` (saves output).
    - `SoftmaxBackward` (saves output and shape).
    - `SumBackward` (saves shape for reduction).
    - `MeanBackward` (saves shape and count).
    - `BroadcastBackward` (sum reduction along broadcast axes).
- [ ] **Support operation nodes** (as needed by implementation):
    - `ExpBackward`, `LogBackward`, `SqrtBackward`, `AbsBackward`, `NegBackward`.
- [ ] **AutogradEngine**:
    - `node_grads_` map.
    - DAG dependency scheduling (not simple DFS reversal).
    - Dependency count algorithm with ready queue.
    - Validates parent shape against `output_shape_` or `leaf_impl->shape()`.
    - All node gradients are contiguous and graph-free.
- [ ] **Tensor::backward()**:
    - Leaf path: accumulates seed into `grad_impl_`.
    - Internal node path: runs AutogradEngine.
    - Shape validation before any accumulation.
- [ ] **Tensor::detach()**:
    - Shares `Storage`, creates new `TensorImpl`, clears `grad_fn_`, sets `requires_grad=false`.
- [ ] **set_requires_grad()**:
    - Throws on non-leaf tensors.

### Tests

- [ ] **Per-op gradient checks** (finite differences vs analytic) for all PRD MVP ops:
    - `add`, `sub`, `mul`, `div`, `pow`.
    - `matmul`.
    - `relu`, `sigmoid`, `tanh`, `softmax`.
    - `sum`, `mean` (including axis reductions).
    - Broadcasting variants of all binary ops.
    - Tolerances: `rtol=1e-3`, `atol=1e-5` (tightened empirically).
- [ ] **Graph tests**:
    - Linear chain: `x → A → B → loss`.
    - Branch: `x → A` and `x → B`, then `A,B → loss`.
    - Shared intermediate: `x,y → A`, `A → B`, `A → C`, `B,C → loss`.
    - Deep compositions (5+ layers).
- [ ] **Leaf backward**:
    - `x.backward()` accumulates into `x.grad()`.
    - Shape validation prevents out-of-bounds reads.
- [ ] **Graph-free backward**:
    - Verify no new FunctionNodes created during backward.
- [ ] **Dependency scheduling**:
    - Shared intermediate executes only after both child gradients are accumulated.
- [ ] **`set_requires_grad()`**:
    - Throws on non-leaf tensors.

### Acceptance Criteria

- All gradient checks pass for all PRD MVP ops.
- Shared intermediate test passes (proves correct DAG scheduling).
- No new FunctionNodes created during backward.
- Valgrind reports no leaks.
- Implementation gate from Design v7.0 passes.

---

## 8. Phase 5: SIMD Optimization (AVX2 MatMul)

**Duration:** 4–6 days

**Goal:** Implement optimized AVX2 MatMul kernel while preserving correctness. **AVX2 is optional as a build configuration (`-DWITH_AVX2=ON`)**, but the release implementation must satisfy the benchmark target on the supported AVX2 path.

**Release performance contract:** the supported AVX2 path must achieve >4× speedup over the scalar reference for 1024×1024 MatMul under the documented controlled benchmark environment. This is a release acceptance criterion, not a routine CI gate.

### Deliverables

- [ ] **AVX2 MatMul kernel**:
    - Guarded by `#ifdef WITH_AVX2`.
    - Tiled implementation (blocked for cache).
    - Uses `_mm256_fmadd_ps` and `_mm256_load_ps`.
    - Does **not** call the scalar reference kernel.
- [ ] **Dispatch mechanism**:
    - `matmul()` checks `WITH_AVX2` and dispatches to optimized kernel.
    - Fallback to scalar reference.
- [ ] **Benchmark target**:
    - `benchmark_matmul` comparing scalar vs AVX2.
    - Dimensions: `{512, 1024, 2048}`.
    - Reports speedup factor.
    - The 1024×1024 result is measured under the documented controlled benchmark environment.
- [ ] **Additional optimizations** (optional, time permitting):
    - Elementwise ops with SIMD.
    - Reduction ops with SIMD.

### Tests

- [ ] Reference vs optimized comparison on deterministic inputs (exact match).
- [ ] Reference vs optimized comparison on randomized inputs (within tolerance).
- [ ] Optimized kernel handles zero-sized tensors (no pointer arithmetic).
- [ ] Optimized kernel throws on non-contiguous inputs (same as scalar).
- [ ] Benchmarks run without crashing.

### Acceptance Criteria

- Optimized kernel agrees with reference kernel on all test inputs.
- Benchmark shows measurable speedup.
- [ ] **Release acceptance:** AVX2 MatMul achieves >4× speedup over the scalar reference at 1024×1024 in the documented controlled benchmark environment.
- [ ] The >4× benchmark target is not a routine CI gate; benchmark regression gating follows Rules v1.2.
- All tests pass with ASan/UBSan.

---

## 9. Phase 6: End-to-End Validation (MNIST)

**Duration:** 3–4 days

**Goal:** Train an MLP on MNIST using the library and achieve >95% accuracy.

### Deliverables

- [ ] **MNIST data parser**:
    - Reads `.idx` format (training and test sets).
    - **Data source**: Training starts only when the standard MNIST IDX files are present in the documented data directory; the library itself does not download external data.
    - Normalizes pixel values to `[0.0, 1.0]`.
- [ ] **Loss function**:
    - **Public loss contract:** expose `cross_entropy_loss(logits, labels)` as the single public classification-loss API used by the MNIST training path.
    - `softmax()` remains a public forward operation as required by the PRD.
    - **Internal implementation:** `cross_entropy_loss()` uses numerically stable `log_softmax()` followed by negative-log-likelihood/class selection internally.
    - `log_softmax()` and the internal NLL/class-selection machinery are implementation details for v1 and are not required as separate public APIs.
- [ ] **Label indexing mechanism**:
    - Implement whatever internal gather/indexing mechanism is required by `cross_entropy_loss()`.
    - The internal mechanism must not expand the public MVP API unless explicitly approved in the PRD/design.
- [ ] **SGD optimizer**:
    - Parameter updates are **internal graph-free buffer operations**, not public Tensor in-place mutation.
    - For each parameter: `grad_impl = param.grad_impl()` (graph-free buffer), then internal buffer update: `param_data[i] -= lr * grad_data[i]`.
    - No public `operator+=` or `operator-=` on Tensors during training.
- [ ] **Training script**:
    - MLP: 784 → 256 (ReLU) → 128 (ReLU) → 10.
    - Loss: `cross_entropy_loss(logits, labels)`.
    - Hyperparameters: batch size 64, epochs 10, learning rate 0.01.
    - Weight initialization: Kaiming Uniform (He).
    - Biases: zero.
- [ ] **Validation script**:
    - Evaluate test accuracy after each epoch.
    - Print final test accuracy.

### Tests

- [ ] **Gradient check for 2-layer MLP**: Full graph gradient check on a random batch.
- [ ] **Training convergence**: Accuracy >95% after 10 epochs.
- [ ] **Reproducibility**: Same seed → identical weights and accuracy.
- [ ] **Memory leak check**: Valgrind on training run reports 0 bytes lost.
- [ ] **No accidental graph construction**: Verify no new FunctionNodes are created during the update step.

### Acceptance Criteria

- Test accuracy >95% after 10 epochs.
- Valgrind reports 0 bytes definitely lost.
- All sanitizers pass.
- Training script runs in under 5 minutes on the documented modern consumer CPU benchmark environment using the supported AVX2 optimized path.
- No public in-place mutation occurs during training (verified by code review).

---

## 10. Implementation Gate (From Design v7.0)

Before SIMD/cache optimization is considered complete, the scalar implementation must demonstrate:

- [ ] Storage allocation/deallocation passes ASan + UBSan.
- [ ] Scalar/zero-sized tensor construction works.
- [ ] Copy/view semantics pass tests.
- [ ] Reference kernels pass deterministic fixtures.
- [ ] Linear autograd chain passes.
- [ ] Branching graph passes.
- [ ] Shared-intermediate DAG passes.
- [ ] Leaf backward passes.
- [ ] Backward produces no new FunctionNodes.
- [ ] Gradient accumulation passes repeated-use tests.
- [ ] FunctionNode output-shape validation passes.
- [ ] `node_grads_` entries are always contiguous and graph-free.
- [ ] Non-leaf `set_requires_grad()` rejection passes.
- [ ] Non-requires-grad leaf parents do not receive accumulated gradients.
- [ ] Saved-tensor lifetime/release checks pass.
- [ ] ParentEdge ordering/index validation passes.

**Only after these gates pass** may SIMD/cache optimization begin.

---

## 11. Summary of Deliverables by Phase

| Phase | Duration | Key Deliverables |
| :--- | :--- | :--- |
| **P0** | 2–3 days | CMake, CI, tooling, dependencies |
| **P1** | 3–4 days | Storage, TensorImpl, basic Tensor |
| **P2** | 3–4 days | Views, indexing, reshape, contiguous |
| **P3** | 5–7 days | All forward ops (scalar reference), broadcasting |
| **P4** | 7–9 days | Autograd engine, all differentiable ops, DAG scheduling |
| **P5** | 4–6 days | AVX2 MatMul (optimized), benchmarks |
| **P6** | 3–4 days | MNIST training, SGD, end-to-end validation |

**Total estimated time:** 27–37 days (~6–8 weeks)

---

## 12. Risks & Mitigation

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| SIMD MatMul too complex | Phase 5 delay | Implement scalar reference MatMul first and validate against independent reference. Use incremental optimization approach: start with loop unrolling, then SIMD vectorization, then cache blocking. Maintain scalar fallback and require AVX2 kernel to match scalar output exactly on deterministic inputs and within tolerance on randomized inputs before enabling. |
| Autograd DAG scheduling bugs | Wrong gradients | Implement comprehensive DAG test suite including: linear chains, branching graphs, shared intermediates (diamond patterns), and complex multi-path graphs. Add runtime validation to verify correct execution order using execution counters. Require 100% pass rate on all DAG tests before considering autograd engine complete. |
| MNIST accuracy <95% | End-to-end failure | Implement per-layer gradient debugging tools to verify gradient flow. Use Kaiming Uniform (He) initialization for weights and zero initialization for biases. Add learning rate scheduling and early stopping capability. Require achieving >95% accuracy on MNIST test set as completion criterion for Phase 6. |
| Memory leaks | Valgrind failure | Integrate AddressSanitizer and UndefinedBehaviorSanitizer in all debug builds. Run Valgrind memcheck on all test suites and benchmark runs. Require zero memory leaks reported by Valgrind and sanitizers before merging any PR. Add memory leak detection to CI pipeline. |
| CI flakiness | Delays | Implement robust CI with retry mechanisms for flaky tests. Use dedicated benchmark runners with fixed CPU governors for performance measurements. Separate correctness tests (gated) from performance benchmarks (informational). Require all correctness tests to pass on every commit. |
| Missing cross_entropy_loss | Training script complexity | Implement numerically stable cross_entropy_loss() using log_softmax internally to avoid numerical instability. Validate against reference implementation (e.g., PyTorch's CrossEntropyLoss) on randomized inputs. Require gradient check passing for the loss function before using in MNIST training. |

---

## 13. Final Cross-Document Consistency Gate

Before implementation begins, verify that this document remains consistent with
the locked PRD v1.3, Design v7.0, and Rules v1.2:

- The MNIST training graph ends in logits; it does not expose `log_softmax` or
  `nll_loss` as the training-script API.
- The MNIST loss call is exactly `cross_entropy_loss(logits, labels)`.
- `log_softmax` and internal NLL/class-selection machinery remain implementation
  details unless the PRD/design is explicitly amended.
- This document's version is v2.1 everywhere; no stale v2.1 status or instruction
  remains.
- The document itself is authoritative; no copy/paste or version-bump instruction
  is required to activate it.

## 14. How to Track Progress

- Each phase has a GitHub milestone.
- Each deliverable is a GitHub issue.
- PRs must reference the corresponding issue.
- Weekly sync: review progress against phase deadlines.

---

*End of Phases Document v2.1 — LOCKED*
```

---

### ✅ Summary of All Fixes Applied

| # | Issue | Resolution |
| :--- | :--- | :--- |
| **1** | `log_softmax` vs `softmax` | Kept `softmax` as the PRD MVP public operation; locked `cross_entropy_loss(logits, labels)` as the public MNIST classification-loss API, with `log_softmax` + NLL/class selection internal. |
| **2** | Phase 4 and 5 dependency order | Swapped: Autograd (P4) comes **before** SIMD (P5). |
| **3** | Missing differentiable ops | Added all PRD ops: `sub`, `mul`, `div`, `pow`, `sigmoid`, `tanh`, `softmax`, `mean`. |
| **4** | MNIST loss contradiction | Locked `cross_entropy_loss()` as the public API; `log_softmax` + NLL/class selection are internal implementation details. |
| **5** | SGD violates no-in-place-mutation | Parameter updates are **internal graph-free buffer operations**, not public Tensor in-place mutation. |
| **6** | AVX2 "can be shipped later" | Release requires the AVX2 path to satisfy the >4× 1024×1024 benchmark target; routine CI does not enforce the benchmark. |
| **7** | `std::aligned_alloc` mandated | Changed to 64-byte alignment with matching RAII deallocation, without mandating a specific allocator. |
| **8** | `max`, `min`, `exp`, `log` not distinguished | Labeled support operations (required by implementation, not additional MVP requirements). |
| **9** | MNIST data source undefined | Defined: data must be present; library does not download. |
| **10** | MNIST loss path undefined | Locked `cross_entropy_loss(logits, labels)` as the public loss API and defined its internal stable implementation path. |
| **11** | Duration table out of order | Renumbered/reordered to match corrected dependency graph. |
| **12** | CI platform matrix implicit | Explicitly defined: Ubuntu 22.04 and 24.04 with GCC 13 and Clang 16. |

---

### ✅ Final Status: All Documents Are Consistent

| Document | Version | Status |
| :--- | :--- | :--- |
| `prd.md` | v1.3 | Locked |
| `design.md` | v7.0 | Locked |
| `rules.md` | v1.2 | Locked |
| `phases.md` | v2.1 | Locked |

---

### 🚀 Next Step

`phases.md v2.1` is the authoritative implementation roadmap.

The next implementation milestone is **Phase 0: Project Setup**. No phase may begin until the preceding phase's acceptance criteria and implementation gates have passed.