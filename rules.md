
```markdown
# Rules: C++ Autograd Tensor Library

**Version:** 1.2 (Global State & SIMD Clarified)  
**Date:** 2026-08-17  
**Based On:** PRD v1.3 + Design v7.0 + Rules v1.1 Corrections  
**Status:** Locked — Implementation must follow these rules

---

## 1. General Principles

1. **Correctness over performance.** We optimize only after tests and benchmarks prove correctness.
2. **No warnings.** The code must compile with `-Wall -Wextra -Wpedantic` (and `-Werror` in CI).
3. **No undefined behavior.** All pointer arithmetic, casts, and memory accesses must be well-defined.
4. **RAII everywhere.** Resources (memory, file handles, locks) are owned by objects with deterministic lifetime.
5. **Minimal dependencies.** Only dependencies explicitly listed in the PRD are allowed. No hidden third-party libraries.
6. **Document why, not what.** Comments explain design decisions and invariants, not obvious code.
7. **No stdout/stderr output from library code.** The library must not print anything unexpectedly during tensor operations, error handling, or training. Applications/examples may print for demo purposes.
8. **No global mutable state.** The library must not use mutable global state for configuration, RNG, autograd, logging, or caches. [FIX #1]

---

## 2. C++ Standard & Compiler

- **Standard:** C++20 (strict). C++23 features are not allowed until the toolchain fully supports them.
- **Compiler:** GCC 13+ or Clang 16+.
- **Target Platform:** Ubuntu 22.04 / 24.04 (x86_64).
- **Build System:** CMake 3.20+ with `FetchContent` for dependencies.
- **Sanitizers:** Debug builds must enable `-fsanitize=address,undefined` (and optionally `-fsanitize=thread` for future multithreading).
- **Optimization flags:** `-O2` or `-O3` for benchmarks; `-Og` for debugging.
- **SIMD:** [FIX #2] AVX2 optimized kernels are optional and enabled via `-DWITH_AVX2=ON`. The scalar reference implementation must always exist. AVX-512 is not required for v1.

---

## 3. Naming Conventions

| Category | Convention | Example |
| :--- | :--- | :--- |
| **Classes/Structs** | `PascalCase` | `Tensor`, `Storage`, `FunctionNode` |
| **Methods/Functions** | `snake_case` | `matmul()`, `reshape()`, `make_contiguous()` |
| **Private member variables** | `trailing_underscore_` | `storage_`, `shape_`, `grad_impl_` |
| **Public member variables** | `snake_case` (rare; prefer getters) | (allowed only for simple POD structs) |
| **Constants** | `kPascalCase` | `kMaxDimensions`, `kAlignment` |
| **Template parameters** | `T` or `PascalCase` | `typename T`, `typename Allocator` |
| **Type aliases** | `PascalCase` | `using TensorMap = ...;` |
| **Enum classes** | `PascalCase` | (Avoid unnecessary examples that imply out-of-scope features.) |
| **Macros** | `SCREAMING_SNAKE_CASE` (avoid macros) | (use `#pragma once` and `constexpr` instead) |
| **File names** | `snake_case.hpp` and `snake_case.cpp` | `tensor.hpp`, `storage.cpp` |
| **Namespace** | `autograd` for public; `internal` for implementation details | `namespace autograd { ... }` |

**Example:**

```cpp
class TensorImpl {
private:
    std::shared_ptr<Storage> storage_;
    std::vector<size_t> shape_;
};

Tensor matmul(const Tensor& a, const Tensor& b) { ... }

constexpr size_t kAlignment = 64;
```

---

## 4. Code Formatting

- **Indentation:** 4 spaces (no tabs).
- **Line width:** 100 characters maximum.
- **Braces:** Opening brace on the same line as the declaration/statement (`K&R` style).
- **Spacing:** One space after `if`, `for`, `while`, `switch`; no space before `(` in function calls.
- **Pointer/reference:** `T* ptr` and `T& ref` (align with type).
- **`auto` usage:** Use `auto` only when the type is obvious from the context (e.g., iterators, lambdas). Avoid `auto` where it harms readability (e.g., for fundamental types where explicit is clearer).

**Example:**

```cpp
if (condition) {
    do_something();
} else {
    do_other();
}

for (size_t i = 0; i < n; ++i) {
    sum += array[i];
}

float* data = storage_->data();
auto it = vec.begin();  // acceptable
size_t size = vec.size();  // explicit
```

We will use **`clang-format`** with a `.clang-format` file based on these rules (to be added to the repo).

---

## 5. Error Handling

- **Use exceptions** for API errors (invalid arguments, out-of-range, shape mismatches). Exceptions are clear and reduce error-checking boilerplate.
- **Specific exception types:**
    - `std::invalid_argument` — dimension mismatch, invalid shape, unsupported operation.
    - `std::out_of_range` — index out of bounds.
    - `std::runtime_error` — internal errors (e.g., `data()` on non-contiguous).
    - `std::logic_error` — invariant violations (e.g., `set_requires_grad` on non-leaf).
    - `std::bad_alloc` — allocation failure (thrown by the standard library).
- **No exceptions** for expected control flow. Use `if`/`throw` only for exceptional cases.
- **All exceptions must have descriptive messages** that explain what went wrong and, where possible, what the expected values were.
- **Exception safety:** The library must provide at least the basic guarantee (no resource leaks; objects are in a valid state). Most operations provide the strong guarantee (no change on failure) because they are functional (no in-place mutation).

---

## 6. Memory Management & Ownership

- **Prefer smart pointers** (`std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr`) over raw `new`/`delete`.
- **Use `std::make_shared` and `std::make_unique`** whenever possible.
- **Raw pointers are allowed** for non-owning views (e.g., `data()` returning `float*`) but must be clearly documented.
- **No manual memory management** in user-facing code. Only the `Storage` class is allowed to call `std::aligned_alloc`/`std::free` via a custom deleter.
- **RAII for all resources.** Any class that allocates resources must have a destructor that releases them.

---

## 7. Testing Requirements

- **Google Test** is the unit-testing framework.
- **Google Benchmark** is the performance-testing framework.
- **Every public API** must have at least one unit test covering normal behavior and error paths.
- **Autograd gradient checks** must use finite differences and compare with analytic gradients with specified tolerances.
- **Reference vs optimized kernels** must be tested on deterministic and randomized inputs.
- **Memory tests:** Valgrind and sanitizers must pass before merging any PR.
- **Coverage:** Aim for >80% line coverage (measured with `gcov`). Critical code (autograd, kernels) should be >95%.
- **Test naming:** `TestClass_Method_Scenario` (e.g., `TestTensor_Add_WithBroadcasting`).

---

## 8. Commit Message Format

We use **Conventional Commits**:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Allowed types:**
- `feat` — new feature
- `fix` — bug fix
- `perf` — performance improvement
- `refactor` — code restructuring (no functional change)
- `test` — adding/updating tests
- `docs` — documentation
- `style` — formatting, whitespace
- `chore` — build, dependencies, CI

**Scope:** The module/area affected (e.g., `tensor`, `storage`, `autograd`, `matmul`, `cmake`).

**Subject:** Imperative, present tense, ≤50 characters.

**Body:** Explain *what* and *why*, not how.

**Footer:** References issues, PRs, or breaking changes.

**Examples:**

```
feat(tensor): add scalar factory and arange

Add static methods Tensor::scalar() and Tensor::arange() to simplify
creation of single-element and sequential tensors.

Refs #12
```

```
fix(autograd): validate gradient shape before leaf accumulation

Previously, shape validation was performed after leaf accumulation,
which could cause out-of-bounds reads. Now validate before any
accumulation.

Fixes #34
```

---

## 9. Branching & PR Workflow

- **Main branch:** `main` (always production-ready).
- **Development:** Feature branches (e.g., `feat/matmul-simd`) branching from `main`.
- **Pull requests:** Before merging, a PR must:
    1. Pass CI (build, tests, sanitizers, benchmarks).
    2. Have at least one approval from a reviewer.
    3. Be up-to-date with `main` (rebase or merge).
    4. Include tests for new functionality.
- **Commit history:** We prefer a clean, linear history. Use `git rebase` before merging.

---

## 10. Documentation Requirements

- **All public headers** must be documented using Doxygen-style comments (`///` or `/**`).
- **Documentation must include:** brief description, parameters, return value, exceptions thrown, and any notable behavior (e.g., contiguity requirements).
- **Internal code** may use less documentation but should have comments for non-obvious logic.
- **The repository must include**:
    - `README.md` — overview, build instructions, usage examples.
    - `CONTRIBUTING.md` — contribution guidelines (point to `rules.md`).
    - `LICENSE` — (choose MIT or Apache 2.0).

---

## 11. Build System (CMake)

- **Modern CMake** (targets, properties, not global variables).
- **Use `FetchContent`** for Google Test and Google Benchmark.
- **No hardcoded paths.** Use `find_package` for system dependencies (e.g., `Threads`, `zlib`).
- **Set C++ standard per target** using `target_compile_features`, not a global `set(CMAKE_CXX_STANDARD)`:

```cmake
target_compile_features(autograd
    PUBLIC
        cxx_std_20
)

# Similarly for tests and benchmarks:
target_compile_features(autograd_tests
    PRIVATE
        cxx_std_20
)
```

- **Build variants:**
    - `Debug` — with `-O0 -g` and sanitizers enabled.
    - `Release` — with `-O2 -DNDEBUG`.
    - `RelWithDebInfo` — with `-O2 -g` (for profiling).
- **Optional features:**
    - `-DWITH_AVX2=ON` — enable AVX2 optimized kernels.
    - `-DBUILD_BENCHMARKS=ON` — build benchmark targets.
    - `-DENABLE_SANITIZERS=ON` — enable AddressSanitizer and UndefinedBehaviorSanitizer.

---

## 12. Continuous Integration (CI)

- **CI provider:** GitHub Actions or equivalent.
- **Jobs:**
    1. **Build & Test** on Ubuntu 22.04 with GCC 13 and Clang 16.
    2. **Sanitizers** (ASan + UBSan) in Debug mode.
    3. **Valgrind** (memcheck) for memory leak detection.
    4. **Benchmarks** (optional, run on `main` only).
    5. **Code coverage** (optional, using `gcov`).
- **CI must fail** on:
    - Compilation warnings.
    - Test failures.
    - Sanitizer errors.
    - Valgrind errors.
- **Performance regression checks:** [FIX #3]
    - Benchmark results are informational by default.
    - Regressions become **blocking** only when all of the following are met:
        1. The measurement is performed on controlled, consistent hardware/environment (e.g., dedicated benchmarking runner with fixed CPU governor).
        2. The regression is **statistically significant**: median performance is at least **10% worse** than the locked baseline.
        3. The regression is confirmed over **3–5 repeated benchmark runs** (to reduce noise from thermal throttling, background processes, or runner variability).
    - A "locked baseline" is defined as the median performance of the last stable `main` commit measured under the same controlled environment.

This avoids flaky CI due to CPU throttling, background processes, or runner variability.

---

## 13. Code Review Checklist

Before requesting review, the author must verify:

- [ ] Code compiles without warnings on GCC and Clang.
- [ ] All tests pass locally.
- [ ] New functionality has unit tests.
- [ ] Documentation is added/updated for public APIs.
- [ ] No memory leaks (Valgrind/ASan).
- [ ] Commit messages follow the conventional format.
- [ ] No TODOs/FIXMEs without an associated issue number.
- [ ] No `std::cout` or logging in production code (see Rule 1.7).

Reviewers must focus on:
- **Correctness** (logic, edge cases, invariants).
- **Efficiency** (avoiding unnecessary copies, ensuring O(n) vs O(n²) where appropriate).
- **Clarity** (is the code understandable?).
- **Test coverage** (are error paths tested?).

---

## 14. Performance Rules

- **Micro-benchmarks** (Google Benchmark) must be written for compute-heavy functions (MatMul, reductions, broadcasting).
- **Optimizations are optional** and must be guarded by `#ifdef` (e.g., `WITH_AVX2`) with a scalar fallback.
- **No performance pessimization:** Avoid expensive operations (e.g., dynamic allocations) in tight loops. Use preallocated buffers where possible.
- **Profiling:** Use `perf` and `cachegrind` to identify bottlenecks before optimizing.
- **`std::function` usage:** Prohibited in performance-critical runtime hot paths (e.g., inner loops, per-element operations, autograd backward of large tensors). It may be used in initialization, graph discovery, testing, or other non-hot-path code when it improves clarity.

---

## 15. Prohibited Practices

The following are **forbidden** in the codebase:

- ❌ `std::endl` (use `'\n'` instead).
- ❌ Raw `new`/`delete` (except inside `Storage` with custom deleter).
- ❌ C-style casts (use `static_cast`, `reinterpret_cast`, `const_cast`).
- ❌ Global mutable state (see Rule 1.8).
- ❌ `std::vector<bool>` (use `std::vector<char>` or a bit container).
- ❌ `std::shared_ptr` cycles (use `std::weak_ptr` to break them).
- ❌ In-place tensor mutation (not allowed in v1).
- ❌ Using `std::function` in performance-critical paths (see Rule 14).
- ❌ Using `std::bind` (use lambdas).
- ❌ Uncontrolled stdout/stderr output from library code (see Rule 1.7).

---

## 16. Special Rules for Autograd

- **All `backward()` implementations must be graph-free** (they must not create new `FunctionNode`s). Use internal no-grad kernels.
- **Gradient buffers** must have `requires_grad == false` and `grad_fn == nullptr`.
- **`saved_tensors_`** must only contain tensors required for `backward()`. Do not save unnecessary data.
- **`make_parent_edge()`** is the only allowed way to create `ParentEdge` objects from operations.
- **`set_requires_grad()`** on non-leaf tensors is prohibited and must throw `std::logic_error`.

---

## 17. Document Updates

- `rules.md` is a living document and may be updated via PR.
- Any substantive change to the rules requires discussion and approval from at least two contributors.
- The version number and date must be updated with each change.

---

*End of Rules Document v1.2*
```