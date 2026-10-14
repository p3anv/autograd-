# Contributing to AutoGrad++

Thank you for your interest in contributing! This document outlines the guidelines and workflow for contributing to AutoGrad++.

---

## 📋 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to the maintainers.

---

## 🛠 Development Environment

### Prerequisites
- **C++20** compatible compiler (GCC 13+, Clang 16+, MSVC 19.35+)
- **CMake 3.20+**
- **Git**
- **Google Test** (fetched via FetchContent)
- **Google Benchmark** (optional, fetched via FetchContent)

### Recommended Setup
```bash
# Fork and clone your fork
git clone https://github.com/yourusername/autograd-tensor-library.git
cd autograd-tensor-library

# Add upstream remote
git remote add upstream https://github.com/original/autograd-tensor-library.git

# Create development build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON -DWITH_AVX2=ON
cmake --build . -j$(nproc)

# Verify tests pass
ctest --output-on-failure
```

---

## 🔄 Development Workflow

### 1. Create an Issue First
Before starting work, please:
- Check existing [issues](https://github.com/yourusername/autograd-tensor-library/issues)
- Create a new issue describing the problem/feature
- Discuss the approach with maintainers

### 2. Branch Naming
Use descriptive branch names:
```
feature/matmul-avx512-support
fix/gradient-accumulation-broadcast
refactor/tensor-view-api
docs/update-readme-benchmarks
```

### 3. Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
type(scope): short description

Longer description if needed.

Fixes #123
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code restructuring
- `docs` - Documentation changes
- `test` - Test additions/modifications
- `perf` - Performance improvements
- `build` - Build system changes
- `ci` - CI configuration changes

**Examples:**
```
feat(simd): add AVX2 softmax kernel with fused exp/sum pass
fix(autograd): resolve circular reference in MaxBackward
refactor(tensor): simplify view creation logic
test(simd): add equivalence tests for log_softmax
perf(matmul): optimize B matrix packing for column-major access
```

### 4. Pull Request Process

1. **Sync with upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your branch:**
   ```bash
   git push origin feature/your-feature
   ```

3. **Open PR** against `main` branch with:
   - Clear title following conventional commits
   - Description of changes
   - Link to related issue
   - Test results (run `ctest --output-on-failure`)

4. **CI Checks** must pass:
   - Build (Debug + Release)
   - All 177 tests pass
   - Sanitizers clean (ASan, UBSan)
   - Code formatting (clang-format)

5. **Code Review** - Address feedback promptly

6. **Merge** - Maintainers will squash and merge

---

## 🎨 Coding Standards

### C++20 Style Guide

#### General
- **Modern C++20**: Use concepts, ranges, `std::format`, structured bindings
- **RAII**: All resources managed by smart pointers or containers
- **Value semantics**: `Tensor` is a value-semantic handle
- **No raw owning pointers**: Use `std::unique_ptr`, `std::shared_ptr`, `std::vector`
- **Const correctness**: Mark everything `const` by default
- **`noexcept`**: Use where appropriate

#### Naming Conventions
| Entity | Convention | Example |
|--------|------------|---------|
| Classes/Structs | PascalCase | `TensorImpl`, `FunctionNode` |
| Functions/Methods | snake_case | `matmul()`, `make_contiguous()` |
| Variables | snake_case | `input_shape`, `grad_output` |
| Constants | kPascalCase | `kDefaultAlignment` |
| Template Params | PascalCase | `typename Tag`, `size_t Dim` |
| Namespaces | snake_case | `autograd::simd` |
| Macros | UPPER_SNAKE | `AUTOGRAD_ASSERT` |

#### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: 100 characters max
- **Braces**: Allman style for functions, K&R for control flow
- **Includes**: System → Project → Local, alphabetical within groups
- **`auto`**: Use for complex types, explicit for primitives

```cpp
// ✅ Good
auto result = matmul(a, b);           // Complex return type
const float learning_rate = 0.01f;    // Primitive

// ❌ Avoid
auto x = 42;                          // Primitive, use explicit type
```

#### Error Handling
- **Exceptions** for programming errors (invalid arguments, logic errors)
- **`std::expected`** (C++23) or `std::optional` for recoverable failures
- **Assertions** for internal invariants (`AUTOGRAD_ASSERT`)

```cpp
// ✅ Good
if (dim >= shape.size()) {
    throw std::invalid_argument("Dimension out of range");
}

// ✅ Good (internal invariant)
AUTOGRAD_ASSERT(data_ != nullptr, "Storage must be allocated");
```

#### SIMD Code Guidelines
- **Per-TU compilation**: AVX2 kernels in separate `.cpp` with `-mavx2 -mfma`
- **Runtime dispatch only**: Never `#ifdef` in hot paths
- **Tag-based dispatch**: `ScalarTag`, `SseTag`, `Avx2Tag`
- **Alignment**: Assume 64-byte aligned inputs for AVX2
- **Numerical equivalence**: Validate against scalar reference (≤1e-4 rel. error)

---

## 🧪 Testing Requirements

### Test Categories
| Category | Location | When to Add |
|----------|----------|-------------|
| Unit tests | `tests/test_*.cpp` | Every new function/class |
| Integration | `tests/test_autograd.cpp` | Cross-component features |
| Numerical | `tests/test_simd.cpp` | New SIMD kernels |
| Property-based | `tests/hardcore_test.cpp` | Mathematical properties |

### Test Naming
```cpp
TEST(ComponentName, SpecificBehavior) {
    // Test name describes expected behavior
}
```

### Requirements for New Code
- **Unit tests** for all public APIs
- **Edge cases**: empty tensors, zero-sized dims, broadcasting
- **Numerical tests** for SIMD kernels (AVX2 vs scalar)
- **Gradient tests** for autograd operations

### Running Tests
```bash
# All tests
ctest --output-on-failure

# Specific suite
./autograd_tests --gtest_filter="TensorTest.*"
./hardcore_tests

# With ASan (leaks disabled for test infrastructure)
ASAN_OPTIONS=detect_leaks=0 ./autograd_tests

# Valgrind (build without sanitizers)
cd ../build_valgrind && valgrind --leak-check=full ./autograd_tests
```

---

## 📝 Documentation

### API Documentation
- Document all public APIs with Doxygen comments
- Include `@param`, `@return`, `@throws`, `@see`
- Provide usage examples for complex functions

```cpp
/**
 * @brief Matrix multiplication with optional AVX2 acceleration.
 * @param a Left operand [M, K], must be contiguous
 * @param b Right operand [K, N], must be contiguous
 * @return Result tensor [M, N]
 * @throws std::invalid_argument if inner dimensions mismatch or non-contiguous
 * @see matmul(), transpose()
 */
Tensor matmul(const Tensor& a, const Tensor& b);
```

### README Updates
- Update feature tables for new operations
- Add benchmark results for performance changes
- Keep architecture diagram current

---

## 🔍 Code Review Checklist

### For Authors (Self-Review)
- [ ] All tests pass (`ctest --output-on-failure`)
- [ ] No sanitizer warnings (ASan, UBSan)
- [ ] No Valgrind leaks (core library)
- [ ] Follows coding standards
- [ ] Documentation updated
- [ ] No TODO/FIXME without issue reference
- [ ] Commit messages follow conventions

### For Reviewers
- [ ] Correctness: Logic matches specification
- [ ] Safety: No memory leaks, UB, data races
- [ ] Performance: No regressions, SIMD used appropriately
- [ ] Maintainability: Clear, documented, testable
- [ ] Consistency: Follows existing patterns

---

## 🏷 Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
1. Update `CMakeLists.txt` version
2. Update `CHANGELOG.md`
3. Tag release: `git tag -a v1.2.0 -m "Release v1.2.0"`
4. Push tag: `git push origin v1.2.0`
5. CI builds release artifacts
6. Create GitHub Release with notes

---

## 🐛 Reporting Bugs

### Before Reporting
- Search existing issues
- Test with latest `main` branch
- Try to reproduce with minimal example

### Bug Report Template
```markdown
**Description:** Clear description of the bug

**Steps to Reproduce:**
1. Create tensor with shape...
2. Call operation...
3. Observe incorrect result...

**Expected Behavior:** What should happen

**Actual Behavior:** What actually happens

**Environment:**
- OS: Ubuntu 24.04
- Compiler: GCC 13.2
- Build type: Debug/Release
- AVX2: ON/OFF

**Minimal Code Example:**
```cpp
// Minimal reproducing code
```

**Additional Context:** Logs, stack traces, etc.
```

---

## 💡 Feature Requests

### Before Requesting
- Check roadmap in README
- Search existing issues
- Consider if it fits project scope

### Feature Request Template
```markdown
**Problem:** What problem does this solve?

**Proposed Solution:** High-level approach

**Alternatives Considered:** Other approaches

**API Design:** Sketch of public API

**Testing Strategy:** How to validate

**Performance Impact:** Expected speedup/regression
```

---

## 🏗 Architecture Decisions

### When to Create ADR
For significant architectural decisions, create an Architecture Decision Record (ADR) in `docs/adr/`:
- New major component
- API breaking changes
- Performance-critical design choices
- Dependency additions

### ADR Template
```markdown
# ADR-XXX: Title

## Status
Proposed | Accepted | Superseded

## Context
What is the issue?

## Decision
What was decided?

## Consequences
Positive and negative impacts
```

---

## 📞 Getting Help

- **GitHub Discussions**: Questions, ideas, general discussion
- **GitHub Issues**: Bugs, feature requests
- **Discord/Slack**: Real-time chat (if available)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

*Thank you for contributing to AutoGrad++!* 🚀