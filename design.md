# Design: C++ Autograd Tensor Library

**Version:** 7.0 (Final — Fully Correctness-Hardened)  
**Date:** 2026-08-17  
**Based On:** PRD v1.3 + Design v6.0 + final contract/lifetime/gradient-state corrections  
**Status:** LOCKED — READY FOR IMPLEMENTATION (VERIFIED)

---

## 1. Core Design Principle: Layer Separation

Each component has exactly one primary reason to exist. Layers must not bleed into each other.

| Layer | Responsibility | Owns/References |
| :--- | :--- | :--- |
| **Tensor** | User-facing value-semantic handle. | `shared_ptr<TensorImpl>` |
| **TensorImpl** | Tensor metadata + autograd state for one logical view. | `shared_ptr<Storage>` + gradient buffer |
| **Storage** | Raw aligned memory ownership. | RAII-owned allocation |
| **FunctionNode** | Differentiable operation definition + backward rule. | Parent edges + only tensors required by backward |
| **ParentEdge** | Explicit edge from an operation to one of its input parents. | Owning parent node or leaf `TensorImpl` |
| **AutogradEngine** | DAG traversal, gradient propagation, and temporary gradient state. | Root node + temporary maps |
| **Kernel** | Numerical computation only. | Raw pointers / spans; never creates autograd nodes |

**Dtype:** v1 supports `float32` only. `float64` is out of scope for the MVP.

### Fundamental invariants

1. A `Tensor` is a handle; it does not own tensor bytes directly.
2. A `TensorImpl` owns view metadata and references one `Storage`.
3. Copying a Tensor shares its `TensorImpl`.
4. Creating a view creates a new `TensorImpl` sharing the same `Storage`.
5. `Storage` owns the allocation exactly once.
6. `grad_fn != nullptr` implies `requires_grad == true`.
7. A leaf requiring gradients has `grad_fn == nullptr`.
8. Internal gradient buffers never require gradients and never have a `grad_fn`.
9. Backward execution never creates new autograd nodes.
10. Every `ParentEdge` has exactly one target: parent node XOR leaf tensor.
11. Every returned parent gradient corresponds to exactly one `parents_` entry, in identical order.
21. Optimized kernels must agree with independent reference kernels within the documented numerical tolerance.

---

## 2. Core Classes & Ownership Model

### 2.1 Storage (Memory Ownership)

**Purpose:** Owns one contiguous aligned allocation. Storage knows nothing about shapes, strides, views, or autograd.

```cpp
class Storage {
public:
    explicit Storage(size_t size);

    float* data() noexcept;
    const float* data() const noexcept;
    size_t size() const noexcept;

private:
    std::unique_ptr<float[], AlignedDeleter> data_;
    size_t size_ = 0;
};
```

### Allocation contract

- Allocation alignment is 64 bytes.
- `size == 0` is valid and produces no allocation; `data()` returns `nullptr`.
- Before `size * sizeof(float)`, check:
  `size <= SIZE_MAX / sizeof(float)`.
- Before adding the alignment padding, check:
  `bytes <= SIZE_MAX - 63`.
- Compute:
  `aligned_bytes = ((bytes + 63) / 64) * 64`.
- `std::aligned_alloc(64, aligned_bytes)` is used by the v1 implementation.
- Deallocation uses the matching `std::free` through `AlignedDeleter`.
- Allocation failure throws `std::bad_alloc`.
- Overflow is rejected with `std::invalid_argument`.

The implementation must never perform pointer arithmetic on `nullptr`.

---

### 2.2 TensorImpl (Metadata + Autograd State)

**Purpose:** Represents one logical tensor view and its autograd metadata.

```cpp
class TensorImpl {
public:
    TensorImpl(std::shared_ptr<Storage> storage,
               std::vector<size_t> shape,
               std::vector<ptrdiff_t> strides,
               size_t offset = 0);

    static std::shared_ptr<TensorImpl>
    make_contiguous(std::vector<size_t> shape);

    size_t numel() const;
    size_t dim() const;
    bool is_contiguous() const;

    float* data();
    const float* data() const;

    bool requires_grad_ = false;
    std::shared_ptr<FunctionNode> grad_fn_;

    // Gradient storage is independent tensor state.
    std::shared_ptr<TensorImpl> grad_impl_;

    void accumulate_grad(const Tensor& grad); // requires_grad_ must be true

private:
    std::shared_ptr<Storage> storage_;
    std::vector<size_t> shape_;
    std::vector<ptrdiff_t> strides_;
    size_t offset_ = 0;
};
```

### Copy vs view

- `Tensor B = A` shares the exact same `TensorImpl`.
- `Tensor B = A.slice(...)` creates a new `TensorImpl`.
- The new view shares `Storage` but has its own shape, strides, and offset.

### Shape and stride invariants

- Rank is at most 8.
- Shape dimensions are non-negative by construction (`size_t`).
- `numel` multiplication is overflow-checked.
- Zero-sized tensors are valid.
- A scalar tensor has rank 0 and `numel == 1`.
- v1 views use non-negative strides; reverse/negative-stride views are out of scope.
- The logical address of every valid element must remain within the underlying storage.
- View construction validates shape/stride/offset against storage bounds.
- No view may reference bytes outside its storage.
- Overlapping views are allowed only when created by explicitly defined view operations; v1 does not provide mutable in-place operations.

### Zero-sized tensors

If any dimension is zero, `numel == 0`.

For such tensors:

- `data()` returns `nullptr`.
- No pointer arithmetic is performed.
- Element access is invalid and throws.
- Reference/optimized kernels must correctly handle `n == 0`.

### `data()` contract

`data()` is only a contiguous-data API.

- Contiguous, non-empty tensor → pointer to logical element 0.
- Contiguous, empty tensor → `nullptr`.
- Non-contiguous tensor → throws `std::runtime_error`.
- It never silently materializes a contiguous copy.

---

### 2.3 Tensor (Public Handle)

```cpp
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    static Tensor scalar(float value, bool requires_grad = false);

    explicit Tensor(std::vector<size_t> shape,
                    bool requires_grad = false);

    static Tensor zeros(std::vector<size_t> shape,
                        bool requires_grad = false);
    static Tensor ones(std::vector<size_t> shape,
                       bool requires_grad = false);
    static Tensor arange(size_t start, size_t end);

    size_t numel() const;
    size_t dim() const;
    bool is_contiguous() const;
    std::vector<size_t> shape() const;

    float* data();
    const float* data() const;
    float& operator()(std::initializer_list<size_t> idx);

    bool requires_grad() const;
    void set_requires_grad(bool val);
    Tensor detach() const;
    Tensor grad() const;

    void backward(const Tensor& grad_output);
    void backward();  // scalar tensors only

    Tensor slice(size_t dim, size_t start, size_t end) const;
    Tensor narrow(size_t dim, size_t start, size_t length) const;
    Tensor reshape(std::vector<size_t> new_shape) const;
    Tensor contiguous() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    // Internal access only; not part of the public user API.
    std::shared_ptr<TensorImpl> impl() const noexcept { return impl_; }

private:
    std::shared_ptr<TensorImpl> impl_;
};
```

### `set_requires_grad()` contract

`set_requires_grad()` may only change the flag on leaf tensors.

- Setting it on a leaf is allowed.
- Attempting to change it on a non-leaf tensor with `grad_fn != nullptr` throws `std::logic_error`.
- Disabling gradients on a leaf clears its accumulated gradient buffer.
- Enabling gradients on a leaf does not create a `FunctionNode`.

This preserves the invariant:

```text
grad_fn != nullptr → requires_grad == true
```

### `backward()` contract

```text
backward(seed):
    1. Reject an empty Tensor.
    2. Require seed.shape == output.shape.
    3. If requires_grad == false, return without building/traversing a graph.
    4. If leaf (grad_fn == nullptr), accumulate seed into the leaf gradient.
    5. Otherwise run AutogradEngine from grad_fn with the seed.
```

`backward()` without an argument is allowed only when `numel() == 1`; it seeds the graph with scalar `1.0f`.

A non-scalar output requires an explicit seed with exactly the same shape.

---

## 3. ParentEdge and Graph Ownership

### 3.1 ParentEdge

```cpp
struct ParentEdge {
    std::shared_ptr<FunctionNode> parent_node;
    std::shared_ptr<TensorImpl> leaf_impl;
    size_t parent_index;

    ParentEdge(std::shared_ptr<FunctionNode> node, size_t idx);
    ParentEdge(std::shared_ptr<TensorImpl> leaf, size_t idx);

    bool is_node() const noexcept;
    bool is_leaf() const noexcept;
    bool is_valid() const noexcept;
};
```

### Invariant

Exactly one target is present:

```text
(parent_node != nullptr) XOR (leaf_impl != nullptr)
```

Constructors reject null targets.

`parent_index` is the index of the corresponding parent in the operation's input ordering.

Each FunctionNode must validate before execution that:
- `parents_[i].parent_index == i` for every parent entry; and
- every parent index is within `[0, parents_.size())`.

The constructors validate the target pointer. Parent-order/index validation is performed when the operation node is finalized/constructed because only the completed parent list knows its valid range.

### Ownership

Parent edges are **owning** references. They keep graph nodes or leaf tensor implementations alive for the duration of the graph.

No `shared_ptr` ownership cycle is permitted.

A FunctionNode may retain tensors required for backward, but must not retain unnecessary tensors.

Saved tensors may retain upstream graph state when required to keep backward valid.
After a node has executed its `backward()` successfully, implementations should
release saved tensors as soon as the node is no longer needed. v1 does not retain
saved tensors for a second backward unless explicitly supported by the API.

---

### 3.2 `make_parent_edge`

All operation nodes must use one helper:

```cpp
ParentEdge make_parent_edge(const Tensor& parent, size_t index) {
    auto impl = parent.impl();

    if (!impl) {
        throw std::invalid_argument("Cannot create graph edge from empty tensor");
    }

    if (impl->grad_fn_) {
        if (!impl->requires_grad_) {
            throw std::logic_error(
                "Tensor invariant violated: grad_fn exists but requires_grad is false");
        }
        return ParentEdge(impl->grad_fn_, index);
    }

    return ParentEdge(impl, index);
}
```

This prevents operations from accidentally treating intermediate tensors as leaves.

---

## 4. FunctionNode

```cpp
class FunctionNode {
public:
    explicit FunctionNode(std::vector<size_t> output_shape)
        : output_shape_(std::move(output_shape)) {}

    virtual ~FunctionNode() = default;

    // v1 operations are single-output. output_shape_ records the
    // exact shape of that operation's output.
    const std::vector<size_t>& output_shape() const noexcept {
        return output_shape_;
    }

    // grad_outputs.size() == number of outputs (exactly 1 in v1).
    // Returned vector size == parents_.size().
    // Returned entry i is the gradient for parents_[i].
    virtual std::vector<Tensor>
    backward(const std::vector<Tensor>& grad_outputs) = 0;

    std::vector<ParentEdge> parents_;

    // Only tensors actually required by backward().
    std::vector<Tensor> saved_tensors_;

protected:
    std::vector<size_t> output_shape_;
};
```

### FunctionNode construction invariant

Every concrete FunctionNode constructor must initialize the base `FunctionNode`
with the exact output shape of the Tensor it represents. This makes
`output_shape()` mandatory and prevents an operation from being inserted into
the graph without usable output-shape metadata.

### Critical backward invariant

**Backward implementations are graph-free.**

Every tensor returned by `FunctionNode::backward()` must have:

```text
requires_grad == false
grad_fn == nullptr
```

Backward code must use internal no-grad kernels/helpers. It must never call public autograd-tracked operations in a way that creates new FunctionNodes.

This invariant prevents backward from recursively constructing another computation graph.

---

### 4.1 AddBackward

Addition needs no saved tensors:

```cpp
class AddBackward : public FunctionNode {
public:
    AddBackward(const Tensor& a, const Tensor& b, const std::vector<size_t>& output_shape)
        : FunctionNode(output_shape) {
        parents_.push_back(make_parent_edge(a, 0));
        parents_.push_back(make_parent_edge(b, 1));
    }

    std::string name() const override { return "AddBackward"; }

    std::vector<Tensor>
    backward(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs.at(0);

        // Graph-free internal copies/views as required.
        return {
            internal::no_grad_copy_or_view(grad),
            internal::no_grad_copy_or_view(grad)
        };
    }
};
```

For broadcasted addition, each returned gradient is reduced back to its parent's original shape using an internal graph-free broadcast-gradient kernel.

---

### 4.2 MatMulBackward

v1 MatMul is rank-2 only.

```cpp
class MatMulBackward : public FunctionNode {
public:
    MatMulBackward(const Tensor& a, const Tensor& b, const std::vector<size_t>& output_shape)
        : FunctionNode(output_shape) {
        parents_.push_back(make_parent_edge(a, 0));
        parents_.push_back(make_parent_edge(b, 1));
        saved_tensors_ = {a, b};
    }

    std::string name() const override { return "MatMulBackward"; }

    std::vector<Tensor>
    backward(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs.at(0);
        const Tensor& a = saved_tensors_.at(0);
        const Tensor& b = saved_tensors_.at(1);

        // Backward MUST NOT call public autograd-tracked matmul().
        // Transpose may be a non-contiguous view, so explicitly materialize
        // contiguous operands before the rank-2 contiguous-only kernel.
        Tensor bt = internal::transpose_no_grad(b).contiguous();
        Tensor at = internal::transpose_no_grad(a).contiguous();

        Tensor grad_a =
            internal::matmul_no_grad(grad, bt);

        Tensor grad_b =
            internal::matmul_no_grad(at, grad);

        return {grad_a, grad_b};
    }
};
```

The exact internal implementation may optimize these paths later, but the semantics are fixed:

```text
backward operation
    ↓
internal no-grad computation
    ↓
gradient tensor with no grad_fn
```

---

## 5. Autograd Engine

### 5.1 Temporary gradient map

```cpp
class AutogradEngine {
public:
    void run(std::shared_ptr<FunctionNode> root_node,
             const Tensor& root_grad);

private:
    std::unordered_map<FunctionNode*, Tensor> node_grads_;
};
```

`node_grads_` belongs to one backward invocation and is destroyed when that invocation completes.

It is not stored globally and is not part of FunctionNode state.

---

## 6. Correct DAG Traversal and Backward Execution

### 6.1 Why ordinary DFS reversal is insufficient

The computation graph is a DAG, not necessarily a tree.

Example:

```text
          root
         /    \
       A        B
        \      /
         shared
```

`shared` must not execute until **both** `A` and `B` have propagated their gradients into it.

Therefore a simple DFS followed by reversal is not sufficient.

### 6.2 Required algorithm

The engine constructs a topological execution order using dependency counts.

Graph direction:

```text
child operation → parent operation
```

A node may execute only after **all of its child operations in the reachable graph have executed**.

Algorithm:

1. Discover every reachable FunctionNode from the root.
2. For every discovered node, count its number of reachable child nodes.
   - This is the number of downstream nodes that depend on it.
3. Initialize the ready queue with the root node, whose child count is zero.
4. Pop a ready node.
5. Execute its `backward()` using the fully accumulated `node_grads_[node]`.
6. Propagate each returned gradient:
   - leaf edge → accumulate into leaf gradient;
   - node edge → accumulate into `node_grads_[parent]`.
7. Decrement the pending-child count of every parent node.
8. When a parent's pending-child count reaches zero, enqueue it.
9. Continue until all reachable nodes have executed.

This guarantees:

```text
Every child gradient contribution
        ↓
is accumulated
        ↓
before the parent node's backward()
```

### 6.3 Gradient normalization and parent-shape validation

Before a gradient is inserted into `node_grads_`, the engine normalizes it to a
contiguous, graph-free tensor. This guarantees that later internal buffer
operations can safely use `data()`.

```text
incoming gradient
      ↓
validate shape
      ↓
ensure contiguous
      ↓
ensure requires_grad == false and grad_fn == nullptr
      ↓
store in node_grads_
```

For a leaf edge, the expected parent shape is `edge.leaf_impl->shape()`.

For a node edge, the expected parent shape is `edge.parent_node->output_shape()`.

Every v1 FunctionNode has exactly one output, so `output_shape_` is the shape
of the Tensor represented by that node.

`normalize_gradient_for_engine()` must return a graph-free contiguous Tensor.
It may reuse an already-contiguous graph-free buffer or create a new one.

### 6.4 Pseudocode

```cpp
void AutogradEngine::run(
    std::shared_ptr<FunctionNode> root_node,
    const Tensor& root_grad)
{
    node_grads_.clear();

    if (!root_node)
        throw std::invalid_argument("Autograd root node is null");

    if (root_grad.shape() != root_node->output_shape())
        throw std::invalid_argument(
            "Root gradient shape must match root output shape");

    if (root_grad.requires_grad() || root_grad.impl()->grad_fn_)
        throw std::logic_error(
            "Autograd root gradient must be graph-free");

    node_grads_[root_node.get()] =
        internal::normalize_gradient_for_engine(root_grad);

    std::vector<FunctionNode*> nodes;
    std::unordered_set<FunctionNode*> visited;

    std::function<void(FunctionNode*)> discover =
        [&](FunctionNode* node) {
            if (!node || !visited.insert(node).second)
                return;

            nodes.push_back(node);

            for (const auto& edge : node->parents_) {
                if (edge.is_node())
                    discover(edge.parent_node.get());
            }
        };

    discover(root_node.get());

    // Number of downstream child nodes that must execute before
    // each node can execute.
    std::unordered_map<FunctionNode*, size_t> pending_children;

    for (FunctionNode* node : nodes)
        pending_children[node] = 0;

    for (FunctionNode* child : nodes) {
        std::unordered_set<FunctionNode*> unique_parents;

        for (const auto& edge : child->parents_) {
            if (edge.is_node())
                unique_parents.insert(edge.parent_node.get());
        }

        for (FunctionNode* parent : unique_parents)
            ++pending_children[parent];
    }

    std::deque<FunctionNode*> ready;
    ready.push_back(root_node.get());

    size_t executed = 0;

    while (!ready.empty()) {
        FunctionNode* node = ready.front();
        ready.pop_front();

        auto grad_it = node_grads_.find(node);
        if (grad_it == node_grads_.end())
            throw std::logic_error(
                "Autograd invariant violated: node has no accumulated gradient");

        Tensor grad_of_output = grad_it->second;

        if (grad_of_output.shape() != node->output_shape())
            throw std::logic_error(
                "Accumulated node gradient has incorrect shape");

        // backward() must be graph-free.
        std::vector<Tensor> grads_for_parents =
            node->backward({grad_of_output});

        if (grads_for_parents.size() != node->parents_.size())
            throw std::logic_error(
                "Backward returned incorrect number of parent gradients");

        for (size_t i = 0; i < node->parents_.size(); ++i) {
            const auto& edge = node->parents_[i];

            if (edge.parent_index != i)
                throw std::logic_error(
                    "ParentEdge index does not match parent ordering");

            Tensor grad =
                internal::normalize_gradient_for_engine(grads_for_parents[i]);

            const std::vector<size_t>& expected_shape =
                edge.is_leaf()
                    ? edge.leaf_impl->shape()
                    : edge.parent_node->output_shape();

            if (grad.shape() != expected_shape)
                throw std::logic_error(
                    "Backward returned gradient with incorrect shape");

            if (grad.requires_grad() || grad.impl()->grad_fn_)
                throw std::logic_error(
                    "Backward produced a tensor requiring gradients");

            if (edge.is_leaf()) {
                if (edge.leaf_impl->requires_grad_) {
                    edge.leaf_impl->accumulate_grad(grad);
                }
            } else {
                FunctionNode* parent = edge.parent_node.get();

                auto it = node_grads_.find(parent);

                if (it == node_grads_.end()) {
                    node_grads_.emplace(parent, std::move(grad));
                } else {
                    internal::add_to_buffer(
                        it->second.data(),
                        grad.data(),
                        it->second.numel());
                }

                if (pending_children[parent] == 0)
                    throw std::logic_error(
                        "Autograd dependency count underflow");

                --pending_children[parent];

                if (pending_children[parent] == 0)
                    ready.push_back(parent);
            }
        }

        ++executed;
    }

    if (executed != nodes.size())
        throw std::logic_error(
            "Autograd graph traversal did not execute all reachable nodes");
}
```cpp
void TensorImpl::accumulate_grad(const Tensor& grad) {
    if (!requires_grad_) {
        throw std::logic_error(
            "Cannot accumulate a gradient into a tensor that does not require gradients");
    }

    if (grad.shape() != shape_) {
        throw std::invalid_argument(
            "Gradient shape must match tensor shape");
    }

    if (grad.requires_grad() || grad.impl()->grad_fn_) {
        throw std::logic_error(
            "Cannot accumulate a gradient that requires gradients");
    }

    if (!grad_impl_) {
        auto fresh = TensorImpl::make_contiguous(shape_);

        internal::copy_buffer(
            fresh->data(),
            grad.data(),
            numel());

        fresh->requires_grad_ = false;
        fresh->grad_fn_.reset();

        grad_impl_ = std::move(fresh);
        return;
    }

    internal::add_to_buffer(
        grad_impl_->data(),
        grad.data(),
        numel());
}
```

### Gradient invariants

- `grad_impl_` is always contiguous.
- `grad_impl_->requires_grad_ == false`.
- `grad_impl_->grad_fn_ == nullptr`.
- Gradient shape equals source tensor shape.
- First accumulation copies into fresh storage.
- Later accumulations use graph-free raw buffer addition.
- No atomic floating-point gradient updates are used in v1.

---

## 8. Views, Reshape, Detach, and Mutation

### No in-place tensor mutation in v1

The public API does not provide:

```text
x += y
x *= y
x[index] = value
```

This prevents storage mutation from invalidating saved autograd values.

### `detach()`

`detach()`:

- shares the same underlying Storage;
- creates a new TensorImpl;
- preserves shape/stride/offset;
- sets `requires_grad == false`;
- has `grad_fn == nullptr`;
- has no gradient buffer.

### `reshape()`

- Zero-copy when the existing layout permits it.
- Non-contiguous incompatible reshape throws.
- No silent copy.
- Caller may explicitly call `contiguous()` first.

### `contiguous()`

- Returns the same logical values in row-major contiguous storage.
- If already contiguous, it may return a Tensor sharing the existing TensorImpl.
- Otherwise it allocates new Storage and copies values through a graph-free internal kernel.

---

## 9. Operation Dispatch

### Public forward operations

Public operations may construct FunctionNodes when one or more inputs require gradients.
Every created FunctionNode must initialize `output_shape_` from the exact shape of
the resulting Tensor and must validate parent ordering before returning the result.

Example:

```cpp
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2)
        throw std::invalid_argument(
            "matmul requires rank-2 tensors");

    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument(
            "matmul dimension mismatch");

    if (!a.is_contiguous() || !b.is_contiguous())
        throw std::invalid_argument(
            "matmul requires contiguous inputs. Call .contiguous() first.");

    Tensor result = internal::matmul_no_grad(a, b);

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<MatMulBackward>(
            a, b, result.shape());
        result.impl()->grad_fn_ = node;
        result.impl()->requires_grad_ = true;
    }

    return result;
}
```

### Critical separation

```text
Public operation
    → may create FunctionNode

Internal no-grad operation
    → NEVER creates FunctionNode
```

Backward implementations must use the second category.

---

## 10. Error Handling

| Scenario | Exception |
| :--- | :--- |
| Dimension mismatch | `std::invalid_argument` |
| Non-contiguous MatMul input | `std::invalid_argument` |
| MatMul on non-rank-2 | `std::invalid_argument` |
| Out-of-bounds indexing | `std::out_of_range` |
| Incompatible reshape | `std::invalid_argument` |
| Storage size overflow | `std::invalid_argument` |
| Shape product overflow | `std::invalid_argument` |
| Wrong backward seed shape | `std::invalid_argument` |
| Null ParentEdge target | `std::invalid_argument` |
| Internal autograd invariant violation | `std::logic_error` |
| Allocation failure | `std::bad_alloc` |

---

## 11. Kernel Architecture

### Reference kernels

Reference kernels are deliberately simple and independent of optimized indexing/tiling logic.

```cpp
namespace internal {

void add_to_buffer(float* dst,
                   const float* src,
                   size_t n);

void copy_buffer(float* dst,
                 const float* src,
                 size_t n);

void matmul_ref(const float* a,
                const float* b,
                float* out,
                size_t m,
                size_t k,
                size_t n);
}
```

### Optimized kernels

```cpp
#ifdef WITH_AVX2
namespace optimized {

void matmul(const float* a,
            const float* b,
            float* out,
            size_t m,
            size_t k,
            size_t n);

}
#endif
```

Optimized kernels:

- never call reference kernels internally;
- must be tested against reference kernels;
- must preserve tensor shape and numerical semantics;
- must not create autograd nodes;
- may use SIMD/cache blocking/threading according to the PRD and later performance evidence.

---

## 12. Required Design-Level Tests

Before optimization, tests must establish:

### Tensor/storage

- scalar tensors;
- zero-sized tensors;
- rank 1–8 tensors;
- shape-product overflow;
- storage-byte overflow;
- 64-byte alignment;
- copy vs view semantics;
- offset/stride correctness;
- contiguous vs non-contiguous data access.

### Autograd

At minimum:

```text
x + y
x * y
x / y
sum
mean
reshape
broadcasting
matmul
ReLU
Sigmoid
Tanh
Softmax
cross-entropy
```

### Graph tests

Explicitly test:

```text
linear chain:
x → A → B → loss

branch:
x → A
x → B
A,B → loss

shared intermediate:
x,y → A
A → B
A → C
B,C → loss
```

The shared-intermediate case is mandatory because it detects incorrect DFS/reverse traversal.

### Backward graph-free test

Record the number of FunctionNodes before backward and verify it does not increase after backward.

### Gradient execution-state tests

Explicitly verify:

```text
root gradient is contiguous and graph-free
node_grads_ entries are always contiguous and graph-free
parent gradient shape is checked against leaf TensorImpl shape
parent gradient shape is checked against FunctionNode output_shape
non-leaf set_requires_grad(false) is rejected
saved tensors are released after backward when no longer needed
```

### Gradient tests

Compare analytical gradients against finite differences with dtype-appropriate `rtol` and `atol`.

### Reference-kernel tests

For deterministic and randomized inputs:

```text
optimized output ≈ reference output
```

within the documented tolerance.

---

## 13. Final Architecture Invariants

The following rules are locked for implementation:

1. Storage owns bytes; TensorImpl owns view metadata.
2. Tensor copies share TensorImpl; views create TensorImpls sharing Storage.
3. No silent copies from performance-sensitive APIs.
4. v1 is float32-only.
5. No public in-place tensor mutation.
6. Graph edges explicitly distinguish intermediate nodes from leaves.
7. ParentEdge ownership is explicit and cycle-free.
8. `grad_fn != nullptr` implies `requires_grad`.
9. `set_requires_grad()` may only change leaf tensors.
10. Leaf backward accumulates directly into the leaf gradient.
11. A leaf that does not require gradients never receives an accumulated gradient.
12. Every FunctionNode stores the exact shape of its single v1 output.
13. Gradient seed shape must always match the output shape.
14. Every operation returns one parent gradient per ParentEdge, in parent order.
15. Gradient buffers and `node_grads_` entries are contiguous, independent, and graph-free.
16. Backward implementations never construct new FunctionNodes.
17. Backward execution uses a true DAG dependency schedule, not simple DFS reversal.
18. A node executes only after every reachable child has propagated its gradient.
19. Shared intermediates therefore receive all child gradient contributions before executing backward.
20. Saved tensors are retained only as long as required for a valid backward.
21. Optimized kernels are validated against independent reference kernels.
22. All shape, allocation, and view bounds calculations are overflow-safe.
23. Empty tensors never perform pointer arithmetic on null storage.
24. Sanitizers and tests must pass before performance optimization is accepted.

---## 14. Implementation Gate

`design.md` is considered locked only when the first scalar implementation demonstrates:

```text
- storage allocation/deallocation passes ASan + UBSan
- scalar/zero-sized tensor construction works
- copy/view semantics pass tests
- reference kernels pass deterministic fixtures
- linear autograd chain passes
- branching graph passes
- shared-intermediate DAG passes
- leaf backward passes
- backward produces no new FunctionNodes
- gradient accumulation passes repeated-use tests
- FunctionNode output-shape validation passes
- `node_grads_` entries are always contiguous and graph-free
- non-leaf `set_requires_grad()` rejection passes
- non-requires-grad leaf parents do not receive accumulated gradients
- saved-tensor lifetime/release checks pass
- ParentEdge ordering/index validation passes
```

Only after these gates pass may SIMD/cache optimization begin.

---

**End of Design Document v7.0**
