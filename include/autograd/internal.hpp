#pragma once

#include <cstddef>
#include <memory>
#include <algorithm>
#include <stdexcept>

namespace autograd {
    class TensorImpl;
    class FunctionNode;
}

namespace autograd {
namespace internal {

/**
 * @brief Helper class representing an edge from a parent to a child in the autograd graph.
 * A ParentEdge can point to either a leaf tensor (TensorImpl) or an intermediate node (FunctionNode),
 * but not both (XOR invariant).
 */
struct ParentEdge {
    std::shared_ptr<TensorImpl> leaf_parent_;
    std::shared_ptr<FunctionNode> intermediate_parent_;
    std::size_t index_; // index of the parent's output that this edge connects to

    // Constructors enforce XOR invariant: exactly one of leaf_parent_ or intermediate_parent_ is non-null
    ParentEdge() = delete;

    explicit ParentEdge(std::shared_ptr<TensorImpl> leaf_parent, std::size_t index)
        : leaf_parent_(std::move(leaf_parent)), intermediate_parent_(nullptr), index_(index) {
        if (!leaf_parent_) {
            throw std::invalid_argument("leaf_parent must not be null");
        }
    }

    explicit ParentEdge(std::shared_ptr<FunctionNode> intermediate_parent, std::size_t index)
        : leaf_parent_(nullptr), intermediate_parent_(std::move(intermediate_parent)), index_(index) {
        if (!intermediate_parent_) {
            throw std::invalid_argument("intermediate_parent must not be null");
        }
    }

    // Check if this edge points to a leaf parent
    bool is_leaf() const noexcept {
        return leaf_parent_ != nullptr;
    }

    // Check if this edge points to an intermediate parent
    bool is_intermediate() const noexcept {
        return intermediate_parent_ != nullptr;
    }

    // Get the parent tensor impl (if leaf) or null
    std::shared_ptr<TensorImpl> leaf_parent() const noexcept { return leaf_parent_; }

    // Get the parent function node (if intermediate) or null
    std::shared_ptr<FunctionNode> intermediate_parent() const noexcept { return intermediate_parent_; }

    // Get the index
    std::size_t index() const noexcept { return index_; }
};

/**
 * @brief Factory function to create a ParentEdge pointing to a leaf tensor.
 * @param leaf_parent Shared pointer to the leaf tensor
 * @param index Index of the parent's output
 * @return ParentEdge object
 */
inline ParentEdge make_parent_edge(std::shared_ptr<TensorImpl> leaf_parent, std::size_t index) {
    return ParentEdge(std::move(leaf_parent), index);
}

/**
 * @brief Factory function to create a ParentEdge pointing to an intermediate node.
 * @param intermediate_parent Shared pointer to the function node
 * @param index Index of the parent's output
 * @return ParentEdge object
 */
inline ParentEdge make_parent_edge(std::shared_ptr<FunctionNode> intermediate_parent, std::size_t index) {
    return ParentEdge(std::move(intermediate_parent), index);
}

/**
 * @brief Add source buffer to destination buffer element-wise.
 * @param dst Destination buffer
 * @param src Source buffer
 * @param n Number of elements
 */
inline void add_to_buffer(float* dst,
                          const float* src,
                          std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

/**
 * @brief Copy source buffer to destination buffer.
 * @param dst Destination buffer
 * @param src Source buffer
 * @param n Number of elements
 */
inline void copy_buffer(float* dst,
                        const float* src,
                        std::size_t n) {
    std::copy_n(src, n, dst);
}

} // namespace internal
} // namespace autograd