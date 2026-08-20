#pragma once

#include <memory>
#include <vector>
#include <string>
#include "autograd/tensor.hpp"

namespace autograd {

class TensorImpl;
class Storage;

/**
 * @brief Base class for all autograd function nodes.
 * Represents a differentiable operation in the computation graph.
 */
class FunctionNode {
public:
    explicit FunctionNode(std::vector<std::size_t> output_shape);
    virtual ~FunctionNode() = default;

    /**
     * @brief Get the exact shape of this operation's output.
     * @return Reference to output shape vector
     */
    const std::vector<std::size_t>& output_shape() const noexcept;

    /**
     * @brief Compute backward pass of the operation.
     * @param grad_outputs Gradients with respect to outputs (exactly 1 in v1)
     * @return Gradients with respect to inputs (one per parent)
     * @note Must be graph-free: returned tensors must have requires_grad==false and grad_fn==nullptr
     */
    virtual std::vector<Tensor>
    backward(const std::vector<Tensor>& grad_outputs) = 0;

    /**
     * @brief Get name of the operation (for debugging).
     * @return Operation name
     */
    virtual std::string name() const { return "FunctionNode"; }

    // Parent edges (owning references to parents)
    std::vector<std::pair<std::shared_ptr<FunctionNode>, std::size_t>> parents_; // (parent_node, parent_index)
    std::vector<std::pair<std::shared_ptr<TensorImpl>, std::size_t>> parent_leaves_; // (leaf_tensor, parent_index)

    // Tensors saved for backward computation (only those actually needed)
    std::vector<Tensor> saved_tensors_;

protected:
    std::vector<std::size_t> output_shape_;
};

} // namespace autograd