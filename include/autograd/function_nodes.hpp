#pragma once

#include <memory>
#include <vector>
#include <string>
#include "autograd/tensor.hpp"
#include "autograd/internal.hpp"

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
    std::vector<internal::ParentEdge> parents_; // (parent, index)

    // Tensors saved for backward computation (only those actually needed)
    std::vector<Tensor> saved_tensors_;

protected:
    std::vector<std::size_t> output_shape_;
};

/**
 * @brief Backward node for addition operation.
 */
class AddBackward : public FunctionNode {
public:
    AddBackward(const std::vector<std::size_t>& shape_a, const std::vector<std::size_t>& shape_b);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "AddBackward"; }

private:
    std::vector<std::size_t> shape_a_;
    std::vector<std::size_t> shape_b_;
};

/**
 * @brief Backward node for subtraction operation.
 */
class SubBackward : public FunctionNode {
public:
    SubBackward(const std::vector<std::size_t>& shape_a, const std::vector<std::size_t>& shape_b);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "SubBackward"; }

private:
    std::vector<std::size_t> shape_a_;
    std::vector<std::size_t> shape_b_;
};

/**
 * @brief Backward node for multiplication operation.
 */
class MulBackward : public FunctionNode {
public:
    MulBackward(const Tensor& a, const Tensor& b);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "MulBackward"; }

private:
    Tensor a_;
    Tensor b_;
};

/**
 * @brief Backward node for division operation.
 */
class DivBackward : public FunctionNode {
public:
    DivBackward(const Tensor& a, const Tensor& b);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "DivBackward"; }

private:
    Tensor a_;
    Tensor b_;
};

/**
 * @brief Backward node for power operation.
 */
class PowBackward : public FunctionNode {
public:
    PowBackward(const Tensor& base, const Tensor& exponent);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "PowBackward"; }

private:
    Tensor base_;
    Tensor exponent_;
};

/**
 * @brief Backward node for negation operation.
 */
class NegBackward : public FunctionNode {
public:
    NegBackward();
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "NegBackward"; }
};

/**
 * @brief Backward node for exponential operation.
 */
class ExpBackward : public FunctionNode {
public:
    ExpBackward(const Tensor& output);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "ExpBackward"; }

private:
    Tensor output_;
};

/**
 * @brief Backward node for natural logarithm operation.
 */
class LogBackward : public FunctionNode {
public:
    LogBackward(const Tensor& input);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "LogBackward"; }

private:
    Tensor input_;
};

/**
 * @brief Backward node for square root operation.
 */
class SqrtBackward : public FunctionNode {
public:
    SqrtBackward(const Tensor& input);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "SqrtBackward"; }

private:
    Tensor input_;
};

/**
 * @brief Backward node for absolute value operation.
 */
class AbsBackward : public FunctionNode {
public:
    AbsBackward(const Tensor& input);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "AbsBackward"; }

private:
    Tensor input_;
};

/**
 * @brief Backward node for ReLU activation function.
 */
class ReLUBackward : public FunctionNode {
public:
    ReLUBackward(const Tensor& input);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "ReLUBackward"; }

private:
    Tensor input_;
};

/**
 * @brief Backward node for sigmoid activation function.
 */
class SigmoidBackward : public FunctionNode {
public:
    SigmoidBackward(const Tensor& output);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "SigmoidBackward"; }

private:
    Tensor output_;
};

/**
 * @brief Backward node for hyperbolic tangent activation function.
 */
class TanhBackward : public FunctionNode {
public:
    TanhBackward(const Tensor& output);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "TanhBackward"; }

private:
    Tensor output_;
};

/**
 * @brief Backward node for softmax operation.
 */
class SoftmaxBackward : public FunctionNode {
public:
    SoftmaxBackward(const Tensor& output);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "SoftmaxBackward"; }

private:
    Tensor output_;
};

/**
 * @brief Backward node for sum reduction operation.
 */
class SumBackward : public FunctionNode {
public:
    SumBackward(const std::vector<std::size_t>& input_shape, std::size_t dim);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "SumBackward"; }

private:
    std::vector<std::size_t> input_shape_;
    std::size_t dim_;
};

/**
 * @brief Backward node for mean reduction operation.
 */
class MeanBackward : public FunctionNode {
public:
    MeanBackward(const std::vector<std::size_t>& input_shape, std::size_t dim);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "MeanBackward"; }

private:
    std::vector<std::size_t> input_shape_;
    std::size_t dim_;
};

/**
 * @brief Backward node for max reduction operation.
 */
class MaxBackward : public FunctionNode {
public:
    MaxBackward(const Tensor& input, const Tensor& output, std::size_t dim);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "MaxBackward"; }

private:
    Tensor input_;
    Tensor output_;
    std::size_t dim_;
};

/**
 * @brief Backward node for min reduction operation.
 */
class MinBackward : public FunctionNode {
public:
    MinBackward(const Tensor& input, const Tensor& output, std::size_t dim);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "MinBackward"; }

private:
    Tensor input_;
    Tensor output_;
    std::size_t dim_;
};

/**
 * @brief Backward node for matrix multiplication operation.
 */
class MatMulBackward : public FunctionNode {
public:
    MatMulBackward(const Tensor& a, const Tensor& b);
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "MatMulBackward"; }

private:
    Tensor a_;
    Tensor b_;
};

} // namespace autograd