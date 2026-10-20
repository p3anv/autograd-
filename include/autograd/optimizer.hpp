#pragma once

#include "autograd/tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace autograd {

/**
 * @brief Base optimizer class.
 *
 * Optimizers perform parameter updates using gradients.
 * All operations are graph-free (no autograd tracking).
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /**
     * @brief Perform one optimization step.
     * @param params Pointers to parameter tensors (must have gradients)
     */
    virtual void step(std::vector<Tensor*> params) = 0;

    /**
     * @brief Zero out gradients for all parameters.
     * @param params Pointers to parameter tensors
     */
    virtual void zero_grad(std::vector<Tensor*> params) = 0;

    /**
     * @brief Get optimizer name.
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Stochastic Gradient Descent with optional momentum.
 */
class SGD : public Optimizer {
public:
    /**
     * @param learning_rate Learning rate (default: 0.01)
     * @param momentum Momentum factor (default: 0.0)
     * @param weight_decay Weight decay (L2 regularization, default: 0.0)
     */
    SGD(float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);

    void step(std::vector<Tensor*> params) override;
    void zero_grad(std::vector<Tensor*> params) override;
    std::string name() const override { return "SGD"; }

    float learning_rate() const { return lr_; }
    void set_learning_rate(float lr) { lr_ = lr; }

private:
    float lr_;
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocity_;  // Momentum buffers (graph-free)
};

/**
 * @brief Adam optimizer.
 */
class Adam : public Optimizer {
public:
    /**
     * @param learning_rate Learning rate (default: 0.001)
     * @param beta1 Exponential decay rate for 1st moment (default: 0.9)
     * @param beta2 Exponential decay rate for 2nd moment (default: 0.999)
     * @param eps Term added to denominator for numerical stability (default: 1e-8)
     * @param weight_decay Weight decay (default: 0.0)
     */
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f);

    void step(std::vector<Tensor*> params) override;
    void zero_grad(std::vector<Tensor*> params) override;
    std::string name() const override { return "Adam"; }

    float learning_rate() const { return lr_; }
    void set_learning_rate(float lr) { lr_ = lr; }

private:
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    size_t step_ = 0;
    std::vector<Tensor> m_;  // 1st moment estimates
    std::vector<Tensor> v_;  // 2nd moment estimates
};

/**
 * @brief Factory function to create optimizer by name.
 */
std::unique_ptr<Optimizer> create_optimizer(const std::string& name, float lr = 0.01f);

} // namespace autograd