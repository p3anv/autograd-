#pragma once

#include "autograd/tensor.hpp"
#include "autograd/optimizer.hpp"
#include "autograd/losses.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace autograd {

/**
 * @brief Multi-Layer Perceptron for MNIST classification.
 *
 * Architecture: 784 -> 256 (ReLU) -> 128 (ReLU) -> 10 (logits)
 * Uses Kaiming/He initialization for weights.
 */
class MLP {
public:
    /**
     * @brief Construct MLP with given hidden layer sizes.
     * @param input_size Input dimension (784 for MNIST)
     * @param hidden_sizes Vector of hidden layer sizes (e.g., {256, 128})
     * @param output_size Output dimension (10 for MNIST)
     * @param seed Random seed for reproducibility
     */
    MLP(size_t input_size = 784,
        const std::vector<size_t>& hidden_sizes = {256, 128},
        size_t output_size = 10,
        unsigned int seed = 42);

    /**
     * @brief Forward pass - returns logits.
     * @param x Input tensor of shape [batch_size, 784]
     * @return Logits tensor of shape [batch_size, 10]
     */
    Tensor forward(const Tensor& x);

    /**
     * @brief Training step.
     * @param x Input batch [batch_size, 784]
     * @param y One-hot targets [batch_size, 10]
     * @param optimizer Optimizer to use for parameter updates
     * @return Loss value as float
     */
    float train_step(const Tensor& x, const Tensor& y, Optimizer& optimizer);

    /**
     * @brief Evaluate model accuracy on a dataset.
     * @param x Input tensor [N, 784]
     * @param y One-hot targets [N, 10]
     * @return Accuracy as percentage (0-100)
     */
    float evaluate(const Tensor& x, const Tensor& y);

    /**
     * @brief Get all parameters for optimizer.
     * @return Vector of pointers to parameter tensors
     */
    std::vector<Tensor*> parameters();

    /**
     * @brief Set model to training mode (enables dropout, etc. if added later).
     */
    void train() { is_training_ = true; }

    /**
     * @brief Set model to evaluation mode.
     */
    void eval() { is_training_ = false; }

    /**
     * @brief Get number of parameters.
     */
    size_t num_parameters() const;

private:
    struct Layer {
        Tensor weight;
        Tensor bias;
    };

    std::vector<Layer> layers_;
    bool is_training_ = true;

    /**
     * @brief Initialize weights using Kaiming/He uniform initialization.
     * @param fan_in Number of input units
     * @param fan_out Number of output units
     * @param shape Weight shape [fan_out, fan_in]
     * @param seed Random seed
     */
    Tensor kaiming_uniform_(size_t fan_in, size_t fan_out, unsigned int& seed);

    /**
     * @brief Linear layer forward: x @ W.T + b
     */
    Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);
};

} // namespace autograd