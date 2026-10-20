#pragma once

#include "autograd/tensor.hpp"

namespace autograd {

/**
 * @brief Cross-entropy loss for multi-class classification.
 *
 * Numerically stable implementation using log_softmax internally.
 *
 * @param logits Logits tensor of shape [N, C] where N=batch size, C=num classes
 * @param targets One-hot encoded labels tensor of shape [N, C]
 * @return Scalar loss tensor (0-dimensional)
 * @throws std::invalid_argument if shapes don't match
 */
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);

/**
 * @brief Negative log-likelihood loss for pre-computed log-probabilities.
 *
 * @param log_probs Log-probabilities tensor of shape [N, C] (output of log_softmax)
 * @param targets One-hot encoded labels tensor of shape [N, C]
 * @return Scalar loss tensor (0-dimensional)
 */
Tensor nll_loss(const Tensor& log_probs, const Tensor& targets);

/**
 * @brief Binary cross-entropy loss.
 *
 * @param probs Probabilities tensor of shape [N, 1] or [N] (output of sigmoid)
 * @param targets Binary targets tensor of shape [N, 1] or [N] (0 or 1)
 * @return Scalar loss tensor
 */
Tensor binary_cross_entropy(const Tensor& probs, const Tensor& targets);

/**
 * @brief Mean squared error loss.
 *
 * @param predictions Predicted values
 * @param targets Target values
 * @return Scalar loss tensor
 */
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

} // namespace autograd