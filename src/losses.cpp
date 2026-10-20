#include "autograd/losses.hpp"
#include "autograd/operations.hpp"
#include "autograd/internal.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace autograd {

// Cross-entropy loss: combines log_softmax + nll_loss for numerical stability
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    // Validate shapes
    if (logits.dim() != 2 || targets.dim() != 2) {
        throw std::invalid_argument("cross_entropy_loss expects 2D tensors [N, C]");
    }
    if (logits.shape() != targets.shape()) {
        throw std::invalid_argument("logits and targets must have same shape");
    }

    // log_softmax for numerical stability
    Tensor log_probs = log_softmax(logits);

    // NLL loss: -sum(targets * log_probs) / N
    // Since targets are one-hot, this simplifies to -sum(log_probs at target indices) / N
    Tensor nll = nll_loss(log_probs, targets);

    return nll;
}

// NLL loss: -sum(targets * log_probs) / N
Tensor nll_loss(const Tensor& log_probs, const Tensor& targets) {
    if (log_probs.dim() != 2 || targets.dim() != 2) {
        throw std::invalid_argument("nll_loss expects 2D tensors [N, C]");
    }
    if (log_probs.shape() != targets.shape()) {
        throw std::invalid_argument("log_probs and targets must have same shape");
    }

    // Compute -sum(targets * log_probs) / N
    // Since targets are one-hot, element-wise multiply picks out the log_prob at the target index
    Tensor elementwise = targets * log_probs;
    Tensor sum_per_sample = elementwise.sum(1, false);  // [N] - sum over classes
    Tensor neg_sum = -sum_per_sample;
    Tensor mean_loss = neg_sum.mean();  // scalar

    return mean_loss;
}

// Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)] averaged
Tensor binary_cross_entropy(const Tensor& probs, const Tensor& targets) {
    // Validate shapes
    auto probs_shape = probs.shape();
    auto targets_shape = targets.shape();

    // Handle both [N, 1] and [N] shapes
    size_t N_probs = (probs_shape.size() == 2) ? probs_shape[0] : probs_shape[0];
    size_t N_targets = (targets_shape.size() == 2) ? targets_shape[0] : targets_shape[0];

    if (N_probs != N_targets) {
        throw std::invalid_argument("probs and targets must have same batch size");
    }

    // Clamp probabilities to avoid log(0)
    Tensor eps = Tensor::scalar(1e-7f);
    Tensor one = Tensor::scalar(1.0f);
    Tensor probs_clamped = maximum(minimum(probs, one - eps), eps);

    // BCE = -[y*log(p) + (1-y)*log(1-p)]
    Tensor log_p = log(probs_clamped);
    Tensor log_1mp = log(one - probs_clamped);

    Tensor term1 = targets * log_p;
    Tensor term2 = (one - targets) * log_1mp;
    Tensor bce_per_sample = -(term1 + term2);

    return bce_per_sample.mean();
}

// Mean squared error loss
Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("predictions and targets must have same shape");
    }

    Tensor diff = predictions - targets;
    Tensor squared = diff * diff;
    return squared.mean();
}

} // namespace autograd