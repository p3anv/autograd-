#include "autograd/optimizer.hpp"
#include "autograd/tensor.hpp"
#include "autograd/internal.hpp"
#include <stdexcept>
#include <cmath>

namespace autograd {

// ============================================================================
// SGD Implementation
// ============================================================================

SGD::SGD(float learning_rate, float momentum, float weight_decay)
    : lr_(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::zero_grad(std::vector<Tensor*> params) {
    for (Tensor* param : params) {
        if (param && param->impl()->grad_impl_) {
            // Zero the gradient buffer (graph-free operation)
            float* grad_data = param->impl()->grad_impl_->data();
            std::fill(grad_data, grad_data + param->numel(), 0.0f);
        }
    }
}

void SGD::step(std::vector<Tensor*> params) {
    // Initialize velocity buffers if needed
    if (velocity_.empty()) {
        velocity_.reserve(params.size());
        for (Tensor* param : params) {
            if (param) {
                velocity_.emplace_back(param->shape(), false);
                std::fill(velocity_.back().data(), velocity_.back().data() + param->numel(), 0.0f);
            } else {
                velocity_.emplace_back();  // Empty tensor for null param
            }
        }
    }

    // Gradient clipping: compute global gradient norm
    float grad_norm = 0.0f;
    bool has_inf_nan = false;
    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* param = params[i];
        if (!param || !param->impl()->grad_impl_) continue;
        Tensor grad_tensor(param->impl()->grad_impl_);
        const float* grad_data = grad_tensor.data();
        size_t n = param->numel();
        for (size_t j = 0; j < n; ++j) {
            float g = grad_data[j];
            if (!std::isfinite(g)) {
                has_inf_nan = true;
            }
            grad_norm += g * g;
        }
    }
    grad_norm = std::sqrt(grad_norm);

    // Check for inf/nan in grad_norm
    if (!std::isfinite(grad_norm)) {
        has_inf_nan = true;
    }

    // Clip gradient norm to 5.0 (less aggressive clipping)
    const float max_grad_norm = 5.0f;
    float clip_coef = 1.0f;
    if (grad_norm > max_grad_norm || has_inf_nan) {
        clip_coef = max_grad_norm / (grad_norm + 1e-6f);
        // If grad_norm is inf or nan, clip_coef will be 0
        if (!std::isfinite(clip_coef) || clip_coef == 0.0f) {
            clip_coef = 0.0f;
        }
    }

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* param = params[i];
        if (!param || !param->impl()->grad_impl_) continue;

        // Get gradient tensor from grad_impl_
        Tensor grad_tensor(param->impl()->grad_impl_);
        Tensor& vel = velocity_[i];
        float* param_data = param->data();
        const float* grad_data = grad_tensor.data();
        float* vel_data = vel.data();
        size_t n = param->numel();

        if (momentum_ > 0.0f) {
            // Velocity = momentum * velocity - lr * (grad + weight_decay * param)
            for (size_t j = 0; j < n; ++j) {
                float g = grad_data[j] * clip_coef;
                if (!std::isfinite(g)) {
                    g = 0.0f;
                }
                if (weight_decay_ > 0.0f) {
                    g += weight_decay_ * param_data[j];
                }
                vel_data[j] = momentum_ * vel_data[j] - lr_ * g;
                param_data[j] += vel_data[j];
            }
        } else {
            // Simple SGD: param -= lr * (grad + weight_decay * param)
            for (size_t j = 0; j < n; ++j) {
                float g = grad_data[j] * clip_coef;
                if (!std::isfinite(g)) {
                    g = 0.0f;
                }
                if (weight_decay_ > 0.0f) {
                    g += weight_decay_ * param_data[j];
                }
                param_data[j] -= lr_ * g;
            }
        }
    }
}

// ============================================================================
// Adam Implementation
// ============================================================================

Adam::Adam(float learning_rate, float beta1, float beta2, float eps, float weight_decay)
    : lr_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {}

void Adam::zero_grad(std::vector<Tensor*> params) {
    for (Tensor* param : params) {
        if (param && param->impl()->grad_impl_) {
            float* grad_data = param->impl()->grad_impl_->data();
            std::fill(grad_data, grad_data + param->numel(), 0.0f);
        }
    }
}

void Adam::step(std::vector<Tensor*> params) {
    // Initialize moment buffers if needed
    if (m_.empty()) {
        m_.reserve(params.size());
        v_.reserve(params.size());
        for (Tensor* param : params) {
            if (param) {
                m_.emplace_back(param->shape(), false);
                v_.emplace_back(param->shape(), false);
                std::fill(m_.back().data(), m_.back().data() + param->numel(), 0.0f);
                std::fill(v_.back().data(), v_.back().data() + param->numel(), 0.0f);
            } else {
                m_.emplace_back();
                v_.emplace_back();
            }
        }
    }

    step_++;
    float lr_t = lr_ * std::sqrt(1.0f - std::pow(beta2_, step_)) / (1.0f - std::pow(beta1_, step_));

    // Gradient clipping: compute global gradient norm
    float grad_norm = 0.0f;
    bool has_inf_nan = false;
    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* param = params[i];
        if (!param || !param->impl()->grad_impl_) continue;
        Tensor grad_tensor(param->impl()->grad_impl_);
        const float* grad_data = grad_tensor.data();
        size_t n = param->numel();
        for (size_t j = 0; j < n; ++j) {
            float g = grad_data[j];
            if (!std::isfinite(g)) {
                has_inf_nan = true;
            }
            grad_norm += g * g;
        }
    }
    grad_norm = std::sqrt(grad_norm);

    // Check for inf/nan in grad_norm
    if (!std::isfinite(grad_norm)) {
        has_inf_nan = true;
    }

    // Clip gradient norm to 5.0 (less aggressive clipping)
    const float max_grad_norm = 5.0f;
    float clip_coef = 1.0f;
    if (grad_norm > max_grad_norm || has_inf_nan) {
        clip_coef = max_grad_norm / (grad_norm + 1e-6f);
        // If grad_norm is inf or nan, clip_coef will be 0
        if (!std::isfinite(clip_coef) || clip_coef == 0.0f) {
            clip_coef = 0.0f;
        }
    }

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* param = params[i];
        if (!param || !param->impl()->grad_impl_) continue;

        // Get gradient tensor from grad_impl_
        Tensor grad_tensor(param->impl()->grad_impl_);
        Tensor& m = m_[i];
        Tensor& v = v_[i];
        float* param_data = param->data();
        const float* grad_data = grad_tensor.data();
        float* m_data = m.data();
        float* v_data = v.data();
        size_t n = param->numel();

        for (size_t j = 0; j < n; ++j) {
            float g = grad_data[j] * clip_coef;
            // Ensure g is finite
            if (!std::isfinite(g)) {
                g = 0.0f;
            }
            if (weight_decay_ > 0.0f) {
                g += weight_decay_ * param_data[j];
            }

            // m = beta1 * m + (1 - beta1) * g
            m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
            // v = beta2 * v + (1 - beta2) * g^2
            v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;

            // param -= lr_t * m / (sqrt(v) + eps)
            param_data[j] -= lr_t * m_data[j] / (std::sqrt(v_data[j]) + eps_);
        }
    }
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Optimizer> create_optimizer(const std::string& name, float lr) {
    if (name == "sgd" || name == "SGD") {
        return std::make_unique<SGD>(lr);
    } else if (name == "adam" || name == "Adam") {
        return std::make_unique<Adam>(lr);
    } else {
        throw std::invalid_argument("Unknown optimizer: " + name);
    }
}

} // namespace autograd