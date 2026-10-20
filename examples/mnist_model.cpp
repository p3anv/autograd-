#include "autograd/mnist_model.hpp"
#include "autograd/operations.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <iostream>

namespace autograd {

MLP::MLP(size_t input_size, const std::vector<size_t>& hidden_sizes, size_t output_size, unsigned int seed) {
    // Build layer sizes: [input_size, hidden_sizes..., output_size]
    std::vector<size_t> layer_sizes;
    layer_sizes.push_back(input_size);
    for (size_t h : hidden_sizes) layer_sizes.push_back(h);
    layer_sizes.push_back(output_size);

    unsigned int layer_seed = seed;
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        size_t fan_in = layer_sizes[i];
        size_t fan_out = layer_sizes[i + 1];

        Layer layer;
        layer.weight = kaiming_uniform_(fan_in, fan_out, layer_seed);
        layer.bias = Tensor::zeros({fan_out});
        // Enable gradients for parameters
        layer.weight.set_requires_grad(true);
        layer.bias.set_requires_grad(true);
        layers_.push_back(std::move(layer));

        layer_seed += 12345;  // Different seed per layer
    }
}

Tensor MLP::kaiming_uniform_(size_t fan_in, size_t fan_out, unsigned int& seed) {
    // Kaiming/He initialization: uniform distribution in [-bound, bound]
    // bound = gain * sqrt(3 / fan_in)
    // For ReLU, gain = sqrt(2)
    float gain = std::sqrt(2.0f);
    float bound = gain * std::sqrt(3.0f / fan_in);

    // Use uniform distribution [-bound, bound]
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-bound, bound);

    Tensor weight = Tensor::empty({fan_out, fan_in}, false);
    float* data = weight.data();
    for (size_t i = 0; i < fan_out * fan_in; ++i) {
        data[i] = dist(gen);
    }

    return weight;
}

Tensor MLP::linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    // x: [batch_size, in_features]
    // weight: [out_features, in_features]
    // bias: [out_features]
    // output: [batch_size, out_features]

    Tensor out = matmul(x, transpose(weight));
    out = out + bias;  // Broadcasting add
    return out;
}

Tensor MLP::forward(const Tensor& x) {
    Tensor h = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        h = linear(h, layers_[i].weight, layers_[i].bias);
        if (i < layers_.size() - 1) {
            // Apply ReLU for hidden layers
            h = relu(h);
        }
        // Last layer returns logits (no activation)
    }
    return h;
}

float MLP::train_step(const Tensor& x, const Tensor& y, Optimizer& optimizer) {
    // Zero gradients
    auto params = parameters();
    optimizer.zero_grad(params);

    // Forward pass
    Tensor logits = forward(x);

    // Compute loss (cross-entropy with log_softmax internally)
    Tensor loss = cross_entropy_loss(logits, y);

    // Backward pass
    loss.backward();

    // Debug: check gradients for first layer weight (only first step of first epoch)
    static int global_step = 0;
    if (global_step == 0) {
        std::cout << "  DEBUG STEP 0:" << std::endl;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i] && params[i]->impl()->grad_impl_) {
                const float* grad_data = params[i]->impl()->grad_impl_->data();
                float grad_norm = 0.0f;
                int inf_count = 0, nan_count = 0;
                float max_val = -FLT_MAX, min_val = FLT_MAX;
                for (size_t j = 0; j < params[i]->numel(); ++j) {
                    float g = grad_data[j];
                    if (!std::isfinite(g)) {
                        if (std::isinf(g)) inf_count++;
                        if (std::isnan(g)) nan_count++;
                    } else {
                        grad_norm += g * g;
                        if (g > max_val) max_val = g;
                        if (g < min_val) min_val = g;
                    }
                }
                grad_norm = std::sqrt(grad_norm);
                std::cout << "    Param " << i << " (shape: [";
                for (size_t s = 0; s < params[i]->shape().size(); ++s) {
                    std::cout << params[i]->shape()[s] << (s+1 < params[i]->shape().size() ? ", " : "");
                }
                std::cout << "]) grad norm: " << grad_norm << " (inf=" << inf_count << " nan=" << nan_count
                          << " min=" << min_val << " max=" << max_val << ")" << std::endl;

                // Print first 5 gradient values
                std::cout << "    Param " << i << " grad first 5: ";
                for (size_t j = 0; j < std::min<size_t>(5, params[i]->numel()); ++j) {
                    std::cout << grad_data[j] << " ";
                }
                std::cout << std::endl;

                // Print first 5 weight values before step
                const float* param_data = params[i]->data();
                std::cout << "    Param " << i << " weight first 5: ";
                for (size_t j = 0; j < std::min<size_t>(5, params[i]->numel()); ++j) {
                    std::cout << param_data[j] << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "    Param " << i << " has NO grad_impl_" << std::endl;
            }
        }

        // Check logits
        std::cout << "    Logits min/max: ";
        const float* logits_data = logits.data();
        float min_val = logits_data[0];
        float max_val = logits_data[0];
        for (size_t i = 1; i < logits.numel(); ++i) {
            if (logits_data[i] < min_val) min_val = logits_data[i];
            if (logits_data[i] > max_val) max_val = logits_data[i];
        }
        std::cout << min_val << " / " << max_val << std::endl;
        std::cout << "    Loss: " << loss.item() << std::endl;
        std::cout.flush();
    }

    // Optimizer step
    optimizer.step(params);

    // Debug: check weight after step (only first step)
    if (global_step == 0) {
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]) {
                const float* param_data = params[i]->data();
                std::cout << "    Param " << i << " weight after step (first 5): ";
                for (size_t j = 0; j < std::min<size_t>(5, params[i]->numel()); ++j) {
                    std::cout << param_data[j] << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout.flush();
    }
    global_step++;

    float loss_val = loss.item();
    // Return loss as float
    return loss_val;
}

float MLP::evaluate(const Tensor& x, const Tensor& y) {
    // Forward pass (no gradient tracking needed for eval)
    Tensor logits = forward(x);

    // Get predictions (argmax over classes)
    Tensor predictions = logits.argmax(1);  // [batch_size] - indices

    // Get target indices (argmax over one-hot)
    Tensor targets = y.argmax(1);  // [batch_size] - indices

    // Compare predictions with targets
    Tensor correct = equal(predictions, targets);

    // Sum correct and compute accuracy
    Tensor correct_sum = correct.sum(0, false);
    float accuracy = correct_sum.item() / x.shape()[0] * 100.0f;

    return accuracy;
}

std::vector<Tensor*> MLP::parameters() {
    std::vector<Tensor*> params;
    params.reserve(layers_.size() * 2);
    for (auto& layer : layers_) {
        params.push_back(&layer.weight);
        params.push_back(&layer.bias);
    }
    return params;
}

size_t MLP::num_parameters() const {
    size_t total = 0;
    for (const auto& layer : layers_) {
        total += layer.weight.numel();
        total += layer.bias.numel();
    }
    return total;
}

} // namespace autograd