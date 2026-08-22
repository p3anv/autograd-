#include "autograd/function_nodes.hpp"
#include "autograd/tensor_impl.hpp"
#include "autograd/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace autograd {

FunctionNode::FunctionNode(std::vector<std::size_t> output_shape)
    : output_shape_(std::move(output_shape))
{
}

const std::vector<std::size_t>& FunctionNode::output_shape() const noexcept {
    return output_shape_;
}

// AddBackward
AddBackward::AddBackward(const std::vector<std::size_t>& shape_a, const std::vector<std::size_t>& shape_b)
    : FunctionNode(shape_a), shape_a_(shape_a), shape_b_(shape_b) {
    // Note: output shape should be the broadcasted shape of shape_a and shape_b.
    // However, we are not checking here; we assume the forward pass already computed the correct output shape.
    // We'll use the first shape as the output shape for the base constructor, but we should actually compute the broadcasted shape.
    // Let's compute the broadcasted shape properly.
    std::vector<std::size_t> broadcasted_shape;
    std::size_t max_dim = std::max(shape_a.size(), shape_b.size());
    std::vector<std::size_t> a_padded = shape_a;
    std::vector<std::size_t> b_padded = shape_b;
    while (a_padded.size() < max_dim) {
        a_padded.insert(a_padded.begin(), 1);
    }
    while (b_padded.size() < max_dim) {
        b_padded.insert(b_padded.begin(), 1);
    }
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_padded[i] == b_padded[i]) {
            broadcasted_shape.push_back(a_padded[i]);
        } else if (a_padded[i] == 1) {
            broadcasted_shape.push_back(b_padded[i]);
        } else if (b_padded[i] == 1) {
            broadcasted_shape.push_back(a_padded[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting in AddBackward");
        }
    }
    output_shape_ = broadcasted_shape;
}

std::vector<Tensor> AddBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // Gradient of add: grad_a = grad_output, grad_b = grad_output
    // But we need to sum over broadcasted dimensions.
    Tensor grad_output = grad_outputs[0];
    Tensor grad_a, grad_b;

    // Compute grad_a by summing grad_output over dimensions where a was broadcasted
    if (!shape_a_.empty()) {
        grad_a = grad_output;
        // Align shapes from the right (trailing dimensions)
        std::size_t a_ndim = shape_a_.size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - a_ndim;
        for (std::size_t i = 0; i < a_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (shape_a_[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                // Sum over this dimension
                grad_a = grad_a.sum(dim_idx);
            }
        }
        // If a had fewer dimensions than output, sum over the leading dimensions
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_a = grad_a.sum(i);
                }
            }
        }
    } else {
        // a is scalar
        grad_a = Tensor::scalar(grad_output.sum().item());
    }

    // Similarly for grad_b
    if (!shape_b_.empty()) {
        grad_b = grad_output;
        std::size_t b_ndim = shape_b_.size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - b_ndim;
        for (std::size_t i = 0; i < b_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (shape_b_[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_b = grad_b.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_b = grad_b.sum(i);
                }
            }
        }
    } else {
        // b is scalar
        grad_b = Tensor::scalar(grad_output.sum().item());
    }

    return {grad_a, grad_b};
}

// SubBackward
SubBackward::SubBackward(const std::vector<std::size_t>& shape_a, const std::vector<std::size_t>& shape_b)
    : FunctionNode(shape_a), shape_a_(shape_a), shape_b_(shape_b) {
    // Compute broadcasted shape
    std::vector<std::size_t> broadcasted_shape;
    std::size_t max_dim = std::max(shape_a.size(), shape_b.size());
    std::vector<std::size_t> a_padded = shape_a;
    std::vector<std::size_t> b_padded = shape_b;
    while (a_padded.size() < max_dim) {
        a_padded.insert(a_padded.begin(), 1);
    }
    while (b_padded.size() < max_dim) {
        b_padded.insert(b_padded.begin(), 1);
    }
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_padded[i] == b_padded[i]) {
            broadcasted_shape.push_back(a_padded[i]);
        } else if (a_padded[i] == 1) {
            broadcasted_shape.push_back(b_padded[i]);
        } else if (b_padded[i] == 1) {
            broadcasted_shape.push_back(a_padded[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting in SubBackward");
        }
    }
    output_shape_ = broadcasted_shape;
}

std::vector<Tensor> SubBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    Tensor grad_a, grad_b;

    // grad_a = grad_output (same as add)
    if (!shape_a_.empty()) {
        grad_a = grad_output;
        std::size_t a_ndim = shape_a_.size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - a_ndim;
        for (std::size_t i = 0; i < a_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (shape_a_[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_a = grad_a.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_a = grad_a.sum(i);
                }
            }
        }
    } else {
        grad_a = Tensor::scalar(grad_output.sum().item());
    }

    // grad_b = -grad_output
    if (!shape_b_.empty()) {
        grad_b = grad_output;
        // Negate
        grad_b = grad_b.neg();
        // Now sum over broadcasted dimensions
        std::size_t b_ndim = shape_b_.size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - b_ndim;
        for (std::size_t i = 0; i < b_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (shape_b_[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_b = grad_b.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_b = grad_b.sum(i);
                }
            }
        }
    } else {
        grad_b = Tensor::scalar(-grad_output.sum().item());
    }

    return {grad_a, grad_b};
}

// MulBackward
MulBackward::MulBackward(const Tensor& a, const Tensor& b)
    : FunctionNode(a.shape()), a_(a), b_(b) {
    // We assume the forward pass already computed the correct output shape (broadcasted)
    // and set it in the FunctionNode base constructor via the tensor's shape.
    // However, we need to compute the broadcasted shape for the output.
    // Let's compute it here and override the output_shape_.
    std::vector<std::size_t> broadcasted_shape;
    std::size_t max_dim = std::max(a.shape().size(), b.shape().size());
    std::vector<std::size_t> a_padded = a.shape();
    std::vector<std::size_t> b_padded = b.shape();
    while (a_padded.size() < max_dim) {
        a_padded.insert(a_padded.begin(), 1);
    }
    while (b_padded.size() < max_dim) {
        b_padded.insert(b_padded.begin(), 1);
    }
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_padded[i] == b_padded[i]) {
            broadcasted_shape.push_back(a_padded[i]);
        } else if (a_padded[i] == 1) {
            broadcasted_shape.push_back(b_padded[i]);
        } else if (b_padded[i] == 1) {
            broadcasted_shape.push_back(a_padded[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting in MulBackward");
        }
    }
    output_shape_ = broadcasted_shape;
}

std::vector<Tensor> MulBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // dy/da = b, dy/db = a
    Tensor grad_a = grad_output * b_;
    Tensor grad_b = grad_output * a_;

    // Now we need to sum over broadcasted dimensions for each gradient.
    // For grad_a, we need to sum over dimensions where a was broadcasted.
    if (!a_.shape().empty()) {
        grad_a = grad_output * b_;
        std::size_t a_ndim = a_.shape().size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - a_ndim;
        for (std::size_t i = 0; i < a_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (a_.shape()[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_a = grad_a.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_a = grad_a.sum(i);
                }
            }
        }
    } else {
        grad_a = Tensor::scalar((grad_output * b_).sum().item());
    }

    // For grad_b
    if (!b_.shape().empty()) {
        grad_b = grad_output * a_;
        std::size_t b_ndim = b_.shape().size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - b_ndim;
        for (std::size_t i = 0; i < b_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (b_.shape()[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_b = grad_b.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_b = grad_b.sum(i);
                }
            }
        }
    } else {
        grad_b = Tensor::scalar((grad_output * a_).sum().item());
    }

    return {grad_a, grad_b};
}

// DivBackward
DivBackward::DivBackward(const Tensor& a, const Tensor& b)
    : FunctionNode(a.shape()), a_(a), b_(b) {
    // Compute broadcasted shape for output
    std::vector<std::size_t> broadcasted_shape;
    std::size_t max_dim = std::max(a.shape().size(), b.shape().size());
    std::vector<std::size_t> a_padded = a.shape();
    std::vector<std::size_t> b_padded = b.shape();
    while (a_padded.size() < max_dim) {
        a_padded.insert(a_padded.begin(), 1);
    }
    while (b_padded.size() < max_dim) {
        b_padded.insert(b_padded.begin(), 1);
    }
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (a_padded[i] == b_padded[i]) {
            broadcasted_shape.push_back(a_padded[i]);
        } else if (a_padded[i] == 1) {
            broadcasted_shape.push_back(b_padded[i]);
        } else if (b_padded[i] == 1) {
            broadcasted_shape.push_back(a_padded[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting in DivBackward");
        }
    }
    output_shape_ = broadcasted_shape;
}

std::vector<Tensor> DivBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // dy/da = 1/b
    // dy/db = -a/(b^2)
    Tensor grad_a = grad_output / b_;
    Tensor grad_b = -(grad_output * a_) / (b_ * b_);

    // Sum over broadcasted dimensions for grad_a
    if (!a_.shape().empty()) {
        grad_a = grad_output / b_;
        std::size_t a_ndim = a_.shape().size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - a_ndim;
        for (std::size_t i = 0; i < a_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (a_.shape()[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_a = grad_a.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_a = grad_a.sum(i);
                }
            }
        }
    } else {
        grad_a = Tensor::scalar((grad_output / b_).sum().item());
    }

    // Sum over broadcasted dimensions for grad_b
    if (!b_.shape().empty()) {
        grad_b = -(grad_output * a_) / (b_ * b_);
        std::size_t b_ndim = b_.shape().size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - b_ndim;
        for (std::size_t i = 0; i < b_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (b_.shape()[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_b = grad_b.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_b = grad_b.sum(i);
                }
            }
        }
    } else {
        grad_b = Tensor::scalar((-(grad_output * a_) / (b_ * b_)).sum().item());
    }

    return {grad_a, grad_b};
}

// PowBackward
PowBackward::PowBackward(const Tensor& base, const Tensor& exponent)
    : FunctionNode(base.shape()), base_(base), exponent_(exponent) {
    // Compute broadcasted shape for output
    std::vector<std::size_t> broadcasted_shape;
    std::size_t max_dim = std::max(base.shape().size(), exponent.shape().size());
    std::vector<std::size_t> base_padded = base.shape();
    std::vector<std::size_t> exp_padded = exponent.shape();
    while (base_padded.size() < max_dim) {
        base_padded.insert(base_padded.begin(), 1);
    }
    while (exp_padded.size() < max_dim) {
        exp_padded.insert(exp_padded.begin(), 1);
    }
    for (std::size_t i = 0; i < max_dim; ++i) {
        if (base_padded[i] == exp_padded[i]) {
            broadcasted_shape.push_back(base_padded[i]);
        } else if (base_padded[i] == 1) {
            broadcasted_shape.push_back(exp_padded[i]);
        } else if (exp_padded[i] == 1) {
            broadcasted_shape.push_back(base_padded[i]);
        } else {
            throw std::invalid_argument("Incompatible shapes for broadcasting in PowBackward");
        }
    }
    output_shape_ = broadcasted_shape;
}

std::vector<Tensor> PowBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // y = base^exponent
    // dy/dbase = exponent * base^(exponent - 1)
    // dy/d exponent = base^exponent * ln(base)
    Tensor base_pow_exp = base_.pow(exponent_); // This is y
    Tensor grad_base = grad_output * (exponent_ * base_.pow(exponent_ - Tensor::scalar(1.0f)));
    Tensor grad_exp = grad_output * (base_pow_exp * base_.log());

    // Sum over broadcasted dimensions for grad_base
    if (!base_.shape().empty()) {
        grad_base = grad_output * (exponent_ * base_.pow(exponent_ - Tensor::scalar(1.0f)));
        std::size_t base_ndim = base_.shape().size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - base_ndim;
        for (std::size_t i = 0; i < base_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (base_.shape()[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_base = grad_base.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_base = grad_base.sum(i);
                }
            }
        }
    } else {
        grad_base = Tensor::scalar((grad_output * (exponent_ * base_.pow(exponent_ - Tensor::scalar(1.0f)))).sum().item());
    }

    // Sum over broadcasted dimensions for grad_exp
    if (!exponent_.shape().empty()) {
        grad_exp = grad_output * (base_pow_exp * base_.log());
        std::size_t exp_ndim = exponent_.shape().size();
        std::size_t output_ndim = grad_output.shape().size();
        std::size_t offset = output_ndim - exp_ndim;
        for (std::size_t i = 0; i < exp_ndim; ++i) {
            std::size_t dim_idx = offset + i;
            if (exponent_.shape()[i] == 1 && grad_output.shape()[dim_idx] > 1) {
                grad_exp = grad_exp.sum(dim_idx);
            }
        }
        if (offset > 0) {
            for (std::size_t i = 0; i < offset; ++i) {
                if (grad_output.shape()[i] > 1) {
                    grad_exp = grad_exp.sum(i);
                }
            }
        }
    } else {
        grad_exp = Tensor::scalar((grad_output * (base_pow_exp * base_.log())).sum().item());
    }

    return {grad_base, grad_exp};
}

// NegBackward
NegBackward::NegBackward() : FunctionNode({}) {}

std::vector<Tensor> NegBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = -x => dy/dx = -1
    Tensor grad_output = grad_outputs[0];
    Tensor grad_input = -grad_output;
    // No broadcasting to worry about because negation doesn't change shape
    return {grad_input};
}

// ExpBackward
ExpBackward::ExpBackward(const Tensor& output)
    : FunctionNode(output.shape()), output_(output) {}

std::vector<Tensor> ExpBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = exp(x) => dy/dx = exp(x) = y
    Tensor grad_output = grad_outputs[0];
    Tensor grad_input = grad_output * output_;
    // No broadcasting
    return {grad_input};
}

// LogBackward
LogBackward::LogBackward(const Tensor& input)
    : FunctionNode(input.shape()), input_(input) {}

std::vector<Tensor> LogBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = ln(x) => dy/dx = 1/x
    Tensor grad_output = grad_outputs[0];
    Tensor grad_input = grad_output / input_;
    // No broadcasting
    return {grad_input};
}

// SqrtBackward
SqrtBackward::SqrtBackward(const Tensor& input)
    : FunctionNode(input.shape()), input_(input) {}

std::vector<Tensor> SqrtBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = sqrt(x) => dy/dx = 1/(2*sqrt(x))
    Tensor grad_output = grad_outputs[0];
    Tensor two = Tensor::scalar(2.0f);
    Tensor sqrt_x = input_.sqrt();
    Tensor grad_input = grad_output / (two * sqrt_x);
    // No broadcasting
    return {grad_input};
}

// AbsBackward
AbsBackward::AbsBackward(const Tensor& input)
    : FunctionNode(input.shape()), input_(input) {}

std::vector<Tensor> AbsBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = |x| => dy/dx = sign(x) where sign(x) is 1 if x>0, -1 if x<0, 0 if x==0
    Tensor grad_output = grad_outputs[0];
    Tensor zero = Tensor::scalar(0.0f);
    Tensor one = Tensor::scalar(1.0f);
    Tensor neg_one = Tensor::scalar(-1.0f);
    Tensor pos_part = (input_ > zero).to_float(); // 1 where input>0, else 0
    Tensor neg_part = (input_ < zero).to_float(); // 1 where input<0, else 0
    Tensor grad_input = grad_output * (pos_part - neg_part);
    // No broadcasting
    return {grad_input};
}

// ReLUBackward
ReLUBackward::ReLUBackward(const Tensor& input)
    : FunctionNode(input.shape()), input_(input) {}

std::vector<Tensor> ReLUBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = max(0, x) => dy/dx = 1 if x>0, else 0
    Tensor grad_output = grad_outputs[0];
    Tensor zero = Tensor::scalar(0.0f);
    Tensor grad_input = grad_output * (input_ > zero).to_float();
    // No broadcasting
    return {grad_input};
}

// SigmoidBackward
SigmoidBackward::SigmoidBackward(const Tensor& output)
    : FunctionNode(output.shape()), output_(output) {}

std::vector<Tensor> SigmoidBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = sigmoid(x) => dy/dx = y * (1 - y)
    Tensor grad_output = grad_outputs[0];
    Tensor one = Tensor::scalar(1.0f);
    Tensor grad_input = grad_output * output_ * (one - output_);
    // No broadcasting
    return {grad_input};
}

// TanhBackward
TanhBackward::TanhBackward(const Tensor& output)
    : FunctionNode(output.shape()), output_(output) {}

std::vector<Tensor> TanhBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // y = tanh(x) => dy/dx = 1 - y^2
    Tensor grad_output = grad_outputs[0];
    Tensor one = Tensor::scalar(1.0f);
    Tensor grad_input = grad_output * (one - output_ * output_);
    // No broadcasting
    return {grad_input};
}

// SoftmaxBackward
SoftmaxBackward::SoftmaxBackward(const Tensor& output)
    : FunctionNode(output.shape()), output_(output) {}

std::vector<Tensor> SoftmaxBackward::backward(const std::vector<Tensor>& grad_outputs) {
    // For softmax, the Jacobian-vector product is:
    // grad_input = grad_output * output - output * sum(grad_output * output, dim=-1, keepdim=True)
    Tensor grad_output = grad_outputs[0];
    Tensor one = Tensor::scalar(1.0f);
    // Compute sum over the last dimension
    std::size_t last_dim = output_.shape().size() - 1;
    Tensor sum_term = (grad_output * output_).sum(last_dim, true); // keepdim=true
    Tensor grad_input = grad_output * output_ - output_ * sum_term;
    // No broadcasting
    return {grad_input};
}

// SumBackward
SumBackward::SumBackward(const std::vector<std::size_t>& input_shape, std::size_t dim)
    : FunctionNode({}), input_shape_(input_shape), dim_(dim) {
    // Compute output shape: remove the dim-th dimension
    std::vector<std::size_t> output_shape;
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (i != dim) {
            output_shape.push_back(input_shape[i]);
        }
    }
    output_shape_ = output_shape;
}

std::vector<Tensor> SumBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // We need to broadcast grad_output back to the input shape.
    // The gradient of sum is a tensor of ones with the same shape as the input,
    // but only non-zero in the reduced dimension? Actually, the derivative of sum over dim
    // is 1 for each element in the reduced dimension, so we need to broadcast grad_output
    // to the input shape by inserting the reduced dimension and setting grad_output to 1 in that dimension?
    // More precisely: if y = sum(x, dim), then dy/dx_ij...k = 1 for all i,j,...,k.
    // So, the gradient w.r.t. x is a tensor of the same shape as x, where each slice along dim
    // is filled with the value of grad_output at the corresponding position.

    // We can achieve this by expanding grad_output to the input shape.
    // Steps:
    // 1. Insert a dimension of size 1 at position dim in grad_output.
    // 2. Expand that dimension to size input_shape[dim] by repeating the values.

    // However, note that grad_output might have fewer dimensions than input.
    // We need to align the shapes.

    // Let's create a tensor of ones with the input shape, then multiply by grad_output expanded appropriately.

    // First, reshape grad_output to have the same number of dimensions as input_shape,
    // with size 1 in the reduced dimension.
    std::vector<std::size_t> expanded_shape = input_shape_;
    expanded_shape[dim_] = 1;
    Tensor expanded_grad_output = grad_output.reshape(expanded_shape);
    // Now expand the reduced dimension to the full size.
    Tensor ones = Tensor::ones(input_shape_, false);
    Tensor grad_input = ones * expanded_grad_output;
    // Note: The above multiplication will broadcast expanded_grad_output to the full shape.

    // However, we must be cautious: if the input had zero size in the reduced dimension,
    // then the sum would be zero and the gradient should be zero. But the above would
    // produce NaN? Actually, if input_shape[dim] == 0, then the sum over an empty dimension
    // is zero, and the gradient should be zero. Our method: expanded_grad_output has shape
    // with size 1 in that dimension, but the original grad_output had shape with that dimension
    // removed. If input_shape[dim] == 0, then the forward sum would have produced a tensor
    // with that dimension removed, and grad_output would have that dimension removed.
    // Then expanded_shape would have size 1 in that dimension, but we cannot reshape grad_output
    // to that shape because the total number of elements would not match? Let's see:
    // Original input shape: [D0, D1, ..., D_{dim}=0, ...]
    // After sum over dim: output shape: [D0, D1, ..., D_{dim+1}, ...] (the dim-th dimension is removed)
    // grad_output shape: same as output shape.
    // expanded_shape: [D0, D1, ..., 1, D_{dim+1}, ...] -> total elements = D0*...*1*D_{dim+1}*... = same as output shape?
    // Actually, the output shape has one less dimension, so the total elements of output shape is
    // (product of input_shape) / input_shape[dim]. Since input_shape[dim] is 0, the output shape
    // would have 0 elements? But we cannot have a tensor with zero elements in one dimension?
    // Actually, we can have a tensor with shape [2,0,3] which has 0 elements.
    // Then grad_output would have shape [2,3] (if we removed the dim=1 dimension) and 6 elements.
    // expanded_shape would be [2,1,3] -> 6 elements, so we can reshape.

    // However, if input_shape[dim] == 0, then the sum over that dimension is a vector of zeros
    // (with length equal to the product of the other dimensions). So grad_output should be a tensor
    // of zeros with shape equal to the input shape with the dim-th dimension removed.
    // Then expanded_grad_output would be a tensor of zeros with shape [D0,1,D_{dim+1},...] and
    // multiplying by ones (which is [D0, D_{dim}, D_{dim+1},...]) would yield zeros because
    // we are multiplying zeros by ones? Actually, the broadcasted multiplication would be:
    //   [D0,1,D_{dim+1},...] * [D0, D_{dim}, D_{dim+1},...] -> [D0, D_{dim}, D_{dim+1},...]
    //   where each element is 0 * 1 = 0? Wait, no: the broadcasting rules: the first tensor has
    //   size 1 in the dim-th dimension, so it gets repeated D_{dim} times. So each slice
    //   along the dim-th dimension in the result is the slice from the first tensor.
    //   Since the first tensor is all zeros, the result is all zeros.

    // So the method works even for zero-sized dimensions.

    // Return the gradient as a list with one element (since sum has one input).
    return {grad_input};
}

// MeanBackward
MeanBackward::MeanBackward(const std::vector<std::size_t>& input_shape, std::size_t dim)
    : FunctionNode({}), input_shape_(input_shape), dim_(dim) {
    // Compute output shape: remove the dim-th dimension
    std::vector<std::size_t> output_shape;
    for (std::size_t i = 0; i < input_shape.size(); ++i) {
        if (i != dim) {
            output_shape.push_back(input_shape[i]);
        }
    }
    output_shape_ = output_shape;
}

std::vector<Tensor> MeanBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // The gradient of mean is similar to sum, but we divide by the number of elements in the reduced dimension.
    // So, we first compute the gradient as for sum, then divide by input_shape[dim].
    // However, if input_shape[dim] is 0, we should avoid division by zero. But mean over zero elements
    // is undefined, so we assume input_shape[dim] > 0.

    // Compute the sum-like gradient (as in SumBackward)
    std::vector<std::size_t> expanded_shape = input_shape_;
    expanded_shape[dim_] = 1;
    Tensor expanded_grad_output = grad_output.reshape(expanded_shape);
    Tensor ones = Tensor::ones(input_shape_, false);
    Tensor grad_input = ones * expanded_grad_output;
    // Now divide by the size of the reduced dimension
    Tensor size = Tensor::scalar(static_cast<float>(input_shape_[dim_]));
    grad_input = grad_input / size;

    return {grad_input};
}

// MaxBackward
MaxBackward::MaxBackward(const Tensor& input, const Tensor& output, std::size_t dim)
    : FunctionNode(output.shape()), input_(input), output_(output), dim_(dim) {
    // Output shape is already set by the base constructor using output.shape()
}

std::vector<Tensor> MaxBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // We need to find where the input equals the max (output) and distribute the gradient accordingly.
    // For each position in the output, the gradient flows back to the position(s) in the input
    // that had the maximum value in that slice along the reduced dimension.
    // If there are multiple maxima, the gradient is split equally among them? Or we can choose
    // to send the gradient to all maxima (as in PyTorch). We'll implement the latter:
    // gradient is sent to all elements that equal the max in that slice.

    // Steps:
    // 1. Broadcast the output tensor to the input shape (inserting a dimension of size 1 at dim_)
    //    and then expanding to the full size of the reduced dimension.
    // 2. Compare the input tensor with the broadcasted output tensor to find where they are equal.
    // 3. The gradient w.r.t. input is grad_output broadcasted to the input shape, but only at
    //    positions where input == broadcasted_output, and zero elsewhere.
    //    However, if there are multiple maxima, we need to split the gradient? Actually,
    //    if we send the full gradient to all maxima, then the sum of gradients would be
    //    (number of maxima) * grad_output, which is not correct.
    //    The correct approach is to distribute the gradient equally among the maxima?
    //    But the max function is not differentiable at points where there are multiple maxima.
    //    In practice, frameworks like PyTorch compute the gradient as if the max is unique
    //    by selecting the first occurrence (or using a subgradient). However, the common
    //    practice in deep learning is to use the gradient as if the max is unique and
    //    arbitrarily choose one index (e.g., the first). This is what we will do for simplicity.
    //
    //    Alternatively, we can note that the subgradient of the max function at a point with
    //    multiple maxima is the set of vectors where the components corresponding to the
    //    maxima are non-negative and sum to 1, and the other components are zero.
    //    Choosing a particular subgradient (e.g., uniform distribution) is acceptable.
    //    We'll choose to distribute the gradient equally among all maxima.
    //
    //    However, to keep it simple and match common implementations, we will assume that
    //    the max is unique (or we break ties by taking the first). We'll implement by
    //    creating a mask that is 1 at the first occurrence of the max in each slice and 0 elsewhere.
    //
    //    Given the complexity and time, we will implement a simple version that works for
    //    the case where the max is unique (which is typical in practice with random data).
    //    We will leave a note that this needs improvement for handling multiple maxima.
    //
    //    For now, we will implement by:
    //    - Creating a tensor of zeros with the input shape.
    //    - For each slice along the reduced dimension, find the index of the first occurrence
    //      of the max value (which is the output value for that slice).
    //    - Set the gradient at that index to the grad_output value for that slice.
    //
    //    This is not fully correct but will work for many cases.
    //
    //    A better approach would be to use the fact that the max operation can be thought of
    //    as: y = max(x, dim) and then the backward pass is:
    //    grad_x = (x == y_expanded) * grad_y_expanded / sum(x == y_expanded, dim, keepdim=True)
    //    where y_expanded is y broadcasted to the shape of x.
    //    This distributes the gradient equally among all maxima.
    //    We'll implement this version.

    // Step 1: Expand grad_output to the input shape by inserting a dimension of size 1 at dim_
    //         and then expanding to the full size.
    std::vector<std::size_t> expanded_shape = input_.shape();
    expanded_shape[dim_] = 1;
    Tensor expanded_grad_output = grad_output.reshape(expanded_shape);
    // Now we need to expand the reduced dimension to the full size.
    // We can do this by repeating the values along that dimension.
    // However, we can use broadcasting in multiplication: if we multiply by a tensor of ones
    // with the same shape as input_, it will broadcast.
    Tensor ones = Tensor::ones(input_.shape(), false);
    Tensor grad_output_expanded = ones * expanded_grad_output; // This broadcasts expanded_grad_output to the shape of input_

    // Step 2: Create a mask where input_ equals the max (which is output_ expanded to input shape)
    //         We need to expand the output tensor to the input shape for comparison.
    std::vector<std::size_t> output_expanded_shape = input_.shape();
    output_expanded_shape[dim_] = 1;
    Tensor output_expanded = output_.reshape(output_expanded_shape);
    Tensor ones_for_output = Tensor::ones(input_.shape(), false);
    Tensor output_expanded_full = ones_for_output * output_expanded; // Broadcast output to input shape

    // Step 3: Create a boolean mask where input_ equals output_expanded_full
    Tensor mask = (input_ == output_expanded_full).to_float();

    // Step 4: Count the number of maxima along the reduced dimension for each slice.
    //         We sum the mask along the reduced dimension.
    Tensor mask_sum = mask.sum(dim_, true); // keepdim=true

    // Step 5: To avoid division by zero, we clamp the mask_sum to at least 1.
    //         However, if there are no elements (should not happen), we set to 1.
    Tensor mask_sum_clamped = mask_sum.clamp(Tensor::scalar(1.0f), Tensor::scalar(std::numeric_limits<float>::max()));

    // Step 6: The gradient w.r.t. input is (mask * grad_output_expanded) / mask_sum_clamped
    Tensor grad_input = (mask * grad_output_expanded) / mask_sum_clamped;

    return {grad_input};
}

// MinBackward
MinBackward::MinBackward(const Tensor& input, const Tensor& output, std::size_t dim)
    : FunctionNode(output.shape()), input_(input), output_(output), dim_(dim) {
    // Output shape is already set by the base constructor using output.shape()
}

std::vector<Tensor> MinBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // Similar to max, but for min.
    // We'll use the same approach: distribute the gradient equally among all minima.

    // Step 1: Expand grad_output to the input shape by inserting a dimension of size 1 at dim_
    //         and then expanding to the full size.
    std::vector<std::size_t> expanded_shape = input_.shape();
    expanded_shape[dim_] = 1;
    Tensor expanded_grad_output = grad_output.reshape(expanded_shape);
    Tensor ones = Tensor::ones(input_.shape(), false);
    Tensor grad_output_expanded = ones * expanded_grad_output; // This broadcasts expanded_grad_output to the shape of input_

    // Step 2: Create a mask where input_ equals the min (which is output_ expanded to input shape)
    std::vector<std::size_t> output_expanded_shape = input_.shape();
    output_expanded_shape[dim_] = 1;
    Tensor output_expanded = output_.reshape(output_expanded_shape);
    Tensor ones_for_output = Tensor::ones(input_.shape(), false);
    Tensor output_expanded_full = ones_for_output * output_expanded; // Broadcast output to input shape

    // Step 3: Create a boolean mask where input_ equals output_expanded_full
    Tensor mask = (input_ == output_expanded_full).to_float();

    // Step 4: Count the number of minima along the reduced dimension for each slice.
    Tensor mask_sum = mask.sum(dim_, true); // keepdim=true

    // Step 5: Avoid division by zero
    Tensor mask_sum_clamped = mask_sum.clamp(Tensor::scalar(1.0f), Tensor::scalar(std::numeric_limits<float>::max()));

    // Step 6: The gradient w.r.t. input is (mask * grad_output_expanded) / mask_sum_clamped
    Tensor grad_input = (mask * grad_output_expanded) / mask_sum_clamped;

    return {grad_input};
}

// MatMulBackward
MatMulBackward::MatMulBackward(const Tensor& a, const Tensor& b)
    : FunctionNode({}), a_(a), b_(b) {
    // For matmul, output shape is [a.rows, b.cols] assuming a is [M, K] and b is [K, N]
    // We assume the tensors are 2D (as per the forward pass restriction).
    std::vector<std::size_t> a_shape = a.shape();
    std::vector<std::size_t> b_shape = b.shape();
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::invalid_argument("MatMulBackward expects 2D tensors");
    }
    std::size_t M = a_shape[0];
    std::size_t K = a_shape[1];
    std::size_t N = b_shape[1];
    if (K != b_shape[0]) {
        throw std::invalid_argument("Inner dimensions mismatch for matmul");
    }
    output_shape_ = {M, N};
}

std::vector<Tensor> MatMulBackward::backward(const std::vector<Tensor>& grad_outputs) {
    Tensor grad_output = grad_outputs[0];
    // grad_a = grad_output * b^T
    // grad_b = a^T * grad_output
    // Note: we need to transpose b and a.
    Tensor b_transposed = b_.transpose();
    Tensor a_transposed = a_.transpose();
    Tensor grad_a = grad_output.matmul(b_transposed);
    Tensor grad_b = a_transposed.matmul(grad_output);

    // No broadcasting to worry about because matmul doesn't broadcast in our implementation.
    return {grad_a, grad_b};
}

} // namespace autograd