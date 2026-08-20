#pragma once

#include <vector>
#include "autograd/tensor.hpp"

namespace autograd {

// Forward declarations of operations to be implemented in Phase 3
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor pow(const Tensor& base, const Tensor& exponent);

// Unary operations
Tensor neg(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor abs(const Tensor& a);

// Activation functions
Tensor relu(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor softmax(const Tensor& a);

// Reductions
Tensor sum(const Tensor& a, std::size_t dim);
Tensor mean(const Tensor& a, std::size_t dim);
Tensor max(const Tensor& a, std::size_t dim);
Tensor min(const Tensor& a, std::size_t dim);

// Normalization
Tensor log_softmax(const Tensor& a);

// Matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace autograd