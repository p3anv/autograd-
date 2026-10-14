#pragma once

#include "autograd/dispatch.hpp"
#include <cstddef>
#include <cstdint>

namespace autograd {
namespace simd {

// ============================================================================
// Elementwise Binary Operations
// ============================================================================
template <typename Tag>
void add_kernel(Tag, float* dst, const float* a, const float* b, size_t n);
template <typename Tag>
void sub_kernel(Tag, float* dst, const float* a, const float* b, size_t n);
template <typename Tag>
void mul_kernel(Tag, float* dst, const float* a, const float* b, size_t n);
template <typename Tag>
void div_kernel(Tag, float* dst, const float* a, const float* b, size_t n);

// Broadcast variants (scalar + vector)
template <typename Tag>
void add_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n);
template <typename Tag>
void sub_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n);
template <typename Tag>
void rsub_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n);
template <typename Tag>
void mul_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n);
template <typename Tag>
void div_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n);
template <typename Tag>
void rdiv_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n);

// ============================================================================
// Elementwise Unary Operations
// ============================================================================
template <typename Tag>
void neg_kernel(Tag, float* dst, const float* a, size_t n);
template <typename Tag>
void abs_kernel(Tag, float* dst, const float* a, size_t n);
template <typename Tag>
void exp_kernel(Tag, float* dst, const float* a, size_t n);
template <typename Tag>
void log_kernel(Tag, float* dst, const float* a, size_t n);
template <typename Tag>
void sqrt_kernel(Tag, float* dst, const float* a, size_t n);
template <typename Tag>
void relu_kernel(Tag, float* dst, const float* a, size_t n);
template <typename Tag>
void sigmoid_kernel(Tag, float* dst, const float* a, size_t n);
template <typename Tag>
void tanh_kernel(Tag, float* dst, const float* a, size_t n);

// ============================================================================
// Reduction Operations
// ============================================================================
template <typename Tag>
float sum_kernel(Tag, const float* data, size_t n);
template <typename Tag>
float mean_kernel(Tag, const float* data, size_t n);

// Sum along a specific dimension (strided)
template <typename Tag>
void sum_dim_kernel(Tag, float* dst, const float* src,
                    const size_t* shape, const std::ptrdiff_t* strides,
                    size_t dim, size_t ndim, size_t outer_size, size_t inner_size);

// ============================================================================
// Activation Functions (Multi-pass)
// ============================================================================
template <typename Tag>
void softmax_kernel(Tag, float* dst, const float* src, size_t rows, size_t cols);
template <typename Tag>
void log_softmax_kernel(Tag, float* dst, const float* src, size_t rows, size_t cols);

// ============================================================================
// Matrix Multiplication
// ============================================================================
template <typename Tag>
void matmul_kernel(Tag, float* C, const float* A, const float* B,
                   size_t M, size_t N, size_t K);

} // namespace simd
} // namespace autograd