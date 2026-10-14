#pragma once

#include "autograd/dispatch.hpp"
#include <cstddef>

namespace autograd {
namespace simd {

/**
 * @brief AVX2 + FMA kernel declarations.
 *
 * These symbols are defined in src/simd/kernels_avx2.cpp, which is compiled
 * with `-mavx2 -mfma` at the translation-unit level (never globally). Callers
 * must only reach them through runtime dispatch (see dispatch.hpp), which
 * guarantees the host CPU supports AVX2+FMA before the tag is selected.
 *
 * The machine-code safety relies entirely on that runtime gate: the process
 * would fault with SIGILL if an Avx2Tag kernel were executed on a CPU lacking
 * AVX2.
 */

// Elementwise binary
void add_kernel(Avx2Tag, float* dst, const float* a, const float* b, size_t n);
void sub_kernel(Avx2Tag, float* dst, const float* a, const float* b, size_t n);
void mul_kernel(Avx2Tag, float* dst, const float* a, const float* b, size_t n);
void div_kernel(Avx2Tag, float* dst, const float* a, const float* b, size_t n);

// Broadcast variants (scalar + vector)
void add_scalar_kernel(Avx2Tag, float* dst, const float* a, float b, size_t n);
void sub_scalar_kernel(Avx2Tag, float* dst, const float* a, float b, size_t n);
void rsub_scalar_kernel(Avx2Tag, float* dst, float a, const float* b, size_t n);
void mul_scalar_kernel(Avx2Tag, float* dst, const float* a, float b, size_t n);
void div_scalar_kernel(Avx2Tag, float* dst, const float* a, float b, size_t n);
void rdiv_scalar_kernel(Avx2Tag, float* dst, float a, const float* b, size_t n);

// Elementwise unary
void neg_kernel(Avx2Tag, float* dst, const float* a, size_t n);
void abs_kernel(Avx2Tag, float* dst, const float* a, size_t n);
void exp_kernel(Avx2Tag, float* dst, const float* a, size_t n);
void log_kernel(Avx2Tag, float* dst, const float* a, size_t n);
void sqrt_kernel(Avx2Tag, float* dst, const float* a, size_t n);
void relu_kernel(Avx2Tag, float* dst, const float* a, size_t n);
void sigmoid_kernel(Avx2Tag, float* dst, const float* a, size_t n);
void tanh_kernel(Avx2Tag, float* dst, const float* a, size_t n);

// Reductions
float sum_kernel(Avx2Tag, const float* data, size_t n);
float mean_kernel(Avx2Tag, const float* data, size_t n);
void sum_dim_kernel(Avx2Tag, float* dst, const float* src,
                    const size_t* shape, const std::ptrdiff_t* strides,
                    size_t dim, size_t ndim, size_t outer_size, size_t inner_size);

// Activations
void softmax_kernel(Avx2Tag, float* dst, const float* src, size_t rows, size_t cols);
void log_softmax_kernel(Avx2Tag, float* dst, const float* src, size_t rows, size_t cols);

// Matmul
void matmul_kernel(Avx2Tag, float* C, const float* A, const float* B,
                   size_t M, size_t N, size_t K);

} // namespace simd
} // namespace autograd