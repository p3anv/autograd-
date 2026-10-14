#include "autograd/kernels.hpp"
#include "autograd/dispatch.hpp"
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace autograd {
namespace simd {

// ============================================================================
// Elementwise Binary Operations
// ============================================================================

template <typename Tag>
void add_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
}

template <typename Tag>
void sub_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] - b[i];
    }
}

template <typename Tag>
void mul_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
}

template <typename Tag>
void div_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] / b[i];
    }
}

// Broadcast variants (scalar + vector)
template <typename Tag>
void add_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] + b;
    }
}

template <typename Tag>
void sub_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] - b;
    }
}

template <typename Tag>
void rsub_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a - b[i];
    }
}

template <typename Tag>
void radd_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a + b[i];
    }
}

template <typename Tag>
void mul_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] * b;
    }
}

template <typename Tag>
void div_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] / b;
    }
}

template <typename Tag>
void rdiv_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a / b[i];
    }
}

// ============================================================================
// Elementwise Unary Operations
// ============================================================================

template <typename Tag>
void neg_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = -a[i];
    }
}

template <typename Tag>
void abs_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::fabs(a[i]);
    }
}

template <typename Tag>
void exp_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::exp(a[i]);
    }
}

template <typename Tag>
void log_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::log(a[i]);
    }
}

template <typename Tag>
void sqrt_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::sqrt(a[i]);
    }
}

template <typename Tag>
void relu_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

template <typename Tag>
void sigmoid_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
}

template <typename Tag>
void tanh_kernel(Tag, float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::tanh(a[i]);
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

template <typename Tag>
float sum_kernel(Tag, const float* data, size_t n) {
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        result += data[i];
    }
    return result;
}

template <typename Tag>
float mean_kernel(Tag, const float* data, size_t n) {
    return sum_kernel(Tag{}, data, n) / static_cast<float>(n);
}

// Sum along a specific dimension (strided)
template <typename Tag>
void sum_dim_kernel(Tag, float* dst, const float* src,
                    const size_t* shape, const std::ptrdiff_t* strides,
                    size_t dim, size_t ndim, size_t outer_size, size_t inner_size) {
    // shape: full tensor shape, strides: full tensor strides in elements
    // dim: dimension to reduce. outer_size/inner_size describe the dims
    // before/after `dim`; the output size is the product of all other dims.
    (void)outer_size;
    (void)inner_size;
    const size_t dim_size = shape[dim];
    size_t total_out = 1;
    for (size_t d = 0; d < ndim; ++d) {
        if (d != dim) total_out *= shape[d];
    }

    for (size_t out = 0; out < total_out; ++out) {
        // Decode the linear output index into coordinates over the
        // output shape (input shape with `dim` removed), then map onto
        // the input element offset.
        size_t remaining = out;
        size_t offset = 0;
        // Output coords are ordered: [0, dim) then (dim, ndim).
        // Decode from the fastest-varying (last) output dim backwards.
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (static_cast<size_t>(d) == dim) continue;
            const size_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset += coord * strides[d];
        }

        float acc = 0.0f;
        for (size_t i = 0; i < dim_size; ++i) {
            acc += src[offset + i * strides[dim]];
        }
        dst[out] = acc;
    }
}

// ============================================================================
// Activation Functions (Multi-pass)
// ============================================================================

template <typename Tag>
void softmax_kernel(Tag, float* dst, const float* src, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row_src = src + r * cols;
        float* row_dst = dst + r * cols;

        // Pass 1: Find max
        float max_val = row_src[0];
        for (size_t c = 1; c < cols; ++c) {
            if (row_src[c] > max_val) max_val = row_src[c];
        }

        // Pass 2: Compute exp and sum
        float exp_sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = std::exp(row_src[c] - max_val);
            row_dst[c] = v;
            exp_sum += v;
        }

        // Pass 3: Normalize
        for (size_t c = 0; c < cols; ++c) {
            row_dst[c] /= exp_sum;
        }
    }
}

template <typename Tag>
void log_softmax_kernel(Tag, float* dst, const float* src, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row_src = src + r * cols;
        float* row_dst = dst + r * cols;

        // Find max
        float max_val = row_src[0];
        for (size_t c = 1; c < cols; ++c) {
            if (row_src[c] > max_val) max_val = row_src[c];
        }

        // Compute exp sum
        float exp_sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            exp_sum += std::exp(row_src[c] - max_val);
        }

        float log_sum = std::log(exp_sum);

        // Compute log_softmax = (x - max) - log_sum
        for (size_t c = 0; c < cols; ++c) {
            row_dst[c] = (row_src[c] - max_val) - log_sum;
        }
    }
}

// ============================================================================
// Matrix Multiplication (Naive scalar - for reference)
// ============================================================================

template <typename Tag>
void matmul_kernel(Tag, float* C, const float* A, const float* B,
                   size_t M, size_t N, size_t K) {
    // C[M,N] = A[M,K] * B[K,N]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Explicit Instantiations
// ============================================================================

// Binary
template void add_kernel(ScalarTag, float*, const float*, const float*, size_t);
template void sub_kernel(ScalarTag, float*, const float*, const float*, size_t);
template void mul_kernel(ScalarTag, float*, const float*, const float*, size_t);
template void div_kernel(ScalarTag, float*, const float*, const float*, size_t);

// Broadcast binary
template void add_scalar_kernel(ScalarTag, float*, const float*, float, size_t);
template void sub_scalar_kernel(ScalarTag, float*, const float*, float, size_t);
template void rsub_scalar_kernel(ScalarTag, float*, float, const float*, size_t);
template void radd_scalar_kernel(ScalarTag, float*, float, const float*, size_t);
template void mul_scalar_kernel(ScalarTag, float*, const float*, float, size_t);
template void div_scalar_kernel(ScalarTag, float*, const float*, float, size_t);
template void rdiv_scalar_kernel(ScalarTag, float*, float, const float*, size_t);

// Unary
template void neg_kernel(ScalarTag, float*, const float*, size_t);
template void abs_kernel(ScalarTag, float*, const float*, size_t);
template void exp_kernel(ScalarTag, float*, const float*, size_t);
template void log_kernel(ScalarTag, float*, const float*, size_t);
template void sqrt_kernel(ScalarTag, float*, const float*, size_t);
template void relu_kernel(ScalarTag, float*, const float*, size_t);
template void sigmoid_kernel(ScalarTag, float*, const float*, size_t);
template void tanh_kernel(ScalarTag, float*, const float*, size_t);

// Reductions
template float sum_kernel(ScalarTag, const float*, size_t);
template float mean_kernel(ScalarTag, const float*, size_t);
template void sum_dim_kernel(ScalarTag, float*, const float*,
                            const size_t*, const std::ptrdiff_t*,
                            size_t, size_t, size_t, size_t);

// Activations
template void softmax_kernel(ScalarTag, float*, const float*, size_t, size_t);
template void log_softmax_kernel(ScalarTag, float*, const float*, size_t, size_t);

// Matmul
template void matmul_kernel(ScalarTag, float*, const float*, const float*, size_t, size_t, size_t);

// SSE4.2 instantiations (fallback to scalar implementations)
template void add_kernel(SseTag, float*, const float*, const float*, size_t);
template void sub_kernel(SseTag, float*, const float*, const float*, size_t);
template void mul_kernel(SseTag, float*, const float*, const float*, size_t);
template void div_kernel(SseTag, float*, const float*, const float*, size_t);

template void add_scalar_kernel(SseTag, float*, const float*, float, size_t);
template void sub_scalar_kernel(SseTag, float*, const float*, float, size_t);
template void rsub_scalar_kernel(SseTag, float*, float, const float*, size_t);
template void radd_scalar_kernel(SseTag, float*, float, const float*, size_t);
template void mul_scalar_kernel(SseTag, float*, const float*, float, size_t);
template void div_scalar_kernel(SseTag, float*, const float*, float, size_t);
template void rdiv_scalar_kernel(SseTag, float*, float, const float*, size_t);

template void neg_kernel(SseTag, float*, const float*, size_t);
template void abs_kernel(SseTag, float*, const float*, size_t);
template void exp_kernel(SseTag, float*, const float*, size_t);
template void log_kernel(SseTag, float*, const float*, size_t);
template void sqrt_kernel(SseTag, float*, const float*, size_t);
template void relu_kernel(SseTag, float*, const float*, size_t);
template void sigmoid_kernel(SseTag, float*, const float*, size_t);
template void tanh_kernel(SseTag, float*, const float*, size_t);

template float sum_kernel(SseTag, const float*, size_t);
template float mean_kernel(SseTag, const float*, size_t);
template void sum_dim_kernel(SseTag, float*, const float*,
                            const size_t*, const std::ptrdiff_t*,
                            size_t, size_t, size_t, size_t);

template void softmax_kernel(SseTag, float*, const float*, size_t, size_t);
template void log_softmax_kernel(SseTag, float*, const float*, size_t, size_t);

template void matmul_kernel(SseTag, float*, const float*, const float*, size_t, size_t, size_t);

} // namespace simd
} // namespace autograd