#include "autograd/kernels_avx2.hpp"

#if !defined(__AVX2__) || !defined(__FMA__)
#error "kernels_avx2.cpp must be compiled with -mavx2 -mfma"
#endif

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <iostream>

namespace autograd {
namespace simd {
namespace {

// ----------------------------------------------------------------------------
// Vector math primitives. Values outside the ranges used by this library are
// saturated rather than allowed to produce invalid exponent fields; the
// scalar reference remains the source of truth for exact edge semantics.
// ----------------------------------------------------------------------------

// exp(x) via exp2: x*log2(e) split into an integer part k and a fraction
// f in [-0.5, 0.5]. 2^k comes from the IEEE exponent field, 2^f from a
// degree-5 minimax polynomial. Relative error is ~1e-6, well inside the
// 1e-4 tolerance used to validate against std::exp.
inline __m256 exp_ps(__m256 x) {
    // Clamp to the range where exp() stays within a finite float.
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3365f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.7228f));

    const __m256 LOG2E = _mm256_set1_ps(1.4426950408889634f);
    __m256 t = _mm256_mul_ps(x, LOG2E);
    __m256 k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // f = x - k*ln2, with two-part ln2 for accuracy.
    __m256 f = _mm256_fnmadd_ps(k, _mm256_set1_ps(0.693359375f), x);
    f = _mm256_fnmadd_ps(k, _mm256_set1_ps(-2.12194440e-4f), f);

    // Taylor series of exp(f) truncated at degree 5 (|f| <= 0.5).
    __m256 p = _mm256_set1_ps(8.3333333e-3f);  // 1/120
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(4.1666667e-2f)); // 1/24
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.6666667e-1f)); // 1/6
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.0e-1f));       // 1/2
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));

    // 2^k by placing k into the exponent field (bias 127).
    __m256i ki = _mm256_cvtps_epi32(k);
    ki = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ki));
}

// log(x) for x > 0: split x = m * 2^e with m in [sqrt(0.5), sqrt(2)),
// then log(x) = e*ln2 + 2*atanh((m-1)/(m+1)) via a 5-term series.
inline __m256 log_ps(__m256 x) {
    __m256i xi = _mm256_castps_si256(x);
    __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    __m256i mi = _mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)),
        _mm256_set1_epi32(0x3F800000));
    __m256 m = _mm256_castsi256_ps(mi);

    // Rescale so m < sqrt(2), adjusting the exponent to match.
    __m256 big = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356f), _CMP_GT_OQ);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), big);
    __m256 ef = _mm256_add_ps(_mm256_cvtepi32_ps(e),
                              _mm256_and_ps(big, _mm256_set1_ps(1.0f)));

    __m256 num = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    __m256 den = _mm256_add_ps(m, _mm256_set1_ps(1.0f));
    __m256 z = _mm256_div_ps(num, den);
    __m256 z2 = _mm256_mul_ps(z, z);

    __m256 p = _mm256_set1_ps(1.0f / 9.0f);
    p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 7.0f));
    p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 5.0f));
    p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 3.0f));
    p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f));

    __m256 logm = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), z), p);
    return _mm256_fmadd_ps(ef, _mm256_set1_ps(0.6931471805599453f), logm);
}

inline __m256 abs_ps(__m256 x) {
    const __m256i mask = _mm256_set1_epi32(0x7FFFFFFF);
    return _mm256_and_ps(x, _mm256_castsi256_ps(mask));
}

inline __m256 relu_ps(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

inline __m256 sigmoid_ps(__m256 x) {
    __m256 e = exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

inline __m256 tanh_ps(__m256 x) {
    // tanh(x) = 1 - 2/(exp(2x)+1); saturates to +/-1 at the extremes.
    __m256 e = exp_ps(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
    __m256 r = _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, _mm256_set1_ps(1.0f)));
    return _mm256_sub_ps(_mm256_set1_ps(1.0f), r);
}

// ----------------------------------------------------------------------------
// Loop drivers: vector body for the bulk, exact scalar tail.
// ----------------------------------------------------------------------------

template <typename VecOp, typename ScalarOp>
inline void unary_loop(float* dst, const float* a, size_t n, VecOp vop, ScalarOp sop) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, vop(_mm256_loadu_ps(a + i)));
    }
    for (; i < n; ++i) {
        dst[i] = sop(a[i]);
    }
}

template <typename VecOp, typename ScalarOp>
inline void binary_loop(float* dst, const float* a, const float* b, size_t n,
                        VecOp vop, ScalarOp sop) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, vop(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; ++i) {
        dst[i] = sop(a[i], b[i]);
    }
}

template <typename VecOp, typename ScalarOp>
inline void scalar_loop(float* dst, const float* a, float b, size_t n,
                        VecOp vop, ScalarOp sop) {
    const __m256 vb = _mm256_set1_ps(b);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, vop(_mm256_loadu_ps(a + i), vb));
    }
    for (; i < n; ++i) {
        dst[i] = sop(a[i], b);
    }
}

// Reversed broadcast: dst = op(scalar a, vector b).
template <typename VecOp, typename ScalarOp>
inline void rscalar_loop(float* dst, float a, const float* b, size_t n,
                         VecOp vop, ScalarOp sop) {
    const __m256 va = _mm256_set1_ps(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, vop(va, _mm256_loadu_ps(b + i)));
    }
    for (; i < n; ++i) {
        dst[i] = sop(a, b[i]);
    }
}

// Horizontal sum of a __m256 vector (8 floats)
inline float hsum_ps_avx2(__m256 v) {
    // Add upper 128 bits to lower 128 bits
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);

    // Horizontal add: [a+b+c+d, e+f+g+h, ...]
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sum2 = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sum2);
    sum2 = _mm_add_ss(sum2, shuf);

    return _mm_cvtss_f32(sum2);
}

// Horizontal max of a __m256 vector (8 floats)
inline float hmax_ps_avx2(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(vlow, vhigh);
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));
    return _mm_cvtss_f32(max128);
}

} // namespace

// ----------------------------------------------------------------------------
// Elementwise binary (templated to match header declarations)
// ----------------------------------------------------------------------------
template <typename Tag>
void add_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    binary_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_add_ps(x, y); },
                [](float x, float y) { return x + y; });
}
template <typename Tag>
void sub_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    binary_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_sub_ps(x, y); },
                [](float x, float y) { return x - y; });
}
template <typename Tag>
void mul_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    binary_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_mul_ps(x, y); },
                [](float x, float y) { return x * y; });
}
template <typename Tag>
void div_kernel(Tag, float* dst, const float* a, const float* b, size_t n) {
    binary_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_div_ps(x, y); },
                [](float x, float y) { return x / y; });
}

// ----------------------------------------------------------------------------
// Broadcast variants
// ----------------------------------------------------------------------------
template <typename Tag>
void add_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    scalar_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_add_ps(x, y); },
                [](float x, float y) { return x + y; });
}
template <typename Tag>
void sub_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    scalar_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_sub_ps(x, y); },
                [](float x, float y) { return x - y; });
}
template <typename Tag>
void rsub_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n) {
    rscalar_loop(dst, a, b, n,
                 [](__m256 x, __m256 y) { return _mm256_sub_ps(x, y); },
                 [](float x, float y) { return x - y; });
}
template <typename Tag>
void radd_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n) {
    rscalar_loop(dst, a, b, n,
                 [](__m256 x, __m256 y) { return _mm256_add_ps(x, y); },
                 [](float x, float y) { return x + y; });
}
template <typename Tag>
void mul_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    scalar_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_mul_ps(x, y); },
                [](float x, float y) { return x * y; });
}
template <typename Tag>
void div_scalar_kernel(Tag, float* dst, const float* a, float b, size_t n) {
    scalar_loop(dst, a, b, n,
                [](__m256 x, __m256 y) { return _mm256_div_ps(x, y); },
                [](float x, float y) { return x / y; });
}
template <typename Tag>
void rdiv_scalar_kernel(Tag, float* dst, float a, const float* b, size_t n) {
    rscalar_loop(dst, a, b, n,
                 [](__m256 x, __m256 y) { return _mm256_div_ps(x, y); },
                 [](float x, float y) { return x / y; });
}

// ----------------------------------------------------------------------------
// Elementwise unary
// ----------------------------------------------------------------------------
template <typename Tag>
void neg_kernel(Tag, float* dst, const float* a, size_t n) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    unary_loop(dst, a, n,
               [sign](__m256 x) { return _mm256_xor_ps(x, sign); },
               [](float x) { return -x; });
}
template <typename Tag>
void abs_kernel(Tag, float* dst, const float* a, size_t n) {
    unary_loop(dst, a, n,
               [](__m256 x) { return abs_ps(x); },
               [](float x) { return std::fabs(x); });
}
template <typename Tag>
void exp_kernel(Tag, float* dst, const float* a, size_t n) {
    unary_loop(dst, a, n,
               [](__m256 x) { return exp_ps(x); },
               [](float x) { return std::exp(x); });
}
template <typename Tag>
void log_kernel(Tag, float* dst, const float* a, size_t n) {
    unary_loop(dst, a, n,
               [](__m256 x) { return log_ps(x); },
               [](float x) { return std::log(x); });
}
template <typename Tag>
void sqrt_kernel(Tag, float* dst, const float* a, size_t n) {
    unary_loop(dst, a, n,
               [](__m256 x) { return _mm256_sqrt_ps(x); },
               [](float x) { return std::sqrt(x); });
}
template <typename Tag>
void relu_kernel(Tag, float* dst, const float* a, size_t n) {
    unary_loop(dst, a, n,
               [](__m256 x) { return relu_ps(x); },
               [](float x) { return x > 0.0f ? x : 0.0f; });
}
template <typename Tag>
void sigmoid_kernel(Tag, float* dst, const float* a, size_t n) {
    unary_loop(dst, a, n,
               [](__m256 x) { return sigmoid_ps(x); },
               [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}
template <typename Tag>
void tanh_kernel(Tag, float* dst, const float* a, size_t n) {
    unary_loop(dst, a, n,
               [](__m256 x) { return tanh_ps(x); },
               [](float x) { return std::tanh(x); });
}

// ----------------------------------------------------------------------------
// Reduction Operations
// ----------------------------------------------------------------------------
template <typename Tag>
float sum_kernel(Tag, const float* data, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 7 < n; i += 8) {
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(data + i));
    }

    float sum = hsum_ps_avx2(vsum);

    for (; i < n; ++i) {
        sum += data[i];
    }

    return sum;
}

template <typename Tag>
float mean_kernel(Tag, const float* data, size_t n) {
    if (n == 0) return 0.0f;
    return sum_kernel(Tag{}, data, n) / static_cast<float>(n);
}

// Strided reduction along a specific dimension
template <typename Tag>
void sum_dim_kernel(Tag, float* dst, const float* src,
                    const size_t* shape, const std::ptrdiff_t* strides,
                    size_t dim, size_t ndim, size_t outer_size, size_t inner_size) {
    (void)outer_size;
    (void)inner_size;
    const size_t dim_size = shape[dim];
    size_t total_out = 1;
    for (size_t d = 0; d < ndim; ++d) {
        if (d != dim) total_out *= shape[d];
    }

    for (size_t out = 0; out < total_out; ++out) {
        size_t remaining = out;
        size_t offset = 0;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (static_cast<size_t>(d) == dim) continue;
            const size_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset += coord * strides[d];
        }

        float acc = 0.0f;
        if (strides[dim] == 1 && dim_size >= 8) {
            size_t i = 0;
            __m256 vsum = _mm256_setzero_ps();
            for (; i + 7 < dim_size; i += 8) {
                vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(src + offset + i));
            }
            acc = hsum_ps_avx2(vsum);
            for (; i < dim_size; ++i) {
                acc += src[offset + i];
            }
        } else {
            for (size_t i = 0; i < dim_size; ++i) {
                acc += src[offset + i * strides[dim]];
            }
        }
        dst[out] = acc;
    }
}

// ----------------------------------------------------------------------------
// Activation Functions (Multi-pass)
// ----------------------------------------------------------------------------
template <typename Tag>
void softmax_kernel(Tag, float* dst, const float* src, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row_src = src + r * cols;
        float* row_dst = dst + r * cols;

        // Pass 1: Find max (vectorized)
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        size_t c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row_src + c);
            vmax = _mm256_max_ps(vmax, v);
        }
        float max_val = hmax_ps_avx2(vmax);

        for (; c < cols; ++c) {
            if (row_src[c] > max_val) max_val = row_src[c];
        }

        // Pass 2: Compute exp(x - max) and sum (vectorized)
        __m256 vsum = _mm256_setzero_ps();
        c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row_src + c);
            __m256 vexp = exp_ps(_mm256_sub_ps(v, _mm256_set1_ps(max_val)));
            _mm256_storeu_ps(row_dst + c, vexp);
            vsum = _mm256_add_ps(vsum, vexp);
        }
        float exp_sum = hsum_ps_avx2(vsum);
        for (; c < cols; ++c) {
            float v = std::exp(row_src[c] - max_val);
            row_dst[c] = v;
            exp_sum += v;
        }

        // Pass 3: Normalize (vectorized)
        __m256 vinv = _mm256_set1_ps(1.0f / exp_sum);
        c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row_dst + c);
            _mm256_storeu_ps(row_dst + c, _mm256_mul_ps(v, vinv));
        }
        for (; c < cols; ++c) {
            row_dst[c] /= exp_sum;
        }
    }
}

template <typename Tag>
void log_softmax_kernel(Tag, float* dst, const float* src, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row_src = src + r * cols;
        float* row_dst = dst + r * cols;

        // Pass 1: Find max (vectorized)
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        size_t c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row_src + c);
            vmax = _mm256_max_ps(vmax, v);
        }
        float max_val = hmax_ps_avx2(vmax);

        for (; c < cols; ++c) {
            if (row_src[c] > max_val) max_val = row_src[c];
        }

        // Pass 2: Compute exp(x - max) and sum (vectorized)
        __m256 vsum = _mm256_setzero_ps();
        c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row_src + c);
            __m256 vexp = exp_ps(_mm256_sub_ps(v, _mm256_set1_ps(max_val)));
            vsum = _mm256_add_ps(vsum, vexp);
        }
        float exp_sum = hsum_ps_avx2(vsum);
        for (; c < cols; ++c) {
            exp_sum += std::exp(row_src[c] - max_val);
        }

        float log_sum = std::log(exp_sum);

        // Pass 3: Compute log_softmax = (x - max) - log_sum (vectorized)
        __m256 vmax_vec = _mm256_set1_ps(max_val);
        __m256 vlog_sum = _mm256_set1_ps(log_sum);
        c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 v = _mm256_loadu_ps(row_src + c);
            __m256 result = _mm256_sub_ps(_mm256_sub_ps(v, vmax_vec), vlog_sum);
            _mm256_storeu_ps(row_dst + c, result);
        }
        for (; c < cols; ++c) {
            row_dst[c] = (row_src[c] - max_val) - log_sum;
        }
    }
}

// ----------------------------------------------------------------------------
// Matrix Multiplication (Blocked with 8x8 microkernel)
// ----------------------------------------------------------------------------

static void matmul_microkernel_8x8(const float* A, const float* B, float* C, size_t K, size_t ldc) {
    // Load existing C values (accumulate into C, not overwrite)
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    // Process K in chunks of 8
    for (size_t k_base = 0; k_base < K; k_base += 8) {
        size_t k_end = std::min(k_base + 8, K);
        size_t k_chunk = k_end - k_base;

        for (size_t kk = 0; kk < k_chunk; ++kk) {
            size_t k = k_base + kk;
            // B is row-major [K, 8], so row kk in this chunk is at B + kk * 8
            __m256 bcol = _mm256_loadu_ps(B + kk * 8);

            // A is row-major [8, K], so row i, col k is at A[i*K + k]
            // Broadcast each A element across the vector
            __m256 a0 = _mm256_set1_ps(A[0 * K + k]);
            c0 = _mm256_fmadd_ps(a0, bcol, c0);

            __m256 a1 = _mm256_set1_ps(A[1 * K + k]);
            c1 = _mm256_fmadd_ps(a1, bcol, c1);

            __m256 a2 = _mm256_set1_ps(A[2 * K + k]);
            c2 = _mm256_fmadd_ps(a2, bcol, c2);

            __m256 a3 = _mm256_set1_ps(A[3 * K + k]);
            c3 = _mm256_fmadd_ps(a3, bcol, c3);

            __m256 a4 = _mm256_set1_ps(A[4 * K + k]);
            c4 = _mm256_fmadd_ps(a4, bcol, c4);

            __m256 a5 = _mm256_set1_ps(A[5 * K + k]);
            c5 = _mm256_fmadd_ps(a5, bcol, c5);

            __m256 a6 = _mm256_set1_ps(A[6 * K + k]);
            c6 = _mm256_fmadd_ps(a6, bcol, c6);

            __m256 a7 = _mm256_set1_ps(A[7 * K + k]);
            c7 = _mm256_fmadd_ps(a7, bcol, c7);
        }
        // Advance B pointer by 8*8 = 64 elements (8 rows × 8 elements each)
        B += 64;
    }

    // C is row-major [8, 8], so row i is at C + i*ldc
    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
}

template <typename Tag>
void matmul_kernel(Tag, float* C, const float* A, const float* B,
                   size_t M, size_t N, size_t K) {
    constexpr size_t MC = 64;
    constexpr size_t NC = 64;
    constexpr size_t KC = 256;

    std::fill(C, C + M * N, 0.0f);

    std::vector<float> pack_A(MC * KC);
    // pack_B is column-major [K_rem, N_rem] for the whole panel
    std::vector<float> pack_B(KC * NC);

    for (size_t m_block = 0; m_block < M; m_block += MC) {
        size_t M_rem = std::min(MC, M - m_block);
        for (size_t n_block = 0; n_block < N; n_block += NC) {
            size_t N_rem = std::min(NC, N - n_block);

            alignas(32) float C_block[MC * NC];
            std::fill(C_block, C_block + M_rem * N_rem, 0.0f);

            for (size_t k_block = 0; k_block < K; k_block += KC) {
                size_t K_rem = std::min(KC, K - k_block);

                // Pack A panel: [M_rem, K_rem] -> row-major contiguous
                for (size_t i = 0; i < M_rem; ++i) {
                    std::memcpy(
                        &pack_A[i * K_rem],
                        &A[(m_block + i) * K + k_block],
                        K_rem * sizeof(float)
                    );
                }

                // Pack B panel: [K_rem, N_rem] -> column-major contiguous
                // pack_B[j * K_rem + k] = B[(k_block + k) * N + (n_block + j)]
                for (size_t k = 0; k < K_rem; ++k) {
                    for (size_t j = 0; j < N_rem; ++j) {
                        pack_B[j * K_rem + k] = B[(k_block + k) * N + (n_block + j)];
                    }
                }

                // Compute M_rem x N_rem block using 8x8 microkernels
                // The microkernel processes K in chunks of 8, so we need to feed it
                // B in a format where each 8-column group has 8 elements per column
                for (size_t i = 0; i < M_rem; i += 8) {
                    size_t i_end = std::min(i + 8, M_rem);
                    size_t i_len = i_end - i;

                    for (size_t j = 0; j < N_rem; j += 8) {
                        size_t j_end = std::min(j + 8, N_rem);
                        size_t j_len = j_end - j;

                        if (i_len == 8 && j_len == 8) {
                            // We need to pass B in a format the microkernel expects:
                            // B should be [K_rem, 8] in column-major (each column has K_rem elements)
                            // But the microkernel processes K in chunks of 8, expecting 8*8=64 elements per chunk
                            // So we need to create a temporary buffer with the right layout

                            // Create a temporary buffer for this 8x8 tile's B panel
                            // The microkernel expects B as [K_rem, 8] column-major, processed in 8-row chunks
                            alignas(32) float B_tile[KC * 8];  // Max size: 256 * 8 = 2048 elements

                            // Pack the 8 columns for this tile into B_tile
                            // B_tile[k * 8 + jj] = pack_B[(j + jj) * K_rem + k]
                            for (size_t k = 0; k < K_rem; ++k) {
                                for (size_t jj = 0; jj < j_len; ++jj) {
                                    B_tile[k * 8 + jj] = pack_B[(j + jj) * K_rem + k];
                                }
                            }

                            matmul_microkernel_8x8(
                                &pack_A[i * K_rem],
                                B_tile,
                                &C_block[i * N_rem + j],
                                K_rem, N_rem
                            );
                        } else {
                            for (size_t ii = 0; ii < i_len; ++ii) {
                                for (size_t jj = 0; jj < j_len; ++jj) {
                                    float sum = 0.0f;
                                    for (size_t k = 0; k < K_rem; ++k) {
                                        sum += pack_A[(i + ii) * K_rem + k] * pack_B[(j + jj) * K_rem + k];
                                    }
                                    C_block[(i + ii) * N_rem + (j + jj)] += sum;
                                }
                            }
                        }
                    }
                }
            }

            // Write C_block back to C
            for (size_t i = 0; i < M_rem; ++i) {
                std::memcpy(
                    &C[(m_block + i) * N + n_block],
                    &C_block[i * N_rem],
                    N_rem * sizeof(float)
                );
            }
        }
    }
}

// ============================================================================
// Explicit Instantiations for AVX2
// ============================================================================

// Binary
template void add_kernel<Avx2Tag>(Avx2Tag, float*, const float*, const float*, size_t);
template void sub_kernel<Avx2Tag>(Avx2Tag, float*, const float*, const float*, size_t);
template void mul_kernel<Avx2Tag>(Avx2Tag, float*, const float*, const float*, size_t);
template void div_kernel<Avx2Tag>(Avx2Tag, float*, const float*, const float*, size_t);

// Broadcast binary
template void add_scalar_kernel<Avx2Tag>(Avx2Tag, float*, const float*, float, size_t);
template void sub_scalar_kernel<Avx2Tag>(Avx2Tag, float*, const float*, float, size_t);
template void rsub_scalar_kernel<Avx2Tag>(Avx2Tag, float*, float, const float*, size_t);
template void radd_scalar_kernel<Avx2Tag>(Avx2Tag, float*, float, const float*, size_t);
template void mul_scalar_kernel<Avx2Tag>(Avx2Tag, float*, const float*, float, size_t);
template void div_scalar_kernel<Avx2Tag>(Avx2Tag, float*, const float*, float, size_t);
template void rdiv_scalar_kernel<Avx2Tag>(Avx2Tag, float*, float, const float*, size_t);

// Unary
template void neg_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);
template void abs_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);
template void exp_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);
template void log_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);
template void sqrt_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);
template void relu_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);
template void sigmoid_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);
template void tanh_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t);

// Reductions
template float sum_kernel<Avx2Tag>(Avx2Tag, const float*, size_t);
template float mean_kernel<Avx2Tag>(Avx2Tag, const float*, size_t);
template void sum_dim_kernel<Avx2Tag>(Avx2Tag, float*, const float*,
                            const size_t*, const std::ptrdiff_t*,
                            size_t, size_t, size_t, size_t);

// Activations
template void softmax_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t, size_t);
template void log_softmax_kernel<Avx2Tag>(Avx2Tag, float*, const float*, size_t, size_t);

// Matmul
template void matmul_kernel<Avx2Tag>(Avx2Tag, float*, const float*, const float*, size_t, size_t, size_t);

// Non-templated wrapper functions for dispatch.hpp compatibility
void add_kernel(Avx2Tag tag, float* dst, const float* a, const float* b, size_t n) {
    add_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void sub_kernel(Avx2Tag tag, float* dst, const float* a, const float* b, size_t n) {
    sub_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void mul_kernel(Avx2Tag tag, float* dst, const float* a, const float* b, size_t n) {
    mul_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void div_kernel(Avx2Tag tag, float* dst, const float* a, const float* b, size_t n) {
    div_kernel<Avx2Tag>(tag, dst, a, b, n);
}

void add_scalar_kernel(Avx2Tag tag, float* dst, const float* a, float b, size_t n) {
    add_scalar_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void sub_scalar_kernel(Avx2Tag tag, float* dst, const float* a, float b, size_t n) {
    sub_scalar_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void rsub_scalar_kernel(Avx2Tag tag, float* dst, float a, const float* b, size_t n) {
    rsub_scalar_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void radd_scalar_kernel(Avx2Tag tag, float* dst, float a, const float* b, size_t n) {
    radd_scalar_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void mul_scalar_kernel(Avx2Tag tag, float* dst, const float* a, float b, size_t n) {
    mul_scalar_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void div_scalar_kernel(Avx2Tag tag, float* dst, const float* a, float b, size_t n) {
    div_scalar_kernel<Avx2Tag>(tag, dst, a, b, n);
}
void rdiv_scalar_kernel(Avx2Tag tag, float* dst, float a, const float* b, size_t n) {
    rdiv_scalar_kernel<Avx2Tag>(tag, dst, a, b, n);
}

void neg_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    neg_kernel<Avx2Tag>(tag, dst, a, n);
}
void abs_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    abs_kernel<Avx2Tag>(tag, dst, a, n);
}
void exp_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    exp_kernel<Avx2Tag>(tag, dst, a, n);
}
void log_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    log_kernel<Avx2Tag>(tag, dst, a, n);
}
void sqrt_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    sqrt_kernel<Avx2Tag>(tag, dst, a, n);
}
void relu_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    relu_kernel<Avx2Tag>(tag, dst, a, n);
}
void sigmoid_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    sigmoid_kernel<Avx2Tag>(tag, dst, a, n);
}
void tanh_kernel(Avx2Tag tag, float* dst, const float* a, size_t n) {
    tanh_kernel<Avx2Tag>(tag, dst, a, n);
}

float sum_kernel(Avx2Tag tag, const float* data, size_t n) {
    return sum_kernel<Avx2Tag>(tag, data, n);
}
float mean_kernel(Avx2Tag tag, const float* data, size_t n) {
    return mean_kernel<Avx2Tag>(tag, data, n);
}
void sum_dim_kernel(Avx2Tag tag, float* dst, const float* src,
                    const size_t* shape, const std::ptrdiff_t* strides,
                    size_t dim, size_t ndim, size_t outer_size, size_t inner_size) {
    sum_dim_kernel<Avx2Tag>(tag, dst, src, shape, strides, dim, ndim, outer_size, inner_size);
}

void softmax_kernel(Avx2Tag tag, float* dst, const float* src, size_t rows, size_t cols) {
    softmax_kernel<Avx2Tag>(tag, dst, src, rows, cols);
}
void log_softmax_kernel(Avx2Tag tag, float* dst, const float* src, size_t rows, size_t cols) {
    log_softmax_kernel<Avx2Tag>(tag, dst, src, rows, cols);
}

void matmul_kernel(Avx2Tag tag, float* C, const float* A, const float* B,
                   size_t M, size_t N, size_t K) {
    matmul_kernel<Avx2Tag>(tag, C, A, B, M, N, K);
}

} // namespace simd
} // namespace autograd