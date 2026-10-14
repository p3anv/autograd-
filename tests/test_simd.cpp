#include "autograd/cpu_capabilities.hpp"
#include "autograd/dispatch.hpp"
#include "autograd/kernels.hpp"
#if defined(HAS_AVX2_KERNELS)
#include "autograd/kernels_avx2.hpp"
#endif
#include "autograd/tensor.hpp"
#include "autograd/operations.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace autograd;

TEST(SimdTest, CpuCapabilitiesDetection) {
    const auto& caps = CpuCapabilities::get();

    std::cout << "CPU Capabilities:" << std::endl;
    std::cout << "  SSE:      " << caps.has_sse << std::endl;
    std::cout << "  SSE2:     " << caps.has_sse2 << std::endl;
    std::cout << "  SSE3:     " << caps.has_sse3 << std::endl;
    std::cout << "  SSSE3:    " << caps.has_ssse3 << std::endl;
    std::cout << "  SSE4.1:   " << caps.has_sse41 << std::endl;
    std::cout << "  SSE4.2:   " << caps.has_sse42 << std::endl;
    std::cout << "  AVX:      " << caps.has_avx << std::endl;
    std::cout << "  AVX2:     " << caps.has_avx2 << std::endl;
    std::cout << "  FMA:      " << caps.has_fma << std::endl;
    std::cout << "  BMI1:     " << caps.has_bmi1 << std::endl;
    std::cout << "  BMI2:     " << caps.has_bmi2 << std::endl;
    std::cout << "  LZCNT:    " << caps.has_lzcnt << std::endl;
    std::cout << "  POPCNT:   " << caps.has_popcnt << std::endl;

    // At minimum, x86_64 should have SSE2
    EXPECT_TRUE(caps.has_sse2);
    EXPECT_TRUE(caps.has_any_simd());

    // AVX2+FMA check
    bool has_avx2_fma = caps.has_avx2_fma();
    std::cout << "  AVX2+FMA: " << has_avx2_fma << std::endl;
}

TEST(SimdTest, DispatchMechanism) {
    int result = 0;

    // Test runtime dispatch
    simd::dispatch([&](auto tag) {
        using Tag = decltype(tag);
        if constexpr (std::is_same_v<Tag, simd::Avx2Tag>) {
            result = 3;
        } else if constexpr (std::is_same_v<Tag, simd::SseTag>) {
            result = 2;
        } else {
            result = 1;
        }
    });

    std::cout << "Dispatch result: " << result << std::endl;

    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx2_fma()) {
        EXPECT_EQ(result, 3);
    } else if (caps.has_sse42) {
        EXPECT_EQ(result, 2);
    } else {
        EXPECT_EQ(result, 1);
    }
}

TEST(SimdTest, StaticDispatch) {
    int result = 0;

    simd::static_dispatch<simd::ScalarTag>([&](auto) {
        result = 1;
    });
    EXPECT_EQ(result, 1);

    simd::static_dispatch<simd::SseTag>([&](auto) {
        result = 2;
    });
    EXPECT_EQ(result, 2);

    simd::static_dispatch<simd::Avx2Tag>([&](auto) {
        result = 3;
    });
    EXPECT_EQ(result, 3);
}

TEST(SimdTest, GetBestTag) {
    auto tag = simd::get_best_tag();

    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx2_fma()) {
        EXPECT_TRUE(std::holds_alternative<simd::Avx2Tag>(tag));
    } else if (caps.has_sse42) {
        EXPECT_TRUE(std::holds_alternative<simd::SseTag>(tag));
    } else {
        EXPECT_TRUE(std::holds_alternative<simd::ScalarTag>(tag));
    }
}

// ============================================================================
// Kernel Correctness Tests (Scalar reference)
// ============================================================================

TEST(SimdKernelTest, AddScalar) {
    const size_t n = 100;
    std::vector<float> a(n), b(n), dst(n), expected(n);

    // Initialize with random values
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(i) * 0.2f + 1.0f;
        expected[i] = a[i] + b[i];
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::add_kernel(tag, dst.data(), a.data(), b.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(SimdKernelTest, SubScalar) {
    const size_t n = 50;
    std::vector<float> a(n), b(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 1.5f;
        b[i] = static_cast<float>(i) * 0.5f;
        expected[i] = a[i] - b[i];
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::sub_kernel(tag, dst.data(), a.data(), b.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, MulScalar) {
    const size_t n = 64;
    std::vector<float> a(n), b(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.1f + 1.0f;
        b[i] = static_cast<float>(i) * 0.2f + 0.5f;
        expected[i] = a[i] * b[i];
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::mul_kernel(tag, dst.data(), a.data(), b.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, DivScalar) {
    const size_t n = 32;
    std::vector<float> a(n), b(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1) * 2.0f;
        b[i] = static_cast<float>(i + 1) * 0.5f + 1.0f;
        expected[i] = a[i] / b[i];
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::div_kernel(tag, dst.data(), a.data(), b.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, AddBroadcastScalar) {
    const size_t n = 100;
    std::vector<float> a(n), dst(n), expected(n);
    float b = 3.5f;

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.1f;
        expected[i] = a[i] + b;
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::add_scalar_kernel(tag, dst.data(), a.data(), b, n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, MulBroadcastScalar) {
    const size_t n = 64;
    std::vector<float> a(n), dst(n), expected(n);
    float b = 2.5f;

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.1f + 1.0f;
        expected[i] = a[i] * b;
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::mul_scalar_kernel(tag, dst.data(), a.data(), b, n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, NegUnary) {
    const size_t n = 50;
    std::vector<float> a(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.5f - 10.0f;
        expected[i] = -a[i];
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::neg_kernel(tag, dst.data(), a.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, AbsUnary) {
    const size_t n = 50;
    std::vector<float> a(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = (i % 2 == 0) ? static_cast<float>(i) : -static_cast<float>(i);
        expected[i] = std::fabs(a[i]);
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::abs_kernel(tag, dst.data(), a.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, ExpUnary) {
    const size_t n = 20;
    std::vector<float> a(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.1f - 1.0f; // Range [-1, 1]
        expected[i] = std::exp(a[i]);
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::exp_kernel(tag, dst.data(), a.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, LogUnary) {
    const size_t n = 20;
    std::vector<float> a(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1) * 0.5f; // Range [0.5, 10]
        expected[i] = std::log(a[i]);
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::log_kernel(tag, dst.data(), a.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, ReluUnary) {
    const size_t n = 30;
    std::vector<float> a(n), dst(n), expected(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) - 10.0f; // Range [-10, 20]
        expected[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::relu_kernel(tag, dst.data(), a.data(), n);
    });

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], expected[i]);
    }
}

TEST(SimdKernelTest, SumReduction) {
    const size_t n = 1000;
    std::vector<float> a(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0f;
    }

    float result = simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        return simd::sum_kernel(tag, a.data(), n);
    });

    EXPECT_FLOAT_EQ(result, static_cast<float>(n));
}

TEST(SimdKernelTest, MeanReduction) {
    const size_t n = 100;
    std::vector<float> a(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1);
    }

    float result = simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        return simd::mean_kernel(tag, a.data(), n);
    });

    float expected = static_cast<float>(n + 1) / 2.0f;
    EXPECT_FLOAT_EQ(result, expected);
}

TEST(SimdKernelTest, SoftmaxActivation) {
    const size_t rows = 4, cols = 5;
    std::vector<float> src(rows * cols), dst(rows * cols);

    // Known values
    float vals[20] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
        10.0f, 20.0f, 30.0f, 40.0f, 50.0f
    };
    std::copy(vals, vals + 20, src.begin());

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::softmax_kernel(tag, dst.data(), src.data(), rows, cols);
    });

    // Verify each row sums to 1
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            sum += dst[r * cols + c];
            EXPECT_GE(dst[r * cols + c], 0.0f);
            EXPECT_LE(dst[r * cols + c], 1.0f);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(SimdKernelTest, LogSoftmaxActivation) {
    const size_t rows = 3, cols = 4;
    std::vector<float> src(rows * cols), dst(rows * cols);

    for (size_t i = 0; i < rows * cols; ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::log_softmax_kernel(tag, dst.data(), src.data(), rows, cols);
    });

    // Verify exp(log_softmax) ≈ softmax
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            sum += std::exp(dst[r * cols + c]);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(SimdKernelTest, MatmulNaive) {
    const size_t M = 4, N = 5, K = 3;
    std::vector<float> A(M * K), B(K * N), C(M * N), expected(M * N);

    // A = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    // B = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
    for (size_t i = 0; i < M * K; ++i) A[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < K * N; ++i) B[i] = static_cast<float>(i + 1);

    // Compute expected
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::matmul_kernel(tag, C.data(), A.data(), B.data(), M, N, K);
    });

    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(C[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(SimdKernelTest, SumDimReduction) {
    // 2x3 tensor, sum along dim=1 -> 2x1
    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    std::vector<float> dst(2);
    std::vector<size_t> shape = {2, 3};
    std::vector<ptrdiff_t> strides = {3, 1};

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::sum_dim_kernel(tag, dst.data(), src.data(),
                            shape.data(), strides.data(),
                            1, 2, 2, 3);
    });

    EXPECT_FLOAT_EQ(dst[0], 6.0f);  // 1+2+3
    EXPECT_FLOAT_EQ(dst[1], 15.0f); // 4+5+6
}

TEST(SimdKernelTest, SumDimReductionDim0) {
    // 2x3 tensor, sum along dim=0 -> 1x3
    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    std::vector<float> dst(3);
    std::vector<size_t> shape = {2, 3};
    std::vector<ptrdiff_t> strides = {3, 1};

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::sum_dim_kernel(tag, dst.data(), src.data(),
                            shape.data(), strides.data(),
                            0, 2, 1, 3);
    });

    EXPECT_FLOAT_EQ(dst[0], 5.0f); // 1+4
    EXPECT_FLOAT_EQ(dst[1], 7.0f); // 2+5
    EXPECT_FLOAT_EQ(dst[2], 9.0f); // 3+6
}

// ============================================================================
// AVX2 equivalence tests: each kernel must agree with its scalar reference.
// Sizes deliberately include non-multiples of 8 to cover the scalar tail.
// ============================================================================
#if defined(HAS_AVX2_KERNELS)

namespace {

const std::vector<size_t> kEquivSizes = {0, 1, 3, 7, 8, 9, 15, 16, 17, 33, 64, 100, 1000};

std::vector<float> make_vec(size_t n, float base, float step) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = base + step * static_cast<float>((i * 7) % 17);
    }
    return v;
}

template <typename Fn>
void expect_unary_equiv(Fn fn, float rtol, float atol,
                        float base, float step) {
    for (size_t n : kEquivSizes) {
        auto a = make_vec(n, base, step);
        std::vector<float> ds(n), dv(n);
        simd::static_dispatch<simd::ScalarTag>([&](auto tag) { fn(tag, ds.data(), a.data(), n); });
        fn(simd::Avx2Tag{}, dv.data(), a.data(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(ds[i], dv[i], atol + rtol * std::fabs(ds[i]))
                << "n=" << n << " i=" << i;
        }
    }
}

template <typename Fn>
void expect_binary_equiv(Fn fn, float rtol, float atol,
                         float base_a, float step_a, float base_b, float step_b) {
    for (size_t n : kEquivSizes) {
        auto a = make_vec(n, base_a, step_a);
        auto b = make_vec(n, base_b, step_b);
        std::vector<float> ds(n), dv(n);
        simd::static_dispatch<simd::ScalarTag>([&](auto tag) { fn(tag, ds.data(), a.data(), b.data(), n); });
        fn(simd::Avx2Tag{}, dv.data(), a.data(), b.data(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(ds[i], dv[i], atol + rtol * std::fabs(ds[i]))
                << "n=" << n << " i=" << i;
        }
    }
}

template <typename Fn>
void expect_scalar_bcast_equiv(Fn fn, float bval, float rtol, float atol) {
    for (size_t n : kEquivSizes) {
        auto a = make_vec(n, -2.0f, 0.23f);
        std::vector<float> ds(n), dv(n);
        simd::static_dispatch<simd::ScalarTag>([&](auto tag) { fn(tag, ds.data(), a.data(), bval, n); });
        fn(simd::Avx2Tag{}, dv.data(), a.data(), bval, n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(ds[i], dv[i], atol + rtol * std::fabs(ds[i]))
                << "n=" << n << " i=" << i;
        }
    }
}

template <typename Fn>
void expect_rscalar_bcast_equiv(Fn fn, float aval, float rtol, float atol) {
    for (size_t n : kEquivSizes) {
        auto b = make_vec(n, 0.5f, 0.31f); // positive: safe for rdiv
        std::vector<float> ds(n), dv(n);
        simd::static_dispatch<simd::ScalarTag>([&](auto tag) { fn(tag, ds.data(), aval, b.data(), n); });
        fn(simd::Avx2Tag{}, dv.data(), aval, b.data(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(ds[i], dv[i], atol + rtol * std::fabs(ds[i]))
                << "n=" << n << " i=" << i;
        }
    }
}

} // namespace

class Avx2EquivTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!simd::has_avx2_fma()) {
            GTEST_SKIP() << "CPU does not support AVX2+FMA";
        }
    }
};

// --- Binary: these are the same IEEE operations, so results must be exact. ---
TEST_F(Avx2EquivTest, Add) {
    expect_binary_equiv([](auto t, float* d, const float* a, const float* b, size_t n) { simd::add_kernel(t, d, a, b, n); },
                        0.0f, 0.0f, -2.0f, 0.23f, 1.0f, 0.31f);
}
TEST_F(Avx2EquivTest, Sub) {
    expect_binary_equiv([](auto t, float* d, const float* a, const float* b, size_t n) { simd::sub_kernel(t, d, a, b, n); },
                        0.0f, 0.0f, -2.0f, 0.23f, 1.0f, 0.31f);
}
TEST_F(Avx2EquivTest, Mul) {
    expect_binary_equiv([](auto t, float* d, const float* a, const float* b, size_t n) { simd::mul_kernel(t, d, a, b, n); },
                        0.0f, 0.0f, -2.0f, 0.23f, 1.0f, 0.31f);
}
TEST_F(Avx2EquivTest, Div) {
    expect_binary_equiv([](auto t, float* d, const float* a, const float* b, size_t n) { simd::div_kernel(t, d, a, b, n); },
                        0.0f, 0.0f, 0.5f, 0.23f, 0.5f, 0.31f);
}

// --- Broadcast variants ---
TEST_F(Avx2EquivTest, AddScalarBcast) {
    expect_scalar_bcast_equiv([](auto t, float* d, const float* a, float b, size_t n) { simd::add_scalar_kernel(t, d, a, b, n); },
                              3.5f, 0.0f, 0.0f);
}
TEST_F(Avx2EquivTest, SubScalarBcast) {
    expect_scalar_bcast_equiv([](auto t, float* d, const float* a, float b, size_t n) { simd::sub_scalar_kernel(t, d, a, b, n); },
                              1.25f, 0.0f, 0.0f);
}
TEST_F(Avx2EquivTest, RsubScalarBcast) {
    expect_rscalar_bcast_equiv([](auto t, float* d, float a, const float* b, size_t n) { simd::rsub_scalar_kernel(t, d, a, b, n); },
                               5.0f, 0.0f, 0.0f);
}
TEST_F(Avx2EquivTest, MulScalarBcast) {
    expect_scalar_bcast_equiv([](auto t, float* d, const float* a, float b, size_t n) { simd::mul_scalar_kernel(t, d, a, b, n); },
                              -0.75f, 0.0f, 0.0f);
}
TEST_F(Avx2EquivTest, DivScalarBcast) {
    expect_scalar_bcast_equiv([](auto t, float* d, const float* a, float b, size_t n) { simd::div_scalar_kernel(t, d, a, b, n); },
                              2.0f, 0.0f, 0.0f);
}
TEST_F(Avx2EquivTest, RdivScalarBcast) {
    expect_rscalar_bcast_equiv([](auto t, float* d, float a, const float* b, size_t n) { simd::rdiv_scalar_kernel(t, d, a, b, n); },
                               4.0f, 0.0f, 0.0f);
}

// --- Unary: exact operations ---
TEST_F(Avx2EquivTest, Neg) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::neg_kernel(t, d, a, n); },
                       0.0f, 0.0f, -2.0f, 0.23f);
}
TEST_F(Avx2EquivTest, Abs) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::abs_kernel(t, d, a, n); },
                       0.0f, 0.0f, -2.0f, 0.23f);
}
TEST_F(Avx2EquivTest, Relu) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::relu_kernel(t, d, a, n); },
                       0.0f, 0.0f, -2.0f, 0.23f);
}
TEST_F(Avx2EquivTest, Sqrt) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::sqrt_kernel(t, d, a, n); },
                       0.0f, 0.0f, 0.5f, 0.23f);
}

// --- Unary: polynomial approximations, validated within tolerance ---
TEST_F(Avx2EquivTest, Exp) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::exp_kernel(t, d, a, n); },
                       1e-4f, 1e-4f, -3.0f, 0.31f);
}
TEST_F(Avx2EquivTest, Log) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::log_kernel(t, d, a, n); },
                       1e-4f, 1e-4f, 0.5f, 0.31f);
}
TEST_F(Avx2EquivTest, Sigmoid) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::sigmoid_kernel(t, d, a, n); },
                       1e-4f, 1e-4f, -4.0f, 0.41f);
}
TEST_F(Avx2EquivTest, Tanh) {
    expect_unary_equiv([](auto t, float* d, const float* a, size_t n) { simd::tanh_kernel(t, d, a, n); },
                       1e-4f, 1e-4f, -4.0f, 0.41f);
}

// --- Reductions ---
TEST_F(Avx2EquivTest, SumReduction) {
    for (size_t n : kEquivSizes) {
        auto a = make_vec(n, -2.0f, 0.23f);
        float ds = simd::static_dispatch<simd::ScalarTag>([&](auto tag) { return simd::sum_kernel(tag, a.data(), n); });
        float dv = simd::sum_kernel(simd::Avx2Tag{}, a.data(), n);
        EXPECT_NEAR(ds, dv, 1e-4f + 1e-4f * std::fabs(ds)) << "n=" << n;
    }
}

TEST_F(Avx2EquivTest, MeanReduction) {
    for (size_t n : kEquivSizes) {
        if (n == 0) continue; // mean(0) is undefined
        auto a = make_vec(n, -2.0f, 0.23f);
        float ds = simd::static_dispatch<simd::ScalarTag>([&](auto tag) { return simd::mean_kernel(tag, a.data(), n); });
        float dv = simd::mean_kernel(simd::Avx2Tag{}, a.data(), n);
        EXPECT_NEAR(ds, dv, 1e-4f + 1e-4f * std::fabs(ds)) << "n=" << n;
    }
}

TEST_F(Avx2EquivTest, SumDimReduction) {
    // 2x3 tensor, sum along dim=1 -> 2x1
    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    std::vector<float> ds(2), dv(2);
    std::vector<size_t> shape = {2, 3};
    std::vector<ptrdiff_t> strides = {3, 1};

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::sum_dim_kernel(tag, ds.data(), src.data(),
                            shape.data(), strides.data(),
                            1, 2, 2, 3);
    });
    simd::sum_dim_kernel(simd::Avx2Tag{}, dv.data(), src.data(),
                         shape.data(), strides.data(),
                         1, 2, 2, 3);

    EXPECT_FLOAT_EQ(ds[0], dv[0]);
    EXPECT_FLOAT_EQ(ds[1], dv[1]);
}

TEST_F(Avx2EquivTest, SumDimReductionDim0) {
    // 2x3 tensor, sum along dim=0 -> 1x3
    std::vector<float> src = {1, 2, 3, 4, 5, 6};
    std::vector<float> ds(3), dv(3);
    std::vector<size_t> shape = {2, 3};
    std::vector<ptrdiff_t> strides = {3, 1};

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::sum_dim_kernel(tag, ds.data(), src.data(),
                            shape.data(), strides.data(),
                            0, 2, 1, 3);
    });
    simd::sum_dim_kernel(simd::Avx2Tag{}, dv.data(), src.data(),
                         shape.data(), strides.data(),
                         0, 2, 1, 3);

    EXPECT_FLOAT_EQ(ds[0], dv[0]);
    EXPECT_FLOAT_EQ(ds[1], dv[1]);
    EXPECT_FLOAT_EQ(ds[2], dv[2]);
}

// --- Activations ---
TEST_F(Avx2EquivTest, Softmax) {
    const size_t rows = 4, cols = 5;
    std::vector<float> src(rows * cols), ds(rows * cols), dv(rows * cols);

    float vals[20] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
        10.0f, 20.0f, 30.0f, 40.0f, 50.0f
    };
    std::copy(vals, vals + 20, src.begin());

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::softmax_kernel(tag, ds.data(), src.data(), rows, cols);
    });
    simd::softmax_kernel(simd::Avx2Tag{}, dv.data(), src.data(), rows, cols);

    for (size_t i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(ds[i], dv[i], 1e-4f + 1e-4f * std::fabs(ds[i])) << "i=" << i;
    }
}

TEST_F(Avx2EquivTest, LogSoftmax) {
    const size_t rows = 3, cols = 4;
    std::vector<float> src(rows * cols), ds(rows * cols), dv(rows * cols);

    for (size_t i = 0; i < rows * cols; ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
    }

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::log_softmax_kernel(tag, ds.data(), src.data(), rows, cols);
    });
    simd::log_softmax_kernel(simd::Avx2Tag{}, dv.data(), src.data(), rows, cols);

    for (size_t i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(ds[i], dv[i], 1e-4f + 1e-4f * std::fabs(ds[i])) << "i=" << i;
    }
}

// --- Matmul ---
TEST_F(Avx2EquivTest, Matmul) {
    const size_t M = 4, N = 5, K = 3;
    std::vector<float> A(M * K), B(K * N), ds(M * N), dv(M * N);

    for (size_t i = 0; i < M * K; ++i) A[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < K * N; ++i) B[i] = static_cast<float>(i + 1);

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::matmul_kernel(tag, ds.data(), A.data(), B.data(), M, N, K);
    });
    simd::matmul_kernel(simd::Avx2Tag{}, dv.data(), A.data(), B.data(), M, N, K);

    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(ds[i], dv[i], 1e-4f + 1e-4f * std::fabs(ds[i])) << "i=" << i;
    }
}

TEST_F(Avx2EquivTest, MatmulLarge) {
    // Test with sizes that require blocking
    const size_t M = 16, N = 16, K = 16;
    std::vector<float> A(M * K), B(K * N), ds(M * N), dv(M * N);

    for (size_t i = 0; i < M * K; ++i) A[i] = static_cast<float>((i * 7) % 17) * 0.1f;
    for (size_t i = 0; i < K * N; ++i) B[i] = static_cast<float>((i * 11) % 13) * 0.1f;

    simd::static_dispatch<simd::ScalarTag>([&](auto tag) {
        simd::matmul_kernel(tag, ds.data(), A.data(), B.data(), M, N, K);
    });
    simd::matmul_kernel(simd::Avx2Tag{}, dv.data(), A.data(), B.data(), M, N, K);

    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(ds[i], dv[i], 1e-3f + 1e-3f * std::fabs(ds[i])) << "i=" << i;
    }
}

#endif // HAS_AVX2_KERNELS