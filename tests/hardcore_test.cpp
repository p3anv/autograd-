#include "autograd/tensor.hpp"
#include "autograd/operations.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>
#include <cassert>

// Helper function to compare floats with tolerance
bool approx_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

// Helper function to compare tensors
bool tensors_equal(const autograd::Tensor& a, const autograd::Tensor& b, float eps = 1e-5f) {
    if (a.shape() != b.shape()) {
        return false;
    }
    const float* a_data = a.data();
    const float* b_data = b.data();
    for (size_t i = 0; i < a.numel(); ++i) {
        if (!approx_equal(a_data[i], b_data[i], eps)) {
            return false;
        }
    }
    return true;
}

// Helper to create tensor from vector (since we don't have from_vector)
autograd::Tensor tensor_from_vector(const std::vector<float>& data, const std::vector<std::size_t>& shape) {
    autograd::Tensor result(shape, false);
    std::copy(data.begin(), data.end(), result.data());
    return result;
}

// Helper to print tensor
void print_tensor(const autograd::Tensor& t, const std::string& name) {
    std::cout << name << " [shape: ";
    auto shape = t.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << "x";
    }
    std::cout << "]:\n";
    const float* data = t.data();
    for (size_t i = 0; i < t.numel(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << data[i] << " ";
        if (!shape.empty() && (i + 1) % shape.back() == 0) {
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== HARDCORE TEST FOR PHASE 3 OPERATIONS ===\n\n";

    int passed = 0;
    int failed = 0;

    auto test_case = [&](const std::string& name, bool condition) {
        if (condition) {
            std::cout << "[PASS] " << name << std::endl;
            passed++;
        } else {
            std::cout << "[FAIL] " << name << std::endl;
            failed++;
        }
    };

    try {
    
    // Test 1: POW operation
    std::cout << "\n--- Testing POW operation ---\n";
    {
        // Scalar pow
        autograd::Tensor base = autograd::Tensor::scalar(2.0f);
        autograd::Tensor exp = autograd::Tensor::scalar(3.0f);
        autograd::Tensor result = autograd::pow(base, exp);
        test_case("Scalar pow: 2^3 = 8", approx_equal(result({}), 8.0f));
        
        // Vector pow (scalar base, vector exponent)
        autograd::Tensor base2 = autograd::Tensor::scalar(2.0f);
        autograd::Tensor exp2 = autograd::Tensor::zeros({3}, false);
        exp2({0}) = 0.0f; // 2^0 = 1
        exp2({1}) = 1.0f; // 2^1 = 2
        exp2({2}) = 2.0f; // 2^2 = 4
        autograd::Tensor result2 = autograd::pow(base2, exp2);
        std::vector<float> expected_data2 = {1.0f, 2.0f, 4.0f};
        autograd::Tensor expected2 = tensor_from_vector(expected_data2, {3});
        test_case("Vector pow (scalar base): 2^[0,1,2] = [1,2,4]", 
                  tensors_equal(result2, expected2));
        
        // Vector pow (vector base, scalar exponent)
        std::vector<float> base3_data = {2.0f, 3.0f, 4.0f};
        autograd::Tensor base3 = tensor_from_vector(base3_data, {3});
        autograd::Tensor exp3 = autograd::Tensor::scalar(2.0f);
        autograd::Tensor result3 = autograd::pow(base3, exp3);
        std::vector<float> expected3_data = {4.0f, 9.0f, 16.0f};
        autograd::Tensor expected3 = tensor_from_vector(expected3_data, {3});
        test_case("Vector pow (vector base): [2,3,4]^2 = [4,9,16]", 
                  tensors_equal(result3, expected3));
        
        // Broadcasting test
        std::vector<float> base4_data = {2.0f, 3.0f};
        autograd::Tensor base4 = tensor_from_vector(base4_data, {2}); // shape [2]
        std::vector<float> exp4_data = {1.0f, 2.0f, 3.0f, 4.0f};
        autograd::Tensor exp4 = tensor_from_vector(exp4_data, {2, 2}); // shape [2,2]
        // Should broadcast base4 to [2,2] -> [2,3] ^ [1,2;3,4]
        autograd::Tensor result4 = autograd::pow(base4, exp4);
        std::vector<float> expected4_data = {
            std::pow(2.0f, 1.0f), std::pow(3.0f, 2.0f),
            std::pow(2.0f, 3.0f), std::pow(3.0f, 4.0f)
        };
        autograd::Tensor expected4 = tensor_from_vector(expected4_data, {2, 2}); // [2,9;8,81]
        test_case("Broadcasting pow: [2,3] ^ [[1,2],[3,4]] = [[2,9],[8,81]]",
                  tensors_equal(result4, expected4));
    }
    
    // Test 2: SOFTMAX operation (numerical stability)
    std::cout << "\n--- Testing SOFTMAX operation ---\n";
    {
        // Basic softmax
        std::vector<float> logits_data = {1.0f, 2.0f, 3.0f};
        autograd::Tensor logits = tensor_from_vector(logits_data, {3});
        autograd::Tensor softmax_result = autograd::softmax(logits);
        // Manual calculation: e^x / sum(e^x)
        float e1 = std::exp(1.0f);
        float e2 = std::exp(2.0f);
        float e3 = std::exp(3.0f);
        float sum1 = e1 + e2 + e3;
        std::vector<float> expected_data = {e1/sum1, e2/sum1, e3/sum1};
        autograd::Tensor expected = tensor_from_vector(expected_data, {3});
        test_case("Basic softmax: [1,2,3]", tensors_equal(softmax_result, expected, 1e-5f));
        
        // Numerical stability test with large values
        std::vector<float> large_logits_data = {1000.0f, 1001.0f, 1002.0f};
        autograd::Tensor large_logits = tensor_from_vector(large_logits_data, {3});
        autograd::Tensor stable_result = autograd::softmax(large_logits);
        // Should not produce NaN or inf
        const float* stable_data = stable_result.data();
        bool no_nan_inf = true;
        for (size_t i = 0; i < 3; ++i) {
            if (std::isnan(stable_data[i]) || std::isinf(stable_data[i])) {
                no_nan_inf = false;
                break;
            }
        }
        test_case("Numerical stability with large values", no_nan_inf);
        
        // Test that result sums to 1
        float sum2 = 0.0f;
        for (size_t i = 0; i < 3; ++i) {
            sum2 += stable_data[i];
        }
        test_case("Softmax sums to 1 (large values)", approx_equal(sum2, 1.0f, 1e-5f));
        
        // Test 2D softmax (applied to last dimension)
        std::vector<float> logits2d_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        autograd::Tensor logits2d = tensor_from_vector(logits2d_data, {2, 3});
        autograd::Tensor softmax2d = autograd::softmax(logits2d);
        // Debug: print shape
        std::cerr << "softmax2d shape: [";
        auto s2 = softmax2d.shape();
        for (size_t i = 0; i < s2.size(); ++i) {
            std::cerr << s2[i];
            if (i < s2.size()-1) std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        // Each row should sum to 1
        float sum3 = softmax2d({0,0}) + softmax2d({0,1}) + softmax2d({0,2});
        float sum4 = softmax2d({1,0}) + softmax2d({1,1}) + softmax2d({1,2});
        test_case("2D softmax row 1 sums to 1", approx_equal(sum3, 1.0f, 1e-5f));
        test_case("2D softmax row 2 sums to 1", approx_equal(sum4, 1.0f, 1e-5f));
        
        // Edge case: zeros
        autograd::Tensor zeros = autograd::Tensor::zeros({3}, false);
        autograd::Tensor softmax_zeros = autograd::softmax(zeros);
        std::vector<float> expected_zeros_data = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
        autograd::Tensor expected_zeros = tensor_from_vector(expected_zeros_data, {3});
        test_case("Softmax of zeros", tensors_equal(softmax_zeros, expected_zeros, 1e-5f));
    }
    
    // Test 3: SUM operation
    std::cout << "\n--- Testing SUM operation ---\n";
    std::cout << "DEBUG: Entering SUM test section" << std::endl << std::flush;
    {
        // 1D sum - scalar result
        std::vector<float> vec_data = {1.0f, 2.0f, 3.0f, 4.0f};
        autograd::Tensor vec = tensor_from_vector(vec_data, {4});
        autograd::Tensor sum_result = autograd::sum(vec, 0);
        test_case("1D sum: sum([1,2,3,4]) = 10", approx_equal(sum_result({}), 10.0f));
        
        // 2D sum along dim 0 (rows)
        std::vector<float> mat_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        autograd::Tensor mat = tensor_from_vector(mat_data, {2, 3});
        autograd::Tensor sum_dim0 = autograd::sum(mat, 0); // sum along rows -> [1+4, 2+5, 3+6] = [5,7,9]
        std::vector<float> expected_sum0_data = {5.0f, 7.0f, 9.0f};
        autograd::Tensor expected_sum0 = tensor_from_vector(expected_sum0_data, {3}); // dim removed
        // Debug: print actual result
        std::cerr << "sum_dim0 shape: [";
        auto s0 = sum_dim0.shape();
        for (size_t i = 0; i < s0.size(); ++i) {
            std::cerr << s0[i];
            if (i < s0.size()-1) std::cerr << "x";
        }
        std::cerr << "] data: ";
        const float* sum0_data = sum_dim0.data();
        for (size_t i = 0; i < sum_dim0.numel(); ++i) {
            std::cerr << sum0_data[i] << " ";
        }
        std::cerr << std::endl;
        test_case("2D sum along dim 0", tensors_equal(sum_dim0, expected_sum0));

        // 2D sum along dim 1 (columns)
        autograd::Tensor sum_dim1 = autograd::sum(mat, 1); // sum along columns -> [1+2+3, 4+5+6] = [6,15]
        std::vector<float> expected_sum1_data = {6.0f, 15.0f};
        autograd::Tensor expected_sum1 = tensor_from_vector(expected_sum1_data, {2}); // dim removed
        // Debug: print actual result
        std::cerr << "sum_dim1 shape: [";
        auto s1 = sum_dim1.shape();
        for (size_t i = 0; i < s1.size(); ++i) {
            std::cerr << s1[i];
            if (i < s1.size()-1) std::cerr << "x";
        }
        std::cerr << "] data: ";
        const float* sum1_data = sum_dim1.data();
        for (size_t i = 0; i < sum_dim1.numel(); ++i) {
            std::cerr << sum1_data[i] << " ";
        }
        std::cerr << std::endl;
        test_case("2D sum along dim 1", tensors_equal(sum_dim1, expected_sum1));
        
        // Zero-sized tensor
        autograd::Tensor zero_tensor = autograd::Tensor::zeros({0, 5}, false);
        autograd::Tensor sum_zero = autograd::sum(zero_tensor, 1);
        test_case("Sum of zero-sized tensor", sum_zero.numel() == 0);
        
        // Sum behavior (our implementation removes the dimension, following NumPy convention)
        autograd::Tensor sum_keepdim = autograd::sum(vec, 0);
        test_case("Sum removes dimension (scalar output)", sum_keepdim.numel() == 1 && sum_keepdim.dim() == 0);
    }
    
    // Test 4: MEAN operation
    std::cout << "\n--- Testing MEAN operation ---\n";
    {
        // 1D mean - scalar result
        std::vector<float> vec_data = {1.0f, 2.0f, 3.0f, 4.0f};
        autograd::Tensor vec = tensor_from_vector(vec_data, {4});
        autograd::Tensor mean_result = autograd::mean(vec, 0);
        test_case("1D mean: mean([1,2,3,4]) = 2.5", approx_equal(mean_result({}), 2.5f));
        
        // 2D mean along dim 0
        std::vector<float> mat_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        autograd::Tensor mat = tensor_from_vector(mat_data, {2, 3});
        autograd::Tensor mean_dim0 = autograd::mean(mat, 0); // [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        std::vector<float> expected_mean0_data = {2.5f, 3.5f, 4.5f};
        autograd::Tensor expected_mean0 = tensor_from_vector(expected_mean0_data, {3}); // dim removed
        test_case("2D mean along dim 0", tensors_equal(mean_dim0, expected_mean0));

        // 2D mean along dim 1
        autograd::Tensor mean_dim1 = autograd::mean(mat, 1); // [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
        std::vector<float> expected_mean1_data = {2.0f, 5.0f};
        autograd::Tensor expected_mean1 = tensor_from_vector(expected_mean1_data, {2}); // dim removed
        test_case("2D mean along dim 1", tensors_equal(mean_dim1, expected_mean1));
        
        // Mean of zeros
        autograd::Tensor zeros = autograd::Tensor::zeros({4}, false);
        autograd::Tensor mean_zeros = autograd::mean(zeros, 0);
        test_case("Mean of zeros", approx_equal(mean_zeros({}), 0.0f));
    }
    
    // Test 5: MAX operation
    std::cout << "\n--- Testing MAX operation ---\n";
    {
        // 1D max - scalar result
        std::vector<float> vec_data = {1.0f, 5.0f, 3.0f, 2.0f};
        autograd::Tensor vec = tensor_from_vector(vec_data, {4});
        autograd::Tensor max_result = autograd::max(vec, 0);
        test_case("1D max: max([1,5,3,2]) = 5", approx_equal(max_result({}), 5.0f));
        
        // 2D max along dim 0
        std::vector<float> mat_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 0.0f, 6.0f
        };
        autograd::Tensor mat = tensor_from_vector(mat_data, {2, 3});
        autograd::Tensor max_dim0 = autograd::max(mat, 0); // [max(1,4), max(2,0), max(3,6)] = [4,2,6]
        std::vector<float> expected_max0_data = {4.0f, 2.0f, 6.0f};
        autograd::Tensor expected_max0 = tensor_from_vector(expected_max0_data, {3}); // dim removed
        test_case("2D max along dim 0", tensors_equal(max_dim0, expected_max0));
        
        // 2D max along dim 1
        autograd::Tensor max_dim1 = autograd::max(mat, 1); // [max(1,2,3), max(4,0,6)] = [3,6]
        std::vector<float> expected_max1_data = {3.0f, 6.0f};
        autograd::Tensor expected_max1 = tensor_from_vector(expected_max1_data, {2}); // dim removed
        test_case("2D max along dim 1", tensors_equal(max_dim1, expected_max1));
        
        // Test with negative values
        std::vector<float> neg_vec_data = {-5.0f, -2.0f, -10.0f, -1.0f};
        autograd::Tensor neg_vec = tensor_from_vector(neg_vec_data, {4});
        autograd::Tensor max_neg = autograd::max(neg_vec, 0);
        test_case("Max with negatives: max([-5,-2,-10,-1]) = -1", approx_equal(max_neg({}), -1.0f));
    }
    
    // Test 6: MIN operation
    std::cout << "\n--- Testing MIN operation ---\n";
    {
        // 1D min
        std::vector<float> vec_data = {1.0f, 5.0f, 3.0f, 2.0f};
        autograd::Tensor vec = tensor_from_vector(vec_data, {4});
        autograd::Tensor min_result = autograd::min(vec, 0);
        test_case("1D min: min([1,5,3,2]) = 1", approx_equal(min_result({}), 1.0f));
        
        // 2D min along dim 0
        std::vector<float> mat_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 0.0f, 6.0f
        };
        autograd::Tensor mat = tensor_from_vector(mat_data, {2, 3});
        autograd::Tensor min_dim0 = autograd::min(mat, 0); // [min(1,4), min(2,0), min(3,6)] = [1,0,3]
        std::vector<float> expected_min0_data = {1.0f, 0.0f, 3.0f};
        autograd::Tensor expected_min0 = tensor_from_vector(expected_min0_data, {3}); // dim removed
        test_case("2D min along dim 0", tensors_equal(min_dim0, expected_min0));
        
        // 2D min along dim 1
        std::cout << "DEBUG: About to create min_dim1" << std::endl;
        autograd::Tensor min_dim1 = autograd::min(mat, 1); // [min(1,2,3), min(4,0,6)] = [1,0]
        std::cout << "DEBUG: min_dim1 created, shape = [";
        auto m1shape = min_dim1.shape();
        for (size_t i = 0; i < m1shape.size(); ++i) {
            std::cout << m1shape[i];
            if (i < m1shape.size()-1) std::cout << ",";
        }
        std::cout << "], numel = " << min_dim1.numel() << std::endl;
        std::cout << "DEBUG: About to create expected_min1_data" << std::endl;
        std::vector<float> expected_min1_data = {1.0f, 0.0f};
        std::cout << "DEBUG: About to create expected_min1 tensor" << std::endl;
        autograd::Tensor expected_min1 = tensor_from_vector(expected_min1_data, {2}); // dim removed
        std::cout << "DEBUG: expected_min1 created, shape = [";
        auto eshape = expected_min1.shape();
        for (size_t i = 0; i < eshape.size(); ++i) {
            std::cout << eshape[i];
            if (i < eshape.size()-1) std::cout << ",";
        }
        std::cout << "], numel = " << expected_min1.numel() << std::endl;
        std::cout << "DEBUG: About to call tensors_equal" << std::endl;
        bool equal = tensors_equal(min_dim1, expected_min1);
        std::cout << "DEBUG: tensors_equal returned " << (equal ? "true" : "false") << std::endl;
        test_case("2D min along dim 1", equal);
        
        // Test with negative values
        std::cout << "DEBUG: Entering Test with negative values section" << std::endl;
        std::vector<float> neg_vec_data = {-5.0f, -2.0f, -10.0f, -1.0f};
        std::cout << "DEBUG: Creating neg_vec tensor" << std::endl;
        autograd::Tensor neg_vec = tensor_from_vector(neg_vec_data, {4});
        std::cout << "DEBUG: Calling min on neg_vec" << std::endl;
        autograd::Tensor min_neg = autograd::min(neg_vec, 0);
        std::cout << "DEBUG: min_neg shape = [";
        auto mshape = min_neg.shape();
        for (size_t i = 0; i < mshape.size(); ++i) {
            std::cout << mshape[i];
            if (i < mshape.size()-1) std::cout << ",";
        }
        std::cout << "], numel = " << min_neg.numel() << std::endl;
        std::cout << "DEBUG: Accessing min_neg({})" << std::endl;
        float min_val = min_neg({});
        std::cout << "DEBUG: min_val = " << min_val << std::endl;
        test_case("Min with negatives: min([-5,-2,-10,-1]) = -10", approx_equal(min_val, -10.0f));
    }
    std::cout << "DEBUG: Just finished MIN test section, about to start LOG_SOFTMAX" << std::endl;
    
    // Test 7: LOG_SOFTMAX operation
    {
        try {
            std::cout << "\n--- Testing LOG_SOFTMAX operation ---\n" << std::endl;
            // Basic log_softmax
            std::vector<float> logits_data = {1.0f, 2.0f, 3.0f};
            autograd::Tensor logits = tensor_from_vector(logits_data, {3});
            autograd::Tensor log_softmax_result = autograd::log_softmax(logits);
            // log_softmax(x) = log(softmax(x))
            autograd::Tensor softmax_ref = autograd::softmax(logits);
            // Debug: print shape of softmax_ref
            // std::cerr << "softmax_ref shape: [";
            // auto s = softmax_ref.shape();
            // for (size_t i = 0; i < s.size(); ++i) {
            //     std::cerr << s[i];
            //     if (i < s.size()-1) std::cerr << ", ";
            // }
            // std::cerr << "]" << std::endl;
            std::vector<float> expected_data = {
                std::log(softmax_ref({0})),
                std::log(softmax_ref({1})),
                std::log(softmax_ref({2}))
            };
            autograd::Tensor expected = tensor_from_vector(expected_data, {3});
            test_case("Basic log_softmax matches log(softmax)", tensors_equal(log_softmax_result, expected, 1e-5f));

            // Numerical stability test
            std::vector<float> large_logits_data = {1000.0f, 1001.0f, 1002.0f};
            autograd::Tensor large_logits = tensor_from_vector(large_logits_data, {3});
            autograd::Tensor stable_log_softmax = autograd::log_softmax(large_logits);
            const float* lsm_data = stable_log_softmax.data();
            bool no_nan_inf = true;
            for (size_t i = 0; i < 3; ++i) {
                if (std::isnan(lsm_data[i]) || std::isinf(lsm_data[i])) {
                    no_nan_inf = false;
                    break;
                }
            }
            test_case("Log softmax numerical stability", no_nan_inf);

            // Test that exponentiating log_softmax gives softmax (within tolerance)
            autograd::Tensor from_log = autograd::exp(stable_log_softmax);
            autograd::Tensor direct_softmax = autograd::softmax(large_logits);
            test_case("exp(log_softmax) ≈ softmax", tensors_equal(from_log, direct_softmax, 1e-5f));

            // 2D test
            std::vector<float> logits2d_data = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            };
            autograd::Tensor logits2d = tensor_from_vector(logits2d_data, {2, 3});
            autograd::Tensor log_softmax2d = autograd::log_softmax(logits2d);
            // Each row of exp(log_softmax2d) should sum to 1
            autograd::Tensor exp_result = autograd::exp(log_softmax2d);
            float sum5 = exp_result({0,0}) + exp_result({0,1}) + exp_result({0,2});
            float sum6 = exp_result({1,0}) + exp_result({1,1}) + exp_result({1,2});
            test_case("2D log_softmax: exp(row1) sums to 1", approx_equal(sum5, 1.0f, 1e-5f));
            test_case("2D log_softmax: exp(row2) sums to 1", approx_equal(sum6, 1.0f, 1e-5f));
        } catch (const std::exception& e) {
            std::cerr << "Exception in LOG_SOFTMAX test: " << e.what() << std::endl;
            throw;
        }
    }
    
    // Test 8: MATMUL operation
    std::cout << "\n--- Testing MATMUL operation ---\n";
    {
        // Basic 2x2 matmul
        std::vector<float> a_data = {
            1.0f, 2.0f,
            3.0f, 4.0f
        };
        autograd::Tensor a = tensor_from_vector(a_data, {2, 2}); // [1,2;3,4]
        std::vector<float> b_data = {
            5.0f, 6.0f,
            7.0f, 8.0f
        };
        autograd::Tensor b = tensor_from_vector(b_data, {2, 2}); // [5,6;7,8]
        // Result: [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19,22;43,50]
        autograd::Tensor result = autograd::matmul(a, b);
        std::vector<float> expected_data = {
            19.0f, 22.0f,
            43.0f, 50.0f
        };
        autograd::Tensor expected = tensor_from_vector(expected_data, {2, 2});
        test_case("Basic 2x2 matmul", tensors_equal(result, expected));
        
        // Non-square matrices: 2x3 * 3x2 = 2x2
        std::vector<float> a2_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        autograd::Tensor a2 = tensor_from_vector(a2_data, {2, 3}); // [1,2,3;4,5,6]
        std::vector<float> b2_data = {
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f
        };
        autograd::Tensor b2 = tensor_from_vector(b2_data, {3, 2}); // [7,8;9,10;11,12]
        // Result[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33=58
        // Result[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36=64
        // Result[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66=139
        // Result[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72=154
        autograd::Tensor result2 = autograd::matmul(a2, b2);
        std::vector<float> expected2_data = {
            58.0f, 64.0f,
            139.0f, 154.0f
        };
        autograd::Tensor expected2 = tensor_from_vector(expected2_data, {2, 2});
        test_case("2x3 * 3x2 matmul", tensors_equal(result2, expected2));
        
        // Test with identity matrix
        std::vector<float> I_data = {
            1.0f, 0.0f,
            0.0f, 1.0f
        };
        autograd::Tensor I = tensor_from_vector(I_data, {2, 2}); // 2x2 identity
        std::vector<float> x_data = {
            5.0f, 6.0f,
            7.0f, 8.0f
        };
        autograd::Tensor x = tensor_from_vector(x_data, {2, 2});
        autograd::Tensor resultI = autograd::matmul(I, x);
        test_case("Identity matrix * X = X", tensors_equal(resultI, x));
        
        autograd::Tensor resultI2 = autograd::matmul(x, I);
        test_case("X * Identity matrix = X", tensors_equal(resultI2, x));
        
        // Test zero matrix
        autograd::Tensor zero = autograd::Tensor::zeros({2, 2}, false);
        autograd::Tensor resultZero = autograd::matmul(zero, x);
        autograd::Tensor expectedZero = autograd::Tensor::zeros({2, 2}, false);
        test_case("Zero matrix * X = Zero", tensors_equal(resultZero, expectedZero));
        
        // Test error conditions - non-contiguous tensors
        try {
            autograd::Tensor noncontig = autograd::Tensor::zeros({2, 3}, false).slice(1, 0, 2); // Makes non-contiguous
            autograd::Tensor other = autograd::Tensor::zeros({3, 2}, false);
            autograd::matmul(noncontig, other); // Should throw
            test_case("Matmul rejects non-contiguous first argument", false);
        } catch (const std::invalid_argument& e) {
            test_case("Matmul rejects non-contiguous first argument", true);
        } catch (...) {
            test_case("Matmul rejects non-contiguous first argument (wrong exception)", false);
        }
        
        try {
            autograd::Tensor a = autograd::Tensor::zeros({2, 3}, false);
            autograd::Tensor noncontig = autograd::Tensor::zeros({3, 2}, false).slice(0, 0, 2); // Makes non-contiguous
            autograd::matmul(a, noncontig); // Should throw
            test_case("Matmul rejects non-contiguous second argument", false);
        } catch (const std::invalid_argument& e) {
            test_case("Matmul rejects non-contiguous second argument", true);
        } catch (...) {
            test_case("Matmul rejects non-contiguous second argument (wrong exception)", false);
        }
        
        // Test error conditions - wrong dimensions
        try {
            autograd::Tensor a = autograd::Tensor::zeros({2, 3}, false); // 2x3
            autograd::Tensor b = autograd::Tensor::zeros({2, 2}, false); // 2x2
            autograd::matmul(a, b); // Inner dimensions 3 != 2 should throw
            test_case("Matmul rejects incompatible inner dimensions", false);
        } catch (const std::invalid_argument& e) {
            test_case("Matmul rejects incompatible inner dimensions", true);
        } catch (...) {
            test_case("Matmul rejects incompatible inner dimensions (wrong exception)", false);
        }
        
        // Test error conditions - not rank-2
        try {
            std::vector<float> a3d_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            autograd::Tensor a = tensor_from_vector(a3d_data, {2, 2, 2}); // 3D tensor
            autograd::Tensor b = autograd::Tensor::zeros({2, 2}, false); // 2D tensor
            autograd::matmul(a, b); // First arg not rank-2 should throw
            test_case("Matmul rejects non-rank-2 first argument", false);
        } catch (const std::invalid_argument& e) {
            test_case("Matmul rejects non-rank-2 first argument", true);
        } catch (...) {
            test_case("Matmul rejects non-rank-2 first argument (wrong exception)", false);
        }
    }
    
    // Test 9: Edge cases and error handling
    std::cout << "\n--- Testing EDGE CASES and ERROR HANDLING ---\n";
    {
        // Test dimension out of bounds for reductions
        try {
            autograd::Tensor t = autograd::Tensor::zeros({2, 3}, false);
            autograd::sum(t, 5); // dim 5 >= 2 dimensions should throw
            test_case("Sum rejects out-of-bounds dimension", false);
        } catch (const std::invalid_argument& e) {
            test_case("Sum rejects out-of-bounds dimension", true);
        } catch (...) {
            test_case("Sum rejects out-of-bounds dimension (wrong exception)", false);
        }
        
        try {
            autograd::Tensor t = autograd::Tensor::zeros({2, 3}, false);
            autograd::mean(t, -1); // negative dimension should throw
            test_case("Mean rejects negative dimension", false);
        } catch (const std::invalid_argument& e) {
            test_case("Mean rejects negative dimension", true);
        } catch (...) {
            test_case("Mean rejects negative dimension (wrong exception)", false);
        }
        
        // Test broadcasting edge cases
        {
            // Scalar + tensor
            autograd::Tensor scalar = autograd::Tensor::scalar(5.0f);
            std::vector<float> vec_data = {1.0f, 2.0f, 3.0f};
            autograd::Tensor vec = tensor_from_vector(vec_data, {3});
            autograd::Tensor result = autograd::add(scalar, vec);
            std::vector<float> expected_data = {6.0f, 7.0f, 8.0f};
            autograd::Tensor expected = tensor_from_vector(expected_data, {3});
            test_case("Scalar + tensor broadcasting", tensors_equal(result, expected));
            
            // Tensor + scalar
            autograd::Tensor result2 = autograd::add(vec, scalar);
            test_case("Tensor + scalar broadcasting", tensors_equal(result2, expected));
            
            // Broadcasting with singleton dimensions
            std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
            autograd::Tensor a = tensor_from_vector(a_data, {2, 2}); // [2,2]
            std::vector<float> b_data = {10.0f, 20.0f};
            autograd::Tensor b = tensor_from_vector(b_data, {2, 1}); // [2,1] -> should broadcast to [2,2]
            autograd::Tensor result3 = autograd::add(a, b);
            std::vector<float> expected3_data = {11.0f, 12.0f, 23.0f, 24.0f};
            autograd::Tensor expected3 = tensor_from_vector(expected3_data, {2, 2});
            test_case("Broadcasting with singleton dimensions", tensors_equal(result3, expected3));
        }
        
        // Test zero-sized tensors
        {
            autograd::Tensor zero1d = autograd::Tensor::zeros({0}, false);
            autograd::Tensor zero2d = autograd::Tensor::zeros({0, 5}, false);
            
            autograd::Tensor sum_zero1d = autograd::sum(zero1d, 0);
            test_case("Sum of zero-sized 1D tensor", sum_zero1d.numel() == 1);
            
            autograd::Tensor sum_zero2d = autograd::sum(zero2d, 1);
            test_case("Sum of zero-sized 2D tensor (dim=1)", sum_zero2d.numel() == 0);
            
            autograd::Tensor mean_zero1d = autograd::mean(zero1d, 0);
            // Mean of zero-sized tensor is NaN (not a number)
            bool is_nan = std::isnan(mean_zero1d({}));
            test_case("Mean of zero-sized 1D tensor", is_nan);
            
            // Note: softmax of zero-sized tensor should return zero-sized tensor
            autograd::Tensor softmax_zero = autograd::softmax(zero1d);
            test_case("Softmax of zero-sized tensor", softmax_zero.numel() == 0);
            
            autograd::Tensor logsoftmax_zero = autograd::log_softmax(zero1d);
            test_case("Log softmax of zero-sized tensor", logsoftmax_zero.numel() == 0);
        }
    }
    
    // Test 10: Mathematical properties and consistency
    std::cout << "\n--- Testing MATHEMATICAL PROPERTIES ---\n";
    {
        // Test that mean(x) = sum(x) / size for reductions
        std::vector<float> vec_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        autograd::Tensor vec = tensor_from_vector(vec_data, {5});
        autograd::Tensor sum_result = autograd::sum(vec, 0);
        autograd::Tensor mean_result = autograd::mean(vec, 0);
        float expected_mean = sum_result({}) / 5.0f;
        test_case("Mean = Sum / size (1D)", approx_equal(mean_result({}), expected_mean));
        
        // Test 2D case
        std::vector<float> mat_data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f
        };
        autograd::Tensor mat = tensor_from_vector(mat_data, {3, 3}); // [3,3]
        autograd::Tensor sum_dim0 = autograd::sum(mat, 0); // sum along rows -> [12,15,18]
        autograd::Tensor mean_dim0 = autograd::mean(mat, 0); // mean along rows -> [4,5,6]
        std::vector<float> expected_mean0_data = {
            12.0f/3.0f, 15.0f/3.0f, 18.0f/3.0f
        };
        autograd::Tensor expected_mean0 = tensor_from_vector(expected_mean0_data, {3});
        test_case("Mean = Sum / size (2D, dim=0)", tensors_equal(mean_dim0, expected_mean0));
        
        // Test softmax properties: output in [0,1] and sums to 1
        std::vector<float> logits_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        autograd::Tensor logits = tensor_from_vector(logits_data, {5});
        autograd::Tensor softmax_out = autograd::softmax(logits);
        bool in_range = true;
        float sum7 = 0.0f;
        const float* sm_data = softmax_out.data();
        for (size_t i = 0; i < 5; ++i) {
            if (sm_data[i] < 0.0f || sm_data[i] > 1.0f) {
                in_range = false;
                break;
            }
            sum7 += sm_data[i];
        }
        test_case("Softmax output in [0,1]", in_range);
        test_case("Softmax sums to 1", approx_equal(sum7, 1.0f, 1e-5f));
        
        // Test log_softmax properties: exp(log_softmax) = softmax
        autograd::Tensor log_softmax_out = autograd::log_softmax(logits);
        autograd::Tensor exp_log_softmax = autograd::exp(log_softmax_out);
        test_case("Exp(log_softmax) = softmax", tensors_equal(exp_log_softmax, softmax_out, 1e-5f));
        
        // Test that adding constant to logits doesn't change softmax (invariance property)
        std::vector<float> shifted_logits_data = {
            -2.0f + 5.0f, -1.0f + 5.0f, 0.0f + 5.0f, 1.0f + 5.0f, 2.0f + 5.0f
        };
        autograd::Tensor shifted_logits = tensor_from_vector(shifted_logits_data, {5}); // added 5.0 to each
        autograd::Tensor softmax_original = autograd::softmax(logits);
        autograd::Tensor softmax_shifted = autograd::softmax(shifted_logits);
        test_case("Softmax invariant to constant shift", tensors_equal(softmax_original, softmax_shifted, 1e-5f));
        
        // Same for log_softmax (should be invariant to constant shift)
        autograd::Tensor log_softmax_original = autograd::log_softmax(logits);
        autograd::Tensor log_softmax_shifted = autograd::log_softmax(shifted_logits);
        // log_softmax(x + c) = log_softmax(x)
        test_case("Log_softmax invariant to constant shift", tensors_equal(log_softmax_shifted, log_softmax_original, 1e-5f));
    }
    
    std::cout << "\n=== TEST SUMMARY ===\n";
    std::cout << "PASSED: " << passed << "\n";
    std::cout << "FAILED: " << failed << "\n";
    std::cout << "TOTAL:  " << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n🎉 ALL HARDCORE TESTS PASSED! 🎉\n";
        return 0;
    } else {
        std::cout << "\n❌ " << failed << " TESTS FAILED\n";
        return 1;
    }
} catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
} catch (...) {
    std::cerr << "Caught unknown exception" << std::endl;
    return 1;
}
}
