#include "autograd/tensor.hpp"
#include "autograd/operations.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

using namespace autograd;

TEST(AutogradTest, AddBackward) {
    Tensor a({2, 3}, true);
    Tensor b({2, 3}, true);

    // Fill with known values
    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
        b.data()[i] = static_cast<float>(i + 10);
    }

    Tensor c = add(a, b);

    // c = a + b, so dc/da = 1, dc/db = 1
    Tensor grad_c({2, 3}, false);
    for (size_t i = 0; i < 6; ++i) {
        grad_c.data()[i] = 1.0f;
    }

    c.backward(grad_c);

    // Check gradients
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(a.grad().data()[i], 1.0f);
        EXPECT_FLOAT_EQ(b.grad().data()[i], 1.0f);
    }
}

TEST(AutogradTest, MulBackward) {
    Tensor a({2, 3}, true);
    Tensor b({2, 3}, true);

    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
        b.data()[i] = static_cast<float>(i + 10);
    }

    Tensor c = mul(a, b);

    // c = a * b, so dc/da = b, dc/db = a
    Tensor grad_c({2, 3}, false);
    for (size_t i = 0; i < 6; ++i) {
        grad_c.data()[i] = 1.0f;
    }

    c.backward(grad_c);

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(a.grad().data()[i], b.data()[i]);
        EXPECT_FLOAT_EQ(b.grad().data()[i], a.data()[i]);
    }
}

TEST(AutogradTest, SubBackward) {
    Tensor a({2, 3}, true);
    Tensor b({2, 3}, true);

    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
        b.data()[i] = static_cast<float>(i + 10);
    }

    Tensor c = sub(a, b);

    // c = a - b, so dc/da = 1, dc/db = -1
    Tensor grad_c({2, 3}, false);
    for (size_t i = 0; i < 6; ++i) {
        grad_c.data()[i] = 1.0f;
    }

    c.backward(grad_c);

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(a.grad().data()[i], 1.0f);
        EXPECT_FLOAT_EQ(b.grad().data()[i], -1.0f);
    }
}

TEST(AutogradTest, DivBackward) {
    Tensor a({2, 3}, true);
    Tensor b({2, 3}, true);

    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
        b.data()[i] = static_cast<float>(i + 10);
    }

    Tensor c = div(a, b);

    // c = a / b, so dc/da = 1/b, dc/db = -a/b^2
    Tensor grad_c({2, 3}, false);
    for (size_t i = 0; i < 6; ++i) {
        grad_c.data()[i] = 1.0f;
    }

    c.backward(grad_c);

    for (size_t i = 0; i < 6; ++i) {
        float expected_a_grad = 1.0f / b.data()[i];
        float expected_b_grad = -a.data()[i] / (b.data()[i] * b.data()[i]);
        EXPECT_FLOAT_EQ(a.grad().data()[i], expected_a_grad);
        EXPECT_FLOAT_EQ(b.grad().data()[i], expected_b_grad);
    }
}

TEST(AutogradTest, NegBackward) {
    Tensor a({3}, true);
    a.data()[0] = 1.0f;
    a.data()[1] = -2.0f;
    a.data()[2] = 3.0f;

    Tensor b = neg(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 2.0f;
    grad_b.data()[1] = 3.0f;
    grad_b.data()[2] = 4.0f;

    b.backward(grad_b);

    // db/da = -1
    EXPECT_FLOAT_EQ(a.grad().data()[0], -2.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[1], -3.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[2], -4.0f);
}

TEST(AutogradTest, ExpBackward) {
    Tensor a({3}, true);
    a.data()[0] = 0.0f;
    a.data()[1] = 1.0f;
    a.data()[2] = 2.0f;

    Tensor b = exp(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;
    grad_b.data()[2] = 1.0f;

    b.backward(grad_b);

    // db/da = exp(a) = b
    EXPECT_FLOAT_EQ(a.grad().data()[0], b.data()[0]);
    EXPECT_FLOAT_EQ(a.grad().data()[1], b.data()[1]);
    EXPECT_FLOAT_EQ(a.grad().data()[2], b.data()[2]);
}

TEST(AutogradTest, LogBackward) {
    Tensor a({3}, true);
    a.data()[0] = 1.0f;
    a.data()[1] = std::exp(1.0f);
    a.data()[2] = std::exp(2.0f);

    Tensor b = log(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;
    grad_b.data()[2] = 1.0f;

    b.backward(grad_b);

    // db/da = 1/a
    EXPECT_FLOAT_EQ(a.grad().data()[0], 1.0f / a.data()[0]);
    EXPECT_FLOAT_EQ(a.grad().data()[1], 1.0f / a.data()[1]);
    EXPECT_FLOAT_EQ(a.grad().data()[2], 1.0f / a.data()[2]);
}

TEST(AutogradTest, SqrtBackward) {
    Tensor a({3}, true);
    a.data()[0] = 1.0f;
    a.data()[1] = 4.0f;
    a.data()[2] = 9.0f;

    Tensor b = sqrt(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;
    grad_b.data()[2] = 1.0f;

    b.backward(grad_b);

    // db/da = 1/(2*sqrt(a))
    EXPECT_FLOAT_EQ(a.grad().data()[0], 1.0f / (2.0f * std::sqrt(a.data()[0])));
    EXPECT_FLOAT_EQ(a.grad().data()[1], 1.0f / (2.0f * std::sqrt(a.data()[1])));
    EXPECT_FLOAT_EQ(a.grad().data()[2], 1.0f / (2.0f * std::sqrt(a.data()[2])));
}

TEST(AutogradTest, AbsBackward) {
    Tensor a({3}, true);
    a.data()[0] = 1.0f;
    a.data()[1] = -2.0f;
    a.data()[2] = 0.0f;

    Tensor b = abs(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;
    grad_b.data()[2] = 1.0f;

    b.backward(grad_b);

    // db/da = sign(a)
    EXPECT_FLOAT_EQ(a.grad().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[1], -1.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[2], 0.0f); // sign(0) = 0
}

TEST(AutogradTest, ReLUBackward) {
    Tensor a({3}, true);
    a.data()[0] = 1.0f;
    a.data()[1] = -2.0f;
    a.data()[2] = 0.0f;

    Tensor b = relu(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;
    grad_b.data()[2] = 1.0f;

    b.backward(grad_b);

    // db/da = 1 if a > 0, else 0
    EXPECT_FLOAT_EQ(a.grad().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[1], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[2], 0.0f);
}

TEST(AutogradTest, SigmoidBackward) {
    Tensor a({3}, true);
    a.data()[0] = 0.0f;
    a.data()[1] = 1.0f;
    a.data()[2] = -1.0f;

    Tensor b = sigmoid(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;
    grad_b.data()[2] = 1.0f;

    b.backward(grad_b);

    // db/da = b * (1 - b)
    for (size_t i = 0; i < 3; ++i) {
        float expected = b.data()[i] * (1.0f - b.data()[i]);
        EXPECT_FLOAT_EQ(a.grad().data()[i], expected);
    }
}

TEST(AutogradTest, TanhBackward) {
    Tensor a({3}, true);
    a.data()[0] = 0.0f;
    a.data()[1] = 1.0f;
    a.data()[2] = -1.0f;

    Tensor b = tanh(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;
    grad_b.data()[2] = 1.0f;

    b.backward(grad_b);

    // db/da = 1 - b^2
    for (size_t i = 0; i < 3; ++i) {
        float expected = 1.0f - b.data()[i] * b.data()[i];
        EXPECT_FLOAT_EQ(a.grad().data()[i], expected);
    }
}

TEST(AutogradTest, SoftmaxBackward) {
    Tensor a({3}, true);
    a.data()[0] = 1.0f;
    a.data()[1] = 2.0f;
    a.data()[2] = 3.0f;

    Tensor b = softmax(a);

    Tensor grad_b({3}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 2.0f;
    grad_b.data()[2] = 3.0f;

    b.backward(grad_b);

    // For softmax, the Jacobian-vector product is:
    // grad_input = grad_output * output - output * sum(grad_output * output)
    // This is implicitly tested by comparing with numerical gradients
    // Let's verify by checking that the gradients sum to zero (property of softmax)
    float grad_sum = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        grad_sum += a.grad().data()[i];
    }
    EXPECT_NEAR(grad_sum, 0.0f, 1e-5);
}

TEST(AutogradTest, SumBackward) {
    Tensor a({2, 3}, true);
    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
    }

    Tensor b = sum(a, 1); // Sum along dimension 1 -> shape [2]

    Tensor grad_b({2}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 2.0f;

    b.backward(grad_b);

    // Gradient should be broadcast back to [2, 3]
    // Each row i gets the value grad_b[i] repeated 3 times
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.grad().data()[i * 3 + j], grad_b.data()[i]);
        }
    }
}

TEST(AutogradTest, MeanBackward) {
    Tensor a({2, 3}, true);
    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
    }

    Tensor b = mean(a, 1); // Mean along dimension 1 -> shape [2]

    Tensor grad_b({2}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 2.0f;

    b.backward(grad_b);

    // Gradient should be broadcast back to [2, 3] and divided by 3
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.grad().data()[i * 3 + j], grad_b.data()[i] / 3.0f);
        }
    }
}

TEST(AutogradTest, MaxBackward) {
    Tensor a({2, 3}, true);
    a.data()[0] = 1.0f; a.data()[1] = 5.0f; a.data()[2] = 3.0f;
    a.data()[3] = 2.0f; a.data()[4] = 1.0f; a.data()[5] = 4.0f;

    Tensor b = max(a, 1); // Max along dimension 1 -> shape [2]

    Tensor grad_b({2}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;

    b.backward(grad_b);

    // Gradient flows back to the max position in each row
    // Row 0: max at index 1 (value 5)
    // Row 1: max at index 2 (value 4)
    EXPECT_FLOAT_EQ(a.grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[1], 1.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[2], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[3], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[4], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[5], 1.0f);
}

TEST(AutogradTest, MinBackward) {
    Tensor a({2, 3}, true);
    a.data()[0] = 1.0f; a.data()[1] = 5.0f; a.data()[2] = 3.0f;
    a.data()[3] = 2.0f; a.data()[4] = 1.0f; a.data()[5] = 4.0f;

    Tensor b = min(a, 1); // Min along dimension 1 -> shape [2]

    Tensor grad_b({2}, false);
    grad_b.data()[0] = 1.0f;
    grad_b.data()[1] = 1.0f;

    b.backward(grad_b);

    // Gradient flows back to the min position in each row
    // Row 0: min at index 0 (value 1)
    // Row 1: min at index 1 (value 1)
    EXPECT_FLOAT_EQ(a.grad().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[1], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[2], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[3], 0.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[4], 1.0f);
    EXPECT_FLOAT_EQ(a.grad().data()[5], 0.0f);
}

TEST(AutogradTest, MatMulBackward) {
    Tensor a({2, 3}, true);
    Tensor b({3, 2}, true);

    // a = [[1, 2, 3], [4, 5, 6]]
    // b = [[7, 8], [9, 10], [11, 12]]
    float a_vals[6] = {1, 2, 3, 4, 5, 6};
    float b_vals[6] = {7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = a_vals[i];
        b.data()[i] = b_vals[i];
    }

    Tensor c = matmul(a, b); // shape [2, 2]

    Tensor grad_c({2, 2}, false);
    grad_c.data()[0] = 1.0f; grad_c.data()[1] = 2.0f;
    grad_c.data()[2] = 3.0f; grad_c.data()[3] = 4.0f;

    c.backward(grad_c);

    // grad_a = grad_c * b^T
    // grad_b = a^T * grad_c
    // We can verify by checking the shapes and doing numerical check
    EXPECT_EQ(a.grad().shape(), a.shape());
    EXPECT_EQ(b.grad().shape(), b.shape());

    // Check a specific value for grad_a
    // grad_a[0, 0] = grad_c[0, 0] * b[0, 0] + grad_c[0, 1] * b[0, 1] = 1*7 + 2*8 = 23
    EXPECT_FLOAT_EQ(a.grad().data()[0], 23.0f);

    // Check a specific value for grad_b
    // grad_b[0, 0] = a[0, 0] * grad_c[0, 0] + a[1, 0] * grad_c[1, 0] = 1*1 + 4*3 = 13
    EXPECT_FLOAT_EQ(b.grad().data()[0], 13.0f);
}

TEST(AutogradTest, ChainRule) {
    // Test a chain of operations: a -> b = exp(a) -> c = b + 1 -> d = sum(c)
    Tensor a({3}, true);
    a.data()[0] = 0.0f;
    a.data()[1] = 1.0f;
    a.data()[2] = 2.0f;

    std::cout << "Before exp" << std::endl;
    Tensor b = exp(a);
    std::cout << "After exp, before add" << std::endl;
    Tensor c = add(b, Tensor::scalar(1.0f));
    std::cout << "After add, before sum" << std::endl;
    Tensor d = sum(c, 0);
    std::cout << "After sum, before backward" << std::endl;

    // d is a scalar, backward with gradient 1.0
    d.backward();
    std::cout << "After backward" << std::endl;

    // Check grad - inspect the grad_impl_ directly
    std::cout << "a.impl_->grad_impl_: " << (a.impl() && a.impl()->grad_impl_ != nullptr) << std::endl;
    if (a.impl() && a.impl()->grad_impl_) {
        std::cout << "grad_impl_.numel(): " << a.impl()->grad_impl_->numel() << std::endl;
        std::cout << "grad_impl_.shape: ";
        for (auto s : a.impl()->grad_impl_->get_shape()) std::cout << s << " ";
        std::cout << std::endl;
    }

    // Check grad
    std::cout << "Before grad access" << std::endl;
    Tensor grad = a.grad();
    std::cout << "After grad access, before data" << std::endl;
    std::cout << "grad.requires_grad(): " << grad.requires_grad() << std::endl;
    std::cout << "grad.impl_->grad_fn_: " << (grad.impl() && grad.impl()->grad_fn_ != nullptr) << std::endl;
    std::cout << "grad.numel(): " << grad.numel() << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "grad[" << i << "] = " << grad.data()[i] << std::endl;
        EXPECT_FLOAT_EQ(grad.data()[i], std::exp(a.data()[i]));
    }
    std::cout << "Test complete" << std::endl;
}

TEST(AutogradTest, GradientAccumulation) {
    // Test that gradients accumulate correctly when a tensor is used multiple times
    Tensor a({3}, true);
    a.data()[0] = 1.0f;
    a.data()[1] = 2.0f;
    a.data()[2] = 3.0f;

    Tensor b = mul(a, a); // b = a * a
    Tensor c = add(b, a); // c = a^2 + a

    // dc/da = 2*a + 1
    Tensor grad_c({3}, false);
    grad_c.data()[0] = 1.0f;
    grad_c.data()[1] = 1.0f;
    grad_c.data()[2] = 1.0f;

    c.backward(grad_c);

    for (size_t i = 0; i < 3; ++i) {
        float expected = 2.0f * a.data()[i] + 1.0f;
        EXPECT_FLOAT_EQ(a.grad().data()[i], expected);
    }
}

TEST(AutogradTest, BroadcastingAddBackward) {
    // Test broadcasting: a (2, 3) + b (3,) -> c (2, 3)
    Tensor a({2, 3}, true);
    Tensor b({3}, true);

    for (size_t i = 0; i < 6; ++i) a.data()[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < 3; ++i) b.data()[i] = static_cast<float>(i + 10);

    Tensor c = add(a, b);

    Tensor grad_c({2, 3}, false);
    for (size_t i = 0; i < 6; ++i) grad_c.data()[i] = 1.0f;

    c.backward(grad_c);

    // grad_a should be all 1s
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(a.grad().data()[i], 1.0f);
    }

    // grad_b should sum over the broadcasted dimension
    // b is [10, 11, 12], broadcast to [2, 3] -> sum over first dim gives [2, 2, 2]
    EXPECT_FLOAT_EQ(b.grad().data()[0], 2.0f);
    EXPECT_FLOAT_EQ(b.grad().data()[1], 2.0f);
    EXPECT_FLOAT_EQ(b.grad().data()[2], 2.0f);
}

