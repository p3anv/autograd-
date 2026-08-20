#include "autograd/tensor.hpp"
#include "autograd/operations.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace autograd {

TEST(TensorTest, Constructors) {
    // Default constructor
    Tensor t1;
    EXPECT_EQ(t1.numel(), 0);
    EXPECT_EQ(t1.dim(), 0);

    // Constructor from shape
    Tensor t2({2, 3, 4}, false);
    EXPECT_EQ(t2.numel(), 24);
    EXPECT_EQ(t2.dim(), 3);
    EXPECT_FALSE(t2.requires_grad());

    Tensor t3({2, 3, 4}, true);
    EXPECT_EQ(t3.numel(), 24);
    EXPECT_EQ(t3.dim(), 3);
    EXPECT_TRUE(t3.requires_grad());
}

TEST(TensorTest, Factories) {
    // Scalar
    Tensor s = Tensor::scalar(3.14f, true);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_EQ(s.dim(), 0);
    EXPECT_TRUE(s.requires_grad());
    EXPECT_FLOAT_EQ(s({}), 3.14f);

    // Zeros
    Tensor z = Tensor::zeros({2, 3}, false);
    EXPECT_EQ(z.numel(), 6);
    EXPECT_EQ(z.dim(), 2);
    EXPECT_FALSE(z.requires_grad());
    for (size_t i = 0; i < z.numel(); ++i) {
        EXPECT_FLOAT_EQ(z.data()[i], 0.0f);
    }

    // Ones
    Tensor o = Tensor::ones({2, 3}, true);
    EXPECT_EQ(o.numel(), 6);
    EXPECT_EQ(o.dim(), 2);
    EXPECT_TRUE(o.requires_grad());
    for (size_t i = 0; i < o.numel(); ++i) {
        EXPECT_FLOAT_EQ(o.data()[i], 1.0f);
    }

    // Arange
    Tensor a = Tensor::arange(0, 5);
    EXPECT_EQ(a.numel(), 5);
    EXPECT_EQ(a.dim(), 1);
    EXPECT_FALSE(a.requires_grad());
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(a.data()[i], static_cast<float>(i));
    }
}

TEST(TensorTest, Accessors) {
    Tensor t = Tensor::zeros({2, 3, 4}, false);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.dim(), 3);
    EXPECT_TRUE(t.is_contiguous());

    auto shape = t.shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);

    // Test data access
    EXPECT_NE(t.data(), nullptr);
    EXPECT_EQ(t.data(), t.data()); // const and non-const should return same pointer for non-const tensor
}

TEST(TensorTest, ElementAccess) {
    Tensor t = Tensor::zeros({2, 3}, false);

    // Set and get values
    t({0, 0}) = 1.5f;
    t({1, 2}) = -2.3f;

    EXPECT_FLOAT_EQ(t({0, 0}), 1.5f);
    EXPECT_FLOAT_EQ(t({1, 2}), -2.3f);

    // Test bounds checking
    EXPECT_THROW(t({2, 0}), std::out_of_range); // row out of bounds
    EXPECT_THROW(t({0, 3}), std::out_of_range); // col out of bounds
    EXPECT_THROW(t({0, 0, 0}), std::invalid_argument); // too many indices
}

TEST(TensorTest, ZeroSizedTensor) {
    Tensor t = Tensor::zeros({0, 5}, false);
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.dim(), 2);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.data(), nullptr);
    // Should not be able to access elements
    EXPECT_THROW(t({0, 0}), std::out_of_range);
}

TEST(TensorTest, ScalarTensor) {
    Tensor s = Tensor::scalar(42.0f, false);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_EQ(s.dim(), 0);
    EXPECT_TRUE(s.is_contiguous());
    EXPECT_NE(s.data(), nullptr);
    EXPECT_FLOAT_EQ(s({}), 42.0f);

    // Test setting value
    s({}) = 3.14f;
    EXPECT_FLOAT_EQ(s({}), 3.14f);
}

TEST(TensorTest, Detach) {
    Tensor t = Tensor::zeros({2, 3}, true);
    EXPECT_TRUE(t.requires_grad());

    Tensor detached = t.detach();
    EXPECT_FALSE(detached.requires_grad());
    EXPECT_EQ(detached.numel(), 6);
    EXPECT_EQ(detached.dim(), 2);
    EXPECT_TRUE(detached.is_contiguous());

    // Original tensor should still require grad
    EXPECT_TRUE(t.requires_grad());

    // Detached tensor should not share grad_fn
    EXPECT_EQ(detached.impl()->grad_fn_, nullptr);
}

TEST(TensorTest, GradInitiallyNull) {
    Tensor t = Tensor::zeros({2, 3}, true);
    EXPECT_EQ(t.grad().numel(), 0); // No gradient initially
}

TEST(TensorTest, BackwardNotImplemented) {
    Tensor t = Tensor::zeros({2, 3}, true);
    Tensor grad_output = Tensor::ones({2, 3}, false);

    EXPECT_THROW(t.backward(grad_output), std::runtime_error);
    EXPECT_THROW(t.backward(), std::runtime_error); // scalar only
}

TEST(TensorTest, AddOperation) {
    // Test basic element-wise addition
    Tensor a = Tensor::zeros({2, 3}, false);
    Tensor b = Tensor::ones({2, 3}, false);

    // Set some values in a
    a({0, 0}) = 1.0f;
    a({1, 2}) = 2.0f;

    Tensor result = add(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values
    EXPECT_FLOAT_EQ(result({0, 0}), 2.0f); // 1.0 + 1.0
    EXPECT_FLOAT_EQ(result({0, 1}), 1.0f); // 0.0 + 1.0
    EXPECT_FLOAT_EQ(result({0, 2}), 1.0f); // 0.0 + 1.0
    EXPECT_FLOAT_EQ(result({1, 0}), 1.0f); // 0.0 + 1.0
    EXPECT_FLOAT_EQ(result({1, 1}), 1.0f); // 0.0 + 1.0
    EXPECT_FLOAT_EQ(result({1, 2}), 3.0f); // 2.0 + 1.0
}

TEST(TensorTest, AddBroadcasting) {
    // Test broadcasting: (2,3) + (3,) -> (2,3)
    Tensor a = Tensor::zeros({2, 3}, false);
    Tensor b = Tensor::zeros({3}, false);

    // Set values
    a({0, 0}) = 1.0f;
    a({1, 2}) = 2.0f;
    b({0}) = 0.5f;
    b({1}) = 1.5f;
    b({2}) = 2.5f;

    Tensor result = add(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: each row gets b added to it
    EXPECT_FLOAT_EQ(result({0, 0}), 1.5f); // 1.0 + 0.5
    EXPECT_FLOAT_EQ(result({0, 1}), 1.5f); // 0.0 + 1.5
    EXPECT_FLOAT_EQ(result({0, 2}), 2.5f); // 0.0 + 2.5
    EXPECT_FLOAT_EQ(result({1, 0}), 0.5f); // 0.0 + 0.5
    EXPECT_FLOAT_EQ(result({1, 1}), 1.5f); // 0.0 + 1.5
    EXPECT_FLOAT_EQ(result({1, 2}), 4.5f); // 2.0 + 2.5
}

TEST(TensorTest, AddScalarBroadcasting) {
    // Test scalar broadcasting
    Tensor a = Tensor::zeros({2, 2}, false);
    Tensor b = Tensor::scalar(3.0f, false);

    // Set values in a
    a({0, 0}) = 1.0f;
    a({1, 1}) = 2.0f;

    Tensor result = add(a, b);

    EXPECT_EQ(result.numel(), 4);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values
    EXPECT_FLOAT_EQ(result({0, 0}), 4.0f); // 1.0 + 3.0
    EXPECT_FLOAT_EQ(result({0, 1}), 3.0f); // 0.0 + 3.0
    EXPECT_FLOAT_EQ(result({1, 0}), 3.0f); // 0.0 + 3.0
    EXPECT_FLOAT_EQ(result({1, 1}), 5.0f); // 2.0 + 3.0
}

TEST(TensorTest, AddWithZerosOnes) {
    Tensor a = Tensor::ones({2, 3}, true);  // requires_grad = true
    Tensor b = Tensor::zeros({2, 3}, false);

    Tensor result = add(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_FALSE(result.requires_grad()); // Adding non-grad tensor should not require grad

    // Check values
    for (std::size_t i = 0; i < result.numel(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], 1.0f);
    }
}

TEST(TensorTest, SubOperation) {
    // Test basic element-wise subtraction
    Tensor a = Tensor::ones({2, 3}, false);  // All ones
    Tensor b = Tensor::zeros({2, 3}, false); // All zeros

    // Set some values in b
    b({0, 0}) = 2.0f;
    b({1, 2}) = 1.0f;

    Tensor result = sub(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: a - b
    EXPECT_FLOAT_EQ(result({0, 0}), -1.0f); // 1.0 - 2.0
    EXPECT_FLOAT_EQ(result({0, 1}), 1.0f);  // 1.0 - 0.0
    EXPECT_FLOAT_EQ(result({0, 2}), 1.0f);  // 1.0 - 0.0
    EXPECT_FLOAT_EQ(result({1, 0}), 1.0f);  // 1.0 - 0.0
    EXPECT_FLOAT_EQ(result({1, 1}), 1.0f);  // 1.0 - 0.0
    EXPECT_FLOAT_EQ(result({1, 2}), 0.0f);  // 1.0 - 1.0
}

TEST(TensorTest, SubBroadcasting) {
    // Test broadcasting: (2,3) - (3,) -> (2,3)
    Tensor a = Tensor::zeros({2, 3}, false);
    Tensor b = Tensor::zeros({3}, false);

    // Set values
    a({0, 0}) = 1.0f;
    a({1, 2}) = 2.0f;
    b({0}) = 0.5f;
    b({1}) = 1.5f;
    b({2}) = 2.5f;

    Tensor result = sub(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: each row gets b subtracted from it
    EXPECT_FLOAT_EQ(result({0, 0}), 0.5f);  // 1.0 - 0.5
    EXPECT_FLOAT_EQ(result({0, 1}), -1.5f); // 0.0 - 1.5
    EXPECT_FLOAT_EQ(result({0, 2}), -2.5f); // 0.0 - 2.5
    EXPECT_FLOAT_EQ(result({1, 0}), -0.5f); // 0.0 - 0.5
    EXPECT_FLOAT_EQ(result({1, 1}), -1.5f); // 0.0 - 1.5
    EXPECT_FLOAT_EQ(result({1, 2}), -0.5f); // 2.0 - 2.5
}

TEST(TensorTest, SubScalarBroadcasting) {
    // Test scalar broadcasting
    Tensor a = Tensor::zeros({2, 2}, false);
    Tensor b = Tensor::scalar(3.0f, false);

    // Set values in a
    a({0, 0}) = 5.0f;
    a({1, 1}) = 2.0f;

    Tensor result = sub(a, b);

    EXPECT_EQ(result.numel(), 4);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: a - 3.0
    EXPECT_FLOAT_EQ(result({0, 0}), 2.0f); // 5.0 - 3.0
    EXPECT_FLOAT_EQ(result({0, 1}), -3.0f); // 0.0 - 3.0
    EXPECT_FLOAT_EQ(result({1, 0}), -3.0f); // 0.0 - 3.0
    EXPECT_FLOAT_EQ(result({1, 1}), -1.0f); // 2.0 - 3.0
}

TEST(TensorTest, MulOperation) {
    // Test basic element-wise multiplication
    Tensor a = Tensor::ones({2, 3}, false);  // All ones
    Tensor b = Tensor::zeros({2, 3}, false); // All zeros

    // Set some values in b
    b({0, 0}) = 2.0f;
    b({1, 2}) = 3.0f;

    Tensor result = mul(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: a * b
    EXPECT_FLOAT_EQ(result({0, 0}), 2.0f); // 1.0 * 2.0
    EXPECT_FLOAT_EQ(result({0, 1}), 0.0f); // 1.0 * 0.0
    EXPECT_FLOAT_EQ(result({0, 2}), 0.0f); // 1.0 * 0.0
    EXPECT_FLOAT_EQ(result({1, 0}), 0.0f); // 1.0 * 0.0
    EXPECT_FLOAT_EQ(result({1, 1}), 0.0f); // 1.0 * 0.0
    EXPECT_FLOAT_EQ(result({1, 2}), 3.0f); // 1.0 * 3.0
}

TEST(TensorTest, MulBroadcasting) {
    // Test broadcasting: (2,3) * (3,) -> (2,3)
    Tensor a = Tensor::zeros({2, 3}, false);
    Tensor b = Tensor::zeros({3}, false);

    // Set values
    a({0, 0}) = 2.0f;
    a({1, 2}) = 3.0f;
    b({0}) = 0.5f;
    b({1}) = 2.0f;
    b({2}) = 4.0f;

    Tensor result = mul(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: each row gets multiplied by b
    EXPECT_FLOAT_EQ(result({0, 0}), 1.0f);  // 2.0 * 0.5
    EXPECT_FLOAT_EQ(result({0, 1}), 0.0f);  // 0.0 * 2.0
    EXPECT_FLOAT_EQ(result({0, 2}), 0.0f);  // 0.0 * 4.0
    EXPECT_FLOAT_EQ(result({1, 0}), 0.0f);  // 0.0 * 0.5
    EXPECT_FLOAT_EQ(result({1, 1}), 0.0f);  // 0.0 * 2.0
    EXPECT_FLOAT_EQ(result({1, 2}), 12.0f); // 3.0 * 4.0
}

TEST(TensorTest, MulScalarBroadcasting) {
    // Test scalar broadcasting
    Tensor a = Tensor::zeros({2, 2}, false);
    Tensor b = Tensor::scalar(3.0f, false);

    // Set values in a
    a({0, 0}) = 2.0f;
    a({1, 1}) = 4.0f;

    Tensor result = mul(a, b);

    EXPECT_EQ(result.numel(), 4);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: a * 3.0
    EXPECT_FLOAT_EQ(result({0, 0}), 6.0f); // 2.0 * 3.0
    EXPECT_FLOAT_EQ(result({0, 1}), 0.0f); // 0.0 * 3.0
    EXPECT_FLOAT_EQ(result({1, 0}), 0.0f); // 0.0 * 3.0
    EXPECT_FLOAT_EQ(result({1, 1}), 12.0f); // 4.0 * 3.0
}

TEST(TensorTest, MulWithZerosOnes) {
    Tensor a = Tensor::ones({2, 3}, true);  // requires_grad = true
    Tensor b = Tensor::zeros({2, 3}, false);

    Tensor result = mul(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_FALSE(result.requires_grad()); // Multiplying by non-grad tensor should not require grad

    // Check values
    for (std::size_t i = 0; i < result.numel(); ++i) {
        EXPECT_FLOAT_EQ(result.data()[i], 0.0f);
    }
}

TEST(TensorTest, DivOperation) {
    // Test basic element-wise division
    Tensor a = Tensor::zeros({2, 3}, false);
    Tensor b = Tensor::ones({2, 3}, false);

    // Set some values in a
    a({0, 0}) = 6.0f;
    a({1, 2}) = 4.0f;

    // Set some values in b (avoid division by zero)
    b({0, 0}) = 2.0f;
    b({1, 2}) = 2.0f;
    b({0, 1}) = 3.0f;
    b({1, 0}) = 1.0f;

    Tensor result = div(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: a / b
    EXPECT_FLOAT_EQ(result({0, 0}), 3.0f); // 6.0 / 2.0
    EXPECT_FLOAT_EQ(result({0, 1}), 0.0f); // 0.0 / 3.0
    EXPECT_FLOAT_EQ(result({0, 2}), 0.0f); // 0.0 / 1.0 (default)
    EXPECT_FLOAT_EQ(result({1, 0}), 0.0f); // 0.0 / 1.0
    EXPECT_FLOAT_EQ(result({1, 1}), 0.0f); // 0.0 / 1.0
    EXPECT_FLOAT_EQ(result({1, 2}), 2.0f); // 4.0 / 2.0
}

TEST(TensorTest, DivBroadcasting) {
    // Test broadcasting: (2,3) / (3,) -> (2,3)
    Tensor a = Tensor::zeros({2, 3}, false);
    Tensor b = Tensor::zeros({3}, false);

    // Set values in a
    a({0, 0}) = 6.0f;
    a({1, 2}) = 4.0f;

    // Set values in b (avoid division by zero)
    b({0}) = 2.0f;
    b({1}) = 2.0f;
    b({2}) = 4.0f;

    Tensor result = div(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: each row gets divided by b
    EXPECT_FLOAT_EQ(result({0, 0}), 3.0f);  // 6.0 / 2.0
    EXPECT_FLOAT_EQ(result({0, 1}), 0.0f);  // 0.0 / 2.0
    EXPECT_FLOAT_EQ(result({0, 2}), 0.0f);  // 0.0 / 4.0
    EXPECT_FLOAT_EQ(result({1, 0}), 0.0f);  // 0.0 / 2.0
    EXPECT_FLOAT_EQ(result({1, 1}), 0.0f);  // 0.0 / 2.0
    EXPECT_FLOAT_EQ(result({1, 2}), 1.0f);  // 4.0 / 4.0
}

TEST(TensorTest, DivScalarBroadcasting) {
    // Test scalar broadcasting
    Tensor a = Tensor::zeros({2, 2}, false);
    Tensor b = Tensor::scalar(2.0f, false);

    // Set values in a
    a({0, 0}) = 8.0f;
    a({1, 1}) = 4.0f;

    Tensor result = div(a, b);

    EXPECT_EQ(result.numel(), 4);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_TRUE(result.is_contiguous());

    // Check values: a / 2.0
    EXPECT_FLOAT_EQ(result({0, 0}), 4.0f); // 8.0 / 2.0
    EXPECT_FLOAT_EQ(result({0, 1}), 0.0f); // 0.0 / 2.0
    EXPECT_FLOAT_EQ(result({1, 0}), 0.0f); // 0.0 / 2.0
    EXPECT_FLOAT_EQ(result({1, 1}), 2.0f); // 4.0 / 2.0
}

TEST(TensorTest, DivWithZerosOnes) {
    Tensor a = Tensor::ones({2, 3}, true);  // requires_grad = true
    Tensor b = Tensor::zeros({2, 3}, false);

    Tensor result = div(a, b);

    EXPECT_EQ(result.numel(), 6);
    EXPECT_EQ(result.dim(), 2);
    EXPECT_FALSE(result.requires_grad()); // Dividing by non-grad tensor should not require grad

    // Check values (division by zero will produce inf)
    for (std::size_t i = 0; i < result.numel(); ++i) {
        EXPECT_TRUE(std::isinf(result.data()[i]));
    }
}

TEST(TensorTest, OperatorsNotImplemented) {
    Tensor a = Tensor::zeros({2, 2}, false);
    Tensor b = Tensor::zeros({2, 2}, false);

    EXPECT_THROW(a + b, std::runtime_error);
    EXPECT_THROW(a * b, std::runtime_error);
}

TEST(TensorTest, SliceView) {
    Tensor base = Tensor::zeros({4, 6}, false);
    // Slice the first dimension: [0:2, :] -> shape {2, 6}
    Tensor sliced = base.slice(0, 0, 2);
    EXPECT_EQ(sliced.numel(), 12);
    EXPECT_EQ(sliced.dim(), 2);
    auto shape = sliced.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 6);
    // Since the base is contiguous and we sliced the first dimension, the result should be contiguous
    EXPECT_TRUE(sliced.is_contiguous());
    // The underlying storage should be shared (zero-copy view)
    // We can check by comparing the data pointers (but note: offset may differ)
    // For simplicity, we trust the TensorImpl tests and just check that the tensor is valid.
    // Access an element to ensure it works
    sliced({0, 0}) = 1.0f;
    EXPECT_FLOAT_EQ(sliced({0, 0}), 1.0f);
    // The original base should be unchanged at that location because it's a different offset?
    // Actually, the slice [0:2, :] starts at offset 0, so the first element of the slice is the first element of the base.
    EXPECT_FLOAT_EQ(base({0, 0}), 1.0f);

    // Slice the second dimension: [:, 1:4] -> shape {4, 3}
    Tensor sliced2 = base.slice(1, 1, 4);
    EXPECT_EQ(sliced2.numel(), 12);
    EXPECT_EQ(sliced2.dim(), 2);
    shape = sliced2.shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 3);
    // After slicing a non-contiguous dimension, the result is not contiguous
    EXPECT_FALSE(sliced2.is_contiguous());
    sliced2({0, 0}) = 2.0f;
    EXPECT_FLOAT_EQ(sliced2({0, 0}), 2.0f);
    // This element corresponds to base({0, 1})
    EXPECT_FLOAT_EQ(base({0, 1}), 2.0f);
}

TEST(TensorTest, NarrowView) {
    Tensor base = Tensor::zeros({4, 6}, false);
    // Narrow the first dimension to 2 rows starting at index 1: [1:3, :] -> shape {2, 6}
    Tensor narrowed = base.narrow(0, 1, 2);
    EXPECT_EQ(narrowed.numel(), 12);
    EXPECT_EQ(narrowed.dim(), 2);
    auto shape = narrowed.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 6);
    EXPECT_TRUE(narrowed.is_contiguous());
    narrowed({0, 0}) = 3.0f;
    EXPECT_FLOAT_EQ(narrowed({0, 0}), 3.0f);
    // This element corresponds to base({1, 0})
    EXPECT_FLOAT_EQ(base({1, 0}), 3.0f);

    // Narrow the second dimension to 3 columns starting at index 2: [:, 2:5] -> shape {4, 3}
    Tensor narrowed2 = base.narrow(1, 2, 3);
    EXPECT_EQ(narrowed2.numel(), 12);
    EXPECT_EQ(narrowed2.dim(), 2);
    shape = narrowed2.shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 3);
    // After narrowing a non-contiguous dimension, the result is not contiguous
    EXPECT_FALSE(narrowed2.is_contiguous());
    narrowed2({0, 0}) = 4.0f;
    EXPECT_FLOAT_EQ(narrowed2({0, 0}), 4.0f);
    // This element corresponds to base({0, 2})
    EXPECT_FLOAT_EQ(base({0, 2}), 4.0f);
}

TEST(TensorTest, ReshapeView) {
    Tensor base = Tensor::zeros({2, 3, 4}, false); // 24 elements
    // Reshape to {4, 6}
    Tensor reshaped = base.reshape({4, 6});
    EXPECT_EQ(reshaped.numel(), 24);
    EXPECT_EQ(reshaped.dim(), 2);
    auto shape = reshaped.shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 6);
    EXPECT_TRUE(reshaped.is_contiguous());
    reshaped({0, 0}) = 5.0f;
    EXPECT_FLOAT_EQ(reshaped({0, 0}), 5.0f);
    // This element corresponds to base({0, 0, 0})
    EXPECT_FLOAT_EQ(base({0, 0, 0}), 5.0f);

    // Reshape to {24}
    Tensor reshaped2 = base.reshape({24});
    EXPECT_EQ(reshaped2.numel(), 24);
    EXPECT_EQ(reshaped2.dim(), 1);
    shape = reshaped2.shape();
    EXPECT_EQ(shape[0], 24);
    EXPECT_TRUE(reshaped2.is_contiguous());
    reshaped2({5}) = 6.0f;
    EXPECT_FLOAT_EQ(reshaped2({5}), 6.0f);
    // This element corresponds to base({0, 1, 1}) because 2*3*0 + 3*1 + 1 = 4? Let's compute:
    // shape {2,3,4}: index [i,j,k] -> i*(3*4) + j*4 + k
    // For i=0, j=1, k=1: 0*12 + 1*4 + 1 = 5 -> yes.
    EXPECT_FLOAT_EQ(base({0, 1, 1}), 6.0f);

    // Reshape to {2, 2, 2, 3}
    Tensor reshaped3 = base.reshape({2, 2, 2, 3});
    EXPECT_EQ(reshaped3.numel(), 24);
    EXPECT_EQ(reshaped3.dim(), 4);
    shape = reshaped3.shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 2);
    EXPECT_EQ(shape[3], 3);
    EXPECT_TRUE(reshaped3.is_contiguous());
    reshaped3({0, 0, 0, 0}) = 7.0f;
    EXPECT_FLOAT_EQ(reshaped3({0, 0, 0, 0}), 7.0f);
    // This element corresponds to base({0, 0, 0})
    EXPECT_FLOAT_EQ(base({0, 0, 0}), 7.0f);
}

TEST(TensorTest, ReshapeIncompatibleShape) {
    Tensor base = Tensor::zeros({2, 3, 4}, false);
    EXPECT_THROW(base.reshape({2, 3, 5}), std::invalid_argument);
    EXPECT_THROW(base.reshape({2, 3}), std::invalid_argument);
}

TEST(TensorTest, ReshapeNonContiguousThrows) {
    // Create a non-contiguous tensor by slicing in a way that makes it non-contiguous?
    // Actually, slicing a contiguous tensor with stride 1 in the sliced dimension remains contiguous.
    // We need to create a non-contiguous tensor first. Let's use narrow to change the stride?
    // Narrow also preserves contiguity if we narrow a dimension and keep the stride the same.
    // Instead, we can create a view by manually constructing a TensorImpl? But we don't have that in Tensor.
    // Alternatively, we can note that the Tensor::reshape method delegates to TensorImpl::reshape,
    // which throws if the tensor is not contiguous. We already tested that in TensorImplTest.
    // For Tensor, we can just test that calling reshape on a non-contiguous tensor throws.
    // How to make a non-contiguous Tensor? We can use slice with a step? But we don't have step.
    // Actually, our slice and narrow do not change the stride to non-contiguous values; they just adjust offset and shape.
    // So a tensor created by slice or narrow from a contiguous tensor is still contiguous if the sliced/narrowed dimension is stride-aligned?
    // Let's check:
    //   For a contiguous tensor with shape [d0, d1, ...] and strides [s0, s1, ...] where s0 = d1*d2*..., s1 = d2*d3*, etc.
    //   If we slice dimension i from start to end, the new shape[i] = end-start, and the stride remains the same.
    //   The tensor remains contiguous if the sliced dimension is the last dimension? Actually, no.
    //   Contiguity requires that the strides are exactly [d1*d2*..., d2*d3*..., ..., 1].
    //   After slicing a dimension (not the last), the stride for that dimension is still the old stride, which is the product of the lower dimensions.
    //   But the lower dimensions have changed? Actually, the lower dimensions (dimensions with higher index) are unchanged.
    //   Example: shape [4,6] -> strides [6,1]. Slice dimension 0: [2,6] -> strides [6,1].
    //   For shape [2,6], the expected strides for contiguous are [6,1] -> still contiguous.
    //   Slice dimension 1: [4,2] -> strides [6,1].
    //   For shape [4,2], expected strides are [2,1] -> but we have [6,1] -> not contiguous.
    //   So slicing a non-last dimension can produce a non-contiguous tensor.
    Tensor base = Tensor::zeros({4, 6}, false);
    Tensor sliced = base.slice(1, 0, 3); // shape [4,3], strides [6,1] -> expected strides for [4,3] are [3,1] -> not contiguous.
    EXPECT_FALSE(sliced.is_contiguous());
    EXPECT_THROW(sliced.reshape({2, 6}), std::invalid_argument);
}

TEST(TensorTest, ContiguousView) {
    Tensor base = Tensor::zeros({3, 4}, false);
    // Already contiguous, contiguous() should return a tensor that shares storage
    Tensor contig = base.contiguous();
    EXPECT_TRUE(contig.is_contiguous());
    // Check that the data is shared (we can check by modifying one and seeing the other change)
    contig({0, 0}) = 8.0f;
    EXPECT_FLOAT_EQ(base({0, 0}), 8.0f);
    // Reset
    base({0, 0}) = 0.0f;

    // Create a non-contiguous tensor by slicing the second dimension (as above)
    Tensor noncontig = base.slice(1, 0, 3); // shape [4,3], non-contiguous
    EXPECT_FALSE(noncontig.is_contiguous());
    Tensor contig2 = noncontig.contiguous();
    EXPECT_TRUE(contig2.is_contiguous());
    // The contiguous tensor should have its own storage (copy)
    // Modify the contiguous tensor and check that the original non-contiguous tensor is unchanged
    contig2({0, 0}) = 9.0f;
    EXPECT_FLOAT_EQ(contig2({0, 0}), 9.0f);
    EXPECT_FLOAT_EQ(noncontig({0, 0}), 0.0f); // Should be unchanged
    // Also, the shape should be the same
    EXPECT_EQ(contig2.shape(), noncontig.shape());
}

} // namespace autograd