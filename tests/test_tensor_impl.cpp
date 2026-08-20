#include "autograd/tensor_impl.hpp"
#include "autograd/tensor.hpp"
#include <gtest/gtest.h>

namespace autograd {

TEST(TensorImplTest, Construction) {
    auto storage = std::make_shared<Storage>(12);
    std::vector<std::size_t> shape = {3, 4};
    std::vector<std::ptrdiff_t> strides = {4, 1};
    TensorImpl impl(storage, shape, strides, 0);

    EXPECT_EQ(impl.numel(), 12);
    EXPECT_EQ(impl.dim(), 2);
    EXPECT_TRUE(impl.is_contiguous());
}

TEST(TensorImplTest, MakeContiguous) {
    auto impl = TensorImpl::make_contiguous({3, 4, 5});

    EXPECT_EQ(impl->numel(), 60);
    EXPECT_EQ(impl->dim(), 3);
    EXPECT_TRUE(impl->is_contiguous());

    // Check strides for row-major order
    auto strides = impl->get_strides();
    EXPECT_EQ(strides[0], 20); // 4*5
    EXPECT_EQ(strides[1], 5);  // 5
    EXPECT_EQ(strides[2], 1);

    // Check shape
    auto shape = impl->get_shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 5);

    // Check offset
    EXPECT_EQ(impl->get_offset(), 0);
}

TEST(TensorImplTest, ZeroSizedTensor) {
    auto storage = std::make_shared<Storage>(0);
    std::vector<std::size_t> shape = {0, 5};
    std::vector<std::ptrdiff_t> strides = {5, 1};
    TensorImpl impl(storage, shape, strides, 0);

    EXPECT_EQ(impl.numel(), 0);
    EXPECT_EQ(impl.dim(), 2);
    EXPECT_TRUE(impl.is_contiguous());
    EXPECT_EQ(impl.data(), nullptr);
}

TEST(TensorImplTest, ScalarTensor) {
    auto storage = std::make_shared<Storage>(1);
    std::vector<std::size_t> shape = {};
    std::vector<std::ptrdiff_t> strides = {};
    TensorImpl impl(storage, shape, strides, 0);

    EXPECT_EQ(impl.numel(), 1);
    EXPECT_EQ(impl.dim(), 0);
    EXPECT_TRUE(impl.is_contiguous());
}

TEST(TensorImplTest, NonContiguousView) {
    auto base = TensorImpl::make_contiguous({4, 6});
    // Create a view with different strides (still contiguous in this case, but different layout)
    auto view = std::make_shared<TensorImpl>(
        base->get_storage(),
        std::vector<std::size_t>{4, 6},
        std::vector<std::ptrdiff_t>{6, 1}, // Same as base actually
        0);

    // Actually let's create a truly non-contiguous view by changing strides
    // Use strides {2, 3} for shape {4, 6} - this should be valid and non-contiguous
    auto noncontiguous_view = std::make_shared<TensorImpl>(
        base->get_storage(),
        std::vector<std::size_t>{4, 6},
        std::vector<std::ptrdiff_t>{2, 3}, // Valid non-contiguous strides
        0);

    EXPECT_FALSE(noncontiguous_view->is_contiguous());
    // Non-contiguous tensors should not allow data() access
    EXPECT_THROW(noncontiguous_view->data(), std::runtime_error);
}

TEST(TensorImplTest, BoundsChecking) {
    auto storage = std::make_shared<Storage>(12);

    // Test shape that exceeds storage bounds
    EXPECT_THROW(
        TensorImpl(storage, {4, 4}, {4, 1}, 0), // 16 elements but storage only has 12
        std::invalid_argument);
}

TEST(TensorImplTest, RankLimit) {
    auto storage = std::make_shared<Storage>(100);

    // Test rank exceeding limit
    std::vector<std::size_t> shape(9, 1); // 9 dimensions
    std::vector<std::ptrdiff_t> strides(9, 1);
    EXPECT_THROW(
        TensorImpl(storage, shape, strides, 0),
        std::invalid_argument);
}

TEST(TensorImplTest, ShapeAndStridesSizeMismatch) {
    auto storage = std::make_shared<Storage>(12);
    EXPECT_THROW(
        TensorImpl(storage, {3, 4}, {4}, 0), // shape size 2, strides size 1
        std::invalid_argument);
}

TEST(TensorImplTest, SliceView) {
    auto base = TensorImpl::make_contiguous({4, 6});
    auto storage = base->get_storage();

    // Slice the first dimension: [0:2, :] -> shape {2, 6}
    auto sliced = base->slice(0, 0, 2);
    EXPECT_EQ(sliced->numel(), 12);
    EXPECT_EQ(sliced->dim(), 2);
    auto shape = sliced->get_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 6);
    // Strides should be the same as base (since we are slicing the first dimension, the stride for the second dimension remains 1, and for the first dimension it's 6)
    auto strides = sliced->get_strides();
    EXPECT_EQ(strides[0], 6); // 6 elements per row
    EXPECT_EQ(strides[1], 1);
    // Offset should be 0 (start at 0)
    EXPECT_EQ(sliced->get_offset(), 0);
    // Share the same storage
    EXPECT_EQ(sliced->get_storage(), storage);

    // Slice the second dimension: [:, 1:4] -> shape {4, 3}
    auto sliced2 = base->slice(1, 1, 4);
    EXPECT_EQ(sliced2->numel(), 12);
    EXPECT_EQ(sliced2->dim(), 2);
    shape = sliced2->get_shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 3);
    // Strides: first dimension stride is 6 (unchanged), second dimension stride is 1 (unchanged)
    strides = sliced2->get_strides();
    EXPECT_EQ(strides[0], 6);
    EXPECT_EQ(strides[1], 1);
    // Offset should be 1 * stride[1] = 1
    EXPECT_EQ(sliced2->get_offset(), 1);
    // Share the same storage
    EXPECT_EQ(sliced2->get_storage(), storage);
}

TEST(TensorImplTest, NarrowView) {
    auto base = TensorImpl::make_contiguous({4, 6});
    auto storage = base->get_storage();

    // Narrow the first dimension to 2 rows starting at index 1: [1:3, :] -> shape {2, 6}
    auto narrowed = base->narrow(0, 1, 2);
    EXPECT_EQ(narrowed->numel(), 12);
    EXPECT_EQ(narrowed->dim(), 2);
    auto shape = narrowed->get_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 6);
    // Strides should be the same as base
    auto strides = narrowed->get_strides();
    EXPECT_EQ(strides[0], 6);
    EXPECT_EQ(strides[1], 1);
    // Offset should be 1 * stride[0] = 6
    EXPECT_EQ(narrowed->get_offset(), 6);
    // Share the same storage
    EXPECT_EQ(narrowed->get_storage(), storage);

    // Narrow the second dimension to 3 columns starting at index 2: [:, 2:5] -> shape {4, 3}
    auto narrowed2 = base->narrow(1, 2, 3);
    EXPECT_EQ(narrowed2->numel(), 12);
    EXPECT_EQ(narrowed2->dim(), 2);
    shape = narrowed2->get_shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 3);
    // Strides should be the same as base
    strides = narrowed2->get_strides();
    EXPECT_EQ(strides[0], 6);
    EXPECT_EQ(strides[1], 1);
    // Offset should be 2 * stride[1] = 2
    EXPECT_EQ(narrowed2->get_offset(), 2);
    // Share the same storage
    EXPECT_EQ(narrowed2->get_storage(), storage);
}

TEST(TensorImplTest, ReshapeView) {
    auto base = TensorImpl::make_contiguous({2, 3, 4}); // 24 elements
    auto storage = base->get_storage();

    // Reshape to {4, 6} (same number of elements)
    auto reshaped = base->reshape({4, 6});
    EXPECT_EQ(reshaped->numel(), 24);
    EXPECT_EQ(reshaped->dim(), 2);
    auto shape = reshaped->get_shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 6);
    // Strides should be row-major for the new shape: [6, 1]
    auto strides = reshaped->get_strides();
    EXPECT_EQ(strides[0], 6);
    EXPECT_EQ(strides[1], 1);
    // Offset should remain 0
    EXPECT_EQ(reshaped->get_offset(), 0);
    // Share the same storage
    EXPECT_EQ(reshaped->get_storage(), storage);

    // Reshape to {24} (vector)
    auto reshaped2 = base->reshape({24});
    EXPECT_EQ(reshaped2->numel(), 24);
    EXPECT_EQ(reshaped2->dim(), 1);
    shape = reshaped2->get_shape();
    EXPECT_EQ(shape[0], 24);
    // Strides should be [1]
    strides = reshaped2->get_strides();
    EXPECT_EQ(strides[0], 1);
    // Offset should remain 0
    EXPECT_EQ(reshaped2->get_offset(), 0);
    // Share the same storage
    EXPECT_EQ(reshaped2->get_storage(), storage);

    // Reshape to {2, 2, 2, 3} (still 24)
    auto reshaped3 = base->reshape({2, 2, 2, 3});
    EXPECT_EQ(reshaped3->numel(), 24);
    EXPECT_EQ(reshaped3->dim(), 4);
    shape = reshaped3->get_shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 2);
    EXPECT_EQ(shape[3], 3);
    // Strides should be [12, 6, 3, 1] for row-major
    strides = reshaped3->get_strides();
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 6);
    EXPECT_EQ(strides[2], 3);
    EXPECT_EQ(strides[3], 1);
    // Offset should remain 0
    EXPECT_EQ(reshaped3->get_offset(), 0);
    // Share the same storage
    EXPECT_EQ(reshaped3->get_storage(), storage);
}

TEST(TensorImplTest, ReshapeIncompatibleShape) {
    auto base = TensorImpl::make_contiguous({2, 3, 4}); // 24 elements
    // Try to reshape to {2, 3, 5} (30 elements) -> should throw
    EXPECT_THROW(base->reshape({2, 3, 5}), std::invalid_argument);
    // Try to reshape to {2, 3} (6 elements) -> should throw
    EXPECT_THROW(base->reshape({2, 3}), std::invalid_argument);
}

TEST(TensorImplTest, ReshapeNonContiguousThrows) {
    // Create a non-contiguous tensor by transposing a matrix
    auto base = TensorImpl::make_contiguous({4, 6});
    // Create a view with column-major strides (valid, but non-contiguous)
    auto noncontiguous = std::make_shared<TensorImpl>(
        base->get_storage(),
        std::vector<std::size_t>{4, 6},
        std::vector<std::ptrdiff_t>{1, 4}, // column-major strides
        0);
    EXPECT_FALSE(noncontiguous->is_contiguous());
    // Trying to reshape a non-contiguous tensor should throw
    EXPECT_THROW(noncontiguous->reshape({2, 12}), std::invalid_argument);
}

TEST(TensorImplTest, ContiguousView) {
    auto base = TensorImpl::make_contiguous({3, 4});
    auto storage = base->get_storage();

    // Already contiguous, should return self
    auto contig = base->contiguous();
    EXPECT_TRUE(contig->is_contiguous());
    EXPECT_EQ(contig->get_storage(), storage);
    // Note: shared_from_this returns a shared_ptr that shares ownership with the original shared_ptr.
    // Since we are comparing the storage, it's the same.

    // Create a non-contiguous view (by transposing strides)
    auto noncontig = std::make_shared<TensorImpl>(
        base->get_storage(),
        std::vector<std::size_t>{3, 4},
        std::vector<std::ptrdiff_t>{1, 3}, // column-major strides
        0);
    EXPECT_FALSE(noncontig->is_contiguous());
    auto contig2 = noncontig->contiguous();
    EXPECT_TRUE(contig2->is_contiguous());
    // The contiguous tensor should have its own storage (a copy)
    EXPECT_NE(contig2->get_storage(), base->get_storage());
    // But the size should be the same
    EXPECT_EQ(contig2->get_storage()->size(), base->get_storage()->size());
    // And the data should be copied
    // We can check by setting data in the original and seeing if the contiguous copy changed?
    // But we don't have a way to set data directly. Instead, we can check that the offset is 0 and the shape is the same.
    EXPECT_EQ(contig2->get_offset(), 0);
    EXPECT_EQ(contig2->get_shape(), noncontig->get_shape());
}

} // namespace autograd