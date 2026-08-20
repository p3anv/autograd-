#include "autograd/tensor_impl.hpp"
#include "autograd/tensor.hpp"
#include "autograd/internal.hpp"
#include <numeric>
#include <stdexcept>

namespace autograd {

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
                       std::vector<std::size_t> shape,
                       std::vector<std::ptrdiff_t> strides,
                       std::size_t offset)
    : storage_(std::move(storage))
    , shape_(std::move(shape))
    , strides_(std::move(strides))
    , offset_(offset)
{
    // Validate that shape and strides have same size
    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument("Shape and strides must have the same number of dimensions");
    }

    // Validate rank limit (max 8 dimensions)
    if (shape_.size() > 8) {
        throw std::invalid_argument("Tensor rank exceeds maximum of 8 dimensions");
    }

    // Validate that all shape dimensions are non-negative (size_t ensures this)
    // Validate that logical address of every element remains within storage
    if (!shape_.empty()) {
        std::size_t max_index = 0;
        bool zero_dimension = false;
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            // If any dimension is zero, the tensor has zero elements
            if (shape_[i] == 0) {
                zero_dimension = true;
                break;
            }
            // Since we assume non-negative strides, the maximum index is when each coordinate is at its max.
            max_index += (shape_[i] - 1) * static_cast<std::size_t>(strides_[i]);
        }

        if (!zero_dimension) {
            // Check that offset + max_index is within storage bounds (in elements)
            if (storage_) {
                if (offset_ + max_index > storage_->size()) {
                    throw std::invalid_argument("Tensor view exceeds storage bounds");
                }
            }
        }
    }
}

std::shared_ptr<TensorImpl>
TensorImpl::make_contiguous(std::vector<std::size_t> shape)
{
    // Check for overflow in shape product
    std::size_t total_elements = 1;
    for (std::size_t dim : shape) {
        if (dim != 0 && total_elements > std::size_t(-1) / dim) {
            throw std::invalid_argument("Tensor shape too large: overflow in element count");
        }
        total_elements *= dim;
    }

    auto storage = std::make_shared<Storage>(total_elements);
    std::vector<std::ptrdiff_t> strides;

    // Compute strides for row-major (C-style) layout
    if (!shape.empty()) {
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (std::size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * static_cast<std::ptrdiff_t>(shape[i]);
        }
    }

    return std::make_shared<TensorImpl>(storage, shape, strides, 0);
}

std::size_t TensorImpl::numel() const
{
    std::size_t total = 1;
    for (std::size_t dim : shape_) {
        if (dim != 0 && total > std::size_t(-1) / dim) {
            // Overflow protection - though constructor should have caught this
            return 0;
        }
        total *= dim;
    }
    return total;
}

std::size_t TensorImpl::dim() const
{
    return shape_.size();
}

bool TensorImpl::is_contiguous() const
{
    // Empty or scalar tensors are considered contiguous
    if (shape_.size() <= 1) {
        return true;
    }

    // Check if strides match row-major contiguous layout
    std::ptrdiff_t expected_stride = 1;
    for (std::size_t i = shape_.size(); i > 0; --i) {
        std::size_t idx = i - 1;
        if (strides_[idx] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<std::ptrdiff_t>(shape_[idx]);
    }
    return true;
}

float* TensorImpl::data()
{
    if (!is_contiguous()) {
        throw std::runtime_error("Cannot get data() pointer for non-contiguous tensor");
    }

    if (!storage_ || storage_->size() == 0) {
        return nullptr;
    }

    return storage_->data() + offset_;
}

const float* TensorImpl::data() const
{
    if (!is_contiguous()) {
        throw std::runtime_error("Cannot get data() pointer for non-contiguous tensor");
    }

    if (!storage_ || storage_->size() == 0) {
        return nullptr;
    }

    return storage_->data() + offset_;
}

void TensorImpl::accumulate_grad(const Tensor& grad)
{
    if (!requires_grad_) {
        throw std::logic_error(
            "Cannot accumulate a gradient into a tensor that does not require gradients");
    }

    if (grad.shape() != shape_) {
        throw std::invalid_argument(
            "Gradient shape must match tensor shape");
    }

    if (grad.requires_grad() || grad.impl()->grad_fn_) {
        throw std::logic_error(
            "Cannot accumulate a gradient that requires gradients");
    }

    if (!grad_impl_) {
        auto fresh = TensorImpl::make_contiguous(shape_);

        internal::copy_buffer(
            fresh->data(),
            grad.data(),
            numel());

        fresh->requires_grad_ = false;
        fresh->grad_fn_.reset();

        grad_impl_ = std::move(fresh);
        return;
    }

    internal::add_to_buffer(
        grad_impl_->data(),
        grad.data(),
        numel());
}

std::shared_ptr<TensorImpl>
TensorImpl::slice(std::size_t dim, std::size_t start, std::size_t end) const
{
    if (dim >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    if (start > end || end > shape_[dim]) {
        throw std::out_of_range("Invalid slice indices");
    }

    // Create new shape with the sliced dimension
    std::vector<std::size_t> new_shape = shape_;
    new_shape[dim] = end - start;

    // Strides remain the same
    std::vector<std::ptrdiff_t> new_strides = strides_;

    // Update offset: old offset + start * stride[dim]
    std::size_t new_offset = offset_ + start * static_cast<std::size_t>(strides_[dim]);

    return std::make_shared<TensorImpl>(storage_, new_shape, new_strides, new_offset);
}

std::shared_ptr<TensorImpl>
TensorImpl::narrow(std::size_t dim, std::size_t start, std::size_t length) const
{
    if (dim >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    if (start + length > shape_[dim]) {
        throw std::out_of_range("Narrow exceeds dimension bounds");
    }

    // Create new shape with the narrowed dimension
    std::vector<std::size_t> new_shape = shape_;
    new_shape[dim] = length;

    // Strides remain the same
    std::vector<std::ptrdiff_t> new_strides = strides_;

    // Update offset: old offset + start * stride[dim]
    std::size_t new_offset = offset_ + start * static_cast<std::size_t>(strides_[dim]);

    return std::make_shared<TensorImpl>(storage_, new_shape, new_strides, new_offset);
}

std::shared_ptr<TensorImpl>
TensorImpl::reshape(std::vector<std::size_t> new_shape) const
{
    // Check if the total number of elements matches
    if (numel() != 0) {
        std::size_t new_numel = 1;
        for (std::size_t dim : new_shape) {
            if (new_numel > std::size_t(-1) / dim) {
                throw std::invalid_argument("Tensor shape too large: overflow in element count");
            }
            new_numel *= dim;
        }

        if (new_numel != numel()) {
            throw std::invalid_argument("Incompatible shape for reshape");
        }
    } else {
        // Handle zero-sized tensor case
        for (std::size_t dim : new_shape) {
            if (dim != 0) {
                throw std::invalid_argument("Cannot reshape zero-sized tensor to non-zero shape");
            }
        }
    }

    // For zero-copy reshape, we need to check if the tensor is contiguous
    // According to the spec, reshape should be zero-copy if compatible, otherwise throw
    // A reshape is compatible without copying if the tensor is contiguous
    if (!is_contiguous()) {
        throw std::invalid_argument("reshape() requires contiguous tensor");
    }

    // Compute new strides for row-major layout
    std::vector<std::ptrdiff_t> new_strides;
    if (!new_shape.empty()) {
        new_strides.resize(new_shape.size());
        new_strides[new_shape.size() - 1] = 1;
        for (std::size_t i = new_shape.size() - 1; i > 0; --i) {
            new_strides[i - 1] = new_strides[i] * static_cast<std::ptrdiff_t>(new_shape[i]);
        }
    }

    return std::make_shared<TensorImpl>(storage_, new_shape, new_strides, offset_);
}

std::shared_ptr<TensorImpl>
TensorImpl::contiguous() const
{
    if (is_contiguous()) {
        // Already contiguous, return a new TensorImpl sharing the same storage
        return std::make_shared<TensorImpl>(storage_, shape_, strides_, offset_);
    }

    // Need to create a copy
    auto contiguous_storage = std::make_shared<Storage>(numel());
    std::vector<std::ptrdiff_t> strides;
    if (!shape_.empty()) {
        strides.resize(shape_.size());
        strides[shape_.size() - 1] = 1;
        for (std::size_t i = shape_.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * static_cast<std::ptrdiff_t>(shape_[i]);
        }
    }

    auto result = std::make_shared<TensorImpl>(contiguous_storage, shape_, strides, 0);

    // Copy the data
    if (numel() > 0) {
        float* dst_ptr = result->data();
        const float* src_ptr = storage_->data() + offset_;

        // If the tensor is contiguous, we can do a fast copy
        if (is_contiguous()) {
            internal::copy_buffer(dst_ptr, src_ptr, numel());
        } else {
            // For non-contiguous tensors, we need to copy element by element
            // using the proper indexing
            std::vector<std::size_t> indices(shape_.size(), 0);
            std::size_t linear_index = 0;

            while (linear_index < numel()) {
                // Compute source offset for current indices
                std::size_t src_offset = offset_;
                for (std::size_t i = 0; i < shape_.size(); ++i) {
                    src_offset += static_cast<std::size_t>(strides_[i]) * indices[i];
                }

                // Copy element
                dst_ptr[linear_index] = src_ptr[src_offset - offset_]; // src_ptr already includes offset_

                // Increment indices (like an odometer)
                for (std::size_t i = shape_.size(); i > 0; --i) {
                    std::size_t idx = i - 1;
                    if (++indices[idx] < shape_[idx]) {
                        break;
                    }
                    indices[idx] = 0;
                }

                ++linear_index;
            }
        }
    }

    return result;
}

} // namespace autograd