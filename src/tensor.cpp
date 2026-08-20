#include "autograd/tensor.hpp"
#include <algorithm>
#include <stdexcept>

namespace autograd {

// Constructors
// Tensor::Tensor() = default; // Already defaulted in header, remove definition to avoid warning

Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
    : impl_(std::move(impl))
{
}

Tensor Tensor::scalar(float value, bool requires_grad)
{
    auto result = zeros({});
    if (requires_grad) {
        result.set_requires_grad(true);
    }
    // Set the scalar value
    if (result.numel() == 1) {
        *result.data() = value;
    }
    return result;
}

Tensor::Tensor(std::vector<std::size_t> shape, bool requires_grad)
{
    impl_ = TensorImpl::make_contiguous(shape);
    if (requires_grad) {
        impl_->requires_grad_ = true;
    }
}

Tensor Tensor::zeros(std::vector<std::size_t> shape, bool requires_grad)
{
    Tensor result(shape, requires_grad);
    if (result.numel() > 0) {
        std::fill(result.data(), result.data() + result.numel(), 0.0f);
    }
    return result;
}

Tensor Tensor::ones(std::vector<std::size_t> shape, bool requires_grad)
{
    Tensor result(shape, requires_grad);
    if (result.numel() > 0) {
        std::fill(result.data(), result.data() + result.numel(), 1.0f);
    }
    return result;
}

Tensor Tensor::arange(std::size_t start, std::size_t end)
{
    if (end <= start) {
        return zeros({});
    }

    std::size_t size = end - start;
    Tensor result = zeros({size}, false);

    for (std::size_t i = 0; i < size; ++i) {
        result.data()[i] = static_cast<float>(start + i);
    }

    return result;
}

// Accessors
std::size_t Tensor::numel() const
{
    return impl_ ? impl_->numel() : 0;
}

std::size_t Tensor::dim() const
{
    return impl_ ? impl_->dim() : 0;
}

bool Tensor::is_contiguous() const
{
    return impl_ ? impl_->is_contiguous() : true;
}

std::vector<std::size_t> Tensor::shape() const
{
    return impl_ ? impl_->get_shape() : std::vector<std::size_t>{};
}

float* Tensor::data()
{
    return impl_ ? impl_->data() : nullptr;
}

const float* Tensor::data() const
{
    return impl_ ? impl_->data() : nullptr;
}

float& Tensor::operator()(std::initializer_list<std::size_t> idx)
{
    if (!impl_) {
        throw std::runtime_error("Cannot access data of null tensor");
    }

    // Convert initializer_list to vector for easier handling
    std::vector<std::size_t> indices(idx);

    if (indices.size() != impl_->get_shape().size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }

    // Calculate linear offset
    std::size_t offset = impl_->get_offset();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= impl_->get_shape()[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        offset += static_cast<std::size_t>(impl_->get_strides()[i]) * indices[i];
    }

    return impl_->get_storage()->data()[offset];
}

// Autograd interface
bool Tensor::requires_grad() const
{
    return impl_ ? impl_->requires_grad_ : false;
}

void Tensor::set_requires_grad(bool val)
{
    if (!impl_) {
        throw std::runtime_error("Cannot set requires_grad on null tensor");
    }

    if (impl_->grad_fn_ && !val) {
        throw std::logic_error("Cannot disable requires_grad on non-leaf tensor");
    }

    impl_->requires_grad_ = val;

    // If disabling gradients, clear gradient buffer
    if (!val && impl_->grad_impl_) {
        impl_->grad_impl_.reset();
    }
}

Tensor Tensor::detach() const
{
    if (!impl_) {
        return Tensor();
    }

    auto new_impl = std::make_shared<TensorImpl>(
        impl_->get_storage(),
        impl_->get_shape(),
        impl_->get_strides(),
        impl_->get_offset()
    );

    new_impl->requires_grad_ = false;
    new_impl->grad_fn_.reset();
    new_impl->grad_impl_.reset(); // No gradient buffer for detached tensor

    return Tensor(new_impl);
}

Tensor Tensor::grad() const
{
    if (!impl_ || !impl_->grad_impl_) {
        return Tensor(); // Return null tensor if no gradient
    }
    return Tensor(impl_->grad_impl_);
}

void Tensor::backward(const Tensor& grad_output)
{
    // To be implemented in Phase 4
    (void)grad_output; // Suppress unused parameter warning
    throw std::runtime_error("backward() not yet implemented");
}

void Tensor::backward()
{
    if (numel() != 1) {
        throw std::runtime_error("backward() without argument only valid for scalar tensors");
    }

    // To be implemented in Phase 4
    throw std::runtime_error("backward() not yet implemented");
}

// Views (zero-copy when possible)
Tensor Tensor::slice(std::size_t dim, std::size_t start, std::size_t end) const
{
    if (!impl_) {
        return Tensor();
    }
    return Tensor(impl_->slice(dim, start, end));
}

Tensor Tensor::narrow(std::size_t dim, std::size_t start, std::size_t length) const
{
    if (!impl_) {
        return Tensor();
    }
    return Tensor(impl_->narrow(dim, start, length));
}

Tensor Tensor::reshape(std::vector<std::size_t> shape) const
{
    if (!impl_) {
        return Tensor();
    }
    return Tensor(impl_->reshape(shape));
}

Tensor Tensor::contiguous() const
{
    if (!impl_) {
        return Tensor();
    }
    return Tensor(impl_->contiguous());
}

// Operators
Tensor Tensor::operator+(const Tensor& other) const
{
    // To be implemented in Phase 3
    (void)other; // Suppress unused parameter warning
    throw std::runtime_error("operator+ not yet implemented");
}

Tensor Tensor::operator*(const Tensor& other) const
{
    // To be implemented in Phase 3
    (void)other; // Suppress unused parameter warning
    throw std::runtime_error("operator* not yet implemented");
}

} // namespace autograd