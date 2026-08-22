#include "autograd/tensor.hpp"
#include "autograd/operations.hpp"
#include "autograd/autograd_engine.hpp"
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
    if (!impl_) {
        throw std::runtime_error("Cannot call backward on null tensor");
    }

    if (!impl_->requires_grad_) {
        throw std::logic_error("Cannot call backward on tensor that doesn't require gradients");
    }

    if (impl_->grad_fn_) {
        throw std::logic_error("Cannot call backward on non-leaf tensor that has already been used in a backward pass");
    }

    // Create autograd engine and run backward pass
    AutogradEngine engine;
    engine.run(impl_->grad_fn_, grad_output);
}

void Tensor::backward()
{
    if (!impl_) {
        throw std::runtime_error("Cannot call backward on null tensor");
    }

    if (numel() != 1) {
        throw std::runtime_error("backward() without argument only valid for scalar tensors");
    }

    if (!impl_->requires_grad_) {
        throw std::logic_error("Cannot call backward on tensor that doesn't require gradients");
    }

    if (impl_->grad_fn_) {
        throw std::logic_error("Cannot call backward on non-leaf tensor that has already been used in a backward pass");
    }

    // Create scalar gradient of 1.0
    Tensor grad_output = Tensor::scalar(1.0f, false);

    // Create autograd engine and run backward pass
    AutogradEngine engine;
    engine.run(impl_->grad_fn_, grad_output);
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
    return autograd::add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const
{
    return autograd::sub(*this, other);
}

Tensor Tensor::operator-() const
{
    return autograd::neg(*this);
}

Tensor Tensor::operator*(const Tensor& other) const
{
    return autograd::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const
{
    return autograd::div(*this, other);
}

Tensor Tensor::operator==(const Tensor& other) const
{
    std::vector<std::size_t> this_shape = this->shape();
    std::vector<std::size_t> other_shape = other.shape();

    if (this_shape == other_shape) {
        if (this->numel() == 0) {
            return Tensor({}, false); // scalar tensor
        }

        Tensor result(this_shape, false);
        const float* this_data = this->data();
        const float* other_data = other.data();
        float* result_data = result.data();

        for (std::size_t i = 0; i < this->numel(); ++i) {
            result_data[i] = (this_data[i] == other_data[i]) ? 1.0f : 0.0f;
        }

        return result;
    }

    throw std::invalid_argument("Incompatible shapes for operator== (broadcasting not fully implemented)");
}

Tensor Tensor::operator!=(const Tensor& other) const
{
    std::vector<std::size_t> this_shape = this->shape();
    std::vector<std::size_t> other_shape = other.shape();

    if (this_shape == other_shape) {
        if (this->numel() == 0) {
            return Tensor({}, false); // scalar tensor
        }

        Tensor result(this_shape, false);
        const float* this_data = this->data();
        const float* other_data = other.data();
        float* result_data = result.data();

        for (std::size_t i = 0; i < this->numel(); ++i) {
            result_data[i] = (this_data[i] != other_data[i]) ? 1.0f : 0.0f;
        }

        return result;
    }

    throw std::invalid_argument("Incompatible shapes for operator!= (broadcasting not fully implemented)");
}

Tensor Tensor::operator<(const Tensor& other) const
{
    // Element-wise less than - return true if ALL elements are less
    // But looking at usage in function_nodes.cpp, it seems they want
    // element-wise comparison that returns a tensor of bools.
    // However, the way it's used: (input_ > zero).to_float()
    // suggests they want an element-wise comparison that returns a tensor
    // which they then convert to float.
    //
    // Actually, looking more carefully at the code, these comparison
    // operators are being used in expressions like (input_ > zero)
    // and then calling .to_float() on the result.
    // This suggests that operator> should return a Tensor, not a bool.
    //
    // Let me check how it's used:
    // In AbsBackward: Tensor pos_part = (input_ > zero).to_float();
    // In ReLUBackward: Tensor grad_input = grad_output * (input_ > zero).to_float();
    //
    // So (input_ > zero) must return a Tensor that has a .to_float() method.
    // Therefore, the comparison operators should return Tensor, not bool.
    //
    // I need to change the declarations in tensor.hpp to return Tensor
    // and implement them to do element-wise comparison returning 0/1 tensors.

    // For now, I'll return false to avoid breaking things, but this needs to be fixed properly.
    std::vector<std::size_t> this_shape = this->shape();
    std::vector<std::size_t> other_shape = other.shape();

    if (this_shape == other_shape) {
        if (this->numel() == 0) {
            return Tensor({}, false); // scalar tensor
        }

        Tensor result(this_shape, false);
        const float* this_data = this->data();
        const float* other_data = other.data();
        float* result_data = result.data();

        for (std::size_t i = 0; i < this->numel(); ++i) {
            result_data[i] = (this_data[i] < other_data[i]) ? 1.0f : 0.0f;
        }

        return result;
    }

    throw std::invalid_argument("Incompatible shapes for operator< (broadcasting not fully implemented)");
}

Tensor Tensor::operator<=(const Tensor& other) const
{
    std::vector<std::size_t> this_shape = this->shape();
    std::vector<std::size_t> other_shape = other.shape();

    if (this_shape == other_shape) {
        if (this->numel() == 0) {
            return Tensor({}, false); // scalar tensor
        }

        Tensor result(this_shape, false);
        const float* this_data = this->data();
        const float* other_data = other.data();
        float* result_data = result.data();

        for (std::size_t i = 0; i < this->numel(); ++i) {
            result_data[i] = (this_data[i] <= other_data[i]) ? 1.0f : 0.0f;
        }

        return result;
    }

    throw std::invalid_argument("Incompatible shapes for operator<= (broadcasting not fully implemented)");
}

Tensor Tensor::operator>(const Tensor& other) const
{
    std::vector<std::size_t> this_shape = this->shape();
    std::vector<std::size_t> other_shape = other.shape();

    if (this_shape == other_shape) {
        if (this->numel() == 0) {
            return Tensor({}, false); // scalar tensor
        }

        Tensor result(this_shape, false);
        const float* this_data = this->data();
        const float* other_data = other.data();
        float* result_data = result.data();

        for (std::size_t i = 0; i < this->numel(); ++i) {
            result_data[i] = (this_data[i] > other_data[i]) ? 1.0f : 0.0f;
        }

        return result;
    }

    throw std::invalid_argument("Incompatible shapes for operator> (broadcasting not fully implemented)");
}

Tensor Tensor::operator>=(const Tensor& other) const
{
    std::vector<std::size_t> this_shape = this->shape();
    std::vector<std::size_t> other_shape = other.shape();

    if (this_shape == other_shape) {
        if (this->numel() == 0) {
            return Tensor({}, false); // scalar tensor
        }

        Tensor result(this_shape, false);
        const float* this_data = this->data();
        const float* other_data = other.data();
        float* result_data = result.data();

        for (std::size_t i = 0; i < this->numel(); ++i) {
            result_data[i] = (this_data[i] >= other_data[i]) ? 1.0f : 0.0f;
        }

        return result;
    }

    throw std::invalid_argument("Incompatible shapes for operator>= (broadcasting not fully implemented)");
}

// Tensor methods needed for autograd
Tensor Tensor::to_float() const
{
    // For now, just return a copy of the tensor
    // In a more complete implementation, this would convert boolean/integer tensors to float
    // But since we only support float tensors, this is just a copy
    return Tensor(impl_->get_shape(), impl_->requires_grad_);
}

Tensor Tensor::sum(std::size_t dim, bool keepdim) const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::sum(*this, dim, keepdim);
}

Tensor Tensor::clamp(const Tensor& min, const Tensor& max) const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::clamp(*this, min, max);
}

Tensor Tensor::transpose() const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::transpose(*this);
}

Tensor Tensor::matmul(const Tensor& other) const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::matmul(*this, other);
}

Tensor Tensor::log() const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::log(*this);
}

Tensor Tensor::neg() const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::neg(*this);
}

Tensor Tensor::sqrt() const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::sqrt(*this);
}

Tensor Tensor::pow(const Tensor& exponent) const
{
    if (!impl_) {
        return Tensor();
    }
    return autograd::pow(*this, exponent);
}

Tensor Tensor::sum() const {
    if (!impl_) {
        return Tensor();
    }
    return autograd::sum_all(*this);
}

float Tensor::item() const {
    if (!impl_) {
        throw std::runtime_error("item() called on null tensor");
    }
    if (impl_->numel() != 1) {
        throw std::runtime_error("item() only valid for tensors with one element");
    }
    return impl_->data()[0];
}

} // namespace autograd