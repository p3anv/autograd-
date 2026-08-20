#pragma once

#include <memory>
#include <vector>
#include <initializer_list>
#include <stdexcept>

#include "autograd/tensor_impl.hpp"
#include "autograd/storage.hpp"

namespace autograd {

/**
 * @brief User-facing value-semantic handle to a tensor.
 */
class Tensor {
public:
    // Constructors
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    /** Create a scalar tensor */
    static Tensor scalar(float value, bool requires_grad = false);

    /** Create tensor from shape */
    explicit Tensor(std::vector<std::size_t> shape,
                    bool requires_grad = false);

    /** Create tensor filled with zeros */
    static Tensor zeros(std::vector<std::size_t> shape,
                        bool requires_grad = false);

    /** Create tensor filled with ones */
    static Tensor ones(std::vector<std::size_t> shape,
                       bool requires_grad = false);

    /** Create tensor with values [start, end) */
    static Tensor arange(std::size_t start, std::size_t end);

    // Accessors
    std::size_t numel() const;
    std::size_t dim() const;
    bool is_contiguous() const;
    std::vector<std::size_t> shape() const;

    float* data();
    const float* data() const;
    float& operator()(std::initializer_list<std::size_t> idx);

    // Autograd interface
    bool requires_grad() const;
    void set_requires_grad(bool val);
    Tensor detach() const;
    Tensor grad() const;

    void backward(const Tensor& grad_output);
    void backward();  // scalar tensors only

    // Views (zero-copy when possible)
    Tensor slice(std::size_t dim, std::size_t start, std::size_t end) const;
    Tensor narrow(std::size_t dim, std::size_t start, std::size_t length) const;
    Tensor reshape(std::vector<std::size_t> shape) const;
    Tensor contiguous() const;

    // Operators (to be implemented)
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    // Internal access only; not part of the public user API.
    std::shared_ptr<TensorImpl> impl() const noexcept { return impl_; }

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace autograd