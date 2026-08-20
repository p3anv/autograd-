#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace autograd {

class Storage;
class FunctionNode;
class Tensor; // Forward declaration

/**
 * @brief Represents one logical tensor view and its autograd metadata.
 */
class TensorImpl {
public:
    /**
     * @brief Construct a new TensorImpl with given storage, shape, strides, and offset.
     * @param storage Shared pointer to the underlying storage
     * @param shape Tensor shape (dimensions)
     * @param strides Stride array (number of elements to skip in each dimension)
     * @param offset Offset into the storage (default 0)
     */
    TensorImpl(std::shared_ptr<Storage> storage,
               std::vector<std::size_t> shape,
               std::vector<std::ptrdiff_t> strides,
               std::size_t offset = 0);

    /**
     * @brief Factory method to create a contiguous tensor.
     * @param shape Tensor shape
     * @return Shared pointer to a new contiguous TensorImpl
     */
    static std::shared_ptr<TensorImpl>
    make_contiguous(std::vector<std::size_t> shape);

    /**
     * @brief Get the total number of elements.
     * @return Number of elements
     */
    std::size_t numel() const;

    /**
     * @brief Get the number of dimensions.
     * @return Number of dimensions
     */
    std::size_t dim() const;

    /**
     * @brief Check if the tensor is contiguous in memory.
     * @return true if contiguous, false otherwise
     */
    bool is_contiguous() const;

    /**
     * @brief Get pointer to the underlying data (only valid for contiguous tensors).
     * @return Pointer to float data
     * @throws std::runtime_error if tensor is not contiguous
     */
    float* data();
    const float* data() const;

    /**
     * @brief Get the underlying storage.
     * @return Shared pointer to storage
     */
    std::shared_ptr<Storage> get_storage() const { return storage_; }

    /**
     * @brief Get the shape of the tensor.
     * @return Shape vector
     */
    const std::vector<std::size_t>& get_shape() const { return shape_; }

    /**
     * @brief Get the strides of the tensor.
     * @return Strides vector
     */
    const std::vector<std::ptrdiff_t>& get_strides() const { return strides_; }

    /**
     * @brief Get the offset into the storage.
     * @return Offset
     */
    std::size_t get_offset() const { return offset_; }

    // Autograd state
    bool requires_grad_ = false;
    std::shared_ptr<FunctionNode> grad_fn_;

    // Gradient storage (independent tensor state)
    std::shared_ptr<TensorImpl> grad_impl_;

    /**
     * @brief Accumulate gradient into the gradient buffer.
     * @param grad Gradient tensor to accumulate
     * @requires requires_grad_ must be true
     */
    void accumulate_grad(const Tensor& grad);

    /**
     * @brief Create a zero-copy view slicing the specified dimension.
     * @param dim Dimension to slice
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @return Shared pointer to new TensorImpl sharing the same storage
     * @throws std::out_of_range if dim is out of range or indices are invalid
     */
    std::shared_ptr<TensorImpl> slice(std::size_t dim, std::size_t start, std::size_t end) const;

    /**
     * @brief Create a zero-copy view narrowing the specified dimension.
     * @param dim Dimension to narrow
     * @param start Start index (inclusive)
     * @param length Number of elements to include
     * @return Shared pointer to new TensorImpl sharing the same storage
     * @throws std::out_of_range if dim is out of range or indices are invalid
     */
    std::shared_ptr<TensorImpl> narrow(std::size_t dim, std::size_t start, std::size_t length) const;

    /**
     * @brief Create a zero-copy view reshaping the tensor.
     * @param new_shape New shape for the tensor
     * @return Shared pointer to new TensorImpl sharing the same storage if compatible
     * @throws std::invalid_argument if the new shape is incompatible with current size
     */
    std::shared_ptr<TensorImpl> reshape(std::vector<std::size_t> new_shape) const;

    /**
     * @brief Create a contiguous copy of the tensor.
     * @return Shared pointer to new contiguous TensorImpl with copied data
     */
    std::shared_ptr<TensorImpl> contiguous() const;

private:
    std::shared_ptr<Storage> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::ptrdiff_t> strides_;
    std::size_t offset_ = 0;
};

} // namespace autograd