#include "autograd/storage.hpp"
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace autograd {

Storage::Storage(std::size_t size) {
    // Handle zero-sized tensors specially - no allocation needed
    if (size == 0) {
        data_ = nullptr;
        size_ = 0;
        return;
    }

    // Check for overflow in size * sizeof(float)
    if (size > std::size_t(-1) / sizeof(float)) {
        throw std::invalid_argument("Storage size too large: overflow in size * sizeof(float)");
    }

    const std::size_t bytes = size * sizeof(float);

    // Check for overflow in bytes + alignment padding
    if (bytes > std::size_t(-1) - 63) {
        throw std::invalid_argument("Storage size too large: overflow in bytes + alignment padding");
    }

    // Calculate aligned bytes (64-byte alignment)
    const std::size_t aligned_bytes = ((bytes + 63) / 64) * 64;

    // Allocate aligned memory
    data_.reset(static_cast<float*>(std::aligned_alloc(64, aligned_bytes)));
    if (!data_ && bytes != 0) {
        throw std::bad_alloc();
    }

    size_ = size;
}

float* Storage::data() noexcept {
    return data_.get();
}

const float* Storage::data() const noexcept {
    return data_.get();
}

std::size_t Storage::size() const noexcept {
    return size_;
}

} // namespace autograd