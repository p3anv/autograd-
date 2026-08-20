#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <limits>

namespace autograd {

class Storage {
public:
    explicit Storage(std::size_t size);

    float* data() noexcept;
    const float* data() const noexcept;
    std::size_t size() const noexcept;

private:
    struct AlignedDeleter {
        void operator()(float* ptr) const noexcept {
            std::free(ptr);
        }
    };

    std::unique_ptr<float[], AlignedDeleter> data_;
    std::size_t size_ = 0;
};

} // namespace autograd