#pragma once

#include <cstddef>
#include <algorithm>

namespace autograd {
namespace internal {

/**
 * @brief Add source buffer to destination buffer element-wise.
 * @param dst Destination buffer
 * @param src Source buffer
 * @param n Number of elements
 */
inline void add_to_buffer(float* dst,
                          const float* src,
                          std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

/**
 * @brief Copy source buffer to destination buffer.
 * @param dst Destination buffer
 * @param src Source buffer
 * @param n Number of elements
 */
inline void copy_buffer(float* dst,
                        const float* src,
                        std::size_t n) {
    std::copy_n(src, n, dst);
}

} // namespace internal
} // namespace autograd