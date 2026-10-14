#pragma once

#include "autograd/cpu_capabilities.hpp"
#include <utility>
#include <variant>

namespace autograd {
namespace simd {

// Tag types for compile-time dispatch
struct ScalarTag { constexpr explicit ScalarTag() = default; };
struct SseTag { constexpr explicit SseTag() = default; };
struct Avx2Tag { constexpr explicit Avx2Tag() = default; };

// Compile-time best tag available
#if defined(__AVX2__) && defined(__FMA__)
using BestTag = Avx2Tag;
#elif defined(__SSE4_2__)
using BestTag = SseTag;
#else
using BestTag = ScalarTag;
#endif

/**
 * @brief Runtime dispatch to the best available SIMD implementation.
 *
 * @tparam Op Callable accepting (Tag, Args...) where Tag is ScalarTag/SseTag/Avx2Tag
 * @tparam Args Arguments to forward to the callable
 * @param op Callable that implements the operation for each SIMD level
 * @param args Arguments to forward
 * @return Result of the callable
 *
 * Usage:
 *   simd::dispatch([&](auto tag) {
 *       if constexpr (std::is_same_v<decltype(tag), Avx2Tag>) {
 *           return avx2_kernel(args...);
 *       } else if constexpr (std::is_same_v<decltype(tag), SseTag>) {
 *           return sse_kernel(args...);
 *       } else {
 *           return scalar_kernel(args...);
 *       }
 *   });
 */
template <typename Op, typename... Args>
decltype(auto) dispatch(Op&& op, Args&&... args) {
    const auto& caps = CpuCapabilities::get();

    if (caps.has_avx2_fma()) {
        return std::forward<Op>(op)(Avx2Tag{}, std::forward<Args>(args)...);
    }
    if (caps.has_sse42) {
        return std::forward<Op>(op)(SseTag{}, std::forward<Args>(args)...);
    }
    return std::forward<Op>(op)(ScalarTag{}, std::forward<Args>(args)...);
}

/**
 * @brief Compile-time dispatch (for templates where runtime dispatch isn't needed).
 *
 * @tparam Tag One of ScalarTag, SseTag, Avx2Tag
 * @tparam Op Callable
 * @tparam Args Arguments
 * @param op Callable
 * @param args Arguments
 * @return Result of op(Tag{}, args...)
 */
template <typename Tag, typename Op, typename... Args>
decltype(auto) static_dispatch(Op&& op, Args&&... args) {
    return std::forward<Op>(op)(Tag{}, std::forward<Args>(args)...);
}

/**
 * @brief Check if AVX2+FMA is available at runtime.
 */
inline bool has_avx2_fma() noexcept {
    return CpuCapabilities::get().has_avx2_fma();
}

/**
 * @brief Check if SSE4.2 is available at runtime.
 */
inline bool has_sse42() noexcept {
    return CpuCapabilities::get().has_sse42;
}

// Type-erased tag for runtime use
using AnyTag = std::variant<ScalarTag, SseTag, Avx2Tag>;

/**
 * @brief Get the best available SIMD tag at runtime (type-erased).
 */
inline AnyTag get_best_tag() {
    const auto& caps = CpuCapabilities::get();
    if (caps.has_avx2_fma()) return Avx2Tag{};
    if (caps.has_sse42) return SseTag{};
    return ScalarTag{};
}

} // namespace simd
} // namespace autograd