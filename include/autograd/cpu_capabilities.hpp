#pragma once

#include <cstdint>

namespace autograd {

/**
 * @brief CPU capability flags detected at runtime via CPUID.
 */
struct CpuCapabilities {
    bool has_sse = false;
    bool has_sse2 = false;
    bool has_sse3 = false;
    bool has_ssse3 = false;
    bool has_sse41 = false;
    bool has_sse42 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_fma = false;
    bool has_bmi1 = false;
    bool has_bmi2 = false;
    bool has_lzcnt = false;
    bool has_popcnt = false;
    bool has_xsave = false; // OS XSAVE support

    /**
     * @brief Get the singleton instance (detected on first call).
     * @return Reference to detected capabilities.
     */
    static const CpuCapabilities& get();

    /**
     * @brief Manually trigger detection (for testing).
     * @return Detected capabilities.
     */
    static CpuCapabilities detect();

    /**
     * @brief Check if AVX2 + FMA are available (minimum for fast path).
     * @return True if AVX2 and FMA are both supported.
     */
    bool has_avx2_fma() const noexcept {
        return has_avx2 && has_fma;
    }

    /**
     * @brief Check if any SIMD is available.
     * @return True if at least SSE2 is supported.
     */
    bool has_any_simd() const noexcept {
        return has_sse2;
    }
};

} // namespace autograd