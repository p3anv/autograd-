#include "autograd/cpu_capabilities.hpp"
#include <array>
#include <cstdint>
#include <mutex>

namespace autograd {

// CPUID leaf/function codes
constexpr uint32_t CPUID_VENDOR_ID = 0x00000000;
constexpr uint32_t CPUID_FEATURES = 0x00000001;
constexpr uint32_t CPUID_EXT_FEATURES = 0x00000007;
constexpr uint32_t CPUID_EXTENDED = 0x80000000;
constexpr uint32_t CPUID_EXT_FEATURES_ECX = 0x80000001;

// Helper to invoke CPUID
#if defined(_MSC_VER)
    #include <intrin.h>
    static void cpuid(std::array<int, 4>& out, int eax, int ecx = 0) {
        __cpuidex(out.data(), eax, ecx);
    }
#else
    static void cpuid(std::array<int, 4>& out, int eax, int ecx = 0) {
        asm volatile(
            "cpuid"
            : "=a"(out[0]), "=b"(out[1]), "=c"(out[2]), "=d"(out[3])
            : "a"(eax), "c"(ecx)
        );
    }
#endif

// Check if OS supports XSAVE/XRSTOR (required for AVX)
static bool check_os_xsave() {
#if defined(_MSC_VER)
    unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 0x6) == 0x6; // XMM and YMM state enabled
#else
    uint32_t eax, edx;
    asm volatile(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(0)
    );
    return (eax & 0x6) == 0x6;
#endif
}

CpuCapabilities CpuCapabilities::detect() {
    CpuCapabilities caps;
    std::array<int, 4> regs;

    // Basic CPUID support check
    cpuid(regs, CPUID_VENDOR_ID);
    uint32_t max_basic = static_cast<uint32_t>(regs[0]);

    if (max_basic >= CPUID_FEATURES) {
        cpuid(regs, CPUID_FEATURES);
        // ECX (regs[2])
        caps.has_sse3     = (regs[2] >> 0)  & 1;
        caps.has_ssse3    = (regs[2] >> 9)  & 1;
        caps.has_sse41    = (regs[2] >> 19) & 1;
        caps.has_sse42    = (regs[2] >> 20) & 1;
        caps.has_popcnt   = (regs[2] >> 23) & 1;
        caps.has_avx      = (regs[2] >> 28) & 1;
        caps.has_fma      = (regs[2] >> 12) & 1;
        caps.has_xsave    = (regs[2] >> 26) & 1; // OS XSAVE support
        // EDX (regs[3])
        caps.has_sse      = (regs[3] >> 25) & 1;
        caps.has_sse2     = (regs[3] >> 26) & 1;
    }

    if (max_basic >= CPUID_EXT_FEATURES) {
        cpuid(regs, CPUID_EXT_FEATURES, 0);
        // EBX (regs[1])
        caps.has_bmi1     = (regs[1] >> 3)  & 1;
        caps.has_avx2     = (regs[1] >> 5)  & 1;
        caps.has_bmi2     = (regs[1] >> 8)  & 1;
        // ECX (regs[2])
        caps.has_lzcnt    = (regs[2] >> 5)  & 1; // Actually in extended features leaf 0x80000001
    }

    // Extended features (0x80000001)
    cpuid(regs, CPUID_EXTENDED);
    uint32_t max_extended = static_cast<uint32_t>(regs[0]);
    if (max_extended >= CPUID_EXT_FEATURES_ECX) {
        cpuid(regs, CPUID_EXT_FEATURES_ECX);
        caps.has_lzcnt = (regs[2] >> 5) & 1; // LZCNT in ECX bit 5
    }

    // AVX requires OS support for XSAVE/XRSTOR (YMM state)
    if (caps.has_avx && !check_os_xsave()) {
        caps.has_avx = false;
        caps.has_avx2 = false; // AVX2 implies AVX
        caps.has_fma = false;  // FMA implies AVX
    }

    return caps;
}

const CpuCapabilities& CpuCapabilities::get() {
    static const CpuCapabilities instance = detect();
    return instance;
}

} // namespace autograd