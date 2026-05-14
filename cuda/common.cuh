// common.cuh — CUDA-side quant block layout + PTX helper stubs.
//
// The struct layouts below mirror include/ds4cuda.h verbatim so kernels
// can do reinterpret_cast<const block_q8_0*>(weight_ptr) on a mmap pointer.

#ifndef DS4CUDA_COMMON_CUH
#define DS4CUDA_COMMON_CUH

#include <cstdint>
#include <cuda_fp16.h>

namespace ds4cuda {

// Device-side IEEE fp16 -> fp32 conversion. Reinterprets a uint16_t
// (Q8_0 block scale) as __half and uses the hardware conversion
// intrinsic, which gives a true IEEE 754 binary16 -> binary32 cast.
// Bit-exact with the host CPU reference in ds4/ds4.c:1485 f16_to_f32
// for all finite & subnormal inputs (NaN payload bits are preserved on
// both sides).
__device__ __forceinline__ float fp16_bits_to_fp32(uint16_t h) {
    __half_raw r;
    r.x = h;
    return __half2float(__half(r));
}

constexpr int QK_K = 256;

// Q8_0 — 32 elems, 34 bytes.
struct __align__(2) block_q8_0 {
    uint16_t d;        // fp16 scale
    int8_t   qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size");

// Q2_K — 256 elems, 84 bytes.
struct __align__(2) block_q2_K {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
};
static_assert(sizeof(block_q2_K) == 84, "block_q2_K size");

// IQ2_XXS — 256 elems, 66 bytes.
struct __align__(2) block_iq2_xxs {
    uint16_t d;
    uint16_t qs[32];
};
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs size");

// Activation-only (not weight): Q8_K — 292 bytes.
struct block_q8_K {
    float    d;
    int8_t   qs[256];
    int16_t  bsums[16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K size");

// PTX dequant helpers go here.
// __device__ inline float dequant_q8_0(...) { ... }
// __device__ inline float dequant_q2_K(...) { ... }
// __device__ inline float dequant_iq2_xxs(...) { ... }

} // namespace ds4cuda

#endif // DS4CUDA_COMMON_CUH
