// fp8_kv.cu — DeepSeek-V4 KV-cache FP8 (E4M3FN) round-trip CUDA kernel.
//
// Stage coverage: `KVcur` (KV row after tail-RoPE, after FP8
// quantize+dequantize round-trip on the non-RoPE prefix). The RoPE
// tail (last `n_rot` dims) is copied through unchanged.
//
// ============================================================
// Spec (cited verbatim from ds4 CPU + Metal references)
// ============================================================
//
//   ds4/ds4.c:1561-1574 — E4M3FN value table:
//     static const float exp_scale[16] = {
//         0.0f, 0.015625f, 0.03125f, 0.0625f,
//         0.125f, 0.25f,  0.5f,   1.0f,
//         2.0f,  4.0f,    8.0f,   16.0f,
//         32.0f, 64.0f,   128.0f, 256.0f,
//     };
//     value(i)  =  (exp==0) ? mant * 0.001953125
//                           : (1 + mant * 0.125) * exp_scale[exp]
//        where exp = (i >> 3) & 0xF, mant = i & 0x7;
//     value(0)=0, value(1)=1/512, ..., value(126)=448.
//
//   ds4/ds4.c:1576-1601 — nearest-value picker (binary search +
//   bankers' tiebreak: when two adjacent table entries are equally
//   close, prefer the even index).
//
//   ds4/ds4.c:1606-1624 — per-row inplace round-trip:
//     n_nope = head_dim - n_rot
//     for off = 0; off < n_nope; off += 64:
//        amax  = max_{i in [off, off+64)} |x[i]|
//        amax  = max(amax, 1.0e-4)
//        scale = 2 ** ceil(log2(amax / 448))
//        for i in [off, off+64):
//           v        = clamp(x[i] / scale, -448, 448)
//           x[i]     = nearest_value(v) * scale
//
//   ds4/metal/dsv4_kv.metal:79-127 — Metal-side kernel; same formula,
//   one threadgroup per row, 64 threads, scratch[64] reduction in
//   shared memory.
//
// ============================================================
// Block layout
// ============================================================
// One CUDA block per KV row. Block has 64 threads — exactly two warps
// — so the per-block amax reduction folds neatly through butterfly
// shuffles plus one shared-memory crossover between the two warps.
// 7 quantized blocks (n_nope=448, block size=64) are walked
// sequentially within the same CUDA block; threads cooperate on each.
// The trailing n_rot elements are copied through with the same 64
// threads.
//
// Numerics:
//   - All math fp32 (ldexpf, log2f, ceilf — same intrinsics as the CPU
//     ref). We use exp2f(ceilf(log2f(amax/448))) which is one fma/log2
//     pair per block (= 7 calls total per row); the alternative
//     `ldexpf(1, (int)ceilf(...))` would need a host-side typed cast.
//     Both produce the same power-of-2 — the input amax/448 is always
//     positive and in [1e-4/448, ...] so log2f is well-defined.
//   - The dequant table is materialized once into constant memory.
//     Binary search [0, 126] picks the nearest index, with bankers'
//     tiebreak matching the CPU reference exactly.

#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"
#include "fp8_kv.cuh"

namespace ds4cuda {

namespace {

// E4M3FN per-exp scale table, in __constant__ memory. Cite:
// ds4/metal/dsv4_kv.metal:1-6 and ds4/ds4.c:1562-1567.
__constant__ float c_e4m3fn_exp_scale[16] = {
    0.0f,    0.015625f, 0.03125f, 0.0625f,
    0.125f,  0.25f,     0.5f,     1.0f,
    2.0f,    4.0f,      8.0f,     16.0f,
    32.0f,   64.0f,     128.0f,   256.0f,
};

constexpr int   FP8_BLOCK     = 64;     // elems per fp8 scale block
constexpr int   FP8_THREADS   = 64;     // threads per CUDA block
constexpr float FP8_E4M3_MAX  = 448.0f; // E4M3FN finite max
constexpr float FP8_AMAX_FLR  = 1.0e-4f;

// Reproduce dsv4_e4m3fn_value(i) on device. Cite: ds4.c:1561-1574.
__device__ __forceinline__ float e4m3fn_value(int i) {
    const int exp_  = (i >> 3) & 0xF;
    const int mant  = i & 0x7;
    return (exp_ == 0)
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * c_e4m3fn_exp_scale[exp_];
}

// Nearest-value picker matching dsv4_e4m3fn_dequant_cpu (ds4.c:1576-1601)
// + dsv4_kv.metal:49-74. Bankers' tiebreak: if two adjacent table
// entries are equally close, the CPU code rounds up only when the
// upper neighbor has an even index AND the lower has an odd index
// (i.e. (best+1)%2==0 && best%2==1). That parity test is reproduced
// exactly here.
__device__ __forceinline__ float e4m3fn_dequant(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float ax   = fminf(fabsf(x), FP8_E4M3_MAX);

    // Binary search the largest i in [0, 126] with value(i) <= ax.
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int   best       = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - e4m3fn_value(best));
        const float next_diff = fabsf(ax - e4m3fn_value(best + 1));
        const bool  prefer_up = (next_diff < best_diff)
                              || (next_diff == best_diff
                                  && (((best + 1) & 1) == 0)
                                  && ((best & 1) != 0));
        if (prefer_up) {
            best = best + 1;
        }
    }
    return sign * e4m3fn_value(best);
}

// Warp-level max reduction over 32 lanes via butterfly shuffle.
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

// One CUDA block = one KV row. 64 threads cooperate on each 64-elem
// fp8 quantize block; n_nope/64 such blocks are walked sequentially.
__global__ void fp8_kv_quantize_round_trip_f32_kernel(
        const float *__restrict__ kv_in,
        float       *__restrict__ kv_out,
        int                       head_dim,
        int                       n_rot)
{
    const int row     = blockIdx.x;
    const int tid     = (int)threadIdx.x;          // 0 .. 63
    const int n_nope  = head_dim - n_rot;          // 448 for default config
    const int row_off = row * head_dim;

    // 2-warp partial-max crossover. Each warp's lane-0 deposits its
    // max into shared[warp_id]; both warps then read both and take the
    // pairwise max. Single __syncthreads barrier per fp8 block.
    __shared__ float s_warp_max[2];

    // ---- 1) sequential walk over quantize blocks ----
    for (int off = 0; off < n_nope; off += FP8_BLOCK) {
        const float v = kv_in[row_off + off + tid];

        // amax over 64 elements: warp-internal butterfly, then 2-warp
        // crossover via shared memory. After the second warp_max,
        // every thread holds the same (block) amax.
        const int   warp_id = tid >> 5;
        const int   lane    = tid & 31;
        float       wmax    = warp_max(fabsf(v));
        if (lane == 0) {
            s_warp_max[warp_id] = wmax;
        }
        __syncthreads();
        const float amax_raw = fmaxf(s_warp_max[0], s_warp_max[1]);
        const float amax     = fmaxf(amax_raw, FP8_AMAX_FLR);
        const float scale    = exp2f(ceilf(log2f(amax / FP8_E4M3_MAX)));

        const float v_clamped = fminf(fmaxf(v / scale, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        const float q         = e4m3fn_dequant(v_clamped) * scale;
        kv_out[row_off + off + tid] = q;

        // Only one barrier needed before reusing s_warp_max in the
        // next iteration. Since every thread already wrote its own
        // output slot above, the shared array is the only race-prone
        // resource.
        __syncthreads();
    }

    // ---- 2) copy through the RoPE tail (last n_rot elems) ----
    // 64 threads + n_rot=64 -> exactly one element per thread.
    // Use a strided loop so the kernel still works if n_rot grows.
    for (int j = tid; j < n_rot; j += FP8_THREADS) {
        kv_out[row_off + n_nope + j] = kv_in[row_off + n_nope + j];
    }
}

} // namespace

void launch_fp8_kv_quantize_round_trip_f32(const float *kv_in,
                                           float       *kv_out,
                                           int          n_rows,
                                           int          head_dim,
                                           int          n_rot,
                                           cudaStream_t stream)
{
    // One block per KV row, 64 threads per block.
    const dim3 grid(n_rows);
    const dim3 block(FP8_THREADS);
    fp8_kv_quantize_round_trip_f32_kernel<<<grid, block, 0, stream>>>(
        kv_in, kv_out, head_dim, n_rot);
}

} // namespace ds4cuda
