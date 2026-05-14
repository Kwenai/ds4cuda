// fp8_kv.cuh — host-callable launcher for the DeepSeek-V4 KV-cache FP8
// (E4M3FN) round-trip CUDA kernel. Implementation: cuda/fp8_kv.cu.
//
// What this stage does
// --------------------
// After tail-RoPE produces a [n_heads, head_dim] KV tensor (KVrope), the
// DeepSeek-V4 reference path quantizes the **non-RoPE prefix** of every
// KV row through E4M3FN — a power-of-2 per-block fp8 scale, then a
// 7-bit-magnitude dequant table for the value itself — and writes the
// dequantized fp32 back to the same buffer. The RoPE tail (last
// `n_rot` dims) is copied through verbatim.
//
// This mirrors how the Metal kernel `kernel_dsv4_fp8_kv_quantize_f32`
// (ds4/metal/dsv4_kv.metal:79-127) and the CPU helper
// `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4/ds4.c:1606-1624) keep the
// runtime KV cache float-addressable while still matching FP8 cache
// semantics. The CPU dump stage `KVcur` (ds4/ds4.c:7214-7216) is
// produced by exactly this round-trip, so this kernel's output should
// match the dump element-wise.
//
// E4M3FN spec (cite: ds4/ds4.c:1561-1574, ds4/metal/dsv4_kv.metal:1-46):
//     index i in [0, 127]; exp = (i >> 3) & 0xF; mant = i & 0x7.
//     value(i) = (exp == 0) ? mant * 0.001953125
//                           : (1 + mant * 0.125) * exp_scale[exp]
//   where exp_scale[16] = {0, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4,
//                          8, 16, 32, 64, 128, 256}. value(126)=448.
//
// Per-block (64 elems) quantize formula
//   (cite: ds4/ds4.c:1606-1624 + dsv4_kv.metal:100-122):
//     amax  = max_{j in block} |x[j]|;  amax = max(amax, 1e-4)
//     scale = 2 ** ceil(log2(amax / 448))      // power-of-2
//     for j in block:
//        v       = clamp(x[j] / scale, -448, 448)
//        q_table = nearest(v) in E4M3FN table   (ties -> even index)
//        x[j]    = q_table * scale
//
// Reference: ds4 `kernel_dsv4_fp8_kv_quantize_f32` /
//            `kernel_dsv4_kv_fp8_store_f32`.

#ifndef DS4CUDA_FP8_KV_CUH
#define DS4CUDA_FP8_KV_CUH

#include <cuda_runtime.h>

namespace ds4cuda {

// Run E4M3FN round-trip on a [n_rows, head_dim] fp32 KV buffer. Each
// row's first `head_dim - n_rot` elements are quantized in 64-elem
// blocks (per-block fp8 scale + E4M3FN nearest), then dequantized back
// to fp32. The final `n_rot` elements are copied through unchanged.
//
//   kv_in / kv_out  device fp32, [n_rows * head_dim]; in-place safe
//                   (kernel reads then writes per-block under a single
//                   __syncthreads barrier).
//   n_rows          number of KV rows (=1 for the stage test;
//                   batch_kv during decode/prefill packs more).
//   head_dim        DS4_N_HEAD_DIM = 512.
//   n_rot           DS4_N_ROT = 64. Must be a multiple of 64.
//                   Caller's contract: head_dim - n_rot must also be a
//                   multiple of 64 (true for 512-64=448 = 7*64).
//   stream          CUDA stream (default 0).
void launch_fp8_kv_quantize_round_trip_f32(const float *kv_in,
                                           float       *kv_out,
                                           int          n_rows,
                                           int          head_dim,
                                           int          n_rot,
                                           cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_FP8_KV_CUH
