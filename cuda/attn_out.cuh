// attn_out.cuh — host-callable launchers for attention output
// projection chain (kqv_back / attn_low / attn_out).
//
// Implementations: cuda/attn_out.cu.
//
// Stage chain (cite ds4/ds4.c:6968-6970, layer_grouped_out_one):
//
//     kqv_out  (heads after flash attention,        fp32 [N_HEAD*HEAD_DIM = 32768])
//        | rope_tail_layer_inplace(..., inverse=true)
//     kqv_back (heads after inverse-tail RoPE,      fp32 [32768])
//        | matvec_q8_0_grouped_rows(attn_output_a)
//     attn_low (group-block-diagonal projection,    fp32 [N_OUT_GROUP*N_LORA_O = 8192])
//        | matvec_q8_0(attn_output_b)
//     attn_out (full LoRA-output projection,        fp32 [N_EMBD = 4096])
//
// The intermediate `attn_low` kernel is the only piece that needs new
// CUDA code — `attn_out` reuses launch_mul_mv_q8_0_q8_0_f32 from
// dense_q8.cu directly. We expose a separate inverse-RoPE entry point
// here (rather than extending rope.cu) so the flash-attention work in
// rope.cu can stay pinned.
//
// References:
//   - ds4/ds4.c:3516-3552  matvec_q8_0_grouped_rows (CPU reference)
//   - ds4/ds4.c:3166-3179  matvec_q8_0_grouped_worker (per-output dot)
//   - ds4/ds4.c:4665-4713  rope_tail_ext_inplace (sin_sign=-1 for inverse)
//   - ds4/metal/moe.metal:842 kernel_dsv4_attn_out_low_q8_0_f32 (Metal ref)

#ifndef DS4CUDA_ATTN_OUT_CUH
#define DS4CUDA_ATTN_OUT_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"

namespace ds4cuda {

// Inverse tail-RoPE YaRN. Same per-pair math as the forward kernel in
// rope.cu but with sin_sign = -1.0f (cite ds4/ds4.c:4681,
// `sin_sign = inverse ? -1.0f : 1.0f`). Used by the kqv_back stage —
// undoes the rotation that was applied to the queries before flash
// attention so the heads return to a Y-axis frame for the grouped
// output projection.
//
//   x_in / x_out  device fp32, [n_heads * head_dim]. In-place safe.
//   n_heads       N_HEAD = 64.
//   head_dim      DS4_N_HEAD_DIM = 512.
//   n_rot         DS4_N_ROT = 64.
//   pos           Token position (matches the forward call earlier).
//   il            Layer index (drives compress_ratio routing).
//   stream        CUDA stream (default 0).
void launch_tail_rope_yarn_inverse_f32(const float *x_in,
                                       float       *x_out,
                                       int          n_heads,
                                       int          head_dim,
                                       int          n_rot,
                                       int          pos,
                                       int          il,
                                       cudaStream_t stream = 0);

// Q8_0 × Q8_0 → fp32 grouped matvec.
//
// Semantics (cite ds4/ds4.c:3166-3179):
//
//     for r in [0, n_groups * rank):
//         g = r / rank
//         y[r] = dot(W_row[r,  : group_dim],
//                    x_quant[g * group_dim : (g+1) * group_dim])
//
// The activation is split into n_groups slices of `group_dim` fp32
// elements. Each slice is independently quantized to (int8 qs, fp32
// scale) using ds4.c:3094 quantize_q8_0_activation semantics, then
// the per-row dot uses the matching slice's quantized data.
//
// Preconditions:
//   - W_dev is (n_groups * rank) * (group_dim/32) on-disk block_q8_0
//     records, row-major. (Same flat layout as the regular [in, out]
//     Q8_0 weight; only the per-row indexing into x changes.)
//   - x_fp32_dev is (n_groups * group_dim) fp32.
//   - group_dim % 32 == 0.
//   - y_dev is (n_groups * rank) fp32.
//
// The launcher uses stream-ordered cudaMallocAsync/cudaFreeAsync for
// quantized activation scratch and does not synchronize internally.
// Callers must keep inputs/outputs valid until dependent stream work is done.
void launch_mul_mv_q8_0_q8_0_grouped_f32(const block_q8_0 *W_dev,
                                         const float      *x_fp32_dev,
                                         float            *y_dev,
                                         int               n_groups,
                                         int               group_dim,
                                         int               rank,
                                         cudaStream_t      stream = 0);

// Same grouped Q8_0 matvec contract as above, with caller-owned scratch:
//   scratch_xq_dev     : int8  [n_groups * group_dim]
//   scratch_xscale_dev : float [n_groups * group_dim / 32]
// Performs no device allocation/free and does not synchronize.
void launch_mul_mv_q8_0_q8_0_grouped_f32_prealloc(
                                         const block_q8_0 *W_dev,
                                         const float      *x_fp32_dev,
                                         float            *y_dev,
                                         int               n_groups,
                                         int               group_dim,
                                         int               rank,
                                         int8_t           *scratch_xq_dev,
                                         float            *scratch_xscale_dev,
                                         cudaStream_t      stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_ATTN_OUT_CUH
