// rope.cuh — host-callable launcher for the DeepSeek-V4 tail-RoPE YaRN
// CUDA kernel. Implementation: cuda/rope.cu.
//
// Computes per-head, per dim-pair (i, i+1) within the last n_rot dims of
// each head:
//
//     theta_extrap = (float)pos * (freq_base ** (-2.0f * (i/2) / n_rot))
//     YaRN-scaled  -> cos/sin via rope_yarn(...) (see rope.cu for cite)
//     y[i  ]       = x[i  ] * cos - x[i+1] * sin
//     y[i+1]       = x[i  ] * sin + x[i+1] * cos
//
// The first (head_dim - n_rot) dims of every head are passed through
// untouched. Single kernel covers both Qcur stage (n_heads=64)
// and the KVrope stage (n_heads=1) because the inner per-head math is
// identical — see ds4.c:6911-6912 where rope_tail_layer_inplace is
// called with the same (head_dim, n_rot, pos, il) for both buffers,
// only n_head differs.
//
// All RoPE constants (freq_base, scale_factor, orig_ctx, beta_fast,
// beta_slow) and the per-layer routing (compress_ratio -> freq_base /
// freq_scale / ext_factor / attn_factor) are baked into rope.cu —
// callers only pass the layer index `il` and the position `pos`.
//
// Reference: ds4/ds4.c:4646-4760 (YaRN ramp + corr_dims + rope_tail_layer_inplace)
// and ds4/metal/dsv4_rope.metal:27-49 (Metal-side YaRN helper, used to
// double-check formula).

#ifndef DS4CUDA_ROPE_CUH
#define DS4CUDA_ROPE_CUH

#include <cuda_runtime.h>

namespace ds4cuda {

// Launch tail-RoPE YaRN on a [n_heads, head_dim] fp32 buffer. The last
// `n_rot` dims of each head get rotated; the leading `head_dim - n_rot`
// dims are copied through. Suitable for Q (n_heads=N_HEAD=64) and KV
// (n_heads=N_HEAD_KV=1) on any layer.
//
//   x_in / x_out  device fp32, [n_heads * head_dim] elements (may overlap
//                 — kernel writes after reading the pair, so in-place is
//                 safe).
//   n_heads       N_HEAD (=64) for Q, N_HEAD_KV (=1) for KV.
//   head_dim      DS4_N_HEAD_DIM = 512.
//   n_rot         DS4_N_ROT = 64.
//   pos           Token position (uint32 in CPU ref). Determines the
//                 rotation angle — layer 0 prefill tok 9 uses pos=9.
//   il            Layer index. Drives compress_ratio, which in turn
//                 picks freq_base / freq_scale / ext_factor / attn_factor.
//   stream        CUDA stream (default 0).
void launch_tail_rope_yarn_f32(const float *x_in,
                               float       *x_out,
                               int          n_heads,
                               int          head_dim,
                               int          n_rot,
                               int          pos,
                               int          il,
                               cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_ROPE_CUH
