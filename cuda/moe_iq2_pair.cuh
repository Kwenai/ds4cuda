// moe_iq2_pair.cuh — host-callable launcher for the routed MoE
// gate+up paired IQ2_XXS matvec fused with SwiGLU + clamp + router weight.
//
// Stage coverage: `routed_expert_mid` (per-slot, per-row hidden output
// of the routed-expert SwiGLU body, weighted by router_topk_w).
//
// Pipeline mirror (cite ds4/ds4.c):
//   1. ds4_quantize_row_q8_K(x, xq, in_dim)              ds4.c:1628
//   2. for slot in 0..N_USED:
//      for row in 0..out_dim (= FF_EXP):
//        gate = ds4_vec_dot_iq2_xxs_pair_q8_K(           ds4.c:1877 (gate side)
//        up   = ds4_vec_dot_iq2_xxs_pair_q8_K(             same   (up side)
//        if c > 1e-6: gate=min(gate,c); up=clamp(up,-c,c) ds4.c:3805
//        mid[slot,row] = silu(gate) * up * weight[slot]  ds4.c:3810
//
// Internally this launcher first invokes a Q8_K activation quantizer
// on the fp32 input vector (see launch_quantize_fp32_to_q8_K), then a
// fused per-(slot,row) pair-dot kernel with the immediate SwiGLU tail.
// The gate+up dot share the IQ2_XXS aux-decode trip — a single grid
// table lookup feeds both. This matches the Metal kernel
// kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 (moe.metal:959) but uses
// a Q8_K activation to bit-match the CPU dump (Metal uses fp32 directly
// — Metal’s 0.25 scale × signed-grid×fp32 path produces a slightly
// different result than the int8×int8 dot path used by ds4 CPU; we
// pick the CPU path because the dumps come from CPU).
//
// Block layout: 1 block per (slot,row_group) where row_group = 4 rows
// per block (N_R0_IQ2_XXS in metal). 1 warp per block; lane processes
// every 32nd ib32 sub-block. Final simd_sum reduces across the warp;
// lane 0 writes mid[slot,row] for each of the 4 rows.

#ifndef DS4CUDA_MOE_IQ2_PAIR_CUH
#define DS4CUDA_MOE_IQ2_PAIR_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"

namespace ds4cuda {

// Quantize fp32 activation [n_elems] into Q8_K super-blocks.
//   n_elems must be a multiple of 256.
//   xq_dev receives n_blocks * 256 int8 (qs[]).
//   xscale_dev receives n_blocks fp32 (block_q8_K::d).
//   xbsums_dev receives n_blocks * 16 int16 (block_q8_K::bsums[]).
//
// Mirrors ds4_quantize_row_q8_K in ds4/ds4.c:1628 exactly:
//     amax = max |x[j]|, max = signed value with max |x|
//     iscale = -127.0f / max          (note: -127, not 127, so the sign
//                                      of the largest-magnitude element
//                                      gets pinned to -127 / 127 cleanly)
//     y[b].qs[j] = clamp(lrintf(iscale * x[j]), -128, 127)
//     y[b].bsums[i] = sum_{j in [16i, 16i+16)} qs[j]
//     y[b].d = 1.0f / iscale          (raw fp32, not fp16-round-tripped)
//
// All-zero blocks set d = 0, qs = 0, bsums = 0 (early return).
void launch_quantize_fp32_to_q8_K(const float *x_dev,
                                  int8_t      *xq_dev,
                                  float       *xscale_dev,
                                  int16_t     *xbsums_dev,
                                  int          n_blocks,
                                  cudaStream_t stream = 0);

// Launch the fused gate+up pair-IQ2_XXS · Q8_K matvec + SwiGLU + clamp +
// route-weight kernel. Full-tensor variant with caller-owned Q8_K
// activation scratch:
//   scratch_xq_dev     : int8   [in_dim]
//   scratch_xscale_dev : float  [in_dim / 256]
//   scratch_xbsums_dev : int16  [(in_dim / 256) * 16]
// W_gate/W_up point at the full expert tensor
// [n_experts][out_dim][in_dim/256]; the kernel indexes expert slices
// with topk_ids[slot] directly. Performs no device allocation/free and
// does not synchronize.
//
// Preconditions: in_dim % 256 == 0; n_used <= 6.
void launch_routed_moe_pair_swiglu_full_f32_prealloc(
                                            const block_iq2_xxs *W_gate,
                                            const block_iq2_xxs *W_up,
                                            const float         *x_fp32,
                                            const int32_t       *topk_ids,
                                            const float         *topk_w,
                                            float               *mid_out,
                                            int n_used,
                                            int out_dim,
                                            int in_dim,
                                            int n_experts,
                                            float clamp_value,
                                            int8_t  *scratch_xq_dev,
                                            float   *scratch_xscale_dev,
                                            int16_t *scratch_xbsums_dev,
                                            cudaStream_t stream = 0);

// Same routed MoE gate/up SwiGLU contract as
// launch_routed_moe_pair_swiglu_full_f32_prealloc, but reading SoA v2
// packed weights (qs/d separated) instead of AoS block_iq2_xxs[].
//
// Lane->data mapping is inverted vs. the AoS kernel: lane l processes
// super-block l of the row (one super-block per lane, exclusively),
// turning the AoS kernel's 66-B-stride loads into coalesced 64-B warp
// transactions across n_blocks_in adjacent uint16s in qs_soa.
//
// Math is identical to the AoS kernel; output is ULP-equivalent but NOT
// byte-equal (the warp reduction tree sees partials in a permuted order).
//
// Constraint: in_dim/256 <= 32 (production = 16). Larger super-block
// counts would need lane multiplexing — early-exit instead.
//
// SoA tensor shapes (per gate/up tensor, byte layout matches
// ds4_iq2_xxs_soa_v2_layout()):
//   qs_soa : uint16 [n_experts][out_dim][32][in_dim/256]
//   d_soa  : uint16 [n_experts][out_dim][in_dim/256]
//
// Activation scratch is caller-owned (same shapes as the AoS prealloc
// launcher above). Performs no device allocation/free and does not
// synchronize.
void launch_routed_moe_pair_swiglu_resident_soa_v2_f32_prealloc(
                                            const uint16_t *qs_soa_gate,
                                            const uint16_t *d_soa_gate,
                                            const uint16_t *qs_soa_up,
                                            const uint16_t *d_soa_up,
                                            const float    *x_fp32,
                                            const int32_t  *topk_ids,
                                            const float    *topk_w,
                                            float          *mid_out,
                                            int n_used,
                                            int out_dim,
                                            int in_dim,
                                            int n_experts,
                                            float clamp_value,
                                            int8_t  *scratch_xq_dev,
                                            float   *scratch_xscale_dev,
                                            int16_t *scratch_xbsums_dev,
                                            cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_MOE_IQ2_PAIR_CUH
