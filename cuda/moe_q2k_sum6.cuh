// moe_q2k_sum6.cuh — host-callable launcher for the routed-MoE Q2_K
// down-projection summed across the 6 selected experts.
//
// Stage coverage: `ffn_moe_out` (final routed-expert FFN output,
// fp32 [N_EMBD=4096]).
//
// Pipeline mirror (cite ds4/ds4.c):
//   for slot in 0..N_USED=6:
//     ds4_quantize_row_q8_K(mid + slot * in_dim, midq[slot], in_dim)  ds4.c:5292
//   matvec_q2_k_experts_accum_prequant(out, ..., midq, selected, 6)    ds4.c:5296
//
// where matvec_q2_k_experts_accum_prequant runs the per-row body
// (ds4.c:3919-3927):
//     for row d in 0..out_dim:
//       acc = 0
//       for slot in 0..6:
//         dot = ds4_vec_dot_q2_K_q8_K(in_dim, &v, &W_down[slot,d,:],
//                                      midq[slot])
//         acc += dot
//       out[d] = acc
//
// This launcher mirrors the CPU path: it first allocates per-slot Q8_K
// activation scratch (n_used * n_blocks_in * sizeof(block_q8_K) ≈
// 6 * 8 * 292 = 14 KiB for in_dim=2048), quantizes each slot's mid row
// to Q8_K, then launches a single sum6 kernel where each block computes
// one output element via in-register accumulation across the 6 expert
// slots (NO atomicAdd — design §4 explicitly forbids).
//
// Block layout: 1 CTA = 1 output row d (grid.x = out_dim = 4096),
// blockDim = 32 (one warp). Lane partition over n_blocks_in × 6 slots
// = 6 * 8 = 48 super-block-slots (per row). Each lane handles 1 or 2
// of those, accumulates float sumf, warp-reduce at the end.
//
// Note on slot order: `topk_ids` is N/A here — the W_down pointer
// already refers to the 6 selected slices in the slot order matching
// `mid` (which is what `routed_expert_mid` stores). Caller passes
// W_down packed as [n_used][out_dim][n_blocks_in].

#ifndef DS4CUDA_MOE_Q2K_SUM6_CUH
#define DS4CUDA_MOE_Q2K_SUM6_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"

namespace ds4cuda {

// Routed MoE Q2_K down-projection summed over n_used slots, full-tensor
// variant with caller-owned per-slot Q8_K activation scratch.
// W_down points at the full expert tensor [n_experts][out_dim][blocks];
// the kernel indexes selected experts with topk_ids[slot] directly.
//   scratch_midq_dev     : int8  [n_used * in_dim]
//   scratch_midscale_dev : float [n_used * in_dim / 256]
//   scratch_midbsums_dev : int16 [n_used * (in_dim / 256) * 16]
// Performs no device allocation/free and does not synchronize.
//
// Preconditions: in_dim % 256 == 0; n_used in [1, 6]; n_used * out_dim
// fits in int.
void launch_routed_moe_q2k_sum6_full_f32_prealloc(
                                         const block_q2_K *W_down,
                                         const float      *mid,
                                         const int32_t    *topk_ids,
                                         float            *out,
                                         int               n_used,
                                         int               out_dim,
                                         int               in_dim,
                                         int               n_experts,
                                         int8_t           *scratch_midq_dev,
                                         float            *scratch_midscale_dev,
                                         int16_t          *scratch_midbsums_dev,
                                         cudaStream_t      stream = 0);

// Build a model/session-resident full-expert Q2_K SoA mirror from the
// source AoS tensor:
//   W_down     [n_experts][out_dim][n_blocks_in] block_q2_K
//   scales_soa [n_experts][out_dim][16][n_blocks_in]
//   qs_soa     [n_experts][out_dim][64][n_blocks_in]
//   d_soa      [n_experts][out_dim][n_blocks_in]
//   dmin_soa   [n_experts][out_dim][n_blocks_in]
// This is a load/session-init operation, not a per-token packer.
void launch_build_moe_q2k_sum6_resident_soa(
                                         const block_q2_K *W_down,
                                         uint8_t          *scales_soa,
                                         uint8_t          *qs_soa,
                                         uint16_t         *d_soa,
                                         uint16_t         *dmin_soa,
                                         int               n_experts,
                                         int               out_dim,
                                         int               in_dim,
                                         cudaStream_t      stream = 0);

// Resident SoA v2 routed MoE Q2_K down-projection sum, with caller-owned
// activation scratch. Performs fp32->Q8_K quantization, Q8_K SoA
// transpose, and the resident dot. Performs no allocation/free and
// does not synchronize. This is the production hot-path launcher.
void launch_routed_moe_q2k_sum6_resident_soa_v2_f32_prealloc(
                                         const uint8_t    *scales_soa,
                                         const uint8_t    *qs_soa,
                                         const uint16_t   *d_soa,
                                         const uint16_t   *dmin_soa,
                                         const float      *mid,
                                         const int32_t    *topk_ids,
                                         float            *out,
                                         int               n_used,
                                         int               out_dim,
                                         int               in_dim,
                                         int               n_experts,
                                         int8_t           *scratch_midq_dev,
                                         float            *scratch_midscale_dev,
                                         int16_t          *scratch_midbsums_dev,
                                         int8_t           *scratch_midq_soa_dev,
                                         int16_t          *scratch_midbsums_soa_dev,
                                         cudaStream_t      stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_MOE_Q2K_SUM6_CUH
