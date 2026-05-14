// router.cuh — host-callable launchers for the router stages.
//
// Implementation: cuda/router.cu.
//
// Stage coverage (matches ds4/ds4.c:5118-5180 hash-router fast path
// used by layers 0/1):
//
//   1) router_logits     fp32 [N_EXPERT=256]
//      = matvec_f16(layer->ffn_gate_inp, ffn_norm)
//      cited at ds4/ds4.c:5147 + ds4/ds4.c:5935
//
//   2) router_probs      fp32 [N_EXPERT=256]
//      = sqrt(softplus_stable(logits[i]))
//      cited at ds4/ds4.c:5149 + ds4/ds4.c:5937; softplus at ds4.c:4987
//      stable form: x>20 -> x; x<-20 -> exp(x); else log1p(exp(x)).
//
//   3) router_topk_ids   int32 [N_EXPERT_USED=6]
//      = ffn_gate_tid2eid[token_id*6 + i] for i in 0..6
//      cited at ds4/ds4.c:5119 layer_hash_selected_experts (data layout
//      `table + (uint64_t)token * DS4_N_EXPERT_USED` -> contiguous 6
//      int32 per token id).  Hash table dim layout in GGUF metadata is
//      I32 [6, 129280] with dim[0]=6, dim[1]=vocab — but the on-disk
//      bytes are token-major (token * 6 stride). Verified: dim[0]=6.
//
//   4) router_topk_w     fp32 [N_EXPERT_USED=6]
//      = expert_weight[i] from layer_hash_router_weights_from_probs
//      cited at ds4/ds4.c:5153-5168:
//          weights_out[i] = probs[selected[i]];
//          sum += weights_out[i];
//          ...
//          if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
//          weights_out[i] = weights_out[i] / sum * DS4_EXPERT_WEIGHT_SCALE; (=1.5)
//
// Numerical notes:
//   - F16 matvec uses __half2float (IEEE 754 binary16->binary32, identical
//     to ds4/ds4.c f16_to_f32 fallback). One warp per output row, fp32
//     accumulation.
//   - softplus_stable mirrors ds4.c:4987 verbatim, with the same x>20 /
//     x<-20 cutoffs.  sqrt is the IEEE sqrtf (single-precision rounded).
//   - hash topk_ids has zero math; just a 6-element int32 gather.
//   - topk_w sum-floor 6.103515625e-5f is fp16 min-subnormal-ish — kept
//     as a bit-for-bit constant.

#ifndef DS4CUDA_ROUTER_CUH
#define DS4CUDA_ROUTER_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ds4cuda {

// router_logits: F16 weight [out_dim, in_dim] row-major (row o starts at
// w_f16 + o*in_dim, length in_dim) × fp32 activation [in_dim] → fp32 [out_dim].
//
// Numerical contract: each output row is a serial fp32 dot product within
// one warp; lanes of the warp stride the input by 32 and reduce via
// __shfl_xor_sync.  The accumulation order can differ from the host
// `dot_f16_row` scalar loop, so 1-ULP-ish drift is expected (see ds4.c:2707).
//
// All buffers device memory.  in_dim need not be a multiple of 32; tail
// elements are handled by lane-bounded loads.
void launch_mul_mv_f16_f32(const uint16_t *w_f16,   // f16 [out_dim * in_dim]
                           const float    *x_f32,   // f32 [in_dim]
                           float          *y_f32,   // f32 [out_dim]
                           int             out_dim,
                           int             in_dim,
                           cudaStream_t    stream = 0);

// router_probs: y[i] = sqrt(softplus_stable(x[i])) element-wise, fp32 in/out,
// length n.  Pointwise — no reduction, no shared memory.
void launch_sqrt_softplus_f32(const float *x_f32,
                              float       *y_f32,
                              int          n,
                              cudaStream_t stream = 0);

// router_topk_ids (hash-routing fast path): one 6-element int32 gather
// from ffn_gate_tid2eid[token_id*6 .. token_id*6+6].
//
// table_i32     : on-disk int32 [vocab, 6] in token-major order
//                 (matches ds4.c:5134 indexing).
// token_id      : the prompt token id.
// out_i32       : int32 [k] selected expert ids.
// k             : usually DS4_N_EXPERT_USED = 6.
void launch_hash_router_topk_ids_i32(const int32_t *table_i32,
                                     int            token_id,
                                     int32_t       *out_i32,
                                     int            k,
                                     cudaStream_t   stream = 0);

// router_topk_w (hash-routing fast path):
//   sum = sum_i probs[selected[i]];   if sum < 6.103515625e-5f: sum = 6.103515625e-5f;
//   w[i] = probs[selected[i]] / sum * scale;
//
// All device fp32 / int32.  scale is DS4_EXPERT_WEIGHT_SCALE = 1.5f.
//
// Implemented as a single block of `k` threads — the gather + sum + divide
// fits in one warp for k <= 32.
void launch_hash_router_topk_w_f32(const float   *probs_f32,
                                   const int32_t *selected_i32,
                                   float         *w_out_f32,
                                   int            k,
                                   float          scale,
                                   float          sum_floor,
                                   cudaStream_t   stream = 0);

// Top-k routing with bias (layer 3..42, ds4.c:5217 layer_topk_selected_experts_from_probs):
//
//   selection[i] = probs[i] + exp_probs_b[i]   (i in 0..n_expert)
//   topk_ids     = argsort_desc(selection)[0..n_used]   (ds4.c:5182 topk_desc)
//   sum          = sum_{i in topk_ids} probs[i];        (UNBIASED probs!)
//   if (sum < sum_floor) sum = sum_floor;
//   topk_w[s]    = probs[topk_ids[s]] / sum * weight_scale;
//
// Note: the SELECTION uses (probs + bias), but the WEIGHTING uses the
// UNBIASED probs (not probs + bias).  This matches ds4.c:5234-5242.
//
// Inputs (device):
//   - router_probs   : fp32 [n_expert]    (= sqrt(softplus(logits)))
//   - exp_probs_b    : fp32 [n_expert]    (per-expert scalar bias, F32 from
//                      blk.<il>.exp_probs_b.bias).
// Outputs (device):
//   - topk_ids       : int32 [n_used]
//   - topk_w         : fp32  [n_used]
// Constants:
//   - n_expert       = 256, n_used = 6 in DS4.  weight_scale = 1.5,
//     sum_floor = 6.103515625e-5f.
//
// Implementation note: ds4.c:5182 `topk_desc` is an O(n*k) insertion-sort
// (256 * 6 = 1536 cmps) — small enough that a single block running the
// same algorithm in shared memory is bit-equal to the host loop.  We
// reproduce that scalar order to avoid any selection ambiguity at ties
// (per-expert prob+bias is fp32; ties at exact equality are extremely
// rare but the deterministic order matters for byte-identical alignment).
void launch_topk_selected_experts_f32(const float   *router_probs,
                                      const float   *exp_probs_b,
                                      int32_t       *topk_ids_out,
                                      float         *topk_w_out,
                                      int            n_expert,
                                      int            n_used,
                                      float          weight_scale,
                                      float          sum_floor,
                                      cudaStream_t   stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_ROUTER_CUH
