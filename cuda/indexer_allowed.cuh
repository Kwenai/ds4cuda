// indexer_allowed.cuh — host-callable launcher for the indexer top-K
// "allowed compressed rows" mask.  Two paths:
//   - short-circuit branch (n_comp <= top_k => all entries == 1).
//   - long-prompt top-K scoring path (matvec_f16 indexer_attn_q_b +
//     matvec_f16 indexer_proj + dot product over all comp rows +
//     iterative argmax) — triggered only when n_index_comp > top_k
//     (=DS4_N_INDEXER_TOP_K=512). Short prompt "Hello" never triggers
//     it because n_index_comp peaks at 2.
//
// Reference: ds4/ds4.c:6900-6959 (indexer_allowed_decode_one).
//
// Dump layout (cite: patches/ds4_cpu_stage_dump.patch lines 484-495):
//   On disk the bool[n_comp] mask is converted to int32[n_comp] with
//   value 1 = allowed, 0 = not.  The DSST file is dtype=2 (I32).

#ifndef DS4CUDA_INDEXER_ALLOWED_CUH
#define DS4CUDA_INDEXER_ALLOWED_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ds4cuda {

// Short-circuit version.  Sets out[0..n_comp) := 1 if n_comp <= top_k,
// otherwise this launcher aborts (callers MUST use the long path
// `launch_indexer_score_topk_i32` below when n_comp > top_k).  All
// values written as int32 (matches the on-disk dump layout).
//
//   out_i32 : device int32 [n_comp].
//   n_comp  : current cache->n_index_comp (= number of indexer-side
//             compressed rows accumulated so far).  Must be > 0.
//   top_k   : DS4_N_INDEXER_TOP_K = 512.
void launch_indexer_allowed_short_circuit_i32(int32_t     *out_i32,
                                              int          n_comp,
                                              int          top_k,
                                              cudaStream_t stream = 0);

// Long-prompt indexer scoring + top-K.
//
// Mirrors the n_comp > top_k branch of ds4.c:6926 indexer_allowed_decode_one.
// Inputs:
//   q              [n_head * head_dim] fp32  — post-RoPE indexer Q.
//   weights        [n_head]            fp32  — per-head weights; the caller
//                                              MUST have already multiplied
//                                              them by 1/sqrt(head_dim*n_head).
//   index_comp     [n_comp * head_dim] f16   — indexer compressed KV rows.
//   n_comp                                   — number of compressed rows.
//   n_head                                   — DS4_N_INDEXER_HEAD = 64.
//   head_dim                                 — DS4_N_INDEXER_HEAD_DIM = 128.
//   top_k                                    — DS4_N_INDEXER_TOP_K = 512.
//
// Output:
//   allowed_out    [n_comp] int32  — 0/1 mask, exactly `top_k` ones.
//   scratch_scores [n_comp] fp32   — caller-owned scratch, kernel-internal.
//
// Two-phase kernel launch:
//   1. score kernel: grid = n_comp blocks of n_head threads. Thread h
//      computes ReLU(dot(q[h*head_dim..], kv_c)) * weights[h]; the block
//      reduces these n_head terms into scores[c].
//   2. top-K kernel: single block. Iteratively picks argmax of remaining
//      scores, marks it allowed (1 if selected), top_k iterations total.
//      For n_comp <= ~16384 and top_k=512 this is microseconds-scale.
//
// ULP-class drift on top-K boundaries is accepted (ReLU + score sort
// can swap a handful of indices near the cutoff vs. ds4 CPU); attention
// math is downstream-tolerant.
void launch_indexer_score_topk_i32(const float    *q,
                                   const float    *weights,
                                   const uint16_t *index_comp,
                                   int             n_comp,
                                   int             n_head,
                                   int             head_dim,
                                   int             top_k,
                                   int32_t        *allowed_out,
                                   float          *scratch_scores,
                                   cudaStream_t    stream = 0);

// FP32-KV variant of launch_indexer_score_topk_i32. Our session-state
// indexer cache stores compressed rows as fp32 (cuda/session_state.cu
// carves index_comp_kv as a float buffer), so the in-place CUDA path
// uses this variant. The math is identical to the f16 variant except
// for the KV dequant step.
void launch_indexer_score_topk_f32_i32(const float *q,
                                       const float *weights,
                                       const float *index_comp_f32,
                                       int          n_comp,
                                       int          n_head,
                                       int          head_dim,
                                       int          top_k,
                                       int32_t     *allowed_out,
                                       float       *scratch_scores,
                                       cudaStream_t stream = 0);

// Small per-element scale kernel: y[i] *= scale, for i in [0, n).
// Used to scale the indexer head weights by 1/sqrt(head_dim*n_head)
// after the F16 matvec produces them, before invoking the score kernel.
void launch_scale_inplace_f32(float        *y,
                              int           n,
                              float         scale,
                              cudaStream_t  stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_INDEXER_ALLOWED_CUH
