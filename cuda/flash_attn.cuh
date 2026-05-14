// flash_attn.cuh — host-callable launcher for the DeepSeek-V4
// raw-SWA decode-raw flash-attention CUDA kernel (`kqv_out` stage).
//
// Implements the per-head softmax over the sliding-window KV rows, with
// attention-sink logits folded into the softmax denominator. Single
// query token (the decoded position), MLA layout (n_head_kv = 1 so K and
// V are the same row buffer of dim head_dim).
//
// Spec — cite ds4/ds4.c:4868-4906 layer_attention_rows_one:
//
//     sinks         = blk.<il>.attn_sinks.weight    // f32[N_HEAD]
//     kq_scale      = 1.0f / sqrtf((float)head_dim)
//
//     for h in 0..n_head:
//         qh        = Q[h, :head_dim]
//         max       = sinks[h]
//         for k in 0..n_kv:
//             scores[k] = dot_f32(qh, K[k, :head_dim]) * kq_scale
//             max       = fmaxf(max, scores[k])
//         denom = exp(sinks[h] - max)              // sink contrib to S
//         out[h, :head_dim] = 0
//         for k in 0..n_kv:
//             w        = exp(scores[k] - max)
//             denom   += w
//             out[h]  += w * V[k, :head_dim]      // axpy
//         out[h] /= denom
//
// Block layout
// ------------
// Grid: n_head blocks (default 64).
// Block: 256 threads (8 warps). Each block handles one query head.
//   - First pass: compute scores[k] for all k (cooperative dot product
//     per row, warp + cross-warp shfl reduce, broadcast to all threads).
//   - Track running max (init = sinks[h]) on the fly.
//   - Second pass: each thread accumulates its strided slice of the
//     value-weighted sum of K rows + the per-step exp(score-max) into
//     the denominator. Final divide is broadcast.
//
// Numerical contract
// ------------------
//   - scores stored in shared memory [n_kv] floats.
//   - Score reduction: warp butterfly via __shfl_xor_sync, then 8-way
//     cross-warp via shared memory + a second warp-0 butterfly.
//   - Sink seeded into max_score before the K loop (same as CPU).
//   - All accumulation fp32, matching ds4/ds4.c CPU reference.
//
// Reference: ds4/metal/flash_attn.metal (three variants — this kernel
// covers `decode-raw`); sink logic at lines 1274-1290 mirrors the CPU form.

#ifndef DS4CUDA_FLASH_ATTN_CUH
#define DS4CUDA_FLASH_ATTN_CUH

#include <cuda_runtime.h>

namespace ds4cuda {

// Decode-raw flash-attention launcher. fp32 Q/K/V/sinks/out.
//
// Args:
//   Q        device fp32, [n_head * head_dim]
//   K        device fp32, [n_kv  * head_dim] (in MLA: K == V buffer)
//   V        device fp32, [n_kv  * head_dim] (== K for n_head_kv=1)
//   sinks    device fp32, [n_head]            // attention sinks
//   out      device fp32, [n_head * head_dim]
//   n_head   number of query heads (DS4_N_HEAD = 64)
//   n_kv     number of K/V rows in the (sub-128) sliding window
//   head_dim per-head dim (DS4_N_HEAD_DIM = 512)
//   stream   CUDA stream (default 0)
//
// The kernel asserts head_dim == 512 and n_kv <= 128 (DS4_N_SWA) at
// launch time; failures are reported via cudaGetLastError after the
// launch (kernel returns early on bad shapes).
void launch_flash_attn_decode_raw_f32(const float *Q,
                                      const float *K,
                                      const float *V,
                                      const float *sinks,
                                      float       *out,
                                      int          n_head,
                                      int          n_kv,
                                      int          head_dim,
                                      cudaStream_t stream = 0);

// Decode-mixed flash-attention launcher (layer 2+ ratio-4 path).
//
// Mirrors ds4/ds4.c:6657-6717 layer_attention_mixed_one. Computes one
// query token's heads against (n_raw raw-SWA rows + n_comp compressed
// rows), with per-compressed-row top-k mask `comp_allowed` from the
// indexer.
//
// Spec — cite ds4/ds4.c:6657-6717:
//
//   sinks    = blk.<il>.attn_sinks.weight             // f32[N_HEAD]
//   kq_scale = 1.0f / sqrtf((float)head_dim)
//   n_total  = n_raw + n_comp
//
//   for h in 0..n_head:
//       qh        = Q[h, :head_dim]
//       max       = sinks[h]
//       for r in 0..n_raw:
//           score[r] = dot_f32(qh, raw_kv[r]) * kq_scale
//           max      = fmaxf(max, score[r])
//       for r in 0..n_comp:
//           if comp_allowed && !comp_allowed[r]:
//               score[n_raw+r] = -INF
//               continue
//           score[n_raw+r] = dot_f32(qh, comp_kv[r]) * kq_scale
//           max            = fmaxf(max, score[n_raw+r])
//       denom = exp(sinks[h] - max)
//       out[h, :] = 0
//       for r in 0..n_raw:
//           w = exp(score[r] - max)
//           denom += w; out[h] += w * raw_kv[r]
//       for r in 0..n_comp:
//           if score[n_raw+r] <= -INF/2: continue
//           w = exp(score[n_raw+r] - max)
//           denom += w; out[h] += w * comp_kv[r]
//       out[h] /= denom
//
// Both raw_kv and comp_kv on disk are pre-f16-roundtrip (kv_cache_push_raw
// at ds4.c:6353-6363 and kv_cache_push_comp at ds4.c:6366-6371 round each
// row through fp16 on insert). The kernel applies the f16 round-trip on
// the read side for both buffers so the caller's launch contract stays
// fp32-in/fp32-out, matching the decode-raw pattern.
//
// Args:
//   Q             device fp32, [n_head * head_dim]
//   raw_kv        device fp32, [n_raw  * head_dim] (used for both K and V; MLA)
//   comp_kv       device fp32, [n_comp * head_dim] (used for both K and V)
//   comp_allowed  device int32, [n_comp]  // 1 = allowed, 0 = masked.
//                                          // Pass NULL to disable the mask
//                                          // (treats every comp row as allowed).
//   sinks         device fp32, [n_head]
//   out           device fp32, [n_head * head_dim]
//   n_head        number of query heads (DS4_N_HEAD = 64)
//   n_raw         number of raw SWA rows (<= DS4_N_SWA = 128)
//   n_comp        number of compressed rows
//   head_dim      per-head dim (DS4_N_HEAD_DIM = 512)
//   stream        CUDA stream (default 0)
//
// The kernel asserts head_dim == 512 and (n_raw + n_comp) <= 256 at
// launch time; failures are reported on stderr (kernel returns early).
void launch_flash_attn_decode_mixed_f32(const float   *Q,
                                        const float   *raw_kv,
                                        const float   *comp_kv,
                                        const int32_t *comp_allowed,
                                        const float   *sinks,
                                        float         *out,
                                        int            n_head,
                                        int            n_raw,
                                        int            n_comp,
                                        int            head_dim,
                                        cudaStream_t   stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_FLASH_ATTN_CUH
