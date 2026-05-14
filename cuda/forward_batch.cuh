// forward_batch.cuh — host-callable layer-major chunked prefill.
//
// "chunked prefill 是 layer-major + 绝对 chunk 边界对齐"
//
// The chunked prefill reorganizes ds4_forward_token's outer/inner loop
// from token-major (per token: walk all 43 layers) to layer-major
// (per layer: process all N tokens in the chunk).  The crucial property
// is that each layer's weights are loaded and consumed once per chunk
// instead of once per token — a 2048x reduction in weight-band stress
// on a long prompt (chunk_size = 2048).
//
// Pipeline (high level):
//
//   1. Embed all N tokens into a per-token residual_hc buffer
//      [n_tokens, N_HC * N_EMBD].  Same kernel as forward_token.cu
//      (launch_embed_token_f16_to_f32 + 4-replica HC build), one
//      launch per token but trivial cost.
//
//   2. for il in 0..N_LAYER=43:
//          for t in 0..n_tokens:
//              ds4_forward_layer(m, s, il, token_ids[t], pos=chunk_start+t,
//                                in_resid_hc + t*HC_DIM,
//                                out_resid_hc + t*HC_DIM,
//                                stream, capture=NULL)
//          swap(in_resid_hc, out_resid_hc)
//
//      Within layer `il` the per-layer launchers re-read the same
//      weights N times — the minimal version does NOT yet batch
//      the matvec kernels.  However the *order* matches the layer-major
//      contract and the KV/compressor/indexer rings advance in the
//      same emit order as ds4's prefill_layer_major_cpu (cite
//      ds4.c:7910 prefill_layer_major_cpu): per-layer iterate over
//      all tokens before moving to the next layer.  This is sufficient
//      to validate the chunk boundary at the next step where the
//      inner per-token loop is replaced by a true batched matvec.
//
//   3. Final stage on the LAST token of the chunk only
//      (forward_token's tail path: output_hc_head + output_norm +
//      output projection).  All other tokens' residual_hc are discarded
//      because prefill never needs their logits.
//
// Memory:
//   - per-chunk activation arena overhead ≈ N_TOKENS * 2 * HC_DIM * 4 B
//     = N * 128 KiB.  At chunk_size=2048: 256 MiB on the device.
//     For "Hello" (N=10): 1.28 MiB — trivial.  Allocated separately
//     from session_state.activation_arena so the per-layer call's
//     bump arena starts fresh at cur=0.
//
// This is the minimal viable contract — same numerical floor as
// streaming forward_token because both use the identical per-layer
// kernels.  argmax(logits) at chunk_end MUST equal the streaming
// 10x forward_token argmax (== 2581 for "Hello").  Future steps
// will replace the inner per-token loop with batched matvec kernels
// (launch_mul_mv_q8_0_q8_0_batched, launch_flash_attn_prefill_masked,
// etc.) without changing this entry point's contract.

#ifndef DS4CUDA_FORWARD_BATCH_CUH
#define DS4CUDA_FORWARD_BATCH_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "ds4cuda.h"
#include "common.cuh"

namespace ds4cuda {

// Forward one chunk of `n_tokens` prompt tokens through the full ds4
// stack in layer-major order, producing the LAST token's logits.
//
// Returns 0 on success, negative on failure.  On success the host
// buffer `out_logits_last` (size N_VOCAB fp32 = 129280 floats) is
// populated with the last-token logits, copied D2H synchronously
// before return.
//
// Pre-conditions:
//   - m              : model loaded via ds4_model_load_to_managed.
//   - s              : session_state allocated via ds4_session_state_alloc;
//                      KV cap_raw must accommodate `chunk_start_pos +
//                      n_tokens` per layer.  cap_comp must accommodate
//                      ceil((chunk_start_pos+n_tokens)/4) + slack at
//                      each ratio-4 layer.
//   - token_ids      : host int array [n_tokens] of prompt tokens.
//   - chunk_start_pos: position of the FIRST token in the chunk
//                      (== 0 for the first chunk of a fresh session).
//   - n_tokens       : <= 2048.
//   - out_logits_last: host fp32 buffer of size N_VOCAB elements;
//                      receives the last-token logits.
//
// Synchronizes the stream internally before returning.
//
// Side effects:
//   - session_state KV / compressor / indexer rings advance by exactly
//     n_tokens slots (per layer's compress ratio).  After return the
//     session is ready for further chunks or per-token decode.
int ds4_forward_chunk(const struct ds4_model       *m,
                      struct ds4_session_state     *s,
                      const int                    *token_ids,
                      int                           chunk_start_pos,
                      int                           n_tokens,
                      float                        *out_logits_last,
                      cudaStream_t                  stream = 0);

// Maximum tokens per chunk supported by the minimal entry point
// (matches design doc §1 (F) "chunk size = 2048 token").  Callers
// passing n_tokens > this value should split into multiple chunks
// (multi-chunk boundary alignment is a future enhancement).
static constexpr int DS4_CHUNK_SIZE_MAX = 2048;

} // namespace ds4cuda

#endif // DS4CUDA_FORWARD_BATCH_CUH
