// forward_layer.cuh — host-callable streaming forward for a single
// DeepSeek-V4 layer.
//
// Covers all 43 layers (il=0..42). Layer 2 exercises the layer-2 specific
// compressor + indexer + decode-mixed flash-attention chain, while layers
// 0 and 1 use the dense ratio-0 path.
//
// Implementation: cuda/forward_layer.cu.
//
// Pipeline for layer 0 / 1 (dense ratio-0, matches ds4/ds4.c:7050-7155
// layer_attention_raw_swa_one + layer_ffn_one for il<2):
//
//   Input:
//     - input_residual_hc : device fp32 [N_HC=4 * N_EMBD=4096] = 16384
//     - token_id          : prompt token id (used only by hash router for
//                           layers 0,1; ignored after that).
//     - pos               : token position (drives RoPE angle).
//
//   Output:
//     - output_residual_hc: device fp32 [N_HC=4 * N_EMBD=4096] = 16384
//     - session_state.layers[il].raw_kv ring updated in place; n_raw++.
//
//   Side effects: temporary intermediates written into
//   session_state.activation_arena (16 MiB bump arena — caller responsible
//   for not running concurrent forward calls that share the arena).
//
//   Stage chain (31 launchers for layer 0/1, 35 for layer 2):
//     [Attention half]
//       1.  hc_attn_pre  (RMSNorm + matvec + sinkhorn + weighted-sum)
//       2.  attn_norm    (RMSNorm)
//       3.  q_lora       (Q8_0 matvec, attn_q_a)
//       4.  q_lora_norm  (RMSNorm)
//       5.  KVraw        (Q8_0 matvec, attn_kv_a, == 1 KV head)
//       6.  KVnorm       (RMSNorm with attn_kv_a_norm.weight)
//       7.  Qraw         (Q8_0 matvec, attn_q_b)
//       8.  Qnorm        (per-head RMSNorm, no weight)
//       9.  Qcur         (tail-RoPE YaRN, n_heads=64)
//      10.  KVrope       (tail-RoPE YaRN, n_heads=1)
//      11.  KVcur        (FP8 round-trip)
//      12.  push raw_kv  (cudaMemcpyDeviceToDevice into session_state)
//
//      LAYER 2 ONLY — between push raw_kv and flash_attn:
//        12a. attn_compressor_decode_step
//             cite ds4.c:7093  (compressor on attn_norm input,
//                               weights blk.<il>.attn_compressor_*)
//             writes state row, on emit (pos+1)%4==0 emits comp_kv ->
//             pushed to comp_kv ring (with fp16 round-trip on push,
//             cite ds4.c:6395).
//        12b. indexer_compressor_decode_step
//             cite ds4.c:7111 (compressor on attn_norm input,
//                              weights blk.<il>.indexer_compressor_*)
//             same ring push to index_comp_kv (fp16 round-trip).
//        12c. indexer_allowed_short_circuit (n_index_comp <= 512)
//             cite ds4.c:6926, fills [n_index_comp] int32 mask of 1s.
//
//      13.  flash_attn   (decode_raw for layer 0/1, decode_mixed for layer 2)
//      14.  kqv_back     (inverse tail-RoPE)
//      15.  attn_low     (Q8_0 grouped matvec, attn_output_a)
//      16.  attn_out     (Q8_0 matvec, attn_output_b)
//      17.  hc_post_attn (HC expand+add+split using hc_attn_pre's post/comb)
//
//     [FFN half]
//      18.  hc_ffn_pre   (RMSNorm + matvec + sinkhorn + weighted-sum)
//      19.  ffn_norm     (RMSNorm)
//      20.  shared_gate  (Q8_0 matvec, ffn_gate_shexp)
//      21.  shared_up    (Q8_0 matvec, ffn_up_shexp)
//      22.  shared_silu_mul (SwiGLU body)
//      23.  ffn_shexp    (Q8_0 matvec, ffn_down_shexp)
//      24.  router_logits (F16 matvec, ffn_gate_inp)
//      25.  router_probs (sqrt+softplus)
//      26.  router_topk_ids (hash-router gather; il<3 uses hash router)
//      27.  router_topk_w  (sum-floor + scale)
//      28.  routed_expert_mid (IQ2_XXS pair gate+up + clamp + SwiGLU + weight)
//      29.  ffn_moe_out  (Q2_K sum-6 down)
//      30.  ffn_out      (add ffn_shexp + ffn_moe_out)
//      31.  hc_post_ffn  (HC expand+add+split using hc_ffn_pre's post/comb)
//
// Output residual_hc = hc_post_ffn output.
//
// MoE expert-slice handling: at each forward call we read 6 selected
// experts' gate + up (IQ2_XXS) + down (Q2_K) slices from the model's
// managed weight buffer and pack them into the activation arena as a
// contiguous [6][...] block, since the existing routed_moe launchers
// expect packed input.  Total temporary expert-pack footprint:
//   - gate slice : 6 * 2048 * 16 * 66 = 12.4 MiB
//   - up slice   : 6 * 2048 * 16 * 66 = 12.4 MiB
//   - down slice : 6 * 4096 *  8 * 84 = 16.1 MiB
//   total          ~ 41 MiB, allocated separately from activation_arena.

#ifndef DS4CUDA_FORWARD_LAYER_CUH
#define DS4CUDA_FORWARD_LAYER_CUH

#include <cuda_runtime.h>

#include "ds4cuda.h"

namespace ds4cuda {

// Generic entry point: forward one layer's streaming step for il in
// 0..42.  All buffers are device pointers; input/output residual_hc
// are fp32 [N_HC * N_EMBD] = 16384 floats.
//
// Returns 0 on success, negative on failure (reports to stderr and either
// returns -1 or aborts on internal CUDA errors via the launcher CK macro).
//
// Preconditions:
//   - m            : model loaded via ds4_model_load_to_managed.
//   - s            : session_state allocated via ds4_session_state_alloc;
//                    s->layers[il].raw_kv has at least n_raw_post=n_raw+1
//                    entries available (cap_raw must be >= n_raw_post).
//                    For il>=2 the caller must additionally pre-load the
//                    compressor (and, on ratio-4 layers, indexer) state
//                    buffers + comp_kv / index_comp_kv rings to mirror the
//                    per-token streaming state that ds4 would have built up
//                    before tok `pos`.
//   - il in 0..42  : full layer dispatch scope.  Per-layer dispatch:
//                    - il<2 (ratio=0): dense raw SWA (decode_raw); no
//                      compressor; hash router (DS4_N_HASH_LAYER=3).
//                    - il==2 (ratio=4): hash router; attn compressor +
//                      indexer compressor + decode_mixed.
//                    - il==3 odd ratio=128: top-k router (biased argsort);
//                      attn compressor only (no indexer); decode_mixed.
//                      For "Hello" 10-token prompt n_comp=0 (ratio=128 emit
//                      threshold not reached), so decode_mixed degenerates
//                      to the raw-SWA loop.
//                    - il>=4 even (ratio=4): top-k router; attn + indexer
//                      compressor; decode_mixed.
//                    - il>=5 odd  (ratio=128): top-k router; attn
//                      compressor only; decode_mixed (degenerate as il=3).
//   - token_id     : the prompt/generated token id; hash router lookups
//                    use it for layers 0,1,2 (DS4_N_HASH_LAYER=3).  Ignored
//                    by the top-k path (il>=3).
//   - pos          : token position (0-based).
//
// The function enqueues work on the caller's stream and returns without a
// per-layer stream synchronize.  The caller must synchronize before
// reusing/freeing stream-ordered device outputs.
int ds4_forward_layer(const struct ds4_model      *m,
                      struct ds4_session_state    *s,
                      int                          il,           // 0..42
                      int                          token_id,
                      int                          pos,
                      const float                 *input_residual_hc,
                      float                       *output_residual_hc,
                      cudaStream_t                 stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_FORWARD_LAYER_CUH
