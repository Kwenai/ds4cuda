// forward_token.cuh — host-callable streaming forward for one DeepSeek-V4
// token through the full 43-layer stack + final HC head + output
// projection.
//
// Wraps ds4_forward_layer (il in {0..42}) and the final stage chain
// (output_hc_head + output_norm + output projection).  Dev-era terminal
// argmax gate (since dropped): argmax of final logits == 2581 for the
// "Hello" prompt.
//
// Pipeline (matches ds4/ds4.c:7822-7847 forward_token_raw_swa_cpu_decode_scratch
// + ds4.c:8145-8168 output_logits_one_decode_scratch):
//
//   1.  embed_token_f16(token_id) -> token_embd_f32 [N_EMBD]
//                                    (cite ds4.c:7833 forward_token line 7833)
//   2.  for h in 0..N_HC=4: residual_hc[h, :] = token_embd_f32     (4 replica)
//                                    (cite ds4.c:7836 hc_from_plain_embedding)
//   3.  for il in 0..N_LAYER=43:
//          residual_hc = ds4_forward_layer(il, residual_hc, ...)   (cite ds4.c:7843)
//   4.  output_logits_one(residual_hc) -> fp32 [N_VOCAB=129280]    (cite ds4.c:7846)
//
// output_logits_one_decode_scratch (ds4.c:8145-8168) inner pipeline:
//   5.  output_hc_head_one(residual_hc) -> output_collapsed [N_EMBD]
//                                    (cite ds4.c:8154 output_hc_head_one)
//   6.  output_norm = rms_norm_weight(output_collapsed, output_norm.w, eps)
//                                    (cite ds4.c:8158 rms_norm_weight)
//   7.  logits = matvec_q8_0(output.weight, output_norm) -> fp32 [N_VOCAB]
//                                    (cite ds4.c:8162 matvec_q8_0)
//
// The activation_arena (16 MiB) inside session_state is a bump pointer
// pool — each call frees nothing, just rewinds.  See cuda/forward_layer.cu
// for the per-layer arena carve helper used as a template.

#ifndef DS4CUDA_FORWARD_TOKEN_CUH
#define DS4CUDA_FORWARD_TOKEN_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "ds4cuda.h"
#include "common.cuh"

namespace ds4cuda {

// Forward one token through the full ds4 stack and produce final logits.
//
// Returns 0 on success, negative on failure.  On success `*logits_out`
// is set to a device fp32 pointer of length N_VOCAB=129280, carved from
// the session's activation_arena.  The pointer is valid until the next
// ds4_forward_token call (which will rewind the arena and reuse the
// memory).
//
// Pre-conditions:
//   - m  : model loaded via ds4_model_load_to_managed.  Required tensors:
//          token_embd.weight, blk.<il>.* for il in 0..42, output_hc_fn.weight,
//          output_hc_scale.weight, output_hc_base.weight, output_norm.weight,
//          output.weight.
//   - s  : session_state allocated via ds4_session_state_alloc;
//          s->layers[il].raw_kv has at least n_raw_post=n_raw+1 entries
//          available for all il.  Caller is responsible for any layer-2+
//          compressor / indexer state mirroring needed before the call.
//
// Note:
//   - This wrapper relies on ds4_forward_layer accepting il in {0..42}.
//     The companion test bypasses the layer chain entirely and exercises
//     only the tail (steps 5..7 above) using the il42 hc_ffn_post dump
//     as residual_hc input.
int ds4_forward_token(const struct ds4_model       *m,
                      struct ds4_session_state     *s,
                      int                           token_id,
                      int                           pos,
                      const float                 **logits_out,
                      cudaStream_t                  stream = 0);

// Final stage chain (steps 5..7 above) factored out so the alignment
// test can drive it directly with the il42 hc_ffn_post dump as input,
// without needing ds4_forward_layer to handle il>2.
//
// Inputs (device):
//   - residual_hc      : fp32 [N_HC * N_EMBD = 16384] (last layer hc_ffn_post)
// Weights (device, managed pointers):
//   - output_hc_fn     : F16 [N_HC * N_EMBD, N_HC]
//   - output_hc_scale  : F32 [1]
//   - output_hc_base   : F32 [N_HC]
//   - output_norm_w    : F32 [N_EMBD]
//   - output_w_q8_0    : Q8_0 [N_EMBD, N_VOCAB]
// Scratch (device, sized per the spec):
//   - scratch_collapsed: fp32 [N_EMBD]   (output_hc_head output)
//   - scratch_norm     : fp32 [N_EMBD]   (output_norm output)
//   - scratch_q8_xq    : int8  [N_EMBD]   (output projection Q8 activation)
//   - scratch_q8_scale : fp32 [N_EMBD/32] (output projection Q8 scales)
// Output (device):
//   - logits_out       : fp32 [N_VOCAB]
//
// Enqueues all work on `stream`; performs no dynamic device allocation and
// does not synchronize internally.
int ds4_forward_token_final_stage(
        const float           *residual_hc,
        const uint16_t        *output_hc_fn,
        const float           *output_hc_scale,
        const float           *output_hc_base,
        const float           *output_norm_w,
        const block_q8_0      *output_w_q8_0,
        int                    n_embd,
        int                    n_hc,
        int                    n_vocab,
        float                  rms_eps,
        float                 *scratch_collapsed,
        float                 *scratch_norm,
        float                 *logits_out,
        int8_t                *scratch_q8_xq,
        float                 *scratch_q8_scale,
        cudaStream_t           stream = 0,
        int                    perf_token = -1);

} // namespace ds4cuda

#endif // DS4CUDA_FORWARD_TOKEN_CUH
