// hc_sinkhorn.cuh — host-callable launcher for the HC pre (sinkhorn mixer)
// CUDA kernel covering the `hc_attn_pre_post_weights`, `hc_attn_pre_comb`,
// and `hc_attn_pre` stages.
//
// Implementation: cuda/hc_sinkhorn.cu.
//
// Pipeline (matches ds4/ds4.c:4255 hc_pre_from_state_one_scratch and
// ds4/metal/dsv4_hc.metal:83 kernel_dsv4_hc_split_sinkhorn):
//   1) RMSNorm (no weight) over the full hc_dim = N_HC * N_EMBD residual.
//   2) F16 matvec: mix[24] = hc_attn_fn ([16384, 24]) * flat[16384].
//   3) Sinkhorn split:
//        - pre_weights[4]   = sigmoid(mix[0..4]   * scale[0] + base[0..4])  + eps
//        - post_weights[4]  = 2 * sigmoid(mix[4..8] * scale[1] + base[4..8])
//        - comb[16] (4x4)   = softmax-row(mix[8..24] * scale[2] + base[8..24])
//                             then DS4_N_HC_SINKHORN_ITER (=20) doubly-stochastic
//                             normalization sweeps starting from a column-norm.
//   4) HC weighted sum:
//        out[d] = sum_{h=0..3} pre_weights[h] * residual_hc[h * N_EMBD + d]
//
// Outputs (all device fp32):
//   - post_weights : [N_HC]            stage `hc_attn_pre_post_weights`
//   - comb         : [N_HC * N_HC]     stage `hc_attn_pre_comb`
//   - out          : [N_EMBD]          stage `hc_attn_pre`
//
// All three are emitted from one kernel launch; the test compares each
// against its own DSST reference dump.

#ifndef DS4CUDA_HC_SINKHORN_CUH
#define DS4CUDA_HC_SINKHORN_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ds4cuda {

// HC pre kernel for one token (batch dim = 1). Layout follows ds4/ds4.c:
//   - residual_hc : [N_HC, N_EMBD] row-major (h-major, h * N_EMBD + d)
//   - hc_attn_fn  : [out_dim=24, in_dim=16384] row-major fp16, packed as
//                   uint16_t bits (matches GGUF on-disk layout — row o
//                   starts at hc_attn_fn + o * 16384).
//   - hc_attn_scale : fp32 [3]  (pre, post, comb scales)
//   - hc_attn_base  : fp32 [24] (pre[0..4], post[4..8], comb[8..24])
//
// Outputs are written without any concurrency between the three buffers:
// post_weights and comb are written once by lane 0 of the block; the row
// of `out` is written cooperatively by all threads.
//
// eps is the HC sinkhorn epsilon (DS4_RMS_EPS = 1e-6, also reused as the
// sinkhorn epsilon — ds4.c:4281 passes 1.0e-6f).
//
// n_embd/n_hc/sinkhorn_iters are passed as runtime parameters even though
// DS4 fixes them to 4096 / 4 / 20: the kernel is specialized for n_hc=4
// and n_embd a multiple of 4, but the iteration count is read from the
// model KV at session start.
void launch_hc_attn_pre_f32(const float    *residual_hc,    // [n_hc * n_embd]
                            const uint16_t *hc_attn_fn,     // f16 [24 * (n_hc*n_embd)]
                            const float    *hc_attn_scale,  // f32 [3]
                            const float    *hc_attn_base,   // f32 [24]
                            float          *post_weights,   // f32 [n_hc]
                            float          *comb,           // f32 [n_hc * n_hc]
                            float          *out,            // f32 [n_embd]
                            int             n_embd,
                            int             n_hc,
                            int             sinkhorn_iters,
                            float           eps,
                            cudaStream_t    stream = 0);

// HC sinkhorn multi-CTA: 3-kernel split of launch_hc_attn_pre_f32. Replaces
// the single-CTA serialization of the mix_len matvec rows with mix_len
// parallel blocks. Math is identical to v1; per-row mix output is byte-
// equal to v1.
//
// Adds two scratch pointers vs v1:
//   rms_scratch : float[1]                 (1/sqrt(mean+eps) handoff)
//   mix_scratch : float[n_hc * (2 + n_hc)] (24 fp32 for DS4)
//
// Both must be device-allocated and may live anywhere in the activation
// arena — they are written by hc_pre_rms_kernel + hc_pre_matvec_kernel
// and read by hc_pre_sinkhorn_finish_kernel inside this single launcher.
void launch_hc_attn_pre_v2_f32(const float    *residual_hc,
                               const uint16_t *hc_attn_fn,
                               const float    *hc_attn_scale,
                               const float    *hc_attn_base,
                               float          *post_weights,
                               float          *comb,
                               float          *out,
                               int             n_embd,
                               int             n_hc,
                               int             sinkhorn_iters,
                               float           eps,
                               float          *rms_scratch,
                               float          *mix_scratch,
                               cudaStream_t    stream = 0);

// FFN variant — same kernel chain, named separately so call sites read
// symmetrically with the v1 spelling (which uses launch_hc_attn_pre_f32
// for both attn and ffn). The shared chain is implemented under the
// attn name; this one forwards.
void launch_hc_ffn_pre_v2_f32(const float    *residual_hc,
                              const uint16_t *hc_ffn_fn,
                              const float    *hc_ffn_scale,
                              const float    *hc_ffn_base,
                              float          *post_weights,
                              float          *comb,
                              float          *out,
                              int             n_embd,
                              int             n_hc,
                              int             sinkhorn_iters,
                              float           eps,
                              float          *rms_scratch,
                              float          *mix_scratch,
                              cudaStream_t    stream = 0);

// HC post step for one sublayer output (one token, batch dim = 1). Mirrors
// ds4/ds4.c:4337 hc_post_one (cited):
//   for (dst = 0..n_hc):
//     for (d = 0..n_embd):
//       acc = block_out[d] * post[dst]
//       for (src = 0..n_hc):
//         acc += comb[dst + src * n_hc] * residual_hc[src * n_embd + d]
//       out_hc[dst * n_embd + d] = acc
//
// Layout (h-major, identical to launch_hc_attn_pre_f32):
//   - residual_hc : [n_hc, n_embd]   (h * n_embd + d)
//   - out_hc      : [n_hc, n_embd]   (h * n_embd + d)
//   - block_out   : [n_embd]
//   - post        : [n_hc]
//   - comb        : [n_hc * n_hc]    (dst + src * n_hc — comb is
//                                      addressed [dst_hc, src_hc] per
//                                      ds4.c:4350 comment "The HC combine
//                                      matrix is addressed as
//                                      [dst_hc, src_hc]")
//
// Stage coverage: `hc_attn_post` and `hc_ffn_post` (both same kernel,
// only the inputs differ). out_hc and residual_hc may not alias; block_out
// may safely overlap with neither. The kernel is specialized for n_hc=4
// (DS4_N_HC) but the launcher passes n_hc as a runtime parameter.
void launch_hc_post_f32(const float *block_out,    // [n_embd]
                        const float *residual_hc,  // [n_hc, n_embd]
                        const float *post,          // [n_hc]
                        const float *comb,          // [n_hc * n_hc]
                        float       *out_hc,        // [n_hc, n_embd]
                        int          n_hc,
                        int          n_embd,
                        cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_HC_SINKHORN_CUH
