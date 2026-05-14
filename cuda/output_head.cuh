// output_head.cuh — host-callable launcher for the final HC collapse
// (output_hc_head_one) kernel.
//
// Implementation: cuda/output_head.cu.
//
// Pipeline (matches ds4/ds4.c:8099 output_hc_head_one):
//
//     rms_norm_no_weight(flat, inp_hc, n_hc * n_embd, eps);
//     matvec_f16(pre, model, output_hc_fn, flat);   // F16 [hc_dim, n_hc]
//     for (h = 0..n_hc):
//         w[h] = sigmoid(pre[h] * scale[0] + base[h]) + eps;   // <-- scalar scale
//     hc_weighted_sum_one(out, inp_hc, w, n_embd, n_hc);
//
// Differences from hc_post_one (hc_post launcher):
//   - hc_post uses 4 stream-specific post[dst] scalars + a 4×4 comb matrix.
//   - output_hc_head_one uses a SINGLE scalar `scale[0]` broadcast across
//     all 4 HC streams (no per-stream scaling), a per-stream `base[h]`,
//     and NO comb matrix — the four streams are collapsed straight into
//     one fp32 [n_embd] via a weighted sum (not a 4x4 mix).
//   - The activation transform is sigmoid (not 2*sigmoid, not softmax),
//     and there's a `+ eps` floor before the weighted sum.
//
// This kernel is structurally similar to launch_hc_attn_pre_f32 but the
// post-matvec transform and final reduction differ enough that a fused
// stand-alone kernel is the cleanest mapping.  All five sub-steps (RMS sum,
// F16 matvec[4], sigmoid+eps, weighted sum) run inside one block.

#ifndef DS4CUDA_OUTPUT_HEAD_CUH
#define DS4CUDA_OUTPUT_HEAD_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ds4cuda {

// Final HC collapse for one token (batch=1).
//
// Inputs (device):
//   - residual_hc      : fp32 [n_hc, n_embd] row-major (h * n_embd + d).
//                        The model's last-layer hc_ffn_post output, also
//                        the input to the rms_norm step.
//   - output_hc_fn     : F16 [hc_dim_in=n_hc*n_embd, n_hc=4] row-major.
//                        On-disk dim[0]=hc_dim_in, dim[1]=n_hc, row o starts
//                        at output_hc_fn + o*hc_dim_in (matvec_f16
//                        convention from ds4.c:2740).
//   - output_hc_scale  : fp32 [1].  Single scalar broadcast across all 4
//                        streams (different from hc_attn_scale which has 3).
//   - output_hc_base   : fp32 [n_hc=4].
//
// Output (device):
//   - out              : fp32 [n_embd] — the final pre-output_norm vector.
//
// Numerical contract:
//   - RMSNorm: ss accumulated in fp32 inside the block; final scale =
//     1 / sqrtf(ss/N + eps).  ds4.c uses fp64 for ss; the typical drift is
//     sub-ULP at hc_dim=16384.  Test gates with rel/abs ULP-aware tolerances.
//   - F16 matvec: __half2float per element, fp32 accumulate per warp,
//     intra-warp butterfly reduction — same shape as launch_mul_mv_f16_f32
//     (ds4.c row order may differ by reduction-tree; tolerated under ULP).
//   - sigmoid: 1.0f / (1.0f + expf(-z)) — direct, no fastmath transform.
//   - hc_weighted_sum: per-d serial sum_{h=0..n_hc-1} w[h]*x[h,d] (fp32).
//
// eps is DS4_RMS_EPS = 1e-6 (used both inside the RMSNorm sqrt AND added
// to each `w[h]` per ds4.c:8116 — DS4_HC_EPS reuses the same constant).
//
// n_hc must be 4 (DS4_N_HC) and n_embd must be a multiple of 4 (DS4 uses
// 4096).  hc_dim_in = n_hc * n_embd.

void launch_output_hc_head_f32(const float    *residual_hc,    // fp32 [n_hc * n_embd]
                               const uint16_t *output_hc_fn,    // f16  [n_hc * (n_hc * n_embd)]
                               const float    *output_hc_scale, // fp32 [1]
                               const float    *output_hc_base,  // fp32 [n_hc]
                               float          *out,              // fp32 [n_embd]
                               int             n_embd,
                               int             n_hc,
                               float           eps,
                               cudaStream_t    stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_OUTPUT_HEAD_CUH
