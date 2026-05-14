// norm.cuh — host-callable launcher for the fp32 RMSNorm CUDA kernel.
//
// Implementation: cuda/norm.cu.
//
// Computes:
//     y[i] = (x[i] / sqrt(mean(x*x) + eps)) * w[i]
// for one row of n elements, matching ds4/ds4.c:2680 rms_norm_weight.
// One CUDA block processes one row; reduction is fp32 in-kernel (the
// CPU reference accumulates in fp64 then casts at the sqrt, so a small
// ULP-class diff is expected — see norm.cu comment).
//
// Used by stage `attn_norm` (and later ffn_norm / KV-norm / etc.).

#ifndef DS4CUDA_NORM_CUH
#define DS4CUDA_NORM_CUH

#include <cuda_runtime.h>

namespace ds4cuda {

// Single-row RMSNorm. n is the number of elements in the row (e.g.
// 4096 for attn_norm). eps is added inside the sqrt (DS4_RMS_EPS=1e-6).
// All buffers are device fp32.
void launch_rms_norm_f32(const float *x, const float *w, float *y, int n,
                         float eps, cudaStream_t stream = 0);

// Batched per-row RMSNorm. Each of `n_rows` rows of length `n_per_row`
// is normalized independently:
//
//     ss[r]    = sum_{i} x[r*n_per_row + i] * x[r*n_per_row + i]
//     scale[r] = 1 / sqrt(ss[r] / n_per_row + eps)
//     y[r,i]   = x[r,i] * scale[r] * w_eff[r,i]
//
// where w_eff depends on `weight_dim`:
//   weight_dim == 0                : unit gain (w ignored, may be NULL)
//   weight_dim == n_per_row        : w shared across rows  (w[i])
//   weight_dim == n_per_row*n_rows : per-row weight        (w[r,i])
//
// Used by stage `Qnorm` (per-head RMSNorm with no weight: 64 rows
// of 512 elements). Grid is `n_rows` blocks; each block reduces one
// row in fp32, mirroring launch_rms_norm_f32.
void launch_rms_norm_batch_f32(const float *x, const float *w,
                               int weight_dim,
                               float *y,
                               int n_rows, int n_per_row,
                               float eps,
                               cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_NORM_CUH
