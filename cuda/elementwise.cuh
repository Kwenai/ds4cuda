// elementwise.cuh — host-callable launchers for trivial pointwise CUDA
// kernels that don't deserve a TU of their own.
//
// Implementation: cuda/elementwise.cu.
//
// Computes (so far):
//     y[i] = a[i] + b[i]
// element-wise, matching ds4/ds4.c:5613 / 5702
//     ffn_out[i] = moe[i] + shared[i]
// (the shared-expert-and-routed-expert combine that feeds hc_ffn_post).
//
// Pointwise — no cross-element math, no reduction. One thread per element.
// Bit-equal to the host reference (single fp32 add per element, no
// reduction tree drift) so the alignment gate is rel_tol=1e-5 with no
// ULP-aware fallback.

#ifndef DS4CUDA_ELEMENTWISE_CUH
#define DS4CUDA_ELEMENTWISE_CUH

#include <cuda_runtime.h>

namespace ds4cuda {

// Element-wise add: y[i] = a[i] + b[i]. All buffers are device fp32,
// length n. a, b, y may not alias. n need not be a multiple of any
// particular factor.
void launch_add_f32(const float *a, const float *b, float *y, int n,
                    cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_ELEMENTWISE_CUH
