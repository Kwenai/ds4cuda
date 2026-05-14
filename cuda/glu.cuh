// glu.cuh — host-callable launcher for the SwiGLU element-wise CUDA kernel.
//
// Implementation: cuda/glu.cu.
//
// Computes:
//     y[i] = silu(g[i]) * u[i] = g[i] * sigmoid(g[i]) * u[i]
// element-wise, matching ds4/ds4.c:4983 silu + ds4/ds4.c:4993 swiglu (which
// in turn delegates to ds4/ds4.c:4856 sigmoid_stable). The CPU reference
// uses the branchless-stable form
//     sigmoid(x) = (x>=0) ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))
// to avoid overflow at large negative x. This kernel mirrors that branch
// rather than using the naive 1/(1+exp(-x)), so |g| can be ~88 (the fp32
// expf overflow threshold) without producing inf/NaN.
//
// Used by stage `shared_silu_mul` (shared expert hidden, dim=2048).
// Also reusable for the routed-expert SwiGLU body in later milestones.

#ifndef DS4CUDA_GLU_CUH
#define DS4CUDA_GLU_CUH

#include <cuda_runtime.h>

namespace ds4cuda {

// Element-wise SwiGLU: y[i] = silu(g[i]) * u[i].
//
// All buffers are device fp32, length n. g, u, y may not alias.
// Caller picks block size internally; n need not be a multiple of any
// particular factor.
void launch_silu_mul_f32(const float *g, const float *u, float *y, int n,
                         cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_GLU_CUH
