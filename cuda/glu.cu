// glu.cu — element-wise SwiGLU CUDA kernel for the shared-expert FFN body.
//
// Stage coverage: `shared_silu_mul` (shared expert hidden, dim=2048).
//
// ============================================================
// Spec (cited verbatim from ds4 CPU reference)
// ============================================================
//
//   ds4/ds4.c:4856 — sigmoid_stable:
//     static float sigmoid_stable(float x) {
//         if (x >= 0.0f) {
//             const float e = expf(-x);
//             return 1.0f / (1.0f + e);
//         } else {
//             const float e = expf(x);
//             return e / (1.0f + e);
//         }
//     }
//
//   ds4/ds4.c:4983 — silu:
//     static float silu(float x) {
//         return x * sigmoid_stable(x);
//     }
//
//   ds4/ds4.c:4993 — swiglu:
//     static void swiglu(float *out, const float *gate, const float *up,
//                        uint64_t n) {
//         for (uint64_t i = 0; i < n; i++) {
//             out[i] = silu(gate[i]) * up[i];
//         }
//     }
//
// Pointwise — no cross-element math, no reduction, no shared memory.
// One thread per element; 256 threads per block was picked to keep the
// 2048-element shared expert at exactly 8 blocks (one wave of 8 SMs on
// the consumer Blackwell sm_120 target). Larger n still works: grid
// stride is implicit in the launch sizing.
//
// Numerical contract: bit-equal to the CPU reference for normal inputs
// because we (a) use the same branched sigmoid_stable form, (b) call
// expf (which on CUDA dispatches to __expf for sm_120; we use the
// non-fast variant to keep IEEE-rounded results matching the host).
// The element-wise SwiGLU has no reduction tree, so no cross-impl drift
// is expected — gate is rel_tol=1e-5 (no ULP-aware fallback needed).

#include "glu.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace ds4cuda {

namespace {

// Branchless-stable sigmoid; matches ds4/ds4.c:4856 exactly.
__device__ __forceinline__ float sigmoid_stable_dev(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}

__global__ void silu_mul_f32_kernel(const float *__restrict__ g,
                                    const float *__restrict__ u,
                                    float *__restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i];
    const float ui = u[i];
    const float si = gi * sigmoid_stable_dev(gi);  // silu(gi)
    y[i] = si * ui;
}

} // namespace

void launch_silu_mul_f32(const float *g, const float *u, float *y, int n,
                         cudaStream_t stream) {
    constexpr int BS = 256;
    const int grid = (n + BS - 1) / BS;
    silu_mul_f32_kernel<<<grid, BS, 0, stream>>>(g, u, y, n);
}

} // namespace ds4cuda
