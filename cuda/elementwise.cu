// elementwise.cu — trivial pointwise CUDA kernels (currently: fp32 add).
//
// Stage coverage: `ffn_out` (shared + routed-expert combine,
// dim=4096, fp32). One thread per element; no reduction.
//
// ============================================================
// Spec (cited verbatim from ds4 CPU reference)
// ============================================================
//
//   ds4/ds4.c:5613 — layer_shared_ffn_one combine:
//     for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
//         ffn_out[i] = moe[i] + shared[i];
//     }
//
//   ds4/ds4.c:5702 — layer_shared_ffn_batch combine (per-token row):
//     scratch->ffn_out[i] = scratch->ffn_moe[i] + scratch->ffn_shared[i];
//
//   ds4/ds4.c:6027–6033 — debug-dump path that produces the
//   il00_tok09_ffn_out.bin reference: same `ffn_out = moe + shared` over
//   N_EMBD=4096.
//
// Numerical contract: bit-equal to the CPU reference for normal inputs
// (single fp32 add per element, no reduction). The element-wise alignment
// gate is therefore rel_tol=1e-5 with no ULP-aware fallback. 256 threads
// per block was picked to match cuda/glu.cu (8 blocks for the dim=2048
// SwiGLU; 16 blocks for dim=4096 here).

#include "elementwise.cuh"

#include <cuda_runtime.h>

namespace ds4cuda {

namespace {

__global__ void add_f32_kernel(const float *__restrict__ a,
                               const float *__restrict__ b,
                               float       *__restrict__ y,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

} // namespace

void launch_add_f32(const float *a, const float *b, float *y, int n,
                    cudaStream_t stream) {
    constexpr int BS = 256;
    const int grid = (n + BS - 1) / BS;
    add_f32_kernel<<<grid, BS, 0, stream>>>(a, b, y, n);
}

} // namespace ds4cuda
