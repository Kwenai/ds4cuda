// output_head.cu — final HC collapse kernel.
//
// One block computes one token, mirroring ds4.c:8099 output_hc_head_one:
//
//   1) rms_norm_no_weight over the entire HC residual (n_hc * n_embd
//      = 16384 elements). Block-wide reduction in fp32.  ss/N inside sqrt.
//   2) F16 matvec output_hc_fn × flat → pre[n_hc] (n_hc=4 outputs).  One
//      warp per output row computes a fp32 dot of length hc_dim.  We have
//      8 warps per block; warps 0..3 each compute one row, warps 4..7 idle
//      for that step.
//   3) Per-h activation: w[h] = sigmoid(pre[h] * scale[0] + base[h]) + eps.
//      Lane 0 of warp 0 owns the broadcast.  Result lands in shared mem.
//   4) hc_weighted_sum: for each d, out[d] = sum_h w[h] * residual_hc[h,d].
//      All 256 threads stride d in lockstep; the inner sum is 4 fp32
//      multiply-adds.
//
// Numerical contract: see output_head.cuh.  The test gate is ULP-aware
// (rel ~ 1e-4, abs ~ 1e-5) — 16384-wide RMS reduction + 16384-wide F16
// matvec dot products amplify reduction-order drift, but stay sub-ULP
// at the magnitudes seen in practice (|pre| ~ O(1), |out[d]| < ~1).

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "output_head.cuh"

namespace ds4cuda {

namespace {

constexpr int OH_TPB    = 256;
constexpr int OH_WARPS  = OH_TPB / 32;   // 8
constexpr int OH_MAX_HC = 4;

__device__ __forceinline__ float oh_block_sum(float val,
                                              float (&warp_sums)[OH_WARPS],
                                              int tid_lane, int tid_warp)
{
    // intra-warp butterfly
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, off);
    }
    if (tid_lane == 0) warp_sums[tid_warp] = val;
    __syncthreads();

    float total = 0.0f;
    if (tid_warp == 0) {
        total = (tid_lane < OH_WARPS) ? warp_sums[tid_lane] : 0.0f;
        #pragma unroll
        for (int off = OH_WARPS / 2; off > 0; off >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, off);
        }
        if (tid_lane == 0) warp_sums[0] = total;
    }
    __syncthreads();
    return warp_sums[0];
}

__global__ void output_hc_head_kernel(const float    *__restrict__ residual_hc,
                                      const uint16_t *__restrict__ output_hc_fn,
                                      const float    *__restrict__ output_hc_scale,
                                      const float    *__restrict__ output_hc_base,
                                      float          *__restrict__ out,
                                      int             n_embd,
                                      int             n_hc,
                                      float           eps)
{
    const int tid       = threadIdx.x;
    const int tid_lane  = tid & 31;
    const int tid_warp  = tid >> 5;
    const int hc_dim    = n_hc * n_embd;

    __shared__ float warp_sums[OH_WARPS];
    __shared__ float s_scale;
    __shared__ float s_base[OH_MAX_HC];
    __shared__ float s_pre[OH_MAX_HC];
    __shared__ float s_w[OH_MAX_HC];

    // ---- 1) RMSNorm (no weight) over the entire HC residual ----
    float ss = 0.0f;
    for (int i = tid; i < hc_dim; i += OH_TPB) {
        const float v = residual_hc[i];
        ss += v * v;
    }
    ss = oh_block_sum(ss, warp_sums, tid_lane, tid_warp);
    const float rms_scale = 1.0f / sqrtf(ss / (float)hc_dim + eps);

    if (tid == 0) {
        s_scale = output_hc_scale[0];
    }
    if (tid < n_hc) {
        s_base[tid] = output_hc_base[tid];
    }
    __syncthreads();

    // ---- 2) F16 matvec: pre[h] = dot(output_hc_fn[h, :], flat) ----
    // flat[i] = residual_hc[i] * rms_scale (computed implicitly inside the dot).
    // Warp `tid_warp` handles row `tid_warp` (idle if tid_warp >= n_hc).
    if (tid_warp < n_hc) {
        const uint16_t *wrow = output_hc_fn + (size_t)tid_warp * (size_t)hc_dim;
        float acc = 0.0f;
        for (int i = tid_lane; i < hc_dim; i += 32) {
            const float w = fp16_bits_to_fp32(wrow[i]);
            const float v = residual_hc[i] * rms_scale;
            acc += w * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, off);
        }
        if (tid_lane == 0) s_pre[tid_warp] = acc;
    }
    __syncthreads();

    // ---- 3) Per-h activation: w[h] = sigmoid(pre[h]*scale + base[h]) + eps ----
    if (tid < n_hc) {
        const float z = s_pre[tid] * s_scale + s_base[tid];
        const float sig = 1.0f / (1.0f + expf(-z));
        s_w[tid] = sig + eps;
    }
    __syncthreads();

    // ---- 4) hc_weighted_sum_one: out[d] = sum_h w[h] * residual_hc[h, d] ----
    // Specialized for n_hc = 4 (DS4_MAX_HC) — unrolled accumulation.
    const float w0 = s_w[0];
    const float w1 = s_w[1];
    const float w2 = s_w[2];
    const float w3 = s_w[3];

    const float *r0 = residual_hc + (size_t)0 * n_embd;
    const float *r1 = residual_hc + (size_t)1 * n_embd;
    const float *r2 = residual_hc + (size_t)2 * n_embd;
    const float *r3 = residual_hc + (size_t)3 * n_embd;

    for (int d = tid; d < n_embd; d += OH_TPB) {
        // ds4.c:4244 hc_weighted_sum_one does scalar fp32 accum:
        //   acc = 0; for h in 0..n_hc: acc += x[h*n_embd+d] * weights[h];
        // Strict left-to-right at n_hc=4.
        float acc = 0.0f;
        acc += r0[d] * w0;
        acc += r1[d] * w1;
        acc += r2[d] * w2;
        acc += r3[d] * w3;
        out[d] = acc;
    }
}

} // namespace

void launch_output_hc_head_f32(const float    *residual_hc,
                               const uint16_t *output_hc_fn,
                               const float    *output_hc_scale,
                               const float    *output_hc_base,
                               float          *out,
                               int             n_embd,
                               int             n_hc,
                               float           eps,
                               cudaStream_t    stream)
{
    // n_hc must be 4 (DS4_N_HC), n_embd a multiple of 4 (DS4 uses 4096).
    // One block per token (batch=1).
    output_hc_head_kernel<<<1, OH_TPB, 0, stream>>>(residual_hc,
                                                    output_hc_fn,
                                                    output_hc_scale,
                                                    output_hc_base,
                                                    out,
                                                    n_embd,
                                                    n_hc,
                                                    eps);
}

} // namespace ds4cuda
