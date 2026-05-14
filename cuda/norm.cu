// norm.cu — fp32 RMSNorm CUDA kernel for the `attn_norm` stage and
// later RMSNorm sites (ffn_norm, q_a_norm, kv_norm, etc.).
//
// Spec (ds4/ds4.c:2680 rms_norm_weight):
//     ss     = sum_{i=0..n-1} (double)x[i] * (double)x[i]
//     scale  = 1.0f / sqrtf((float)(ss / (double)n) + eps)
//     y[i]   = x[i] * scale * w[i]
//
// On the device we deliberately accumulate in fp32 (not fp64): GB10
// fp64 throughput is 1/64 of fp32, and the CPU's fp64-then-cast model
// only matters in pathological cancellation regimes that don't occur
// here (x is a residual stream activation, all components same sign
// magnitude). The expected mismatch vs the CPU dump is < 1e-5 relative
// (same tolerance gate as the Q8_0 dot product kernel in dense_q8.cu).
//
// Block layout:
//   - 1 CUDA block per row.
//   - 256 threads = 8 warps. Each thread handles n/256 = 16 elements
//     for the n=4096 attn_norm. (We support arbitrary n with a strided
//     loop so the same kernel covers q_a_norm n=1024 etc. later.)
//   - In-block reduction: warp-internal butterfly via __shfl_xor_sync,
//     then lane-0 of each warp writes to a 32-slot shared array, then
//     warp-0 reduces those 8 partials again via butterfly. Result is
//     broadcast to all threads via shared memory.
//
// Numerical contract:
//   - eps is added INSIDE the sqrt (matches CPU).
//   - 1.0f / sqrtf(...) is one fp32 sqrt + one fp32 reciprocal. Could
//     use rsqrtf for slightly better throughput, but stays with the
//     CPU sqrtf+divide form to avoid an extra ULP of drift.
//
// Reference: RMS_EPS = 1e-6 (ds4/ds4.c)

#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"
#include "norm.cuh"

namespace ds4cuda {

namespace {

constexpr int RMS_NORM_TPB    = 256;
constexpr int RMS_NORM_WARPS  = RMS_NORM_TPB / 32;   // 8

// Single-row kernel. One CUDA block computes one row.
//
// Grid is launched with grid.x=1 because the stage tests do
// per-row alignment; the matvec milestones will batch via grid.x =
// n_rows once we promote this kernel into a real residual-stream op.
__global__ void rms_norm_f32_kernel(const float *__restrict__ x,
                                    const float *__restrict__ w,
                                    float *__restrict__ y,
                                    int n,
                                    float eps)
{
    const int tid           = threadIdx.x;
    const int lane          = tid & 31;
    const int warp_in_block = tid >> 5;

    // ---- 1) per-thread sum of squares (fp32 accumulator) ----------
    float ss = 0.0f;
    for (int i = tid; i < n; i += RMS_NORM_TPB) {
        const float v = x[i];
        ss += v * v;
    }

    // ---- 2) intra-warp butterfly reduce ---------------------------
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    }
    // After this, every lane in the warp holds the warp's sum.

    // ---- 3) cross-warp reduce via shared memory -------------------
    __shared__ float warp_sums[RMS_NORM_WARPS];
    if (lane == 0) warp_sums[warp_in_block] = ss;
    __syncthreads();

    // Warp 0 reduces the per-warp partials. Only the first
    // RMS_NORM_WARPS lanes carry valid data; mask out the rest.
    float total = 0.0f;
    if (warp_in_block == 0) {
        total = (lane < RMS_NORM_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = RMS_NORM_WARPS / 2; off > 0; off >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, off);
        }
    }

    // Broadcast scale to every thread via shared memory slot 0.
    __shared__ float scale_s;
    if (tid == 0) {
        const float mean = total / (float)n;
        scale_s = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();
    const float scale = scale_s;

    // ---- 4) elementwise output y[i] = x[i] * scale * w[i] ---------
    for (int i = tid; i < n; i += RMS_NORM_TPB) {
        y[i] = x[i] * scale * w[i];
    }
}

} // namespace

void launch_rms_norm_f32(const float *x, const float *w, float *y, int n,
                         float eps, cudaStream_t stream)
{
    rms_norm_f32_kernel<<<1, RMS_NORM_TPB, 0, stream>>>(x, w, y, n, eps);
}

namespace {

// Batched RMSNorm. One CUDA block per row; threadIdx ranges over the
// columns of that row. Reduction logic is identical to the single-row
// kernel above; the only differences are:
//   - row stride: x and y are indexed via `blockIdx.x * n_per_row`,
//   - flexible weight: weight_dim selects unit-gain / shared / per-row
//     gain (matches the contract documented in norm.cuh).
//
// Used by Qnorm: 64 heads × 512 elems with weight_dim == 0 (unit gain
// per ds4/ds4.c:2689 head_rms_norm_inplace, no weight tensor).
__global__ void rms_norm_batch_f32_kernel(const float *__restrict__ x,
                                          const float *__restrict__ w,
                                          int weight_dim,
                                          float *__restrict__ y,
                                          int n_per_row,
                                          float eps)
{
    const int row           = blockIdx.x;
    const int tid           = threadIdx.x;
    const int lane          = tid & 31;
    const int warp_in_block = tid >> 5;

    const float *xr = x + (size_t)row * n_per_row;
    float       *yr = y + (size_t)row * n_per_row;

    // ---- 1) per-thread sum of squares ----------------------------
    float ss = 0.0f;
    for (int i = tid; i < n_per_row; i += RMS_NORM_TPB) {
        const float v = xr[i];
        ss += v * v;
    }

    // ---- 2) intra-warp butterfly reduce --------------------------
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    }

    // ---- 3) cross-warp reduce via shared memory ------------------
    __shared__ float warp_sums[RMS_NORM_WARPS];
    if (lane == 0) warp_sums[warp_in_block] = ss;
    __syncthreads();

    float total = 0.0f;
    if (warp_in_block == 0) {
        total = (lane < RMS_NORM_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = RMS_NORM_WARPS / 2; off > 0; off >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, off);
        }
    }

    __shared__ float scale_s;
    if (tid == 0) {
        const float mean = total / (float)n_per_row;
        scale_s = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();
    const float scale = scale_s;

    // ---- 4) elementwise output ----------------------------------
    // weight_dim chooses the gain layout. Branch is uniform across
    // the block so it does not divergence-cost; compiler can also
    // hoist the comparison out of the loop body.
    if (weight_dim == 0) {
        for (int i = tid; i < n_per_row; i += RMS_NORM_TPB) {
            yr[i] = xr[i] * scale;
        }
    } else if (weight_dim == n_per_row) {
        // Shared weight across rows: w[i].
        for (int i = tid; i < n_per_row; i += RMS_NORM_TPB) {
            yr[i] = xr[i] * scale * w[i];
        }
    } else {
        // Per-row weight: w[row, i]. weight_dim is asserted to be
        // n_per_row * n_rows by the host launcher.
        const float *wr = w + (size_t)row * n_per_row;
        for (int i = tid; i < n_per_row; i += RMS_NORM_TPB) {
            yr[i] = xr[i] * scale * wr[i];
        }
    }
}

} // namespace

void launch_rms_norm_batch_f32(const float *x, const float *w,
                               int weight_dim,
                               float *y,
                               int n_rows, int n_per_row,
                               float eps,
                               cudaStream_t stream)
{
    // weight_dim must be one of: 0 (unit), n_per_row (shared),
    // n_per_row*n_rows (per-row). The kernel's else-branch handles
    // the per-row case so any other value would silently mis-index;
    // the caller is contractually responsible for passing one of
    // these three values.
    rms_norm_batch_f32_kernel<<<n_rows, RMS_NORM_TPB, 0, stream>>>(
        x, w, weight_dim, y, n_per_row, eps);
}

} // namespace ds4cuda
