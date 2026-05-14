// router.cu — CUDA kernels backing the router stages (layer 0/1
// hash-routing fast path):
//
//   - mul_mv_f16_f32          (router_logits: F16 weight × fp32 act → fp32)
//   - sqrt_softplus_f32       (router_probs:  pointwise sqrt(softplus(x)))
//   - hash_router_topk_ids    (router_topk_ids: 6-elem int32 gather)
//   - hash_router_topk_w      (router_topk_w: normalize+scale 6 weights)
//
// ============================================================
// Spec citations (verbatim from ds4 CPU reference)
// ============================================================
//
//   ds4/ds4.c:2707-2727 — dot_f16_row (host fallback, scalar fp32 accum):
//     float acc = 0.0f;
//     for (uint64_t i = 0; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
//
//   ds4/ds4.c:2740 — matvec_f16:
//     in_dim = w->dim[0];  out_dim = w->dim[1];
//     out[o] = dot_f16_row(data + o*in_dim, x, in_dim);   for o in [0..out_dim)
//
//   ds4/ds4.c:4987 — softplus_stable:
//     if (x > 20.0f) return x;
//     if (x < -20.0f) return expf(x);
//     return log1pf(expf(x));
//
//   ds4/ds4.c:5118-5135 — layer_hash_selected_experts:
//     row = table + (uint64_t)token * DS4_N_EXPERT_USED;
//     for i in 0..6: selected[i] = row[i];
//
//   ds4/ds4.c:5153-5168 — layer_hash_router_weights_from_probs:
//     for i in 0..6: weights_out[i] = probs[selected[i]]; sum += weights_out[i];
//     if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
//     for i in 0..6: weights_out[i] = weights_out[i] / sum * 1.5f;
//
// Numerical contracts:
//   - F16 matvec is bit-close (warp-strided fp32 accumulation drifts vs.
//     scalar order at most ~1 ULP class on 4096-element rows; we use
//     the same ULP-aware test threshold as q_lora — REL_TOL=1e-4,
//     ABS_TOL=1e-5).
//   - sqrt_softplus is element-wise so it should be bit-equal modulo
//     log1pf/expf rounding (1 ULP). REL_TOL=1e-5 is comfortable.
//   - hash topk_ids is exact (just a memcpy of 6 int32 from a row).
//   - topk_w has only 6 fp32 sums + 6 fp32 divides + 6 fp32 muls in a
//     single warp, fp32-bit-stable ordering. REL_TOL=1e-5.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "router.cuh"

namespace ds4cuda {

namespace {

// ---------------------------------------------------------------------
// F16 weight × fp32 activation matvec.
//
// Layout: w_f16 row-major, row o at w_f16 + o*in_dim, length in_dim.
// One warp computes one output row.  Lanes 0..31 stride i by 32 and
// fp32-accumulate, then __shfl_xor_sync butterfly reduces.
//
// Block layout: 4 warps per block (128 threads), each warp handles one
// output row. Grid = ceil_div(out_dim, 4). Reasonable for out_dim=256
// (= 64 blocks) and the 4096-wide reduction (each lane does 128
// multiply-adds).
// ---------------------------------------------------------------------
__global__ void mul_mv_f16_f32_kernel(const uint16_t *__restrict__ w_f16,
                                      const float    *__restrict__ x_f32,
                                      float          *__restrict__ y_f32,
                                      int             out_dim,
                                      int             in_dim)
{
    const int warp_id_in_block = threadIdx.x >> 5;     // 0..3
    const int lane            = threadIdx.x & 31;
    const int row             = blockIdx.x * (blockDim.x >> 5) + warp_id_in_block;
    if (row >= out_dim) return;

    const uint16_t *wrow = w_f16 + (size_t)row * in_dim;

    float acc = 0.0f;
    for (int i = lane; i < in_dim; i += 32) {
        const float w = fp16_bits_to_fp32(wrow[i]);
        const float v = x_f32[i];
        acc += w * v;
    }

    // Warp-level butterfly reduction.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }

    if (lane == 0) y_f32[row] = acc;
}

// ---------------------------------------------------------------------
// router_probs: y[i] = sqrt(softplus_stable(x[i])).
//
// Pointwise. softplus_stable mirrors ds4.c:4987 branch for branch.
// ---------------------------------------------------------------------
__device__ __forceinline__ float softplus_stable_dev(float x) {
    if (x > 20.0f)  return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ void sqrt_softplus_f32_kernel(const float *__restrict__ x,
                                         float       *__restrict__ y,
                                         int          n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = sqrtf(softplus_stable_dev(x[i]));
}

// ---------------------------------------------------------------------
// router_topk_ids: 6-element int32 gather from a token-major hash table.
//
// On-disk layout per ds4.c:5134:
//     row = table + (uint64_t)token * DS4_N_EXPERT_USED;
// so the indexing is `table[token*6 + i]`.
//
// One block, one thread per output element. k <= 32 in practice.
// ---------------------------------------------------------------------
__global__ void hash_router_topk_ids_kernel(const int32_t *__restrict__ table,
                                            int            token_id,
                                            int32_t       *__restrict__ out,
                                            int            k)
{
    const int i = threadIdx.x;
    if (i >= k) return;
    out[i] = table[(size_t)token_id * (size_t)k + i];
}

// ---------------------------------------------------------------------
// router_topk_w: normalize+scale, single warp.
//
// w[i] = probs[selected[i]] / max(sum, sum_floor) * scale
// sum  = sum_{i in 0..k} probs[selected[i]]
//
// One block, k threads (k <= 32 in practice — DS4_N_EXPERT_USED=6).
// Uses warp shuffles to compute the sum.
// ---------------------------------------------------------------------
__global__ void hash_router_topk_w_kernel(const float   *__restrict__ probs,
                                          const int32_t *__restrict__ selected,
                                          float         *__restrict__ w_out,
                                          int            k,
                                          float          scale,
                                          float          sum_floor)
{
    const int i = threadIdx.x;

    // Per-thread partial: gather one element if in range, 0 otherwise.
    float v = 0.0f;
    int32_t idx = 0;
    if (i < k) {
        idx = selected[i];
        v   = probs[idx];
    }

    // Warp-wide sum (k <= 32, so a single warp covers all lanes).
    float sum = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }

    // Floor on the denominator (fp32 constant 6.103515625e-5f from
    // ds4.c:5164 — the fp16 min-normal limit reused as a divide guard).
    if (sum < sum_floor) sum = sum_floor;

    if (i < k) {
        w_out[i] = v / sum * scale;
    }
}

// ---------------------------------------------------------------------
// Top-k routing with exp_probs_b bias (layer 3..42).  Mirrors
// ds4.c:5217 layer_topk_selected_experts_from_probs.
//
// We run the host insertion-sort (ds4.c:5182 topk_desc) verbatim in a
// single thread inside one block so the selection is bit-identical to the
// CPU reference (this matters for tie-handling at exactly equal
// (prob+bias) values — deterministic order is "first index wins" because
// `score[i] > score[idx[j]]` is strict-greater).
//
// Then sum 6 unbiased probs across `n_used` lanes for the weight
// normalization.  All math is fp32; n_expert is fixed at 256 in DS4 but
// passed as a runtime parameter for completeness.
// ---------------------------------------------------------------------
__global__ void topk_selected_experts_kernel(const float   *__restrict__ router_probs,
                                             const float   *__restrict__ exp_probs_b,
                                             int32_t       *__restrict__ topk_ids_out,
                                             float         *__restrict__ topk_w_out,
                                             int            n_expert,
                                             int            n_used,
                                             float          weight_scale,
                                             float          sum_floor)
{
    // n_used is small (6); kept on stack via shared memory for sharing.
    extern __shared__ int32_t s_ids[];   // s_ids[0..n_used)

    const int tid = threadIdx.x;

    // ---- 1) Selection: thread 0 runs the insertion-sort ----
    // Uses scalar fp32 ops in the same order as ds4.c topk_desc + the
    // outer loop in layer_topk_selected_experts_from_probs (ds4.c:5217).
    if (tid == 0) {
        // Initialize selected[] to -1, mirroring topk_desc:5183.
        for (int j = 0; j < n_used; ++j) s_ids[j] = -1;

        // Single-pass insertion-sort over (probs + bias).
        // ds4.c reads `probs[i] + bias[i]` only inside the loop — we do
        // the same so the score is recomputed fresh per i (no cache).
        for (int i = 0; i < n_expert; ++i) {
            const float si = router_probs[i] + exp_probs_b[i];
            for (int j = 0; j < n_used; ++j) {
                const int32_t cur = s_ids[j];
                bool insert;
                if (cur < 0) {
                    insert = true;
                } else {
                    const float sc = router_probs[cur] + exp_probs_b[cur];
                    insert = si > sc;
                }
                if (insert) {
                    // Shift the tail right.
                    for (int m = n_used - 1; m > j; --m) s_ids[m] = s_ids[m - 1];
                    s_ids[j] = i;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // ---- 2) Weight normalization: read unbiased probs[selected[i]],
    // sum across 32 lanes, divide and scale.
    float v = 0.0f;
    int32_t idx = 0;
    if (tid < n_used) {
        idx = s_ids[tid];
        v   = router_probs[idx];
    }

    // Warp-wide sum (n_used <= 32 — DS4 uses 6).
    float sum = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    if (sum < sum_floor) sum = sum_floor;

    if (tid < n_used) {
        topk_ids_out[tid] = idx;
        topk_w_out[tid]   = v / sum * weight_scale;
    }
}

} // namespace

// =====================================================================
// Public launchers.
// =====================================================================

void launch_mul_mv_f16_f32(const uint16_t *w_f16,
                           const float    *x_f32,
                           float          *y_f32,
                           int             out_dim,
                           int             in_dim,
                           cudaStream_t    stream)
{
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int BS = WARPS_PER_BLOCK * 32;
    const int grid = (out_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    mul_mv_f16_f32_kernel<<<grid, BS, 0, stream>>>(w_f16, x_f32, y_f32,
                                                   out_dim, in_dim);
}

void launch_sqrt_softplus_f32(const float *x_f32,
                              float       *y_f32,
                              int          n,
                              cudaStream_t stream)
{
    constexpr int BS = 256;
    const int grid = (n + BS - 1) / BS;
    sqrt_softplus_f32_kernel<<<grid, BS, 0, stream>>>(x_f32, y_f32, n);
}

void launch_hash_router_topk_ids_i32(const int32_t *table_i32,
                                     int            token_id,
                                     int32_t       *out_i32,
                                     int            k,
                                     cudaStream_t   stream)
{
    // k tiny (<=32). One warp's worth of threads.
    const int bs = (k <= 32) ? 32 : k;
    hash_router_topk_ids_kernel<<<1, bs, 0, stream>>>(table_i32, token_id,
                                                      out_i32, k);
}

void launch_hash_router_topk_w_f32(const float   *probs_f32,
                                   const int32_t *selected_i32,
                                   float         *w_out_f32,
                                   int            k,
                                   float          scale,
                                   float          sum_floor,
                                   cudaStream_t   stream)
{
    // k <= 32 (DS4_N_EXPERT_USED = 6).  One warp.
    hash_router_topk_w_kernel<<<1, 32, 0, stream>>>(probs_f32, selected_i32,
                                                    w_out_f32, k, scale,
                                                    sum_floor);
}

void launch_topk_selected_experts_f32(const float   *router_probs,
                                      const float   *exp_probs_b,
                                      int32_t       *topk_ids_out,
                                      float         *topk_w_out,
                                      int            n_expert,
                                      int            n_used,
                                      float          weight_scale,
                                      float          sum_floor,
                                      cudaStream_t   stream)
{
    // One block, 32 threads (one warp). n_used <= 32. n_expert (256)
    // sets the inner-loop count, not the launch shape.  Shared mem holds
    // n_used int32 selected IDs — n_used*4 bytes (24 B for DS4).
    const size_t shmem = (size_t)n_used * sizeof(int32_t);
    topk_selected_experts_kernel<<<1, 32, shmem, stream>>>(router_probs,
                                                           exp_probs_b,
                                                           topk_ids_out,
                                                           topk_w_out,
                                                           n_expert,
                                                           n_used,
                                                           weight_scale,
                                                           sum_floor);
}

} // namespace ds4cuda
