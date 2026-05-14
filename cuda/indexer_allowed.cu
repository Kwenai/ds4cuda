// indexer_allowed.cu — indexer top-K allowed mask kernels.
//
// Two paths, both mirroring ds4.c:6926 indexer_allowed_decode_one:
//
//   1. short-circuit (n_comp <= top_k):  fill out[0..n_comp) = 1.
//   2. long-prompt   (n_comp >  top_k):  compute scores[c] over all
//      n_comp compressed rows, then mark the top-K by argmax.
//
// References: ds4/ds4.c:6900-6985 (CPU reference), see indexer_allowed.cuh
// for the long-form contract.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "common.cuh"
#include "indexer_allowed.cuh"

namespace ds4cuda {

namespace {

// ---------------------------------------------------------------------
// Short-circuit kernel: fill int32 [n_comp] with 1.
// ---------------------------------------------------------------------
__global__ void indexer_allowed_fill_one_kernel(int32_t *__restrict__ out,
                                                int n_comp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_comp) return;
    out[i] = 1;
}

// ---------------------------------------------------------------------
// In-place scale: y[i] *= s, used to bake the 1/sqrt(head_dim*n_head)
// factor into the per-head weights before scoring.
// ---------------------------------------------------------------------
__global__ void scale_inplace_f32_kernel(float *__restrict__ y, int n,
                                         float s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] *= s;
}

// ---------------------------------------------------------------------
// Score kernel: per compressed row c, compute
//
//   scores[c] = sum over h in [0, n_head):
//       w[h] * max(0, dot(q[h*head_dim .. (h+1)*head_dim],
//                          kv[c*head_dim .. (c+1)*head_dim]))
//
// Grid: n_comp blocks. Each block has n_head threads. Thread h owns the
// dot for one head — n_head=64 ⇒ exactly two warps per block. The
// reduction across n_head reuses warp shuffles + a tiny shared array.
//
// head_dim=128 is small enough that one thread doing the whole dot in
// fp32 is the cheapest scheme (no warp split, no shared-memory broadcast
// of q/kv). q is shared across all blocks, so the q reads coalesce; kv
// is read once per block from device memory.
//
// Numerical contract: each head's dot product accumulates left-to-right
// in fp32, ReLU is applied, and the n_head terms are reduced via warp
// shuffle. Up to ULP-class drift vs ds4.c:6957-6967 is expected — the
// indexer top-K mask is downstream tolerant of small score perturbations.
// ---------------------------------------------------------------------
__global__ void indexer_score_kernel(const float    *__restrict__ q,
                                     const float    *__restrict__ weights,
                                     const uint16_t *__restrict__ index_comp,
                                     float          *__restrict__ scores_out,
                                     int             n_comp,
                                     int             n_head,
                                     int             head_dim)
{
    const int c = blockIdx.x;
    if (c >= n_comp) return;

    const int h = threadIdx.x;     // 0..n_head-1
    const int tid_lane = h & 31;
    const int tid_warp = h >> 5;

    float partial = 0.0f;
    if (h < n_head) {
        const float    *qh = q          + (size_t)h * head_dim;
        const uint16_t *kv = index_comp + (size_t)c * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * fp16_bits_to_fp32(kv[d]);
        }
        if (dot < 0.0f) dot = 0.0f;   // ReLU
        partial = dot * weights[h];
    }

    constexpr int MAX_WARPS_FP16 = 8;
    __shared__ float warp_sums[MAX_WARPS_FP16];

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    }
    const int n_warps = (n_head + 31) >> 5;
    if (tid_lane == 0 && tid_warp < n_warps) warp_sums[tid_warp] = partial;
    __syncthreads();

    if (tid_warp == 0) {
        float v = (tid_lane < n_warps) ? warp_sums[tid_lane] : 0.0f;
        #pragma unroll
        for (int off = MAX_WARPS_FP16 / 2; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (tid_lane == 0) scores_out[c] = v;
    }
}

// FP32 KV variant. Identical to indexer_score_kernel above except for
// the kv dequant step (no fp16->fp32 cast). Used by the in-place CUDA
// forward path where index_comp_kv is fp32.
__global__ void indexer_score_kernel_fp32kv(const float *__restrict__ q,
                                            const float *__restrict__ weights,
                                            const float *__restrict__ index_comp,
                                            float       *__restrict__ scores_out,
                                            int          n_comp,
                                            int          n_head,
                                            int          head_dim)
{
    const int c = blockIdx.x;
    if (c >= n_comp) return;

    const int h = threadIdx.x;
    const int tid_lane = h & 31;
    const int tid_warp = h >> 5;

    float partial = 0.0f;
    if (h < n_head) {
        const float *qh = q          + (size_t)h * head_dim;
        const float *kv = index_comp + (size_t)c * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kv[d];
        }
        if (dot < 0.0f) dot = 0.0f;
        partial = dot * weights[h];
    }

    // Block-wide reduction over n_head threads. n_head expected to be
    // 64 (two warps); we still write the code for any multiple of 32
    // up to 32*MAX_WARPS so the kernel generalises in future.
    constexpr int MAX_WARPS = 8;  // up to 256 heads
    __shared__ float warp_sums[MAX_WARPS];

    // intra-warp butterfly
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    }
    const int n_warps = (n_head + 31) >> 5;
    if (tid_lane == 0 && tid_warp < n_warps) warp_sums[tid_warp] = partial;
    __syncthreads();

    if (tid_warp == 0) {
        float v = (tid_lane < n_warps) ? warp_sums[tid_lane] : 0.0f;
        #pragma unroll
        for (int off = MAX_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (tid_lane == 0) scores_out[c] = v;
    }
}

// ---------------------------------------------------------------------
// Top-K + mask emit kernel: single block, top_k iterations.
//
// Pre: scores[0..n_comp) populated, allowed[0..n_comp) is uninitialised.
// Post: allowed[c] = 1 for the top_k highest-scoring indices, 0 elsewhere.
//
// Algorithm:
//   step 0: every thread zeros a stripe of allowed[]
//   step 1: for k in [0, top_k):
//             warp-parallel reduce over (allowed[c]==0 ? scores[c] : -INF)
//             to find argmax (tid 0 holds the result).
//             tid 0 sets allowed[best] = 1.
//
// Block size 256 threads. n_comp up to ~33K per layer at max_context=131072
// (which we don't run in practice), so this is fine for an inner-loop
// kernel. For top_k=512, expected runtime is microseconds.
//
// Mathematical equivalence to ds4.c:6969-6979's serial argmax:
// arg-max with ties — ds4 picks the lowest c index on equal scores (its
// loop strict-greater than is initialised at -INF). The CUDA kernel
// uses the same strict-greater pairing during the warp/block reduction
// and breaks ties on the lowest c index; for finite floats this matches
// ds4 byte-for-byte. (Two terms produced by independent fp32 reductions
// can compare equal where ds4 would see a strictly-greater pair; in
// that case the tie-break picks the lowest c, which is the same as
// ds4's behaviour.)
// ---------------------------------------------------------------------
__global__ void indexer_topk_emit_kernel(const float *__restrict__ scores,
                                         int32_t     *__restrict__ allowed,
                                         int          n_comp,
                                         int          top_k)
{
    constexpr int TPB = 256;
    constexpr int N_WARPS = TPB / 32;
    const int tid       = threadIdx.x;
    const int tid_lane  = tid & 31;
    const int tid_warp  = tid >> 5;

    // ---- step 0: zero the allowed mask in parallel ------------------
    for (int i = tid; i < n_comp; i += TPB) {
        allowed[i] = 0;
    }
    __syncthreads();

    // ---- step 1: top_k iterations of argmax over (allowed==0) ------
    __shared__ float warp_best_v[N_WARPS];
    __shared__ int   warp_best_i[N_WARPS];
    __shared__ int   block_best_i;

    for (int k = 0; k < top_k; ++k) {
        float my_best_v = -INFINITY;
        int   my_best_i = -1;

        for (int c = tid; c < n_comp; c += TPB) {
            const int   a = allowed[c];
            const float s = scores[c];
            if (a == 0 && s > my_best_v) {
                my_best_v = s;
                my_best_i = c;
            }
        }

        // Intra-warp pairwise reduction by score, tie-break by lower index.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float other_v = __shfl_xor_sync(0xffffffff, my_best_v, off);
            const int   other_i = __shfl_xor_sync(0xffffffff, my_best_i, off);
            const bool take_other =
                (other_v > my_best_v) ||
                (other_v == my_best_v && other_i >= 0 &&
                 (my_best_i < 0 || other_i < my_best_i));
            if (take_other) {
                my_best_v = other_v;
                my_best_i = other_i;
            }
        }
        if (tid_lane == 0) {
            warp_best_v[tid_warp] = my_best_v;
            warp_best_i[tid_warp] = my_best_i;
        }
        __syncthreads();

        // Final reduction across warps in warp 0.
        if (tid_warp == 0) {
            float v = (tid_lane < N_WARPS) ? warp_best_v[tid_lane] : -INFINITY;
            int   i = (tid_lane < N_WARPS) ? warp_best_i[tid_lane] : -1;
            #pragma unroll
            for (int off = N_WARPS / 2; off > 0; off >>= 1) {
                const float other_v = __shfl_xor_sync(0xffffffff, v, off);
                const int   other_i = __shfl_xor_sync(0xffffffff, i, off);
                const bool take_other =
                    (other_v > v) ||
                    (other_v == v && other_i >= 0 &&
                     (i < 0 || other_i < i));
                if (take_other) {
                    v = other_v;
                    i = other_i;
                }
            }
            if (tid_lane == 0) {
                block_best_i = (i >= 0) ? i : 0;
                allowed[block_best_i] = 1;
            }
        }
        __syncthreads();
    }
}

} // namespace

void launch_indexer_allowed_short_circuit_i32(int32_t     *out_i32,
                                              int          n_comp,
                                              int          top_k,
                                              cudaStream_t stream)
{
    if (n_comp <= 0) return;

    if (n_comp > top_k) {
        // Caller must route long n_comp through the long-prompt path
        // (launch_indexer_score_topk_i32). Hard-fail here so a future
        // bug can't silently get the wrong mask.
        std::fprintf(stderr,
                     "indexer_allowed_short_circuit: n_comp=%d > top_k=%d "
                     "(use launch_indexer_score_topk_i32 instead)\n",
                     n_comp, top_k);
        std::abort();
    }

    const int TPB = 128;
    const dim3 grid((n_comp + TPB - 1) / TPB);
    const dim3 block(TPB);
    indexer_allowed_fill_one_kernel<<<grid, block, 0, stream>>>(out_i32, n_comp);
}

void launch_scale_inplace_f32(float       *y,
                              int          n,
                              float        scale,
                              cudaStream_t stream)
{
    if (n <= 0) return;
    const int TPB = 64;
    const dim3 grid((n + TPB - 1) / TPB);
    const dim3 block(TPB);
    scale_inplace_f32_kernel<<<grid, block, 0, stream>>>(y, n, scale);
}

void launch_indexer_score_topk_i32(const float    *q,
                                   const float    *weights,
                                   const uint16_t *index_comp,
                                   int             n_comp,
                                   int             n_head,
                                   int             head_dim,
                                   int             top_k,
                                   int32_t        *allowed_out,
                                   float          *scratch_scores,
                                   cudaStream_t    stream)
{
    if (n_comp <= 0) return;
    if (n_head <= 0 || head_dim <= 0) return;
    if (top_k <= 0) {
        // No allowed indices; just zero the mask.
        cudaMemsetAsync(allowed_out, 0, (size_t)n_comp * sizeof(int32_t), stream);
        return;
    }

    // Round up to a warp so the warp shuffles in the score kernel
    // see a well-defined active mask. n_head=64 hits this perfectly.
    int tpb = ((n_head + 31) / 32) * 32;
    if (tpb < 32) tpb = 32;
    if (tpb > 256) tpb = 256;
    indexer_score_kernel<<<n_comp, tpb, 0, stream>>>(
        q, weights, index_comp, scratch_scores, n_comp, n_head, head_dim);

    // Single-block top-K + mask emit. 256 threads, n_comp stripes.
    indexer_topk_emit_kernel<<<1, 256, 0, stream>>>(
        scratch_scores, allowed_out, n_comp, top_k);
}

void launch_indexer_score_topk_f32_i32(const float *q,
                                       const float *weights,
                                       const float *index_comp_f32,
                                       int          n_comp,
                                       int          n_head,
                                       int          head_dim,
                                       int          top_k,
                                       int32_t     *allowed_out,
                                       float       *scratch_scores,
                                       cudaStream_t stream)
{
    if (n_comp <= 0) return;
    if (n_head <= 0 || head_dim <= 0) return;
    if (top_k <= 0) {
        cudaMemsetAsync(allowed_out, 0, (size_t)n_comp * sizeof(int32_t), stream);
        return;
    }

    int tpb = ((n_head + 31) / 32) * 32;
    if (tpb < 32) tpb = 32;
    if (tpb > 256) tpb = 256;
    indexer_score_kernel_fp32kv<<<n_comp, tpb, 0, stream>>>(
        q, weights, index_comp_f32, scratch_scores, n_comp, n_head, head_dim);

    indexer_topk_emit_kernel<<<1, 256, 0, stream>>>(
        scratch_scores, allowed_out, n_comp, top_k);
}

} // namespace ds4cuda
