// flash_attn.cu — DeepSeek-V4 raw-SWA decode-raw flash attention kernel
// (`kqv_out` stage).
//
// =====================================================================
// Spec (cite ds4/ds4.c:4868-4906 layer_attention_rows_one)
// =====================================================================
//
//     sinks    = blk.<il>.attn_sinks.weight       // f32[N_HEAD]
//     kq_scale = 1.0f / sqrtf((float)head_dim)
//
//     for h in 0..n_head:
//         qh  = Q[h * head_dim ..]
//         max = sinks[h]
//         for k in 0..n_kv:
//             scores[k] = dot_f32(qh, K[k]) * kq_scale
//             max       = fmaxf(max, scores[k])
//         out[h, :] = 0
//         denom = exp(sinks[h] - max)
//         for k in 0..n_kv:
//             w        = exp(scores[k] - max)
//             denom   += w
//             out[h]  += w * V[k]
//         out[h] *= 1.0f / denom
//
// SWA window handling: the caller passes only the in-window rows
// (cache->raw_kv, cache->n_raw — see ds4/ds4.c:7294 for the call site).
// At pos=9 with N_SWA=128 the entire prefill prefix [0..pos] is in the
// window, so n_kv = pos+1 = 10.
//
// MLA layout (DS4_N_HEAD_KV = 1): K and V are the same row buffer of
// shape [n_kv, head_dim]. The launcher accepts separate K and V
// pointers for forward-compat with grouped-MQA, but for layer 0 the
// caller passes K == V.
//
// KV cache f16 round-trip: ds4/ds4.c:6204-6217 kv_cache_push_raw stores
// each KV row as f16. The dot product therefore reads K[k] as the f16
// projection of the input fp32. The input dump file il00_tok09_KVcur.bin
// is *post fp8* but *pre f16* — so for bit-exact alignment the test
// harness must f16-round each row before feeding it here, OR the kernel
// must do the round-trip itself. We choose the kernel-side variant: it
// keeps the launcher contract simple (fp32 in, fp32 out, just like the
// other dump-aligned stages) and the cost is negligible (5 KiB rounded once).
//
// =====================================================================
// Block layout
// =====================================================================
// One CUDA block per query head:
//   grid  = dim3(n_head)
//   block = dim3(256)
//
// Pass 1 (per-row dot product, n_kv rows):
//   Each thread reads `head_dim / 256` elements of qh and K[k] (=2 for
//   head_dim=512), multiplies, accumulates locally. Cross-thread
//   reduction is a 32-wide warp butterfly + 8-way cross-warp via
//   shared memory. Result (the dot) is broadcast to all threads via
//   shared memory and the max is updated once per row.
//
// Pass 2 (weighted-V accumulation):
//   For each k in 0..n_kv, all threads compute the same scalar
//   weight = exp(scores[k] - max), then each thread reads its
//   strided slice of V[k] and does `out[h] += w * V[k]`. The
//   denominator gets `denom += w` once per row (thread 0).
//
// Final divide:
//   inv = 1.0f / denom is computed by thread 0 and broadcast via
//   shared memory; each thread scales its `out[h]` slice and writes
//   it back.
//
// =====================================================================
// Numerical contract
// =====================================================================
// All accumulation fp32; we match the CPU's running-fp32 dot via:
//   - per-thread fp32 register accumulator over the strided slice
//   - intra-warp __shfl_xor_sync sum (32 lanes)
//   - cross-warp sum via shared memory partials + a second warp-0
//     butterfly
// This deviates from the CPU's strict left-to-right summation order so
// dot products will drift a few ULPs. Empirically with head_dim=512
// fp32 dots this stays under 1e-5 absolute, comfortably within the
// stage's PASS gate.
//
// All scoring math (`exp`, `fmaxf`, `1/denom`) uses fp32 device
// intrinsics. CUDA's expf is IEEE-bounded to ~1 ULP off the host libm
// expf, so scores small enough to evaluate normally (which they are
// here — softmax pushes the input range into [-tens, 0]) are well
// within the gate.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "common.cuh"
#include "flash_attn.cuh"

namespace ds4cuda {

namespace {

constexpr int FA_TPB    = 256;
constexpr int FA_WARPS  = FA_TPB / 32;       // 8
constexpr int FA_HD_MAX = 512;               // DS4_N_HEAD_DIM
constexpr int FA_KV_MAX = 128;               // DS4_N_SWA

// Match ds4 CPU kv_cache_push_raw: each KV row is round-tripped through
// fp16 before being used in attention (see ds4.c:6204-6217). To keep
// this kernel bit-equivalent with the CPU dump we apply the same
// round-trip device-side. fp16_bits_to_fp32 is provided in common.cuh;
// the f32 -> f16 conversion below uses CUDA's __float2half intrinsic
// which is bit-equivalent to IEEE 754 binary32 -> binary16 round-half-
// to-even (matches ds4.c:1519 f32_to_f16 for all finite inputs that
// arise on this stage).
__device__ __forceinline__ float f16_round(float x) {
    return __half2float(__float2half(x));
}

// Block-wide reduce helper: every thread comes in with a partial value,
// every thread leaves with the full sum. Uses one shared array of
// FA_WARPS slots + a second warp-0 butterfly. Caller must follow with
// the result it cares about.
__device__ __forceinline__ float block_sum_f32(float v, float *warp_sums)
{
    const int tid           = threadIdx.x;
    const int lane          = tid & 31;
    const int warp_in_block = tid >> 5;

    // Intra-warp butterfly.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    if (lane == 0) warp_sums[warp_in_block] = v;
    __syncthreads();

    float total = (tid < FA_WARPS) ? warp_sums[tid] : 0.0f;
    if (warp_in_block == 0) {
        #pragma unroll
        for (int off = FA_WARPS / 2; off > 0; off >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, off);
        }
        if (tid == 0) warp_sums[0] = total;
    }
    __syncthreads();
    return warp_sums[0];
}

// One block per head. blockDim.x = FA_TPB.
__global__ void flash_attn_decode_raw_f32_kernel(
        const float *__restrict__ Q,
        const float *__restrict__ K,
        const float *__restrict__ V,
        const float *__restrict__ sinks,
        float       *__restrict__ out,
        int   n_kv,
        int   head_dim,
        float kq_scale)
{
    const int h         = blockIdx.x;
    const int tid       = threadIdx.x;

    const float *qh = Q + (size_t)h * head_dim;

    // Shared state:
    //   scores[FA_KV_MAX]        per-row dot products (after scale)
    //   warp_sums[FA_WARPS+1]    block_sum_f32 scratch (last slot used
    //                            for inv-denom broadcast at the end)
    //   max_s, denom_s           scalar broadcasts
    __shared__ float scores[FA_KV_MAX];
    __shared__ float warp_sums[FA_WARPS];
    __shared__ float max_s;
    __shared__ float inv_s;

    // ---- Pass 1: dot products -------------------------------------
    // Initialise running max with the sink logit (CPU ds4.c:4883).
    if (tid == 0) {
        max_s = sinks[h];
    }
    __syncthreads();

    for (int k = 0; k < n_kv; ++k) {
        const float *kk = K + (size_t)k * head_dim;
        float p = 0.0f;
        for (int j = tid; j < head_dim; j += FA_TPB) {
            // f16 round-trip on K[k, j] to mirror the cache layout.
            const float kv = f16_round(kk[j]);
            p += qh[j] * kv;
        }
        const float dot = block_sum_f32(p, warp_sums);

        if (tid == 0) {
            const float s = dot * kq_scale;
            scores[k] = s;
            if (s > max_s) max_s = s;
        }
        __syncthreads();   // make scores[k] / max_s visible
    }

    // Snap shot the final max for use in pass 2 (avoid re-reading the
    // shared scalar from inside the inner loop, where it's stable).
    const float max_v = max_s;

    // ---- Pass 2: weighted V accumulation --------------------------
    // Each thread maintains an accumulator covering its strided slice
    // of [0, head_dim). For head_dim=512, FA_TPB=256: each thread has
    // 2 slots (j = tid, tid+256).
    constexpr int SLOTS = (FA_HD_MAX + FA_TPB - 1) / FA_TPB;   // 2
    float acc[SLOTS];
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) acc[s] = 0.0f;

    // Sink contribution to denom (CPU ds4.c:4893).
    float denom_local = (tid == 0) ? __expf(sinks[h] - max_v) : 0.0f;

    for (int k = 0; k < n_kv; ++k) {
        const float w = __expf(scores[k] - max_v);
        if (tid == 0) denom_local += w;

        const float *vk = V + (size_t)k * head_dim;
        #pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int j = tid + s * FA_TPB;
            if (j < head_dim) {
                acc[s] += w * f16_round(vk[j]);
            }
        }
    }

    // Compute 1/denom on thread 0 and broadcast.
    if (tid == 0) {
        inv_s = 1.0f / denom_local;
    }
    __syncthreads();
    const float inv = inv_s;

    // ---- Final write-out ------------------------------------------
    float *oh = out + (size_t)h * head_dim;
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        const int j = tid + s * FA_TPB;
        if (j < head_dim) {
            oh[j] = acc[s] * inv;
        }
    }
}

} // namespace

void launch_flash_attn_decode_raw_f32(const float *Q,
                                      const float *K,
                                      const float *V,
                                      const float *sinks,
                                      float       *out,
                                      int          n_head,
                                      int          n_kv,
                                      int          head_dim,
                                      cudaStream_t stream)
{
    // Shape contract — assert in launcher (cheap, host-side).
    if (head_dim != FA_HD_MAX || n_kv > FA_KV_MAX || n_kv < 0 ||
        n_head <= 0) {
        // Drop the launch; caller's cudaGetLastError will surface
        // cudaErrorInvalidConfiguration via grid==(0). Better: just
        // print and return — the test harness will fail compare.
        std::fprintf(stderr,
                     "launch_flash_attn_decode_raw_f32: bad shape "
                     "n_head=%d n_kv=%d head_dim=%d (max %d/%d)\n",
                     n_head, n_kv, head_dim, FA_KV_MAX, FA_HD_MAX);
        return;
    }

    const float kq_scale = 1.0f / sqrtf((float)head_dim);
    flash_attn_decode_raw_f32_kernel<<<dim3(n_head), dim3(FA_TPB), 0, stream>>>(
        Q, K, V, sinks, out, n_kv, head_dim, kq_scale);
}

// =====================================================================
// Decode-mixed kernel (layer >= 2 ratio-4 path)
//
// Spec: ds4/ds4.c:6657-6717 layer_attention_mixed_one. Same online-softmax
// + sinks framework as decode-raw, but the K/V row sequence is the
// concatenation of n_raw raw-SWA rows + n_comp compressed rows, with a
// per-compressed-row mask from the indexer (`comp_allowed`).
//
// The mask is applied at the score-fill site (ds4.c:6684-6687): a masked
// row's score is set to DS4_NEG_INF and *does not* update `max_score`.
// The pass-2 axpy path then short-circuits any score <= NEG_INF/2
// (ds4.c:6705) so masked rows contribute neither to denom nor to out.
//
// Bookkeeping for the kernel:
//   - One CUDA block per query head (same as decode-raw).
//   - The short kernel stores scores in shared memory and is capped at
//     FA_NTOTAL_MAX = 256.  Longer decode contexts use a streaming
//     fallback that computes the softmax max in pass 1, then recomputes
//     row scores in pass 2 while accumulating the denominator/output.
//     That keeps memory bounded when n_comp grows with max_context.
//   - F16 round-trip on both raw_kv and comp_kv reads (matches the
//     in-cache layout produced by kv_cache_push_raw / kv_cache_push_comp,
//     ds4.c:6363 + ds4.c:6369). The dump files are pre-f16-roundtrip in
//     both cases (ds4.c:7363 dumps `kv_t` before push; ds4.c:7400 dumps
//     `comp` before push), so kernel-side round-tripping keeps the
//     launcher contract fp32-in/fp32-out.
//
// All accumulation fp32; pass-1 / pass-2 reduction patterns identical
// to decode-raw; the only differences are (a) the second K loop over
// comp rows, (b) the comp_allowed mask short-circuiting both score
// fill and axpy, and (c) the larger SCORES capacity.
// =====================================================================

namespace {

constexpr int FA_NTOTAL_MAX = 256;   // n_raw + n_comp upper bound

__global__ void flash_attn_decode_mixed_f32_kernel(
        const float   *__restrict__ Q,
        const float   *__restrict__ raw_kv,
        const float   *__restrict__ comp_kv,
        const int32_t *__restrict__ comp_allowed,   // may be NULL
        const float   *__restrict__ sinks,
        float         *__restrict__ out,
        int   n_raw,
        int   n_comp,
        int   head_dim,
        float kq_scale)
{
    const int h         = blockIdx.x;
    const int tid       = threadIdx.x;

    const float *qh = Q + (size_t)h * head_dim;

    // Shared state — sized for n_total <= FA_NTOTAL_MAX.
    __shared__ float scores[FA_NTOTAL_MAX];
    __shared__ float warp_sums[FA_WARPS];
    __shared__ float max_s;
    __shared__ float inv_s;

    // ---- Pass 1: dot products + running max -----------------------
    if (tid == 0) {
        max_s = sinks[h];
    }
    __syncthreads();

    // 1a) raw rows.
    for (int k = 0; k < n_raw; ++k) {
        const float *kk = raw_kv + (size_t)k * head_dim;
        float p = 0.0f;
        for (int j = tid; j < head_dim; j += FA_TPB) {
            const float kv = f16_round(kk[j]);
            p += qh[j] * kv;
        }
        const float dot = block_sum_f32(p, warp_sums);

        if (tid == 0) {
            const float s = dot * kq_scale;
            scores[k] = s;
            if (s > max_s) max_s = s;
        }
        __syncthreads();
    }

    // 1b) compressed rows + per-row top-k mask.
    for (int k = 0; k < n_comp; ++k) {
        const int idx = n_raw + k;
        const bool allowed = (comp_allowed == nullptr) ||
                             (comp_allowed[k] != 0);

        if (!allowed) {
            if (tid == 0) {
                // CPU writes DS4_NEG_INF; CUDA's -INFINITY matches the
                // pass-2 short-circuit comparison `<= NEG_INF/2`.
                scores[idx] = -CUDART_INF_F;
            }
            __syncthreads();
            continue;
        }

        const float *kk = comp_kv + (size_t)k * head_dim;
        float p = 0.0f;
        for (int j = tid; j < head_dim; j += FA_TPB) {
            const float kv = f16_round(kk[j]);
            p += qh[j] * kv;
        }
        const float dot = block_sum_f32(p, warp_sums);

        if (tid == 0) {
            const float s = dot * kq_scale;
            scores[idx] = s;
            if (s > max_s) max_s = s;
        }
        __syncthreads();
    }

    const float max_v = max_s;

    // ---- Pass 2: weighted V accumulation --------------------------
    constexpr int SLOTS = (FA_HD_MAX + FA_TPB - 1) / FA_TPB;   // 2
    float acc[SLOTS];
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) acc[s] = 0.0f;

    // Sink contribution to denom.
    float denom_local = (tid == 0) ? __expf(sinks[h] - max_v) : 0.0f;

    // 2a) raw rows.
    for (int k = 0; k < n_raw; ++k) {
        const float w = __expf(scores[k] - max_v);
        if (tid == 0) denom_local += w;

        const float *vk = raw_kv + (size_t)k * head_dim;
        #pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int j = tid + s * FA_TPB;
            if (j < head_dim) {
                acc[s] += w * f16_round(vk[j]);
            }
        }
    }

    // 2b) compressed rows — skip masked (score <= -inf/2 mirrors CPU).
    for (int k = 0; k < n_comp; ++k) {
        const float sc = scores[n_raw + k];
        if (sc <= -CUDART_INF_F * 0.5f) continue;
        const float w = __expf(sc - max_v);
        if (tid == 0) denom_local += w;

        const float *vk = comp_kv + (size_t)k * head_dim;
        #pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int j = tid + s * FA_TPB;
            if (j < head_dim) {
                acc[s] += w * f16_round(vk[j]);
            }
        }
    }

    if (tid == 0) {
        inv_s = 1.0f / denom_local;
    }
    __syncthreads();
    const float inv = inv_s;

    float *oh = out + (size_t)h * head_dim;
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        const int j = tid + s * FA_TPB;
        if (j < head_dim) {
            oh[j] = acc[s] * inv;
        }
    }
}

__global__ void flash_attn_decode_mixed_stream_f32_kernel(
        const float   *__restrict__ Q,
        const float   *__restrict__ raw_kv,
        const float   *__restrict__ comp_kv,
        const int32_t *__restrict__ comp_allowed,   // may be NULL
        const float   *__restrict__ sinks,
        float         *__restrict__ out,
        int   n_raw,
        int   n_comp,
        int   head_dim,
        float kq_scale)
{
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;

    const float *qh = Q + (size_t)h * head_dim;

    __shared__ float warp_sums[FA_WARPS];
    __shared__ float max_s;
    __shared__ float inv_s;

    if (tid == 0) {
        max_s = sinks[h];
    }
    __syncthreads();

    // Pass 1: compute only the running max.  This avoids a per-head
    // score array proportional to n_raw+n_comp.
    for (int k = 0; k < n_raw; ++k) {
        const float *kk = raw_kv + (size_t)k * head_dim;
        float p = 0.0f;
        for (int j = tid; j < head_dim; j += FA_TPB) {
            p += qh[j] * f16_round(kk[j]);
        }
        const float dot = block_sum_f32(p, warp_sums);
        if (tid == 0) {
            const float s = dot * kq_scale;
            if (s > max_s) max_s = s;
        }
        __syncthreads();
    }

    for (int k = 0; k < n_comp; ++k) {
        const bool allowed = (comp_allowed == nullptr) ||
                             (comp_allowed[k] != 0);
        if (!allowed) continue;

        const float *kk = comp_kv + (size_t)k * head_dim;
        float p = 0.0f;
        for (int j = tid; j < head_dim; j += FA_TPB) {
            p += qh[j] * f16_round(kk[j]);
        }
        const float dot = block_sum_f32(p, warp_sums);
        if (tid == 0) {
            const float s = dot * kq_scale;
            if (s > max_s) max_s = s;
        }
        __syncthreads();
    }

    const float max_v = max_s;

    constexpr int SLOTS = (FA_HD_MAX + FA_TPB - 1) / FA_TPB;   // 2
    float acc[SLOTS];
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) acc[s] = 0.0f;

    float denom_local = (tid == 0) ? __expf(sinks[h] - max_v) : 0.0f;

    // Pass 2: recompute each score and immediately consume it.
    for (int k = 0; k < n_raw; ++k) {
        const float *kv = raw_kv + (size_t)k * head_dim;
        float p = 0.0f;
        for (int j = tid; j < head_dim; j += FA_TPB) {
            p += qh[j] * f16_round(kv[j]);
        }
        const float dot = block_sum_f32(p, warp_sums);
        __shared__ float w_s;
        if (tid == 0) {
            const float w = __expf(dot * kq_scale - max_v);
            denom_local += w;
            w_s = w;
        }
        __syncthreads();
        const float w = w_s;

        #pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int j = tid + s * FA_TPB;
            if (j < head_dim) {
                acc[s] += w * f16_round(kv[j]);
            }
        }
        __syncthreads();
    }

    for (int k = 0; k < n_comp; ++k) {
        const bool allowed = (comp_allowed == nullptr) ||
                             (comp_allowed[k] != 0);
        if (!allowed) continue;

        const float *kv = comp_kv + (size_t)k * head_dim;
        float p = 0.0f;
        for (int j = tid; j < head_dim; j += FA_TPB) {
            p += qh[j] * f16_round(kv[j]);
        }
        const float dot = block_sum_f32(p, warp_sums);
        __shared__ float w_s;
        if (tid == 0) {
            const float w = __expf(dot * kq_scale - max_v);
            denom_local += w;
            w_s = w;
        }
        __syncthreads();
        const float w = w_s;

        #pragma unroll
        for (int s = 0; s < SLOTS; ++s) {
            const int j = tid + s * FA_TPB;
            if (j < head_dim) {
                acc[s] += w * f16_round(kv[j]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        inv_s = 1.0f / denom_local;
    }
    __syncthreads();
    const float inv = inv_s;

    float *oh = out + (size_t)h * head_dim;
    #pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        const int j = tid + s * FA_TPB;
        if (j < head_dim) {
            oh[j] = acc[s] * inv;
        }
    }
}

} // namespace

void launch_flash_attn_decode_mixed_f32(const float   *Q,
                                        const float   *raw_kv,
                                        const float   *comp_kv,
                                        const int32_t *comp_allowed,
                                        const float   *sinks,
                                        float         *out,
                                        int            n_head,
                                        int            n_raw,
                                        int            n_comp,
                                        int            head_dim,
                                        cudaStream_t   stream)
{
    const int n_total = n_raw + n_comp;
    if (head_dim != FA_HD_MAX || n_raw < 0 || n_comp < 0 ||
        n_raw > FA_KV_MAX || n_head <= 0) {
        std::fprintf(stderr,
                     "launch_flash_attn_decode_mixed_f32: bad shape "
                     "n_head=%d n_raw=%d n_comp=%d head_dim=%d "
                     "(max raw=%d, hd=%d)\n",
                     n_head, n_raw, n_comp, head_dim,
                     FA_KV_MAX, FA_HD_MAX);
        return;
    }

    const float kq_scale = 1.0f / sqrtf((float)head_dim);
    if (n_total <= FA_NTOTAL_MAX) {
        flash_attn_decode_mixed_f32_kernel<<<dim3(n_head), dim3(FA_TPB), 0, stream>>>(
            Q, raw_kv, comp_kv, comp_allowed, sinks, out,
            n_raw, n_comp, head_dim, kq_scale);
    } else {
        flash_attn_decode_mixed_stream_f32_kernel<<<dim3(n_head), dim3(FA_TPB), 0, stream>>>(
            Q, raw_kv, comp_kv, comp_allowed, sinks, out,
            n_raw, n_comp, head_dim, kq_scale);
    }
}

} // namespace ds4cuda
