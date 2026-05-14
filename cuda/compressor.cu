// compressor.cu — DeepSeek-V4 streaming compressor decode-one step.
//
// Reference: ds4/ds4.c:6420-6471 (compressor_pool_decode_state) and
//            ds4/ds4.c:6475-6568 (compressor_decode_one).
//
// The launcher orchestrates a chain of small kernels + the existing
// launch_mul_mv_f16_f32 to mirror compressor_decode_one byte-for-byte:
//
//   1) kv_cur, sc_cur = matvec_f16(W_kv, x), matvec_f16(W_gate, x)
//        — reuse cuda/router.cu's launch_mul_mv_f16_f32.
//   2) sc_cur += APE bias (F16 -> F32 widened on the fly).
//   3) Write kv_cur and sc_cur into state row (compress_ratio + pos_mod).
//   4) If (pos+1) % compress_ratio == 0:
//        a) pool 8 rows -> head_dim_out via 4-way LSE-weighted average
//           with the pair-row "second-half" indexing per ds4.c:6432-6444.
//        b) RMSNorm: scale by 1/sqrt(mean(pooled^2)+eps), multiply by
//           norm[i].
//        c) tail-RoPE YaRN at comp_pos = pos+1-ratio (reuse rope.cu).
//        d) attn path only: fp8 E4M3FN round-trip on the non-RoPE prefix
//           (reuse fp8_kv.cu).
//        e) Ring-shift the state buffer (rows 4..7 -> 0..3, then dup
//           4..7 := 0..3) per ds4.c:6547-6563.
//
// Numerical notes:
//   - Pair matmul uses fp32 accumulation (already the case in
//     launch_mul_mv_f16_f32; warp-strided fp32 reduction).  A 1-ULP
//     drift vs the CPU's scalar dot_f16_row order is expected.
//   - APE F16 -> F32 conversion uses fp16_bits_to_fp32 (IEEE binary16
//     -> binary32 via __half2float), bit-exact with the CPU helper.
//   - pool_decode_state uses one CUDA block per output column (head_dim_out
//     blocks), small per-column reduction (4 or 8 input rows).  Result is
//     written to a fp32 [head_dim_out] scratch buffer that is then fed
//     into the RMSNorm / RoPE / fp8 chain.
//   - The state ring-shift is a single kernel that copies in place; ds4
//     does it in two memcpy passes (4..7 -> 0..3 then 0..3 -> 4..7).  We
//     do it as a single read-then-write per element with a syncthreads
//     barrier.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "common.cuh"
#include "compressor.cuh"
#include "fp8_kv.cuh"
#include "rope.cuh"
#include "router.cuh"

namespace ds4cuda {

namespace {

constexpr float COMP_NEG_INF = -1.0e30f;     // ds4.c:50 DS4_NEG_INF
constexpr float COMP_RMS_EPS =  1.0e-6f;     // ds4.c:52 DS4_RMS_EPS
constexpr int   COMP_N_ROT   = 64;           // ds4.c:90 DS4_N_ROT

// ---- Step 2 + 3: APE bias add + write state row ---------------------
// One block over `width` columns.  After this kernel:
//   sc_cur[j] += ape[pos_mod * width + j] (F16 -> F32 widened)
//   state_kv[(ratio+pos_mod) * width + j]    = kv_cur[j]
//   state_score[(ratio+pos_mod) * width + j] = sc_cur[j]
//
// Cite: ds4.c:6517-6522.
__global__ void ape_add_and_write_state_kernel(
    const float    *__restrict__ kv_cur,    // [width]
    float          *__restrict__ sc_cur,    // [width] in/out
    const uint16_t *__restrict__ ape_f16,   // [width * compress_ratio]
    float          *__restrict__ state_kv,
    float          *__restrict__ state_score,
    int             width,
    int             compress_ratio,
    int             pos_mod)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;

    const uint16_t ape_h = ape_f16[(size_t)pos_mod * (size_t)width + (size_t)j];
    const float    ape_v = fp16_bits_to_fp32(ape_h);

    const float kv_v = kv_cur[j];
    const float sc_v = sc_cur[j] + ape_v;

    // Row index per ds4.c:6492 — for ratio=4, row = 4 + pos_mod (slot 4..7).
    // For ratio=128, row = pos_mod (coff=1, only 128 rows).
    const int row = (compress_ratio == 4) ? (compress_ratio + pos_mod) : pos_mod;
    const size_t off = (size_t)row * (size_t)width + (size_t)j;
    state_kv[off]    = kv_v;
    state_score[off] = sc_v;
    sc_cur[j]        = sc_v;  // not strictly needed; kept for debug
}

// ---- Step 4a: pool_decode_state -------------------------------------
// One block per output column j (j in [0, head_dim)).
// For ratio=4 (coff=2):
//     for r in 0..4:
//        sp = state_score[r * width + j]                (first half row r)
//        sc = state_score[(4+r) * width + head_dim + j] (second half row 4+r)
//        track max
//     denom = sum exp(s - max), sum = sum exp(s - max) * kv
//     out[j] = sum / denom
// For ratio=128 (coff=1):
//     for r in 0..128:
//        s = state_score[r * width + j];
//        track max, then weighted average.
//
// Cite: ds4.c:6420-6471 (compressor_pool_decode_state).
__global__ void compressor_pool_decode_state_kernel(
    float       *__restrict__ pooled,       // [head_dim]
    const float *__restrict__ state_kv,
    const float *__restrict__ state_score,
    int          head_dim,
    int          compress_ratio)
{
    const int j = blockIdx.x;
    if (j >= head_dim) return;
    if (threadIdx.x != 0) return;     // small reduction, single-thread is fine

    const int coff  = (compress_ratio == 4) ? 2 : 1;
    const int width = coff * head_dim;

    // ---- max ----
    float max_score = COMP_NEG_INF;
    if (compress_ratio == 4) {
        for (int r = 0; r < 4; r++) {
            const float sp = state_score[(size_t)r           * width + j];
            const float sc = state_score[(size_t)(4 + r)     * width + head_dim + j];
            if (sp > max_score) max_score = sp;
            if (sc > max_score) max_score = sc;
        }
    } else {
        for (int r = 0; r < compress_ratio; r++) {
            const float s = state_score[(size_t)r * width + j];
            if (s > max_score) max_score = s;
        }
    }

    if (max_score <= COMP_NEG_INF * 0.5f) {
        pooled[j] = 0.0f;
        return;
    }

    // ---- weighted sum ----
    float denom = 0.0f;
    float sum   = 0.0f;
    if (compress_ratio == 4) {
        for (int r = 0; r < 4; r++) {
            const float sp = state_score[(size_t)r       * width + j];
            const float sc = state_score[(size_t)(4 + r) * width + head_dim + j];
            const float wp = expf(sp - max_score);
            const float wc = expf(sc - max_score);
            denom += wp + wc;
            sum   += wp * state_kv[(size_t)r       * width + j];
            sum   += wc * state_kv[(size_t)(4 + r) * width + head_dim + j];
        }
    } else {
        for (int r = 0; r < compress_ratio; r++) {
            const float s = state_score[(size_t)r * width + j];
            const float w = expf(s - max_score);
            denom += w;
            sum   += w * state_kv[(size_t)r * width + j];
        }
    }
    pooled[j] = (denom > 0.0f) ? (sum / denom) : 0.0f;
}

// ---- Step 4b: RMSNorm in-place on pooled vector ---------------------
// One block, 256 threads, fp32 reduction.  Mirrors norm.cu but writes
// in place (so the chain stays with one buffer through emit_out).
//
// Spec (ds4.c:6534-6539):
//     ss = sum_i pooled[i]^2 (fp64 accum on CPU; fp32 here for GB10 throughput)
//     rms = 1 / sqrt(ss / head_dim + RMS_EPS)
//     pooled[i] = pooled[i] * rms * norm[i]
__global__ void compressor_rmsnorm_inplace_kernel(
    float       *__restrict__ x,            // [head_dim] in/out
    const float *__restrict__ w_norm,
    int          head_dim,
    float        eps)
{
    constexpr int TPB = 256;
    constexpr int N_WARPS = TPB / 32;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += TPB) {
        const float v = x[i];
        ss += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    }
    __shared__ float warp_sums[N_WARPS];
    if (lane == 0) warp_sums[warp] = ss;
    __syncthreads();

    float total = 0.0f;
    if (warp == 0) {
        total = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = N_WARPS / 2; off > 0; off >>= 1) {
            total += __shfl_xor_sync(0xffffffff, total, off);
        }
    }
    __shared__ float scale_s;
    if (tid == 0) {
        const float mean = total / (float)head_dim;
        scale_s = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();
    const float scale = scale_s;

    for (int i = tid; i < head_dim; i += TPB) {
        x[i] = x[i] * scale * w_norm[i];
    }
}

// ---- Step 4e: ring-shift state buffer (ratio==4 only) ---------------
// Cite: ds4.c:6547-6563.  Net effect: rows 0..3 become a copy of the
// emit-time rows 4..7 (the post-update top-half), and rows 4..7 then
// get duplicated to match — i.e. all 8 rows hold the post-emit "carry"
// state.  We do it in-place with a single swap+dup pass over 4*width
// elements.  Per element:
//     old_top    = state[(4+r) * width + j]
//     state[r * width + j]     = old_top
//     state[(4+r) * width + j] = old_top
__global__ void compressor_ring_shift_kernel(
    float *__restrict__ state_kv,
    float *__restrict__ state_score,
    int    width)
{
    const int r = blockIdx.y;          // 0..3
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;

    const size_t top_off = (size_t)(4 + r) * (size_t)width + (size_t)j;
    const size_t bot_off = (size_t)r       * (size_t)width + (size_t)j;

    const float kv_top = state_kv[top_off];
    const float sc_top = state_score[top_off];

    state_kv[bot_off]    = kv_top;
    state_kv[top_off]    = kv_top;
    state_score[bot_off] = sc_top;
    state_score[top_off] = sc_top;
}

} // namespace

// ---------------------------------------------------------------------

void launch_compressor_decode_step_f32(
    bool             is_attn,
    const float     *x_f32,
    const uint16_t  *w_kv_f16,
    const uint16_t  *w_gate_f16,
    const uint16_t  *w_ape_f16,
    const float     *w_norm_f32,
    int              in_dim,
    int              head_dim_out,
    int              compress_ratio,
    int              pos,
    int              il,
    float           *state_kv,
    float           *state_score,
    float           *emit_out,
    int             *emitted_out_host,
    cudaStream_t     stream)
{
    const int coff      = (compress_ratio == 4) ? 2 : 1;
    const int width     = coff * head_dim_out;
    const int pos_mod   = pos % compress_ratio;
    const bool emit     = (((pos + 1) % compress_ratio) == 0);

    // ---- 1) Pair matvec (kv + gate) into temporary buffers ----------
    float *d_kv_cur = nullptr;
    float *d_sc_cur = nullptr;
    cudaMallocAsync(&d_kv_cur, (size_t)width * sizeof(float), stream);
    cudaMallocAsync(&d_sc_cur, (size_t)width * sizeof(float), stream);

    // launch_mul_mv_f16_f32 expects W [out_dim * in_dim] row-major (out_dim
    // rows of length in_dim).  matches matvec_f16 (ds4.c:2740) which reads
    // data[o*in_dim + i].  Here out_dim = width = coff*head_dim_out.
    launch_mul_mv_f16_f32(w_kv_f16,   x_f32, d_kv_cur, width, in_dim, stream);
    launch_mul_mv_f16_f32(w_gate_f16, x_f32, d_sc_cur, width, in_dim, stream);

    // ---- 2 + 3) APE bias add + write state row ---------------------
    {
        const int TPB = 128;
        const dim3 grid((width + TPB - 1) / TPB);
        const dim3 block(TPB);
        ape_add_and_write_state_kernel<<<grid, block, 0, stream>>>(
            d_kv_cur, d_sc_cur, w_ape_f16,
            state_kv, state_score,
            width, compress_ratio, pos_mod);
    }

    cudaFreeAsync(d_kv_cur, stream);
    cudaFreeAsync(d_sc_cur, stream);

    if (emitted_out_host) *emitted_out_host = emit ? 1 : 0;
    if (!emit) {
        return;
    }

    // ---- 4a) pool decode state -> emit_out --------------------------
    {
        const dim3 grid(head_dim_out);
        const dim3 block(1);          // single-thread per column reduction
        compressor_pool_decode_state_kernel<<<grid, block, 0, stream>>>(
            emit_out, state_kv, state_score, head_dim_out, compress_ratio);
    }

    // ---- 4b) RMSNorm in place ---------------------------------------
    {
        const dim3 grid(1);
        const dim3 block(256);
        compressor_rmsnorm_inplace_kernel<<<grid, block, 0, stream>>>(
            emit_out, w_norm_f32, head_dim_out, COMP_RMS_EPS);
    }

    // ---- 4c) tail-RoPE YaRN at comp_pos -----------------------------
    {
        const int comp_pos = pos + 1 - compress_ratio;
        launch_tail_rope_yarn_f32(emit_out, emit_out,
                                  /*n_heads*/1, head_dim_out, COMP_N_ROT,
                                  comp_pos, il, stream);
    }

    // ---- 4d) fp8 E4M3FN round-trip (attn path only) -----------------
    if (is_attn) {
        launch_fp8_kv_quantize_round_trip_f32(emit_out, emit_out,
                                              /*n_rows*/1,
                                              head_dim_out, COMP_N_ROT,
                                              stream);
    }

    // ---- 4e) ring-shift state (ratio==4) ----------------------------
    if (compress_ratio == 4) {
        const int TPB = 128;
        const dim3 grid((width + TPB - 1) / TPB, /*4 rows*/4);
        const dim3 block(TPB);
        compressor_ring_shift_kernel<<<grid, block, 0, stream>>>(
            state_kv, state_score, width);
    }
}

void launch_compressor_decode_step_f32_prealloc(
    bool             is_attn,
    const float     *x_f32,
    const uint16_t  *w_kv_f16,
    const uint16_t  *w_gate_f16,
    const uint16_t  *w_ape_f16,
    const float     *w_norm_f32,
    int              in_dim,
    int              head_dim_out,
    int              compress_ratio,
    int              pos,
    int              il,
    float           *state_kv,
    float           *state_score,
    float           *emit_out,
    int             *emitted_out_host,
    float           *scratch_kv_cur,
    float           *scratch_sc_cur,
    cudaStream_t     stream)
{
    const int coff      = (compress_ratio == 4) ? 2 : 1;
    const int width     = coff * head_dim_out;
    const int pos_mod   = pos % compress_ratio;
    const bool emit     = (((pos + 1) % compress_ratio) == 0);

    launch_mul_mv_f16_f32(w_kv_f16,   x_f32, scratch_kv_cur, width, in_dim, stream);
    launch_mul_mv_f16_f32(w_gate_f16, x_f32, scratch_sc_cur, width, in_dim, stream);

    {
        const int TPB = 128;
        const dim3 grid((width + TPB - 1) / TPB);
        const dim3 block(TPB);
        ape_add_and_write_state_kernel<<<grid, block, 0, stream>>>(
            scratch_kv_cur, scratch_sc_cur, w_ape_f16,
            state_kv, state_score,
            width, compress_ratio, pos_mod);
    }

    if (emitted_out_host) *emitted_out_host = emit ? 1 : 0;
    if (!emit) {
        return;
    }

    {
        const dim3 grid(head_dim_out);
        const dim3 block(1);
        compressor_pool_decode_state_kernel<<<grid, block, 0, stream>>>(
            emit_out, state_kv, state_score, head_dim_out, compress_ratio);
    }

    {
        const dim3 grid(1);
        const dim3 block(256);
        compressor_rmsnorm_inplace_kernel<<<grid, block, 0, stream>>>(
            emit_out, w_norm_f32, head_dim_out, COMP_RMS_EPS);
    }

    {
        const int comp_pos = pos + 1 - compress_ratio;
        launch_tail_rope_yarn_f32(emit_out, emit_out,
                                  /*n_heads*/1, head_dim_out, COMP_N_ROT,
                                  comp_pos, il, stream);
    }

    if (is_attn) {
        launch_fp8_kv_quantize_round_trip_f32(emit_out, emit_out,
                                              /*n_rows*/1,
                                              head_dim_out, COMP_N_ROT,
                                              stream);
    }

    if (compress_ratio == 4) {
        const int TPB = 128;
        const dim3 grid((width + TPB - 1) / TPB, /*4 rows*/4);
        const dim3 block(TPB);
        compressor_ring_shift_kernel<<<grid, block, 0, stream>>>(
            state_kv, state_score, width);
    }
}

} // namespace ds4cuda
