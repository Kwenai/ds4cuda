// attn_out.cu — attention output projection chain (CUDA kernels).
//
// Two kernels:
//   1. tail_rope_yarn_inverse_f32_kernel — inverse-rotation tail RoPE
//      YaRN, same as the forward kernel in cuda/rope.cu but with
//      sin_sign = -1. Used by the kqv_back stage.
//   2. mul_mv_q8_0_q8_0_grouped_f32_kernel — Q8_0 × Q8_0 matvec where
//      each output row r reads x[g*group_dim..] with g = r/rank. Used
//      by the attn_low stage. Per-group activation quantization is
//      handled by quantize_grouped_split_kernel.
//
// References (full citations near each code block):
//   - ds4/ds4.c:4665-4713  rope_tail_ext_inplace (forward & inverse)
//   - ds4/ds4.c:3166-3179  matvec_q8_0_grouped_worker
//   - ds4/ds4.c:3516-3552  matvec_q8_0_grouped_rows (CPU launcher)
//   - ds4/ds4.c:3094       quantize_q8_0_activation (per-block scale)
//   - ds4/metal/moe.metal:842 kernel_dsv4_attn_out_low_q8_0_f32

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "attn_out.cuh"
#include "common.cuh"
#include "dense_q8.cuh"

namespace ds4cuda {

namespace {

// ---- Project-wide YaRN constants (cite ds4/ds4.c:52-61) -------------
constexpr float ROPE_FREQ_BASE          = 10000.0f;
constexpr float ROPE_SCALE_FACTOR       = 16.0f;
constexpr float ROPE_YARN_BETA_FAST     = 32.0f;
constexpr float ROPE_YARN_BETA_SLOW     = 1.0f;
constexpr float COMPRESS_ROPE_FREQ_BASE = 160000.0f;
constexpr int   ROPE_ORIG_CTX           = 65536;

__host__ static inline uint32_t layer_compress_ratio(int il) {
    if (il < 2) return 0u;
    return ((uint32_t)il & 1u) == 0u ? 4u : 128u;
}

__device__ __forceinline__ float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// Inverse tail-RoPE kernel. Mirrors tail_rope_yarn_f32_kernel in
// cuda/rope.cu byte-for-byte except the sin term is negated, matching
// ds4.c:4681 `sin_sign = inverse ? -1.0f : 1.0f` and ds4.c:4703
// `s = sin_sign * sinf(theta) * mscale`.
__global__ void tail_rope_yarn_inverse_f32_kernel(const float *__restrict__ x_in,
                                                  float *__restrict__       x_out,
                                                  int   head_dim,
                                                  int   n_rot,
                                                  int   pos,
                                                  float freq_base,
                                                  float freq_scale,
                                                  float ext_factor,
                                                  float attn_factor,
                                                  float corr_dim_low,
                                                  float corr_dim_high)
{
    const int head     = blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int tpb      = (int)blockDim.x;
    const int n_nope   = head_dim - n_rot;
    const int row_base = head * head_dim;

    // Pass-through prefix.
    for (int j = tid; j < n_nope; j += tpb) {
        x_out[row_base + j] = x_in[row_base + j];
    }

    // Per-pair rotation. theta_extrap is built via a `tid`-step running
    // product of theta_scale, identical to the forward kernel; this
    // keeps the fp32 reduction order matching the CPU `theta_extrap *=
    // theta_scale` walk.
    const int   i           = 2 * tid;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    float       theta_extrap = (float)pos;
    for (int k = 0; k < tid; ++k) {
        theta_extrap *= theta_scale;
    }
    const float theta_interp = freq_scale * theta_extrap;

    float theta  = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp(corr_dim_low, corr_dim_high, i) * ext_factor;
        theta  = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    const float c = cosf(theta) * mscale;
    // sin_sign = -1 for inverse rotation.
    const float s = -1.0f * sinf(theta) * mscale;

    const int idx0 = row_base + n_nope + i;
    const int idx1 = idx0 + 1;
    const float x0 = x_in[idx0];
    const float x1 = x_in[idx1];
    x_out[idx0] = x0 * c - x1 * s;
    x_out[idx1] = x0 * s + x1 * c;
}

// ---- Grouped activation quantization --------------------------------
//
// One warp per Q8_0 block. We treat the activation as a flat array of
// (n_groups * blocks_per_group) blocks; per-warp logic is identical to
// quantize_fp32_to_q8_0_split_kernel in dense_q8.cu — each group's
// scales are independent because the warp scope is one block (32
// elements, all in the same group).
//
// Cite: ds4/ds4.c:3094 quantize_q8_0_activation, called per group at
// ds4/ds4.c:3534-3537 inside matvec_q8_0_grouped_rows.
__global__ void quantize_grouped_split_kernel(const float *__restrict__ x,
                                              int8_t *__restrict__       xq,
                                              float  *__restrict__       xscale,
                                              int    n_blocks_total) {
    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int b             = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (b >= n_blocks_total) return;

    const float xv = x[b * 32 + lane];
    float ax = fabsf(xv);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ax = fmaxf(ax, __shfl_xor_sync(0xffffffff, ax, offset));
    }

    const float d  = ax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    int v = __float2int_rn(xv * id);
    if (v >  127) v =  127;
    if (v < -128) v = -128;

    xq[b * 32 + lane] = (int8_t)v;
    if (lane == 0) xscale[b] = d;
}

// ---- Grouped matvec kernel ------------------------------------------
//
// 1 warp per output row. For row r in [0, n_groups*rank):
//   group g       = r / rank
//   xq_base       = g * blocks_per_group * 32
//   xscale_base   = g * blocks_per_group
//   y[r]          = sum_b f16_to_f32(W[r,b].d) * xscale[g,b] *
//                       sum_i (W[r,b].qs[i] * xq[g,b,i])
//
// Cite: ds4/ds4.c:3166-3179 matvec_q8_0_grouped_worker (the CPU
// reference partitions output index `idx` into (group, row_in_group)
// then walks the matching xq/xscale slice).
__global__ void mul_mv_q8_0_q8_0_grouped_f32_kernel(const block_q8_0 *__restrict__ W,
                                                    const int8_t *__restrict__     xq,
                                                    const float  *__restrict__     xscale,
                                                    float        *__restrict__     y,
                                                    int n_groups,
                                                    int rank,
                                                    int blocks_per_group) {
    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int row           = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (row >= n_groups * rank) return;

    const int g = row / rank;
    const block_q8_0 *row_ptr = W + (size_t)row * blocks_per_group;
    const int8_t     *xq_g    = xq     + (size_t)g * blocks_per_group * 32;
    const float      *xscale_g = xscale + (size_t)g * blocks_per_group;

    float partial = 0.0f;
    for (int b = lane; b < blocks_per_group; b += 32) {
        const block_q8_0 &blk = row_ptr[b];
        const float dw = fp16_bits_to_fp32(blk.d);
        const float dx = xscale_g[b];
        const int8_t *__restrict__ wq = blk.qs;
        const int8_t *__restrict__ xb = xq_g + b * 32;

        int sum_i32 = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            sum_i32 += (int)wq[i] * (int)xb[i];
        }
        partial = __fmaf_rn(dw * dx, (float)sum_i32, partial);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    if (lane == 0) y[row] = partial;
}

} // namespace

// ---- Inverse RoPE launcher ------------------------------------------
//
// Same per-layer routing logic as launch_tail_rope_yarn_f32 in
// cuda/rope.cu. The routing is duplicated here (rather than extending
// rope.cu to take an inverse flag) to keep rope.cu and attn_out.cu
// independent.
void launch_tail_rope_yarn_inverse_f32(const float *x_in,
                                       float       *x_out,
                                       int          n_heads,
                                       int          head_dim,
                                       int          n_rot,
                                       int          pos,
                                       int          il,
                                       cudaStream_t stream)
{
    const uint32_t cratio    = layer_compress_ratio(il);
    const bool     compressed = (cratio != 0u);

    const float freq_base  = (compressed && COMPRESS_ROPE_FREQ_BASE > 0.0f)
                                 ? COMPRESS_ROPE_FREQ_BASE
                                 : ROPE_FREQ_BASE;
    const float freq_scale = (!compressed || ROPE_SCALE_FACTOR <= 0.0f)
                                 ? 1.0f
                                 : (1.0f / ROPE_SCALE_FACTOR);
    const float ext_factor = (compressed && ROPE_SCALE_FACTOR > 1.0f) ? 1.0f : 0.0f;

    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    float corr_low  = 0.0f;
    float corr_high = 0.0f;
    if (ext_factor != 0.0f) {
        const int   n_ctx_orig = ROPE_ORIG_CTX;
        const auto corr_dim = [&](float n_rot_target) -> float {
            return (float)n_rot * logf((float)n_ctx_orig / (n_rot_target * 2.0f * (float)CUDART_PI_F))
                 / (2.0f * logf(freq_base));
        };
        const float start = floorf(corr_dim(ROPE_YARN_BETA_FAST));
        const float end   = ceilf (corr_dim(ROPE_YARN_BETA_SLOW));
        corr_low  = fmaxf(0.0f, start);
        corr_high = fminf((float)(n_rot - 1), end);
    }

    const dim3 grid(n_heads);
    const dim3 block(n_rot / 2);
    tail_rope_yarn_inverse_f32_kernel<<<grid, block, 0, stream>>>(
        x_in, x_out, head_dim, n_rot, pos,
        freq_base, freq_scale, ext_factor, attn_factor,
        corr_low, corr_high);
}

// ---- Grouped Q8_0 matvec launcher -----------------------------------
void launch_mul_mv_q8_0_q8_0_grouped_f32(const block_q8_0 *W_dev,
                                         const float      *x_fp32_dev,
                                         float            *y_dev,
                                         int               n_groups,
                                         int               group_dim,
                                         int               rank,
                                         cudaStream_t      stream)
{
    // group_dim % 32 == 0 is required (Q8_0 blocks are 32 elements).
    const int blocks_per_group = group_dim / 32;
    const int n_blocks_total   = n_groups * blocks_per_group;
    const int n_rows           = n_groups * rank;

    int8_t *xq     = nullptr;
    float  *xscale = nullptr;
    cudaMallocAsync(&xq,     (size_t)n_blocks_total * 32, stream);
    cudaMallocAsync(&xscale, (size_t)n_blocks_total * sizeof(float), stream);

    {
        constexpr int TPB = 128;
        constexpr int WARPS_PER_CTA = TPB / 32;
        const int n_cta = (n_blocks_total + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        quantize_grouped_split_kernel<<<n_cta, TPB, 0, stream>>>(
            x_fp32_dev, xq, xscale, n_blocks_total);
    }

    {
        constexpr int TPB = 128;
        constexpr int WARPS_PER_CTA = TPB / 32;
        const int n_cta = (n_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        mul_mv_q8_0_q8_0_grouped_f32_kernel<<<n_cta, TPB, 0, stream>>>(
            W_dev, xq, xscale, y_dev, n_groups, rank, blocks_per_group);
    }

    cudaFreeAsync(xq, stream);
    cudaFreeAsync(xscale, stream);
}

void launch_mul_mv_q8_0_q8_0_grouped_f32_prealloc(
                                         const block_q8_0 *W_dev,
                                         const float      *x_fp32_dev,
                                         float            *y_dev,
                                         int               n_groups,
                                         int               group_dim,
                                         int               rank,
                                         int8_t           *scratch_xq_dev,
                                         float            *scratch_xscale_dev,
                                         cudaStream_t      stream)
{
    const int blocks_per_group = group_dim / 32;
    const int n_blocks_total   = n_groups * blocks_per_group;
    const int n_rows           = n_groups * rank;

    {
        constexpr int TPB = 128;
        constexpr int WARPS_PER_CTA = TPB / 32;
        const int n_cta = (n_blocks_total + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        quantize_grouped_split_kernel<<<n_cta, TPB, 0, stream>>>(
            x_fp32_dev, scratch_xq_dev, scratch_xscale_dev, n_blocks_total);
    }

    {
        constexpr int TPB = 128;
        constexpr int WARPS_PER_CTA = TPB / 32;
        const int n_cta = (n_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        mul_mv_q8_0_q8_0_grouped_f32_kernel<<<n_cta, TPB, 0, stream>>>(
            W_dev, scratch_xq_dev, scratch_xscale_dev,
            y_dev, n_groups, rank, blocks_per_group);
    }
}

} // namespace ds4cuda
