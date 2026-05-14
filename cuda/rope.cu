// rope.cu — DeepSeek-V4 tail-RoPE YaRN CUDA kernel.
//
// Stage coverage: `Qcur` (Q upper after per-head norm, n_heads=64)
// and `KVrope` (KV upper, n_heads=1). Both call this same kernel with
// matching (head_dim=512, n_rot=64, il, pos); only n_heads differs.
//
// ============================================================
// Spec (cited verbatim from ds4 CPU + Metal references)
// ============================================================
//
//   ds4/ds4.c:52-61 — YaRN constants (project-wide):
//     #define DS4_RMS_EPS                 ( 1.0e-6f)
//     #define DS4_ROPE_FREQ_BASE          (10000.0f)
//     #define DS4_ROPE_SCALE_FACTOR       (16.0f)
//     #define DS4_ROPE_YARN_BETA_FAST     (32.0f)
//     #define DS4_ROPE_YARN_BETA_SLOW     (1.0f)
//     #define DS4_COMPRESS_ROPE_FREQ_BASE (160000.0f)
//     #define DS4_ROPE_ORIG_CTX           UINT64_C(65536)
//
//   ds4/ds4.c:407-411 — layer compress ratio:
//     if (il < 2) return 0;                       // layer 0,1 dense
//     return (il & 1u) == 0 ? 4u : 128u;          // even=ratio4, odd=ratio128
//
//   ds4/ds4.c:4646-4660 — YaRN helpers:
//     rope_yarn_ramp(low, high, i0)  = 1 - clamp((i0/2 - low)/(high-low), 0, 1)
//     rope_yarn_corr_dim(n_dims, n_ctx_orig, n_rot, base) =
//        n_dims * log(n_ctx_orig / (n_rot * 2pi)) / (2 * log(base))
//
//   ds4/ds4.c:4665-4713 — rope_tail_ext_inplace, the core math:
//     theta_scale  = freq_base ** (-2/n_rot)
//     theta_extrap = (float)pos                      // start at pos
//     for i = 0; i < n_rot; i += 2:
//        theta_interp = freq_scale * theta_extrap
//        theta        = theta_interp                 // ext_factor==0 path
//        mscale       = attn_factor
//        if ext_factor != 0:
//            ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i) * ext_factor
//            theta    = theta_interp*(1-ramp_mix) + theta_extrap*ramp_mix
//            mscale  *= 1 + 0.1 * log(1/freq_scale)
//        c = cosf(theta) * mscale
//        s = sin_sign * sinf(theta) * mscale         // sin_sign=+1 (forward)
//        x0 = tail[i]; x1 = tail[i+1]
//        tail[i]   = x0*c - x1*s
//        tail[i+1] = x0*s + x1*c
//        theta_extrap *= theta_scale
//
//   ds4/ds4.c:4729-4760 — per-layer routing:
//     compressed   = (compress_ratio(il) != 0)
//     freq_base    = compressed ? 160000.0f : 10000.0f
//     freq_scale   = compressed ? 1/16     : 1.0f
//     ext_factor   = (compressed && SCALE_FACTOR>1) ? 1.0f : 0.0f
//     attn_factor  = (ext_factor!=0) ? 1/(1 + 0.1*log(SCALE_FACTOR)) : 1.0f
//                    // CPU comment notes the helper applies its own mscale
//                    // pass-through scaling; ds4 cancels by prepre-dividing
//     n_ctx_orig   = compressed ? 65536 : 0   // unused when ext_factor==0
//     inverse      = false  (forward rotate; true is reserved for the
//                            attn-output reverse path elsewhere)
//
//   ds4/metal/dsv4_rope.metal:27-49 — Metal-side YaRN helper (sanity
//   check only; matches the CPU formula above). Note Metal also supports
//   freq_factor (per-dim mscale src2); ds4 CPU code does NOT use that
//   for the tail RoPE Q/KV path (line 4751 passes no equivalent), so we
//   skip it here too.
//
// ============================================================
// Layer 0 specialization
// ============================================================
// For the stage tests il=0 → compress_ratio=0 → ext_factor=0 →
// attn_factor=1, freq_scale=1, freq_base=10000. The kernel still
// implements the full YaRN ramp branch so it remains correct on
// compressed layers (il>=2) where the dump-driven test suite will
// later exercise the ext_factor!=0 path.
//
// ============================================================
// Block layout
// ============================================================
// One CUDA block per head. Within a block we launch n_rot/2 = 32
// threads — one per (i, i+1) dim-pair within the rotated tail. The
// non-rotated head_dim - n_rot = 448 prefix is copied through by the
// same threads in a strided loop (head_dim/threadsPerBlock = 14
// elements per thread).
//
// We pack 32 threads = exactly one warp; that keeps the math one-shot
// without a __syncthreads() barrier. Each thread:
//   - copies its share of the no-rot prefix (head_dim - n_rot = 448
//     elems / 32 threads = 14 each).
//   - computes one (i, i+1) RoPE rotation in the tail (i = 2*tid).
//
// Constants are baked into __constant__ memory so callers only pass
// (x_in, x_out, n_heads, head_dim, n_rot, pos, il, stream) per the
// design intent of a DeepSeek-V4-specific engine.
//
// Numerical notes:
//   - Use `sincosf(theta, &s, &c)` (intrinsic) to compute both in one
//     call. Both ds4 CPU and Metal use the IEEE library cosf+sinf, so
//     bit-identical results aren't expected, but `sincosf` shares the
//     reduction step and is within ~1 ULP of separate calls — well
//     under the 1e-6 absolute tolerance.
//   - All accumulation is fp32. The CPU also accumulates in fp32 here
//     (theta_extrap is float; theta_scale is float).

#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "common.cuh"
#include "rope.cuh"

namespace ds4cuda {

namespace {

// --- Project-wide YaRN constants (cite: ds4/ds4.c:52-61) ----------
constexpr float ROPE_FREQ_BASE          = 10000.0f;
constexpr float ROPE_SCALE_FACTOR       = 16.0f;
constexpr float ROPE_YARN_BETA_FAST     = 32.0f;
constexpr float ROPE_YARN_BETA_SLOW     = 1.0f;
constexpr float COMPRESS_ROPE_FREQ_BASE = 160000.0f;
constexpr int   ROPE_ORIG_CTX           = 65536;

// Compress ratio (cite: ds4/ds4.c:407-411). Encoded as a host inline
// helper used only at launch-time to derive per-layer parameters.
__host__ static inline uint32_t layer_compress_ratio(int il) {
    if (il < 2) return 0u;
    return ((uint32_t)il & 1u) == 0u ? 4u : 128u;
}

// rope_yarn_ramp — cite: ds4/ds4.c:4646-4649.
//   ramp(low, high, i0) = 1 - clamp((floor(i0/2) - low)/max(0.001, high-low), 0, 1)
__device__ __forceinline__ float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// One CUDA block = one head. blockDim.x = n_rot/2 (=32 = exactly one warp).
__global__ void tail_rope_yarn_f32_kernel(const float *__restrict__ x_in,
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
    const int tid      = (int)threadIdx.x;     // 0 .. n_rot/2 - 1 (=31)
    const int tpb      = (int)blockDim.x;      // n_rot/2
    const int n_nope   = head_dim - n_rot;     // 448

    const int row_base = head * head_dim;

    // ---- 1) copy through the no-rot prefix (n_nope elements) -----
    //   stride-tpb loop over [0, n_nope).
    for (int j = tid; j < n_nope; j += tpb) {
        x_out[row_base + j] = x_in[row_base + j];
    }

    // ---- 2) compute per-pair YaRN rotation in the tail -----------
    //   pair index i = 2*tid in [0, n_rot). The CPU code (ds4.c:4680,
    //   4710) computes a single fp32 `theta_scale = powf(freq_base,
    //   -2/n_rot)` then walks `theta_extrap *= theta_scale` once per
    //   pair. To stay bit-equivalent with that fp32 reduction tree
    //   (rather than calling powf with a different exponent per
    //   thread, which routes through libdevice's __log/__exp pair and
    //   drifts ~1 ULP per thread), each thread reproduces the CPU's
    //   sequential running product: do `tid` fp32 multiplications by
    //   theta_scale. That's at most 31 mults — negligible vs the
    //   sincosf+powf cost.
    const int   i           = 2 * tid;                 // 0,2,...,n_rot-2
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

    // Match the CPU reference (ds4.c:4702-4703) which calls cosf and
    // sinf separately. CUDA's sincosf shares argument reduction so it
    // can drift ~1 ULP from the IEEE-correct sinf/cosf pair on edge
    // cases — use the explicit pair for tighter parity.
    const float c = cosf(theta) * mscale;
    const float s = sinf(theta) * mscale;

    const int idx0 = row_base + n_nope + i;       // tail[i]
    const int idx1 = idx0 + 1;                    // tail[i+1]
    const float x0 = x_in[idx0];
    const float x1 = x_in[idx1];
    x_out[idx0] = x0 * c - x1 * s;
    x_out[idx1] = x0 * s + x1 * c;
}

} // namespace

void launch_tail_rope_yarn_f32(const float *x_in,
                               float       *x_out,
                               int          n_heads,
                               int          head_dim,
                               int          n_rot,
                               int          pos,
                               int          il,
                               cudaStream_t stream)
{
    // ---- Per-layer YaRN parameter routing ------------------------
    // Cite: ds4/ds4.c:4716-4760 (layer_rope_freq_base / freq_scale,
    // rope_tail_layer_inplace).
    const uint32_t cratio    = layer_compress_ratio(il);
    const bool     compressed = (cratio != 0u);

    const float freq_base  = (compressed && COMPRESS_ROPE_FREQ_BASE > 0.0f)
                                 ? COMPRESS_ROPE_FREQ_BASE
                                 : ROPE_FREQ_BASE;
    const float freq_scale = (!compressed || ROPE_SCALE_FACTOR <= 0.0f)
                                 ? 1.0f
                                 : (1.0f / ROPE_SCALE_FACTOR);
    const float ext_factor = (compressed && ROPE_SCALE_FACTOR > 1.0f) ? 1.0f : 0.0f;

    // attn_factor: see ds4.c:4741-4749. ds4's helper-cancellation
    // factor when ext_factor!=0; otherwise 1.0.
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    // YaRN correction-dim window. Only used when ext_factor!=0; we
    // still compute it (cheap; matches CPU fall-through).
    // Cite: ds4/ds4.c:4651-4660.
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

    // One block per head; n_rot/2 threads per block (== one warp at n_rot=64).
    const dim3 grid(n_heads);
    const dim3 block(n_rot / 2);
    tail_rope_yarn_f32_kernel<<<grid, block, 0, stream>>>(
        x_in, x_out, head_dim, n_rot, pos,
        freq_base, freq_scale, ext_factor, attn_factor,
        corr_low, corr_high);
}

} // namespace ds4cuda
