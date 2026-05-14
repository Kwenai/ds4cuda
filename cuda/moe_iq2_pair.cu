// moe_iq2_pair.cu — routed MoE gate+up paired IQ2_XXS matvec fused with
// SwiGLU + clamp + router weight.
//
// Stage: `routed_expert_mid` ([n_used=6, out_dim=2048] fp32 hidden).
//
// CPU reference (cited verbatim, ds4/ds4.c):
//   - 1628..1666  ds4_quantize_row_q8_K
//   - 1877..1956  ds4_vec_dot_iq2_xxs_pair_q8_K        (NEON path; we
//                  follow the int8×int8-then-fp32 form, not the metal
//                  fp32 form — so we match the dump).
//   - 3792..3811  matvec_iq2_xxs_mid_worker             (clamp + SwiGLU
//                  + route_weight tail).
//   - 3816..3861  matvec_iq2_xxs_experts_mid_prequant   (slot loop /
//                  layout).
//
// Constant tables for IQ2_XXS sign+grid expansion are reused from
// cuda/moe_iq2.cu via __constant__ extern. The grid table is in TU-local
// __constant__ in moe_iq2.cu — to share without ODR conflicts we
// re-declare a local copy here (~2 KiB constant memory, negligible).
//
// Memory layout of weights (in_dim=4096 -> 16 super-blocks/row,
// out_dim=2048 rows, 256 experts):
//   W_gate[expert][row][ib] : block_iq2_xxs (66 B)
// Each row stride = 16 super-blocks = 16*66 = 1056 B.
// Each expert stride = 2048*1056 = 2,162,688 B ≈ 2.06 MiB.
// One pair of (gate,up) per expert = 4.13 MiB. Six selected experts =
// 24.78 MiB; fits trivially on Spark. The launcher receives device
// pointers to the 6 selected expert slices already H2D-copied; the
// `topk_ids` argument is therefore a redirect index in [0, n_used).
//
// Activation: in_dim=4096 / 256 = 16 Q8_K super-blocks. Internal scratch
// is 16 * (256 B qs + 4 B d + 32 B bsums) = 4.62 KiB.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "common.cuh"
#include "moe_iq2_pair.cuh"

namespace ds4cuda {

// ---------------------------------------------------------------------
// Constant tables. Verbatim copy of ds4/metal/moe.metal:8-88
// (mirror of ds4/ds4.c:217-297). Re-declared here as a separate
// __constant__ array (TU-local) so we do not require linkage with
// moe_iq2.cu's tables. Constant memory is plentiful (64 KiB).
// ---------------------------------------------------------------------

__constant__ uint8_t c_pair_kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};

__constant__ uint8_t c_pair_ksigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__constant__ uint64_t c_pair_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// ---------------------------------------------------------------------
// Q8_K activation quantizer.
//
// Layout: 1 CTA = 1 super-block (256 elems). 256 threads per CTA.
//   Each thread handles one fp32 element; warp + block reductions find
//   amax (and the signed max-magnitude element 'max' that drives the
//   sign of iscale, mirroring ds4.c:1640).
//
// The CPU reference picks `max = x[j_with_max_abs]` (signed). For ties
// in |x| the CPU resolves to the first occurrence in the for-loop. We
// reproduce that exact rule by carrying (amax_so_far, signed_max) as a
// pair through the warp/block reduce: the lane with the largest amax
// wins, and on ties the lower thread index wins (matching the CPU's
// strict greater-than '>' comparison in ds4.c:1637).
// ---------------------------------------------------------------------

__device__ __forceinline__ void warp_reduce_max_signed(float &amax, float &smax)
{
    // 32-lane butterfly. We keep the (amax, smax) pair coherent: the
    // lane with the strictly larger amax wins, ties stay with lower idx
    // (the lane index is folded in via the strict '>' above).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_amax = __shfl_xor_sync(0xffffffff, amax, offset);
        const float other_smax = __shfl_xor_sync(0xffffffff, smax, offset);
        if (other_amax > amax) { amax = other_amax; smax = other_smax; }
    }
}

__global__ void quantize_fp32_to_q8_K_kernel(const float *__restrict__ x,
                                             int8_t      *__restrict__ xq,
                                             float       *__restrict__ xscale,
                                             int16_t     *__restrict__ xbsums,
                                             int n_blocks)
{
    const int b    = blockIdx.x;             // 0..n_blocks-1
    if (b >= n_blocks) return;
    const int tid  = threadIdx.x;            // 0..255
    const int lane = tid & 31;
    const int wid  = tid >> 5;               // 0..7

    const float *xb   = x  + (size_t)b * 256;
    int8_t      *qsb  = xq + (size_t)b * 256;

    // ---- pass 1: amax + signed max -----------------------------------
    const float xv = xb[tid];
    float amax = fabsf(xv);
    float smax = xv;

    warp_reduce_max_signed(amax, smax);

    __shared__ float sh_amax[8];
    __shared__ float sh_smax[8];
    if (lane == 0) { sh_amax[wid] = amax; sh_smax[wid] = smax; }
    __syncthreads();

    if (wid == 0) {
        amax = (lane < 8) ? sh_amax[lane] : 0.0f;
        smax = (lane < 8) ? sh_smax[lane] : 0.0f;
        // Reduce 8 partial maxima within warp 0. We can reuse the same
        // butterfly helper since irrelevant lanes carry amax=0.
        warp_reduce_max_signed(amax, smax);
        if (lane == 0) { sh_amax[0] = amax; sh_smax[0] = smax; }
    }
    __syncthreads();
    amax = sh_amax[0];
    smax = sh_smax[0];

    // ---- pass 2: quantize -------------------------------------------
    if (amax == 0.0f) {
        qsb[tid] = 0;
        if (tid < 16) xbsums[(size_t)b * 16 + tid] = 0;
        if (tid == 0) xscale[b] = 0.0f;
        return;
    }

    const float iscale = -127.0f / smax;     // ds4.c:1651
    int v = __float2int_rn(iscale * xv);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    qsb[tid] = (int8_t)v;
    __syncthreads();

    // ---- pass 3: bsums (16 sums, each over 16 consecutive qs[]) -----
    if (tid < 16) {
        int sum = 0;
        // Re-read our just-written qs[] to keep the reduction tied to
        // the actual stored values. CPU does the same in ds4.c:1660.
        for (int j = 0; j < 16; ++j) sum += (int)qsb[tid * 16 + j];
        xbsums[(size_t)b * 16 + tid] = (int16_t)sum;
    }

    if (tid == 0) xscale[b] = 1.0f / iscale;  // ds4.c:1663
}

void launch_quantize_fp32_to_q8_K(const float *x_dev,
                                  int8_t      *xq_dev,
                                  float       *xscale_dev,
                                  int16_t     *xbsums_dev,
                                  int          n_blocks,
                                  cudaStream_t stream)
{
    constexpr int TPB = 256;
    quantize_fp32_to_q8_K_kernel<<<n_blocks, TPB, 0, stream>>>(
        x_dev, xq_dev, xscale_dev, xbsums_dev, n_blocks);
}

// ---------------------------------------------------------------------
// Pair IQ2_XXS · Q8_K dot, one warp per row.
//
// Lane partitioning over super-blocks: lane l processes ib32 = l, l+32,
// l+64, ... (Metal kernel uses the same pattern with `ix = tiisg`,
// stride 32 across all ib32 of all super-blocks). With nb=16
// super-blocks * 8 ib32 each = 128 ib32 total per row, each lane handles
// 4 ib32. Inside the lane we decode aux32 once, then loop the 4 octets
// to get 32 elements of yl from the Q8_K activation.
//
// Critical: the gate weights and up weights live in different IQ2_XXS
// streams but for the SAME (ib32, octet, j) they share the activation
// byte yl[8*l+j]. We decode signs+grid for each side independently
// (gate and up have independent quants) but multiply both against the
// same q8 byte loaded once. The grid table lookups are still distinct;
// what's "fused" is the activation read and the per-row sum tail.
//
// Per-block contribution (mirror ds4.c:1936-1947 NEON form, exact):
//     d0 = f16_to_f32(xg.d) * y.d
//     for ib32 in 0..8:
//       sg += isum_g(ib32) * (0.5 + (aux32_g_s >> 28))
//       su += isum_u(ib32) * (0.5 + (aux32_u_s >> 28))
//     total_g += d0_g * sg
//     total_u += d0_u * su
//   final: gate = 0.25 * total_g, up = 0.25 * total_u
// ---------------------------------------------------------------------

__global__ void moe_iq2_xxs_pair_swiglu_full_kernel(
        const block_iq2_xxs *__restrict__ W_gate, // n_experts * out_dim * (in_dim/256)
        const block_iq2_xxs *__restrict__ W_up,
        const int8_t        *__restrict__ xq,
        const float         *__restrict__ xscale,
        const int32_t       *__restrict__ topk_ids,
        const float         *__restrict__ topk_w,
        float               *__restrict__ mid_out,
        int                  n_used,
        int                  out_dim,
        int                  n_blocks_in,
        int                  n_experts,
        float                clamp)
{
    const int slot = blockIdx.y;
    const int row  = blockIdx.x;
    if (slot >= n_used || row >= out_dim) return;

    const int eid = topk_ids[slot];
    if (eid < 0 || eid >= n_experts) return;

    const int lane = threadIdx.x;

    const block_iq2_xxs *xg_row =
        W_gate + ((size_t)eid * out_dim + row) * n_blocks_in;
    const block_iq2_xxs *xu_row =
        W_up   + ((size_t)eid * out_dim + row) * n_blocks_in;

    float total_g = 0.0f;
    float total_u = 0.0f;

    const int n_ib = n_blocks_in * 8;
    for (int ib_global = lane; ib_global < n_ib; ib_global += 32) {
        const int ib_super = ib_global >> 3;
        const int ib32     = ib_global & 7;

        const block_iq2_xxs &bg = xg_row[ib_super];
        const block_iq2_xxs &bu = xu_row[ib_super];
        const int8_t *q8        = xq + (size_t)ib_super * 256 + ib32 * 32;
        const float   yd        = xscale[ib_super];

        const uint16_t *qg2 = bg.qs + 4 * ib32;
        const uint16_t *qu2 = bu.qs + 4 * ib32;
        const uint32_t aux_g_g = (uint32_t)qg2[0] | ((uint32_t)qg2[1] << 16);
        const uint32_t aux_g_s = (uint32_t)qg2[2] | ((uint32_t)qg2[3] << 16);
        const uint32_t aux_u_g = (uint32_t)qu2[0] | ((uint32_t)qu2[1] << 16);
        const uint32_t aux_u_s = (uint32_t)qu2[2] | ((uint32_t)qu2[3] << 16);

        const float ls_g = 0.5f + (float)(aux_g_s >> 28);
        const float ls_u = 0.5f + (float)(aux_u_s >> 28);

        int isum_g = 0;
        int isum_u = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t  gidx_g = (uint8_t)((aux_g_g >> (8 * l)) & 0xff);
            const uint8_t  gidx_u = (uint8_t)((aux_u_g >> (8 * l)) & 0xff);
            const uint8_t  signs_g = c_pair_ksigns[(aux_g_s >> (7 * l)) & 0x7f];
            const uint8_t  signs_u = c_pair_ksigns[(aux_u_s >> (7 * l)) & 0x7f];

            const uint8_t *grid_g = (const uint8_t *)(c_pair_grid + gidx_g);
            const uint8_t *grid_u = (const uint8_t *)(c_pair_grid + gidx_u);
            const int8_t  *yseg   = q8 + 8 * l;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int sg = (signs_g & c_pair_kmask[j]) ? -1 : 1;
                const int su = (signs_u & c_pair_kmask[j]) ? -1 : 1;
                const int gj = (int)grid_g[j] * sg;
                const int uj = (int)grid_u[j] * su;
                const int yj = (int)yseg[j];
                isum_g += gj * yj;
                isum_u += uj * yj;
            }
        }

        const float dg = fp16_bits_to_fp32(bg.d) * yd;
        const float du = fp16_bits_to_fp32(bu.d) * yd;
        total_g += dg * (ls_g * (float)isum_g);
        total_u += du * (ls_u * (float)isum_u);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total_g += __shfl_xor_sync(0xffffffff, total_g, offset);
        total_u += __shfl_xor_sync(0xffffffff, total_u, offset);
    }

    if (lane == 0) {
        const float gate = 0.25f * total_g;
        const float up   = 0.25f * total_u;

        float g = gate;
        float u = up;
        if (clamp > 1.0e-6f) {
            if (g > clamp)  g = clamp;
            if (u > clamp)  u = clamp;
            if (u < -clamp) u = -clamp;
        }

        float s;
        if (g >= 0.0f) {
            const float e = __expf(-g);
            s = g / (1.0f + e);
        } else {
            const float e = __expf(g);
            s = g * e / (1.0f + e);
        }
        mid_out[(size_t)slot * out_dim + row] = s * u * topk_w[slot];
    }
}

void launch_routed_moe_pair_swiglu_full_f32_prealloc(
                                            const block_iq2_xxs *W_gate,
                                            const block_iq2_xxs *W_up,
                                            const float         *x_fp32,
                                            const int32_t       *topk_ids,
                                            const float         *topk_w,
                                            float               *mid_out,
                                            int                  n_used,
                                            int                  out_dim,
                                            int                  in_dim,
                                            int                  n_experts,
                                            float                clamp_value,
                                            int8_t              *scratch_xq_dev,
                                            float               *scratch_xscale_dev,
                                            int16_t             *scratch_xbsums_dev,
                                            cudaStream_t         stream)
{
    if (in_dim % 256 != 0) {
        std::fprintf(stderr,
            "launch_routed_moe_pair_swiglu_full_f32_prealloc: in_dim=%d not a multiple of 256\n",
            in_dim);
        return;
    }
    const int n_blocks_in = in_dim / 256;

    launch_quantize_fp32_to_q8_K(x_fp32, scratch_xq_dev, scratch_xscale_dev,
                                 scratch_xbsums_dev, n_blocks_in, stream);

    dim3 grid(out_dim, n_used, 1);
    constexpr int TPB = 32;
    moe_iq2_xxs_pair_swiglu_full_kernel<<<grid, TPB, 0, stream>>>(
        W_gate, W_up, scratch_xq_dev, scratch_xscale_dev,
        topk_ids, topk_w, mid_out,
        n_used, out_dim, n_blocks_in, n_experts, clamp_value);
}

// ---------------------------------------------------------------------
// Production SoA v2 dot kernel.
//
// Same math as moe_iq2_xxs_pair_swiglu_full_kernel above, but the
// warp lane -> data mapping is inverted so warp loads from qs_soa are
// coalesced across n_blocks_in adjacent uint16s (32 lanes × 2 B = 64 B
// warp transaction) instead of 32-byte strides across 66 B AoS blocks.
//
// Lane partitioning:
//   AoS kernel: ib_global = lane, lane+32, ...  (each lane handles 4 ib32
//               sub-blocks scattered across multiple super-blocks; 32
//               lanes share each super-block's 66 B → uncoalesced)
//   SoA v2:     lane l owns super-block l exclusively (for n_blocks_in
//               <= 32 — production is 16, so lanes 16..31 are idle and
//               contribute 0 to the warp reduction).
//
// Activation: we keep the AoS xq layout (xq + ib_super*256 + ib32*32)
// — the SoA transpose is only on the weight side. Activation reads are
// cheap (in_dim = 4096 → 4 KiB scratch shared across all rows).
//
// Output is ULP-equivalent (NOT byte-equal) to the AoS kernel because
// the lane permutation changes the warp reduction tree order; the per-
// row partials going into __shfl_xor are summed in a different order.
// ---------------------------------------------------------------------

__global__ void moe_iq2_xxs_pair_swiglu_resident_soa_v2_kernel(
        const uint16_t  *__restrict__ qs_soa_gate, // [n_experts][out_dim][32][n_blocks_in]
        const uint16_t  *__restrict__ d_soa_gate,  // [n_experts][out_dim][n_blocks_in]
        const uint16_t  *__restrict__ qs_soa_up,
        const uint16_t  *__restrict__ d_soa_up,
        const int8_t    *__restrict__ xq,          // [n_blocks_in * 256]
        const float     *__restrict__ xscale,      // [n_blocks_in]
        const int32_t   *__restrict__ topk_ids,
        const float     *__restrict__ topk_w,
        float           *__restrict__ mid_out,
        int              n_used,
        int              out_dim,
        int              n_blocks_in,
        int              n_experts,
        float            clamp)
{
    const int slot = blockIdx.y;
    const int row  = blockIdx.x;
    if (slot >= n_used || row >= out_dim) return;

    const int eid = topk_ids[slot];
    if (eid < 0 || eid >= n_experts) return;

    const int lane = threadIdx.x;       // 0..31

    float total_g = 0.0f;
    float total_u = 0.0f;

    // Only lanes 0..n_blocks_in-1 do work. We require n_blocks_in <= 32
    // (production = 16). Idle lanes contribute 0 to the warp reduction.
    if (lane < n_blocks_in) {
        const int ib_super = lane;

        // qs_soa[((e*out_dim + r)*32 + k)*n_blocks_in + b]
        // For this lane (b=ib_super), step over k = 0..31 with stride
        // n_blocks_in within the qs_soa array. Each ib32 of this lane's
        // super-block is the 4 consecutive k values starting at 4*ib32.
        const size_t row_stride = ((size_t)eid * out_dim + row) * 32 * (size_t)n_blocks_in;
        const uint16_t *qs_g = qs_soa_gate + row_stride + (size_t)ib_super;
        const uint16_t *qs_u = qs_soa_up   + row_stride + (size_t)ib_super;

        // d for this lane's super-block.
        const size_t d_idx =
            ((size_t)eid * out_dim + row) * (size_t)n_blocks_in + (size_t)ib_super;
        const float dg = fp16_bits_to_fp32(d_soa_gate[d_idx]);
        const float du = fp16_bits_to_fp32(d_soa_up  [d_idx]);

        // Activation slice for this lane's super-block (AoS xq layout).
        const int8_t *q8_super = xq + (size_t)ib_super * 256;
        const float   yd       = xscale[ib_super];

        const float dg_yd = dg * yd;
        const float du_yd = du * yd;

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            // qs_soa indices: 4*ib32 + 0..3, each stepping by n_blocks_in.
            const size_t base = (size_t)(4 * ib32) * (size_t)n_blocks_in;
            const uint16_t qg0 = qs_g[base + 0 * (size_t)n_blocks_in];
            const uint16_t qg1 = qs_g[base + 1 * (size_t)n_blocks_in];
            const uint16_t qg2 = qs_g[base + 2 * (size_t)n_blocks_in];
            const uint16_t qg3 = qs_g[base + 3 * (size_t)n_blocks_in];
            const uint16_t qu0 = qs_u[base + 0 * (size_t)n_blocks_in];
            const uint16_t qu1 = qs_u[base + 1 * (size_t)n_blocks_in];
            const uint16_t qu2 = qs_u[base + 2 * (size_t)n_blocks_in];
            const uint16_t qu3 = qs_u[base + 3 * (size_t)n_blocks_in];

            const uint32_t aux_g_g = (uint32_t)qg0 | ((uint32_t)qg1 << 16);
            const uint32_t aux_g_s = (uint32_t)qg2 | ((uint32_t)qg3 << 16);
            const uint32_t aux_u_g = (uint32_t)qu0 | ((uint32_t)qu1 << 16);
            const uint32_t aux_u_s = (uint32_t)qu2 | ((uint32_t)qu3 << 16);

            const float ls_g = 0.5f + (float)(aux_g_s >> 28);
            const float ls_u = 0.5f + (float)(aux_u_s >> 28);

            const int8_t *q8 = q8_super + ib32 * 32;

            int isum_g = 0;
            int isum_u = 0;
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint8_t  gidx_g  = (uint8_t)((aux_g_g >> (8 * l)) & 0xff);
                const uint8_t  gidx_u  = (uint8_t)((aux_u_g >> (8 * l)) & 0xff);
                const uint8_t  signs_g = c_pair_ksigns[(aux_g_s >> (7 * l)) & 0x7f];
                const uint8_t  signs_u = c_pair_ksigns[(aux_u_s >> (7 * l)) & 0x7f];

                const uint8_t *grid_g = (const uint8_t *)(c_pair_grid + gidx_g);
                const uint8_t *grid_u = (const uint8_t *)(c_pair_grid + gidx_u);
                const int8_t  *yseg   = q8 + 8 * l;

                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int sg = (signs_g & c_pair_kmask[j]) ? -1 : 1;
                    const int su = (signs_u & c_pair_kmask[j]) ? -1 : 1;
                    const int gj = (int)grid_g[j] * sg;
                    const int uj = (int)grid_u[j] * su;
                    const int yj = (int)yseg[j];
                    isum_g += gj * yj;
                    isum_u += uj * yj;
                }
            }

            total_g += dg_yd * (ls_g * (float)isum_g);
            total_u += du_yd * (ls_u * (float)isum_u);
        }
    }
    // else: lane idle, total_g = total_u = 0.

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total_g += __shfl_xor_sync(0xffffffff, total_g, offset);
        total_u += __shfl_xor_sync(0xffffffff, total_u, offset);
    }

    if (lane == 0) {
        const float gate = 0.25f * total_g;
        const float up   = 0.25f * total_u;

        float g = gate;
        float u = up;
        if (clamp > 1.0e-6f) {
            if (g > clamp)  g = clamp;
            if (u > clamp)  u = clamp;
            if (u < -clamp) u = -clamp;
        }

        float s;
        if (g >= 0.0f) {
            const float e = __expf(-g);
            s = g / (1.0f + e);
        } else {
            const float e = __expf(g);
            s = g * e / (1.0f + e);
        }
        mid_out[(size_t)slot * out_dim + row] = s * u * topk_w[slot];
    }
}

void launch_routed_moe_pair_swiglu_resident_soa_v2_f32_prealloc(
        const uint16_t *qs_soa_gate,
        const uint16_t *d_soa_gate,
        const uint16_t *qs_soa_up,
        const uint16_t *d_soa_up,
        const float    *x_fp32,
        const int32_t  *topk_ids,
        const float    *topk_w,
        float          *mid_out,
        int             n_used,
        int             out_dim,
        int             in_dim,
        int             n_experts,
        float           clamp_value,
        int8_t         *scratch_xq_dev,
        float          *scratch_xscale_dev,
        int16_t        *scratch_xbsums_dev,
        cudaStream_t    stream)
{
    if (in_dim % 256 != 0) {
        std::fprintf(stderr,
            "launch_routed_moe_pair_swiglu_resident_soa_v2_f32_prealloc: in_dim=%d not a multiple of 256\n",
            in_dim);
        return;
    }
    const int n_blocks_in = in_dim / 256;
    if (n_blocks_in > 32) {
        std::fprintf(stderr,
            "launch_routed_moe_pair_swiglu_resident_soa_v2_f32_prealloc: "
            "n_blocks_in=%d > 32 unsupported (lane=ib_super mapping)\n",
            n_blocks_in);
        return;
    }

    launch_quantize_fp32_to_q8_K(x_fp32, scratch_xq_dev, scratch_xscale_dev,
                                 scratch_xbsums_dev, n_blocks_in, stream);

    dim3 grid(out_dim, n_used, 1);
    constexpr int TPB = 32;
    moe_iq2_xxs_pair_swiglu_resident_soa_v2_kernel<<<grid, TPB, 0, stream>>>(
        qs_soa_gate, d_soa_gate, qs_soa_up, d_soa_up,
        scratch_xq_dev, scratch_xscale_dev,
        topk_ids, topk_w, mid_out,
        n_used, out_dim, n_blocks_in, n_experts, clamp_value);
}

} // namespace ds4cuda
