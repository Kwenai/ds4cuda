// moe_q2k_sum6.cu — routed MoE Q2_K down-projection summed over the
// 6 selected expert slots, register-accumulated (NO atomicAdd).
//
// Stage: `ffn_moe_out` (fp32 [N_EMBD=4096]).
//
// CPU reference (cited verbatim, ds4/ds4.c):
//   - 1628..1666  ds4_quantize_row_q8_K (per-slot mid quantizer)
//   - 1748..1786  ds4_vec_dot_q2_K_q8_K (scalar fallback used as our
//                  numerical model: simpler than NEON, and bit-equal to
//                  the NEON path because the int8×int8 partial products
//                  fit in int32 and re-summing is associative).
//   - 3916..3928  matvec_q2_k_accum_worker (slot accumulation tail)
//   - 5273..5296  routed-MoE call site (Q8_K activation order)
//
// Block layout: 1 CTA = 1 output row d (grid.x = out_dim = 4096).
//   blockDim = 32 (single warp). Lane stripe over (slot, super-block):
//   ib_global = slot * n_blocks_in + ib_super, lane l processes
//   ib_global = l, l+32, ... With n_used=6, n_blocks_in=8, total = 48
//   so lanes 0..15 do 2 iterations and 16..31 do 1.
//
// The "sum6" property: the 6-slot accumulation lives in a single fp32
// register sumf per lane; the inner loop over (slot, ib_super) traverses
// all 48 contributions before the final warp reduce. NO atomicAdd, NO
// cross-block sync — design §4 explicitly requires this.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "common.cuh"
#include "moe_iq2_pair.cuh"   // launch_quantize_fp32_to_q8_K
#include "moe_q2k_sum6.cuh"

namespace ds4cuda {

// Per-super-block Q2_K · Q8_K contribution (returns fp32 increment to
// the row sumf). All sub-block math matches the scalar fallback path
// of ds4/ds4.c:1748..1786 exactly:
//   summs    = sum_{i=0..15} bsums[i] * (scales[i] >> 4)
//   isum     = sum_{sb=0..15} (scales[sb] & 0xF) *
//                 sum_{e=0..15} ((qs[idx(sb,e)] >> shift(sb)) & 3) * q8[sb*16+e]
//   contrib  = (yd * f16(W.d)) * isum - (yd * f16(W.dmin)) * summs
//
// where idx(sb, e) = (sb >> 3) * 32 + (sb & 1) * 16 + e
//   and shift(sb) = ((sb >> 1) & 3) * 2.
__device__ __forceinline__ float q2k_q8k_super_dot(const block_q2_K &wb,
                                                   const int8_t     *__restrict__ q8,
                                                   const int16_t    *__restrict__ bsums,
                                                   float             yd)
{
    int summs = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        summs += (int)bsums[i] * (int)(wb.scales[i] >> 4);
    }

    int isum = 0;
    #pragma unroll
    for (int sb = 0; sb < 16; ++sb) {
        const int k             = sb >> 3;
        const int subblock_half = sb & 1;
        const int shift         = ((sb >> 1) & 3) << 1;

        const int sub_scale = (int)(wb.scales[sb] & 0xF);

        int dot = 0;
        const uint8_t *qs_base = wb.qs + k * 32 + subblock_half * 16;
        const int8_t  *y_base  = q8 + sb * 16;
        #pragma unroll
        for (int e = 0; e < 16; ++e) {
            const int q2v = ((int)qs_base[e] >> shift) & 0x3;
            dot += q2v * (int)y_base[e];
        }
        isum += sub_scale * dot;
    }

    const float dall = yd * fp16_bits_to_fp32(wb.d);
    const float dmin = yd * fp16_bits_to_fp32(wb.dmin);
    return dall * (float)isum - dmin * (float)summs;
}

__device__ __forceinline__ float q2k_resident_soa_v2_q8k_super_dot(
                                                   const uint8_t  *__restrict__ scales_row,
                                                   const uint8_t  *__restrict__ qs_row,
                                                   uint16_t        d_bits,
                                                   uint16_t        dmin_bits,
                                                   int             ib_super,
                                                   int             n_blocks_in,
                                                   int             act_it,
                                                   int             n_iter,
                                                   const int8_t   *__restrict__ midq_soa,
                                                   const int16_t  *__restrict__ bsums_soa,
                                                   float           yd)
{
    int summs = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t sc = scales_row[i * n_blocks_in + ib_super];
        const int16_t bsum = bsums_soa[i * n_iter + act_it];
        summs += (int)bsum * (int)(sc >> 4);
    }

    int isum = 0;
    #pragma unroll
    for (int sb = 0; sb < 16; ++sb) {
        const int k             = sb >> 3;
        const int subblock_half = sb & 1;
        const int shift         = ((sb >> 1) & 3) << 1;

        const uint8_t sc = scales_row[sb * n_blocks_in + ib_super];
        const int sub_scale = (int)(sc & 0xF);

        int dot = 0;
        const int q_base = (k * 32 + subblock_half * 16) * n_blocks_in + ib_super;
        const int y_base = (sb * 16) * n_iter + act_it;
        #pragma unroll
        for (int e = 0; e < 16; ++e) {
            const uint8_t qbyte = qs_row[q_base + e * n_blocks_in];
            const int q2v = ((int)qbyte >> shift) & 0x3;
            const int8_t y = midq_soa[y_base + e * n_iter];
            dot += q2v * (int)y;
        }
        isum += sub_scale * dot;
    }

    const float dall = yd * fp16_bits_to_fp32(d_bits);
    const float dmin = yd * fp16_bits_to_fp32(dmin_bits);
    return dall * (float)isum - dmin * (float)summs;
}

// ---------------------------------------------------------------------
// Full-tensor sum6 kernel.
//
// grid: (out_dim, 1, 1). blockDim.x = 32.
// W_down layout: [n_experts][out_dim][n_blocks_in] super-blocks (the
//   selected experts are dereferenced via topk_ids[slot]).
// midq layout (Q8_K split):
//   q8     [n_used * n_blocks_in * 256] int8
//   yscale [n_used * n_blocks_in]       fp32  (block_q8_K::d)
//   ybsums [n_used * n_blocks_in * 16]  int16 (block_q8_K::bsums)
//
// Stride-32 lane iteration over (slot, ib_super) flattened.
// ---------------------------------------------------------------------
__global__ void moe_q2k_sum6_full_kernel(const block_q2_K *__restrict__ W_down,
                                         const int8_t     *__restrict__ midq,
                                         const float      *__restrict__ midscale,
                                         const int16_t    *__restrict__ midbsums,
                                         const int32_t    *__restrict__ topk_ids,
                                         float            *__restrict__ out,
                                         int               n_used,
                                         int               out_dim,
                                         int               n_blocks_in,
                                         int               n_experts)
{
    const int row  = blockIdx.x;
    if (row >= out_dim) return;
    const int lane = threadIdx.x;

    float sumf = 0.0f;

    const int n_iter = n_used * n_blocks_in;
    for (int it = lane; it < n_iter; it += 32) {
        const int slot     = it / n_blocks_in;
        const int ib_super = it - slot * n_blocks_in;
        const int eid      = topk_ids[slot];
        if (eid < 0 || eid >= n_experts) continue;

        const block_q2_K &wb =
            W_down[((size_t)eid * out_dim + row) * n_blocks_in + ib_super];
        const int8_t  *q8b    = midq    + ((size_t)slot * n_blocks_in + ib_super) * 256;
        const int16_t *bsumsb = midbsums + ((size_t)slot * n_blocks_in + ib_super) * 16;
        const float    yd     = midscale[(size_t)slot * n_blocks_in + ib_super];

        sumf += q2k_q8k_super_dot(wb, q8b, bsumsb, yd);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);
    }
    if (lane == 0) out[row] = sumf;
}

// ---------------------------------------------------------------------
// Q8_K activation SoA transpose. Used internally by the resident SoA v2
// f32 prealloc launcher; the original AoS Q8_K scratch is permuted so
// the resident SoA v2 dot kernel can issue coalesced lane reads.
// ---------------------------------------------------------------------
__global__ void pack_moe_q8k_soa_v2_kernel(
                                         const int8_t     *__restrict__ midq,
                                         const int16_t    *__restrict__ midbsums,
                                         int8_t           *__restrict__ midq_soa,
                                         int16_t          *__restrict__ midbsums_soa,
                                         int               n_iter)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int q_fields = 256 * n_iter;
    const int b_fields = 16 * n_iter;

    if (linear < q_fields) {
        const int e = linear / n_iter;
        const int it = linear - e * n_iter;
        midq_soa[(size_t)e * n_iter + it] = midq[(size_t)it * 256 + e];
    }
    if (linear < b_fields) {
        const int e = linear / n_iter;
        const int it = linear - e * n_iter;
        midbsums_soa[(size_t)e * n_iter + it] = midbsums[(size_t)it * 16 + e];
    }
}

// ---------------------------------------------------------------------
// Resident SoA v2 builder kernel (session-init): permutes the full-expert
// AoS Q2_K tensor into the four SoA streams used by the resident dot
// kernel below. Called from the loader / inference engine init path.
// ---------------------------------------------------------------------
__global__ void build_moe_q2k_sum6_resident_soa_kernel(
                                         const block_q2_K *__restrict__ W_down,
                                         uint8_t          *__restrict__ scales_soa,
                                         uint8_t          *__restrict__ qs_soa,
                                         uint16_t         *__restrict__ d_soa,
                                         uint16_t         *__restrict__ dmin_soa,
                                         int               out_dim,
                                         int               n_blocks_in,
                                         int               n_experts)
{
    const int row = blockIdx.x;
    const int expert = blockIdx.z;
    const int linear = blockIdx.y * blockDim.x + threadIdx.x;
    constexpr int FIELDS = 84;

    if (row >= out_dim || expert >= n_experts || linear >= FIELDS * n_blocks_in) {
        return;
    }

    const int field = linear / n_blocks_in;
    const int ib_super = linear - field * n_blocks_in;
    const block_q2_K &wb =
        W_down[((size_t)expert * out_dim + row) * n_blocks_in + ib_super];

    const size_t er = ((size_t)expert * out_dim + row);
    if (field < 16) {
        scales_soa[(er * 16 + field) * n_blocks_in + ib_super] =
            wb.scales[field];
    } else if (field < 80) {
        qs_soa[(er * 64 + (field - 16)) * n_blocks_in + ib_super] =
            wb.qs[field - 16];
    } else if (field == 80) {
        d_soa[er * n_blocks_in + ib_super] = wb.d;
    } else if (field == 81) {
        dmin_soa[er * n_blocks_in + ib_super] = wb.dmin;
    }
}

// ---------------------------------------------------------------------
// Resident SoA v2 routed-MoE Q2_K · Q8_K dot kernel (production hot path).
//
// One CTA per output row. Lane stripe over (slot, ib_super) flattened.
// Selected experts are read directly from the resident SoA tensors via
// topk_ids[slot]. Activation is transposed Q8_K SoA from
// pack_moe_q8k_soa_v2_kernel.
// ---------------------------------------------------------------------
__global__ void moe_q2k_sum6_resident_soa_v2_kernel(
                                         const uint8_t    *__restrict__ scales_soa,
                                         const uint8_t    *__restrict__ qs_soa,
                                         const uint16_t   *__restrict__ d_soa,
                                         const uint16_t   *__restrict__ dmin_soa,
                                         const int8_t     *__restrict__ midq_soa,
                                         const float      *__restrict__ midscale,
                                         const int16_t    *__restrict__ midbsums_soa,
                                         const int32_t    *__restrict__ topk_ids,
                                         float            *__restrict__ out,
                                         int               n_used,
                                         int               out_dim,
                                         int               n_blocks_in,
                                         int               n_experts)
{
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    const int lane = threadIdx.x;

    const int n_iter = n_used * n_blocks_in;
    float sumf = 0.0f;

    for (int it = lane; it < n_iter; it += 32) {
        const int slot = it / n_blocks_in;
        const int ib_super = it - slot * n_blocks_in;
        const int eid = topk_ids[slot];
        if (eid < 0 || eid >= n_experts) continue;

        const size_t er = (size_t)eid * out_dim + row;
        const uint8_t *scales_row = scales_soa + er * 16 * n_blocks_in;
        const uint8_t *qs_row = qs_soa + er * 64 * n_blocks_in;
        const uint16_t *d_row = d_soa + er * n_blocks_in;
        const uint16_t *dmin_row = dmin_soa + er * n_blocks_in;
        const float yd = midscale[it];

        sumf += q2k_resident_soa_v2_q8k_super_dot(
            scales_row, qs_row, d_row[ib_super], dmin_row[ib_super],
            ib_super, n_blocks_in, it, n_iter, midq_soa, midbsums_soa, yd);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);
    }
    if (lane == 0) out[row] = sumf;
}

void launch_routed_moe_q2k_sum6_full_f32_prealloc(
                                         const block_q2_K *W_down,
                                         const float      *mid,
                                         const int32_t    *topk_ids,
                                         float            *out,
                                         int               n_used,
                                         int               out_dim,
                                         int               in_dim,
                                         int               n_experts,
                                         int8_t           *scratch_midq_dev,
                                         float            *scratch_midscale_dev,
                                         int16_t          *scratch_midbsums_dev,
                                         cudaStream_t      stream)
{
    if (in_dim % 256 != 0) {
        std::fprintf(stderr,
            "launch_routed_moe_q2k_sum6_full_f32_prealloc: in_dim=%d not a multiple of 256\n",
            in_dim);
        return;
    }
    const int n_blocks_in = in_dim / 256;
    const size_t total_blocks = (size_t)n_used * n_blocks_in;

    launch_quantize_fp32_to_q8_K(mid, scratch_midq_dev, scratch_midscale_dev,
                                 scratch_midbsums_dev, (int)total_blocks, stream);

    constexpr int TPB = 32;
    moe_q2k_sum6_full_kernel<<<out_dim, TPB, 0, stream>>>(
        W_down, scratch_midq_dev, scratch_midscale_dev, scratch_midbsums_dev,
        topk_ids, out, n_used, out_dim, n_blocks_in, n_experts);
}

// Internal helper: Q8_K activation SoA transpose. Called from
// launch_routed_moe_q2k_sum6_resident_soa_v2_f32_prealloc.
static void launch_pack_moe_q8k_soa_v2(
                                         const int8_t     *midq_dev,
                                         const int16_t    *midbsums_dev,
                                         int8_t           *midq_soa_dev,
                                         int16_t          *midbsums_soa_dev,
                                         int               n_used,
                                         int               in_dim,
                                         cudaStream_t      stream)
{
    if (in_dim % 256 != 0) {
        std::fprintf(stderr,
            "launch_pack_moe_q8k_soa_v2: in_dim=%d not a multiple of 256\n",
            in_dim);
        return;
    }
    const int n_iter = n_used * (in_dim / 256);
    constexpr int TPB = 256;
    const int n = 256 * n_iter;
    const int grid = (n + TPB - 1) / TPB;
    pack_moe_q8k_soa_v2_kernel<<<grid, TPB, 0, stream>>>(
        midq_dev, midbsums_dev, midq_soa_dev, midbsums_soa_dev, n_iter);
}

void launch_build_moe_q2k_sum6_resident_soa(
                                         const block_q2_K *W_down,
                                         uint8_t          *scales_soa,
                                         uint8_t          *qs_soa,
                                         uint16_t         *d_soa,
                                         uint16_t         *dmin_soa,
                                         int               n_experts,
                                         int               out_dim,
                                         int               in_dim,
                                         cudaStream_t      stream)
{
    if (in_dim % 256 != 0) {
        std::fprintf(stderr,
            "launch_build_moe_q2k_sum6_resident_soa: in_dim=%d not a multiple of 256\n",
            in_dim);
        return;
    }
    const int n_blocks_in = in_dim / 256;
    constexpr int TPB = 128;
    constexpr int FIELDS = 84;
    const int y = (FIELDS * n_blocks_in + TPB - 1) / TPB;
    build_moe_q2k_sum6_resident_soa_kernel<<<dim3(out_dim, y, n_experts),
                                             TPB, 0, stream>>>(
        W_down, scales_soa, qs_soa, d_soa, dmin_soa,
        out_dim, n_blocks_in, n_experts);
}

void launch_routed_moe_q2k_sum6_resident_soa_v2_f32_prealloc(
                                         const uint8_t    *scales_soa,
                                         const uint8_t    *qs_soa,
                                         const uint16_t   *d_soa,
                                         const uint16_t   *dmin_soa,
                                         const float      *mid,
                                         const int32_t    *topk_ids,
                                         float            *out,
                                         int               n_used,
                                         int               out_dim,
                                         int               in_dim,
                                         int               n_experts,
                                         int8_t           *scratch_midq_dev,
                                         float            *scratch_midscale_dev,
                                         int16_t          *scratch_midbsums_dev,
                                         int8_t           *scratch_midq_soa_dev,
                                         int16_t          *scratch_midbsums_soa_dev,
                                         cudaStream_t      stream)
{
    if (in_dim % 256 != 0) {
        std::fprintf(stderr,
            "launch_routed_moe_q2k_sum6_resident_soa_v2_f32_prealloc: in_dim=%d not a multiple of 256\n",
            in_dim);
        return;
    }
    const int n_blocks_in = in_dim / 256;
    const size_t total_blocks = (size_t)n_used * n_blocks_in;

    launch_quantize_fp32_to_q8_K(mid, scratch_midq_dev, scratch_midscale_dev,
                                 scratch_midbsums_dev, (int)total_blocks, stream);
    launch_pack_moe_q8k_soa_v2(scratch_midq_dev, scratch_midbsums_dev,
                               scratch_midq_soa_dev, scratch_midbsums_soa_dev,
                               n_used, in_dim, stream);

    constexpr int TPB = 32;
    moe_q2k_sum6_resident_soa_v2_kernel<<<out_dim, TPB, 0, stream>>>(
        scales_soa, qs_soa, d_soa, dmin_soa,
        scratch_midq_soa_dev, scratch_midscale_dev, scratch_midbsums_soa_dev,
        topk_ids, out, n_used, out_dim, n_blocks_in, n_experts);
}

} // namespace ds4cuda
