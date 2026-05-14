// dense_q8.cu — Q8_0 dequant + Q8_0 matvec CUDA kernels for the
// forward path (attention projections, output head, etc.).
//
// Exposes:
//   - launch_dequant_q8_0_to_f32          : full-tensor Q8_0 → fp32
//   - launch_mul_mv_q8_0_q8_0_f32_prealloc: multi-row Q8_0 × Q8_0 matvec
//     (production attention / output projection path; selectable dp4a
//     routing via q8_0_q8_0_use_dp4a — see body)
//
// Reference:
//   - ds4/metal/dense.metal:108–176 (kernel_mul_mv_q8_0_f32_impl, warp-per-row dot)
//   - ds4/ds4.c:1485 (f16_to_f32 reference, IEEE-strict)

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "common.cuh"
#include "dense_q8.cuh"

namespace ds4cuda {

bool q8_0_q8_0_use_dp4a(int n_rows, int n_cols) {
    // Selective dp4a routing for Q8 matvec. A/B data (commit history) shows dp4a
    // wins +1.15-1.4x on cols={2048, 4096} with small/medium row counts.
    // Larger n_rows are flat-or-slower; the output projection
    // (n_rows=129280) must keep the legacy reduction tree intact for
    // the argmax==2581 terminal correctness gate.
    //
    // cols=1024 shapes (typical q_lora/kv_raw output) showed no benefit
    // in prior A/B; route them through the legacy kernel as well.
    if (n_rows >= 65536) return false;     // output projection
    if (n_cols == 1024) return false;      // low-cols path
    return n_cols == 2048 || n_cols == 4096;
}

// ---------------------------------------------------------------------
// Kernel 1: full-tensor dequant.
//
// Layout: each thread handles one int8 element. A block of 256 threads
// covers 8 Q8_0 blocks (8 * 32 = 256 elements). Each warp pre-loads
// the fp16 scale once via lane 0 and broadcasts via __shfl_sync, but
// since 32 elements = 1 block per warp, the shuffle is unnecessary —
// every lane in the same warp targets the same block.
//
// Q8_0 dequant has no accumulation: dst[i] = d * (float)qs[i], so the
// fp32 result is bit-exact against any host implementation that uses
// the same IEEE fp16 -> fp32 cast and the same float multiply order.
// ---------------------------------------------------------------------
__global__ void dequant_q8_0_to_f32_kernel(const block_q8_0 *__restrict__ src,
                                           float *__restrict__ dst,
                                           int n_blocks) {
    // Each warp (= 32 threads) handles exactly 1 block of 32 elements.
    // blockDim is fixed at 256 (8 warps).
    const int warp_in_block = threadIdx.x >> 5;          // 0..7
    const int lane          = threadIdx.x & 31;          // 0..31
    const int block_id      = blockIdx.x * 8 + warp_in_block;
    if (block_id >= n_blocks) return;

    const block_q8_0 &b = src[block_id];
    // Lane 0 reads fp16 scale; broadcast across the warp.
    float d;
    if (lane == 0) d = fp16_bits_to_fp32(b.d);
    d = __shfl_sync(0xffffffff, d, 0);

    const int q  = (int)b.qs[lane];
    dst[block_id * 32 + lane] = d * (float)q;
}

void launch_dequant_q8_0_to_f32(const block_q8_0 *src_dev,
                                float *dst_dev,
                                int n_blocks,
                                cudaStream_t stream) {
    // 256 threads/block → 8 Q8_0 blocks/CUDA-block.
    constexpr int TPB = 256;
    constexpr int BLOCKS_PER_CTA = TPB / 32; // 8
    const int n_cta = (n_blocks + BLOCKS_PER_CTA - 1) / BLOCKS_PER_CTA;
    dequant_q8_0_to_f32_kernel<<<n_cta, TPB, 0, stream>>>(src_dev, dst_dev, n_blocks);
}

// ---------------------------------------------------------------------
// Q8_0 × Q8_0 path (q_lora alignment).
//
// Goal: bit-exact CPU match for matvec_q8_0 in ds4/ds4.c (line 3443).
// CPU recipe (canonical reference, NOT GGML's standard ggml_quantize_q8_0
// which fp16-round-trips d_x):
//
//   ds4.c:3094 quantize_q8_0_activation
//     amax     = max_i |x[i]|                         (line 3101–3103)
//     d        = amax / 127.0f                        (line 3104, raw fp32)
//     id       = (d != 0) ? 1.0f / d : 0.0f           (line 3105, raw fp32)
//     xscale[b]= d                                    (line 3106, fp32 stored)
//     xq[i]    = clamp(lrintf(x[i]*id), -128, 127)    (line 3108–3111,
//                                                      round-half-to-even)
//
//   ds4.c:2857 dot_q8_0_row (NEON+DOTPROD path lines 2863–2904; scalar
//                            fallback 2907–2916)
//     per block b:
//       sum_i32 = sum int8 wq[i] * int8 xq[i]
//       acc    += f16_to_f32(W.d) * xscale[b] * (float)sum_i32
//
// Critical: the activation scale is stored as raw fp32, NOT
// f32_to_f16-then-back. The CUDA kernel must match this precisely to
// hit max_rel_diff < 1e-5 — fp16 round-tripping the activation scale
// introduces ~5e-4 per-block relative error, which breaks ULP-level
// agreement with the CPU reference on q_lora.
//
// We therefore expose a separate `int8 xq[]` + `float xscale[]` buffer
// pair for the activation rather than packing it into block_q8_0
// (which has a fp16 d field). The weight stays in on-disk block_q8_0.
//
// References:
//   - ds4/metal/dense.metal:108-176 (kernel_mul_mv_q8_0_f32 row mapping)
//   - ds4/ds4.c:3094 quantize_q8_0_activation, ds4/ds4.c:2857 dot_q8_0_row

// Quantize fp32 activation → (int8 qs, fp32 scale) per-32-element block.
//
// 1 warp = 1 block_q8_0 worth of activations. 32 lanes each handle one
// fp32 element. Butterfly shfl_xor reduces |x[i]| to amax in lane 0;
// lane 0 broadcasts d and id to all lanes. Each lane writes its int8
// output and lane 0 writes xscale[b] = d.
__global__ void quantize_fp32_to_q8_0_split_kernel(const float *__restrict__ x,
                                                   int8_t *__restrict__ xq,
                                                   float  *__restrict__ xscale,
                                                   int n_blocks) {
    // Block config: TPB=128 = 4 warps; 1 warp = 1 block_q8_0.
    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int b             = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (b >= n_blocks) return;

    const float xv = x[b * 32 + lane];
    float ax = fabsf(xv);

    // Butterfly max-reduce over the warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ax = fmaxf(ax, __shfl_xor_sync(0xffffffff, ax, offset));
    }
    // ax now equals amax in every lane.

    const float d  = ax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    // Round-half-to-even, matching lrintf with default rounding mode.
    // __float2int_rn implements round-to-nearest-even on NVPTX.
    int v = __float2int_rn(xv * id);
    if (v >  127) v =  127;
    if (v < -128) v = -128;

    xq[b * 32 + lane] = (int8_t)v;
    if (lane == 0) xscale[b] = d;
}

// Internal helper: quantize fp32 activation to (int8 qs, fp32 scale)
// per-32-element block. Called from launch_mul_mv_q8_0_q8_0_f32_prealloc.
// Not declared in the public header; only the matvec launcher uses it.
static void launch_quantize_fp32_to_q8_0_split(const float *x_dev,
                                               int8_t *xq_dev,
                                               float *xscale_dev,
                                               int n_blocks,
                                               cudaStream_t stream) {
    constexpr int TPB = 128;
    constexpr int WARPS_PER_CTA = TPB / 32;
    const int n_cta = (n_blocks + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    quantize_fp32_to_q8_0_split_kernel<<<n_cta, TPB, 0, stream>>>(
        x_dev, xq_dev, xscale_dev, n_blocks);
}

// Q8_0 × Q8_0 → fp32 matvec.
//
// 1 warp per row. Lane l processes blocks l, l+32, ... with stride 32.
// Each block:
//   sum_i32  = sum_{i=0..31} (int)W.qs[i] * (int)xq[i]
//   block_f  = f16_to_f32(W.d) * xscale[b] * (float)sum_i32
// Lane partial sum accumulates block_f. Warp butterfly reduces partials
// to lane 0, which writes y[r].
//
// Round-off vs CPU NEON path: the NEON code accumulates two fp32
// partials (`accv0`, `accv1`) then horizontally adds; this kernel uses
// 32 per-lane partials reduced via butterfly tree. With 128 blocks/row
// (4 blocks/lane on a 4096 in-dim) the reorder is harmless to <1e-5
// relative error in practice. If it ever exceeds the gate, the
// fallback is to serialize each warp's partial along block index in a
// shared-memory tree. We do not __dp4a here — that's a prefill
// optimization; this code is bandwidth-bound on decode and the int
// 32×int8 dot in fp32-mul-add form is plenty.
__global__ void mul_mv_q8_0_q8_0_f32_kernel(const block_q8_0 *__restrict__ W,
                                            const int8_t *__restrict__ xq,
                                            const float *__restrict__ xscale,
                                            float *__restrict__ y,
                                            int n_rows,
                                            int n_blocks_per_row) {
    // 4 warps / CTA. One warp = one row.
    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int row           = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (row >= n_rows) return;

    const block_q8_0 *row_ptr = W + (size_t)row * n_blocks_per_row;
    float partial = 0.0f;

    for (int b = lane; b < n_blocks_per_row; b += 32) {
        const block_q8_0 &blk = row_ptr[b];
        const float dw = fp16_bits_to_fp32(blk.d);
        const float dx = xscale[b];
        const int8_t *__restrict__ wq = blk.qs;
        const int8_t *__restrict__ xb = xq + b * 32;

        int sum_i32 = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            sum_i32 += (int)wq[i] * (int)xb[i];
        }
        // FMA: partial = fma(dw*dx, (float)sum_i32, partial). With
        // --fmad=true (nvcc default) this reads as a single rounded
        // multiply-add, matching the NEON `vfmaq_n_f32` semantics in
        // ds4.c:2888.
        partial = __fmaf_rn(dw * dx, (float)sum_i32, partial);
    }

    // Warp butterfly reduction (sum). The host NEON path reduces
    // 8 fp32 partials (accv0[0..3] + accv1[0..3]) via vaddvq_f32; the
    // CUDA path reduces 32 partials via xor butterfly. The two trees
    // differ in summation order but both are within ~1 ULP of the
    // mathematical sum at the working magnitude (per-block partials
    // ~0.5 for q_a). The amplified rel-diff at near-zero output
    // elements is intrinsic to the cancellation in this matvec, not
    // a kernel bug.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    if (lane == 0) y[row] = partial;
}

__device__ __forceinline__ int load_i8x4_as_i32(const int8_t *p) {
    int v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

// Same warp-per-row mapping as the baseline, but each pair of lanes
// cooperates on one Q8 block: lane 2g reads elements 0..15 and lane 2g+1
// reads 16..31. Compared with the baseline lane-per-block mapping, adjacent
// lanes now hit the two halves of the same 34-byte record. Compared with the
// 4-warps-per-row coalesced attempt, this keeps the row parallelism unchanged
// and pays only one shuffle per Q8 block.
__global__ void mul_mv_q8_0_q8_0_f32_dp4a_kernel(
                                            const block_q8_0 *__restrict__ W,
                                            const int8_t *__restrict__ xq,
                                            const float *__restrict__ xscale,
                                            float *__restrict__ y,
                                            int n_rows,
                                            int n_blocks_per_row) {
    const int lane          = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int row           = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (row >= n_rows) return;

    const block_q8_0 *row_ptr = W + (size_t)row * n_blocks_per_row;
    float partial = 0.0f;
    const int pair_lane = lane & 1;
    const int pair_id = lane >> 1;

    for (int b = pair_id; b < n_blocks_per_row; b += 16) {
        const block_q8_0 &blk = row_ptr[b];
        const float dw = fp16_bits_to_fp32(blk.d);
        const float dx = xscale[b];
        const int8_t *__restrict__ wq = blk.qs;
        const int8_t *__restrict__ xb = xq + b * 32;

        int sum_i32 = 0;
        const int start = pair_lane * 16;
        #pragma unroll
        for (int i = start; i < start + 16; i += 4) {
            const int wv = load_i8x4_as_i32(wq + i);
            const int xv = load_i8x4_as_i32(xb + i);
            sum_i32 = __dp4a(wv, xv, sum_i32);
        }
        sum_i32 += __shfl_xor_sync(0xffffffff, sum_i32, 1);
        if (pair_lane == 0) {
            partial = __fmaf_rn(dw * dx, (float)sum_i32, partial);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    if (lane == 0) y[row] = partial;
}

void launch_mul_mv_q8_0_q8_0_f32_prealloc(const block_q8_0 *W_dev,
                                          const float *x_fp32_dev,
                                          float *y_dev,
                                          int n_rows,
                                          int n_cols,
                                          int8_t *scratch_xq_dev,
                                          float *scratch_xscale_dev,
                                          cudaStream_t stream) {
    const int n_blocks = n_cols / 32;

    launch_quantize_fp32_to_q8_0_split(x_fp32_dev, scratch_xq_dev,
                                       scratch_xscale_dev, n_blocks, stream);

    constexpr int TPB = 128;
    constexpr int WARPS_PER_CTA = TPB / 32;
    const int n_cta = (n_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    if (q8_0_q8_0_use_dp4a(n_rows, n_cols)) {
        mul_mv_q8_0_q8_0_f32_dp4a_kernel<<<n_cta, TPB, 0, stream>>>(
            W_dev, scratch_xq_dev, scratch_xscale_dev, y_dev, n_rows, n_blocks);
    } else {
        mul_mv_q8_0_q8_0_f32_kernel<<<n_cta, TPB, 0, stream>>>(
            W_dev, scratch_xq_dev, scratch_xscale_dev, y_dev, n_rows, n_blocks);
    }
}

} // namespace ds4cuda
