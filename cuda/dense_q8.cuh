// dense_q8.cuh — host-callable launchers for the Q8_0 dequant and
// Q8_0×Q8_0 matvec kernels used by the forward path (attention
// projections, output head). Implementation in cuda/dense_q8.cu.
//
// Exposed launchers:
//   1. launch_dequant_q8_0_to_f32  — full-tensor Q8_0 → fp32 dequant.
//      Bit-exact against the host naive (single multiply per element).
//   2. launch_mul_mv_q8_0_q8_0_f32_prealloc — multi-row Q8_0 × Q8_0
//      matvec used by the production attention and output-projection
//      paths (caller-owned activation scratch; selectable dp4a routing
//      via q8_0_q8_0_use_dp4a).
//
// All kernels consume the on-disk block_q8_0 layout (fp16 d + 32 int8)
// directly via reinterpret_cast on the managed pointer; no dequant
// table or shared constants.
//
// See ds4/metal/dense.metal:108–176 (kernel_mul_mv_q8_0_f32_impl) for
// the upstream Metal spec.

#ifndef DS4CUDA_DENSE_Q8_CUH
#define DS4CUDA_DENSE_Q8_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"

namespace ds4cuda {

// Launch dequant_q8_0_to_f32 on `stream`. `src_dev` points at n_blocks
// contiguous block_q8_0 records (34 bytes each); `dst_dev` receives
// n_blocks * 32 floats. Caller is responsible for sizing both buffers.
void launch_dequant_q8_0_to_f32(const block_q8_0 *src_dev,
                                float *dst_dev,
                                int n_blocks,
                                cudaStream_t stream = 0);


// ---------------------------------------------------------------------
// Q8_0 × Q8_0 path (q_lora alignment).
//
// Bit-exact CPU match for matvec_q8_0 in ds4/ds4.c (line 3443). The
// activation gets quantized to (int8 qs, fp32 scale) — the scale is
// stored as raw fp32, NOT round-tripped through fp16, matching
// quantize_q8_0_activation in ds4.c:3094. See dense_q8.cu for full
// citations and rationale.

// Q8_0 × Q8_0 → fp32 matvec with caller-owned activation quantization
// scratch:
//   scratch_xq_dev     : int8  [n_cols]
//   scratch_xscale_dev : float [n_cols / 32]
// This variant performs no device allocation/free and does not synchronize.
//
// Preconditions:
//   - W_dev is n_rows * (n_cols/32) on-disk block_q8_0 records, row-major.
//   - x_fp32_dev is n_cols fp32.
//   - n_cols % 32 == 0.
//   - y_dev is n_rows fp32.
//
// Internally routes between the warp-per-row baseline kernel and a dp4a
// variant via q8_0_q8_0_use_dp4a (see implementation for the cutoff).
void launch_mul_mv_q8_0_q8_0_f32_prealloc(const block_q8_0 *W_dev,
                                          const float *x_fp32_dev,
                                          float *y_dev,
                                          int n_rows,
                                          int n_cols,
                                          int8_t *scratch_xq_dev,
                                          float *scratch_xscale_dev,
                                          cudaStream_t stream = 0);

bool q8_0_q8_0_use_dp4a(int n_rows, int n_cols);

} // namespace ds4cuda

#endif // DS4CUDA_DENSE_Q8_CUH
