#ifndef DS4CUDA_SOA_LAYOUT_H
#define DS4CUDA_SOA_LAYOUT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Q2_K super-block size (must equal DS4_QK_K from include/ds4cuda.h). */
#define DS4_Q2K_SUPERBLOCK 256

/* Byte layout of a Q2_K SoA v2 packed tensor.
 *
 * Source AoS shape:  block_q2_K[n_experts * out_dim * n_blocks_in]
 *                    where n_blocks_in = in_dim / 256.
 *
 * Target SoA v2 layout (single contiguous blob, four sub-arrays in this order):
 *   scales : uint8 [n_experts][out_dim][16][n_blocks_in]
 *   qs     : uint8 [n_experts][out_dim][64][n_blocks_in]
 *   d      : uint16[n_experts][out_dim][n_blocks_in]
 *   dmin   : uint16[n_experts][out_dim][n_blocks_in]
 *
 * Sum-of-bytes-per-block: 16 + 64 + 2 + 2 = 84 — same as AoS.
 *
 * Header dependency-free so both the offline tool (host C) and runtime
 * (CUDA host code) can include it.
 *
 * Byte-compatible with the SoA mirror produced by
 *   cuda/moe_q2k_sum6.cu :: launch_build_moe_q2k_sum6_resident_soa
 * and consumed by the q2k_resident_soa_v2 kernels in the same file.
 */
/* Precondition: n_experts > 0, out_dim > 0, in_dim > 0 AND
 * in_dim % DS4_Q2K_SUPERBLOCK == 0. Behavior undefined otherwise. */
struct ds4_q2k_soa_v2_layout {
    size_t scales_offset;
    size_t qs_offset;
    size_t d_offset;
    size_t dmin_offset;
    size_t scales_bytes;
    size_t qs_bytes;
    size_t d_bytes;
    size_t dmin_bytes;
    size_t total_bytes;
};

static inline struct ds4_q2k_soa_v2_layout
ds4_q2k_soa_v2_layout(int n_experts, int out_dim, int in_dim)
{
    struct ds4_q2k_soa_v2_layout L;
    size_t e = (size_t)n_experts;
    size_t o = (size_t)out_dim;
    size_t nb = (size_t)in_dim / DS4_Q2K_SUPERBLOCK;
    L.scales_bytes = e * o * 16 * nb;
    L.qs_bytes     = e * o * 64 * nb;
    L.d_bytes      = e * o * nb * 2;
    L.dmin_bytes   = e * o * nb * 2;
    L.scales_offset = 0;
    L.qs_offset     = L.scales_bytes;
    L.d_offset      = L.scales_bytes + L.qs_bytes;
    L.dmin_offset   = L.scales_bytes + L.qs_bytes + L.d_bytes;
    L.total_bytes   = L.scales_bytes + L.qs_bytes + L.d_bytes + L.dmin_bytes;
    return L;
}

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_SOA_LAYOUT_H */
