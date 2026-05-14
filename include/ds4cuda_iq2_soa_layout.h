#ifndef DS4CUDA_IQ2_SOA_LAYOUT_H
#define DS4CUDA_IQ2_SOA_LAYOUT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* IQ2_XXS super-block size (must equal DS4_QK_K from include/ds4cuda.h). */
#define DS4_IQ2_XXS_SUPERBLOCK 256

/* Byte layout of an IQ2_XXS SoA v2 packed tensor.
 *
 * Source AoS shape:  block_iq2_xxs[n_experts * out_dim * n_blocks_in]
 *                    where each block is 66 bytes (2B d + 32 x u16 qs)
 *                    and n_blocks_in = in_dim / 256.
 *
 * Target SoA v2 layout (single contiguous blob, two sub-arrays in this order):
 *   qs : uint16 [n_experts][out_dim][32][n_blocks_in]
 *   d  : uint16 [n_experts][out_dim][n_blocks_in]
 *
 * Sum-of-bytes-per-block: 32*2 + 2 = 66 — same as AoS. Total preserved.
 *
 * The offline repack tool (tools/repack_gguf_soa.c, --replace mode) writes
 * tensors with these bytes; the runtime detects `_soa_v2.weight` tensors
 * and slices pointers from the managed blob. Header dependency-free so
 * both the offline tool (host C) and runtime (CUDA host code) can include it.
 *
 * Precondition: n_experts > 0, out_dim > 0, in_dim > 0 AND
 * in_dim % DS4_IQ2_XXS_SUPERBLOCK == 0. Behavior undefined otherwise.
 */
struct ds4_iq2_xxs_soa_v2_layout {
    size_t qs_offset;
    size_t d_offset;
    size_t qs_bytes;
    size_t d_bytes;
    size_t total_bytes;
};

static inline struct ds4_iq2_xxs_soa_v2_layout
ds4_iq2_xxs_soa_v2_layout(int n_experts, int out_dim, int in_dim)
{
    struct ds4_iq2_xxs_soa_v2_layout L;
    size_t e = (size_t)n_experts;
    size_t o = (size_t)out_dim;
    size_t nb = (size_t)in_dim / DS4_IQ2_XXS_SUPERBLOCK;
    L.qs_bytes    = e * o * 32 * nb * 2;
    L.d_bytes     = e * o * nb * 2;
    L.qs_offset   = 0;
    L.d_offset    = L.qs_bytes;
    L.total_bytes = L.qs_bytes + L.d_bytes;
    return L;
}

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_IQ2_SOA_LAYOUT_H */
