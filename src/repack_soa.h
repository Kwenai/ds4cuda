#ifndef DS4CUDA_REPACK_SOA_H
#define DS4CUDA_REPACK_SOA_H

#include <stddef.h>
#include <stdint.h>

struct ds4_block_q2_K;  /* forward decl from <ds4cuda.h> */

#ifdef __cplusplus
extern "C" {
#endif

/* Transpose Q2_K AoS expert weights to SoA v2 layout.
 *
 *   aos shape : block_q2_K[n_experts * out_dim * (in_dim/256)]
 *   soa output: ds4_q2k_soa_v2_layout(n_experts, out_dim, in_dim).total_bytes
 *
 * Output is a single contiguous blob with four sub-arrays at the offsets
 * computed by ds4_q2k_soa_v2_layout. Caller pre-allocates.
 *
 * Byte-compatible with the SoA mirror produced by
 *   cuda/moe_q2k_sum6.cu :: launch_build_moe_q2k_sum6_resident_soa
 * Same byte permutation; just runs on CPU so the offline repack tool can
 * use it without a GPU.
 *
 * Pure CPU; safe for host-only tools.
 */
void ds4_repack_q2k_aos_to_soa_v2(const struct ds4_block_q2_K *aos,
                                  uint8_t *soa,
                                  int n_experts, int out_dim, int in_dim);

struct ds4_block_iq2_xxs;  /* forward decl from <ds4cuda.h> */

/* Transpose IQ2_XXS AoS expert weights to SoA v2 layout.
 *
 *   aos shape : block_iq2_xxs[n_experts * out_dim * (in_dim/256)]
 *   soa output: ds4_iq2_xxs_soa_v2_layout(n_experts, out_dim, in_dim).total_bytes
 *
 * Output blob has the qs and d sub-arrays at the offsets computed by
 * ds4_iq2_xxs_soa_v2_layout. Caller pre-allocates.
 *
 * Byte layout matches ds4_iq2_xxs_soa_v2_layout (see
 * include/ds4cuda_iq2_soa_layout.h); the runtime consumes the same
 * layout when it detects `_soa_v2.weight` tensors. Pure CPU; safe for
 * host-only tools.
 */
void ds4_repack_iq2_xxs_aos_to_soa_v2(const struct ds4_block_iq2_xxs *aos,
                                      uint8_t *soa,
                                      int n_experts, int out_dim, int in_dim);

#ifdef __cplusplus
}
#endif

#endif
