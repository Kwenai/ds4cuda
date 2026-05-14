#include "repack_soa.h"
#include "ds4cuda.h"
#include "ds4cuda_soa_layout.h"
#include "ds4cuda_iq2_soa_layout.h"

#include <stddef.h>

void ds4_repack_q2k_aos_to_soa_v2(const struct ds4_block_q2_K *aos,
                                  uint8_t *soa,
                                  int n_experts, int out_dim, int in_dim)
{
    int n_blocks_in = in_dim / DS4_Q2K_SUPERBLOCK;
    struct ds4_q2k_soa_v2_layout L = ds4_q2k_soa_v2_layout(n_experts, out_dim, in_dim);
    uint8_t  *scales_soa = soa + L.scales_offset;
    uint8_t  *qs_soa     = soa + L.qs_offset;
    uint16_t *d_soa      = (uint16_t *)(soa + L.d_offset);
    uint16_t *dmin_soa   = (uint16_t *)(soa + L.dmin_offset);

    for (int e = 0; e < n_experts; e++) {
        for (int r = 0; r < out_dim; r++) {
            for (int b = 0; b < n_blocks_in; b++) {
                size_t aos_idx = ((size_t)e * out_dim + r) * n_blocks_in + b;
                const struct ds4_block_q2_K *blk = &aos[aos_idx];

                for (int lane = 0; lane < 16; lane++) {
                    size_t soa_idx = (((size_t)e * out_dim + r) * 16 + lane) * n_blocks_in + b;
                    scales_soa[soa_idx] = blk->scales[lane];
                }
                for (int lane = 0; lane < 64; lane++) {
                    size_t soa_idx = (((size_t)e * out_dim + r) * 64 + lane) * n_blocks_in + b;
                    qs_soa[soa_idx] = blk->qs[lane];
                }
                size_t scalar_idx = ((size_t)e * out_dim + r) * n_blocks_in + b;
                d_soa   [scalar_idx] = blk->d;
                dmin_soa[scalar_idx] = blk->dmin;
            }
        }
    }
}

void ds4_repack_iq2_xxs_aos_to_soa_v2(const struct ds4_block_iq2_xxs *aos,
                                      uint8_t *soa,
                                      int n_experts, int out_dim, int in_dim)
{
    int n_blocks_in = in_dim / DS4_IQ2_XXS_SUPERBLOCK;
    struct ds4_iq2_xxs_soa_v2_layout L =
        ds4_iq2_xxs_soa_v2_layout(n_experts, out_dim, in_dim);
    uint16_t *qs_soa = (uint16_t *)(soa + L.qs_offset);
    uint16_t *d_soa  = (uint16_t *)(soa + L.d_offset);

    for (int e = 0; e < n_experts; e++) {
        for (int r = 0; r < out_dim; r++) {
            for (int b = 0; b < n_blocks_in; b++) {
                size_t aos_idx = ((size_t)e * out_dim + r) * n_blocks_in + b;
                const struct ds4_block_iq2_xxs *blk = &aos[aos_idx];

                for (int lane = 0; lane < 32; lane++) {
                    size_t soa_idx = (((size_t)e * out_dim + r) * 32 + lane) * n_blocks_in + b;
                    qs_soa[soa_idx] = blk->qs[lane];
                }
                size_t d_idx = ((size_t)e * out_dim + r) * n_blocks_in + b;
                d_soa[d_idx] = blk->d;
            }
        }
    }
}
