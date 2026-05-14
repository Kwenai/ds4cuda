/* session_state.cu — per-session decode state allocator.
 *
 * Implements ds4_session_state_alloc / _free / _reset / _arena_bytes
 * (declared extern "C" in include/ds4cuda.h). The allocator does ONE
 * cudaMalloc that backs all 43 per-layer KV/state buffers + cross-layer
 * HC residual ping/pong buffers + activation arena. All sub-pointers in
 * ds4_layer_state / ds4_session_state are offsets into this single
 * device arena.
 *
 * Layout decisions (rationale; cite design + ds4 reference):
 *
 * - raw_kv / comp_kv / index_comp_kv stored as fp32 device buffers.
 *   Matches cuda/flash_attn.cu (decode_raw_f32, decode_mixed_f32) ABI:
 *   the kernel wants fp32 in/out and applies an internal fp16 round-
 *   trip on the read side to mirror the producer-side round-trip in
 *   ds4.c:6353-6363 / 6366-6371. fp16 storage would require an extra
 *   conversion kernel before every attention call — skipping that is a
 *   wash on memory (1M-context fp32 KV ~= 14 GB, well under 32 GB
 *   post-weights budget per design §6).
 *
 * - cap_raw  = DS4_N_SWA = 128 for every layer (sliding window cap is
 *   constant; ds4_default_raw_cap clamps to ctx, but at max_context
 *   >= 128 they're identical).
 *
 * - cap_comp = max_context / ratio + 2 (matches ds4.c:6310). The +2
 *   margin gives one extra slot for the partial-window emit at the
 *   end of prefill, plus one for the post-shift state row that
 *   compressor_pool_decode_state writes when (pos+1) % ratio == 0.
 *
 * - state_kv / state_score sized [coff*ratio, coff*head_dim_out] fp32.
 *   See cuda/compressor.cuh + commit 6c6fdf8 message:
 *     coff = 2 if ratio == 4 else 1
 *     attn  ratio=4  : 8 rows x 1024 cols = 8192 fp32 = 32 KB
 *     attn  ratio=128: 128 rows x 512 cols = 65536 fp32 = 256 KB
 *     idx   ratio=4  : 8 rows x 256 cols = 2048 fp32 = 8 KB
 *
 * - score buffer init to -INF (matches ds4.c:6318-6320). All other
 *   buffers init to zero.
 *
 * - residual_hc and residual_hc_scratch shape [N_HC=4, N_EMBD=4096]
 *   fp32 = 16384 floats = 64 KB each. They are the persistent ping/pong
 *   buffers used across layers within a single token (ds4 hyper-
 *   connection mixer), avoiding per-token cudaMalloc/cudaFree.
 *
 * - activation_arena: 16 MB scratch. Bump-allocator territory for the
 *   graph executor; exact per-stage requirements are the graph-build
 *   step's problem. 16 MB covers the largest single-token intermediate set
 *   in cpu_decode_scratch (Q [4096], heads [32 K], routed_mid_all
 *   [12 K], attn_score [4096], hc_flat [16 K], q8 blocks, etc.)
 *   with comfortable margin.
 *
 * Memory budget at max_context = 4096:
 *   raw_kv      :  43 *      128 *  512 * 4 = 11.0 MB
 *   comp_kv 4   :  21 *     1026 *  512 * 4 = 43.1 MB  (even layers >=2)
 *   comp_kv 128 :  21 *       34 *  512 * 4 =  1.4 MB  (odd layers)
 *   attn state  :  21 * 8192 fp32 (ratio=4) +
 *                  21 * 65536 fp32 (ratio=128) ~= 11.4 MB  (kv + score x2)
 *   indexer KV  :  21 *     1026 *  128 * 4 = 10.5 MB  (ratio=4 only)
 *   idx state   :  21 * 2048 fp32 x 2 (kv + score)    =  336 KB
 *   residual_hc :   2 *        4 * 4096 * 4 =  128 KB
 *   activation  :  16 MB
 *   ----------------------------------------------------
 *   TOTAL       : ~93 MB
 *
 * (Real total is reported by ds4_session_state_arena_bytes; the test
 * binary prints it.)
 */

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ds4cuda.h"

#define CK(stmt)                                                          \
    do {                                                                  \
        cudaError_t _e = (stmt);                                          \
        if (_e != cudaSuccess) {                                          \
            std::fprintf(stderr,                                          \
                         "session_state: CUDA error %s (%s) at %s:%d\n",  \
                         cudaGetErrorName(_e), cudaGetErrorString(_e),    \
                         __FILE__, __LINE__);                             \
            return -1;                                                    \
        }                                                                 \
    } while (0)

/* Round up to 256-byte alignment so each device sub-buffer is happily
 * aligned for vector loads. cudaMalloc returns 256-byte aligned base;
 * we keep that promise for every sub-pointer too. */
static size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

/* Hardcoded per-layer ratio table (mirrors ds4_layer_compress_ratio in
 * ds4/ds4.c:407). */
static int layer_ratio(int il) {
    if (il < 2) return 0;
    return (il & 1) == 0 ? 4 : 128;
}

/* Per-layer arena byte requirement.  Sums:
 *   raw_kv                 (cap_raw * head_dim * 4)
 *   comp_kv                (cap_comp * head_dim * 4)              if ratio
 *   attn_state_kv,_score   (coff*ratio * coff*head_dim * 4 * 2)   if ratio
 *   index_comp_kv          (cap_comp * indexer_head_dim * 4)      if ratio==4
 *   index_state_kv,_score  (coff*ratio * coff*idx_head_dim * 4*2) if ratio==4
 * Each summand is align256()'d so the next pointer stays aligned. */
static size_t per_layer_bytes(int ratio, int cap_raw, int cap_comp) {
    size_t total = 0;
    total += align256((size_t)cap_raw * DS4_N_HEAD_DIM * sizeof(float));

    if (ratio != 0) {
        total += align256((size_t)cap_comp * DS4_N_HEAD_DIM * sizeof(float));

        const int coff = (ratio == 4) ? 2 : 1;
        const size_t state_cells =
            (size_t)coff * (size_t)ratio * (size_t)coff * DS4_N_HEAD_DIM;
        /* state_kv + state_score */
        total += align256(state_cells * sizeof(float));
        total += align256(state_cells * sizeof(float));

        if (ratio == 4) {
            total += align256((size_t)cap_comp * DS4_N_INDEXER_HEAD_DIM
                              * sizeof(float));
            const size_t idx_cells = (size_t)coff * (size_t)ratio
                                     * (size_t)coff * DS4_N_INDEXER_HEAD_DIM;
            total += align256(idx_cells * sizeof(float));
            total += align256(idx_cells * sizeof(float));
        }
    }
    return total;
}

/* Cap_comp formula: ds4.c:6310. Mirrors that exactly. */
static int compute_cap_comp(int max_context, int ratio) {
    if (ratio == 0) return 0;
    return max_context / ratio + 2;
}

#define DS4_SESSION_ACTIVATION_ARENA_BYTES (16ull * 1024ull * 1024ull)

extern "C" size_t ds4_session_state_arena_bytes(int max_context) {
    if (max_context <= 0) return 0;
    const int cap_raw = (int)DS4_N_SWA < max_context ? (int)DS4_N_SWA : max_context;
    size_t total = 0;

    for (int il = 0; il < (int)DS4_N_LAYER; il++) {
        const int ratio = layer_ratio(il);
        const int cap_comp = compute_cap_comp(max_context, ratio);
        total += per_layer_bytes(ratio, cap_raw, cap_comp);
    }

    /* residual_hc ping/pong */
    total += align256((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(float));
    total += align256((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(float));
    /* activation arena */
    total += align256(DS4_SESSION_ACTIVATION_ARENA_BYTES);

    return total;
}

/* Init kernels: zero a region and -INF-fill a region. We could use
 * cudaMemset for zero (fast), but -INF needs a real kernel because
 * float -inf has bit pattern 0xff800000. Use one launcher for each. */
__global__ static void k_fill_neg_inf(float *p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = -INFINITY;
}

static int fill_neg_inf(float *p, size_t n) {
    if (!p || n == 0) return 0;
    int tpb = 256;
    size_t bpg = (n + (size_t)tpb - 1) / (size_t)tpb;
    /* CUDA block.x cap is huge but cap to 2^31-1 just in case. */
    if (bpg > (size_t)((1u << 31) - 1u)) bpg = (size_t)((1u << 31) - 1u);
    k_fill_neg_inf<<<(unsigned)bpg, tpb>>>(p, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr,
                     "session_state: fill_neg_inf launch error %s (%s)\n",
                     cudaGetErrorName(e), cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

/* Bump-pointer carve helper: hand out a 256-aligned slice of `bytes`
 * from `*cursor`, advance cursor, return the device pointer. */
static void *carve(uint8_t **cursor, size_t bytes) {
    uint8_t *p = *cursor;
    *cursor = p + align256(bytes);
    return (void *)p;
}

/* Initialize buffers in a freshly-carved state struct. Zero KVs +
 * residual_hc; -INF-fill score buffers; reset cursors / pos. Used by
 * both alloc (after carve) and reset. Returns 0 / -1. */
static int session_init_buffers(struct ds4_session_state *s) {
    s->pos = 0;
    s->n_tokens_processed = 0;

    /* residual_hc */
    if (s->residual_hc) {
        CK(cudaMemset(s->residual_hc, 0,
                      (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(float)));
    }
    if (s->residual_hc_scratch) {
        CK(cudaMemset(s->residual_hc_scratch, 0,
                      (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(float)));
    }
    /* activation arena (zero is fine; per-stage clobbered) */
    if (s->activation_arena && s->arena_size) {
        CK(cudaMemset(s->activation_arena, 0, s->arena_size));
    }

    for (int il = 0; il < (int)DS4_N_LAYER; il++) {
        struct ds4_layer_state *L = &s->layers[il];
        L->n_raw = 0;
        L->n_comp = 0;
        L->n_index_comp = 0;

        /* raw_kv */
        if (L->raw_kv) {
            CK(cudaMemset(L->raw_kv, 0,
                          (size_t)L->cap_raw * DS4_N_HEAD_DIM * sizeof(float)));
        }
        if (L->compress_ratio != 0) {
            const int coff = (L->compress_ratio == 4) ? 2 : 1;
            const size_t state_cells =
                (size_t)coff * (size_t)L->compress_ratio
                * (size_t)coff * DS4_N_HEAD_DIM;

            CK(cudaMemset(L->comp_kv, 0,
                          (size_t)L->cap_comp * DS4_N_HEAD_DIM * sizeof(float)));
            CK(cudaMemset(L->attn_state_kv, 0, state_cells * sizeof(float)));
            if (fill_neg_inf(L->attn_state_score, state_cells) != 0) return -1;

            if (L->has_indexer) {
                CK(cudaMemset(L->index_comp_kv, 0,
                              (size_t)L->cap_comp * DS4_N_INDEXER_HEAD_DIM
                              * sizeof(float)));
                const size_t idx_cells = (size_t)coff * (size_t)L->compress_ratio
                                         * (size_t)coff * DS4_N_INDEXER_HEAD_DIM;
                CK(cudaMemset(L->index_state_kv, 0, idx_cells * sizeof(float)));
                if (fill_neg_inf(L->index_state_score, idx_cells) != 0) return -1;
            }
        }
    }

    /* Wait for all the inits to land so the caller observes a clean
     * state on the first kernel that reads the buffers. */
    CK(cudaDeviceSynchronize());
    return 0;
}

extern "C" int ds4_session_state_alloc(struct ds4_session_state *out,
                                       int max_context) {
    if (!out || max_context <= 0) {
        std::fprintf(stderr,
                     "session_state: invalid args (out=%p, max_context=%d)\n",
                     (void *)out, max_context);
        return -1;
    }
    std::memset(out, 0, sizeof(*out));
    out->max_context = max_context;

    const int cap_raw = (int)DS4_N_SWA < max_context ? (int)DS4_N_SWA : max_context;

    const size_t total = ds4_session_state_arena_bytes(max_context);
    out->state_arena_bytes = total;

    void *base = nullptr;
    cudaError_t e = cudaMalloc(&base, total);
    if (e != cudaSuccess) {
        std::fprintf(stderr,
                     "session_state: cudaMalloc(%zu B) failed: %s (%s)\n",
                     total, cudaGetErrorName(e), cudaGetErrorString(e));
        std::memset(out, 0, sizeof(*out));
        return -1;
    }
    out->state_arena_base = base;

    /* Carve sub-pointers via a bump pointer over the arena. */
    uint8_t *cur = (uint8_t *)base;

    for (int il = 0; il < (int)DS4_N_LAYER; il++) {
        struct ds4_layer_state *L = &out->layers[il];
        const int ratio = layer_ratio(il);
        const int cap_comp = compute_cap_comp(max_context, ratio);

        L->il = il;
        L->compress_ratio = ratio;
        L->has_indexer = (ratio == 4) ? 1 : 0;
        L->cap_raw = cap_raw;
        L->cap_comp = cap_comp;

        L->raw_kv = (float *)carve(&cur,
            (size_t)cap_raw * DS4_N_HEAD_DIM * sizeof(float));

        if (ratio != 0) {
            L->comp_kv = (float *)carve(&cur,
                (size_t)cap_comp * DS4_N_HEAD_DIM * sizeof(float));
            const int coff = (ratio == 4) ? 2 : 1;
            const size_t state_cells = (size_t)coff * (size_t)ratio
                                       * (size_t)coff * DS4_N_HEAD_DIM;
            L->attn_state_kv    = (float *)carve(&cur,
                state_cells * sizeof(float));
            L->attn_state_score = (float *)carve(&cur,
                state_cells * sizeof(float));

            if (ratio == 4) {
                L->index_comp_kv = (float *)carve(&cur,
                    (size_t)cap_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                const size_t idx_cells = (size_t)coff * (size_t)ratio
                                         * (size_t)coff * DS4_N_INDEXER_HEAD_DIM;
                L->index_state_kv    = (float *)carve(&cur,
                    idx_cells * sizeof(float));
                L->index_state_score = (float *)carve(&cur,
                    idx_cells * sizeof(float));
            }
        }
    }

    /* residual_hc ping/pong */
    out->residual_hc = (float *)carve(&cur,
        (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(float));
    out->residual_hc_scratch = (float *)carve(&cur,
        (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(float));

    /* activation arena */
    out->arena_size = DS4_SESSION_ACTIVATION_ARENA_BYTES;
    out->activation_arena = (float *)carve(&cur, out->arena_size);

    /* Sanity: cursor must not exceed the arena. */
    const size_t consumed = (size_t)(cur - (uint8_t *)base);
    if (consumed > total) {
        std::fprintf(stderr,
            "session_state: arena overflow (consumed=%zu > total=%zu)\n",
            consumed, total);
        cudaFree(base);
        std::memset(out, 0, sizeof(*out));
        return -1;
    }

    /* Initialize buffers (zero KVs + residual_hc, -INF score buffers). */
    if (session_init_buffers(out) != 0) {
        cudaFree(base);
        std::memset(out, 0, sizeof(*out));
        return -1;
    }
    return 0;
}

extern "C" void ds4_session_state_free(struct ds4_session_state *s) {
    if (!s) return;
    ds4_session_state_free_resident_moe_down_soa(s);
    if (s->state_arena_base) {
        cudaFree(s->state_arena_base);
    }
    std::memset(s, 0, sizeof(*s));
}

extern "C" void ds4_session_state_free_resident_moe_down_soa(
        struct ds4_session_state *s) {
    if (!s) return;
    for (int il = 0; il < (int)DS4_N_LAYER; il++) {
        struct ds4_layer_state *L = &s->layers[il];
        if (L->ffn_down_soa_scales) cudaFree(L->ffn_down_soa_scales);
        if (L->ffn_down_soa_qs) cudaFree(L->ffn_down_soa_qs);
        if (L->ffn_down_soa_d) cudaFree(L->ffn_down_soa_d);
        if (L->ffn_down_soa_dmin) cudaFree(L->ffn_down_soa_dmin);
        L->ffn_down_soa_scales = nullptr;
        L->ffn_down_soa_qs = nullptr;
        L->ffn_down_soa_d = nullptr;
        L->ffn_down_soa_dmin = nullptr;
    }
}

extern "C" void ds4_session_state_reset(struct ds4_session_state *s) {
    if (!s || !s->state_arena_base) return;
    /* session_init_buffers re-zeros / re-fills using the existing
     * sub-pointers; arena pointers stay valid. */
    (void)session_init_buffers(s);
}
