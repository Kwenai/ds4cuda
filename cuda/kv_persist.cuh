// kv_persist.cuh — disk KV cache: save / load full ds4_session_state
// (KV cache + compressor state) to/from a single binary file so that
// restart-after-prefill arrives at the first decoded token in seconds
// instead of re-running prefill.
//
// File format (v1, self-contained, deterministic, NOT compatible with
// ds4 KVC payload):
//
//   [ ds4cuda_kvc_header ] (fixed-size, 4 KiB padded)
//   [ for il in 0..n_layer-1:
//       [ ds4cuda_kvc_layer_meta ]            (fixed 64 B)
//       [ raw_kv  bytes  (cap_raw  * HEAD_DIM * 4) ]
//       if compress_ratio != 0:
//         [ comp_kv          bytes (cap_comp * HEAD_DIM * 4)        ]
//         [ attn_state_kv    bytes (state_cells * 4)                ]
//         [ attn_state_score bytes (state_cells * 4)                ]
//       if has_indexer:
//         [ index_comp_kv    bytes (cap_comp * INDEXER_HEAD_DIM * 4) ]
//         [ index_state_kv   bytes (idx_cells * 4)                   ]
//         [ index_state_score bytes (idx_cells * 4)                  ]
//   ]
//   [ residual_hc bytes (N_HC * N_EMBD * 4) ]
//   [ trailer magic "DS4CUDA1" (8 B) for sanity ]
//
// Each per-buffer payload is written at full capacity (cap_raw / cap_comp)
// — counts of "used" rows live in the layer meta. This makes restore a
// straight cudaMemcpy HtoD with no per-row gymnastics, and guarantees a
// byte-equal forward result after save+load (the load is a deterministic
// device-side echo of the saved bytes).
//
// Minimal scope:
//   - no sha1(token_ids) auto-path (caller passes a fixed path)
//   - no cold/continued/evict/shutdown trigger logic
//   - no boundary trim / 2048-align retokenize protection
//   - no checkpoint logits in payload
//   - prompt_token_ids stored in header (capped at 64) for the load-time
//     identity check the caller may want; not used by the loader itself.

#ifndef DS4CUDA_KV_PERSIST_CUH
#define DS4CUDA_KV_PERSIST_CUH

#include <stddef.h>
#include <stdint.h>

#include "ds4cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS4CUDA_KVC_MAGIC          "DS4CUDAK"   /* 8 bytes, no NUL */
#define DS4CUDA_KVC_TRAILER        "DS4CUDA1"   /* 8 bytes, no NUL */
#define DS4CUDA_KVC_VERSION        1u
#define DS4CUDA_KVC_MAX_TOKENS_HDR 64           /* prompt_token_ids[] cap */

/* Header (fixed layout, written to disk verbatim). */
struct ds4cuda_kvc_header {
    char     magic[8];                    /* "DS4CUDAK"             */
    uint32_t version;                     /* DS4CUDA_KVC_VERSION    */
    uint32_t n_layer;                     /* DS4_N_LAYER (43)       */
    uint32_t max_context;                 /* session->max_context   */
    uint32_t pos;                         /* session->pos           */
    uint32_t n_tokens_processed;          /* session->n_tokens_processed */
    uint32_t n_prompt_tokens;             /* number of valid entries
                                             in prompt_token_ids[]  */
    uint32_t cap_raw;                     /* cap_raw at save time   */
    uint32_t reserved0;
    int32_t  prompt_token_ids[DS4CUDA_KVC_MAX_TOKENS_HDR];
    uint64_t reserved[8];
};

/* Per-layer meta (64 B). Written before that layer's buffer payload. */
struct ds4cuda_kvc_layer_meta {
    int32_t  il;
    int32_t  compress_ratio;
    int32_t  has_indexer;
    int32_t  cap_raw;
    int32_t  cap_comp;
    int32_t  n_raw;
    int32_t  n_comp;
    int32_t  n_index_comp;
    uint64_t bytes_after;                 /* total payload bytes for this
                                             layer (sanity, not load-critical) */
    uint64_t reserved[3];
};

/* ---- API -------------------------------------------------------- */

/* Serialize the full session_state (KV + compressor state + residual_hc)
 * to `path`. Memory: allocates one host buffer == max-per-layer-bytes
 * for the DtoH stage and reuses it for every layer, so peak host RSS
 * lift is ~max_layer_bytes (~600 KiB at max_ctx=64).
 *
 * Returns 0 on success; -1 on any I/O / CUDA error. The file is opened
 * with O_TRUNC so a partial write leaves the previous file zero-length.
 *
 * `prompt_token_ids` may be NULL when n_tokens == 0; the header field
 * is left zero-filled. If n_tokens > DS4CUDA_KVC_MAX_TOKENS_HDR only the
 * first DS4CUDA_KVC_MAX_TOKENS_HDR entries are stored (verification is
 * caller's choice; the loader does not enforce an identity check).
 */
int ds4_session_save_to_disk(const struct ds4_session_state *s,
                             const int32_t *prompt_token_ids,
                             int n_tokens,
                             const char *path);

/* Load a previously-saved session into `s_out`, which MUST have been
 * pre-allocated via ds4_session_state_alloc with a max_context that
 * matches the file's header.max_context. The function:
 *
 *   1. fread + validate header (magic, version, geometry).
 *   2. for each layer: fread meta + buffer payloads, cudaMemcpy HtoD
 *      into the corresponding device pointers in s_out.
 *   3. fread residual_hc + trailer + magic check.
 *   4. set s_out->pos / n_tokens_processed / per-layer n_raw / n_comp /
 *      n_index_comp from the header + per-layer meta.
 *
 * On success the activation_arena is left as-is (untouched by save/load
 * — the arena is bump-pointer scratch, not persistent state). Returns
 * 0 on success; -1 on any error. On error `s_out`'s counters / pos are
 * left in an indeterminate state — callers should treat the session as
 * invalid (call _reset before reuse).
 *
 * `out_pos` and `out_n_tokens` (both optional) receive the loaded pos
 * / n_tokens_processed for the caller's convenience.
 */
int ds4_session_load_from_disk(struct ds4_session_state *s_out,
                               const char *path,
                               int *out_pos,
                               int *out_n_tokens);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_KV_PERSIST_CUH */
