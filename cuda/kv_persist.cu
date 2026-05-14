// kv_persist.cu — disk KV cache: serialize / restore ds4_session_state.
//
// Implements the extern "C" API declared in cuda/kv_persist.cuh. See that
// header for the file format spec and overall design.
//
// Implementation notes:
//
//   - We dump every persistent device buffer at FULL CAPACITY (cap_raw or
//     cap_comp), not just the n_raw / n_comp prefix.  Two reasons:
//       * The session arena is initialized to zero for KV / -INF for score
//         buffers (session_state.cu:session_init_buffers).  Restoring full
//         capacity preserves byte-exact device memory state, which is the
//         disk KV cache acceptance gate (post-load forward output must be byte-equal
//         to a no-restart baseline).
//       * Skipping the unused tail rows would force the loader to zero /
//         -INF-fill them — extra kernel launches we don't need.  At
//         max_ctx=64 the saved unused rows total <1 MiB; not worth the
//         complexity.
//
//   - Save uses one reusable host bounce buffer sized to the largest
//     per-layer payload (the ratio-128 attn_state at max_ctx=4096 is
//     ~256 KiB, well under 1 MiB).  cudaMemcpy DtoH per buffer; fwrite
//     directly out.  No double-copy.
//
//   - Load is the symmetric inverse.  We validate the header against the
//     pre-allocated session geometry (max_context, n_layer) so that a
//     mismatched cache file is rejected up-front rather than producing
//     a corrupt KV cache.
//
//   - Residual_hc is dumped/restored even though forward_token rebuilds
//     it from the embedding row at the start of every call — including
//     it costs only 64 KiB and keeps the file fully self-contained
//     (future work: prefill checkpoint with non-trivial residual_hc).
//
//   - All cudaMemcpy calls use the default stream and we
//     cudaDeviceSynchronize at the start of save (DtoH wants stable data)
//     and at the end of load (so the caller sees populated buffers on
//     return).  Save is tens of ms at max_ctx=64; one sync per call is
//     fine.

#include "kv_persist.cuh"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(stmt)                                                          \
    do {                                                                  \
        cudaError_t _e = (stmt);                                          \
        if (_e != cudaSuccess) {                                          \
            std::fprintf(stderr,                                          \
                         "kv_persist: CUDA error %s (%s) at %s:%d\n",     \
                         cudaGetErrorName(_e), cudaGetErrorString(_e),    \
                         __FILE__, __LINE__);                             \
            return -1;                                                    \
        }                                                                 \
    } while (0)

namespace {

/* coff (compressor offset) per ratio — mirrors session_state.cu. */
static int compressor_coff(int ratio) { return (ratio == 4) ? 2 : 1; }

/* state_cells helper (attn): coff*ratio rows × coff*HEAD_DIM cols. */
static size_t attn_state_cells(int ratio) {
    int coff = compressor_coff(ratio);
    return (size_t)coff * (size_t)ratio
         * (size_t)coff * (size_t)DS4_N_HEAD_DIM;
}

/* state_cells helper (indexer): coff*ratio rows × coff*INDEXER_HEAD_DIM cols. */
static size_t idx_state_cells(int ratio) {
    int coff = compressor_coff(ratio);
    return (size_t)coff * (size_t)ratio
         * (size_t)coff * (size_t)DS4_N_INDEXER_HEAD_DIM;
}

/* Largest per-layer payload across the table — used to size the host
 * bounce buffer.  raw_kv (cap_raw=128) + comp_kv (cap_comp ≤ max_ctx/4+2)
 * + state buffers + indexer buffers.  We compute it as the per-buffer
 * max, since save/load goes one buffer at a time. */
static size_t max_single_buffer_bytes(const struct ds4_session_state *s) {
    size_t mx = 0;
    /* raw_kv */
    {
        size_t b = (size_t)s->layers[0].cap_raw
                   * (size_t)DS4_N_HEAD_DIM * sizeof(float);
        if (b > mx) mx = b;
    }
    for (int il = 0; il < (int)DS4_N_LAYER; ++il) {
        const struct ds4_layer_state *L = &s->layers[il];
        if (L->compress_ratio == 0) continue;
        size_t b_comp = (size_t)L->cap_comp
                        * (size_t)DS4_N_HEAD_DIM * sizeof(float);
        if (b_comp > mx) mx = b_comp;
        size_t b_state = attn_state_cells(L->compress_ratio) * sizeof(float);
        if (b_state > mx) mx = b_state;
        if (L->has_indexer) {
            size_t b_idx = (size_t)L->cap_comp
                           * (size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            if (b_idx > mx) mx = b_idx;
            size_t b_idxs = idx_state_cells(L->compress_ratio) * sizeof(float);
            if (b_idxs > mx) mx = b_idxs;
        }
    }
    /* residual_hc */
    {
        size_t b = (size_t)DS4_N_HC * (size_t)DS4_N_EMBD * sizeof(float);
        if (b > mx) mx = b;
    }
    return mx;
}

/* Sum of all per-layer payload bytes (used for the ds4cuda_kvc_layer_meta
 * .bytes_after sanity field). */
static uint64_t per_layer_payload_bytes(const struct ds4_layer_state *L) {
    uint64_t total = 0;
    total += (uint64_t)L->cap_raw * (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    if (L->compress_ratio != 0) {
        total += (uint64_t)L->cap_comp * (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
        total += (uint64_t)attn_state_cells(L->compress_ratio) * sizeof(float);
        total += (uint64_t)attn_state_cells(L->compress_ratio) * sizeof(float);
        if (L->has_indexer) {
            total += (uint64_t)L->cap_comp * (uint64_t)DS4_N_INDEXER_HEAD_DIM
                     * sizeof(float);
            total += (uint64_t)idx_state_cells(L->compress_ratio) * sizeof(float);
            total += (uint64_t)idx_state_cells(L->compress_ratio) * sizeof(float);
        }
    }
    return total;
}

/* DtoH + fwrite one device buffer of `bytes` bytes via the host bounce
 * buffer. Returns 0/-1. */
static int dump_buffer(const void *d_src, size_t bytes,
                       void *h_bounce, FILE *fp,
                       const char *tag) {
    if (bytes == 0) return 0;
    cudaError_t e = cudaMemcpy(h_bounce, d_src, bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(stderr,
            "kv_persist: cudaMemcpy DtoH failed at '%s' (%zu B): %s\n",
            tag, bytes, cudaGetErrorString(e));
        return -1;
    }
    size_t w = std::fwrite(h_bounce, 1, bytes, fp);
    if (w != bytes) {
        std::fprintf(stderr,
            "kv_persist: fwrite short for '%s' (%zu / %zu): %s\n",
            tag, w, bytes, std::strerror(errno));
        return -1;
    }
    return 0;
}

/* fread + HtoD into a device buffer.  Returns 0/-1. */
static int load_buffer(void *d_dst, size_t bytes,
                       void *h_bounce, FILE *fp,
                       const char *tag) {
    if (bytes == 0) return 0;
    size_t r = std::fread(h_bounce, 1, bytes, fp);
    if (r != bytes) {
        std::fprintf(stderr,
            "kv_persist: fread short for '%s' (%zu / %zu): %s\n",
            tag, r, bytes,
            std::feof(fp) ? "unexpected eof" : std::strerror(errno));
        return -1;
    }
    cudaError_t e = cudaMemcpy(d_dst, h_bounce, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        std::fprintf(stderr,
            "kv_persist: cudaMemcpy HtoD failed at '%s' (%zu B): %s\n",
            tag, bytes, cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

} /* namespace */

extern "C" int ds4_session_save_to_disk(const struct ds4_session_state *s,
                                        const int32_t *prompt_token_ids,
                                        int n_tokens,
                                        const char *path) {
    if (!s || !path) {
        std::fprintf(stderr, "kv_persist: NULL session_state or path\n");
        return -1;
    }
    if (!s->state_arena_base) {
        std::fprintf(stderr, "kv_persist: session_state has no arena\n");
        return -1;
    }
    if (n_tokens < 0) n_tokens = 0;

    /* Make sure all prior kernels have committed before DtoH copies. */
    CK(cudaDeviceSynchronize());

    FILE *fp = std::fopen(path, "wb");
    if (!fp) {
        std::fprintf(stderr, "kv_persist: fopen('%s', wb) failed: %s\n",
                     path, std::strerror(errno));
        return -1;
    }

    /* ----- Header --------------------------------------------------- */
    struct ds4cuda_kvc_header hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    std::memcpy(hdr.magic, DS4CUDA_KVC_MAGIC, 8);
    hdr.version             = DS4CUDA_KVC_VERSION;
    hdr.n_layer             = DS4_N_LAYER;
    hdr.max_context         = (uint32_t)s->max_context;
    hdr.pos                 = (uint32_t)s->pos;
    hdr.n_tokens_processed  = (uint32_t)s->n_tokens_processed;
    hdr.n_prompt_tokens     = (uint32_t)n_tokens;
    hdr.cap_raw             = (uint32_t)s->layers[0].cap_raw;
    {
        const int n_copy = n_tokens < DS4CUDA_KVC_MAX_TOKENS_HDR
                         ? n_tokens : DS4CUDA_KVC_MAX_TOKENS_HDR;
        if (prompt_token_ids && n_copy > 0) {
            std::memcpy(hdr.prompt_token_ids, prompt_token_ids,
                        (size_t)n_copy * sizeof(int32_t));
        }
    }
    if (std::fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        std::fprintf(stderr, "kv_persist: fwrite header failed: %s\n",
                     std::strerror(errno));
        std::fclose(fp);
        return -1;
    }

    /* ----- Host bounce buffer -------------------------------------- */
    size_t bounce_sz = max_single_buffer_bytes(s);
    void *h_bounce = std::malloc(bounce_sz);
    if (!h_bounce) {
        std::fprintf(stderr, "kv_persist: malloc(%zu) bounce buffer failed\n",
                     bounce_sz);
        std::fclose(fp);
        return -1;
    }

    /* ----- Per-layer payloads -------------------------------------- */
    for (int il = 0; il < (int)DS4_N_LAYER; ++il) {
        const struct ds4_layer_state *L = &s->layers[il];

        struct ds4cuda_kvc_layer_meta meta;
        std::memset(&meta, 0, sizeof(meta));
        meta.il             = L->il;
        meta.compress_ratio = L->compress_ratio;
        meta.has_indexer    = L->has_indexer;
        meta.cap_raw        = L->cap_raw;
        meta.cap_comp       = L->cap_comp;
        meta.n_raw          = L->n_raw;
        meta.n_comp         = L->n_comp;
        meta.n_index_comp   = L->n_index_comp;
        meta.bytes_after    = per_layer_payload_bytes(L);

        if (std::fwrite(&meta, sizeof(meta), 1, fp) != 1) {
            std::fprintf(stderr,
                "kv_persist: fwrite layer meta il=%d failed: %s\n",
                il, std::strerror(errno));
            std::free(h_bounce);
            std::fclose(fp);
            return -1;
        }

        /* raw_kv */
        size_t raw_bytes = (size_t)L->cap_raw
                           * (size_t)DS4_N_HEAD_DIM * sizeof(float);
        if (dump_buffer(L->raw_kv, raw_bytes, h_bounce, fp, "raw_kv") != 0) {
            std::free(h_bounce); std::fclose(fp); return -1;
        }

        if (L->compress_ratio != 0) {
            size_t comp_bytes = (size_t)L->cap_comp
                                * (size_t)DS4_N_HEAD_DIM * sizeof(float);
            if (dump_buffer(L->comp_kv, comp_bytes, h_bounce, fp,
                            "comp_kv") != 0) {
                std::free(h_bounce); std::fclose(fp); return -1;
            }
            size_t state_bytes = attn_state_cells(L->compress_ratio)
                                  * sizeof(float);
            if (dump_buffer(L->attn_state_kv, state_bytes, h_bounce, fp,
                            "attn_state_kv") != 0) {
                std::free(h_bounce); std::fclose(fp); return -1;
            }
            if (dump_buffer(L->attn_state_score, state_bytes, h_bounce, fp,
                            "attn_state_score") != 0) {
                std::free(h_bounce); std::fclose(fp); return -1;
            }

            if (L->has_indexer) {
                size_t idx_kv_bytes = (size_t)L->cap_comp
                                      * (size_t)DS4_N_INDEXER_HEAD_DIM
                                      * sizeof(float);
                if (dump_buffer(L->index_comp_kv, idx_kv_bytes, h_bounce, fp,
                                "index_comp_kv") != 0) {
                    std::free(h_bounce); std::fclose(fp); return -1;
                }
                size_t idx_state_bytes = idx_state_cells(L->compress_ratio)
                                          * sizeof(float);
                if (dump_buffer(L->index_state_kv, idx_state_bytes, h_bounce,
                                fp, "index_state_kv") != 0) {
                    std::free(h_bounce); std::fclose(fp); return -1;
                }
                if (dump_buffer(L->index_state_score, idx_state_bytes, h_bounce,
                                fp, "index_state_score") != 0) {
                    std::free(h_bounce); std::fclose(fp); return -1;
                }
            }
        }
    }

    /* ----- residual_hc --------------------------------------------- */
    if (s->residual_hc) {
        size_t hc_bytes = (size_t)DS4_N_HC * (size_t)DS4_N_EMBD * sizeof(float);
        if (dump_buffer(s->residual_hc, hc_bytes, h_bounce, fp,
                        "residual_hc") != 0) {
            std::free(h_bounce); std::fclose(fp); return -1;
        }
    } else {
        /* Should never happen on a properly-allocated state, but write
         * zero bytes to keep the file format strict. */
        std::fprintf(stderr,
            "kv_persist: warning, residual_hc is NULL; writing zeros\n");
        size_t hc_bytes = (size_t)DS4_N_HC * (size_t)DS4_N_EMBD * sizeof(float);
        std::memset(h_bounce, 0, hc_bytes);
        if (std::fwrite(h_bounce, 1, hc_bytes, fp) != hc_bytes) {
            std::free(h_bounce); std::fclose(fp); return -1;
        }
    }

    /* ----- Trailer -------------------------------------------------- */
    if (std::fwrite(DS4CUDA_KVC_TRAILER, 1, 8, fp) != 8) {
        std::fprintf(stderr, "kv_persist: fwrite trailer failed\n");
        std::free(h_bounce);
        std::fclose(fp);
        return -1;
    }

    std::free(h_bounce);
    if (std::fclose(fp) != 0) {
        std::fprintf(stderr, "kv_persist: fclose failed: %s\n",
                     std::strerror(errno));
        return -1;
    }
    return 0;
}

extern "C" int ds4_session_load_from_disk(struct ds4_session_state *s_out,
                                          const char *path,
                                          int *out_pos,
                                          int *out_n_tokens) {
    if (!s_out || !path) {
        std::fprintf(stderr, "kv_persist: NULL session_state or path\n");
        return -1;
    }
    if (!s_out->state_arena_base) {
        std::fprintf(stderr,
            "kv_persist: session_state has no arena (call _alloc first)\n");
        return -1;
    }

    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "kv_persist: fopen('%s', rb) failed: %s\n",
                     path, std::strerror(errno));
        return -1;
    }

    /* ----- Header --------------------------------------------------- */
    struct ds4cuda_kvc_header hdr;
    if (std::fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        std::fprintf(stderr, "kv_persist: short header read\n");
        std::fclose(fp);
        return -1;
    }
    if (std::memcmp(hdr.magic, DS4CUDA_KVC_MAGIC, 8) != 0) {
        std::fprintf(stderr,
            "kv_persist: bad magic (expected %.8s)\n", DS4CUDA_KVC_MAGIC);
        std::fclose(fp);
        return -1;
    }
    if (hdr.version != DS4CUDA_KVC_VERSION) {
        std::fprintf(stderr,
            "kv_persist: bad version (got %u, expected %u)\n",
            hdr.version, DS4CUDA_KVC_VERSION);
        std::fclose(fp);
        return -1;
    }
    if (hdr.n_layer != DS4_N_LAYER) {
        std::fprintf(stderr,
            "kv_persist: n_layer mismatch (got %u, expected %u)\n",
            hdr.n_layer, (unsigned)DS4_N_LAYER);
        std::fclose(fp);
        return -1;
    }
    if ((int)hdr.max_context != s_out->max_context) {
        std::fprintf(stderr,
            "kv_persist: max_context mismatch (file=%u, session=%d)\n",
            hdr.max_context, s_out->max_context);
        std::fclose(fp);
        return -1;
    }

    /* ----- Host bounce buffer -------------------------------------- */
    size_t bounce_sz = max_single_buffer_bytes(s_out);
    void *h_bounce = std::malloc(bounce_sz);
    if (!h_bounce) {
        std::fprintf(stderr, "kv_persist: malloc(%zu) bounce failed\n",
                     bounce_sz);
        std::fclose(fp);
        return -1;
    }

    /* ----- Per-layer payloads -------------------------------------- */
    for (int il = 0; il < (int)DS4_N_LAYER; ++il) {
        struct ds4_layer_state *L = &s_out->layers[il];

        struct ds4cuda_kvc_layer_meta meta;
        if (std::fread(&meta, sizeof(meta), 1, fp) != 1) {
            std::fprintf(stderr,
                "kv_persist: short read for layer meta il=%d\n", il);
            std::free(h_bounce); std::fclose(fp);
            return -1;
        }
        if (meta.il != il) {
            std::fprintf(stderr,
                "kv_persist: layer meta il mismatch (got %d, expected %d)\n",
                meta.il, il);
            std::free(h_bounce); std::fclose(fp);
            return -1;
        }
        if (meta.compress_ratio != L->compress_ratio
            || meta.has_indexer != L->has_indexer
            || meta.cap_raw     != L->cap_raw
            || meta.cap_comp    != L->cap_comp) {
            std::fprintf(stderr,
                "kv_persist: layer geometry mismatch il=%d "
                "(file ratio=%d ind=%d cap_raw=%d cap_comp=%d, "
                "session ratio=%d ind=%d cap_raw=%d cap_comp=%d)\n",
                il, meta.compress_ratio, meta.has_indexer,
                meta.cap_raw, meta.cap_comp,
                L->compress_ratio, L->has_indexer,
                L->cap_raw, L->cap_comp);
            std::free(h_bounce); std::fclose(fp);
            return -1;
        }
        L->n_raw        = meta.n_raw;
        L->n_comp       = meta.n_comp;
        L->n_index_comp = meta.n_index_comp;

        /* raw_kv */
        size_t raw_bytes = (size_t)L->cap_raw
                           * (size_t)DS4_N_HEAD_DIM * sizeof(float);
        if (load_buffer(L->raw_kv, raw_bytes, h_bounce, fp, "raw_kv") != 0) {
            std::free(h_bounce); std::fclose(fp); return -1;
        }

        if (L->compress_ratio != 0) {
            size_t comp_bytes = (size_t)L->cap_comp
                                * (size_t)DS4_N_HEAD_DIM * sizeof(float);
            if (load_buffer(L->comp_kv, comp_bytes, h_bounce, fp,
                            "comp_kv") != 0) {
                std::free(h_bounce); std::fclose(fp); return -1;
            }
            size_t state_bytes = attn_state_cells(L->compress_ratio)
                                  * sizeof(float);
            if (load_buffer(L->attn_state_kv, state_bytes, h_bounce, fp,
                            "attn_state_kv") != 0) {
                std::free(h_bounce); std::fclose(fp); return -1;
            }
            if (load_buffer(L->attn_state_score, state_bytes, h_bounce, fp,
                            "attn_state_score") != 0) {
                std::free(h_bounce); std::fclose(fp); return -1;
            }

            if (L->has_indexer) {
                size_t idx_kv_bytes = (size_t)L->cap_comp
                                      * (size_t)DS4_N_INDEXER_HEAD_DIM
                                      * sizeof(float);
                if (load_buffer(L->index_comp_kv, idx_kv_bytes, h_bounce, fp,
                                "index_comp_kv") != 0) {
                    std::free(h_bounce); std::fclose(fp); return -1;
                }
                size_t idx_state_bytes = idx_state_cells(L->compress_ratio)
                                          * sizeof(float);
                if (load_buffer(L->index_state_kv, idx_state_bytes, h_bounce,
                                fp, "index_state_kv") != 0) {
                    std::free(h_bounce); std::fclose(fp); return -1;
                }
                if (load_buffer(L->index_state_score, idx_state_bytes, h_bounce,
                                fp, "index_state_score") != 0) {
                    std::free(h_bounce); std::fclose(fp); return -1;
                }
            }
        }
    }

    /* ----- residual_hc --------------------------------------------- */
    if (s_out->residual_hc) {
        size_t hc_bytes = (size_t)DS4_N_HC * (size_t)DS4_N_EMBD * sizeof(float);
        if (load_buffer(s_out->residual_hc, hc_bytes, h_bounce, fp,
                        "residual_hc") != 0) {
            std::free(h_bounce); std::fclose(fp); return -1;
        }
    }

    /* ----- Trailer -------------------------------------------------- */
    char trailer[8] = {0};
    if (std::fread(trailer, 1, 8, fp) != 8) {
        std::fprintf(stderr, "kv_persist: short trailer read\n");
        std::free(h_bounce); std::fclose(fp); return -1;
    }
    if (std::memcmp(trailer, DS4CUDA_KVC_TRAILER, 8) != 0) {
        std::fprintf(stderr,
            "kv_persist: bad trailer (expected %.8s, got %.8s)\n",
            DS4CUDA_KVC_TRAILER, trailer);
        std::free(h_bounce); std::fclose(fp); return -1;
    }

    /* Restore session-level counters. */
    s_out->pos                = (int)hdr.pos;
    s_out->n_tokens_processed = (int)hdr.n_tokens_processed;

    if (out_pos)        *out_pos        = s_out->pos;
    if (out_n_tokens)   *out_n_tokens   = s_out->n_tokens_processed;

    std::free(h_bounce);
    std::fclose(fp);

    /* Make sure HtoD copies have landed before the caller dispatches a
     * forward. */
    CK(cudaDeviceSynchronize());
    return 0;
}
