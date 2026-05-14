/*
 * inference_engine.cu — Single-session FIFO inference engine.
 *
 * Drives ds4_forward_token for the HTTP server endpoints (OpenAI,
 * Anthropic). The engine is C-callable; it is implemented in CUDA C++
 * because it needs to invoke ds4cuda::ds4_forward_token directly. When
 * an endpoint's generator slot is left unset, the endpoint falls back
 * to a built-in echo generator — that's a sanity path, not the live
 * decoder.
 *
 * Lifecycle:
 *
 *   ds4cuda_inference_engine_create
 *     -> ds4_model_open(GGUF)
 *     -> ds4_model_load_to_managed                  (~70 s warm)
 *     -> ds4_session_state_alloc(max_context)
 *     -> ds4cuda_tokenizer_init
 *     -> spawn one worker thread
 *
 *   ds4cuda_real_{buffered,stream,anthropic}_generator
 *     -> push job onto bounded FIFO
 *     -> for buffered: wait on per-job condvar for the worker
 *        to fill `result` then unblock
 *     -> for stream: the worker calls back into emit() one token at a
 *        time on the WORKER thread (we hold a per-job mutex while
 *        emit() runs so the listener cannot tear the job down early).
 *
 *   ds4cuda_inference_engine_destroy
 *     -> set stop flag, broadcast, join worker
 *     -> ds4_session_state_free + ds4_model_managed_free + ds4_model_close
 *
 * Worker loop per job:
 *
 *   1. Tokenize the DSML prompt with ds4cuda_tokenize  (chat-aware).
 *   2. ds4_session_state_reset (each request starts with empty KV).
 *   3. for t in 0..n_prompt-1:
 *          forward_token(prompt[t], pos=t)             [PREFILL]
 *      Take argmax of last-prompt-token logits (= first generated id).
 *   4. for n in 0..max_new_tokens:
 *          if streaming: emit(decode(tok_n))
 *          if EOS: stop with finish_reason=stop
 *          forward_token(tok_n, pos=n_prompt+n)        [DECODE]
 *          tok_n+1 = argmax(logits)
 *      The decode step's logits feed the NEXT step's argmax.  This
 *      mirrors ds4 generate_loop_cpu (ds4.c:8240-8290): each forward
 *      consumes one input token + position and returns the logits for
 *      the NEXT token.
 *
 * Determinism: greedy argmax with no temperature, no sampling.  This
 * matches the terminal argmax acceptance gate (argmax of "Hello" prefill = 2581
 * = "\n").  Tests for byte-equality across two runs rely on this.
 *
 * RSS budget (DGX Spark, 119.6 GiB visible, 113 GiB cap):
 *   81 GB managed weights
 *   ~92 MB session state  (max_context=64; ~700 MB at max_context=4096)
 *   ~16 MB activation arena
 *   negligible host-side state
 *   total ~ 81.2 GB — comfortably under the cap.
 */

extern "C" {
#include "inference_engine.h"
#include "../tokenizer/tokenizer.h"
#include "ds4cuda.h"
#include "ds4cuda_soa_layout.h"
#include "ds4cuda_iq2_soa_layout.h"
}

#include "../../cuda/forward_token.cuh"
#include "../../cuda/argmax.cuh"
#include "../../cuda/kv_persist.cuh"
#include "../../cuda/moe_q2k_sum6.cuh"

#include <cuda_runtime.h>

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* ------------------------------------------------------------------ */
/* Logging helper.                                                      */
/* ------------------------------------------------------------------ */
#define IELOG(fmt, ...) \
    do { fprintf(stderr, "[ds4cuda-engine] " fmt "\n", ##__VA_ARGS__); } while (0)

static uint64_t mem_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return UINT64_MAX;
    char key[64], unit[32];
    unsigned long long value_kib = 0;
    while (fscanf(f, "%63s %llu %31s\n", key, &value_kib, unit) == 3) {
        if (strcmp(key, "MemAvailable:") == 0) {
            fclose(f);
            return (uint64_t)value_kib * 1024ull;
        }
    }
    fclose(f);
    return UINT64_MAX;
}

static int env_flag_enabled(const char *name) {
    const char *v = getenv(name);
    return v && v[0] != '\0' && strcmp(v, "0") != 0 &&
           strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0;
}

static enum ds4_weight_backend env_weight_backend(void) {
    const char *v = getenv("DS4CUDA_WEIGHT_BACKEND");
    if (!v || v[0] == '\0') return DS4_WEIGHT_BACKEND_MANAGED;
    if (strcasecmp(v, "mmap_direct") == 0 ||
        strcasecmp(v, "mmap-direct") == 0 ||
        strcasecmp(v, "mmap") == 0) {
        return DS4_WEIGHT_BACKEND_MMAP_DIRECT;
    }
    return DS4_WEIGHT_BACKEND_MANAGED;
}

static int validate_weight_backend(enum ds4_weight_backend backend) {
    if (backend != DS4_WEIGHT_BACKEND_MMAP_DIRECT) return 0;

    int dev = 0;
    cudaError_t ce = cudaGetDevice(&dev);
    if (ce != cudaSuccess) {
        IELOG("cudaGetDevice failed while enabling mmap_direct: %s",
              cudaGetErrorString(ce));
        return -EIO;
    }

    int pageable = 0;
    ce = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess,
                                dev);
    if (ce != cudaSuccess || !pageable) {
        IELOG("DS4CUDA_WEIGHT_BACKEND=mmap_direct requires "
              "cudaDevAttrPageableMemoryAccess=1 (dev=%d attr_rc=%s value=%d)",
              dev, cudaGetErrorString(ce), pageable);
        return -ENOTSUP;
    }
    return 0;
}

static int env_int_or_default(const char *name, int defval) {
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return defval;
    char *end = NULL;
    long x = strtol(v, &end, 10);
    if (end == v) return defval;
    if (x < 0) return 0;
    if (x > 1000000) return 1000000;
    return (int)x;
}

static int parse_layer_list(const char *spec, int *out, int max_out) {
    if (!spec || spec[0] == '\0' || !out || max_out <= 0) return 0;
    int n = 0;
    const char *p = spec;
    while (*p && n < max_out) {
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p) break;

        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v >= 0 && v < (long)DS4_N_LAYER) {
            int dup = 0;
            for (int i = 0; i < n; i++) {
                if (out[i] == (int)v) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) out[n++] = (int)v;
        }
        p = end;
    }
    return n;
}

static uint64_t resident_moe_down_soa_layer_bytes(void) {
    const uint64_t n_experts = 256ull;
    const uint64_t out_dim = 4096ull;
    const uint64_t n_blocks_in = 2048ull / 256ull;
    return n_experts * out_dim * 16ull * n_blocks_in +
           n_experts * out_dim * 64ull * n_blocks_in +
           n_experts * out_dim * n_blocks_in * sizeof(uint16_t) +
           n_experts * out_dim * n_blocks_in * sizeof(uint16_t);
}

static int find_layer_tensor_name(char *buf, size_t n, int il, const char *suffix) {
    return snprintf(buf, n, "blk.%d.%s", il, suffix) > 0 ? 0 : -1;
}

static int build_resident_moe_down_soa(struct ds4cuda_inference_engine *e);

/* ------------------------------------------------------------------ */
/* Job & queue.                                                         */
/* ------------------------------------------------------------------ */

/* A job produces output one of two ways:
 *   - BUFFERED: worker fills j->result with malloc()'d UTF-8.
 *   - STREAM: worker calls j->emit(token, len, j->emit_user_data)
 *     once per generated token, on the worker thread.  The listener
 *     waits on j->done_cv for the worker to finish, then returns. */
enum job_mode {
    JOB_BUFFERED = 1,
    JOB_STREAM   = 2,
    /* Prefix-sync admin jobs.  Run on the worker thread so they are
     * naturally serialized with generation jobs and observe a quiescent
     * session_state (no in-flight forward).  The worker dispatches by
     * mode and never enters engine_run_job for these. */
    JOB_SAVE     = 3,
    JOB_LOAD     = 4,
};

struct ie_job {
    int                            mode;        /* enum job_mode */
    /* Inputs */
    char                          *prompt;      /* malloc'd; freed by worker */
    int                            max_new_tokens;
    /* Streaming output channel (mode == STREAM) */
    ds4cuda_chat_stream_emit_fn    emit;
    void                          *emit_user_data;
    /* Buffered output (mode == BUFFERED) — set by worker on success */
    char                          *result;
    /* Admin jobs (SAVE / LOAD) — caller-supplied path. */
    char                          *path;
    /* Common: emit_rc on stream, gen_rc on buffered.  0 = ok. */
    int                            rc;
    /* finish_reason text — "stop" or "length".  Caller may inspect. */
    char                           finish_reason[16];
    /* Synchronization: caller waits on done_cv until done==1. */
    pthread_mutex_t                lock;
    pthread_cond_t                 done_cv;
    int                            done;
    /* Linked list */
    struct ie_job                 *next;
};

static void job_init(struct ie_job *j) {
    memset(j, 0, sizeof(*j));
    pthread_mutex_init(&j->lock, NULL);
    pthread_cond_init(&j->done_cv, NULL);
    strcpy(j->finish_reason, "stop");
}

static void job_destroy(struct ie_job *j) {
    pthread_mutex_destroy(&j->lock);
    pthread_cond_destroy(&j->done_cv);
    free(j->prompt);
    free(j->path);
    /* j->result is owned by the caller on success; freed there. On
     * failure the worker also leaves it NULL. */
}

/* ------------------------------------------------------------------ */
/* Engine.                                                              */
/* ------------------------------------------------------------------ */
struct ds4cuda_inference_engine {
    /* CUDA-resident model + session */
    struct ds4_model               model;
    struct ds4_session_state       session;
    int                           *d_argmax;
    int                            moe_down_soa_layers;
    int                            max_context;
    int                            default_max_new_tokens;

    /* Tokenizer (host-only) */
    struct ds4cuda_tokenizer      *tk;
    int                            eos_id;
    int                            bos_id;

    /* Worker thread + queue */
    pthread_t                      worker;
    pthread_mutex_t                qlock;
    pthread_cond_t                 qcv_nonempty;
    pthread_cond_t                 qcv_nonfull;
    struct ie_job                 *q_head;
    struct ie_job                 *q_tail;
    int                            q_count;
    int                            q_capacity;
    int                            stop;
    int                            worker_ready;

    /* Prefix-sync: cached token-id sequence representing the FULL token
     * stream (prompt + generated) that is currently materialized in
     * session_state's KV cache.  Populated at the end of each successful
     * generation, consulted at the start of the next job to compute the
     * longest matching prefix `P` for prefix sync.
     *
     * Ownership:  worker thread reads + writes between jobs (no locking
     * needed because only one job runs at a time and main-thread admin
     * APIs save/load wait on the queue being empty).
     *
     * Invariants:
     *   - cached_n_tokens == 0  (cache empty / freshly reset session)
     *   - cached_n_tokens >  0  (session contains exactly these tokens
     *     in positions 0 .. cached_n_tokens-1; layer KV cursors == this)
     *
     * If the cache disagrees with the actual session_state (e.g. forward
     * failed midway through prefill), we MUST set cached_n_tokens=0 to
     * force a clean reset on the next job.  We treat this as an
     * "invalidate on error" rule. */
    int                           *cached_token_ids;
    int                            cached_n_tokens;
    int                            cached_capacity;
};

/* --- Direct SoA v2 tensor probe (moe_down SoA v2) ---------------------
 *
 * If the loaded GGUF was repacked by tools/repack_gguf_soa, it
 * contains blk.<il>.ffn_down_exps_soa_v2.weight — a Q2_K-typed tensor
 * whose bytes are already in the SoA v2 layout that
 * launch_routed_moe_q2k_sum6_resident_soa_v2_f32_prealloc consumes. In
 * that case we slice four sub-array base pointers directly from the
 * tensor's abs_offset and skip the GPU-side mirror builder entirely
 * (saves ~28 GiB of cudaMalloc + ~5 s of build kernel launches).
 *
 * If the tensor is absent (running against an old GGUF), the caller
 * falls through to the legacy DS4_MOE_DOWN_SOA_V2 mirror-builder path.
 *
 * Returns the number of layers wired up directly (0 if none of the
 * tensors are present; never negative — partial failures are logged and
 * skipped, leaving any successfully-wired layers in place).
 */
static int try_direct_resident_moe_down_soa(struct ds4cuda_inference_engine *e) {
    int direct_soa_layers = 0;
    const int n_experts = (int)e->model.n_expert;
    const int out_dim   = (int)e->model.n_embd;      /* down output = embd */
    const int in_dim    = (int)e->model.expert_ff;
    if (n_experts <= 0 || out_dim <= 0 || in_dim <= 0) return 0;
    if ((in_dim % DS4_Q2K_SUPERBLOCK) != 0) return 0;

    const struct ds4_q2k_soa_v2_layout L =
        ds4_q2k_soa_v2_layout(n_experts, out_dim, in_dim);

    /* SoA v2 GGUF is ~117 GB which exceeds the 81 GB managed allocation cap.
     * If we're on managed backend, the SoA tensors past the cap can't be sliced.
     * Log clearly and let the loop discover failures naturally (the mmap_direct
     * backend has no such cap). */
    if (ds4_model_weight_backend(&e->model) == DS4_WEIGHT_BACKEND_MANAGED) {
        /* Only emit this hint when we actually find the SoA tensor — if it's
         * not present, the user is on an old GGUF and this hint is noise.
         * The lookup is metadata-only and does not fault the weight page. */
        char probe_name[64];
        snprintf(probe_name, sizeof(probe_name),
                 "blk.0.ffn_down_exps_soa_v2.weight");
        if (ds4_model_find_tensor(&e->model, probe_name)) {
            IELOG("WARN: SoA v2 GGUF detected on managed backend — set "
                  "DS4CUDA_WEIGHT_BACKEND=mmap_direct to use the direct SoA path");
            /* Continue: per-layer probes will fail with null base ptr; the
             * loop logs each one and we fall through to the legacy builder
             * (which will also fail, but with a clear-er error chain). */
        }
    }

    for (int il = 0; il < (int)DS4_N_LAYER; il++) {
        char name[64];
        if (find_layer_tensor_name(name, sizeof(name), il,
                                   "ffn_down_exps_soa_v2.weight") != 0) {
            continue;
        }
        const struct ds4_tensor *t = ds4_model_find_tensor(&e->model, name);
        if (!t) continue;
        /* Type guard: offline tool preserves Q2_K typing. */
        if (t->quant != DS4_QUANT_Q2_K) {
            IELOG("ffn_down_exps_soa_v2 layer %d: unexpected quant %d "
                  "(want Q2_K) — skipping", il, (int)t->quant);
            continue;
        }
        if ((uint64_t)L.total_bytes != t->byte_size) {
            IELOG("ffn_down_exps_soa_v2 layer %d size mismatch: "
                  "expected %zu got %" PRIu64,
                  il, L.total_bytes, t->byte_size);
            continue;
        }
        const uint8_t *base = (const uint8_t *)ds4_tensor_device_ptr(&e->model, t);
        if (!base) {
            IELOG("ffn_down_exps_soa_v2 layer %d: no device ptr "
                  "(backend=%s)", il,
                  ds4_weight_backend_name(
                      ds4_model_weight_backend(&e->model)));
            continue;
        }

        struct ds4_layer_state *Lst = &e->session.layers[il];
        Lst->ffn_down_soa_scales = (uint8_t *)(base + L.scales_offset);
        Lst->ffn_down_soa_qs     = (uint8_t *)(base + L.qs_offset);
        Lst->ffn_down_soa_d      = (uint16_t *)(base + L.d_offset);
        Lst->ffn_down_soa_dmin   = (uint16_t *)(base + L.dmin_offset);
        direct_soa_layers++;
    }

    if (direct_soa_layers > 0 && direct_soa_layers < (int)DS4_N_LAYER) {
        /* Build a short list of layers that failed. To keep this simple, we
         * re-scan and find layers where ffn_down_soa_scales is still NULL
         * after the wiring loop. */
        char failed_list[256] = {0};
        size_t pos = 0;
        int failed_count = 0;
        for (int il = 0; il < (int)DS4_N_LAYER; il++) {
            if (e->session.layers[il].ffn_down_soa_scales) continue;
            failed_count++;
            if (failed_count <= 8 && pos + 16 < sizeof(failed_list)) {
                int n = snprintf(failed_list + pos, sizeof(failed_list) - pos,
                                 "%s%d", failed_count > 1 ? "," : "", il);
                if (n > 0) pos += (size_t)n;
            }
        }
        IELOG("resident MoE-down SoA: partial wiring (%d/%d layers); failed layers: %s%s",
              direct_soa_layers, (int)DS4_N_LAYER, failed_list,
              failed_count > 8 ? "..." : "");
    }

    if (direct_soa_layers > 0) {
        e->moe_down_soa_layers = direct_soa_layers;
        IELOG("resident MoE-down SoA active (direct from GGUF) layers=%d",
              direct_soa_layers);
    }
    return direct_soa_layers;
}

/* --- Direct SoA v2 tensor probe for IQ2_XXS gate+up --------------------
 *
 * If the loaded GGUF was repacked by tools/repack_gguf_soa in --replace
 * mode, it contains
 *   blk.<il>.ffn_gate_exps_soa_v2.weight
 *   blk.<il>.ffn_up_exps_soa_v2.weight
 * IQ2_XXS-typed tensors whose bytes match ds4_iq2_xxs_soa_v2_layout()
 * consumed by launch_routed_moe_pair_swiglu_resident_soa_v2_f32_prealloc.
 *
 * Each layer's gate AND up tensors must both be present; otherwise we
 * skip that layer (forward_layer falls through to the AoS launcher).
 *
 * Mirror of try_direct_resident_moe_down_soa, but for the gate/up pair.
 */
static int try_direct_resident_moe_gate_up_soa(struct ds4cuda_inference_engine *e) {
    int direct_soa_layers = 0;
    const int n_experts = (int)e->model.n_expert;
    const int out_dim   = (int)e->model.expert_ff;   /* FF_EXP = 2048 */
    const int in_dim    = (int)e->model.n_embd;      /* EMBD   = 4096 */
    if (n_experts <= 0 || out_dim <= 0 || in_dim <= 0) return 0;
    if ((in_dim % DS4_IQ2_XXS_SUPERBLOCK) != 0) return 0;

    const struct ds4_iq2_xxs_soa_v2_layout L =
        ds4_iq2_xxs_soa_v2_layout(n_experts, out_dim, in_dim);

    for (int il = 0; il < (int)DS4_N_LAYER; il++) {
        char name_g[64];
        char name_u[64];
        if (find_layer_tensor_name(name_g, sizeof(name_g), il,
                                   "ffn_gate_exps_soa_v2.weight") != 0) {
            continue;
        }
        if (find_layer_tensor_name(name_u, sizeof(name_u), il,
                                   "ffn_up_exps_soa_v2.weight") != 0) {
            continue;
        }
        const struct ds4_tensor *tg = ds4_model_find_tensor(&e->model, name_g);
        const struct ds4_tensor *tu = ds4_model_find_tensor(&e->model, name_u);
        if (!tg || !tu) continue;
        if (tg->quant != DS4_QUANT_IQ2_XXS) {
            IELOG("ffn_gate_exps_soa_v2 layer %d: unexpected quant %d "
                  "(want IQ2_XXS) — skipping", il, (int)tg->quant);
            continue;
        }
        if (tu->quant != DS4_QUANT_IQ2_XXS) {
            IELOG("ffn_up_exps_soa_v2 layer %d: unexpected quant %d "
                  "(want IQ2_XXS) — skipping", il, (int)tu->quant);
            continue;
        }
        if ((uint64_t)L.total_bytes != tg->byte_size) {
            IELOG("ffn_gate_exps_soa_v2 layer %d size mismatch: "
                  "expected %zu got %" PRIu64,
                  il, L.total_bytes, tg->byte_size);
            continue;
        }
        if ((uint64_t)L.total_bytes != tu->byte_size) {
            IELOG("ffn_up_exps_soa_v2 layer %d size mismatch: "
                  "expected %zu got %" PRIu64,
                  il, L.total_bytes, tu->byte_size);
            continue;
        }
        const uint8_t *base_g = (const uint8_t *)ds4_tensor_device_ptr(&e->model, tg);
        const uint8_t *base_u = (const uint8_t *)ds4_tensor_device_ptr(&e->model, tu);
        if (!base_g || !base_u) {
            IELOG("ffn_gate/up_exps_soa_v2 layer %d: no device ptr "
                  "(backend=%s)", il,
                  ds4_weight_backend_name(
                      ds4_model_weight_backend(&e->model)));
            continue;
        }

        struct ds4_layer_state *Lst = &e->session.layers[il];
        Lst->ffn_gate_soa_qs = (uint16_t *)(base_g + L.qs_offset);
        Lst->ffn_gate_soa_d  = (uint16_t *)(base_g + L.d_offset);
        Lst->ffn_up_soa_qs   = (uint16_t *)(base_u + L.qs_offset);
        Lst->ffn_up_soa_d    = (uint16_t *)(base_u + L.d_offset);
        direct_soa_layers++;
    }

    if (direct_soa_layers > 0 && direct_soa_layers < (int)DS4_N_LAYER) {
        char failed_list[256] = {0};
        size_t pos = 0;
        int failed_count = 0;
        for (int il = 0; il < (int)DS4_N_LAYER; il++) {
            if (e->session.layers[il].ffn_gate_soa_qs) continue;
            failed_count++;
            if (failed_count <= 8 && pos + 16 < sizeof(failed_list)) {
                int n = snprintf(failed_list + pos, sizeof(failed_list) - pos,
                                 "%s%d", failed_count > 1 ? "," : "", il);
                if (n > 0) pos += (size_t)n;
            }
        }
        IELOG("resident MoE-gate/up SoA: partial wiring (%d/%d layers); "
              "failed layers: %s%s",
              direct_soa_layers, (int)DS4_N_LAYER, failed_list,
              failed_count > 8 ? "..." : "");
    }

    if (direct_soa_layers > 0) {
        IELOG("resident MoE-gate/up SoA active (direct from GGUF) layers=%d",
              direct_soa_layers);
    }
    return direct_soa_layers;
}

static int build_resident_moe_down_soa(struct ds4cuda_inference_engine *e) {
    /* Prefer direct slicing from a pre-packed SoA v2 GGUF (offline
     * repack output).  When the tensor is present we wire pointers
     * straight into the managed weight blob and SKIP the legacy mirror
     * builder entirely — no cudaMalloc, no build kernel. */
    if (try_direct_resident_moe_down_soa(e) > 0) return 0;

    if (!env_flag_enabled("DS4_MOE_DOWN_SOA_V2")) return 0;

    int want_layers = env_int_or_default("DS4_MOE_DOWN_SOA_V2_LAYERS",
                                         (int)DS4_N_LAYER);
    if (want_layers > (int)DS4_N_LAYER) want_layers = (int)DS4_N_LAYER;
    int requested[(int)DS4_N_LAYER];
    int n_requested = parse_layer_list(
        getenv("DS4_MOE_DOWN_SOA_V2_LAYERS_LIST"),
        requested, (int)DS4_N_LAYER);
    if (n_requested == 0) {
        if (want_layers <= 0) return 0;
        for (int il = 0; il < want_layers; il++) requested[n_requested++] = il;
    }

    const uint64_t reserve_gib =
        (uint64_t)env_int_or_default("DS4_MOE_DOWN_SOA_V2_RESERVE_GIB", 20);
    const uint64_t reserve_bytes = reserve_gib * 1024ull * 1024ull * 1024ull;
    const uint64_t per_layer = resident_moe_down_soa_layer_bytes();
    int built = 0;

    IELOG("DS4_MOE_DOWN_SOA_V2 enabled: per-layer resident SoA %.2f MiB, "
          "requested_layers=%d reserve=%llu GiB",
          (double)per_layer / (1024.0 * 1024.0), n_requested,
          (unsigned long long)reserve_gib);

    for (int i = 0; i < n_requested; i++) {
        const int il = requested[i];
        const uint64_t avail = mem_available_bytes();
        if (avail != UINT64_MAX && avail < reserve_bytes + per_layer) {
            IELOG("resident MoE-down SoA stopping before layer %d: "
                  "MemAvailable %.2f GiB < reserve+layer %.2f GiB",
                  il, (double)avail / (1024.0 * 1024.0 * 1024.0),
                  (double)(reserve_bytes + per_layer) /
                      (1024.0 * 1024.0 * 1024.0));
            break;
        }

        char name[64];
        if (find_layer_tensor_name(name, sizeof(name), il,
                                   "ffn_down_exps.weight") != 0) {
            return -EINVAL;
        }
        const struct ds4_tensor *t = ds4_model_find_tensor(&e->model, name);
        if (!t || t->quant != DS4_QUANT_Q2_K) {
            IELOG("resident MoE-down SoA: missing/non-Q2_K tensor %s", name);
            return -EINVAL;
        }
        const ds4cuda::block_q2_K *src =
            (const ds4cuda::block_q2_K *)ds4_tensor_device_ptr(&e->model, t);
        if (!src) {
            IELOG("resident MoE-down SoA: no device pointer for %s "
                  "(backend=%s)", name,
                  ds4_weight_backend_name(
                      ds4_model_weight_backend(&e->model)));
            return -EINVAL;
        }

        struct ds4_layer_state *L = &e->session.layers[il];
        cudaError_t ce = cudaSuccess;
        ce = cudaMalloc((void **)&L->ffn_down_soa_scales,
                        256ull * 4096ull * 16ull * 8ull);
        if (ce == cudaSuccess) {
            ce = cudaMalloc((void **)&L->ffn_down_soa_qs,
                            256ull * 4096ull * 64ull * 8ull);
        }
        if (ce == cudaSuccess) {
            ce = cudaMalloc((void **)&L->ffn_down_soa_d,
                            256ull * 4096ull * 8ull * sizeof(uint16_t));
        }
        if (ce == cudaSuccess) {
            ce = cudaMalloc((void **)&L->ffn_down_soa_dmin,
                            256ull * 4096ull * 8ull * sizeof(uint16_t));
        }
        if (ce != cudaSuccess) {
            IELOG("resident MoE-down SoA cudaMalloc layer %d failed: %s",
                  il, cudaGetErrorString(ce));
            ds4_session_state_free_resident_moe_down_soa(&e->session);
            return -ENOMEM;
        }

        ds4cuda::launch_build_moe_q2k_sum6_resident_soa(
            src, L->ffn_down_soa_scales, L->ffn_down_soa_qs,
            L->ffn_down_soa_d, L->ffn_down_soa_dmin,
            256, 4096, 2048, 0);
        ce = cudaGetLastError();
        if (ce == cudaSuccess) ce = cudaDeviceSynchronize();
        if (ce != cudaSuccess) {
            IELOG("resident MoE-down SoA build layer %d failed: %s",
                  il, cudaGetErrorString(ce));
            ds4_session_state_free_resident_moe_down_soa(&e->session);
            return -EIO;
        }
        built++;
        IELOG("resident MoE-down SoA built layer %d (%d/%d)",
              il, built, n_requested);
    }

    e->moe_down_soa_layers = built;
    IELOG("resident MoE-down SoA active layers=%d", built);
    return 0;
}


/* ------------------------------------------------------------------ */
/* Prefix-cache helpers.                                                */
/* ------------------------------------------------------------------ */

/* Ensure cached_token_ids has space for at least `need` ids.  Returns 0
 * on success, -ENOMEM on alloc failure.  Capacity grows by doubling. */
static int cache_reserve(struct ds4cuda_inference_engine *e, int need) {
    if (need <= e->cached_capacity) return 0;
    int cap = e->cached_capacity > 0 ? e->cached_capacity : 64;
    while (cap < need) cap *= 2;
    int *p = (int *)realloc(e->cached_token_ids, (size_t)cap * sizeof(int));
    if (!p) return -ENOMEM;
    e->cached_token_ids = p;
    e->cached_capacity = cap;
    return 0;
}

/* Longest common prefix length between cache and `ids[0..n_ids)`. */
static int cache_lcp(const struct ds4cuda_inference_engine *e,
                     const int *ids, int n_ids) {
    int n = e->cached_n_tokens < n_ids ? e->cached_n_tokens : n_ids;
    for (int i = 0; i < n; ++i) {
        if (e->cached_token_ids[i] != ids[i]) return i;
    }
    return n;
}

/* Invalidate the prefix cache (force a full reset on the next job). */
static void cache_invalidate(struct ds4cuda_inference_engine *e) {
    e->cached_n_tokens = 0;
}

/* Replace the cache contents with `ids[0..n_ids)`. */
static int cache_replace(struct ds4cuda_inference_engine *e,
                         const int *ids, int n_ids) {
    if (n_ids <= 0) {
        e->cached_n_tokens = 0;
        return 0;
    }
    int rc = cache_reserve(e, n_ids);
    if (rc != 0) return rc;
    memcpy(e->cached_token_ids, ids, (size_t)n_ids * sizeof(int));
    e->cached_n_tokens = n_ids;
    return 0;
}

/* Append `ids[0..n_ids)` to the cache. */
static int cache_append(struct ds4cuda_inference_engine *e,
                        const int *ids, int n_ids) {
    if (n_ids <= 0) return 0;
    int need = e->cached_n_tokens + n_ids;
    int rc = cache_reserve(e, need);
    if (rc != 0) return rc;
    memcpy(e->cached_token_ids + e->cached_n_tokens, ids,
           (size_t)n_ids * sizeof(int));
    e->cached_n_tokens = need;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Token-id sidecar file (`<path>.tokens`).                             */
/*   Format (little-endian):                                            */
/*     int32 count                                                      */
/*     int32 ids[count]                                                 */
/* This is OS-endian-naive; on the DGX Spark target everything is LE,   */
/* and the broader kv_persist file format makes the same assumption.    */
/* ------------------------------------------------------------------ */
#define IE_TOKENS_SUFFIX ".tokens"

static char *make_tokens_sidecar_path(const char *base) {
    if (!base) return NULL;
    size_t n = strlen(base);
    char *p = (char *)malloc(n + sizeof(IE_TOKENS_SUFFIX));
    if (!p) return NULL;
    memcpy(p, base, n);
    memcpy(p + n, IE_TOKENS_SUFFIX, sizeof(IE_TOKENS_SUFFIX));
    return p;
}

static int write_tokens_sidecar(const char *path, const int *ids, int n_ids) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        IELOG("write_tokens_sidecar: fopen('%s') failed: %s",
              path, strerror(errno));
        return -EIO;
    }
    int32_t cnt = (int32_t)(n_ids > 0 ? n_ids : 0);
    if (fwrite(&cnt, sizeof(cnt), 1, fp) != 1) {
        IELOG("write_tokens_sidecar: short write count: %s", strerror(errno));
        fclose(fp);
        return -EIO;
    }
    if (cnt > 0) {
        /* Token ids in the engine cache are stored as host int (4 B on the
         * supported targets), matching int32 byte-for-byte. */
        if (fwrite(ids, sizeof(int32_t), (size_t)cnt, fp) != (size_t)cnt) {
            IELOG("write_tokens_sidecar: short write ids: %s", strerror(errno));
            fclose(fp);
            return -EIO;
        }
    }
    if (fclose(fp) != 0) {
        IELOG("write_tokens_sidecar: fclose failed: %s", strerror(errno));
        return -EIO;
    }
    return 0;
}

static int read_tokens_sidecar(const char *path,
                               int **out_ids, int *out_n) {
    *out_ids = NULL;
    *out_n = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        IELOG("read_tokens_sidecar: fopen('%s') failed: %s",
              path, strerror(errno));
        return -EIO;
    }
    int32_t cnt = 0;
    if (fread(&cnt, sizeof(cnt), 1, fp) != 1) {
        IELOG("read_tokens_sidecar: short read count");
        fclose(fp);
        return -EIO;
    }
    if (cnt < 0 || cnt > (1 << 24)) {  /* sanity cap 16 M tokens */
        IELOG("read_tokens_sidecar: implausible count=%d", (int)cnt);
        fclose(fp);
        return -EIO;
    }
    if (cnt == 0) {
        *out_ids = NULL;
        *out_n = 0;
        fclose(fp);
        return 0;
    }
    int *ids = (int *)malloc((size_t)cnt * sizeof(int));
    if (!ids) {
        fclose(fp);
        return -ENOMEM;
    }
    if (fread(ids, sizeof(int32_t), (size_t)cnt, fp) != (size_t)cnt) {
        IELOG("read_tokens_sidecar: short read ids");
        free(ids);
        fclose(fp);
        return -EIO;
    }
    fclose(fp);
    *out_ids = ids;
    *out_n = (int)cnt;
    return 0;
}

/* Push job onto the FIFO. Returns 0 on success, -EBUSY if full. */
static int engine_enqueue(struct ds4cuda_inference_engine *e, struct ie_job *j) {
    pthread_mutex_lock(&e->qlock);
    if (e->stop) {
        pthread_mutex_unlock(&e->qlock);
        return -EINVAL;
    }
    if (e->q_count >= e->q_capacity) {
        pthread_mutex_unlock(&e->qlock);
        return -EBUSY;
    }
    j->next = NULL;
    if (e->q_tail) {
        e->q_tail->next = j;
        e->q_tail = j;
    } else {
        e->q_head = e->q_tail = j;
    }
    e->q_count++;
    pthread_cond_signal(&e->qcv_nonempty);
    pthread_mutex_unlock(&e->qlock);
    return 0;
}

/* Pop next job, blocking until either a job arrives or stop is set.
 * On stop returns NULL. */
static struct ie_job *engine_dequeue(struct ds4cuda_inference_engine *e) {
    pthread_mutex_lock(&e->qlock);
    while (!e->stop && e->q_head == NULL) {
        pthread_cond_wait(&e->qcv_nonempty, &e->qlock);
    }
    struct ie_job *j = NULL;
    if (e->q_head) {
        j = e->q_head;
        e->q_head = j->next;
        if (!e->q_head) e->q_tail = NULL;
        e->q_count--;
        pthread_cond_signal(&e->qcv_nonfull);
    }
    pthread_mutex_unlock(&e->qlock);
    return j;
}

static void job_signal_done(struct ie_job *j) {
    pthread_mutex_lock(&j->lock);
    j->done = 1;
    pthread_cond_broadcast(&j->done_cv);
    pthread_mutex_unlock(&j->lock);
}

static void job_wait(struct ie_job *j) {
    pthread_mutex_lock(&j->lock);
    while (!j->done) pthread_cond_wait(&j->done_cv, &j->lock);
    pthread_mutex_unlock(&j->lock);
}

/* ------------------------------------------------------------------ */
/* dsbuf — small dynamic byte buffer used to accumulate decoded output. */
/* ------------------------------------------------------------------ */
struct dsbuf { char *p; size_t len; size_t cap; };

static int dsbuf_reserve(struct dsbuf *b, size_t extra) {
    size_t need = b->len + extra + 1;
    if (need <= b->cap) return 0;
    size_t cap = b->cap ? b->cap * 2 : 64;
    while (cap < need) cap *= 2;
    char *p = (char *)realloc(b->p, cap);
    if (!p) return -ENOMEM;
    b->p = p; b->cap = cap; return 0;
}

static int dsbuf_append(struct dsbuf *b, const char *s, size_t n) {
    if (dsbuf_reserve(b, n) != 0) return -ENOMEM;
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
    return 0;
}

static char *dsbuf_take(struct dsbuf *b) {
    if (b->p) {
        b->p[b->len] = '\0';
        char *out = b->p;
        b->p = NULL; b->cap = 0; b->len = 0;
        return out;
    }
    char *out = (char *)malloc(1);
    if (!out) return NULL;
    out[0] = '\0';
    return out;
}

/* ------------------------------------------------------------------ */
/* Argmax over the last-token logits in device memory.                  */
/* d_logits is a device pointer (carved from the activation arena).     */
/* The reduction stays on GPU; only the selected token id is copied back */
/* because decode still needs the id on CPU for tokenizer emission.      */
/* ------------------------------------------------------------------ */
static int argmax_device_f32(const float *d_logits, int n_vocab,
                             int *d_argmax, cudaStream_t stream) {
    ds4cuda::launch_argmax_f32_to_i32(d_logits, d_argmax, n_vocab, stream);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        IELOG("argmax: kernel launch failed: %s", cudaGetErrorString(e));
        return -1;
    }
    int idx = -1;
    e = cudaMemcpy(&idx, d_argmax, sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        IELOG("argmax: token-id DtoH failed: %s", cudaGetErrorString(e));
        return -1;
    }
    return idx;
}

/* ------------------------------------------------------------------ */
/* Decode one token id to UTF-8 bytes via tokenizer.                    */
/* Returns malloc()'d NUL-terminated string with byte length in *out_n  */
/* (excluding NUL).  NULL on error.  Per-token decode reproduces ds4's  */
/* bpe_emit_piece byte sequence.                                        */
/* ------------------------------------------------------------------ */
static char *decode_one_token(const struct ds4cuda_tokenizer *tk, int id,
                              int *out_n) {
    int ids[1] = { id };
    /* Single token decodes to at most ~32 bytes in practice (longest
     * BPE token in V4 vocab is well under that); 64 is safe. */
    char buf[64];
    int n = ds4cuda_detokenize(tk, ids, 1, buf, sizeof(buf));
    if (n < 0) return NULL;
    char *out = (char *)malloc((size_t)n + 1);
    if (!out) return NULL;
    memcpy(out, buf, (size_t)n);
    out[n] = '\0';
    if (out_n) *out_n = n;
    return out;
}

/* ------------------------------------------------------------------ */
/* Run one job: tokenize -> prefill -> decode loop.                     */
/* On success leaves j->result populated (BUFFERED) or has emitted all  */
/* tokens (STREAM) and j->rc=0; on error sets j->rc<0.                  */
/* ------------------------------------------------------------------ */
static void engine_run_job(struct ds4cuda_inference_engine *e, struct ie_job *j) {
    j->rc = 0;
    strcpy(j->finish_reason, "stop");

    /* ---- 1) Tokenize the DSML prompt -------------------------------- */
    /* Cap prompt tokens at max_context - max_new_tokens to leave room for
     * the decode loop's KV cache.  This is a soft guardrail; if it trips
     * we just truncate decode (the prompt tokens still go in). */
    const int n_vocab = ds4cuda_tokenizer_n_vocab(e->tk);
    if (n_vocab <= 0) {
        IELOG("vocab unknown; aborting job");
        j->rc = -EINVAL;
        return;
    }

    /* Probe length first. ds4cuda_tokenize with (NULL, 0) always returns
     * -E2BIG for any non-empty input, so the probe just tells us "fits / does
     * not fit in this cap". Start at 4096 (covers ~95% of single-turn chat
     * prompts) and double until we land under max_context — beyond that the
     * job won't fit anyway and we surface -E2BIG to the caller. This makes
     * long-context prompts (e.g. ~13K tokens for an 80K-char Chinese block)
     * work without a hardcoded cap. */
    int prompt_cap = 4096;
    int *prompt_ids = NULL;
    int n_prompt = -E2BIG;
    int cap_ceiling = e->max_context > prompt_cap ? e->max_context : prompt_cap;
    /* +1024 slack lets us still report a useful E2BIG when the prompt
     * just barely exceeds max_context (engine truncates downstream). */
    cap_ceiling += 1024;
    while (1) {
        free(prompt_ids);
        prompt_ids = (int *)malloc((size_t)prompt_cap * sizeof(int));
        if (!prompt_ids) { j->rc = -ENOMEM; return; }
        n_prompt = ds4cuda_tokenize(e->tk, j->prompt, prompt_ids, prompt_cap);
        if (n_prompt >= 0) break;
        if (n_prompt != -E2BIG) {
            IELOG("tokenize failed rc=%d (prompt_cap=%d)", n_prompt, prompt_cap);
            free(prompt_ids);
            j->rc = n_prompt;
            return;
        }
        if (prompt_cap >= cap_ceiling) {
            IELOG("tokenize: prompt larger than max_context+slack=%d — aborting",
                  cap_ceiling);
            free(prompt_ids);
            j->rc = -E2BIG;
            return;
        }
        prompt_cap *= 2;
        if (prompt_cap > cap_ceiling) prompt_cap = cap_ceiling;
    }

    int max_new = j->max_new_tokens > 0 ? j->max_new_tokens
                                        : e->default_max_new_tokens;
    int total_budget = e->max_context;  /* prefill + decode tokens */
    if (n_prompt + max_new > total_budget) {
        /* Trim max_new — never trim prefill (the user's request). */
        if (n_prompt >= total_budget) {
            IELOG("prompt %d tokens exceeds max_context %d; aborting",
                  n_prompt, total_budget);
            free(prompt_ids);
            j->rc = -E2BIG;
            return;
        }
        max_new = total_budget - n_prompt;
    }
    if (max_new <= 0) {
        IELOG("max_new <= 0 after capping (n_prompt=%d, max_context=%d)",
              n_prompt, e->max_context);
        free(prompt_ids);
        j->rc = -EINVAL;
        return;
    }

    /* ---- 2) Decide reset vs. prefix-sync fast path ------------------ */
    /*
     * Compare the new prompt's token-id sequence against the cached
     * sequence from the previous successful job (= prompt + generated of
     * that job).  Let `P` be the longest common prefix length.
     *
     *   P == cached_n_tokens AND P > 0 AND P < n_prompt:
     *     Fast path.  The cache is a true prefix of the new prompt, and
     *     there are still prompt tokens to feed.  Skip ds4_session_state
     *     _reset; the per-layer KV cursors and `pos` are already at P.
     *     Start prefill at t = P with pos = P.
     *
     *   Otherwise (including P == 0, divergence before cache end, or
     *   cache fully matches but no new prompt tokens):
     *     Full reset + prefill from scratch.  This is the correct-but-
     *     slow fallback.  We do not yet implement KV truncation for the
     *     P < cached_n_tokens case (left as a prefix-sync follow-up).
     */
    int P = cache_lcp(e, prompt_ids, n_prompt);
    int prefill_start = 0;       /* default: full reset path */
    int do_reset = 1;
    if (P > 0 && P == e->cached_n_tokens && P < n_prompt) {
        /* Fast path: extend the cached prefix. */
        prefill_start = P;
        do_reset = 0;
        IELOG("prefix-sync: P=%d (n_prompt=%d, cached=%d) — "
              "skipping %d prefill steps",
              P, n_prompt, e->cached_n_tokens, P);
    } else if (P > 0) {
        IELOG("prefix-sync: P=%d but P!=cached_n_tokens=%d or no new "
              "tokens (n_prompt=%d) — falling back to full reset",
              P, e->cached_n_tokens, n_prompt);
    }
    if (do_reset) {
        ds4_session_state_reset(&e->session);
        e->cached_n_tokens = 0;
    }

    /* Output accumulator (BUFFERED mode). */
    struct dsbuf acc = {0};

    /* Generated-token id collection (so we can append to the prefix
     * cache after generation finishes).  Sized to `max_new`. */
    int *gen_ids = (int *)malloc((size_t)(max_new > 0 ? max_new : 1)
                                  * sizeof(int));
    if (!gen_ids) {
        free(prompt_ids); free(acc.p);
        cache_invalidate(e);
        j->rc = -ENOMEM;
        return;
    }
    int gen_n = 0;

    /* ---- 3) Prefill: feed prompt tokens [prefill_start..n_prompt) --- */
    /* On the fast path, the previous job left logits at position P-1
     * (the last cached token).  Those are not retained — the activation
     * arena is rewound on every ds4_forward_token call.  So we ALWAYS
     * have to issue at least one forward to get logits.  When the new
     * prompt extends the cache (n_prompt > P), the next forward at t=P
     * pos=P naturally produces logits for the t+1 token; the LAST prompt
     * forward (t = n_prompt-1) gives us the first generated token.  No
     * special-case needed. */
    cudaStream_t stream = 0;
    int next_tok = -1;
    for (int t = prefill_start; t < n_prompt; ++t) {
        const float *d_logits = NULL;
        int rc = ds4cuda::ds4_forward_token(&e->model, &e->session,
                                            prompt_ids[t], /*pos=*/t,
                                            &d_logits, stream);
        if (rc != 0) {
            IELOG("forward_token (prefill t=%d tok=%d) failed rc=%d",
                  t, prompt_ids[t], rc);
            free(prompt_ids); free(acc.p); free(gen_ids);
            cache_invalidate(e);
            j->rc = -EIO;
            return;
        }
        if (t == n_prompt - 1) {
            /* Last prompt token's logits predict the FIRST generated
             * token. */
            int am = argmax_device_f32(d_logits, n_vocab, e->d_argmax, stream);
            if (am < 0) {
                free(prompt_ids); free(acc.p); free(gen_ids);
                cache_invalidate(e);
                j->rc = -EIO;
                return;
            }
            next_tok = am;
        }
    }

    /* The session_state KV cache now contains exactly `n_prompt` tokens
     * in positions 0..n_prompt-1.  Update the prefix cache to reflect
     * this so even if generation produces zero tokens (empty prompt
     * edge case below) the next job sees the correct compare baseline. */
    if (cache_replace(e, prompt_ids, n_prompt) != 0) {
        IELOG("cache_replace OOM (n_prompt=%d) — invalidating", n_prompt);
        cache_invalidate(e);
    }

    free(prompt_ids);
    prompt_ids = NULL;

    if (next_tok < 0) {
        /* No prompt tokens? Nothing to generate. */
        IELOG("empty prompt — nothing to generate");
        free(gen_ids);
        if (j->mode == JOB_BUFFERED) j->result = dsbuf_take(&acc);
        return;
    }

    /* ---- 4) Decode loop -------------------------------------------- */
    /* Stop conditions:
     *   - generated == max_new                               -> "length"
     *   - next_tok == EOS (or BOS treated as EOS-equivalent) -> "stop"
     *
     * Tracking: `kv_committed_gens` counts how many of the entries in
     * gen_ids[] have had their KV row written by a forward_token call
     * (and therefore form part of the session's KV state).  Used by
     * the post-loop prefix cache update so we cache exactly the tokens
     * whose KV is actually present.
     */
    int n_generated = 0;
    int kv_committed_gens = 0;
    while (n_generated < max_new) {
        /* Check stop tokens BEFORE emitting / forwarding — the EOS id
         * is the network's signal that generation should end here.  We
         * neither emit nor advance KV for it. */
        if (next_tok == e->eos_id) {
            strcpy(j->finish_reason, "stop");
            break;
        }

        /* Decode and emit/append. */
        int piece_n = 0;
        char *piece = decode_one_token(e->tk, next_tok, &piece_n);
        if (!piece) {
            IELOG("decode_one_token(%d) failed", next_tok);
            free(acc.p); free(gen_ids);
            cache_invalidate(e);
            j->rc = -EIO;
            return;
        }
        if (j->mode == JOB_STREAM) {
            int erc = j->emit(piece, piece_n, j->emit_user_data);
            free(piece);
            if (erc != 0) {
                IELOG("emit returned %d, aborting stream", erc);
                free(acc.p);
                free(gen_ids);
                cache_invalidate(e);
                j->rc = erc;
                return;
            }
        } else {
            int arc = dsbuf_append(&acc, piece, (size_t)piece_n);
            free(piece);
            if (arc != 0) {
                free(acc.p); free(gen_ids);
                cache_invalidate(e);
                j->rc = arc;
                return;
            }
        }
        /* Record the token id for the prefix cache update at job end. */
        gen_ids[gen_n++] = next_tok;
        n_generated++;

        /* If we've reached max_new, set finish_reason and break BEFORE
         * doing one more forward (we already emitted the last token). */
        if (n_generated >= max_new) {
            strcpy(j->finish_reason, "length");
            break;
        }

        /* Advance: consume next_tok at pos = n_prompt + n_generated - 1
         * to update KV, then read logits to pick the following token. */
        const int pos = n_prompt + n_generated - 1;
        const float *d_logits = NULL;
        int rc = ds4cuda::ds4_forward_token(&e->model, &e->session,
                                            next_tok, pos, &d_logits, stream);
        if (rc != 0) {
            IELOG("forward_token (decode tok=%d pos=%d) failed rc=%d",
                  next_tok, pos, rc);
            free(acc.p); free(gen_ids);
            cache_invalidate(e);
            j->rc = -EIO;
            return;
        }
        int am = argmax_device_f32(d_logits, n_vocab, e->d_argmax, stream);
        if (am < 0) {
            free(acc.p); free(gen_ids);
            cache_invalidate(e);
            j->rc = -EIO;
            return;
        }
        /* The forward at pos = n_prompt + (n_generated-1) just consumed
         * gen_ids[n_generated-1], so its KV row is now committed. */
        kv_committed_gens = n_generated;
        next_tok = am;
    }

    /* Post-loop prefix-cache update: append only the generated tokens
     * whose KV rows are actually committed.  This is `kv_committed_gens`,
     * which evolves like so:
     *
     *   - EOS branch:         loop breaks BEFORE the next emit, so the
     *                         last successful forward at the bottom of
     *                         the previous iteration set
     *                         kv_committed_gens = n_generated.  All
     *                         emitted tokens are KV-backed.
     *   - max_new branch:     loop breaks AFTER the new emit but BEFORE
     *                         the next forward.  kv_committed_gens
     *                         remains at the previous iteration's
     *                         n_generated (= the just-emitted token's
     *                         index, n_generated - 1).  The last emitted
     *                         token has no KV row.
     *
     * In both cases gen_n == n_generated and the right number of ids
     * to cache is exactly kv_committed_gens. */
    if (kv_committed_gens > 0) {
        if (cache_append(e, gen_ids, kv_committed_gens) != 0) {
            IELOG("cache_append OOM (gens=%d) — invalidating",
                  kv_committed_gens);
            cache_invalidate(e);
        }
    }
    free(gen_ids);

    if (j->mode == JOB_BUFFERED) {
        j->result = dsbuf_take(&acc);
        if (!j->result) j->rc = -ENOMEM;
    } else {
        free(acc.p);
    }
}

/* ------------------------------------------------------------------ */
/* Admin job handlers (run on the worker thread).                       */
/* ------------------------------------------------------------------ */

static void engine_run_save_job(struct ds4cuda_inference_engine *e,
                                struct ie_job *j) {
    if (!j->path) { j->rc = -EINVAL; return; }
    /* Use the cached prompt token sequence as the prompt_token_ids field
     * in the kv_persist header (capped at 64 there).  We persist the
     * full sequence in the .tokens sidecar regardless. */
    int rc = ds4_session_save_to_disk(&e->session,
                                      e->cached_token_ids,
                                      e->cached_n_tokens,
                                      j->path);
    if (rc != 0) {
        IELOG("save_to_disk('%s') failed rc=%d", j->path, rc);
        j->rc = rc;
        return;
    }
    char *side = make_tokens_sidecar_path(j->path);
    if (!side) { j->rc = -ENOMEM; return; }
    rc = write_tokens_sidecar(side, e->cached_token_ids, e->cached_n_tokens);
    free(side);
    if (rc != 0) {
        IELOG("write_tokens_sidecar failed rc=%d", rc);
        j->rc = rc;
        return;
    }
    j->rc = 0;
}

static void engine_run_load_job(struct ds4cuda_inference_engine *e,
                                struct ie_job *j) {
    if (!j->path) { j->rc = -EINVAL; return; }
    int load_pos = 0, load_n_tok = 0;
    int rc = ds4_session_load_from_disk(&e->session, j->path,
                                        &load_pos, &load_n_tok);
    if (rc != 0) {
        IELOG("load_from_disk('%s') failed rc=%d", j->path, rc);
        cache_invalidate(e);
        j->rc = rc;
        return;
    }
    char *side = make_tokens_sidecar_path(j->path);
    if (!side) { cache_invalidate(e); j->rc = -ENOMEM; return; }
    int *ids = NULL; int n_ids = 0;
    rc = read_tokens_sidecar(side, &ids, &n_ids);
    free(side);
    if (rc != 0) {
        IELOG("read_tokens_sidecar failed rc=%d (KV loaded but cache empty)",
              rc);
        cache_invalidate(e);
        j->rc = rc;
        return;
    }
    int crc = cache_replace(e, ids, n_ids);
    free(ids);
    if (crc != 0) {
        IELOG("cache_replace OOM");
        cache_invalidate(e);
        j->rc = crc;
        return;
    }
    j->rc = 0;
}

/* ------------------------------------------------------------------ */
/* Worker thread.                                                       */
/* ------------------------------------------------------------------ */
static void *engine_worker_main(void *arg) {
    struct ds4cuda_inference_engine *e = (struct ds4cuda_inference_engine *)arg;
    pthread_mutex_lock(&e->qlock);
    e->worker_ready = 1;
    pthread_cond_broadcast(&e->qcv_nonfull);
    pthread_mutex_unlock(&e->qlock);

    for (;;) {
        struct ie_job *j = engine_dequeue(e);
        if (!j) break;     /* stop signaled with empty queue */
        switch (j->mode) {
        case JOB_SAVE:
            engine_run_save_job(e, j);
            break;
        case JOB_LOAD:
            engine_run_load_job(e, j);
            break;
        case JOB_BUFFERED:
        case JOB_STREAM:
        default:
            engine_run_job(e, j);
            break;
        }
        job_signal_done(j);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Public API: create / destroy.                                        */
/* ------------------------------------------------------------------ */
extern "C" int ds4cuda_inference_engine_create(
        struct ds4cuda_inference_engine **out,
        const struct ds4cuda_inference_engine_options *opts)
{
    if (!out) return -EINVAL;
    *out = NULL;
    if (!opts || !opts->gguf_path) return -EINVAL;

    struct ds4cuda_inference_engine *e =
        (struct ds4cuda_inference_engine *)calloc(1, sizeof(*e));
    if (!e) return -ENOMEM;

    e->max_context = opts->max_context > 0 ? opts->max_context : 256;
    e->default_max_new_tokens =
        opts->default_max_new_tokens > 0 ? opts->default_max_new_tokens : 64;
    e->q_capacity = opts->queue_depth > 0 ? opts->queue_depth : 4;

    pthread_mutex_init(&e->qlock, NULL);
    pthread_cond_init(&e->qcv_nonempty, NULL);
    pthread_cond_init(&e->qcv_nonfull, NULL);

    /* ---- 1) ds4_model_open ---------------------------------------- */
    if (ds4_model_open(&e->model, opts->gguf_path) != 0) {
        IELOG("ds4_model_open failed: %s", opts->gguf_path);
        pthread_mutex_destroy(&e->qlock);
        pthread_cond_destroy(&e->qcv_nonempty);
        pthread_cond_destroy(&e->qcv_nonfull);
        free(e);
        return -EIO;
    }

    /* ---- 2) Choose CUDA-visible weight backend ---------------------- */
    enum ds4_weight_backend weight_backend = env_weight_backend();
    int rc = 0;
    rc = validate_weight_backend(weight_backend);
    if (rc != 0) {
        ds4_model_close(&e->model);
        pthread_mutex_destroy(&e->qlock);
        pthread_cond_destroy(&e->qcv_nonempty);
        pthread_cond_destroy(&e->qcv_nonfull);
        free(e);
        return rc;
    }
    ds4_model_set_weight_backend(&e->model, weight_backend);
    IELOG("weight backend=%s",
          ds4_weight_backend_name(ds4_model_weight_backend(&e->model)));
    if (weight_backend == DS4_WEIGHT_BACKEND_MMAP_DIRECT) {
        IELOG("DS4CUDA_WEIGHT_BACKEND=mmap_direct: skipping "
              "ds4_model_load_to_managed; kernels will read GGUF mmap pages");
    } else {
        rc = ds4_model_load_to_managed(&e->model,
                                       4ULL * 1024ULL * 1024ULL * 1024ULL,
                                       opts->verbose_load);
        if (rc != 0) {
            IELOG("ds4_model_load_to_managed failed rc=%d", rc);
            ds4_model_close(&e->model);
            pthread_mutex_destroy(&e->qlock);
            pthread_cond_destroy(&e->qcv_nonempty);
            pthread_cond_destroy(&e->qcv_nonfull);
            free(e);
            return rc;
        }
    }

    /* ---- 3) Session state ----------------------------------------- */
    if (ds4_session_state_alloc(&e->session, e->max_context) != 0) {
        IELOG("ds4_session_state_alloc(%d) failed", e->max_context);
        ds4_model_managed_free(&e->model);
        ds4_model_close(&e->model);
        pthread_mutex_destroy(&e->qlock);
        pthread_cond_destroy(&e->qcv_nonempty);
        pthread_cond_destroy(&e->qcv_nonfull);
        free(e);
        return -ENOMEM;
    }
    ds4_session_state_reset(&e->session);

    rc = build_resident_moe_down_soa(e);
    if (rc != 0) {
        IELOG("build_resident_moe_down_soa failed rc=%d", rc);
        ds4_session_state_free(&e->session);
        ds4_model_managed_free(&e->model);
        ds4_model_close(&e->model);
        pthread_mutex_destroy(&e->qlock);
        pthread_cond_destroy(&e->qcv_nonempty);
        pthread_cond_destroy(&e->qcv_nonfull);
        free(e);
        return rc;
    }

    /* Probe IQ2_XXS gate/up SoA v2 tensors. Same
     * substitution pattern as down: if present, forward_layer reads
     * L->ffn_{gate,up}_soa_* and uses the SoA launcher; otherwise the
     * AoS launcher is used unchanged. No allocations; no failure path. */
    (void)try_direct_resident_moe_gate_up_soa(e);

    /* ---- 4) Tokenizer --------------------------------------------- */
    if (ds4cuda_tokenizer_init(&e->tk, &e->model) != 0) {
        IELOG("ds4cuda_tokenizer_init failed");
        ds4_session_state_free(&e->session);
        ds4_model_managed_free(&e->model);
        ds4_model_close(&e->model);
        pthread_mutex_destroy(&e->qlock);
        pthread_cond_destroy(&e->qcv_nonempty);
        pthread_cond_destroy(&e->qcv_nonfull);
        free(e);
        return -EIO;
    }
    e->eos_id = ds4cuda_tokenizer_eos_id(e->tk);
    e->bos_id = ds4cuda_tokenizer_bos_id(e->tk);

    /* ---- 5) Optional restore from disk ---------------------------- */
    /* If the caller provided restore_path, populate the session_state
     * and the prefix-token cache from disk BEFORE spawning the worker.
     * Doing this on the main thread (with no jobs in flight yet) avoids
     * any worker/restore synchronization concerns. */
    if (opts->restore_path && opts->restore_path[0] != '\0') {
        int load_pos = 0, load_n_tok = 0;
        int lrc = ds4_session_load_from_disk(&e->session, opts->restore_path,
                                             &load_pos, &load_n_tok);
        if (lrc != 0) {
            IELOG("restore_path: ds4_session_load_from_disk('%s') failed rc=%d",
                  opts->restore_path, lrc);
            ds4cuda_tokenizer_free(e->tk);
            ds4_session_state_free(&e->session);
            ds4_model_managed_free(&e->model);
            ds4_model_close(&e->model);
            pthread_mutex_destroy(&e->qlock);
            pthread_cond_destroy(&e->qcv_nonempty);
            pthread_cond_destroy(&e->qcv_nonfull);
            free(e);
            return lrc;
        }
        char *tokens_path = make_tokens_sidecar_path(opts->restore_path);
        if (!tokens_path) {
            IELOG("restore_path: OOM building sidecar path");
            ds4cuda_tokenizer_free(e->tk);
            ds4_session_state_free(&e->session);
            ds4_model_managed_free(&e->model);
            ds4_model_close(&e->model);
            pthread_mutex_destroy(&e->qlock);
            pthread_cond_destroy(&e->qcv_nonempty);
            pthread_cond_destroy(&e->qcv_nonfull);
            free(e);
            return -ENOMEM;
        }
        int *ids = NULL; int n_ids = 0;
        int trc = read_tokens_sidecar(tokens_path, &ids, &n_ids);
        free(tokens_path);
        if (trc != 0) {
            IELOG("restore_path: read_tokens_sidecar failed rc=%d "
                  "(KV restored but token cache empty — next job will "
                  "fall back to full reset)", trc);
            /* Soft-fail: KV is loaded but the prefix-cache is empty.
             * Caller still benefits from the saved state once a job
             * walks the same prompt again, but only via full prefill. */
            cache_invalidate(e);
        } else {
            int crc = cache_replace(e, ids, n_ids);
            free(ids);
            if (crc != 0) {
                IELOG("restore_path: cache_replace OOM — invalidating");
                cache_invalidate(e);
            } else {
                IELOG("restore_path: loaded KV (pos=%d) + %d cached tokens "
                      "from '%s'", load_pos, n_ids, opts->restore_path);
            }
        }
    }

    /* ---- 6) Tiny CUDA scratch owned for the engine lifetime -------- */
    cudaError_t ce = cudaMalloc((void **)&e->d_argmax, sizeof(int));
    if (ce != cudaSuccess) {
        IELOG("cudaMalloc(d_argmax) failed: %s", cudaGetErrorString(ce));
        free(e->cached_token_ids);
        ds4cuda_tokenizer_free(e->tk);
        ds4_session_state_free(&e->session);
        ds4_model_managed_free(&e->model);
        ds4_model_close(&e->model);
        pthread_mutex_destroy(&e->qlock);
        pthread_cond_destroy(&e->qcv_nonempty);
        pthread_cond_destroy(&e->qcv_nonfull);
        free(e);
        return -ENOMEM;
    }

    /* ---- 7) Spawn worker ------------------------------------------ */
    if (pthread_create(&e->worker, NULL, engine_worker_main, e) != 0) {
        IELOG("pthread_create(worker) failed");
        cudaFree(e->d_argmax);
        free(e->cached_token_ids);
        ds4cuda_tokenizer_free(e->tk);
        ds4_session_state_free(&e->session);
        ds4_model_managed_free(&e->model);
        ds4_model_close(&e->model);
        pthread_mutex_destroy(&e->qlock);
        pthread_cond_destroy(&e->qcv_nonempty);
        pthread_cond_destroy(&e->qcv_nonfull);
        free(e);
        return -EIO;
    }
    /* Wait for worker_ready to ensure it's running before any job. */
    pthread_mutex_lock(&e->qlock);
    while (!e->worker_ready) pthread_cond_wait(&e->qcv_nonfull, &e->qlock);
    pthread_mutex_unlock(&e->qlock);

    *out = e;
    return 0;
}

extern "C" void ds4cuda_inference_engine_destroy(struct ds4cuda_inference_engine *e) {
    if (!e) return;
    pthread_mutex_lock(&e->qlock);
    e->stop = 1;
    pthread_cond_broadcast(&e->qcv_nonempty);
    pthread_cond_broadcast(&e->qcv_nonfull);
    pthread_mutex_unlock(&e->qlock);
    pthread_join(e->worker, NULL);

    /* Drain any leftover jobs (shouldn't happen — callers wait on
     * done_cv — but be defensive).  Each leftover job is signaled
     * done with rc=-ECANCELED so any waiters see the failure. */
    while (e->q_head) {
        struct ie_job *j = e->q_head;
        e->q_head = j->next;
        j->rc = -ECANCELED;
        job_signal_done(j);
    }
    e->q_tail = NULL;
    e->q_count = 0;

    if (e->tk) ds4cuda_tokenizer_free(e->tk);
    if (e->d_argmax) cudaFree(e->d_argmax);
    ds4_session_state_free(&e->session);
    ds4_model_managed_free(&e->model);
    ds4_model_close(&e->model);

    free(e->cached_token_ids);

    pthread_mutex_destroy(&e->qlock);
    pthread_cond_destroy(&e->qcv_nonempty);
    pthread_cond_destroy(&e->qcv_nonfull);
    free(e);
}

/* ------------------------------------------------------------------ */
/* Public generators.                                                   */
/* ------------------------------------------------------------------ */

/* Common helper: enqueue + wait + extract result. */
static int run_one_job(struct ds4cuda_inference_engine *e, struct ie_job *j) {
    int rc = engine_enqueue(e, j);
    if (rc != 0) return rc;
    job_wait(j);
    return j->rc;
}

extern "C" int ds4cuda_real_buffered_generator(
        const char *prompt, int max_new_tokens,
        void *user_data, char **out_text)
{
    if (!out_text) return -EINVAL;
    *out_text = NULL;
    struct ds4cuda_inference_engine *e =
        (struct ds4cuda_inference_engine *)user_data;
    if (!e) return -EINVAL;

    struct ie_job j;
    job_init(&j);
    j.mode = JOB_BUFFERED;
    j.prompt = strdup(prompt ? prompt : "");
    if (!j.prompt) { job_destroy(&j); return -ENOMEM; }
    j.max_new_tokens = max_new_tokens;

    int rc = run_one_job(e, &j);
    if (rc == 0) {
        *out_text = j.result;     /* ownership transfer */
        j.result = NULL;
    } else {
        free(j.result);
    }
    job_destroy(&j);
    return rc;
}

extern "C" int ds4cuda_real_stream_generator(
        const char *prompt, int max_new_tokens,
        void *user_data,
        ds4cuda_chat_stream_emit_fn emit, void *emit_user_data)
{
    if (!emit) return -EINVAL;
    struct ds4cuda_inference_engine *e =
        (struct ds4cuda_inference_engine *)user_data;
    if (!e) return -EINVAL;

    struct ie_job j;
    job_init(&j);
    j.mode = JOB_STREAM;
    j.prompt = strdup(prompt ? prompt : "");
    if (!j.prompt) { job_destroy(&j); return -ENOMEM; }
    j.max_new_tokens = max_new_tokens;
    j.emit = emit;
    j.emit_user_data = emit_user_data;

    int rc = run_one_job(e, &j);
    job_destroy(&j);
    return rc;
}

extern "C" int ds4cuda_real_anthropic_generator(
        const char *dsml_prompt, const char *tools_text,
        int max_new_tokens, void *user_data, char **out_text)
{
    /* tools_text ignored (future: real DSML tool_call generation).
     * For now we degrade gracefully — produce plain text. */
    (void)tools_text;
    return ds4cuda_real_buffered_generator(dsml_prompt, max_new_tokens,
                                           user_data, out_text);
}

extern "C" int ds4cuda_inference_engine_generate_sync(
        struct ds4cuda_inference_engine *e,
        const char *prompt, int max_new_tokens,
        char **out_text)
{
    return ds4cuda_real_buffered_generator(prompt, max_new_tokens,
                                           e, out_text);
}

/* ------------------------------------------------------------------ */
/* Disk save / load wrappers (prefix-sync feature).                     */
/* Both are implemented as admin jobs queued onto the worker thread so  */
/* they observe a quiescent session_state (FIFO ordering w.r.t.         */
/* generation jobs already queued, and serialized w.r.t. the in-flight  */
/* generation job that may still be running).                           */
/* ------------------------------------------------------------------ */
extern "C" int ds4cuda_inference_engine_save_session_to_disk(
        struct ds4cuda_inference_engine *e, const char *path) {
    if (!e || !path) return -EINVAL;
    struct ie_job j;
    job_init(&j);
    j.mode = JOB_SAVE;
    j.path = strdup(path);
    if (!j.path) { job_destroy(&j); return -ENOMEM; }
    int rc = run_one_job(e, &j);
    job_destroy(&j);
    return rc;
}

extern "C" int ds4cuda_inference_engine_load_session_from_disk(
        struct ds4cuda_inference_engine *e, const char *path) {
    if (!e || !path) return -EINVAL;
    struct ie_job j;
    job_init(&j);
    j.mode = JOB_LOAD;
    j.path = strdup(path);
    if (!j.path) { job_destroy(&j); return -ENOMEM; }
    int rc = run_one_job(e, &j);
    job_destroy(&j);
    return rc;
}
