/*
 * model_open.c — high-level open/close + config validation.
 *
 * Pipeline: open → mmap → ds4_gguf_parse (KV+tensor info) →
 *           ds4_config_validate_model → return.
 *
 * Matches ds4 model_open behavior in spirit but reimplements from
 * scratch (no copy of ds4.c). Config keys / expected values are taken
 * from ds4.c:2343–2425.
 */
#define _GNU_SOURCE
#include "ds4cuda.h"
#include "ds4cuda_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* config_validate_model — strict KV → struct ds4_model field copy.    */
/* ------------------------------------------------------------------ */

static const struct ds4_kv *expect_kv(const struct ds4_model *m,
                                      const char *key,
                                      enum ds4_gguf_value_type type)
{
    const struct ds4_kv *kv = ds4_model_find_kv(m, key);
    if (!kv)
        ds4_die("config: missing required KV '%s'", key);
    if (kv->type != type)
        ds4_die("config: KV '%s' has type %u, expected %u",
                key, (unsigned)kv->type, (unsigned)type);
    return kv;
}

static uint32_t expect_u32(const struct ds4_model *m, const char *key,
                           uint32_t want)
{
    const struct ds4_kv *kv = expect_kv(m, key, DS4_GGUF_UINT32);
    if (kv->v.u32 != want)
        ds4_die("config: '%s' = %u, expected %u", key, kv->v.u32, want);
    return kv->v.u32;
}

static uint32_t expect_u32_optional(const struct ds4_model *m,
                                    const char *key,
                                    uint32_t want)
{
    const struct ds4_kv *kv = ds4_model_find_kv(m, key);
    if (!kv) return want;          /* optional: take default */
    if (kv->type != DS4_GGUF_UINT32)
        ds4_die("config: optional '%s' has wrong type %u", key, (unsigned)kv->type);
    if (kv->v.u32 != want)
        ds4_die("config: '%s' = %u, expected %u (optional)",
                key, kv->v.u32, want);
    return kv->v.u32;
}

static void expect_f32(const struct ds4_model *m, const char *key, float want)
{
    const struct ds4_kv *kv = expect_kv(m, key, DS4_GGUF_FLOAT32);
    /* approximate match: values like 1e-6 may carry tiny IEEE noise */
    float v = kv->v.f32;
    float aw = want < 0 ? -want : want;
    float diff = v - want; if (diff < 0) diff = -diff;
    float tol = aw * 1e-5f;
    if (tol < 1e-9f) tol = 1e-9f;
    if (diff > tol)
        ds4_die("config: '%s' = %g, expected %g", key, (double)v, (double)want);
}

static void expect_u64(const struct ds4_model *m, const char *key, uint64_t want)
{
    /* Some GGUF files store this as u32, some as u64 — accept either,
     * but fail loud if it's a different type. */
    const struct ds4_kv *kv = ds4_model_find_kv(m, key);
    if (!kv) ds4_die("config: missing required KV '%s'", key);
    uint64_t got;
    if (kv->type == DS4_GGUF_UINT64)      got = kv->v.u64;
    else if (kv->type == DS4_GGUF_UINT32) got = (uint64_t)kv->v.u32;
    else if (kv->type == DS4_GGUF_INT64)  got = (uint64_t)kv->v.i64;
    else if (kv->type == DS4_GGUF_INT32)  got = (uint64_t)(uint32_t)kv->v.i32;
    else ds4_die("config: '%s' has type %u, expected u32/u64/i32/i64",
                 key, (unsigned)kv->type);
    if (got != want)
        ds4_die("config: '%s' = %llu, expected %llu",
                key, (unsigned long long)got, (unsigned long long)want);
}

static void expect_bool(const struct ds4_model *m, const char *key, int want)
{
    const struct ds4_kv *kv = expect_kv(m, key, DS4_GGUF_BOOL);
    int v = kv->v.b ? 1 : 0;
    if (v != !!want)
        ds4_die("config: '%s' = %d, expected %d", key, v, !!want);
}

static void expect_string(const struct ds4_model *m, const char *key,
                          const char *want)
{
    const struct ds4_kv *kv = expect_kv(m, key, DS4_GGUF_STRING);
    if (strcmp(kv->v.s, want) != 0)
        ds4_die("config: '%s' = '%s', expected '%s'",
                key, kv->v.s, want);
}

void ds4_config_validate_model(struct ds4_model *m)
{
    /* Architecture sentinel — must be deepseek4. */
    expect_string(m, "general.architecture", "deepseek4");

    /* A.1 — U32 scalars, all required, all with fixed expected values. */
    m->n_layer        = expect_u32(m, "deepseek4.block_count",                       43);
    m->n_embd         = expect_u32(m, "deepseek4.embedding_length",                  4096);
    m->n_vocab        = expect_u32(m, "deepseek4.vocab_size",                        129280);
    m->n_head         = expect_u32(m, "deepseek4.attention.head_count",              64);
    m->key_len        = expect_u32(m, "deepseek4.attention.key_length",              512);
    m->head_kv        = expect_u32(m, "deepseek4.attention.head_count_kv",           1);
    m->val_len        = expect_u32(m, "deepseek4.attention.value_length",            512);
    m->rope_dim       = expect_u32(m, "deepseek4.rope.dimension_count",              64);
    m->out_group      = expect_u32(m, "deepseek4.attention.output_group_count",      8);
    m->q_lora_rank    = expect_u32(m, "deepseek4.attention.q_lora_rank",             1024);
    m->out_lora_rank  = expect_u32(m, "deepseek4.attention.output_lora_rank",        1024);
    m->n_expert       = expect_u32(m, "deepseek4.expert_count",                      256);
    m->n_expert_used  = expect_u32(m, "deepseek4.expert_used_count",                 6);
    m->expert_ff      = expect_u32(m, "deepseek4.expert_feed_forward_length",        2048);
    m->n_shared_expert= expect_u32(m, "deepseek4.expert_shared_count",               1);
    m->hash_layer_count = expect_u32(m, "deepseek4.hash_layer_count",                3);
    m->expert_group_count = expect_u32_optional(m, "deepseek4.expert_group_count", 0);
    m->expert_group_used  = expect_u32_optional(m, "deepseek4.expert_group_used_count", 0);
    m->sliding_window = expect_u32(m, "deepseek4.attention.sliding_window",          128);
    m->indexer_head   = expect_u32(m, "deepseek4.attention.indexer.head_count",      64);
    m->indexer_key_len= expect_u32(m, "deepseek4.attention.indexer.key_length",      128);
    m->indexer_top_k  = expect_u32(m, "deepseek4.attention.indexer.top_k",           512);
    m->hc_count       = expect_u32(m, "deepseek4.hyper_connection.count",            4);
    m->hc_sinkhorn_iter = expect_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations", 20);

    /* A.2 — float / u64 scalars + bool. */
    expect_u64(m,  "deepseek4.rope.scaling.original_context_length", 65536);
    expect_f32(m,  "deepseek4.rope.freq_base",                       10000.0f);
    expect_f32(m,  "deepseek4.rope.scaling.factor",                  16.0f);
    expect_f32(m,  "deepseek4.rope.scaling.yarn_beta_fast",          32.0f);
    expect_f32(m,  "deepseek4.rope.scaling.yarn_beta_slow",          1.0f);
    expect_f32(m,  "deepseek4.attention.compress_rope_freq_base",    160000.0f);
    expect_f32(m,  "deepseek4.expert_weights_scale",                 1.5f);
    expect_f32(m,  "deepseek4.attention.layer_norm_rms_epsilon",     1e-6f);
    expect_f32(m,  "deepseek4.hyper_connection.epsilon",             1e-6f);
    expect_bool(m, "deepseek4.expert_weights_norm",                  1);

    /* A.3 — array KVs.
     * compress_ratios: u32/i32 array, length >= 43, value at il must
     *   == ds4_layer_compress_ratio(il) := 0 if il < 2, 4 if even, 128 if odd.
     * swiglu_clamp_exp: f32 array length >= 43, every entry == 10.0f.
     */
    {
        const struct ds4_kv *kv = expect_kv(m, "deepseek4.attention.compress_ratios",
                                            DS4_GGUF_ARRAY);
        if (kv->v.arr.length < 43)
            ds4_die("compress_ratios: length %llu < 43",
                    (unsigned long long)kv->v.arr.length);
        if (kv->v.arr.elem_type != DS4_GGUF_UINT32 &&
            kv->v.arr.elem_type != DS4_GGUF_INT32)
            ds4_die("compress_ratios: elem type %u not u32/i32",
                    (unsigned)kv->v.arr.elem_type);
        for (uint32_t il = 0; il < 43; ++il) {
            uint32_t v;
            memcpy(&v, kv->v.arr.raw + il * 4, 4);
            uint32_t want = (il < 2) ? 0u : ((il & 1u) ? 128u : 4u);
            if (v != want)
                ds4_die("compress_ratios[%u] = %u, expected %u",
                        il, v, want);
        }
    }
    {
        const struct ds4_kv *kv = expect_kv(m, "deepseek4.swiglu_clamp_exp",
                                            DS4_GGUF_ARRAY);
        if (kv->v.arr.length < 43)
            ds4_die("swiglu_clamp_exp: length %llu < 43",
                    (unsigned long long)kv->v.arr.length);
        if (kv->v.arr.elem_type != DS4_GGUF_FLOAT32)
            ds4_die("swiglu_clamp_exp: elem type %u not f32",
                    (unsigned)kv->v.arr.elem_type);
        for (uint32_t il = 0; il < 43; ++il) {
            float v;
            memcpy(&v, kv->v.arr.raw + il * 4, 4);
            float diff = v - 10.0f; if (diff < 0) diff = -diff;
            if (diff > 1e-5f)
                ds4_die("swiglu_clamp_exp[%u] = %g, expected 10.0", il, (double)v);
        }
    }
}

/* ------------------------------------------------------------------ */
/* ds4_model_open / ds4_model_close                                    */
/* ------------------------------------------------------------------ */
int ds4_model_open(struct ds4_model *out, const char *gguf_path)
{
    if (!out || !gguf_path) return -1;
    memset(out, 0, sizeof(*out));
    out->fd = -1;

    int fd = open(gguf_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        ds4_die("open('%s'): %s", gguf_path, strerror(errno));

    struct stat st;
    if (fstat(fd, &st) != 0)
        ds4_die("fstat('%s'): %s", gguf_path, strerror(errno));
    if (st.st_size < 32)
        ds4_die("file '%s' too small (%lld bytes)",
                gguf_path, (long long)st.st_size);

    /* MAP_PRIVATE + MAP_NORESERVE so we don't reserve 81 GB of swap.
     * Linux honors MAP_NORESERVE for read-only maps. We never write. */
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ,
                   MAP_PRIVATE | MAP_NORESERVE, fd, 0);
    if (p == MAP_FAILED)
        ds4_die("mmap('%s', %lld bytes): %s",
                gguf_path, (long long)st.st_size, strerror(errno));

    /* Hint: we'll touch the first ~5 MB heavily and (for now) nothing
     * else. Larger backends will issue their own madvise per-tensor. */
    (void)madvise(p, (size_t)st.st_size, MADV_RANDOM);

    out->fd        = fd;
    out->mmap_ptr  = p;
    out->file_size = (size_t)st.st_size;
    out->weight_backend = DS4_WEIGHT_BACKEND_MANAGED;

    ds4_gguf_parse(out);
    ds4_config_validate_model(out);
    return 0;
}

void ds4_model_close(struct ds4_model *m)
{
    if (!m) return;
    /* The managed weights region (if any) MUST have been freed by the
     * caller via ds4_model_managed_free before reaching here — this TU
     * is pure C and cannot link cudaFree. Warn loud rather than silently
     * leak; the test harnesses always pair them. */
    if (m->weights_managed_base) {
        fprintf(stderr,
                "ds4_model_close: weights_managed_base=%p NOT freed; "
                "call ds4_model_managed_free first (managed leak follows)\n",
                m->weights_managed_base);
        m->weights_managed_base = NULL;
        m->weights_managed_size = 0;
    }
    if (m->mmap_ptr) {
        munmap(m->mmap_ptr, m->file_size);
        m->mmap_ptr = NULL;
    }
    if (m->fd >= 0) {
        close(m->fd);
        m->fd = -1;
    }
    free(m->tensors); m->tensors = NULL;
    free(m->kv);      m->kv = NULL;
    free(m->name_pool); m->name_pool = NULL;
    m->n_tensors = 0;
    m->n_kv = 0;
    m->name_pool_size = 0;
    m->name_pool_used = 0;
}
