// tools/repack_gguf_soa.c
/*
 * repack_gguf_soa — offline GGUF rewriter that adds SoA v2 packed copies
 * of the MoE expert weight tensors (ffn_down_exps in Q2_K, plus ffn_gate_exps
 * and ffn_up_exps in IQ2_XXS) for every layer.
 *
 * Modes:
 *   repack_gguf_soa --dump-tensors <input.gguf>     # diagnostic
 *   repack_gguf_soa [--force] <in.gguf> <out.gguf>             # append mode:
 *                                                     keep all originals,
 *                                                     append 129 SoA tensors
 *                                                     (43 down + 43 gate + 43 up)
 *   repack_gguf_soa --replace [--force] <in.gguf> <out.gguf>   # replace mode:
 *                                                     drop 129 original AoS
 *                                                     expert tensors and emit
 *                                                     129 SoA v2 tensors in
 *                                                     their place (~81 GB
 *                                                     output, same n_tensors
 *                                                     as input).
 *
 * Append-mode output (kept for backward compat):
 *   header (24 B)              : magic + version + n_tensors_new + n_kv
 *   KV section                 : copied byte-for-byte from input
 *   original tensor info recs  : copied byte-for-byte from input
 *   N new tensor info recs     : one per replaceable expert tensor, naming
 *                                "blk.<il>.ffn_{down,gate,up}_exps_soa_v2.weight"
 *   padding                    : zero bytes to alignment
 *   original tensor data blob  : copied byte-for-byte from input
 *   N new SoA v2 tensor blobs  : ds4_repack_q2k_aos_to_soa_v2 (down) or
 *                                ds4_repack_iq2_xxs_aos_to_soa_v2 (gate/up)
 *                                of each layer's expert tensor, each aligned
 *                                to alignment.
 *
 * Replace-mode output:
 *   header (24 B)              : same shape, n_tensors_new = m.n_tensors
 *   KV section                 : byte-for-byte
 *   tensor info recs           : per-record write; AoS expert records are
 *                                *dropped* and SoA v2 records appended with
 *                                newly computed rel_offsets.
 *   padding                    : zero bytes to alignment
 *   tensor data blob           : per-tensor copy in input order, AoS expert
 *                                tensors are *skipped*. SoA v2 tensors append
 *                                at the end.
 *
 * Refuses to overwrite an existing output file unless --force is passed.
 */
#define _GNU_SOURCE
#include "ds4cuda.h"
#include "ds4cuda_soa_layout.h"
#include "ds4cuda_iq2_soa_layout.h"
#include "repack_soa.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* DeepSeek-V4 routed-expert geometry (matches config_validate_model). The
 * down/gate/up tensors differ in (in_dim, out_dim) but all share n_experts
 * and the [in_dim, out_dim, n_experts] dim ordering.
 *   down  : Q2_K     dims=[2048, 4096, 256]   in=2048,  out=4096
 *   gate  : IQ2_XXS  dims=[4096, 2048, 256]   in=4096,  out=2048
 *   up    : IQ2_XXS  dims=[4096, 2048, 256]   in=4096,  out=2048
 * We pull in_dim/out_dim from t->dims[0]/t->dims[1] at runtime and only hard-
 * code n_experts as a sanity check. */
#define REPACK_N_EXPERTS 256

#define DROPPED_OFFSET ((uint64_t)-1)

static void die(const char *fmt, ...) __attribute__((noreturn,format(printf,1,2)));
static void die(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "repack_gguf_soa: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* Diagnostic --dump-tensors mode                                      */
/* ------------------------------------------------------------------ */
static int dump_tensors(const char *path)
{
    struct ds4_model m;
    if (ds4_model_open(&m, path) != 0) die("ds4_model_open failed for %s", path);

    int n_found = 0;
    for (size_t i = 0; i < m.n_tensors; i++) {
        const struct ds4_tensor *t = &m.tensors[i];
        /* Match all three MoE expert families plus their SoA v2 variants. */
        if (!strstr(t->name, "ffn_down_exps") &&
            !strstr(t->name, "ffn_gate_exps") &&
            !strstr(t->name, "ffn_up_exps")) continue;
        printf("%-48s  quant=%d  dims=", t->name, (int)t->quant);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            printf("%s%" PRIu64, d ? "x" : "", t->dims[d]);
        }
        printf("  offset=%" PRIu64 "  bytes=%" PRIu64 "\n",
               t->abs_offset, t->byte_size);
        n_found++;
    }
    printf("\nfound %d ffn_{down,gate,up}_exps tensors\n", n_found);
    ds4_model_close(&m);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Pattern helpers                                                     */
/* ------------------------------------------------------------------ */

/* True iff this tensor is one we replace (offline) with a SoA v2 version.
 *
 * Covers ffn_down_exps (Q2_K) plus ffn_gate_exps and
 * ffn_up_exps (both IQ2_XXS). The quant check is a defense against future
 * model variants that might reuse the name with a different encoding — if
 * the encoding is unexpected we leave the tensor alone rather than drop it
 * and risk emitting bogus SoA bytes. */
static int is_ffn_moe_expert_replaceable(const char *name, int quant)
{
    /* Pattern: "blk." | digits | ".ffn_(down|gate|up)_exps.weight" */
    if (strncmp(name, "blk.", 4) != 0) return 0;
    const char *p = name + 4;
    if (!*p || *p < '0' || *p > '9') return 0;
    while (*p && *p != '.') {
        if (*p < '0' || *p > '9') return 0;
        p++;
    }
    if (*p != '.') return 0;
    if (!strcmp(p, ".ffn_down_exps.weight") && quant == DS4_QUANT_Q2_K)    return 1;
    if (!strcmp(p, ".ffn_gate_exps.weight") && quant == DS4_QUANT_IQ2_XXS) return 1;
    if (!strcmp(p, ".ffn_up_exps.weight")   && quant == DS4_QUANT_IQ2_XXS) return 1;
    return 0;
}

/* "blk.<il>.ffn_X_exps.weight"  ->  "blk.<il>.ffn_X_exps_soa_v2.weight"
 * for X in {down, gate, up}. */
static void make_soa_name(char *buf, size_t buflen, const char *orig)
{
    static const char *const suffixes[] = {
        ".ffn_down_exps.weight",
        ".ffn_gate_exps.weight",
        ".ffn_up_exps.weight",
        NULL,
    };
    static const char *const new_suffixes[] = {
        ".ffn_down_exps_soa_v2.weight",
        ".ffn_gate_exps_soa_v2.weight",
        ".ffn_up_exps_soa_v2.weight",
        NULL,
    };

    const char *dot = NULL;
    int match = -1;
    for (int i = 0; suffixes[i]; i++) {
        const char *d = strstr(orig, suffixes[i]);
        if (d) {
            /* Require the suffix to terminate the string — guards against
             * "...ffn_down_exps.weight.something" false positives. */
            if (d[strlen(suffixes[i])] == '\0') {
                dot = d;
                match = i;
                break;
            }
        }
    }
    if (!dot) die("expected ffn_(down|gate|up)_exps tensor name, got %s", orig);

    size_t prefix    = (size_t)(dot - orig);
    const char *suf  = new_suffixes[match];
    size_t suf_len   = strlen(suf);
    if (prefix + suf_len + 1 > buflen) die("name too long: %s", orig);
    memcpy(buf, orig, prefix);
    memcpy(buf + prefix, suf, suf_len);
    buf[prefix + suf_len] = '\0';
}

/* ------------------------------------------------------------------ */
/* SoA v2 layout + transpose dispatch                                  */
/* ------------------------------------------------------------------ */

/* Returns the SoA v2 byte size for a replaceable expert tensor t.
 * Aborts if quant isn't supported. */
static size_t soa_v2_bytes_for(const struct ds4_tensor *t)
{
    int in_dim    = (int)t->dims[0];
    int out_dim   = (int)t->dims[1];
    int n_experts = (int)t->dims[2];
    if (t->quant == DS4_QUANT_Q2_K) {
        struct ds4_q2k_soa_v2_layout L =
            ds4_q2k_soa_v2_layout(n_experts, out_dim, in_dim);
        return L.total_bytes;
    }
    if (t->quant == DS4_QUANT_IQ2_XXS) {
        struct ds4_iq2_xxs_soa_v2_layout L =
            ds4_iq2_xxs_soa_v2_layout(n_experts, out_dim, in_dim);
        return L.total_bytes;
    }
    die("unsupported quant for SoA repack: %d (tensor '%s')",
        (int)t->quant, t->name);
}

/* Write SoA v2 bytes for tensor t into buf (size soa_v2_bytes_for(t)).
 * src points at the AoS bytes in the input mmap. */
static void soa_v2_repack_to(uint8_t *buf, const struct ds4_tensor *t,
                             const void *src)
{
    int in_dim    = (int)t->dims[0];
    int out_dim   = (int)t->dims[1];
    int n_experts = (int)t->dims[2];
    if (t->quant == DS4_QUANT_Q2_K) {
        ds4_repack_q2k_aos_to_soa_v2((const struct ds4_block_q2_K *)src, buf,
                                     n_experts, out_dim, in_dim);
        return;
    }
    if (t->quant == DS4_QUANT_IQ2_XXS) {
        ds4_repack_iq2_xxs_aos_to_soa_v2(
            (const struct ds4_block_iq2_xxs *)src, buf,
            n_experts, out_dim, in_dim);
        return;
    }
    die("unsupported quant for SoA repack: %d (tensor '%s')",
        (int)t->quant, t->name);
}

/* ------------------------------------------------------------------ */
/* Binary write helpers — all little-endian, all check fwrite return    */
/* ------------------------------------------------------------------ */
static void write_bytes(FILE *f, const void *p, size_t n)
{
    if (n == 0) return;
    size_t off = 0;
    const uint8_t *src = (const uint8_t *)p;
    /* fwrite may return a short count on signals or errors; loop and check. */
    while (off < n) {
        size_t w = fwrite(src + off, 1, n - off, f);
        if (w == 0) die("write short: errno=%s", strerror(errno));
        off += w;
    }
}

static void write_u32(FILE *f, uint32_t v) { write_bytes(f, &v, 4); }
static void write_u64(FILE *f, uint64_t v) { write_bytes(f, &v, 8); }

/* GGUF string format: u64 length + bytes (no terminator on disk). */
static void write_str(FILE *f, const char *s)
{
    uint64_t n = (uint64_t)strlen(s);
    write_u64(f, n);
    write_bytes(f, s, n);
}

static uint64_t ftell_or_die(FILE *f)
{
    long p = ftell(f);
    if (p < 0) die("ftell: %s", strerror(errno));
    return (uint64_t)p;
}

static void pad_to_alignment(FILE *f, uint32_t alignment)
{
    uint64_t pos = ftell_or_die(f);
    uint64_t up = (pos + alignment - 1) / alignment * alignment;
    if (up == pos) return;
    static const uint8_t zeros[4096] = {0};
    uint64_t need = up - pos;
    while (need > 0) {
        size_t chunk = need > sizeof(zeros) ? sizeof(zeros) : (size_t)need;
        write_bytes(f, zeros, chunk);
        need -= chunk;
    }
}

/* Bulk copy from src buffer to file, in chunks (we use this to splat the
 * ~85 GB original data blob to disk). Print a coarse progress line every
 * ~4 GB so the user can watch the I/O progress. */
static void copy_range_progress(FILE *fout, const uint8_t *src, uint64_t n,
                                const char *label)
{
    static const uint64_t REPORT_EVERY = 4ull << 30; /* 4 GiB */
    uint64_t off = 0;
    uint64_t next_report = REPORT_EVERY;
    while (off < n) {
        /* fwrite handles huge sizes on modern glibc, but split into 64 MiB
         * pieces so progress reporting works and stdio doesn't choke. */
        size_t chunk = (n - off) > (64ull << 20) ? (64ull << 20)
                                                  : (size_t)(n - off);
        write_bytes(fout, src + off, chunk);
        off += chunk;
        if (off >= next_report || off == n) {
            fprintf(stderr, "  [%s] %.2f / %.2f GiB\n",
                    label,
                    off / (double)(1ull << 30),
                    n   / (double)(1ull << 30));
            fflush(stderr);
            next_report = off + REPORT_EVERY;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Pretty bytes printer                                                */
/* ------------------------------------------------------------------ */
static void fmt_bytes(uint64_t n, char *out, size_t cap)
{
    double v = (double)n;
    const char *units[] = {"B", "KiB", "MiB", "GiB"};
    int u = 0;
    while (v >= 1024.0 && u < 3) { v /= 1024.0; u++; }
    snprintf(out, cap, "%.2f %s", v, units[u]);
}

/* ------------------------------------------------------------------ */
/* SIGINT/SIGTERM handler: unlink the in-flight tmp file before exit.  */
/* Uses a static char buffer (signal-safe; no heap allocation).        */
/* ------------------------------------------------------------------ */
static char g_tmp_path[PATH_MAX];

static void on_fatal_signal(int sig)
{
    (void)sig;
    if (g_tmp_path[0]) {
        /* unlink() is async-signal-safe per POSIX. */
        unlink(g_tmp_path);
    }
    /* 128 + signal number is the conventional exit code (130 for SIGINT). */
    _exit(128 + sig);
}

/* ------------------------------------------------------------------ */
/* Main write path                                                     */
/* ------------------------------------------------------------------ */
static int rewrite(const char *in_path, const char *out_path,
                   bool replace_mode, bool force)
{
    /* 1. Open input via ds4_model_open (mmap + parse). */
    struct ds4_model m;
    if (ds4_model_open(&m, in_path) != 0) die("ds4_model_open failed for %s", in_path);

    /* 2. Refuse to overwrite existing output unless --force. */
    struct stat st;
    if (!force && stat(out_path, &st) == 0) {
        ds4_model_close(&m);
        die("output '%s' already exists (use --force to overwrite)", out_path);
    }

    /* 3. Identify replaceable expert tensors (down/gate/up).
     *
     * Collect them in input order so the output layout walks the input table
     * in the same order. Validate quant + 3D shape + n_experts consistency
     * per tensor; the in/out dims differ between down and gate/up so we read
     * those from t->dims at use time. */
    const struct ds4_tensor **repl = calloc(m.n_tensors, sizeof(*repl));
    if (!repl) die("calloc");
    size_t n_repl = 0;
    size_t n_down = 0, n_gate = 0, n_up = 0;
    for (size_t i = 0; i < m.n_tensors; i++) {
        const struct ds4_tensor *t = &m.tensors[i];
        if (!is_ffn_moe_expert_replaceable(t->name, (int)t->quant)) continue;
        if (t->n_dims != 3)
            die("'%s' n_dims=%u, expected 3", t->name, t->n_dims);
        if (t->dims[2] != REPACK_N_EXPERTS)
            die("'%s' n_experts=%" PRIu64 ", expected %d",
                t->name, t->dims[2], REPACK_N_EXPERTS);
        /* Sanity: SoA size must equal AoS size (byte-preserving permute). */
        size_t soa_bytes = soa_v2_bytes_for(t);
        if (soa_bytes != t->byte_size)
            die("'%s' SoA layout size (%zu) != AoS size (%" PRIu64 ") — "
                "geometry mismatch", t->name, soa_bytes, t->byte_size);

        repl[n_repl++] = t;
        if (strstr(t->name, "ffn_down_exps"))      n_down++;
        else if (strstr(t->name, "ffn_gate_exps")) n_gate++;
        else if (strstr(t->name, "ffn_up_exps"))   n_up++;
    }
    if (n_repl == 0) die("no replaceable ffn_*_exps tensors found in input");

    fprintf(stderr,
            "repack_gguf_soa: mode=%s, found %zu replaceable tensors "
            "(down=%zu gate=%zu up=%zu)\n",
            replace_mode ? "replace" : "append",
            n_repl, n_down, n_gate, n_up);

    /* 5. Precompute relative offsets for every output tensor.
     *
     * The output data blob layout is:
     *   [for each input tensor in input order]
     *     if (replace_mode && is_replaceable(name, quant)): skip
     *     else: align_up(cursor, m.alignment); offset=cursor; cursor+=byte_size
     *   [for each of n_repl SoA tensors, in input-order]
     *     align_up(cursor, m.alignment); offset=cursor;
     *     cursor += soa_v2_bytes_for(t)   (== AoS byte_size; sanity-checked above)
     *
     * In append mode (replace_mode=false) the cursor walks the original blob
     * exactly because no tensor is skipped and align_up is a no-op when
     * already aligned — so new_orig_rel_offsets[i] should equal the input's
     * rel_offset for kept tensors. We still verify this below.
     *
     * Sentinel: dropped tensors get DROPPED_OFFSET so a misuse trips an
     * obvious crash. */
    uint64_t *new_orig_rel_offsets = calloc(m.n_tensors, sizeof(*new_orig_rel_offsets));
    if (!new_orig_rel_offsets) die("calloc");

    uint64_t blob_cursor = 0;
    for (size_t i = 0; i < m.n_tensors; i++) {
        const struct ds4_tensor *t = &m.tensors[i];
        if (replace_mode &&
            is_ffn_moe_expert_replaceable(t->name, (int)t->quant)) {
            new_orig_rel_offsets[i] = DROPPED_OFFSET;
            continue;
        }
        if (blob_cursor % m.alignment != 0)
            blob_cursor = (blob_cursor + m.alignment - 1) / m.alignment * m.alignment;
        new_orig_rel_offsets[i] = blob_cursor;
        blob_cursor += t->byte_size;
    }
    /* End of kept-originals region. SoA tensors start here (after alignment). */
    if (blob_cursor % m.alignment != 0)
        blob_cursor = (blob_cursor + m.alignment - 1) / m.alignment * m.alignment;

    /* 6. Build SoA tensor names + their relative offsets. */
    char (*new_names)[64] = calloc(n_repl, sizeof(*new_names));
    uint64_t *new_soa_rel_offsets = calloc(n_repl, sizeof(*new_soa_rel_offsets));
    size_t   *new_soa_bytes       = calloc(n_repl, sizeof(*new_soa_bytes));
    if (!new_names || !new_soa_rel_offsets || !new_soa_bytes) die("calloc");

    size_t max_soa_bytes = 0;
    for (size_t i = 0; i < n_repl; i++) {
        const struct ds4_tensor *t = repl[i];
        make_soa_name(new_names[i], sizeof(new_names[i]), t->name);
        new_soa_bytes[i] = soa_v2_bytes_for(t);
        if (new_soa_bytes[i] > max_soa_bytes) max_soa_bytes = new_soa_bytes[i];
        if (blob_cursor % m.alignment != 0)
            blob_cursor = (blob_cursor + m.alignment - 1) / m.alignment * m.alignment;
        new_soa_rel_offsets[i] = blob_cursor;
        if (new_soa_rel_offsets[i] % m.alignment != 0)
            die("BUG: SoA tensor %zu rel_offset %" PRIu64 " not aligned to %u",
                i, new_soa_rel_offsets[i], m.alignment);
        blob_cursor += (uint64_t)new_soa_bytes[i];
    }
    uint64_t out_blob_total = blob_cursor;

    /* 7. Sanity for append mode: each kept original's new rel_offset must
     * equal its input rel_offset (because we mirror the input blob byte-
     * for-byte). The input's rel_offset = abs_offset - tensor_data_pos. */
    if (!replace_mode) {
        for (size_t i = 0; i < m.n_tensors; i++) {
            uint64_t input_rel = m.tensors[i].abs_offset - m.tensor_data_pos;
            if (new_orig_rel_offsets[i] != input_rel)
                die("BUG: append mode but new_orig_rel_offsets[%zu]=%" PRIu64
                    " != input rel %" PRIu64,
                    i, new_orig_rel_offsets[i], input_rel);
        }
    }

    /* 8. Open output. Write to "<out_path>.tmp" first, then rename atomically
     * on success. If we crash / get SIGINT mid-write, the tmp file is unlinked
     * (by handler or by next run) and the final out_path is never touched. */
    char tmp_path[PATH_MAX];
    {
        int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", out_path);
        if (n < 0 || (size_t)n >= sizeof(tmp_path))
            die("output path too long for tmp suffix: %s", out_path);
    }

    /* Clean any stale tmp from a previous interrupted run. */
    (void)unlink(tmp_path);

    /* Install signal handlers so Ctrl+C / SIGTERM unlink the tmp file.
     * Snapshot the tmp_path into a static buffer for async-signal safety. */
    {
        size_t tlen = strlen(tmp_path);
        if (tlen >= sizeof(g_tmp_path)) tlen = sizeof(g_tmp_path) - 1;
        memcpy(g_tmp_path, tmp_path, tlen);
        g_tmp_path[tlen] = '\0';
    }
    signal(SIGINT,  on_fatal_signal);
    signal(SIGTERM, on_fatal_signal);

    FILE *fout = fopen(tmp_path, "wb");
    if (!fout) die("fopen('%s', wb): %s", tmp_path, strerror(errno));

    /* Bigger stdio buffer reduces syscall overhead on the 80 GB byte copy. */
    static char outbuf[1u << 20];   /* 1 MiB */
    setvbuf(fout, outbuf, _IOFBF, sizeof(outbuf));

    const uint8_t *mmap_u8 = (const uint8_t *)m.mmap_ptr;
    /* New n_tensors: append=+n_repl, replace=±0 (drop n_repl, add n_repl). */
    uint64_t n_tensors_new = (uint64_t)m.n_tensors + (uint64_t)n_repl;
    if (replace_mode) n_tensors_new -= (uint64_t)n_repl;

    /* ---- (a) new header (24 bytes) ---- */
    write_u32(fout, 0x46554747u);          /* 'GGUF' little-endian */
    write_u32(fout, 3u);                    /* version 3 */
    write_u64(fout, n_tensors_new);
    write_u64(fout, (uint64_t)m.n_kv);

    /* ---- (b) KV section (byte-copy [24, kv_byte_range_end)) ---- */
    {
        uint64_t kv_bytes = m.kv_byte_range_end - 24;
        write_bytes(fout, mmap_u8 + 24, (size_t)kv_bytes);
    }

    /* ---- (c) original tensor info records ----
     * Append mode: byte-copy the whole info section (offsets unchanged).
     * Replace mode: per-record write, skipping dropped tensors and using
     *               recomputed offsets. */
    if (!replace_mode) {
        uint64_t info_bytes = m.tensor_info_byte_range_end - m.kv_byte_range_end;
        write_bytes(fout, mmap_u8 + m.kv_byte_range_end, (size_t)info_bytes);
    } else {
        for (size_t i = 0; i < m.n_tensors; i++) {
            const struct ds4_tensor *t = &m.tensors[i];
            if (new_orig_rel_offsets[i] == DROPPED_OFFSET) continue;
            write_str(fout, t->name);
            write_u32(fout, t->n_dims);
            for (uint32_t d = 0; d < t->n_dims; d++)
                write_u64(fout, t->dims[d]);
            write_u32(fout, (uint32_t)t->quant);
            write_u64(fout, new_orig_rel_offsets[i]);
        }
    }

    /* ---- (d) append n_repl new SoA tensor info records ---- */
    for (size_t i = 0; i < n_repl; i++) {
        const struct ds4_tensor *t = repl[i];
        write_str(fout, new_names[i]);
        write_u32(fout, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++)
            write_u64(fout, t->dims[d]);
        write_u32(fout, (uint32_t)t->quant);
        write_u64(fout, new_soa_rel_offsets[i]);
    }

    /* ---- (e) padding to alignment ---- */
    pad_to_alignment(fout, m.alignment);

    uint64_t out_data_pos = ftell_or_die(fout);
    fprintf(stderr, "repack_gguf_soa: output tensor_data_pos = %" PRIu64 " (input was %" PRIu64 ")\n",
            out_data_pos, m.tensor_data_pos);

    /* ---- (f) write tensor data blob ----
     * Append mode: byte-copy the original blob then append SoA tensors.
     * Replace mode: per-tensor copy in input order, skipping dropped, then
     *               append SoA tensors.
     *
     * Either way, after this section the file position relative to
     * out_data_pos must equal out_blob_total. */
    if (!replace_mode) {
        const uint64_t original_blob_size = m.file_size - m.tensor_data_pos;
        fprintf(stderr, "repack_gguf_soa: copying original data blob (%.2f GiB)...\n",
                original_blob_size / (double)(1ull << 30));
        struct timespec ts_blob_a, ts_blob_b;
        clock_gettime(CLOCK_MONOTONIC, &ts_blob_a);
        copy_range_progress(fout, mmap_u8 + m.tensor_data_pos,
                            original_blob_size, "copy");
        clock_gettime(CLOCK_MONOTONIC, &ts_blob_b);
        fprintf(stderr, "repack_gguf_soa: original blob copy took %.1f s\n",
                (ts_blob_b.tv_sec - ts_blob_a.tv_sec) +
                1e-9 * (ts_blob_b.tv_nsec - ts_blob_a.tv_nsec));

        uint64_t blob_pos_check = ftell_or_die(fout) - out_data_pos;
        if (blob_pos_check != original_blob_size)
            die("BUG: blob_pos_check (%" PRIu64 ") != original_blob_size (%" PRIu64 ")",
                blob_pos_check, original_blob_size);
    } else {
        /* Per-tensor copy: walk input table in order, skip dropped tensors,
         * pad to alignment before each kept tensor. */
        fprintf(stderr, "repack_gguf_soa: copying kept tensors (replace mode)...\n");
        struct timespec ts_blob_a, ts_blob_b;
        clock_gettime(CLOCK_MONOTONIC, &ts_blob_a);

        uint64_t bytes_written_since_report = 0;
        const uint64_t REPORT_EVERY = 8ull << 30; /* 8 GiB */
        uint64_t total_bytes_written = 0;
        for (size_t i = 0; i < m.n_tensors; i++) {
            if (new_orig_rel_offsets[i] == DROPPED_OFFSET) continue;
            const struct ds4_tensor *t = &m.tensors[i];

            /* Pad to alignment before each tensor. */
            pad_to_alignment(fout, m.alignment);

            /* Verify file position matches predicted offset. */
            uint64_t pre = ftell_or_die(fout);
            uint64_t pre_rel = pre - out_data_pos;
            if (pre_rel != new_orig_rel_offsets[i])
                die("BUG: kept tensor '%s' (input idx %zu): ftell rel %" PRIu64
                    " != predicted %" PRIu64,
                    t->name, i, pre_rel, new_orig_rel_offsets[i]);

            /* Write tensor bytes in 64-MiB chunks. */
            const uint8_t *src = mmap_u8 + t->abs_offset;
            uint64_t remaining = t->byte_size;
            while (remaining > 0) {
                size_t chunk = remaining > (64ull << 20) ? (64ull << 20)
                                                        : (size_t)remaining;
                write_bytes(fout, src, chunk);
                src += chunk;
                remaining -= chunk;
            }
            total_bytes_written += t->byte_size;
            bytes_written_since_report += t->byte_size;
            if (bytes_written_since_report >= REPORT_EVERY) {
                fprintf(stderr, "  [copy] %.2f GiB written\n",
                        total_bytes_written / (double)(1ull << 30));
                fflush(stderr);
                bytes_written_since_report = 0;
            }
        }
        fprintf(stderr, "  [copy] %.2f GiB written (total)\n",
                total_bytes_written / (double)(1ull << 30));

        clock_gettime(CLOCK_MONOTONIC, &ts_blob_b);
        fprintf(stderr, "repack_gguf_soa: per-tensor copy took %.1f s\n",
                (ts_blob_b.tv_sec - ts_blob_a.tv_sec) +
                1e-9 * (ts_blob_b.tv_nsec - ts_blob_a.tv_nsec));
    }

    /* ---- (g) align before the new SoA section ---- */
    pad_to_alignment(fout, m.alignment);
    uint64_t soa_start_rel = ftell_or_die(fout) - out_data_pos;
    if (soa_start_rel != new_soa_rel_offsets[0])
        die("BUG: SoA section starts at rel_off %" PRIu64
            " but precomputed offset[0] = %" PRIu64,
            soa_start_rel, new_soa_rel_offsets[0]);

    /* ---- (h) per-tensor: repack AoS -> SoA v2 and write ----
     * Walks repl[] in input order, which matches the order info records were
     * appended in step (d). One scratch buffer sized to the largest SoA tensor
     * (the Q2_K down tensors are largest at ~672 MiB; gate/up are ~528 MiB). */
    uint8_t *soa_buf = (uint8_t *)malloc(max_soa_bytes);
    if (!soa_buf) die("malloc soa_buf (%zu bytes)", max_soa_bytes);

    struct timespec ts_repack_a, ts_repack_b;
    clock_gettime(CLOCK_MONOTONIC, &ts_repack_a);
    for (size_t i = 0; i < n_repl; i++) {
        const struct ds4_tensor *t = repl[i];
        const size_t bytes = new_soa_bytes[i];

        /* Source AoS bytes via mmap (NOT ds4_tensor_managed_ptr — managed
         * region is not populated for this tool). The dispatch picks the
         * right transposer (Q2_K vs IQ2_XXS) based on t->quant. */
        const void *aos = mmap_u8 + t->abs_offset;
        soa_v2_repack_to(soa_buf, t, aos);

        /* Verify file position matches predicted offset before writing. */
        uint64_t pre = ftell_or_die(fout);
        uint64_t pre_rel = pre - out_data_pos;
        if (pre_rel != new_soa_rel_offsets[i])
            die("BUG: SoA tensor %zu ('%s'): ftell rel %" PRIu64 " != predicted %" PRIu64,
                i, new_names[i], pre_rel, new_soa_rel_offsets[i]);

        write_bytes(fout, soa_buf, bytes);

        /* Progress: one line per tensor. */
        fprintf(stderr, "  [soa] %zu/%zu  %s  rel_off=%" PRIu64 "  bytes=%zu\n",
                i + 1, n_repl, new_names[i], new_soa_rel_offsets[i], bytes);
        fflush(stderr);
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_repack_b);
    fprintf(stderr, "repack_gguf_soa: %zu-tensor repack+write took %.1f s\n",
            n_repl,
            (ts_repack_b.tv_sec - ts_repack_a.tv_sec) +
            1e-9 * (ts_repack_b.tv_nsec - ts_repack_a.tv_nsec));

    free(soa_buf);

    /* Final position check: must match precomputed blob total. */
    uint64_t end_rel = ftell_or_die(fout) - out_data_pos;
    if (end_rel != out_blob_total)
        die("BUG: end-of-data rel %" PRIu64 " != precomputed out_blob_total %" PRIu64,
            end_rel, out_blob_total);

    /* ---- (i) flush + fsync + close + atomic rename ---- */
    if (fflush(fout) != 0) die("fflush: %s", strerror(errno));
    /* fsync durably flushes file data to disk before we rename. Without this,
     * a power loss between rename() and the kernel writeback could leave a
     * zero-length or partial file at out_path that still "exists". */
    if (fsync(fileno(fout)) != 0) die("fsync: %s", strerror(errno));
    if (fclose(fout) != 0) die("fclose: %s", strerror(errno));

    /* Atomic rename: tmp file is durably written and complete. After this
     * point a crash leaves either the old out_path (if any) or the new
     * complete out_path, never a partial. */
    if (rename(tmp_path, out_path) != 0)
        die("rename('%s' -> '%s'): %s", tmp_path, out_path, strerror(errno));

    /* Clear the static tmp_path so a late SIGINT (during cleanup below)
     * doesn't unlink the now-renamed final file. */
    g_tmp_path[0] = '\0';

    /* Report final file size. */
    if (stat(out_path, &st) != 0) die("stat output: %s", strerror(errno));
    char sz_a[32], sz_b[32];
    fmt_bytes((uint64_t)st.st_size, sz_a, sizeof(sz_a));
    fmt_bytes(m.file_size, sz_b, sizeof(sz_b));
    fprintf(stderr, "repack_gguf_soa: wrote %s  (input was %s, mode=%s)\n",
            sz_a, sz_b, replace_mode ? "replace" : "append");

    free(new_names);
    free(new_soa_rel_offsets);
    free(new_soa_bytes);
    free(new_orig_rel_offsets);
    free(repl);
    ds4_model_close(&m);
    return 0;
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */
static void usage(const char *prog)
{
    fprintf(stderr, "usage: %s [--replace] [--force] <input.gguf> <output.gguf>\n", prog);
    fprintf(stderr, "       %s --dump-tensors <input.gguf>\n", prog);
}

int main(int argc, char **argv)
{
    bool replace_mode = false;
    bool force = false;
    const char *positional[2] = {NULL, NULL};
    int n_pos = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dump-tensors")) {
            /* Early-return path. Expects exactly one path immediately after
             * the flag; any other prior flags (--replace/--force) are
             * ignored — this is the diagnostic shortcut. */
            if (i + 1 >= argc) {
                fprintf(stderr, "--dump-tensors needs a path\n");
                usage(argv[0]);
                return 2;
            }
            if (i + 2 != argc) {
                fprintf(stderr, "--dump-tensors takes exactly one trailing "
                                "path argument\n");
                usage(argv[0]);
                return 2;
            }
            return dump_tensors(argv[i + 1]);
        }
        if (!strcmp(argv[i], "--replace")) { replace_mode = true; continue; }
        if (!strcmp(argv[i], "--force"))   { force = true;        continue; }
        if (argv[i][0] == '-') {
            fprintf(stderr, "unknown flag: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
        if (n_pos >= 2) {
            fprintf(stderr, "too many positional arguments\n");
            usage(argv[0]);
            return 2;
        }
        positional[n_pos++] = argv[i];
    }

    if (n_pos != 2) {
        usage(argv[0]);
        return 2;
    }

    return rewrite(positional[0], positional[1], replace_mode, force);
}
