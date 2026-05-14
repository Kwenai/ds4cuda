/*
 * tokenizer.c — DeepSeek V4 Flash GPT-2 byte-level BPE tokenizer.
 *
 * Direct port of ds4/ds4.c lines 13919-14588 (tokenizer + chat-template
 * encoding). The original file is mmap-aware (it parks a `ds4_str` view
 * into the mapped GGUF region for every token / merge string); we mirror
 * that strategy here against ds4cuda's `ds4_kv` array refs:
 *
 *   - kv->v.arr.raw points to the first element of a STRING array
 *     (length-prefixed by uint64 + raw bytes — see gguf_parser.c
 *     parse_value GGUF_VALUE_ARRAY case).
 *   - tokens / merges live inside the mmap'd region; the tokenizer
 *     stores `(const char *ptr, uint64_t len)` views for each entry,
 *     i.e. zero data copy.
 *
 * Chat-template encoding (ds4.c:14463/14524) is folded into
 * ds4cuda_tokenize: we scan for the seven hard-coded special-token
 * UTF-8 byte sequences in the input and substitute their cached id;
 * any non-special spans go through bpe_tokenize_text (the JoyAI pre-
 * tokenizer + byte-level BPE merge loop).
 *
 * Detokenization is the byte-decode inverse of byte_encode + GPT-2
 * codepoint mapping (ds4.c:14087). Special tokens (id >= regular
 * vocab range) round-trip via the raw token-string view, which for
 * V4 already encodes them as their canonical UTF-8 form.
 */
#define _GNU_SOURCE
#include "tokenizer.h"

#include "ds4cuda.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Sized string view (matches ds4.c ds4_str).                          */
/* ------------------------------------------------------------------ */
typedef struct {
    const char *ptr;
    uint64_t    len;
} tk_str;

static int tk_str_eq_bytes(tk_str s, const char *p, size_t n)
{
    return s.len == n && memcmp(s.ptr, p, n) == 0;
}

/* ------------------------------------------------------------------ */
/* Hash helpers (FNV-1a 64; mirror ds4.c:427).                         */
/* ------------------------------------------------------------------ */
static uint64_t hash_bytes(const void *ptr, uint64_t len)
{
    const uint8_t *p = (const uint8_t *)ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t next_pow2(uint64_t n)
{
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ------------------------------------------------------------------ */
/* Open-addressed (string -> int32) hash table. Mirrors ds4.c          */
/* str_i32_table (ds4.c:13947). Keys are (ptr,len) views into the      */
/* mmap'd GGUF; we never copy.                                         */
/* ------------------------------------------------------------------ */
typedef struct {
    tk_str  key;
    int32_t value;
    uint8_t used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t       cap;
    uint64_t       n;
} str_i32_table;

static int table_init(str_i32_table *t, uint64_t expected)
{
    /* Load factor ~0.5; +16 to avoid degenerate small caps.
     *
     * Guard the `expected * 2 + 16` against unsigned overflow: a malformed
     * GGUF tokenizer array could declare a token count near UINT64_MAX,
     * which would wrap to a tiny value and produce a too-small hash
     * table.  Reject impossibly large requests up front. */
    if (expected > (UINT64_MAX - 16) / 2) {
        t->cap = 0;
        t->n   = 0;
        t->entry = NULL;
        return -EINVAL;
    }
    uint64_t want = expected * 2 + 16;
    /* next_pow2 returns a uint64_t; the calloc cast goes through size_t,
     * so on 32-bit a 64-bit cap may itself overflow.  Reject explicitly. */
    if (want > SIZE_MAX / sizeof(t->entry[0])) {
        t->cap = 0;
        t->n   = 0;
        t->entry = NULL;
        return -ENOMEM;
    }
    t->cap = next_pow2(want);
    if (t->cap == 0 || t->cap > SIZE_MAX / sizeof(t->entry[0])) {
        /* next_pow2 wrapped at the top of the range. */
        t->cap = 0;
        t->n   = 0;
        t->entry = NULL;
        return -ENOMEM;
    }
    t->n   = 0;
    t->entry = (str_i32_entry *)calloc((size_t)t->cap, sizeof(t->entry[0]));
    return t->entry ? 0 : -ENOMEM;
}

static void table_free(str_i32_table *t)
{
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, tk_str key, int32_t value)
{
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (t->entry[i].key.len == key.len &&
            memcmp(t->entry[i].key.ptr, key.ptr, key.len) == 0) {
            t->entry[i].value = value;   /* keep first id; should not happen */
            return;
        }
        i = (i + 1) & mask;
    }
    t->entry[i].used  = 1;
    t->entry[i].key   = key;
    t->entry[i].value = value;
    t->n++;
}

static int table_get(const str_i32_table *t, const char *ptr, uint64_t len, int32_t *value)
{
    if (t->cap == 0) return 0;
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        tk_str k = t->entry[i].key;
        if (k.len == len && memcmp(k.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return 1;
        }
        i = (i + 1) & mask;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Tokenizer handle.                                                   */
/* ------------------------------------------------------------------ */
struct ds4cuda_tokenizer {
    /* Token-string views; each ds4cuda_tokenizer->token[id] is a            */
    /* (ptr,len) view into the mmap'd GGUF tokenizer.ggml.tokens array.     */
    tk_str       *token;
    int           n_vocab;

    /* Special token ids cached for the chat template / detokenize path.   */
    int32_t bos_id;
    int32_t eos_id;
    int32_t user_id;
    int32_t assistant_id;
    int32_t think_start_id;
    int32_t think_end_id;
    int32_t dsml_id;

    /* Hash tables.                                                         */
    str_i32_table token_to_id;
    str_i32_table merge_rank;

    /* Inverse byte map: codepoint (0..511) -> raw byte (0..255). We      */
    /* materialize it once at init for fast detokenize. -1 = not mapped.   */
    int16_t cp_to_byte[512];
};

/* ------------------------------------------------------------------ */
/* GPT-2 byte->codepoint mapping (mirror of ds4.c:14087).              */
/* ------------------------------------------------------------------ */
static uint32_t gpt2_byte_to_codepoint(uint8_t b)
{
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }
    /* The remaining 68 bytes get mapped to 256..323 in input order.      */
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

/* UTF-8 helpers — port of ds4.c:14069 / 14117 / 14223 / 14259.        */
static void utf8_put(char **p, uint32_t cp)
{
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static int utf8_len_from_first_byte(uint8_t c)
{
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos)
{
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next)
{
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

/* ------------------------------------------------------------------ */
/* Token-id-vector (resizable int buffer).                             */
/* ------------------------------------------------------------------ */
typedef struct {
    int *v;
    int  len;
    int  cap;
} tok_vec;

static int tv_push(tok_vec *tv, int token)
{
    if (tv->len == tv->cap) {
        int nc = tv->cap ? tv->cap * 2 : 64;
        int *nv = (int *)realloc(tv->v, (size_t)nc * sizeof(*nv));
        if (!nv) return -ENOMEM;
        tv->v   = nv;
        tv->cap = nc;
    }
    tv->v[tv->len++] = token;
    return 0;
}

static void tv_free(tok_vec *tv)
{
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

/* ------------------------------------------------------------------ */
/* Byte-level encode of a raw piece -> heap-allocated UTF-8 buffer.    */
/* (mirror of ds4.c:14105 byte_encode).                                */
/* ------------------------------------------------------------------ */
static char *byte_encode(const char *in_ptr, uint64_t in_len, uint64_t *out_len)
{
    /* Each byte expands to <=4 UTF-8 bytes (max codepoint 323 -> 2-byte */
    /* UTF-8); the original ds4 uses *4 to be safe.                     */
    char *out = (char *)malloc((size_t)in_len * 4 + 1);
    if (!out) return NULL;
    char *p = out;

    for (uint64_t i = 0; i < in_len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in_ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

/* ------------------------------------------------------------------ */
/* Owned per-symbol slice (BPE working set).                           */
/* ------------------------------------------------------------------ */
typedef struct {
    char    *ptr;
    uint64_t len;
} owned_str;

static int owned_copy(owned_str *out, const char *ptr, uint64_t len)
{
    out->ptr = (char *)malloc((size_t)len);
    if (!out->ptr) return -ENOMEM;
    memcpy(out->ptr, ptr, (size_t)len);
    out->len = len;
    return 0;
}

/* Look up the merge rank for two adjacent BPE symbols (ds4.c:14139).  */
static int bpe_rank(const struct ds4cuda_tokenizer *t,
                    const owned_str *a, const owned_str *b)
{
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = (len <= sizeof(stack)) ? stack : (char *)malloc((size_t)len);
    if (!buf) return -1;

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int32_t rank = -1;
    table_get(&t->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return (int)rank;
}

/* ------------------------------------------------------------------ */
/* Apply byte-level BPE to one regex-pre-tokenized piece, emit ids.    */
/* (mirror of ds4.c:14156 bpe_emit_piece). Returns 0 on success.        */
/* ------------------------------------------------------------------ */
static int bpe_emit_piece(const struct ds4cuda_tokenizer *t,
                          const char *raw_ptr, uint64_t raw_len,
                          tok_vec *out)
{
    if (raw_len == 0) return 0;

    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_ptr, raw_len, &encoded_len);
    if (!encoded) return -ENOMEM;

    int n_sym  = 0;
    int cap_sym = 32;
    owned_str *sym = (owned_str *)calloc((size_t)cap_sym, sizeof(sym[0]));
    if (!sym) { free(encoded); return -ENOMEM; }

    /* Split encoded UTF-8 into per-codepoint symbols.                  */
    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            int nc = cap_sym * 2;
            owned_str *ns = (owned_str *)realloc(sym, (size_t)nc * sizeof(sym[0]));
            if (!ns) goto oom;
            sym = ns;
            cap_sym = nc;
        }
        if (owned_copy(&sym[n_sym], encoded + off, (uint64_t)n) < 0) goto oom;
        n_sym++;
        off += (uint64_t)n;
    }

    /* Iteratively pick the best (lowest-rank) merge until none apply.  */
    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(t, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = (char *)malloc((size_t)merged.len);
        if (!merged.ptr) goto oom;
        memcpy(merged.ptr,                   sym[best_i].ptr,     (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
        n_sym--;
    }

    /* Emit tokens (per-symbol vocab lookup, with fallback to byte-by-  */
    /* byte if a symbol somehow isn't in the vocab — same fallback as   */
    /* ds4.c:14210).                                                    */
    for (int i = 0; i < n_sym; i++) {
        int32_t token = -1;
        if (table_get(&t->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            if (tv_push(out, token) < 0) goto oom;
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&t->token_to_id, sym[i].ptr + j, 1, &token)) {
                    if (tv_push(out, token) < 0) goto oom;
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
    return 0;

oom:
    for (int i = 0; i < n_sym; i++) free(sym[i].ptr);
    free(sym);
    free(encoded);
    return -ENOMEM;
}

/* ------------------------------------------------------------------ */
/* JoyAI/DeepSeek pre-tokenization. Identical port of ds4.c:14229..    */
/* 14398 (helpers + main loop). The split shape matters: different     */
/* pieces lead to different BPE merges even when the final text bytes  */
/* are identical.                                                      */
/* ------------------------------------------------------------------ */
static int  ascii_alpha(uint8_t c)   { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static int  ascii_digit(uint8_t c)   { return c >= '0' && c <= '9'; }
static int  ascii_space(uint8_t c)   { return c == ' '||c=='\t'||c=='\n'||c=='\r'||c=='\v'||c=='\f'; }
static int  ascii_newline(uint8_t c) { return c == '\n' || c == '\r'; }
static int  joyai_ascii_punct_symbol(uint8_t c)
{
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static int utf8_is_cjk_hira_kata(uint32_t cp)
{
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static int joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos)
{
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);
    /* Same heuristic as ds4.c:14281: any non-ASCII byte that's not a     */
    /* punctuation/CJK symbol counts as a letter. CJK is checked          */
    /* separately upstream so we reach this branch only for accented      */
    /* Latin / Cyrillic / etc.                                            */
    return 1;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos)
{
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static int joyai_cjk_at(const char *s, uint64_t len, uint64_t pos)
{
    if ((uint8_t)s[pos] < 128) return 0;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

/* ds4.c:14331 bpe_tokenize_text — pre-token split + per-piece BPE.   */
static int bpe_tokenize_text(const struct ds4cuda_tokenizer *t,
                             const char *text, uint64_t len,
                             tok_vec *out)
{
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do { pos = next_utf8_char(text, len, pos); }
            while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        int rc = bpe_emit_piece(t, text + start, pos - start, out);
        if (rc < 0) return rc;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* GGUF metadata access: read tokenizer.ggml.tokens / merges as raw    */
/* string-array views into the mmap'd region.                           */
/*                                                                      */
/* Layout (gguf_parser.c parse_value, GGUF_VALUE_ARRAY case):          */
/*   kv->v.arr.elem_type = inner type (DS4_GGUF_STRING).               */
/*   kv->v.arr.length    = # elements.                                  */
/*   kv->v.arr.raw       = pointer JUST AFTER the (inner_type,length)  */
/*                          pair, i.e. first element header.           */
/*                                                                      */
/* Each STRING element is stored as (uint64 len, raw bytes...).        */
/* ------------------------------------------------------------------ */
static int read_string_array_views(const struct ds4_kv *kv,
                                   const uint8_t *file_base, size_t file_size,
                                   tk_str **out_arr, size_t *out_n)
{
    *out_arr = NULL;
    *out_n   = 0;

    if (!kv || kv->type != DS4_GGUF_ARRAY) return -EINVAL;
    if (kv->v.arr.elem_type != DS4_GGUF_STRING) return -EINVAL;

    uint64_t n = kv->v.arr.length;
    const uint8_t *p   = kv->v.arr.raw;
    const uint8_t *end = file_base + file_size;
    if (!p || p < file_base || p > end) return -EINVAL;

    tk_str *arr = (tk_str *)calloc((size_t)n, sizeof(*arr));
    if (n && !arr) return -ENOMEM;

    for (uint64_t i = 0; i < n; i++) {
        if (p + 8 > end) { free(arr); return -EINVAL; }
        uint64_t slen;
        memcpy(&slen, p, 8);
        p += 8;
        if (p + slen > end) { free(arr); return -EINVAL; }
        arr[i].ptr = (const char *)p;
        arr[i].len = slen;
        p += slen;
    }
    *out_arr = arr;
    *out_n   = (size_t)n;
    return 0;
}

/* ------------------------------------------------------------------ */
/* tokenizer init / free                                                */
/* ------------------------------------------------------------------ */
static int vocab_lookup_required(const struct ds4cuda_tokenizer *t, const char *text)
{
    int32_t id = -1;
    if (!table_get(&t->token_to_id, text, strlen(text), &id)) {
        fprintf(stderr,
                "ds4cuda: required tokenizer special token missing: %s\n",
                text);
        return -1;
    }
    return (int)id;
}

int ds4cuda_tokenizer_init(struct ds4cuda_tokenizer **out,
                            const struct ds4_model *m)
{
    if (out) *out = NULL;
    if (!m) return -EINVAL;
    if (!m->mmap_ptr || m->file_size == 0) return -EINVAL;

    const struct ds4_kv *kv_tokens = ds4_model_find_kv(m, "tokenizer.ggml.tokens");
    const struct ds4_kv *kv_merges = ds4_model_find_kv(m, "tokenizer.ggml.merges");
    if (!kv_tokens || !kv_merges) {
        fprintf(stderr, "ds4cuda: GGUF tokenizer KV missing\n");
        return -EINVAL;
    }

    tk_str *tokens_arr = NULL; size_t n_tokens = 0;
    tk_str *merges_arr = NULL; size_t n_merges = 0;

    int rc = read_string_array_views(kv_tokens,
                                     (const uint8_t *)m->mmap_ptr, m->file_size,
                                     &tokens_arr, &n_tokens);
    if (rc < 0) return rc;
    rc = read_string_array_views(kv_merges,
                                 (const uint8_t *)m->mmap_ptr, m->file_size,
                                 &merges_arr, &n_merges);
    if (rc < 0) { free(tokens_arr); return rc; }

    if (n_tokens == 0 || n_tokens > (size_t)INT32_MAX) {
        free(tokens_arr); free(merges_arr);
        return -EINVAL;
    }

    struct ds4cuda_tokenizer *t = (struct ds4cuda_tokenizer *)calloc(1, sizeof(*t));
    if (!t) { free(tokens_arr); free(merges_arr); return -ENOMEM; }

    t->n_vocab = (int)n_tokens;
    t->token   = tokens_arr;   /* takes ownership */

    if (table_init(&t->token_to_id, n_tokens) < 0) {
        free(merges_arr); ds4cuda_tokenizer_free(t); return -ENOMEM;
    }
    for (int i = 0; i < t->n_vocab; i++) {
        table_put(&t->token_to_id, t->token[i], (int32_t)i);
    }

    if (table_init(&t->merge_rank, n_merges) < 0) {
        free(merges_arr); ds4cuda_tokenizer_free(t); return -ENOMEM;
    }
    for (size_t i = 0; i < n_merges; i++) {
        table_put(&t->merge_rank, merges_arr[i], (int32_t)i);
    }
    /* merges_arr backs the merge_rank entries' string views. We have to */
    /* keep it alive (the views point into the mmap, but the array spine */
    /* itself is heap-allocated). Park it on the tokenizer for free.     */
    /* NOTE: we deliberately leak a pointer here for symmetry with the   */
    /* tokens path; we re-locate it inside `t`. Allocate a single owned  */
    /* spine then drop merges_arr.                                       */
    /* (token_to_id keeps tk_str copies by value; merge_rank likewise.   */
    /* Both copy the (ptr,len) tuple, NOT the bytes — which live in mmap. */
    /* The merges_arr array itself is no longer needed after table_put.  */
    free(merges_arr);

    /* Cache special token ids (mirror of ds4.c:14444).                 */
    /* The ｜ character is U+FF5C (UTF-8 ef bd 9c), ▁ is U+2581 (UTF-8   */
    /* e2 96 81). The exact UTF-8 byte sequences below match the GGUF    */
    /* token strings in DeepSeek-V4-Flash.                              */
    int b  = vocab_lookup_required(t, "<\xef\xbd\x9c" "begin\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>");
    int e  = vocab_lookup_required(t, "<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>");
    int u  = vocab_lookup_required(t, "<\xef\xbd\x9c" "User\xef\xbd\x9c>");
    int a  = vocab_lookup_required(t, "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>");
    int ts = vocab_lookup_required(t, "<think>");
    int te = vocab_lookup_required(t, "</think>");
    int dm = vocab_lookup_required(t, "\xef\xbd\x9c" "DSML\xef\xbd\x9c");

    if (b < 0 || e < 0 || u < 0 || a < 0 || ts < 0 || te < 0 || dm < 0) {
        ds4cuda_tokenizer_free(t);
        return -EINVAL;
    }
    t->bos_id         = b;
    t->eos_id         = e;
    t->user_id        = u;
    t->assistant_id   = a;
    t->think_start_id = ts;
    t->think_end_id   = te;
    t->dsml_id        = dm;

    /* Build inverse byte map for detokenize (fixed 0..511 range —      */
    /* gpt2_byte_to_codepoint never emits a cp >= 512).                 */
    for (int i = 0; i < 512; i++) t->cp_to_byte[i] = -1;
    for (int b8 = 0; b8 < 256; b8++) {
        uint32_t cp = gpt2_byte_to_codepoint((uint8_t)b8);
        if (cp < 512 && t->cp_to_byte[cp] < 0) {
            t->cp_to_byte[cp] = (int16_t)b8;
        }
    }

    *out = t;
    return 0;
}

void ds4cuda_tokenizer_free(struct ds4cuda_tokenizer *t)
{
    if (!t) return;
    free(t->token);
    table_free(&t->token_to_id);
    table_free(&t->merge_rank);
    free(t);
}

/* ------------------------------------------------------------------ */
/* Public encode (with chat-template special-token recognition).       */
/* ------------------------------------------------------------------ */
struct special_marker {
    const char *bytes;
    size_t      len;
    int         id_offset;   /* offset of cached id in the tokenizer struct */
};

static bool match_special_at(const struct ds4cuda_tokenizer *t,
                             const char *p, size_t remain,
                             int *out_id, size_t *out_len)
{
    /* Order matters: longer markers first to avoid matching "<think>" inside */
    /* a hypothetical "<thinking>" — but here the seven markers do not        */
    /* overlap on prefix.                                                     */
    static const char *const TXT_BOS  = "<\xef\xbd\x9c" "begin\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>";
    static const char *const TXT_EOS  = "<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>";
    static const char *const TXT_USER = "<\xef\xbd\x9c" "User\xef\xbd\x9c>";
    static const char *const TXT_AST  = "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>";
    static const char *const TXT_THO  = "<think>";
    static const char *const TXT_THC  = "</think>";
    static const char *const TXT_DSML = "\xef\xbd\x9c" "DSML\xef\xbd\x9c";

    struct slot { const char *s; int id; };
    struct slot slots[] = {
        { TXT_BOS,  t->bos_id         },
        { TXT_EOS,  t->eos_id         },
        { TXT_USER, t->user_id        },
        { TXT_AST,  t->assistant_id   },
        { TXT_THC,  t->think_end_id   },  /* check </think> before <think> */
        { TXT_THO,  t->think_start_id },
        { TXT_DSML, t->dsml_id        },
    };

    for (size_t i = 0; i < sizeof(slots) / sizeof(slots[0]); i++) {
        size_t n = strlen(slots[i].s);
        if (remain >= n && memcmp(p, slots[i].s, n) == 0) {
            *out_id  = slots[i].id;
            *out_len = n;
            return true;
        }
    }
    return false;
}

static int tokenize_with_specials(const struct ds4cuda_tokenizer *t,
                                  const char *text, tok_vec *out)
{
    size_t len = strlen(text);
    size_t span = 0;        /* start of the current non-special run */
    size_t i = 0;
    while (i < len) {
        int id = -1;
        size_t mlen = 0;
        if (match_special_at(t, text + i, len - i, &id, &mlen)) {
            if (i > span) {
                int rc = bpe_tokenize_text(t, text + span, i - span, out);
                if (rc < 0) return rc;
            }
            int rc = tv_push(out, id);
            if (rc < 0) return rc;
            i += mlen;
            span = i;
        } else {
            i++;
        }
    }
    if (i > span) {
        int rc = bpe_tokenize_text(t, text + span, i - span, out);
        if (rc < 0) return rc;
    }
    return 0;
}

int ds4cuda_tokenize(const struct ds4cuda_tokenizer *t,
                     const char *text,
                     int *out_ids, int max_ids)
{
    if (!t) return -EINVAL;
    if (!text) text = "";
    if (max_ids < 0) return -EINVAL;
    if (max_ids > 0 && !out_ids) return -EINVAL;

    tok_vec out = { 0 };
    int rc = tokenize_with_specials(t, text, &out);
    if (rc < 0) { tv_free(&out); return rc; }

    if (out.len > max_ids) {
        tv_free(&out);
        return -E2BIG;
    }
    for (int i = 0; i < out.len; i++) out_ids[i] = out.v[i];
    int n = out.len;
    tv_free(&out);
    return n;
}

int ds4cuda_tokenize_raw(const struct ds4cuda_tokenizer *t,
                         const char *text,
                         int *out_ids, int max_ids)
{
    if (!t) return -EINVAL;
    if (!text) text = "";
    if (max_ids < 0) return -EINVAL;
    if (max_ids > 0 && !out_ids) return -EINVAL;

    tok_vec out = { 0 };
    int rc = bpe_tokenize_text(t, text, strlen(text), &out);
    if (rc < 0) { tv_free(&out); return rc; }

    if (out.len > max_ids) {
        tv_free(&out);
        return -E2BIG;
    }
    for (int i = 0; i < out.len; i++) out_ids[i] = out.v[i];
    int n = out.len;
    tv_free(&out);
    return n;
}

/* ------------------------------------------------------------------ */
/* Detokenize                                                           */
/* ------------------------------------------------------------------ */

/* Helper to test if a token id corresponds to a regular byte-level     */
/* BPE token (any of the seven specials short-circuits to a verbatim    */
/* emit).                                                                */
static int id_is_special(const struct ds4cuda_tokenizer *t, int id)
{
    return id == t->bos_id         || id == t->eos_id         ||
           id == t->user_id        || id == t->assistant_id   ||
           id == t->think_start_id || id == t->think_end_id   ||
           id == t->dsml_id;
}

/* Decode one byte-level BPE token's UTF-8 string back to raw bytes.    */
/* Each codepoint maps back to one byte via cp_to_byte (the inverse of  */
/* gpt2_byte_to_codepoint). Codepoints not in the map cause -EINVAL.    */
/* Writes into (*p..end); on overflow returns -E2BIG.                   */
static int decode_bpe_token(const struct ds4cuda_tokenizer *t, tk_str s,
                            char **p, char *end)
{
    uint64_t off = 0;
    while (off < s.len) {
        uint64_t next = off;
        uint32_t cp = utf8_peek_one(s.ptr, s.len, off, &next);
        if (cp >= 512) return -EINVAL;
        int b = t->cp_to_byte[cp];
        if (b < 0) return -EINVAL;
        if (*p >= end) return -E2BIG;
        **p = (char)(uint8_t)b;
        (*p)++;
        off = next;
    }
    return 0;
}

int ds4cuda_detokenize(const struct ds4cuda_tokenizer *t,
                       const int *ids, int n_ids,
                       char *out_text, size_t max_size)
{
    if (!t) return -EINVAL;
    if (max_size == 0) return -EINVAL;
    if (n_ids > 0 && !ids) return -EINVAL;
    out_text[0] = '\0';

    /* Reserve 1 byte for the trailing NUL.                            */
    char *p   = out_text;
    char *end = out_text + max_size - 1;

    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->n_vocab) {
            out_text[0] = '\0';
            return -EINVAL;
        }
        tk_str s = t->token[id];

        if (id_is_special(t, id)) {
            /* Emit the canonical UTF-8 marker verbatim. The token bytes */
            /* in the GGUF for these ids ARE the marker bytes already.  */
            if ((size_t)(end - p) < s.len) {
                out_text[0] = '\0';
                return -E2BIG;
            }
            memcpy(p, s.ptr, (size_t)s.len);
            p += s.len;
        } else {
            int rc = decode_bpe_token(t, s, &p, end);
            if (rc < 0) {
                out_text[0] = '\0';
                return rc;
            }
        }
    }
    *p = '\0';
    return (int)(p - out_text);
}

/* ------------------------------------------------------------------ */
/* Public accessors.                                                    */
/* ------------------------------------------------------------------ */
int ds4cuda_tokenizer_bos_id          (const struct ds4cuda_tokenizer *t) { return t ? t->bos_id          : -1; }
int ds4cuda_tokenizer_eos_id          (const struct ds4cuda_tokenizer *t) { return t ? t->eos_id          : -1; }
int ds4cuda_tokenizer_user_id         (const struct ds4cuda_tokenizer *t) { return t ? t->user_id         : -1; }
int ds4cuda_tokenizer_assistant_id    (const struct ds4cuda_tokenizer *t) { return t ? t->assistant_id    : -1; }
int ds4cuda_tokenizer_think_open_id   (const struct ds4cuda_tokenizer *t) { return t ? t->think_start_id  : -1; }
int ds4cuda_tokenizer_think_close_id  (const struct ds4cuda_tokenizer *t) { return t ? t->think_end_id    : -1; }
int ds4cuda_tokenizer_dsml_id         (const struct ds4cuda_tokenizer *t) { return t ? t->dsml_id         : -1; }
int ds4cuda_tokenizer_n_vocab         (const struct ds4cuda_tokenizer *t) { return t ? t->n_vocab         : 0; }
