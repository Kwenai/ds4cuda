/*
 * gguf_parser.c — minimal GGUF v3 header / KV / tensor-info parser.
 *
 * - mmaps the file with MAP_PRIVATE | MAP_NORESERVE
 * - reads magic + version + counts
 * - decodes every metadata KV (scalars eagerly, arrays as a pointer +
 *   length to be re-read by caller)
 * - decodes every tensor info (name, n_dims, dims, type, rel_offset)
 * - aligns to general.alignment to compute tensor_data_pos and per-
 *   tensor abs_offset / byte_size
 *
 * Aborts (via ds4_die) on:
 *   - non-GGUF magic
 *   - version != 3
 *   - unsupported quant type on a tensor (the size-by-name table accepts
 *     any GGUF v3 type for accounting, but config_validate_model layer
 *     rejects anything not in DS4_QUANT_*)
 *   - mmap / open / fstat failures
 *   - corrupt / truncated header
 *
 * NOTE: "the file" here is the 81 GB DeepSeek-V4-Flash GGUF. We never
 * touch the tensor data section; only the first ~5 MB (header+KV+
 * tensor info) is read off mmap.
 */
#define _GNU_SOURCE
#include "ds4cuda.h"
#include "ds4cuda_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Diagnostic helpers                                                  */
/* ------------------------------------------------------------------ */

void ds4_die(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "ds4cuda fatal: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}

/* ------------------------------------------------------------------ */
/* GGUF type table — same ordering as ds4.c gguf_types[], extended for */
/* every type we may encounter while parsing the header. Not all types */
/* are supported by ds4cuda backend; unsupported ones are still parsed */
/* to keep abs_offset accounting correct. The validate step rejects.   */
/* ------------------------------------------------------------------ */
struct ds4_gguf_type_info {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
};

static const struct ds4_gguf_type_info GGUF_TYPES[] = {
    [0]  = {"f32",       1,   4},
    [1]  = {"f16",       1,   2},
    [2]  = {"q4_0",     32,  18},
    [3]  = {"q4_1",     32,  20},
    [6]  = {"q5_0",     32,  22},
    [7]  = {"q5_1",     32,  24},
    [8]  = {"q8_0",     32,  34},
    [9]  = {"q8_1",     32,  40},
    [10] = {"q2_k",    256,  84},
    [11] = {"q3_k",    256, 110},
    [12] = {"q4_k",    256, 144},
    [13] = {"q5_k",    256, 176},
    [14] = {"q6_k",    256, 210},
    [15] = {"q8_k",    256, 292},
    [16] = {"iq2_xxs", 256,  66},
    [17] = {"iq2_xs",  256,  74},
    [18] = {"iq3_xxs", 256,  98},
    [19] = {"iq1_s",   256, 110},
    [20] = {"iq4_nl",   32,  18},
    [21] = {"iq3_s",   256, 110},
    [22] = {"iq2_s",   256,  82},
    [23] = {"iq4_xs",  256, 136},
    [24] = {"i8",        1,   1},
    [25] = {"i16",       1,   2},
    [26] = {"i32",       1,   4},
    [27] = {"i64",       1,   8},
    [28] = {"f64",       1,   8},
    [29] = {"iq1_m",   256,  56},
    [30] = {"bf16",      1,   2},
};
#define GGUF_TYPES_LEN ((uint32_t)(sizeof(GGUF_TYPES) / sizeof(GGUF_TYPES[0])))

size_t ds4_quant_block_elems(enum ds4_quant_type t)
{
    if ((uint32_t)t >= GGUF_TYPES_LEN || GGUF_TYPES[t].block_bytes == 0)
        ds4_die("unknown quant type %d", (int)t);
    return GGUF_TYPES[t].block_elems;
}
size_t ds4_quant_block_bytes(enum ds4_quant_type t)
{
    if ((uint32_t)t >= GGUF_TYPES_LEN || GGUF_TYPES[t].block_bytes == 0)
        ds4_die("unknown quant type %d", (int)t);
    return GGUF_TYPES[t].block_bytes;
}
const char *ds4_quant_name(enum ds4_quant_type t)
{
    if ((uint32_t)t < GGUF_TYPES_LEN && GGUF_TYPES[t].block_bytes != 0)
        return GGUF_TYPES[t].name;
    return "<unknown>";
}

/* ------------------------------------------------------------------ */
/* Cursor — bounded read against the mmap region. Out-of-bounds reads  */
/* abort, never silently truncate. All header / KV / tensor info reads */
/* go through this.                                                    */
/* ------------------------------------------------------------------ */
struct cursor {
    const uint8_t *buf;
    size_t pos;
    size_t end;
};

static void cur_init(struct cursor *c, const uint8_t *buf, size_t len)
{
    c->buf = buf;
    c->pos = 0;
    c->end = len;
}

static const uint8_t *cur_take(struct cursor *c, size_t n)
{
    /* Check `n > end - pos` BEFORE adding, to avoid size_t wrap when
     * a hostile / corrupt header advertises a huge length. */
    if (n > (size_t)(c->end - c->pos))
        ds4_die("GGUF truncated: want %zu bytes at pos %zu / %zu",
                n, c->pos, c->end);
    const uint8_t *p = c->buf + c->pos;
    c->pos += n;
    return p;
}

static uint32_t cur_u32(struct cursor *c)
{
    uint32_t v;
    memcpy(&v, cur_take(c, 4), 4);
    return v;
}
static uint64_t cur_u64(struct cursor *c)
{
    uint64_t v;
    memcpy(&v, cur_take(c, 8), 8);
    return v;
}

/* Scalar size for GGUF metadata value type. Returns 0 for non-scalar. */
static uint32_t value_scalar_size(enum ds4_gguf_value_type t)
{
    switch (t) {
    case DS4_GGUF_UINT8:   case DS4_GGUF_INT8:    case DS4_GGUF_BOOL:    return 1;
    case DS4_GGUF_UINT16:  case DS4_GGUF_INT16:                         return 2;
    case DS4_GGUF_UINT32:  case DS4_GGUF_INT32:   case DS4_GGUF_FLOAT32: return 4;
    case DS4_GGUF_UINT64:  case DS4_GGUF_INT64:   case DS4_GGUF_FLOAT64: return 8;
    default: return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Name pool — append-only arena for NUL-terminated names. Caller is   */
/* expected to provision enough up front (we count name byte budget    */
/* during a quick first pass). All const char* in ds4_model point      */
/* into this pool.                                                     */
/* ------------------------------------------------------------------ */
static const char *pool_strndup(struct ds4_model *m, const char *s, size_t n)
{
    /* Guard `used + n + 1` against size_t wrap before the bounds check. */
    if (n > SIZE_MAX - 1 || m->name_pool_used > SIZE_MAX - (n + 1))
        ds4_die("name pool size_t overflow (used=%zu, want=%zu)",
                m->name_pool_used, n);
    if (m->name_pool_used + n + 1 > m->name_pool_size)
        ds4_die("name pool exhausted (used=%zu, want=%zu, cap=%zu)",
                m->name_pool_used, n + 1, m->name_pool_size);
    char *dst = m->name_pool + m->name_pool_used;
    memcpy(dst, s, n);
    dst[n] = '\0';
    m->name_pool_used += n + 1;
    return dst;
}

/* ------------------------------------------------------------------ */
/* Decode one metadata value into kv->v, advancing cur. For arrays we  */
/* record (elem_type, length, raw_ptr) and skip the element bytes      */
/* without materializing.                                              */
/* ------------------------------------------------------------------ */
static void parse_value(struct cursor *c,
                        struct ds4_model *m,
                        enum ds4_gguf_value_type t,
                        struct ds4_kv *kv,
                        int depth);

static void parse_string_into(struct cursor *c,
                              struct ds4_model *m,
                              const char **out)
{
    uint64_t n = cur_u64(c);
    const uint8_t *p = cur_take(c, n);
    *out = pool_strndup(m, (const char *)p, n);
}

/* Skip one value of type t without decoding (used for nested arrays   */
/* and string-array elements when we only need the array span).        */
static void skip_value(struct cursor *c, enum ds4_gguf_value_type t, int depth)
{
    if (depth > 8)
        ds4_die("GGUF nested array depth > 8");
    uint32_t sz = value_scalar_size(t);
    if (sz) {
        cur_take(c, sz);
        return;
    }
    if (t == DS4_GGUF_STRING) {
        uint64_t n = cur_u64(c);
        cur_take(c, n);
        return;
    }
    if (t == DS4_GGUF_ARRAY) {
        uint32_t inner = cur_u32(c);
        uint64_t ln = cur_u64(c);
        uint32_t isz = value_scalar_size((enum ds4_gguf_value_type)inner);
        if (isz) {
            cur_take(c, ln * isz);
        } else {
            for (uint64_t i = 0; i < ln; ++i)
                skip_value(c, (enum ds4_gguf_value_type)inner, depth + 1);
        }
        return;
    }
    ds4_die("unknown GGUF metadata type %u", (unsigned)t);
}

static void parse_value(struct cursor *c,
                        struct ds4_model *m,
                        enum ds4_gguf_value_type t,
                        struct ds4_kv *kv,
                        int depth)
{
    if (depth > 8)
        ds4_die("GGUF nested array depth > 8");
    kv->type = t;
    switch (t) {
    case DS4_GGUF_UINT8:   kv->v.u8  = *cur_take(c, 1); return;
    case DS4_GGUF_INT8:    kv->v.i8  = (int8_t)*cur_take(c, 1); return;
    case DS4_GGUF_BOOL:    kv->v.b   = *cur_take(c, 1); return;
    case DS4_GGUF_UINT16:  memcpy(&kv->v.u16, cur_take(c, 2), 2); return;
    case DS4_GGUF_INT16:   memcpy(&kv->v.i16, cur_take(c, 2), 2); return;
    case DS4_GGUF_UINT32:  kv->v.u32 = cur_u32(c); return;
    case DS4_GGUF_INT32:   { uint32_t u = cur_u32(c); memcpy(&kv->v.i32, &u, 4); return; }
    case DS4_GGUF_FLOAT32: { uint32_t u = cur_u32(c); memcpy(&kv->v.f32, &u, 4); return; }
    case DS4_GGUF_UINT64:  kv->v.u64 = cur_u64(c); return;
    case DS4_GGUF_INT64:   { uint64_t u = cur_u64(c); memcpy(&kv->v.i64, &u, 8); return; }
    case DS4_GGUF_FLOAT64: { uint64_t u = cur_u64(c); memcpy(&kv->v.f64, &u, 8); return; }
    case DS4_GGUF_STRING:  parse_string_into(c, m, &kv->v.s); return;
    case DS4_GGUF_ARRAY: {
        uint32_t inner = cur_u32(c);
        uint64_t ln = cur_u64(c);
        kv->v.arr.elem_type = (enum ds4_gguf_value_type)inner;
        kv->v.arr.length = ln;
        kv->v.arr.raw = c->buf + c->pos;
        uint32_t isz = value_scalar_size((enum ds4_gguf_value_type)inner);
        if (isz) {
            cur_take(c, ln * isz);
        } else {
            for (uint64_t i = 0; i < ln; ++i)
                skip_value(c, (enum ds4_gguf_value_type)inner, depth + 1);
        }
        return;
    }
    default:
        ds4_die("unknown GGUF metadata type %u", (unsigned)t);
    }
}

/* ------------------------------------------------------------------ */
/* Two-pass parse:                                                     */
/*   pass 1: scan to count (n_kv, n_tensors, name byte budget)         */
/*   pass 2: allocate name pool + tensors[] + kv[], fill in            */
/* The first pass walks the same bytes as the second, but it doesn't   */
/* allocate or copy anything — just measures string lengths.           */
/* ------------------------------------------------------------------ */
static size_t pass1_string_bytes(struct cursor *c, int strings_only)
{
    /* Read one string; return its length-on-disk including the +1 NUL */
    /* we will allocate. If strings_only is 0 we don't care.            */
    uint64_t n = cur_u64(c);
    cur_take(c, n);
    return strings_only ? (size_t)n + 1 : 0;
}

static size_t pass1_skip_value(struct cursor *c,
                               enum ds4_gguf_value_type t,
                               int depth,
                               int count_strings)
{
    if (depth > 8)
        ds4_die("GGUF nested array depth > 8");
    size_t name_bytes = 0;
    uint32_t sz = value_scalar_size(t);
    if (sz) { cur_take(c, sz); return 0; }
    if (t == DS4_GGUF_STRING) {
        name_bytes += pass1_string_bytes(c, count_strings);
        return name_bytes;
    }
    if (t == DS4_GGUF_ARRAY) {
        uint32_t inner = cur_u32(c);
        uint64_t ln = cur_u64(c);
        uint32_t isz = value_scalar_size((enum ds4_gguf_value_type)inner);
        if (isz) {
            cur_take(c, ln * isz);
        } else {
            /* don't pool string-array elements (could be 129280 entries);
             * they're addressable via kv->v.arr.raw at parse time. */
            for (uint64_t i = 0; i < ln; ++i)
                pass1_skip_value(c, (enum ds4_gguf_value_type)inner, depth + 1, 0);
        }
        return name_bytes;
    }
    ds4_die("unknown GGUF metadata type %u (pass1)", (unsigned)t);
}

/* Forward decl from model_open.c — config validation runs after parse. */
extern void ds4_config_validate_model(struct ds4_model *m);

int ds4_gguf_parse(struct ds4_model *m)
{
    /* mmap already done by ds4_model_open. */
    if (!m->mmap_ptr || m->file_size < 32)
        ds4_die("GGUF mmap not initialized or file too small");

    /* ---- pass 1: count + measure name pool ---- */
    struct cursor c1;
    cur_init(&c1, (const uint8_t *)m->mmap_ptr, m->file_size);

    uint32_t magic = cur_u32(&c1);
    if (magic != 0x46554747u) /* "GGUF" little-endian */
        ds4_die("not a GGUF file: magic = 0x%08x", magic);
    uint32_t version = cur_u32(&c1);
    if (version != 3)
        ds4_die("only GGUF v3 supported (got v%u)", version);
    uint64_t n_tensors = cur_u64(&c1);
    uint64_t n_kv      = cur_u64(&c1);

    /* default alignment 32; may be overridden by general.alignment */
    m->alignment = 32;
    m->n_tensors = (size_t)n_tensors;
    m->n_kv      = (size_t)n_kv;

    size_t name_pool_budget = 0;

    for (uint64_t i = 0; i < n_kv; ++i) {
        /* key string */
        uint64_t klen = cur_u64(&c1);
        cur_take(&c1, klen);
        name_pool_budget += klen + 1;
        uint32_t t = cur_u32(&c1);
        /* For STRING values we pool them; for ARRAY of STRING we don't. */
        if ((enum ds4_gguf_value_type)t == DS4_GGUF_STRING) {
            name_pool_budget += pass1_skip_value(&c1, (enum ds4_gguf_value_type)t, 0, 1);
        } else {
            pass1_skip_value(&c1, (enum ds4_gguf_value_type)t, 0, 0);
        }
    }

    /* tensor info: name + ndim + dims + type + rel_offset */
    for (uint64_t i = 0; i < n_tensors; ++i) {
        uint64_t nlen = cur_u64(&c1);
        cur_take(&c1, nlen);
        name_pool_budget += nlen + 1;
        uint32_t ndim = cur_u32(&c1);
        if (ndim == 0 || ndim > DS4_MAX_DIMS)
            ds4_die("tensor #%llu: bad ndim %u",
                    (unsigned long long)i, ndim);
        cur_take(&c1, (size_t)ndim * 8); /* dims */
        cur_u32(&c1);                    /* type */
        cur_u64(&c1);                    /* rel_offset */
    }

    /* ---- allocate ---- */
    m->name_pool_size = name_pool_budget + 16;  /* small slack */
    m->name_pool      = (char *)malloc(m->name_pool_size);
    m->name_pool_used = 0;
    m->kv             = (struct ds4_kv *)calloc(m->n_kv, sizeof(*m->kv));
    m->tensors        = (struct ds4_tensor *)calloc(m->n_tensors, sizeof(*m->tensors));
    if (!m->name_pool || (m->n_kv && !m->kv) || (m->n_tensors && !m->tensors))
        ds4_die("calloc failed during GGUF parse setup");

    /* ---- pass 2: decode for real ---- */
    struct cursor c2;
    cur_init(&c2, (const uint8_t *)m->mmap_ptr, m->file_size);
    cur_u32(&c2);                   /* magic */
    cur_u32(&c2);                   /* version */
    (void)cur_u64(&c2);             /* n_tensors */
    (void)cur_u64(&c2);             /* n_kv */

    for (size_t i = 0; i < m->n_kv; ++i) {
        struct ds4_kv *kv = &m->kv[i];
        parse_string_into(&c2, m, &kv->key);
        uint32_t t = cur_u32(&c2);
        parse_value(&c2, m, (enum ds4_gguf_value_type)t, kv, 0);

        if (strcmp(kv->key, "general.alignment") == 0 &&
            kv->type == DS4_GGUF_UINT32) {
            if (kv->v.u32 == 0)
                ds4_die("general.alignment = 0");
            m->alignment = kv->v.u32;
        }
    }

    /* Capture file offset of the first tensor info record (= end of KV
     * section). The offline repack tool copies bytes [24, kv_byte_range_end)
     * verbatim from the input to the output. */
    m->kv_byte_range_end = (uint64_t)c2.pos;

    for (size_t i = 0; i < m->n_tensors; ++i) {
        struct ds4_tensor *t = &m->tensors[i];
        parse_string_into(&c2, m, &t->name);
        t->n_dims = cur_u32(&c2);
        if (t->n_dims == 0 || t->n_dims > DS4_MAX_DIMS)
            ds4_die("tensor '%s': bad ndim %u", t->name, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; ++d)
            t->dims[d] = cur_u64(&c2);
        for (uint32_t d = t->n_dims; d < DS4_MAX_DIMS; ++d)
            t->dims[d] = 1;
        uint32_t qt = cur_u32(&c2);
        if (qt >= GGUF_TYPES_LEN || GGUF_TYPES[qt].block_bytes == 0)
            ds4_die("tensor '%s': unknown quant type %u", t->name, qt);
        t->quant = (enum ds4_quant_type)qt;
        /* rel_offset gets resolved to abs_offset below. */
        t->abs_offset = cur_u64(&c2);   /* temporarily store rel_offset */
    }

    /* tensor_data_pos = align_up(end of tensor info section, alignment). */
    size_t info_end = c2.pos;
    /* Capture for the offline repack tool: end of last tensor info record,
     * i.e. start of the alignment-padding zeros before the data blob. */
    m->tensor_info_byte_range_end = (uint64_t)info_end;
    /* Guard `info_end + alignment - 1` against size_t wrap. m->alignment was
     * already rejected at zero in the KV scan above. */
    if (m->alignment == 0 || info_end > SIZE_MAX - (m->alignment - 1))
        ds4_die("alignment overflow: info_end=%zu alignment=%u",
                info_end, m->alignment);
    uint64_t aligned = (info_end + m->alignment - 1) /
                       m->alignment * m->alignment;
    m->tensor_data_pos = aligned;

    /* compute abs_offset and byte_size for each tensor. */
    for (size_t i = 0; i < m->n_tensors; ++i) {
        struct ds4_tensor *t = &m->tensors[i];
        uint64_t rel = t->abs_offset;
        /* abs_offset = tensor_data_pos + rel, guard against u64 wrap. */
        if (rel > UINT64_MAX - m->tensor_data_pos)
            ds4_die("tensor '%s': rel_offset %llu wraps tensor_data_pos %llu",
                    t->name, (unsigned long long)rel,
                    (unsigned long long)m->tensor_data_pos);
        t->abs_offset = m->tensor_data_pos + rel;

        /* elems = product(dims), guard each multiplication against u64 wrap. */
        uint64_t elems = 1;
        for (uint32_t d = 0; d < t->n_dims; ++d) {
            uint64_t dim = t->dims[d];
            if (dim != 0 && elems > UINT64_MAX / dim)
                ds4_die("tensor '%s': dim product overflow at dim[%u]=%llu",
                        t->name, d, (unsigned long long)dim);
            elems *= dim;
        }

        size_t be = GGUF_TYPES[t->quant].block_elems;
        size_t bb = GGUF_TYPES[t->quant].block_bytes;
        if (elems == 0 || be == 0) {
            t->byte_size = 0;
        } else {
            /* blocks = ceil(elems / be); guard the +be-1 against u64 wrap. */
            if (elems > UINT64_MAX - (uint64_t)(be - 1))
                ds4_die("tensor '%s': blocks ceil overflow (elems=%llu be=%zu)",
                        t->name, (unsigned long long)elems, be);
            uint64_t blocks = (elems + be - 1) / be;
            /* byte_size = blocks * bb; guard u64 multiply. */
            if (bb != 0 && blocks > UINT64_MAX / (uint64_t)bb)
                ds4_die("tensor '%s': byte_size overflow (blocks=%llu bb=%zu)",
                        t->name, (unsigned long long)blocks, bb);
            t->byte_size = blocks * (uint64_t)bb;
        }

        /* file-bounds check: abs_offset + byte_size <= file_size, sans wrap. */
        if (t->byte_size > (uint64_t)m->file_size ||
            t->abs_offset > (uint64_t)m->file_size - t->byte_size)
            ds4_die("tensor '%s': abs_offset %llu + bytes %llu > file size %zu",
                    t->name,
                    (unsigned long long)t->abs_offset,
                    (unsigned long long)t->byte_size,
                    m->file_size);
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Lookup helpers                                                      */
/* ------------------------------------------------------------------ */
const struct ds4_tensor *ds4_model_find_tensor(const struct ds4_model *m,
                                               const char *name)
{
    for (size_t i = 0; i < m->n_tensors; ++i) {
        if (strcmp(m->tensors[i].name, name) == 0)
            return &m->tensors[i];
    }
    return NULL;
}

const struct ds4_kv *ds4_model_find_kv(const struct ds4_model *m,
                                       const char *key)
{
    for (size_t i = 0; i < m->n_kv; ++i) {
        if (strcmp(m->kv[i].key, key) == 0)
            return &m->kv[i];
    }
    return NULL;
}
