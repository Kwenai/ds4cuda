/*
 * cjson_min.c — minimal JSON parser + emitter, see cjson_min.h.
 *
 * Implementation is a recursive descent parser modeled after antirez/ds4
 * `ds4_server.c` json_* helpers, packaged as an explicit tree so the
 * OpenAI/Anthropic endpoint glue can keep request/response handling at
 * the structural level (lookups, iteration) rather than a stream of
 * positional callbacks.
 *
 * Memory discipline:
 *   - The parse path (untrusted request bodies) uses malloc-returning-NULL
 *     primitives and propagates the failure as a regular parse error
 *     (NULL from cjson_min_parse*).  A deeply-nested or oversized JSON
 *     body must never abort() the server.
 *   - The build / emit path (server-generated response bodies) still aborts
 *     on OOM: a server unable to allocate a response object has no path to
 *     recover anyway, and callers in src/server/ assume non-NULL builders.
 *   - Parser recursion is bounded by CJSON_MIN_MAX_DEPTH so a pathological
 *     16 MiB body of "[[[[..." cannot blow the stack.
 */
#define _GNU_SOURCE
#include "cjson_min.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum parser/free/emit recursion depth.  Anything beyond this is
 * rejected with a parse error rather than recursed into.  64 is well
 * above any realistic chat-completion / messages payload (typical nesting
 * is 3-6 levels) while keeping the C stack usage bounded. */
#define CJSON_MIN_MAX_DEPTH 64

/* ------------------------------------------------------------------ */
/* Allocator helpers.                                                  */
/*                                                                     */
/* cjson_xmalloc / cjson_xrealloc / cjson_xstrdup return NULL on OOM   */
/* and are used by the parse path.  The build path (cjson_min_new_*,   */
/* cjson_min_emit) calls the abort_-suffixed variants below; see the   */
/* file-level comment for rationale.                                   */
/* ------------------------------------------------------------------ */
static void *cjson_xmalloc(size_t n)
{
    if (!n) n = 1;
    return malloc(n);
}

static void *cjson_xrealloc(void *p, size_t n)
{
    if (!n) n = 1;
    return realloc(p, n);
}

static char *cjson_xstrdup(const char *s)
{
    if (!s) s = "";
    size_t n = strlen(s);
    char *out = cjson_xmalloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

/* Abort-on-OOM allocators for the build / emit path.  Server response
 * emission has no recovery path so failing fast is preferable to
 * silently returning NULL into code that assumes non-NULL builders. */
static void *xmalloc_(size_t n)
{
    void *p = cjson_xmalloc(n);
    if (!p) {
        fprintf(stderr, "cjson_min: out of memory (alloc %zu B)\n", n);
        abort();
    }
    return p;
}

static void *xrealloc_(void *p, size_t n)
{
    void *q = cjson_xrealloc(p, n);
    if (!q) {
        fprintf(stderr, "cjson_min: out of memory (realloc %zu B)\n", n);
        abort();
    }
    return q;
}

static char *xstrdup_(const char *s)
{
    char *out = cjson_xstrdup(s);
    if (!out) {
        fprintf(stderr, "cjson_min: out of memory (strdup)\n");
        abort();
    }
    return out;
}

/* ------------------------------------------------------------------ */
/* Tiny dynamic string buffer for parser temporaries + emitter output. */
/*                                                                     */
/* The emitter calls sbuf_putc / sbuf_puts / sbuf_printf directly, all */
/* of which abort on OOM (xrealloc_).  The parser uses the _safe       */
/* variants which return 0 on success and -1 on OOM so the failure can */
/* be propagated as a regular parse error.                             */
/* ------------------------------------------------------------------ */
typedef struct {
    char *p;
    size_t len;
    size_t cap;
} sbuf;

static int sbuf_putc_safe(sbuf *b, char c)
{
    if (b->len + 1 >= b->cap) {
        size_t newcap = b->cap ? b->cap * 2 : 64;
        char *np = cjson_xrealloc(b->p, newcap);
        if (!np) return -1;
        b->p = np;
        b->cap = newcap;
    }
    b->p[b->len++] = c;
    b->p[b->len] = '\0';
    return 0;
}

static int sbuf_putn_safe(sbuf *b, const char *s, size_t n)
{
    while (n--) {
        if (sbuf_putc_safe(b, *s++) != 0) return -1;
    }
    return 0;
}

/* Emitter-side (abort on OOM). */
static void sbuf_putc(sbuf *b, char c)
{
    if (b->len + 1 >= b->cap) {
        b->cap = b->cap ? b->cap * 2 : 64;
        b->p = xrealloc_(b->p, b->cap);
    }
    b->p[b->len++] = c;
    b->p[b->len] = '\0';
}

static void sbuf_puts(sbuf *b, const char *s)
{
    if (!s) return;
    while (*s) sbuf_putc(b, *s++);
}

static void sbuf_putn(sbuf *b, const char *s, size_t n)
{
    while (n--) sbuf_putc(b, *s++);
}

static void sbuf_printf(sbuf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void sbuf_printf(sbuf *b, const char *fmt, ...)
{
    char tmp[64];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) {
        sbuf_putn(b, tmp, (size_t)n);
        return;
    }
    char *big = xmalloc_((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sbuf_putn(b, big, (size_t)n);
    free(big);
}

/* ------------------------------------------------------------------ */
/* Node lifetime.                                                      */
/* ------------------------------------------------------------------ */

/* Parser-side: return NULL on OOM. */
static cjson *node_new_safe(enum cjson_type t)
{
    cjson *n = cjson_xmalloc(sizeof(*n));
    if (!n) return NULL;
    memset(n, 0, sizeof(*n));
    n->type = t;
    return n;
}

/* Builder-side: abort on OOM (server response emission is unrecoverable). */
static cjson *node_new(enum cjson_type t)
{
    cjson *n = xmalloc_(sizeof(*n));
    memset(n, 0, sizeof(*n));
    n->type = t;
    return n;
}

/* Iterative free of a sibling list, bounded by CJSON_MIN_MAX_DEPTH for
 * nested children.  A parse-tree that exceeds MAX_DEPTH cannot exist
 * because the parser refused to build it, so callers will not hit the
 * abort below in practice; we keep it as a defensive guard. */
static void cjson_min_free_depth(cjson *n, int depth)
{
    if (depth > CJSON_MIN_MAX_DEPTH) {
        /* Should be unreachable: parser caps depth at the same value.
         * Aborting is preferable to silent leaks of a runaway tree. */
        fprintf(stderr, "cjson_min: free recursion depth exceeded\n");
        abort();
    }
    while (n) {
        cjson *next = n->next;
        if (n->child) cjson_min_free_depth(n->child, depth + 1);
        free(n->valuestring);
        free(n->string);
        free(n);
        n = next;
    }
}

void cjson_min_free(cjson *n)
{
    cjson_min_free_depth(n, 0);
}

/* ------------------------------------------------------------------ */
/* Parser.                                                             */
/* ------------------------------------------------------------------ */
static void skip_ws(const char **p, const char *end)
{
    while (*p < end) {
        unsigned char c = (unsigned char)**p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') (*p)++;
        else break;
    }
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_emit(sbuf *b, uint32_t cp)
{
    if (cp <= 0x7f) {
        sbuf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        sbuf_putc(b, (char)(0xc0 | (cp >> 6)));
        sbuf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        sbuf_putc(b, (char)(0xe0 | (cp >> 12)));
        sbuf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        sbuf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        sbuf_putc(b, (char)(0xf0 | (cp >> 18)));
        sbuf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        sbuf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        sbuf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

/* OOM-safe UTF-8 emission for the parser side.  Returns 0 / -1. */
static int utf8_emit_safe(sbuf *b, uint32_t cp)
{
    if (cp <= 0x7f) {
        return sbuf_putc_safe(b, (char)cp);
    } else if (cp <= 0x7ff) {
        if (sbuf_putc_safe(b, (char)(0xc0 | (cp >> 6))) != 0) return -1;
        return sbuf_putc_safe(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        if (sbuf_putc_safe(b, (char)(0xe0 | (cp >> 12))) != 0) return -1;
        if (sbuf_putc_safe(b, (char)(0x80 | ((cp >> 6) & 0x3f))) != 0) return -1;
        return sbuf_putc_safe(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        if (sbuf_putc_safe(b, (char)(0xf0 | (cp >> 18))) != 0) return -1;
        if (sbuf_putc_safe(b, (char)(0x80 | ((cp >> 12) & 0x3f))) != 0) return -1;
        if (sbuf_putc_safe(b, (char)(0x80 | ((cp >> 6) & 0x3f))) != 0) return -1;
        return sbuf_putc_safe(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool parse_u4(const char **p, const char *end, uint32_t *out)
{
    if (*p + 4 > end) return false;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        int d = hex_digit((*p)[i]);
        if (d < 0) return false;
        v = (v << 4) | (uint32_t)d;
    }
    *p += 4;
    *out = v;
    return true;
}

/* Forward decls. */
static cjson *parse_value(const char **p, const char *end, int depth);

/* parse_string does not recurse so it does not need a depth parameter,
 * but it allocates and can fail with OOM — same NULL-return contract as
 * the rest of the parser. */
static cjson *parse_string(const char **p, const char *end)
{
    if (*p >= end || **p != '"') return NULL;
    (*p)++;
    sbuf b = {0};
    while (*p < end && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') {
            if (sbuf_putc_safe(&b, (char)c) != 0) goto fail;
            continue;
        }
        if (*p >= end) goto fail;
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"':  if (sbuf_putc_safe(&b, '"')  != 0) goto fail; break;
        case '\\': if (sbuf_putc_safe(&b, '\\') != 0) goto fail; break;
        case '/':  if (sbuf_putc_safe(&b, '/')  != 0) goto fail; break;
        case 'b':  if (sbuf_putc_safe(&b, '\b') != 0) goto fail; break;
        case 'f':  if (sbuf_putc_safe(&b, '\f') != 0) goto fail; break;
        case 'n':  if (sbuf_putc_safe(&b, '\n') != 0) goto fail; break;
        case 'r':  if (sbuf_putc_safe(&b, '\r') != 0) goto fail; break;
        case 't':  if (sbuf_putc_safe(&b, '\t') != 0) goto fail; break;
        case 'u': {
            uint32_t cp = 0;
            if (!parse_u4(p, end, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                /* high surrogate; expect "\uXXXX" low surrogate */
                if (*p + 2 > end || (*p)[0] != '\\' || (*p)[1] != 'u') goto fail;
                *p += 2;
                uint32_t lo = 0;
                if (!parse_u4(p, end, &lo)) goto fail;
                if (lo < 0xdc00 || lo > 0xdfff) goto fail;
                cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
            }
            if (utf8_emit_safe(&b, cp) != 0) goto fail;
            break;
        }
        default:
            goto fail;
        }
    }
    if (*p >= end || **p != '"') goto fail;
    (*p)++;

    cjson *n = node_new_safe(CJSON_STRING);
    if (!n) goto fail;
    if (b.p) {
        n->valuestring = b.p;
    } else {
        n->valuestring = cjson_xstrdup("");
        if (!n->valuestring) { free(n); goto fail; }
    }
    return n;
fail:
    free(b.p);
    return NULL;
}

static cjson *parse_number(const char **p, const char *end)
{
    if (*p >= end) return NULL;
    /* strtod needs NUL terminated buffer. The input may not be — but
     * cjson_min_parse_with_len always pre-copies into a NUL-terminated
     * scratch and cjson_min_parse passes a strlen-ed pointer + end, so
     * by the time we reach here `*end == '\0'`. */
    char *endp = NULL;
    double v = strtod(*p, &endp);
    if (!endp || endp == *p) return NULL;
    if ((const char *)endp > end) return NULL;
    *p = endp;
    cjson *n = node_new_safe(CJSON_NUMBER);
    if (!n) return NULL;
    n->valuedouble = v;
    return n;
}

static bool eat_lit(const char **p, const char *end, const char *lit)
{
    size_t n = strlen(lit);
    if ((size_t)(end - *p) < n) return false;
    if (memcmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static cjson *parse_array(const char **p, const char *end, int depth)
{
    if (depth > CJSON_MIN_MAX_DEPTH) return NULL;   /* max depth exceeded */
    if (*p >= end || **p != '[') return NULL;
    (*p)++;
    cjson *arr = node_new_safe(CJSON_ARRAY);
    if (!arr) return NULL;
    cjson *tail = NULL;
    skip_ws(p, end);
    if (*p < end && **p == ']') { (*p)++; return arr; }
    for (;;) {
        cjson *item = parse_value(p, end, depth + 1);
        if (!item) { cjson_min_free(arr); return NULL; }
        if (!arr->child) {
            arr->child = item;
            tail = item;
        } else {
            tail->next = item;
            item->prev = tail;
            tail = item;
        }
        skip_ws(p, end);
        if (*p < end && **p == ',') { (*p)++; skip_ws(p, end); continue; }
        if (*p < end && **p == ']') { (*p)++; return arr; }
        cjson_min_free(arr);
        return NULL;
    }
}

static cjson *parse_object(const char **p, const char *end, int depth)
{
    if (depth > CJSON_MIN_MAX_DEPTH) return NULL;   /* max depth exceeded */
    if (*p >= end || **p != '{') return NULL;
    (*p)++;
    cjson *obj = node_new_safe(CJSON_OBJECT);
    if (!obj) return NULL;
    cjson *tail = NULL;
    skip_ws(p, end);
    if (*p < end && **p == '}') { (*p)++; return obj; }
    for (;;) {
        skip_ws(p, end);
        cjson *key = parse_string(p, end);
        if (!key) { cjson_min_free(obj); return NULL; }
        char *kstr = key->valuestring;
        key->valuestring = NULL;
        cjson_min_free(key);
        skip_ws(p, end);
        if (*p >= end || **p != ':') {
            free(kstr);
            cjson_min_free(obj);
            return NULL;
        }
        (*p)++;
        cjson *val = parse_value(p, end, depth + 1);
        if (!val) {
            free(kstr);
            cjson_min_free(obj);
            return NULL;
        }
        val->string = kstr;
        if (!obj->child) {
            obj->child = val;
            tail = val;
        } else {
            tail->next = val;
            val->prev = tail;
            tail = val;
        }
        skip_ws(p, end);
        if (*p < end && **p == ',') { (*p)++; continue; }
        if (*p < end && **p == '}') { (*p)++; return obj; }
        cjson_min_free(obj);
        return NULL;
    }
}

static cjson *parse_value(const char **p, const char *end, int depth)
{
    if (depth > CJSON_MIN_MAX_DEPTH) return NULL;   /* max depth exceeded */
    skip_ws(p, end);
    if (*p >= end) return NULL;
    char c = **p;
    if (c == '"') return parse_string(p, end);
    if (c == '{') return parse_object(p, end, depth);
    if (c == '[') return parse_array(p, end, depth);
    if (eat_lit(p, end, "null"))  return node_new_safe(CJSON_NULL);
    if (eat_lit(p, end, "true"))  return node_new_safe(CJSON_TRUE);
    if (eat_lit(p, end, "false")) return node_new_safe(CJSON_FALSE);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(p, end);
    return NULL;
}

cjson *cjson_min_parse(const char *json)
{
    if (!json) return NULL;
    return cjson_min_parse_with_len(json, strlen(json));
}

cjson *cjson_min_parse_with_len(const char *json, size_t len)
{
    if (!json) return NULL;
    /* Copy to a NUL-terminated scratch so strtod has a sentinel.
     * OOM here is reported as a parse error (NULL) — must never abort
     * on an untrusted request body. */
    char *buf = cjson_xmalloc(len + 1);
    if (!buf) return NULL;
    memcpy(buf, json, len);
    buf[len] = '\0';
    const char *p = buf;
    const char *end = buf + len;
    cjson *n = parse_value(&p, end, 0);
    if (n) {
        skip_ws(&p, end);
        /* trailing junk -> reject */
        if (p != end) {
            cjson_min_free(n);
            n = NULL;
        }
    }
    free(buf);
    return n;
}

/* ------------------------------------------------------------------ */
/* Accessors.                                                          */
/* ------------------------------------------------------------------ */
bool cjson_min_is_string(const cjson *n) { return n && n->type == CJSON_STRING; }
bool cjson_min_is_number(const cjson *n) { return n && n->type == CJSON_NUMBER; }
bool cjson_min_is_array (const cjson *n) { return n && n->type == CJSON_ARRAY;  }
bool cjson_min_is_object(const cjson *n) { return n && n->type == CJSON_OBJECT; }
bool cjson_min_is_bool  (const cjson *n) { return n && (n->type == CJSON_TRUE || n->type == CJSON_FALSE); }
bool cjson_min_is_null  (const cjson *n) { return n && n->type == CJSON_NULL;   }

const char *cjson_min_string(const cjson *n)
{
    return cjson_min_is_string(n) ? n->valuestring : NULL;
}

double cjson_min_number(const cjson *n)
{
    return cjson_min_is_number(n) ? n->valuedouble : 0.0;
}

bool cjson_min_bool(const cjson *n)
{
    return n && n->type == CJSON_TRUE;
}

cjson *cjson_min_obj_get(const cjson *obj, const char *name)
{
    if (!obj || obj->type != CJSON_OBJECT || !name) return NULL;
    for (cjson *c = obj->child; c; c = c->next) {
        if (c->string && !strcmp(c->string, name)) return c;
    }
    return NULL;
}

int cjson_min_array_size(const cjson *arr)
{
    if (!arr || arr->type != CJSON_ARRAY) return 0;
    int n = 0;
    for (cjson *c = arr->child; c; c = c->next) n++;
    return n;
}

cjson *cjson_min_array_item(const cjson *arr, int index)
{
    if (!arr || arr->type != CJSON_ARRAY || index < 0) return NULL;
    cjson *c = arr->child;
    while (c && index--) c = c->next;
    return c;
}

/* ------------------------------------------------------------------ */
/* Builders.                                                           */
/* ------------------------------------------------------------------ */
cjson *cjson_min_new_null  (void)              { return node_new(CJSON_NULL); }
cjson *cjson_min_new_bool  (bool v)            { return node_new(v ? CJSON_TRUE : CJSON_FALSE); }
cjson *cjson_min_new_number(double v)
{
    cjson *n = node_new(CJSON_NUMBER);
    n->valuedouble = v;
    return n;
}
cjson *cjson_min_new_string(const char *s)
{
    cjson *n = node_new(CJSON_STRING);
    n->valuestring = xstrdup_(s ? s : "");
    return n;
}
cjson *cjson_min_new_array (void)              { return node_new(CJSON_ARRAY); }
cjson *cjson_min_new_object(void)              { return node_new(CJSON_OBJECT); }

void cjson_min_array_push(cjson *array, cjson *item)
{
    if (!array || !item || array->type != CJSON_ARRAY) return;
    if (!array->child) {
        array->child = item;
        return;
    }
    cjson *tail = array->child;
    while (tail->next) tail = tail->next;
    tail->next = item;
    item->prev = tail;
}

void cjson_min_object_set(cjson *object, const char *key, cjson *item)
{
    if (!object || !item || object->type != CJSON_OBJECT) return;
    free(item->string);
    item->string = xstrdup_(key ? key : "");
    if (!object->child) {
        object->child = item;
        return;
    }
    cjson *tail = object->child;
    while (tail->next) tail = tail->next;
    tail->next = item;
    item->prev = tail;
}

/* ------------------------------------------------------------------ */
/* Emitter.                                                            */
/* ------------------------------------------------------------------ */
static void emit_string(sbuf *b, const char *s)
{
    sbuf_putc(b, '"');
    for (const char *p = s ? s : ""; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
        case '"':  sbuf_puts(b, "\\\""); break;
        case '\\': sbuf_puts(b, "\\\\"); break;
        case '\b': sbuf_puts(b, "\\b");  break;
        case '\f': sbuf_puts(b, "\\f");  break;
        case '\n': sbuf_puts(b, "\\n");  break;
        case '\r': sbuf_puts(b, "\\r");  break;
        case '\t': sbuf_puts(b, "\\t");  break;
        default:
            if (c < 0x20) sbuf_printf(b, "\\u%04x", c);
            else          sbuf_putc(b, (char)c);
        }
    }
    sbuf_putc(b, '"');
}

static void emit_number(sbuf *b, double v)
{
    /* If integer-valued in 53-bit range, emit without trailing .0 */
    if (v >= -9.0e15 && v <= 9.0e15) {
        long long iv = (long long)v;
        if ((double)iv == v) { sbuf_printf(b, "%lld", iv); return; }
    }
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "%.17g", v);
    if (n < 0 || (size_t)n >= sizeof(tmp)) snprintf(tmp, sizeof(tmp), "%g", v);
    sbuf_puts(b, tmp);
}

static void emit_value(sbuf *b, const cjson *n, int depth)
{
    if (depth > CJSON_MIN_MAX_DEPTH) {
        /* The parser refuses to build trees this deep, so any tree we are
         * emitting was constructed by the builder API.  Server-side emit
         * has no recovery path; refuse to recurse and emit "null" so the
         * output stays valid JSON. */
        sbuf_puts(b, "null");
        return;
    }
    if (!n) { sbuf_puts(b, "null"); return; }
    switch (n->type) {
    case CJSON_NULL:   sbuf_puts(b, "null");  return;
    case CJSON_TRUE:   sbuf_puts(b, "true");  return;
    case CJSON_FALSE:  sbuf_puts(b, "false"); return;
    case CJSON_NUMBER: emit_number(b, n->valuedouble); return;
    case CJSON_STRING: emit_string(b, n->valuestring); return;
    case CJSON_ARRAY:
        sbuf_putc(b, '[');
        for (cjson *c = n->child; c; c = c->next) {
            emit_value(b, c, depth + 1);
            if (c->next) sbuf_putc(b, ',');
        }
        sbuf_putc(b, ']');
        return;
    case CJSON_OBJECT:
        sbuf_putc(b, '{');
        for (cjson *c = n->child; c; c = c->next) {
            emit_string(b, c->string ? c->string : "");
            sbuf_putc(b, ':');
            emit_value(b, c, depth + 1);
            if (c->next) sbuf_putc(b, ',');
        }
        sbuf_putc(b, '}');
        return;
    default:
        sbuf_puts(b, "null");
        return;
    }
}

char *cjson_min_emit(const cjson *item)
{
    sbuf b = {0};
    emit_value(&b, item, 0);
    if (!b.p) return xstrdup_("null");
    return b.p;
}
