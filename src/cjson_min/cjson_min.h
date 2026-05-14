/*
 * cjson_min.h — minimal JSON parser + emitter for the ds4cuda HTTP server.
 *
 * Clean-room reimplementation: no cJSON source code is reused, only the
 * API surface (function names, node-ownership model) is borrowed from
 * Dave Gamble's well-known cJSON. Released under the same GPL-2.0 as the
 * rest of ds4cuda; see LICENSE in repo root.
 *
 * Subset of the cJSON API: nodes own their children + values; build / parse
 * / emit / free. Sufficient for OpenAI /v1/chat/completions and Anthropic
 * /v1/messages request and response shapes; not a full RFC 8259
 * implementation:
 *
 *   - Strings are UTF-8 byte-transparent. The parser decodes \\uXXXX
 *     surrogate pairs; the emitter only escapes the seven mandatory escapes
 *     (" \\ / \b \f \n \r \t) and control chars < 0x20 — output bytes >= 0x20
 *     pass through untouched (UTF-8-safe in practice).
 *   - Numbers are stored as double. INT_MAX-ish ints round-trip exactly.
 *   - Arrays / objects are stored as singly-linked sibling lists keyed off a
 *     parent's `child` pointer. Linear lookup; fine for /v1/chat/completions
 *     where each layer is small.
 *   - No printbuffer pretty-print. Compact output only.
 *
 * Memory: the parser uses malloc-returning-NULL so an OOM on an untrusted
 * request body shows up as a regular parse error (NULL).  The build /
 * emit path (cjson_min_new_*, cjson_min_emit) still aborts on OOM —
 * response emission has no recovery path and callers assume non-NULL
 * builders.  The string value of a JSON string is owned by the node;
 * cjson_min_free frees the whole tree depth-first.
 *
 * Parser recursion is bounded (see CJSON_MIN_MAX_DEPTH in cjson_min.c)
 * so pathological deeply-nested input cannot blow the C stack.
 *
 * Not thread-safe (no globals other than const error pointers).
 */
#ifndef CJSON_MIN_H
#define CJSON_MIN_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cjson_type {
    CJSON_INVALID = 0,
    CJSON_NULL,
    CJSON_FALSE,
    CJSON_TRUE,
    CJSON_NUMBER,
    CJSON_STRING,
    CJSON_ARRAY,
    CJSON_OBJECT,
};

typedef struct cjson {
    struct cjson *next;
    struct cjson *prev;
    struct cjson *child;       /* first child (array element / object key) */
    enum cjson_type type;
    double valuedouble;        /* CJSON_NUMBER */
    char *valuestring;         /* CJSON_STRING */
    char *string;              /* object key (NULL for array elements / root) */
} cjson;

/* Parse a NUL-terminated UTF-8 buffer. Returns NULL on syntax error.
 * If err_pos is non-NULL, on failure it is set to a (caller-readable) point
 * inside the input near the failure. */
cjson *cjson_min_parse(const char *json);
cjson *cjson_min_parse_with_len(const char *json, size_t len);

/* Free the whole tree (item itself + descendants). NULL-safe. */
void cjson_min_free(cjson *item);

/* Look up object child by name (case-sensitive). Returns NULL when key
 * absent or `obj` is not an object. */
cjson *cjson_min_obj_get(const cjson *obj, const char *name);

/* Array helpers. */
int    cjson_min_array_size(const cjson *arr);
cjson *cjson_min_array_item(const cjson *arr, int index);

/* Type predicates / value accessors with safe fallback. */
bool        cjson_min_is_string(const cjson *n);
bool        cjson_min_is_number(const cjson *n);
bool        cjson_min_is_array (const cjson *n);
bool        cjson_min_is_object(const cjson *n);
bool        cjson_min_is_bool  (const cjson *n);
bool        cjson_min_is_null  (const cjson *n);
const char *cjson_min_string   (const cjson *n);
double      cjson_min_number   (const cjson *n);
bool        cjson_min_bool     (const cjson *n);

/* Build helpers (used by the response emitter). All of these abort on
 * OOM and never return NULL — server response emission has no recovery
 * path so the abort is intentional. */
cjson *cjson_min_new_null  (void);
cjson *cjson_min_new_bool  (bool v);
cjson *cjson_min_new_number(double v);
cjson *cjson_min_new_string(const char *s);   /* strdups s; s may be NULL */
cjson *cjson_min_new_array (void);
cjson *cjson_min_new_object(void);

/* Append a child. The container takes ownership; on return `item` is owned
 * by `container`. For objects, `key` is strdup'd and assigned to item->string. */
void cjson_min_array_push(cjson *array, cjson *item);
void cjson_min_object_set(cjson *object, const char *key, cjson *item);

/* Compact JSON serialization. Caller frees with free(). */
char *cjson_min_emit(const cjson *item);

#ifdef __cplusplus
}
#endif

#endif /* CJSON_MIN_H */
