/*
 * chat_template.c — render OpenAI messages into a DSML prompt string.
 *
 * Mirrors ds4_encode_chat_prompt (ds4/ds4.c:14549) + ds4_chat_append_message
 * (ds4.c:14562) at the *text* level. ds4 does the same templating but
 * directly into a token stream, since it owns the BPE tokenizer. We render
 * to text + special-token markers so the integrated build can re-tokenize via
 * the existing ds4_tokenize_rendered_chat (which knows the seven special
 * tokens by their UTF-8 byte sequences).
 *
 * The special-token byte sequences are checked into the binary as static
 * UTF-8 strings; they are NOT generated from cite of any header file, so
 * the chat template module stands alone with zero CUDA / GGUF dependency.
 */
#define _GNU_SOURCE
#include "chat_template.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cite ds4.c:14444-14450:
 *   bos_id       = "<｜begin▁of▁sentence｜>"
 *   user_id      = "<｜User｜>"
 *   assistant_id = "<｜Assistant｜>"
 *   think_start  = "<think>"
 *   think_end    = "</think>"
 *
 * The wide-bar character is U+FF5C (｜); the narrow underscore-arrow style
 * '▁' separator is U+2581. UTF-8 bytes are spelled out here so cross-
 * editor builds (no "fancy unicode glyph at the source level") are safe.
 */
/* Each \x9c separated from the following alphanumeric to avoid GCC's greedy
 * hex-escape parsing eating it as part of the escape sequence. */
const char *const DS4CUDA_TOK_BOS         = "<\xef\xbd\x9c" "begin\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>";
const char *const DS4CUDA_TOK_USER        = "<\xef\xbd\x9c" "User\xef\xbd\x9c>";
const char *const DS4CUDA_TOK_ASSISTANT   = "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>";
const char *const DS4CUDA_TOK_THINK_OPEN  = "<think>";
const char *const DS4CUDA_TOK_THINK_CLOSE = "</think>";

/* ------------------------------------------------------------------ */
/* Bounded printf into a caller-supplied buffer. Returns false if any    */
/* fragment would overflow.                                              */
/* ------------------------------------------------------------------ */
typedef struct {
    char *p;
    size_t cap;     /* total bytes including space for trailing NUL */
    size_t len;     /* bytes written so far, NOT counting NUL */
    bool overflow;
} cb;

static void cb_init(cb *c, char *p, size_t cap)
{
    c->p = p;
    c->cap = cap;
    c->len = 0;
    c->overflow = false;
    if (cap > 0) p[0] = '\0';
}

static void cb_puts(cb *c, const char *s)
{
    if (c->overflow || !s) return;
    size_t n = strlen(s);
    if (c->len + n + 1 > c->cap) { c->overflow = true; return; }
    memcpy(c->p + c->len, s, n);
    c->len += n;
    c->p[c->len] = '\0';
}

/* ------------------------------------------------------------------ */
/* Role classification.                                                */
/* ------------------------------------------------------------------ */
typedef enum {
    R_SYSTEM,
    R_USER,
    R_ASSISTANT,
    R_TOOL,
} role_kind;

static role_kind classify_role(const char *role)
{
    if (!role) return R_USER;
    if (!strcmp(role, "system") || !strcmp(role, "developer")) return R_SYSTEM;
    if (!strcmp(role, "assistant")) return R_ASSISTANT;
    if (!strcmp(role, "tool") || !strcmp(role, "function")) return R_TOOL;
    return R_USER;
}

/* ------------------------------------------------------------------ */
/* Render messages.                                                    */
/* ------------------------------------------------------------------ */
static void render_one(cb *c, const struct ds4cuda_chat_message *m)
{
    const char *content = m->content ? m->content : "";
    role_kind k = classify_role(m->role);
    switch (k) {
    case R_SYSTEM:
        cb_puts(c, content);
        break;
    case R_USER:
        cb_puts(c, DS4CUDA_TOK_USER);
        cb_puts(c, content);
        break;
    case R_TOOL:
        cb_puts(c, DS4CUDA_TOK_USER);
        cb_puts(c, "Tool: ");
        cb_puts(c, content);
        break;
    case R_ASSISTANT:
        cb_puts(c, DS4CUDA_TOK_ASSISTANT);
        /* If body already starts with a <think> marker we don't double-tag. */
        if (strncmp(content, "<think>", 7) != 0 &&
            strncmp(content, "</think>", 8) != 0) {
            cb_puts(c, DS4CUDA_TOK_THINK_CLOSE);
        }
        cb_puts(c, content);
        break;
    }
}

long ds4cuda_render_chat_prompt(
        const struct ds4cuda_chat_message *messages,
        int n_messages,
        bool enable_think,
        char *out, size_t out_size)
{
    if (!out || out_size == 0) return -1;
    cb c;
    cb_init(&c, out, out_size);
    cb_puts(&c, DS4CUDA_TOK_BOS);
    if (messages) {
        for (int i = 0; i < n_messages; i++) {
            render_one(&c, &messages[i]);
        }
    }
    /* Trailing assistant prefix — mirrors ds4_chat_append_assistant_prefix
     * (ds4.c:14584). */
    cb_puts(&c, DS4CUDA_TOK_ASSISTANT);
    cb_puts(&c, enable_think ? DS4CUDA_TOK_THINK_OPEN : DS4CUDA_TOK_THINK_CLOSE);
    if (c.overflow) {
        out[0] = '\0';
        return -1;
    }
    return (long)c.len;
}

char *ds4cuda_render_chat_prompt_alloc(
        const struct ds4cuda_chat_message *messages,
        int n_messages,
        bool enable_think)
{
    /* Two-pass: first compute exact length with a 0-byte buffer to detect
     * overflow + grow until success. Avoids walking the message vector
     * twice with separate counting code. */
    size_t cap = 1024;
    for (;;) {
        char *buf = malloc(cap);
        if (!buf) return NULL;
        long n = ds4cuda_render_chat_prompt(messages, n_messages,
                                            enable_think, buf, cap);
        if (n >= 0) return buf;
        free(buf);
        if (cap > (size_t)64 * 1024 * 1024) return NULL;
        cap *= 2;
    }
}

char *ds4cuda_render_chat_prompt_with_system_alloc(
        const char *system_text,
        const struct ds4cuda_chat_message *messages,
        int n_messages,
        bool enable_think)
{
    /* Fast path: no system text -> identical to the plain alloc path. */
    if (!system_text || !system_text[0]) {
        return ds4cuda_render_chat_prompt_alloc(messages, n_messages, enable_think);
    }
    if (n_messages < 0) n_messages = 0;
    /* Synthesize a {role:"system", content:<system_text>} as the first entry
     * of a freshly built vector. Strings are borrowed (not copied); the
     * vector lives only for the duration of the renderer call. */
    size_t total = (size_t)n_messages + 1;
    struct ds4cuda_chat_message *vec =
        calloc(total, sizeof(struct ds4cuda_chat_message));
    if (!vec) return NULL;
    vec[0].role = "system";
    vec[0].content = system_text;
    if (messages && n_messages > 0) {
        memcpy(vec + 1, messages,
               sizeof(struct ds4cuda_chat_message) * (size_t)n_messages);
    }
    char *out = ds4cuda_render_chat_prompt_alloc(vec, (int)total, enable_think);
    free(vec);
    return out;
}
