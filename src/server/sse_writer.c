/*
 * sse_writer.c — pure formatting helpers; no I/O. See sse_writer.h.
 *
 * The chunk-JSON builder uses cjson_min directly so any string content
 * is properly escaped (newlines, quotes, UTF-8). We do not inline a
 * sprintf("data: {\"content\":\"%s\"}") path because real model output
 * routinely contains characters that would corrupt JSON if spliced raw.
 */
#define _GNU_SOURCE
#include "sse_writer.h"

#include "../cjson_min/cjson_min.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *xstrdup_sse_(const char *s)
{
    if (!s) s = "";
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (!p) abort();
    memcpy(p, s, n);
    return p;
}

char *ds4cuda_sse_format_data(const char *payload)
{
    if (!payload) payload = "";
    size_t pn = strlen(payload);
    /* "data: " (6) + payload + "\n\n" (2) + NUL */
    size_t total = 6 + pn + 2 + 1;
    char *out = malloc(total);
    if (!out) abort();
    memcpy(out, "data: ", 6);
    memcpy(out + 6, payload, pn);
    out[6 + pn]     = '\n';
    out[6 + pn + 1] = '\n';
    out[6 + pn + 2] = '\0';
    return out;
}

char *ds4cuda_sse_format_done(void)
{
    return xstrdup_sse_("data: [DONE]\n\n");
}

char *ds4cuda_sse_build_chunk_json(const char *id,
                                   const char *model,
                                   long created,
                                   const char *role,
                                   const char *content,
                                   const char *finish_reason)
{
    cjson *root = cjson_min_new_object();
    cjson_min_object_set(root, "id",      cjson_min_new_string(id ? id : ""));
    cjson_min_object_set(root, "object",  cjson_min_new_string("chat.completion.chunk"));
    cjson_min_object_set(root, "created", cjson_min_new_number((double)created));
    cjson_min_object_set(root, "model",   cjson_min_new_string(model ? model : ""));

    cjson *choices = cjson_min_new_array();
    cjson *choice  = cjson_min_new_object();
    cjson_min_object_set(choice, "index", cjson_min_new_number(0));

    cjson *delta = cjson_min_new_object();
    if (role && *role) {
        cjson_min_object_set(delta, "role", cjson_min_new_string(role));
    }
    if (content) {
        /* Even the empty string is meaningful here — that's how OpenAI
         * signals the first "open the assistant turn" chunk. */
        cjson_min_object_set(delta, "content", cjson_min_new_string(content));
    }
    cjson_min_object_set(choice, "delta", delta);

    if (finish_reason && *finish_reason) {
        cjson_min_object_set(choice, "finish_reason",
                             cjson_min_new_string(finish_reason));
    } else {
        cjson_min_object_set(choice, "finish_reason", cjson_min_new_null());
    }
    cjson_min_array_push(choices, choice);
    cjson_min_object_set(root, "choices", choices);

    char *out = cjson_min_emit(root);
    cjson_min_free(root);
    return out;
}
