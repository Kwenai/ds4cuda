/*
 * anthropic_endpoint.c — POST /v1/messages handler.
 *
 * Mirrors the OpenAI endpoint's structure but speaks the Anthropic
 * /v1/messages dialect:
 *
 *   - "system" is a top-level field (string or array of {type:"text",text})
 *     instead of a role inside "messages".
 *   - Each message's "content" may be a plain string or an array of typed
 *     content blocks: {type:"text"}, {type:"tool_use"}, {type:"tool_result"},
 *     {type:"thinking"}.
 *   - "tools" entries are flat ({name, description, input_schema}) — no
 *     {type:"function", function:{...}} wrapper.
 *
 * The handler reduces all of that into a flat message vector consumed by
 * ds4cuda_render_chat_prompt_with_system_alloc, calls a caller-supplied
 * generator, then formats the response as
 *   {id, type:"message", role:"assistant", model,
 *    content:[ {type:"text",text:...}, {type:"tool_use",...}* ],
 *    stop_reason, stop_sequence, usage:{input_tokens, output_tokens}}.
 *
 * If the generator hands back a string containing a DSML
 * "<｜DSML｜tool_calls>...</｜DSML｜tool_calls>" block, that segment is split
 * out, fed through ds4cuda_anthropic_render_tool_use, and emitted as a
 * separate {type:"tool_use",...} content entry; the surrounding text becomes
 * the {type:"text",...} entry (or "" if the entire reply was DSML).
 */
#define _GNU_SOURCE
#include "anthropic_endpoint.h"

#include "chat_template.h"
#include "tool_calls.h"
#include "../cjson_min/cjson_min.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* DSML markers — kept in sync with tool_calls.c. We re-spell them here so
 * the .c remains independent. \xef\xbd\x9c is U+FF5C (｜). */
static const char *ATH_DSML_TC_OPEN  = "<\xef\xbd\x9c" "DSML\xef\xbd\x9c" "tool_calls>";
static const char *ATH_DSML_TC_CLOSE = "</\xef\xbd\x9c" "DSML\xef\xbd\x9c" "tool_calls>";

/* ------------------------------------------------------------------ */
/* Options storage. Lives for the lifetime of the server.               */
/* ------------------------------------------------------------------ */
struct anth_state {
    char *model_id;
    int default_max_tokens;
    ds4cuda_anthropic_generator_fn generator;
    void *generator_user_data;
};

static char *xstrdup_(const char *s)
{
    if (!s) s = "";
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (!p) abort();
    memcpy(p, s, n);
    return p;
}

/* ------------------------------------------------------------------ */
/* Built-in fallback generator. Returns "OK\n" regardless of input.     */
/* ------------------------------------------------------------------ */
static int anth_stub_generator(const char *dsml_prompt,
                               const char *tools_text,
                               int max_new_tokens,
                               void *user_data,
                               char **out_text)
{
    (void)dsml_prompt; (void)tools_text; (void)max_new_tokens; (void)user_data;
    *out_text = xstrdup_("OK\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Anthropic-flavored error JSON.                                       */
/* ------------------------------------------------------------------ */
static char *anth_error_body(const char *msg, const char *kind)
{
    cjson *root = cjson_min_new_object();
    cjson_min_object_set(root, "type", cjson_min_new_string("error"));
    cjson *err = cjson_min_new_object();
    cjson_min_object_set(err, "type",    cjson_min_new_string(kind ? kind : "invalid_request_error"));
    cjson_min_object_set(err, "message", cjson_min_new_string(msg ? msg : ""));
    cjson_min_object_set(root, "error", err);
    char *s = cjson_min_emit(root);
    cjson_min_free(root);
    return s;
}

/* ------------------------------------------------------------------ */
/* sbuf — tiny dynamic string used here only for content concat.        */
/* ------------------------------------------------------------------ */
typedef struct { char *p; size_t len; size_t cap; } sb;

static void sb_reserve(sb *b, size_t extra)
{
    if (b->len + extra + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 128;
        while (cap < b->len + extra + 1) cap *= 2;
        char *p = realloc(b->p, cap);
        if (!p) abort();
        b->p = p;
        b->cap = cap;
    }
}
static void sb_puts(sb *b, const char *s)
{
    if (!s) return;
    size_t n = strlen(s);
    sb_reserve(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static void sb_putn(sb *b, const char *s, size_t n)
{
    if (!s || !n) return;
    sb_reserve(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}
static char *sb_take_or_empty(sb *b)
{
    if (b->p) return b->p;
    char *e = malloc(1);
    if (!e) return NULL;
    e[0] = '\0';
    return e;
}

/* ------------------------------------------------------------------ */
/* Flatten one message's "content" field (string OR array of blocks)    */
/* into a single text string. tool_use / tool_result are appended in    */
/* DSML form so the chat template renderer sees them as ordinary turn   */
/* body text. The returned pointer is malloc()'d.                       */
/* ------------------------------------------------------------------ */
static char *flatten_content(const cjson *content, const char *role)
{
    sb b = {0};
    if (cjson_min_is_string(content)) {
        sb_puts(&b, cjson_min_string(content));
        return sb_take_or_empty(&b);
    }
    if (!cjson_min_is_array(content)) return sb_take_or_empty(&b);

    int n = cjson_min_array_size(content);
    /* First emit text + thinking parts in order. */
    for (int i = 0; i < n; i++) {
        const cjson *item = cjson_min_array_item(content, i);
        if (cjson_min_is_string(item)) {
            sb_puts(&b, cjson_min_string(item));
            continue;
        }
        if (!cjson_min_is_object(item)) continue;
        const char *type = cjson_min_string(cjson_min_obj_get(item, "type"));
        if (!type) continue;
        if (!strcmp(type, "text")) {
            const char *t = cjson_min_string(cjson_min_obj_get(item, "text"));
            if (t) sb_puts(&b, t);
        } else if (!strcmp(type, "thinking")) {
            /* Embed in <think>...</think> fences so the chat template's
             * assistant prefix doesn't double-fence later. */
            const char *t = cjson_min_string(cjson_min_obj_get(item, "thinking"));
            if (t && t[0]) {
                sb_puts(&b, "<think>");
                sb_puts(&b, t);
                sb_puts(&b, "</think>");
            }
        } else if (!strcmp(type, "tool_result")) {
            /* Mirror ds4_server.c parse_anthropic_content_block: a
             * tool_result block carries a "content" field (string or array).
             * We escape into <tool_result>...</tool_result> fences. */
            const cjson *c = cjson_min_obj_get(item, "content");
            sb_puts(&b, "<tool_result>");
            if (cjson_min_is_string(c)) {
                sb_puts(&b, cjson_min_string(c));
            } else if (cjson_min_is_array(c)) {
                int m = cjson_min_array_size(c);
                for (int j = 0; j < m; j++) {
                    const cjson *cc = cjson_min_array_item(c, j);
                    if (cjson_min_is_string(cc)) {
                        sb_puts(&b, cjson_min_string(cc));
                    } else if (cjson_min_is_object(cc)) {
                        const char *t = cjson_min_string(cjson_min_obj_get(cc, "text"));
                        if (t) sb_puts(&b, t);
                    }
                }
            }
            sb_puts(&b, "</tool_result>");
        }
    }
    /* Then any tool_use blocks (assistant only) become a single DSML
     * <｜DSML｜tool_calls>...</｜DSML｜tool_calls> blob.  We use the helper
     * defined in tool_calls.c which already knows the framing. */
    if (role && !strcmp(role, "assistant")) {
        bool any = false;
        for (int i = 0; i < n; i++) {
            const cjson *item = cjson_min_array_item(content, i);
            if (!cjson_min_is_object(item)) continue;
            const char *type = cjson_min_string(cjson_min_obj_get(item, "type"));
            if (type && !strcmp(type, "tool_use")) { any = true; break; }
        }
        if (any) {
            char *blob = ds4cuda_dsml_render_anthropic_assistant_blocks(content);
            /* The helper concatenates text first + tool_calls. We want only
             * the tool_calls suffix here (the text was already emitted in
             * the first pass above), so locate the first "<｜DSML｜tool_calls>"
             * marker and copy from there. */
            if (blob) {
                const char *m = strstr(blob, ATH_DSML_TC_OPEN);
                if (m) sb_puts(&b, m);
                free(blob);
            }
        }
    }
    return sb_take_or_empty(&b);
}

/* ------------------------------------------------------------------ */
/* Parse Anthropic top-level "system" field: string OR array of either  */
/* strings or {type:"text", text:"..."} objects. Returns malloc()'d     */
/* concatenated string ("" if absent / empty).                          */
/* ------------------------------------------------------------------ */
static char *flatten_system(const cjson *sys)
{
    sb b = {0};
    if (!sys || cjson_min_is_null(sys)) return sb_take_or_empty(&b);
    if (cjson_min_is_string(sys)) {
        sb_puts(&b, cjson_min_string(sys));
        return sb_take_or_empty(&b);
    }
    if (!cjson_min_is_array(sys)) return sb_take_or_empty(&b);
    int n = cjson_min_array_size(sys);
    for (int i = 0; i < n; i++) {
        const cjson *item = cjson_min_array_item(sys, i);
        if (cjson_min_is_string(item)) {
            if (b.len) sb_puts(&b, "\n");
            sb_puts(&b, cjson_min_string(item));
        } else if (cjson_min_is_object(item)) {
            const char *t = cjson_min_string(cjson_min_obj_get(item, "text"));
            if (t && t[0]) {
                if (b.len) sb_puts(&b, "\n");
                sb_puts(&b, t);
            }
        }
    }
    return sb_take_or_empty(&b);
}

/* ------------------------------------------------------------------ */
/* Build a flat ds4cuda_chat_message vector from the Anthropic messages */
/* array. Each entry's content is malloc()'d (kept alive in `owned`).   */
/* ------------------------------------------------------------------ */
struct anth_flat {
    struct ds4cuda_chat_message *items;
    int count;
    char **owned;       /* malloc'd content strings to free at cleanup */
    int n_owned;
    int cap_owned;
};

static void anth_flat_free(struct anth_flat *m)
{
    if (!m) return;
    free(m->items);
    for (int i = 0; i < m->n_owned; i++) free(m->owned[i]);
    free(m->owned);
    memset(m, 0, sizeof(*m));
}

static char *anth_take(struct anth_flat *m, char *s)
{
    if (m->n_owned == m->cap_owned) {
        m->cap_owned = m->cap_owned ? m->cap_owned * 2 : 8;
        m->owned = realloc(m->owned, sizeof(char *) * (size_t)m->cap_owned);
        if (!m->owned) abort();
    }
    m->owned[m->n_owned++] = s;
    return s;
}

static int anth_build_messages(const cjson *messages_arr, struct anth_flat *out,
                               char *err, size_t err_len)
{
    memset(out, 0, sizeof(*out));
    if (!cjson_min_is_array(messages_arr)) {
        snprintf(err, err_len, "messages must be an array");
        return -1;
    }
    int n = cjson_min_array_size(messages_arr);
    if (n <= 0) {
        snprintf(err, err_len, "messages must be non-empty");
        return -1;
    }
    out->items = calloc((size_t)n, sizeof(*out->items));
    if (!out->items) abort();
    out->count = n;
    for (int i = 0; i < n; i++) {
        const cjson *m = cjson_min_array_item(messages_arr, i);
        if (!cjson_min_is_object(m)) {
            snprintf(err, err_len, "messages[%d] must be an object", i);
            anth_flat_free(out);
            return -1;
        }
        const char *role = cjson_min_string(cjson_min_obj_get(m, "role"));
        if (!role) role = "user";
        const cjson *content = cjson_min_obj_get(m, "content");
        char *flat = flatten_content(content, role);
        if (!flat) flat = xstrdup_("");
        anth_take(out, flat);
        out->items[i].role = role;
        out->items[i].content = flat;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Locate a DSML tool_calls block inside the generator's reply.         */
/* Returns pointers into `text` (no allocation).                        */
/* ------------------------------------------------------------------ */
static void split_dsml_tool_calls(const char *text,
                                  const char **tc_start,
                                  const char **tc_end_inc)
{
    *tc_start = NULL;
    *tc_end_inc = NULL;
    if (!text) return;
    const char *o = strstr(text, ATH_DSML_TC_OPEN);
    if (!o) return;
    const char *c = strstr(o, ATH_DSML_TC_CLOSE);
    if (!c) return;
    *tc_start = o;
    *tc_end_inc = c + strlen(ATH_DSML_TC_CLOSE);
}

/* ------------------------------------------------------------------ */
/* Build the response object.                                           */
/* ------------------------------------------------------------------ */
static char *anth_response_body(const struct anth_state *st,
                                const char *content,
                                int input_tokens, int output_tokens)
{
    unsigned long t = (unsigned long)time(NULL);
    char id[40];
    snprintf(id, sizeof(id), "msg_%08lx", t & 0xfffffffful);

    /* Detect a DSML tool_calls block inside `content`. */
    const char *tc_o = NULL, *tc_e = NULL;
    split_dsml_tool_calls(content, &tc_o, &tc_e);

    cjson *root = cjson_min_new_object();
    cjson_min_object_set(root, "id",    cjson_min_new_string(id));
    cjson_min_object_set(root, "type",  cjson_min_new_string("message"));
    cjson_min_object_set(root, "role",  cjson_min_new_string("assistant"));
    cjson_min_object_set(root, "model", cjson_min_new_string(st->model_id));

    cjson *content_arr = cjson_min_new_array();

    bool has_tool_use = false;
    if (tc_o && tc_e) {
        /* Pre-text portion (may be empty). */
        size_t pre_len = (size_t)(tc_o - content);
        char *pre = malloc(pre_len + 1);
        if (!pre) abort();
        memcpy(pre, content, pre_len);
        pre[pre_len] = '\0';
        if (pre[0]) {
            cjson *tb = cjson_min_new_object();
            cjson_min_object_set(tb, "type", cjson_min_new_string("text"));
            cjson_min_object_set(tb, "text", cjson_min_new_string(pre));
            cjson_min_array_push(content_arr, tb);
        }
        free(pre);
        /* Convert each invoke inside [tc_o..tc_e) into an Anthropic
         * tool_use block. */
        size_t blen = (size_t)(tc_e - tc_o);
        char *blob = malloc(blen + 1);
        if (!blob) abort();
        memcpy(blob, tc_o, blen);
        blob[blen] = '\0';
        char *anth_block = ds4cuda_anthropic_render_tool_use(blob);
        free(blob);
        if (anth_block) {
            cjson *parsed = cjson_min_parse(anth_block);
            free(anth_block);
            if (parsed) {
                cjson_min_array_push(content_arr, parsed);
                has_tool_use = true;
            }
        }
        /* Post-text (rare but possible). */
        if (*tc_e) {
            cjson *tb = cjson_min_new_object();
            cjson_min_object_set(tb, "type", cjson_min_new_string("text"));
            cjson_min_object_set(tb, "text", cjson_min_new_string(tc_e));
            cjson_min_array_push(content_arr, tb);
        }
    } else {
        /* Plain text reply. */
        cjson *tb = cjson_min_new_object();
        cjson_min_object_set(tb, "type", cjson_min_new_string("text"));
        cjson_min_object_set(tb, "text", cjson_min_new_string(content ? content : ""));
        cjson_min_array_push(content_arr, tb);
    }
    cjson_min_object_set(root, "content", content_arr);

    cjson_min_object_set(root, "stop_reason",
                         cjson_min_new_string(has_tool_use ? "tool_use" : "end_turn"));
    cjson_min_object_set(root, "stop_sequence", cjson_min_new_null());

    cjson *usage = cjson_min_new_object();
    cjson_min_object_set(usage, "input_tokens",  cjson_min_new_number((double)input_tokens));
    cjson_min_object_set(usage, "output_tokens", cjson_min_new_number((double)output_tokens));
    cjson_min_object_set(root, "usage", usage);

    char *out = cjson_min_emit(root);
    cjson_min_free(root);
    return out;
}

/* ------------------------------------------------------------------ */
/* Core handler — shared by HTTP path + in-process test path.           */
/* ------------------------------------------------------------------ */
int ds4cuda_anthropic_handle_messages(
        const struct ds4cuda_anthropic_endpoint_options *opts_in,
        const char *body, size_t body_len,
        char **out_body, size_t *out_body_len, int *out_status)
{
    struct anth_state st = {0};
    st.model_id = (char *)(opts_in && opts_in->model_id ? opts_in->model_id : "deepseek-v4-flash");
    st.default_max_tokens = opts_in && opts_in->default_max_tokens > 0
                              ? opts_in->default_max_tokens : 256;
    st.generator = opts_in && opts_in->generator ? opts_in->generator : anth_stub_generator;
    st.generator_user_data = opts_in ? opts_in->generator_user_data : NULL;

    char err[256] = {0};
    cjson *req = cjson_min_parse_with_len(body ? body : "", body_len);
    if (!req) {
        *out_body = anth_error_body("invalid JSON in request body", "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }
    if (!cjson_min_is_object(req)) {
        cjson_min_free(req);
        *out_body = anth_error_body("request body must be a JSON object", "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Anthropic /v1/messages endpoint: SSE streaming is not implemented. */
    const cjson *stream = cjson_min_obj_get(req, "stream");
    if (stream && cjson_min_is_bool(stream) && cjson_min_bool(stream)) {
        cjson_min_free(req);
        *out_body = anth_error_body("stream=true not supported by this server",
                                    "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Validate tools (Anthropic flat shape). */
    const cjson *tools = cjson_min_obj_get(req, "tools");
    if (ds4cuda_validate_anthropic_tools(tools, err, sizeof(err)) != 0) {
        cjson_min_free(req);
        *out_body = anth_error_body(err[0] ? err : "invalid tools schema",
                                    "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Optional max_tokens. Must be > 0 if provided. */
    int max_tokens = st.default_max_tokens;
    const cjson *mt = cjson_min_obj_get(req, "max_tokens");
    if (cjson_min_is_number(mt)) {
        double v = cjson_min_number(mt);
        if (!(v > 0)) {
            cjson_min_free(req);
            *out_body = anth_error_body("max_tokens must be > 0",
                                        "invalid_request_error");
            *out_body_len = strlen(*out_body);
            *out_status = 400;
            return 0;
        }
        max_tokens = (int)v;
    }

    /* Required messages. */
    const cjson *messages = cjson_min_obj_get(req, "messages");
    struct anth_flat flat;
    if (anth_build_messages(messages, &flat, err, sizeof(err)) != 0) {
        cjson_min_free(req);
        *out_body = anth_error_body(err[0] ? err : "invalid messages",
                                    "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Optional system field. */
    char *system_text = flatten_system(cjson_min_obj_get(req, "system"));
    if (!system_text) system_text = xstrdup_("");

    /* Render DSML prompt. */
    char *prompt = ds4cuda_render_chat_prompt_with_system_alloc(
            system_text, flat.items, flat.count, false);
    if (!prompt) {
        free(system_text);
        anth_flat_free(&flat);
        cjson_min_free(req);
        *out_body = anth_error_body("failed to render chat prompt", "server_error");
        *out_body_len = strlen(*out_body);
        *out_status = 500;
        return 0;
    }

    /* Render DSML tools text (may be empty). */
    char *tools_text = ds4cuda_dsml_render_anthropic_tools(tools);
    if (!tools_text) tools_text = xstrdup_("");

    /* Invoke generator. */
    char *content = NULL;
    int rc = st.generator(prompt, tools_text, max_tokens,
                          st.generator_user_data, &content);
    if (rc != 0 || !content) {
        free(content);
        free(tools_text);
        free(prompt);
        free(system_text);
        anth_flat_free(&flat);
        cjson_min_free(req);
        *out_body = anth_error_body("generator failed", "server_error");
        *out_body_len = strlen(*out_body);
        *out_status = 500;
        return 0;
    }

    /* Token estimate. The integrated build replaces with real counters. */
    int input_tokens  = (int)((strlen(prompt) + strlen(system_text)) / 4);
    int output_tokens = (int)(strlen(content) / 4);
    char *resp = anth_response_body(&st, content, input_tokens, output_tokens);

    free(content);
    free(tools_text);
    free(prompt);
    free(system_text);
    anth_flat_free(&flat);
    cjson_min_free(req);

    if (!resp) {
        *out_body = anth_error_body("failed to serialize response", "server_error");
        *out_body_len = strlen(*out_body);
        *out_status = 500;
        return 0;
    }
    *out_body = resp;
    *out_body_len = strlen(resp);
    *out_status = 200;
    return 0;
}

/* ------------------------------------------------------------------ */
/* HTTP-side wrapper.                                                   */
/* ------------------------------------------------------------------ */
static int messages_handler(const char *method, const char *path,
                            const char *body, size_t body_len,
                            char **out_body, size_t *out_body_len,
                            int *out_status, char **out_content_type,
                            void *user_data)
{
    (void)path;
    if (out_content_type) *out_content_type = xstrdup_("application/json");
    if (strcmp(method, "POST") != 0) {
        *out_body = anth_error_body("method not allowed; use POST",
                                    "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 405;
        return 0;
    }
    struct anth_state *st = user_data;
    struct ds4cuda_anthropic_endpoint_options opts = {
        .model_id = st->model_id,
        .default_max_tokens = st->default_max_tokens,
        .generator = st->generator,
        .generator_user_data = st->generator_user_data,
    };
    return ds4cuda_anthropic_handle_messages(&opts, body, body_len,
                                             out_body, out_body_len, out_status);
}

int ds4cuda_anthropic_endpoint_install(
        struct ds4cuda_http_server *server,
        const struct ds4cuda_anthropic_endpoint_options *opts)
{
    if (!server) return -EINVAL;
    struct anth_state *st = malloc(sizeof(*st));
    if (!st) return -ENOMEM;
    memset(st, 0, sizeof(*st));
    st->model_id = xstrdup_(opts && opts->model_id ? opts->model_id : "deepseek-v4-flash");
    st->default_max_tokens = opts && opts->default_max_tokens > 0
                                ? opts->default_max_tokens : 256;
    st->generator = opts && opts->generator ? opts->generator : anth_stub_generator;
    st->generator_user_data = opts ? opts->generator_user_data : NULL;

    int rc = ds4cuda_http_server_register(server, "/v1/messages",
                                          messages_handler, st);
    if (rc != 0) {
        free(st->model_id);
        free(st);
        return rc;
    }
    return 0;
}
