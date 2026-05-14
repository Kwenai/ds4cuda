/*
 * openai_endpoint.c — POST /v1/chat/completions + GET /v1/models handlers.
 *
 * Contract:
 *
 *   POST /v1/chat/completions
 *     - Parse {"model","messages":[{role,content}*],"tools","max_tokens","stream"}
 *     - Buffered path (stream=false): emit one Content-Length response.
 *     - Streaming path (stream=true): emit SSE chat.completion.chunk frames.
 *     - Validate the tools array against the OpenAI schema.
 *     - Render messages via chat_template into a DSML prompt string.
 *     - Invoke the caller-installed generator callback.
 *     - Emit OpenAI-shape response JSON (or SSE frames).
 *
 *   GET /v1/models, GET /v1/models/<id>
 *     - Static metadata for the served model id.
 *
 * Memory: every JSON tree built via cjson_min is freed before return; the
 * malloc()'d response body is handed to the HTTP layer which frees after
 * send.
 */
#define _GNU_SOURCE
#include "openai_endpoint.h"

#include "chat_template.h"
#include "sse_writer.h"
#include "tool_calls.h"
#include "../cjson_min/cjson_min.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Options storage. Lives for the lifetime of the server.                */
/* ------------------------------------------------------------------ */
struct endpoint_state {
    char *model_id;
    int default_max_tokens;
    ds4cuda_chat_generator_fn generator;
    void *generator_user_data;
    ds4cuda_chat_stream_generator_fn stream_generator;
    void *stream_generator_user_data;
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
/* Built-in fallback generator used when no generator is installed.      */
/* Returns "OK\n" so the route is exercisable without a model.           */
/* ------------------------------------------------------------------ */
static int stub_generator(const char *prompt,
                          int max_new_tokens,
                          void *user_data,
                          char **out_text)
{
    (void)prompt; (void)max_new_tokens; (void)user_data;
    *out_text = xstrdup_("OK\n");
    return 0;
}

/* Built-in fallback streaming generator: emits "OK\n" as three single-
 * character tokens. The inference-engine-backed generator follows the
 * same emit() contract. We deliberately do NOT sleep here — tests need
 * to be fast and clients already see the cadence via separate chunked
 * frames. */
static int stub_stream_generator(const char *prompt,
                                 int max_new_tokens,
                                 void *user_data,
                                 ds4cuda_chat_stream_emit_fn emit,
                                 void *emit_user_data)
{
    (void)prompt; (void)max_new_tokens; (void)user_data;
    static const char *toks[] = {"O", "K", "\n"};
    for (int i = 0; i < 3; i++) {
        if (emit(toks[i], -1, emit_user_data) != 0) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Helper: emit an OpenAI-flavored error JSON payload.                   */
/* ------------------------------------------------------------------ */
static char *error_body(const char *msg, const char *kind)
{
    cjson *root = cjson_min_new_object();
    cjson *err = cjson_min_new_object();
    cjson_min_object_set(err, "message", cjson_min_new_string(msg ? msg : ""));
    cjson_min_object_set(err, "type",    cjson_min_new_string(kind ? kind : "invalid_request_error"));
    cjson_min_object_set(root, "error", err);
    char *s = cjson_min_emit(root);
    cjson_min_free(root);
    return s;
}

/* ------------------------------------------------------------------ */
/* Helper: convert a parsed messages array (cjson) into a flat array of  */
/* ds4cuda_chat_message. Each entry's role / content point into strings  */
/* owned by the cjson tree, so the caller must keep the tree alive while */
/* the message vector is in use.                                         */
/* ------------------------------------------------------------------ */
struct flat_messages {
    struct ds4cuda_chat_message *items;
    int count;
    /* For OpenAI multi-part content arrays we may need to allocate joined
     * strings; we collect those here for free-on-cleanup. */
    char **owned_strings;
    int n_owned;
    int cap_owned;
};

static void flat_messages_free(struct flat_messages *m)
{
    if (!m) return;
    free(m->items);
    for (int i = 0; i < m->n_owned; i++) free(m->owned_strings[i]);
    free(m->owned_strings);
    memset(m, 0, sizeof(*m));
}

static char *take_owned(struct flat_messages *m, char *s)
{
    if (m->n_owned == m->cap_owned) {
        m->cap_owned = m->cap_owned ? m->cap_owned * 2 : 8;
        m->owned_strings = realloc(m->owned_strings, sizeof(char *) * (size_t)m->cap_owned);
        if (!m->owned_strings) abort();
    }
    m->owned_strings[m->n_owned++] = s;
    return s;
}

/* OpenAI permits content as either a plain string or an array of
 * {type:"text"|"input_text", text:"..."} objects. Anthropic uses a similar
 * shape with tool_use / tool_result blocks; non-text blocks are ignored
 * here (Anthropic blocks go through anthropic_endpoint.c). Returns owned
 * char* (never NULL). */
static const char *extract_content(const cjson *content, struct flat_messages *m)
{
    if (cjson_min_is_string(content)) return cjson_min_string(content);
    if (!cjson_min_is_array(content)) return "";
    /* concatenate every text-typed part */
    size_t cap = 256, len = 0;
    char *out = malloc(cap);
    if (!out) abort();
    out[0] = '\0';
    int n = cjson_min_array_size(content);
    for (int i = 0; i < n; i++) {
        const cjson *part = cjson_min_array_item(content, i);
        if (!cjson_min_is_object(part)) continue;
        const cjson *txt = cjson_min_obj_get(part, "text");
        if (!txt) txt = cjson_min_obj_get(part, "content");
        const char *s = cjson_min_string(txt);
        if (!s) continue;
        size_t add = strlen(s);
        while (len + add + 1 > cap) {
            cap *= 2;
            char *p = realloc(out, cap);
            if (!p) abort();
            out = p;
        }
        memcpy(out + len, s, add); len += add;
        out[len] = '\0';
    }
    return take_owned(m, out);
}

static int build_messages(const cjson *messages_arr, struct flat_messages *out,
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
            flat_messages_free(out);
            return -1;
        }
        const char *role = cjson_min_string(cjson_min_obj_get(m, "role"));
        if (!role) role = "user";
        const cjson *content = cjson_min_obj_get(m, "content");
        const char *body = content ? extract_content(content, out) : "";
        out->items[i].role = role;
        out->items[i].content = body ? body : "";
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Build the response object.                                            */
/* ------------------------------------------------------------------ */
static char *response_body(const struct endpoint_state *st,
                           const char *content,
                           int prompt_tokens, int completion_tokens)
{
    /* id: "chatcmpl-" + 8 hex chars derived from time + a random nibble. */
    unsigned long t = (unsigned long)time(NULL);
    char id[40];
    snprintf(id, sizeof(id), "chatcmpl-%08lx", t & 0xfffffffful);

    cjson *root = cjson_min_new_object();
    cjson_min_object_set(root, "id", cjson_min_new_string(id));
    cjson_min_object_set(root, "object", cjson_min_new_string("chat.completion"));
    cjson_min_object_set(root, "created", cjson_min_new_number((double)t));
    cjson_min_object_set(root, "model", cjson_min_new_string(st->model_id));

    cjson *choice = cjson_min_new_object();
    cjson_min_object_set(choice, "index", cjson_min_new_number(0));
    cjson *msg = cjson_min_new_object();
    cjson_min_object_set(msg, "role",    cjson_min_new_string("assistant"));
    cjson_min_object_set(msg, "content", cjson_min_new_string(content ? content : ""));
    cjson_min_object_set(choice, "message", msg);
    cjson_min_object_set(choice, "finish_reason", cjson_min_new_string("stop"));
    cjson *choices = cjson_min_new_array();
    cjson_min_array_push(choices, choice);
    cjson_min_object_set(root, "choices", choices);

    cjson *usage = cjson_min_new_object();
    cjson_min_object_set(usage, "prompt_tokens",     cjson_min_new_number((double)prompt_tokens));
    cjson_min_object_set(usage, "completion_tokens", cjson_min_new_number((double)completion_tokens));
    cjson_min_object_set(usage, "total_tokens",      cjson_min_new_number((double)(prompt_tokens + completion_tokens)));
    cjson_min_object_set(root, "usage", usage);

    char *out = cjson_min_emit(root);
    cjson_min_free(root);
    return out;
}

/* ------------------------------------------------------------------ */
/* Core handler — shared by HTTP path + in-process test path.            */
/* ------------------------------------------------------------------ */
int ds4cuda_openai_handle_chat_completion(
        const struct ds4cuda_chat_endpoint_options *opts_in,
        const char *body, size_t body_len,
        char **out_body, size_t *out_body_len, int *out_status)
{
    struct endpoint_state st = {0};
    st.model_id = (char *)(opts_in && opts_in->model_id ? opts_in->model_id : "deepseek-v4-flash");
    st.default_max_tokens = opts_in && opts_in->default_max_tokens > 0
                              ? opts_in->default_max_tokens : 256;
    st.generator = opts_in && opts_in->generator ? opts_in->generator : stub_generator;
    st.generator_user_data = opts_in ? opts_in->generator_user_data : NULL;

    char err[256] = {0};
    cjson *req = cjson_min_parse_with_len(body ? body : "", body_len);
    if (!req) {
        *out_body = error_body("invalid JSON in request body", "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }
    if (!cjson_min_is_object(req)) {
        cjson_min_free(req);
        *out_body = error_body("request body must be a JSON object", "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Reject stream=true on the in-process buffered path. HTTP clients
     * reach the SSE path via chat_stream_handler instead. */
    const cjson *stream = cjson_min_obj_get(req, "stream");
    if (stream && cjson_min_is_bool(stream) && cjson_min_bool(stream)) {
        cjson_min_free(req);
        *out_body = error_body("stream=true not supported on this entry point; "
                               "use POST /v1/chat/completions over HTTP for SSE",
                               "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Validate tools (best-effort). */
    const cjson *tools = cjson_min_obj_get(req, "tools");
    if (ds4cuda_validate_openai_tools(tools, err, sizeof(err)) != 0) {
        cjson_min_free(req);
        *out_body = error_body(err[0] ? err : "invalid tools schema", "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Optional max_tokens. Must be > 0 if provided; otherwise the engine
     * applies the per-session ceiling derived from --max-context. */
    int max_tokens = st.default_max_tokens;
    const cjson *mt = cjson_min_obj_get(req, "max_tokens");
    if (cjson_min_is_number(mt)) {
        double v = cjson_min_number(mt);
        if (!(v > 0)) {
            cjson_min_free(req);
            *out_body = error_body("max_tokens must be > 0",
                                   "invalid_request_error");
            *out_body_len = strlen(*out_body);
            *out_status = 400;
            return 0;
        }
        max_tokens = (int)v;
    }

    /* Required messages. */
    const cjson *messages = cjson_min_obj_get(req, "messages");
    struct flat_messages flat;
    if (build_messages(messages, &flat, err, sizeof(err)) != 0) {
        cjson_min_free(req);
        *out_body = error_body(err[0] ? err : "invalid messages", "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 400;
        return 0;
    }

    /* Render prompt. */
    char *prompt = ds4cuda_render_chat_prompt_alloc(flat.items, flat.count, false);
    if (!prompt) {
        flat_messages_free(&flat);
        cjson_min_free(req);
        *out_body = error_body("failed to render chat prompt", "server_error");
        *out_body_len = strlen(*out_body);
        *out_status = 500;
        return 0;
    }

    /* Invoke generator. */
    char *content = NULL;
    int rc = st.generator(prompt, max_tokens, st.generator_user_data, &content);
    if (rc != 0 || !content) {
        free(content);
        free(prompt);
        flat_messages_free(&flat);
        cjson_min_free(req);
        *out_body = error_body("generator failed", "server_error");
        *out_body_len = strlen(*out_body);
        *out_status = 500;
        return 0;
    }

    /* Naive token counts: prompt bytes / 4 + completion bytes / 4. The
     * integrated build replaces with real tokenizer counts. */
    int prompt_tokens     = (int)(strlen(prompt)  / 4);
    int completion_tokens = (int)(strlen(content) / 4);
    char *resp = response_body(&st, content, prompt_tokens, completion_tokens);

    free(content);
    free(prompt);
    flat_messages_free(&flat);
    cjson_min_free(req);

    if (!resp) {
        *out_body = error_body("failed to serialize response", "server_error");
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
/* Streaming path: parse messages, render prompt, drive the streaming   */
/* generator, frame each token as an SSE chat.completion.chunk event.   */
/* ------------------------------------------------------------------ */
struct sse_emit_state {
    struct ds4cuda_http_stream_ctx *ctx;
    const char *id;
    const char *model_id;
    long created;
    int  any_token_emitted;
    int  emitted_tokens;
};

static int sse_emit_token(const char *token, int len, void *u)
{
    struct sse_emit_state *st = u;
    if (!token) return -EINVAL;
    size_t L = (len < 0) ? strlen(token) : (size_t)len;
    /* Drop empty tokens silently — clients never expect a "" delta and
     * the chunked-transfer layer can't represent zero-byte fragments
     * anyway. */
    if (L == 0) return 0;

    /* Build a sized copy so we can NUL-terminate for the JSON
     * builder's string-set call. */
    char *copy = malloc(L + 1);
    if (!copy) abort();
    memcpy(copy, token, L);
    copy[L] = '\0';

    char *json = ds4cuda_sse_build_chunk_json(st->id, st->model_id, st->created,
                                              /*role*/ NULL, copy,
                                              /*finish_reason*/ NULL);
    free(copy);
    if (!json) return -ENOMEM;
    char *frame = ds4cuda_sse_format_data(json);
    free(json);
    if (!frame) return -ENOMEM;
    int rc = ds4cuda_http_stream_write_chunk(st->ctx, frame, strlen(frame));
    free(frame);
    if (rc != 0) return rc;
    st->any_token_emitted = 1;
    st->emitted_tokens++;
    return 0;
}

/* Encapsulate the SSE output for one accepted streaming request. The
 * caller has already done JSON parsing + tools validation + prompt
 * render and just needs the wire-format produced. */
static int run_sse_stream(struct ds4cuda_http_stream_ctx *ctx,
                          struct endpoint_state *st,
                          const char *prompt,
                          int max_tokens)
{
    /* Same id format as the buffered response so logs line up. */
    long t = (long)time(NULL);
    char id[40];
    snprintf(id, sizeof(id), "chatcmpl-%08lx", (unsigned long)(t & 0xfffffffful));

    int rc = ds4cuda_http_stream_begin(ctx, 200, "text/event-stream");
    if (rc != 0) return rc;

    /* Frame 1: open the assistant turn with role only (OpenAI parity). */
    char *opener = ds4cuda_sse_build_chunk_json(id, st->model_id, t,
                                                "assistant", "", NULL);
    if (!opener) return -ENOMEM;
    char *opener_frame = ds4cuda_sse_format_data(opener);
    free(opener);
    if (!opener_frame) return -ENOMEM;
    rc = ds4cuda_http_stream_write_chunk(ctx, opener_frame, strlen(opener_frame));
    free(opener_frame);
    if (rc != 0) return rc;

    /* Stream tokens. */
    struct sse_emit_state es = {
        .ctx = ctx, .id = id, .model_id = st->model_id, .created = t,
    };
    ds4cuda_chat_stream_generator_fn gen =
        st->stream_generator ? st->stream_generator : stub_stream_generator;
    int gen_rc = gen(prompt, max_tokens,
                     st->stream_generator_user_data,
                     sse_emit_token, &es);

    /* Final non-DONE chunk: empty delta + finish_reason.  The current
     * streaming generator ABI returns only success/failure, not its
     * internal stop reason.  For the real worker, reaching max_tokens
     * means it emitted exactly max_tokens chunks and set its internal
     * finish_reason to "length"; mirror that at the endpoint so clients
     * can distinguish truncation from EOS. */
    const char *finish = "stop";
    if (gen_rc == 0 && max_tokens > 0 && es.emitted_tokens >= max_tokens) {
        finish = "length";
    }
    char *tail = ds4cuda_sse_build_chunk_json(id, st->model_id, t,
                                              NULL, NULL, finish);
    if (!tail) return -ENOMEM;
    char *tail_frame = ds4cuda_sse_format_data(tail);
    free(tail);
    if (!tail_frame) return -ENOMEM;
    rc = ds4cuda_http_stream_write_chunk(ctx, tail_frame, strlen(tail_frame));
    free(tail_frame);
    if (rc != 0) return rc;

    /* Sentinel + terminator. */
    char *done = ds4cuda_sse_format_done();
    if (!done) return -ENOMEM;
    rc = ds4cuda_http_stream_write_chunk(ctx, done, strlen(done));
    free(done);
    if (rc != 0) return rc;

    return ds4cuda_http_stream_end(ctx);
}

/* Parse the body's `stream` field. Returns true iff stream=true is
 * explicitly present and truthy. NULL/missing/false all return false. */
static bool body_requests_stream(const char *body, size_t body_len)
{
    if (!body || body_len == 0) return false;
    cjson *req = cjson_min_parse_with_len(body, body_len);
    if (!req) return false;
    bool yes = false;
    if (cjson_min_is_object(req)) {
        const cjson *s = cjson_min_obj_get(req, "stream");
        if (s && cjson_min_is_bool(s) && cjson_min_bool(s)) yes = true;
    }
    cjson_min_free(req);
    return yes;
}

/* Streaming-aware chat handler. Always registered on
 * /v1/chat/completions when the endpoint is installed.
 *
 *   stream=true  -> SSE / chunked path (run_sse_stream).
 *   otherwise    -> buffered path: delegate to
 *                   ds4cuda_openai_handle_chat_completion and ship the
 *                   result with a Content-Length response so non-streaming
 *                   clients see the same bytes as the in-process helper. */
static int chat_stream_handler(const char *method, const char *path,
                               const char *body, size_t body_len,
                               struct ds4cuda_http_stream_ctx *ctx,
                               void *user_data)
{
    (void)path;
    struct endpoint_state *st = user_data;

    if (strcmp(method, "POST") != 0) {
        char *eb = error_body("method not allowed; use POST", "invalid_request_error");
        size_t el = eb ? strlen(eb) : 0;
        int rc = ds4cuda_http_stream_send_buffered(ctx, 405,
                                                   "application/json",
                                                   eb ? eb : "", el);
        free(eb);
        return rc;
    }

    if (!body_requests_stream(body, body_len)) {
        /* Buffered path. We let the in-process helper do all the JSON +
         * validation + prompt rendering. */
        struct ds4cuda_chat_endpoint_options opts = {
            .model_id = st->model_id,
            .default_max_tokens = st->default_max_tokens,
            .generator = st->generator,
            .generator_user_data = st->generator_user_data,
        };
        char *resp = NULL; size_t resp_len = 0; int status = 0;
        ds4cuda_openai_handle_chat_completion(&opts, body, body_len,
                                              &resp, &resp_len, &status);
        if (status == 0) status = 500;
        int rc = ds4cuda_http_stream_send_buffered(ctx, status,
                                                   "application/json",
                                                   resp ? resp : "", resp_len);
        free(resp);
        return rc;
    }

    /* --- streaming path: validate the request shape before opening
     * SSE headers, so a malformed body can still get a clean 400. */
    char err[256] = {0};
    cjson *req = cjson_min_parse_with_len(body ? body : "", body_len);
    if (!req || !cjson_min_is_object(req)) {
        if (req) cjson_min_free(req);
        char *eb = error_body("invalid JSON in request body", "invalid_request_error");
        size_t el = eb ? strlen(eb) : 0;
        int rc = ds4cuda_http_stream_send_buffered(ctx, 400,
                                                   "application/json",
                                                   eb ? eb : "", el);
        free(eb);
        return rc;
    }
    const cjson *tools = cjson_min_obj_get(req, "tools");
    if (ds4cuda_validate_openai_tools(tools, err, sizeof(err)) != 0) {
        cjson_min_free(req);
        char *eb = error_body(err[0] ? err : "invalid tools schema",
                              "invalid_request_error");
        size_t el = eb ? strlen(eb) : 0;
        int rc = ds4cuda_http_stream_send_buffered(ctx, 400,
                                                   "application/json",
                                                   eb ? eb : "", el);
        free(eb);
        return rc;
    }
    int max_tokens = st->default_max_tokens;
    const cjson *mt = cjson_min_obj_get(req, "max_tokens");
    if (cjson_min_is_number(mt)) {
        double v = cjson_min_number(mt);
        if (!(v > 0)) {
            cjson_min_free(req);
            char *eb = error_body("max_tokens must be > 0",
                                  "invalid_request_error");
            size_t el = eb ? strlen(eb) : 0;
            int rc = ds4cuda_http_stream_send_buffered(ctx, 400,
                                                       "application/json",
                                                       eb ? eb : "", el);
            free(eb);
            return rc;
        }
        max_tokens = (int)v;
    }
    const cjson *messages = cjson_min_obj_get(req, "messages");
    struct flat_messages flat;
    if (build_messages(messages, &flat, err, sizeof(err)) != 0) {
        cjson_min_free(req);
        char *eb = error_body(err[0] ? err : "invalid messages",
                              "invalid_request_error");
        size_t el = eb ? strlen(eb) : 0;
        int rc = ds4cuda_http_stream_send_buffered(ctx, 400,
                                                   "application/json",
                                                   eb ? eb : "", el);
        free(eb);
        return rc;
    }

    char *prompt = ds4cuda_render_chat_prompt_alloc(flat.items, flat.count, false);
    flat_messages_free(&flat);
    cjson_min_free(req);
    if (!prompt) {
        char *eb = error_body("failed to render chat prompt", "server_error");
        size_t el = eb ? strlen(eb) : 0;
        int rc = ds4cuda_http_stream_send_buffered(ctx, 500,
                                                   "application/json",
                                                   eb ? eb : "", el);
        free(eb);
        return rc;
    }

    int rc = run_sse_stream(ctx, st, prompt, max_tokens);
    free(prompt);
    return rc;
}

static int models_handler(const char *method, const char *path,
                          const char *body, size_t body_len,
                          char **out_body, size_t *out_body_len,
                          int *out_status, char **out_content_type,
                          void *user_data)
{
    (void)body; (void)body_len;
    if (out_content_type) *out_content_type = xstrdup_("application/json");
    if (strcmp(method, "GET") != 0) {
        *out_body = error_body("method not allowed; use GET", "invalid_request_error");
        *out_body_len = strlen(*out_body);
        *out_status = 405;
        return 0;
    }
    struct endpoint_state *st = user_data;

    cjson *one = cjson_min_new_object();
    cjson_min_object_set(one, "id", cjson_min_new_string(st->model_id));
    cjson_min_object_set(one, "object", cjson_min_new_string("model"));
    cjson_min_object_set(one, "owned_by", cjson_min_new_string("ds4cuda"));

    cjson *root;
    if (!strcmp(path, "/v1/models")) {
        cjson *arr = cjson_min_new_array();
        cjson_min_array_push(arr, one);
        root = cjson_min_new_object();
        cjson_min_object_set(root, "object", cjson_min_new_string("list"));
        cjson_min_object_set(root, "data", arr);
    } else {
        root = one;
    }
    char *s = cjson_min_emit(root);
    cjson_min_free(root);
    *out_body = s;
    *out_body_len = strlen(s);
    *out_status = 200;
    return 0;
}

int ds4cuda_openai_endpoint_install(
        struct ds4cuda_http_server *server,
        const struct ds4cuda_chat_endpoint_options *opts)
{
    if (!server) return -EINVAL;
    struct endpoint_state *st = malloc(sizeof(*st));
    if (!st) return -ENOMEM;
    memset(st, 0, sizeof(*st));
    st->model_id = xstrdup_(opts && opts->model_id ? opts->model_id : "deepseek-v4-flash");
    st->default_max_tokens = opts && opts->default_max_tokens > 0
                                ? opts->default_max_tokens : 256;
    st->generator = opts && opts->generator ? opts->generator : stub_generator;
    st->generator_user_data = opts ? opts->generator_user_data : NULL;
    st->stream_generator = opts && opts->stream_generator
                              ? opts->stream_generator : stub_stream_generator;
    st->stream_generator_user_data = opts ? opts->stream_generator_user_data : NULL;

    int rc;
    /* Single streaming handler that internally branches on stream=true.
     * This way buffered (Content-Length) and SSE (chunked) requests share
     * one route + one endpoint state with no double-registration risk. */
    rc = ds4cuda_http_server_register_stream(server, "/v1/chat/completions",
                                             chat_stream_handler, st);
    if (rc != 0) goto fail;
    rc = ds4cuda_http_server_register(server, "/v1/models", models_handler, st);
    if (rc != 0) goto fail;
    char idpath[256];
    snprintf(idpath, sizeof(idpath), "/v1/models/%s", st->model_id);
    rc = ds4cuda_http_server_register(server, idpath, models_handler, st);
    if (rc != 0) goto fail;
    return 0;
fail:
    free(st->model_id);
    free(st);
    return rc;
}
