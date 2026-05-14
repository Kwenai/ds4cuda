/*
 * tool_calls.c — OpenAI / Anthropic tool-call schema validation plus
 * conversion between the wire-side function-call JSON and the DSML
 * <｜DSML｜tool_calls>...<invoke name="..."><parameter name="K">V</parameter>
 * ...</invoke>...</｜DSML｜tool_calls> form the model emits and consumes.
 *
 * Used by both openai_endpoint.c and anthropic_endpoint.c: incoming tool
 * specs are validated against the appropriate flavor; outgoing DSML
 * invoke blocks produced by the model are parsed back into OpenAI
 * tool_calls / Anthropic tool_use response content.
 */
#define _GNU_SOURCE
#include "tool_calls.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Local sbuf — tiny dynamic string.                                     */
/* ------------------------------------------------------------------ */
typedef struct { char *p; size_t len; size_t cap; } sbuf;

static void sbuf_reserve(sbuf *b, size_t extra)
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

static void sbuf_putc(sbuf *b, char c) { sbuf_reserve(b, 1); b->p[b->len++] = c; b->p[b->len] = '\0'; }
static void sbuf_puts(sbuf *b, const char *s) { if (!s) return; size_t n = strlen(s); sbuf_reserve(b, n); memcpy(b->p + b->len, s, n); b->len += n; b->p[b->len] = '\0'; }

static char *sbuf_take_or_empty(sbuf *b)
{
    if (b->p) return b->p;
    char *e = malloc(1);
    if (!e) return NULL;
    e[0] = '\0';
    return e;
}

/* ------------------------------------------------------------------ */
/* Tiny safe snprintf into err.                                          */
/* ------------------------------------------------------------------ */
static void seterr(char *err, size_t err_len, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

static void seterr(char *err, size_t err_len, const char *fmt, ...)
{
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

/* ------------------------------------------------------------------ */
/* Validator.                                                          */
/* ------------------------------------------------------------------ */
int ds4cuda_validate_openai_tools(const cjson *tools, char *err, size_t err_len)
{
    if (!tools) return 0;                       /* tools omitted -> ok */
    if (cjson_min_is_null(tools)) return 0;
    if (!cjson_min_is_array(tools)) {
        seterr(err, err_len, "tools must be an array");
        return -1;
    }
    int n = cjson_min_array_size(tools);
    for (int i = 0; i < n; i++) {
        const cjson *entry = cjson_min_array_item(tools, i);
        if (!cjson_min_is_object(entry)) {
            seterr(err, err_len, "tools[%d] must be an object", i);
            return -1;
        }
        const cjson *type = cjson_min_obj_get(entry, "type");
        if (type && cjson_min_is_string(type) &&
            strcmp(cjson_min_string(type), "function") != 0) {
            /* Anthropic-style direct schema (no "type":"function") is also
             * accepted to keep us forward-compatible with /v1/messages. */
            seterr(err, err_len, "tools[%d].type must be \"function\"", i);
            return -1;
        }
        const cjson *fn = cjson_min_obj_get(entry, "function");
        if (!fn) fn = entry;                  /* anthropic flat shape */
        if (!cjson_min_is_object(fn)) {
            seterr(err, err_len, "tools[%d].function missing or not object", i);
            return -1;
        }
        const cjson *name = cjson_min_obj_get(fn, "name");
        if (!cjson_min_is_string(name) || cjson_min_string(name)[0] == '\0') {
            seterr(err, err_len, "tools[%d].function.name missing", i);
            return -1;
        }
        /* parameters / input_schema are optional and free-form JSON Schema. */
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* DSML serialization — minimal "summary" line per tool. Replaced in     */
/* the integrated build with the full ds4_server.c append_tools_prompt_text */
/* boilerplate (which embeds reasoning_mode-aware system text).          */
/* ------------------------------------------------------------------ */
char *ds4cuda_dsml_render_tools(const cjson *openai_tools)
{
    sbuf b = {0};
    if (!openai_tools || !cjson_min_is_array(openai_tools)) return sbuf_take_or_empty(&b);
    int n = cjson_min_array_size(openai_tools);
    for (int i = 0; i < n; i++) {
        const cjson *entry = cjson_min_array_item(openai_tools, i);
        const cjson *fn = cjson_min_obj_get(entry, "function");
        if (!fn) fn = entry;
        const char *name = cjson_min_string(cjson_min_obj_get(fn, "name"));
        if (!name) continue;
        /* One JSON-flavored line per tool, matching ds4_server.c
         * openai_function_schema_from_tool's output shape. */
        char *json = cjson_min_emit(fn);
        if (json) { sbuf_puts(&b, json); sbuf_putc(&b, '\n'); free(json); }
    }
    return sbuf_take_or_empty(&b);
}

/* ------------------------------------------------------------------ */
/* DSML <- OpenAI assistant tool_calls.                                  */
/*                                                                      */
/* Each tool_call element looks like:                                    */
/*   {"id":"call_x","type":"function",                                   */
/*    "function":{"name":"X","arguments":"{\"k\":\"v\"}"}}               */
/* arguments is a *string* containing JSON (OpenAI quirk).               */
/* ------------------------------------------------------------------ */
/* The wide-bar U+FF5C encoded as 0xEF 0xBD 0x9C. We split after \x9c to keep
 * GCC from greedy-eating the next hex digit into the escape. */
static const char *DSML_OPEN   = "<\xef\xbd\x9c" "DSML\xef\xbd\x9c" "tool_calls>";
static const char *DSML_CLOSE  = "</\xef\xbd\x9c" "DSML\xef\xbd\x9c" "tool_calls>";
static const char *DSML_INV_O  = "<\xef\xbd\x9c" "DSML\xef\xbd\x9c" "invoke";
static const char *DSML_INV_C  = "</\xef\xbd\x9c" "DSML\xef\xbd\x9c" "invoke>";
static const char *DSML_PARM_O = "<\xef\xbd\x9c" "DSML\xef\xbd\x9c" "parameter";
static const char *DSML_PARM_C = "</\xef\xbd\x9c" "DSML\xef\xbd\x9c" "parameter>";

static void emit_one_invoke(sbuf *b, const char *name, const cjson *args_obj,
                            const char *raw_args_string)
{
    sbuf_puts(b, DSML_INV_O);
    sbuf_puts(b, " name=\"");
    sbuf_puts(b, name);
    sbuf_puts(b, "\">\n");
    if (args_obj && cjson_min_is_object(args_obj)) {
        for (cjson *p = args_obj->child; p; p = p->next) {
            sbuf_puts(b, DSML_PARM_O);
            sbuf_puts(b, " name=\"");
            sbuf_puts(b, p->string ? p->string : "");
            if (cjson_min_is_string(p)) {
                sbuf_puts(b, "\" string=\"true\">");
                sbuf_puts(b, cjson_min_string(p));
            } else {
                sbuf_puts(b, "\" string=\"false\">");
                char *json = cjson_min_emit(p);
                if (json) { sbuf_puts(b, json); free(json); }
            }
            sbuf_puts(b, DSML_PARM_C);
            sbuf_putc(b, '\n');
        }
    } else if (raw_args_string && raw_args_string[0]) {
        sbuf_puts(b, DSML_PARM_O);
        sbuf_puts(b, " name=\"args\" string=\"false\">");
        sbuf_puts(b, raw_args_string);
        sbuf_puts(b, DSML_PARM_C);
        sbuf_putc(b, '\n');
    }
    sbuf_puts(b, DSML_INV_C);
    sbuf_putc(b, '\n');
}

char *ds4cuda_dsml_render_assistant_tool_calls(const cjson *tool_calls_array)
{
    sbuf b = {0};
    if (!tool_calls_array || !cjson_min_is_array(tool_calls_array))
        return sbuf_take_or_empty(&b);
    int n = cjson_min_array_size(tool_calls_array);
    if (n == 0) return sbuf_take_or_empty(&b);
    sbuf_puts(&b, DSML_OPEN);
    sbuf_putc(&b, '\n');
    for (int i = 0; i < n; i++) {
        const cjson *e = cjson_min_array_item(tool_calls_array, i);
        const cjson *fn = cjson_min_obj_get(e, "function");
        if (!fn) fn = e;
        const char *name = cjson_min_string(cjson_min_obj_get(fn, "name"));
        if (!name) continue;
        const cjson *args = cjson_min_obj_get(fn, "arguments");
        cjson *parsed = NULL;
        const char *raw = NULL;
        if (cjson_min_is_string(args)) {
            raw = cjson_min_string(args);
            parsed = cjson_min_parse(raw);
            emit_one_invoke(&b, name, parsed, raw);
            cjson_min_free(parsed);
        } else if (cjson_min_is_object(args)) {
            emit_one_invoke(&b, name, args, NULL);
        } else {
            emit_one_invoke(&b, name, NULL, NULL);
        }
    }
    sbuf_puts(&b, DSML_CLOSE);
    sbuf_putc(&b, '\n');
    return sbuf_take_or_empty(&b);
}

/* ------------------------------------------------------------------ */
/* OpenAI <- DSML  parser.                                               */
/*                                                                       */
/* Linear scan for the first <invoke ... > ... </invoke> fragment. Each  */
/* <parameter name="K" string="B">V</parameter> contributes a key/value  */
/* pair. The arguments object is built as a JSON object and stringified  */
/* per the OpenAI spec (function.arguments is a string).                 */
/* ------------------------------------------------------------------ */
static const char *find_lit(const char *hay, const char *end, const char *needle)
{
    size_t nlen = strlen(needle);
    if (!nlen) return hay;
    for (const char *p = hay; p + nlen <= end; p++) {
        if (memcmp(p, needle, nlen) == 0) return p;
    }
    return NULL;
}

/* Read an attribute value: name="..."  -> writes value into `out` (capped). */
static bool read_attr(const char *hay, const char *end, const char *attr,
                      char *out, size_t out_cap)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=\"", attr);
    const char *q = find_lit(hay, end, pat);
    if (!q) return false;
    q += strlen(pat);
    const char *r = q;
    while (r < end && *r != '"') r++;
    if (r >= end) return false;
    size_t n = (size_t)(r - q);
    if (n + 1 > out_cap) n = out_cap - 1;
    memcpy(out, q, n);
    out[n] = '\0';
    return true;
}

char *ds4cuda_openai_render_tool_call(const char *dsml_text)
{
    if (!dsml_text) return NULL;
    const char *s = dsml_text;
    const char *e = s + strlen(s);

    const char *inv = find_lit(s, e, DSML_INV_O);
    if (!inv) return NULL;
    const char *gt = inv;
    while (gt < e && *gt != '>') gt++;
    if (gt >= e) return NULL;
    char name[128] = {0};
    if (!read_attr(inv, gt, "name", name, sizeof(name))) return NULL;

    const char *body = gt + 1;
    const char *inv_end = find_lit(body, e, DSML_INV_C);
    if (!inv_end) inv_end = e;

    cjson *args = cjson_min_new_object();
    const char *p = body;
    while (p < inv_end) {
        const char *po = find_lit(p, inv_end, DSML_PARM_O);
        if (!po) break;
        const char *pgt = po;
        while (pgt < inv_end && *pgt != '>') pgt++;
        if (pgt >= inv_end) break;
        char pname[128] = {0};
        char pstring[16] = {0};
        if (!read_attr(po, pgt, "name", pname, sizeof(pname))) {
            p = pgt + 1; continue;
        }
        bool is_string = true;
        if (read_attr(po, pgt, "string", pstring, sizeof(pstring))) {
            if (!strcmp(pstring, "false")) is_string = false;
        }
        const char *pbody = pgt + 1;
        const char *pc = find_lit(pbody, inv_end, DSML_PARM_C);
        if (!pc) break;
        size_t n = (size_t)(pc - pbody);
        char *val = malloc(n + 1);
        if (!val) break;
        memcpy(val, pbody, n); val[n] = '\0';
        cjson *node = NULL;
        if (is_string) {
            node = cjson_min_new_string(val);
        } else {
            node = cjson_min_parse(val);
            if (!node) node = cjson_min_new_string(val);
        }
        free(val);
        cjson_min_object_set(args, pname, node);
        p = pc + strlen(DSML_PARM_C);
    }

    /* Build the wrapper {"id":..., "type":"function", "function":{name, arguments}} */
    char *args_json = cjson_min_emit(args);
    cjson_min_free(args);

    cjson *fn = cjson_min_new_object();
    cjson_min_object_set(fn, "name", cjson_min_new_string(name));
    cjson_min_object_set(fn, "arguments", cjson_min_new_string(args_json ? args_json : "{}"));
    free(args_json);

    cjson *wrap = cjson_min_new_object();
    cjson_min_object_set(wrap, "id", cjson_min_new_string("call_dsml_0"));
    cjson_min_object_set(wrap, "type", cjson_min_new_string("function"));
    cjson_min_object_set(wrap, "function", fn);

    char *out = cjson_min_emit(wrap);
    cjson_min_free(wrap);
    return out;
}

/* ------------------------------------------------------------------ */
/* Anthropic flavor.                                                    */
/*                                                                      */
/* Anthropic tool entries are flat ({name, description, input_schema}); */
/* OpenAI entries are nested ({type:"function", function:{...}}). The   */
/* validator + DSML renderer reuse the same per-entry "name+schema"     */
/* extraction; only the wrapper differs.                                */
/* ------------------------------------------------------------------ */
int ds4cuda_validate_anthropic_tools(const cjson *tools, char *err, size_t err_len)
{
    if (!tools) return 0;
    if (cjson_min_is_null(tools)) return 0;
    if (!cjson_min_is_array(tools)) {
        seterr(err, err_len, "tools must be an array");
        return -1;
    }
    int n = cjson_min_array_size(tools);
    for (int i = 0; i < n; i++) {
        const cjson *entry = cjson_min_array_item(tools, i);
        if (!cjson_min_is_object(entry)) {
            seterr(err, err_len, "tools[%d] must be an object", i);
            return -1;
        }
        const cjson *name = cjson_min_obj_get(entry, "name");
        if (!cjson_min_is_string(name) || cjson_min_string(name)[0] == '\0') {
            seterr(err, err_len, "tools[%d].name missing", i);
            return -1;
        }
        const cjson *schema = cjson_min_obj_get(entry, "input_schema");
        if (schema && !cjson_min_is_object(schema)) {
            seterr(err, err_len, "tools[%d].input_schema must be an object", i);
            return -1;
        }
    }
    return 0;
}

char *ds4cuda_dsml_render_anthropic_tools(const cjson *anthropic_tools)
{
    sbuf b = {0};
    if (!anthropic_tools || !cjson_min_is_array(anthropic_tools))
        return sbuf_take_or_empty(&b);
    int n = cjson_min_array_size(anthropic_tools);
    for (int i = 0; i < n; i++) {
        const cjson *entry = cjson_min_array_item(anthropic_tools, i);
        if (!cjson_min_is_object(entry)) continue;
        const char *name = cjson_min_string(cjson_min_obj_get(entry, "name"));
        if (!name) continue;
        /* Build a normalized schema object for emission: project the
         * Anthropic shape onto the same {name, description, parameters}
         * fields the OpenAI emitter prints, so downstream integrated builds
         * sees one flavor of schema list. */
        cjson *norm = cjson_min_new_object();
        cjson_min_object_set(norm, "name", cjson_min_new_string(name));
        const cjson *desc = cjson_min_obj_get(entry, "description");
        if (cjson_min_is_string(desc)) {
            cjson_min_object_set(norm, "description",
                                 cjson_min_new_string(cjson_min_string(desc)));
        }
        const cjson *schema = cjson_min_obj_get(entry, "input_schema");
        if (schema && cjson_min_is_object(schema)) {
            char *sj = cjson_min_emit(schema);
            if (sj) {
                cjson *clone = cjson_min_parse(sj);
                if (clone) cjson_min_object_set(norm, "parameters", clone);
                free(sj);
            }
        }
        char *json = cjson_min_emit(norm);
        cjson_min_free(norm);
        if (json) { sbuf_puts(&b, json); sbuf_putc(&b, '\n'); free(json); }
    }
    return sbuf_take_or_empty(&b);
}

char *ds4cuda_dsml_render_anthropic_tool_use(const cjson *block)
{
    sbuf b = {0};
    if (!block || !cjson_min_is_object(block)) return sbuf_take_or_empty(&b);
    const char *type = cjson_min_string(cjson_min_obj_get(block, "type"));
    if (!type || strcmp(type, "tool_use") != 0) return sbuf_take_or_empty(&b);
    const char *name = cjson_min_string(cjson_min_obj_get(block, "name"));
    if (!name) return sbuf_take_or_empty(&b);
    const cjson *input = cjson_min_obj_get(block, "input");
    sbuf_puts(&b, DSML_OPEN);
    sbuf_putc(&b, '\n');
    emit_one_invoke(&b, name, cjson_min_is_object(input) ? input : NULL, NULL);
    sbuf_puts(&b, DSML_CLOSE);
    sbuf_putc(&b, '\n');
    return sbuf_take_or_empty(&b);
}

char *ds4cuda_dsml_render_anthropic_assistant_blocks(const cjson *content_array)
{
    sbuf b = {0};
    if (!content_array || !cjson_min_is_array(content_array))
        return sbuf_take_or_empty(&b);
    int n = cjson_min_array_size(content_array);
    /* First pass: text concatenation. */
    for (int i = 0; i < n; i++) {
        const cjson *item = cjson_min_array_item(content_array, i);
        if (!cjson_min_is_object(item)) continue;
        const char *type = cjson_min_string(cjson_min_obj_get(item, "type"));
        if (!type) continue;
        if (!strcmp(type, "text")) {
            const char *t = cjson_min_string(cjson_min_obj_get(item, "text"));
            if (t) sbuf_puts(&b, t);
        }
    }
    /* Second pass: collect tool_use into a single DSML block. */
    bool any_tool = false;
    for (int i = 0; i < n; i++) {
        const cjson *item = cjson_min_array_item(content_array, i);
        if (!cjson_min_is_object(item)) continue;
        const char *type = cjson_min_string(cjson_min_obj_get(item, "type"));
        if (type && !strcmp(type, "tool_use")) { any_tool = true; break; }
    }
    if (any_tool) {
        sbuf_puts(&b, DSML_OPEN);
        sbuf_putc(&b, '\n');
        for (int i = 0; i < n; i++) {
            const cjson *item = cjson_min_array_item(content_array, i);
            if (!cjson_min_is_object(item)) continue;
            const char *type = cjson_min_string(cjson_min_obj_get(item, "type"));
            if (!type || strcmp(type, "tool_use") != 0) continue;
            const char *name = cjson_min_string(cjson_min_obj_get(item, "name"));
            if (!name) continue;
            const cjson *input = cjson_min_obj_get(item, "input");
            emit_one_invoke(&b, name,
                            cjson_min_is_object(input) ? input : NULL, NULL);
        }
        sbuf_puts(&b, DSML_CLOSE);
        sbuf_putc(&b, '\n');
    }
    return sbuf_take_or_empty(&b);
}

char *ds4cuda_anthropic_render_tool_use(const char *dsml_text)
{
    if (!dsml_text) return NULL;
    const char *s = dsml_text;
    const char *e = s + strlen(s);
    const char *inv = find_lit(s, e, DSML_INV_O);
    if (!inv) return NULL;
    const char *gt = inv;
    while (gt < e && *gt != '>') gt++;
    if (gt >= e) return NULL;
    char name[128] = {0};
    if (!read_attr(inv, gt, "name", name, sizeof(name))) return NULL;
    const char *body = gt + 1;
    const char *inv_end = find_lit(body, e, DSML_INV_C);
    if (!inv_end) inv_end = e;

    cjson *input = cjson_min_new_object();
    const char *p = body;
    while (p < inv_end) {
        const char *po = find_lit(p, inv_end, DSML_PARM_O);
        if (!po) break;
        const char *pgt = po;
        while (pgt < inv_end && *pgt != '>') pgt++;
        if (pgt >= inv_end) break;
        char pname[128] = {0};
        char pstring[16] = {0};
        if (!read_attr(po, pgt, "name", pname, sizeof(pname))) {
            p = pgt + 1; continue;
        }
        bool is_string = true;
        if (read_attr(po, pgt, "string", pstring, sizeof(pstring))) {
            if (!strcmp(pstring, "false")) is_string = false;
        }
        const char *pbody = pgt + 1;
        const char *pc = find_lit(pbody, inv_end, DSML_PARM_C);
        if (!pc) break;
        size_t n = (size_t)(pc - pbody);
        char *val = malloc(n + 1);
        if (!val) break;
        memcpy(val, pbody, n); val[n] = '\0';
        cjson *node = NULL;
        if (is_string) {
            node = cjson_min_new_string(val);
        } else {
            node = cjson_min_parse(val);
            if (!node) node = cjson_min_new_string(val);
        }
        free(val);
        cjson_min_object_set(input, pname, node);
        p = pc + strlen(DSML_PARM_C);
    }

    /* Build {"type":"tool_use","id":"toolu_dsml_0","name":"X","input":{...}} */
    cjson *wrap = cjson_min_new_object();
    cjson_min_object_set(wrap, "type", cjson_min_new_string("tool_use"));
    cjson_min_object_set(wrap, "id",   cjson_min_new_string("toolu_dsml_0"));
    cjson_min_object_set(wrap, "name", cjson_min_new_string(name));
    cjson_min_object_set(wrap, "input", input);

    char *out = cjson_min_emit(wrap);
    cjson_min_free(wrap);
    return out;
}
