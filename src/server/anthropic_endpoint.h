/*
 * anthropic_endpoint.h — POST /v1/messages mounted on a ds4cuda_http_server.
 *
 * Anthropic-compatible cousin of openai_endpoint.{c,h}. The handler:
 *
 *   1. Parses an Anthropic /v1/messages request body (top-level "system"
 *      string-or-array, "messages" array of {role, content} where content
 *      may itself be a string or an array of typed blocks: text, tool_use,
 *      tool_result, thinking).
 *   2. Reduces those blocks to a flat (role, content) message vector + a
 *      system text string, then renders the DSML prompt via
 *      ds4cuda_render_chat_prompt_with_system_alloc.
 *   3. Validates Anthropic-flavored "tools" (no "type":"function" wrapper).
 *   4. Invokes the caller-installed generator, which takes the rendered
 *      DSML prompt + the flattened tool spec text and returns either a
 *      plain assistant text reply or a raw
 *      <｜DSML｜tool_calls>...</｜DSML｜tool_calls> blob. When the latter is
 *      detected, ds4cuda_anthropic_render_tool_use parses it and the
 *      content is spliced into the response as tool_use blocks.
 *   5. Serializes an Anthropic /v1/messages response object
 *      ({id, type:"message", role:"assistant", content:[...], model,
 *        stop_reason, stop_sequence, usage:{input_tokens, output_tokens}}).
 *
 * Streaming (stream=true) is rejected with 400 — SSE upgrade is currently
 * OpenAI-only.
 *
 * The endpoint is host C only, decoupled from inference. The default
 * generator returns "OK\n" so the route is exercisable without a model;
 * production builds install an inference-engine-backed generator.
 */
#ifndef DS4CUDA_ANTHROPIC_ENDPOINT_H
#define DS4CUDA_ANTHROPIC_ENDPOINT_H

#include <stdbool.h>
#include <stddef.h>

#include "http_server.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Anthropic generator signature.
 *   dsml_prompt : DSML-rendered prompt text. NUL-terminated. NOT owned.
 *   tools_text  : DSML-rendered tools schema list (one JSON line per tool;
 *                 empty if no tools). NUL-terminated. NOT owned.
 *   max_new_tokens : ceiling from request (or default).
 *   user_data : opaque cookie passed at install time.
 *   out_text : on success, malloc()'d UTF-8 string. The endpoint takes
 *              ownership and frees with free(). May be empty. May contain
 *              a literal "<｜DSML｜tool_calls>...</｜DSML｜tool_calls>" block
 *              (the endpoint will parse it into tool_use content blocks).
 * Returns 0 on success, <0 on refusal (the endpoint replies 500). */
typedef int (*ds4cuda_anthropic_generator_fn)(
    const char *dsml_prompt,
    const char *tools_text,
    int max_new_tokens,
    void *user_data,
    char **out_text);

struct ds4cuda_anthropic_endpoint_options {
    /* Model name string echoed back in responses. Default "deepseek-v4-flash". */
    const char *model_id;
    /* Default max_tokens when the request omits it. Default 256. */
    int default_max_tokens;
    /* Optional generator. NULL -> built-in fallback returning "OK\n". */
    ds4cuda_anthropic_generator_fn generator;
    void *generator_user_data;
};

/* Install POST /v1/messages onto an existing server. The endpoint keeps a
 * heap-allocated copy of `opts` for the lifetime of the server. Returns 0
 * on success, <0 on register failure. */
int ds4cuda_anthropic_endpoint_install(
        struct ds4cuda_http_server *server,
        const struct ds4cuda_anthropic_endpoint_options *opts);

/* ----- Hand-build a /v1/messages response without going through the HTTP
 * layer. Exercises the JSON parser + chat template + tool-call paths
 * without binding sockets. */
int ds4cuda_anthropic_handle_messages(
        const struct ds4cuda_anthropic_endpoint_options *opts,
        const char *body, size_t body_len,
        char **out_body, size_t *out_body_len, int *out_status);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_ANTHROPIC_ENDPOINT_H */
