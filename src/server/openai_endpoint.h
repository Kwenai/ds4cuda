/*
 * openai_endpoint.h — OpenAI /v1/chat/completions and /v1/models endpoints
 * backed by the inference engine, mounted onto a ds4cuda_http_server.
 *
 * The chat completion handler parses the request, validates tools, renders
 * the DSML prompt, and dispatches to a caller-supplied generator function
 * for the actual token production. The buffered path returns one JSON
 * response; the streaming path drives an SSE chunked response.
 *
 * The endpoint is host C only — no CUDA dependency. Production builds
 * install an inference-engine-backed generator at startup; tests install
 * deterministic generators to exercise the JSON / chat-template / tool-call
 * paths without binding a GPU.
 */
#ifndef DS4CUDA_OPENAI_ENDPOINT_H
#define DS4CUDA_OPENAI_ENDPOINT_H

#include <stdbool.h>
#include <stddef.h>

#include "http_server.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Generator signature.
 *   prompt : DSML-rendered prompt text (NUL-terminated UTF-8). NOT owned.
 *   max_new_tokens : ceiling from the request (or default if absent).
 *   user_data : opaque cookie passed at install time.
 *   out_text : on success, malloc()'d string with the assistant content.
 *              The endpoint takes ownership and frees with free().
 * Returns 0 on success, <0 to signal "I refused" (the endpoint will reply
 * 500). */
typedef int (*ds4cuda_chat_generator_fn)(
    const char *prompt,
    int max_new_tokens,
    void *user_data,
    char **out_text);

/* Streaming generator signature. Invoked once per request when stream=true;
 * the implementation calls `emit(token, emit_user_data)` for each output
 * token, then returns. Pass len=-1 to take strlen of the token; len>=0
 * passes a counted slice (zero-length tokens are dropped by the endpoint).
 *
 * Returns 0 on success, <0 if generation aborted mid-stream — the endpoint
 * will close the SSE channel without a [DONE] sentinel in that case. */
typedef int (*ds4cuda_chat_stream_emit_fn)(const char *token, int len,
                                           void *emit_user_data);

typedef int (*ds4cuda_chat_stream_generator_fn)(
    const char *prompt,
    int max_new_tokens,
    void *user_data,
    ds4cuda_chat_stream_emit_fn emit,
    void *emit_user_data);

struct ds4cuda_chat_endpoint_options {
    /* Model name string echoed back in responses. Default "deepseek-v4-flash". */
    const char *model_id;
    /* Default max_tokens when the request omits it. Default 256. */
    int default_max_tokens;
    /* Optional generator callback. NULL -> built-in fallback that returns
     * the string "OK\n" so the route is exercisable without a model. */
    ds4cuda_chat_generator_fn generator;
    void *generator_user_data;
    /* Optional streaming generator. Called when the request body contains
     * stream=true. NULL -> built-in fallback that emits "OK\n" split into
     * three SSE chunks ("O", "K", "\n"), simulating token-by-token
     * delivery. */
    ds4cuda_chat_stream_generator_fn stream_generator;
    void *stream_generator_user_data;
};

/* Install the OpenAI endpoint(s) onto an existing server. The endpoint
 * keeps a heap-allocated copy of `opts` for the lifetime of the server.
 * Returns 0 on success, <0 on register failure (path already taken). */
int ds4cuda_openai_endpoint_install(
        struct ds4cuda_http_server *server,
        const struct ds4cuda_chat_endpoint_options *opts);

/* ----- In-process: hand-build a chat completion response without going
 * through the HTTP layer. Exercises the JSON parser + chat template +
 * tool-call paths without binding sockets. The caller passes a JSON
 * request body; the function writes a JSON response body (malloc()'d)
 * and a status code. */
int ds4cuda_openai_handle_chat_completion(
        const struct ds4cuda_chat_endpoint_options *opts,
        const char *body, size_t body_len,
        char **out_body, size_t *out_body_len, int *out_status);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_OPENAI_ENDPOINT_H */
