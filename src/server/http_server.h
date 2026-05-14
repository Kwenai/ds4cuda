/*
 * http_server.h — minimal sockets-based HTTP/1.1 server for ds4cuda.
 *
 * Sequential single-threaded request loop, no SSE streaming yet (the
 * integrated build adds an inference-thread + chunked SSE channel to
 * the same listener). The shape mirrors the relevant subset of
 * antirez/ds4 ds4_server.c: parse request line + headers + Content-Length
 * body, dispatch by exact path, blocking send_all of one fully buffered
 * response. Connection: close on every reply (no keep-alive, no pipelining).
 *
 * Why hand-roll rather than libmicrohttpd / mongoose: keeps the build a
 * single compiler invocation (no third-party object), matches ds4 sockets
 * style exactly so the eventual streaming upgrade reads naturally, and the
 * total surface needed (one POST endpoint + GET /v1/models for sanity) is
 * < 200 LOC of socket code.
 *
 * Public API is C only. Handlers are invoked on the listener thread (the
 * server is single-connection / serial — see anthropic_endpoint.c +
 * inference_engine.cu for the FIFO worker model).
 */
#ifndef DS4CUDA_HTTP_SERVER_H
#define DS4CUDA_HTTP_SERVER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ds4cuda_http_server;

/*
 * Handler signature.
 *   method, path, body : NUL-terminated byte buffers; method/path live in
 *     a header arena owned by the caller, body is the request body (also
 *     NUL-terminated, but body_len is the authoritative length).
 *   out_body          : on return, a malloc()'d response body (or NULL for
 *     an empty body). The server takes ownership and free()s it.
 *   out_body_len      : explicit length so binary content (NUL bytes) can
 *     pass through untouched. If 0 and out_body != NULL, strlen() is used.
 *   out_status        : HTTP status code (200 / 400 / 404 / 500 / ...).
 *   out_content_type  : NUL-terminated string. Server strdup()s and frees.
 *     NULL -> "application/json".
 *   user_data         : opaque cookie passed to register().
 *
 * Return:
 *   0  success (send out_body with out_status).
 *   <0 the handler failed before it could fill out_body; the server emits
 *      a generic 500 "internal error" JSON and ignores the out_* fields.
 */
typedef int (*ds4cuda_http_handler_fn)(
    const char *method,
    const char *path,
    const char *body,
    size_t body_len,
    char **out_body,
    size_t *out_body_len,
    int *out_status,
    char **out_content_type,
    void *user_data);

/* Lifecycle. ds4cuda_http_server_create binds + listens on (127.0.0.1:port);
 * 0 success, <0 errno-style failure (port busy / EACCES / ...). The caller
 * is expected to call _run() afterwards. */
int  ds4cuda_http_server_create(struct ds4cuda_http_server **out, int port);
void ds4cuda_http_server_destroy(struct ds4cuda_http_server *s);

/* Register a handler. Match is currently exact path equality (no prefix
 * tree, no method-aware routing — the handler itself is responsible for
 * checking `method` and 405-ing if needed). The same path may not be
 * registered twice; the second call returns -1 without replacing.
 *
 * Path is strdup()'d, handler+user_data is stored by reference. */
int  ds4cuda_http_server_register(
        struct ds4cuda_http_server *s,
        const char *path,
        ds4cuda_http_handler_fn handler,
        void *user_data);

/* ----- Streaming (chunked HTTP/1.1, Transfer-Encoding: chunked) ----------
 *
 * The streaming model is a strict superset of the buffered single-response
 * dispatcher above:
 *
 *   1. The streaming handler is invoked with the parsed request just like
 *      the buffered one, but instead of filling out_body the handler is
 *      handed an opaque ds4cuda_http_stream_ctx tied to the live client
 *      socket. The handler MUST call ds4cuda_http_stream_begin() first to
 *      send the response status line + headers and open the chunked body.
 *
 *   2. The handler then calls ds4cuda_http_stream_write_chunk() N times to
 *      ship body fragments. Each call wraps the supplied bytes in a single
 *      RFC 7230 chunk: "<hex-size>\r\n<bytes>\r\n".
 *
 *   3. After the final chunk the handler calls ds4cuda_http_stream_end()
 *      which writes the zero-length terminator chunk ("0\r\n\r\n"). The
 *      server (not the handler) then closes the socket.
 *
 *   - If the same path has both a buffered and a streaming handler
 *     registered, the streaming handler wins. (We expect the streaming
 *     handler to internally branch on the request body — see
 *     openai_endpoint.c which dispatches on the JSON `stream` field.)
 */
struct ds4cuda_http_stream_ctx;

/* Send "HTTP/1.1 <status> <reason>\r\nContent-Type: ...\r\nCache-Control:
 * no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n".
 *
 * Must be called exactly once per stream context, before any write_chunk.
 * Returns 0 on success, <0 on socket failure.
 *
 * For SSE pass content_type = "text/event-stream". */
int ds4cuda_http_stream_begin(struct ds4cuda_http_stream_ctx *ctx,
                              int status, const char *content_type);

/* Write one chunked-transfer fragment. `len` may be 0 — that case is a
 * no-op (the spec reserves 0-length chunks for the terminator, so we
 * silently drop empties from the handler so it doesn't have to special
 * case zero-byte tokens).
 *
 * Returns 0 on success, <0 on socket failure (broken pipe / timeout). */
int ds4cuda_http_stream_write_chunk(struct ds4cuda_http_stream_ctx *ctx,
                                    const char *data, size_t len);

/* Send the "0\r\n\r\n" terminator. After this call the stream context is
 * spent — further write_chunk / end calls are no-ops returning -EINVAL.
 * Returns 0 on success, <0 on socket failure. */
int ds4cuda_http_stream_end(struct ds4cuda_http_stream_ctx *ctx);

/* Escape hatch: from inside a streaming handler, emit an ordinary
 * Content-Length-framed response and mark the context as fully consumed
 * (no chunked body, no terminator). Used when a streaming handler
 * decides at runtime — e.g. by inspecting JSON `stream` — that the
 * request is in fact a buffered one. After this returns the stream
 * context is "ended": subsequent stream_begin/write_chunk/end calls
 * fail with -EINVAL.
 *
 * Returns 0 on success, <0 on socket failure / misuse. */
int ds4cuda_http_stream_send_buffered(struct ds4cuda_http_stream_ctx *ctx,
                                      int status,
                                      const char *content_type,
                                      const char *body, size_t body_len);

/* Streaming handler signature. The contract:
 *
 *   - method/path/body have the same meaning as the buffered handler.
 *   - stream_ctx is owned by the server; the handler MUST call
 *     ds4cuda_http_stream_begin() and ds4cuda_http_stream_end() exactly
 *     once each (between any number of write_chunk calls).
 *   - Return 0 on success. A non-zero return causes the server to attempt
 *     to send a generic 500 if begin() was never called; if begin() was
 *     already called there is nothing the server can do (headers already
 *     out the door) so the connection is simply closed.
 */
typedef int (*ds4cuda_http_stream_handler_fn)(
    const char *method,
    const char *path,
    const char *body,
    size_t body_len,
    struct ds4cuda_http_stream_ctx *stream_ctx,
    void *user_data);

/* Register a streaming handler. Same path-collision rules as
 * ds4cuda_http_server_register; additionally a streaming handler may be
 * registered on a path that already has a buffered handler — the
 * streaming variant takes precedence at dispatch time. */
int  ds4cuda_http_server_register_stream(
        struct ds4cuda_http_server *s,
        const char *path,
        ds4cuda_http_stream_handler_fn handler,
        void *user_data);

/* Blocking accept loop. Returns when ds4cuda_http_server_stop() is called
 * (or on a fatal accept() error). */
int  ds4cuda_http_server_run(struct ds4cuda_http_server *s);

/* Async stop signal: closes the listening socket so accept() returns
 * EBADF, which the run loop interprets as a clean exit. Safe to call from
 * a signal handler or another thread. */
void ds4cuda_http_server_stop(struct ds4cuda_http_server *s);

/* The bound port. Useful when port=0 was passed (kernel-assigned). */
int  ds4cuda_http_server_port(const struct ds4cuda_http_server *s);

/* ----- In-process dispatch ---------------------------------------------- */

/* Run a single HTTP exchange against the supplied request line + body
 * directly through the in-process handler dispatch (no sockets).
 * Exercises the parsers without binding a port. status/out_body/
 * out_body_len are filled the same way as a network request would
 * receive them. */
int  ds4cuda_http_server_dispatch_in_process(
        struct ds4cuda_http_server *s,
        const char *method,
        const char *path,
        const char *body,
        size_t body_len,
        int *out_status,
        char **out_body,
        size_t *out_body_len,
        char **out_content_type);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_HTTP_SERVER_H */
