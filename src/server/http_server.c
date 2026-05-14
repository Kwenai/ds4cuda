/*
 * http_server.c — implementation of the minimal HTTP server.
 *
 * Parsing strategy is a direct adaptation of antirez/ds4 ds4_server.c:
 *
 *   1. recv() into a growable buffer until "\r\n\r\n" (header end).
 *   2. sscanf the first line for method + path; query string stripped.
 *   3. scan headers for Content-Length (case-insensitive).
 *   4. recv() until the body has been fully received.
 *   5. dispatch to a registered handler keyed on exact path; on no match,
 *      emit 404.
 *   6. write a plain non-chunked HTTP/1.1 response with Connection: close.
 *
 * Stream / SSE responses are handled by the endpoint handler itself, not
 * by this layer — the OpenAI /v1/chat/completions handler takes ownership
 * of the connection fd after the request line + headers are parsed and
 * writes its own chunked-transfer or text/event-stream frames.
 */
#define _GNU_SOURCE
#include "http_server.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Local xmalloc / sbuf — duplicated from cjson_min.c on purpose so we    */
/* don't pull in the JSON layer for pure HTTP transport.                  */
/* ------------------------------------------------------------------ */
static void *xmalloc_(size_t n)
{
    if (!n) n = 1;
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "http_server: out of memory (alloc %zu B)\n", n);
        abort();
    }
    return p;
}

static void *xrealloc_(void *p, size_t n)
{
    if (!n) n = 1;
    void *q = realloc(p, n);
    if (!q) {
        fprintf(stderr, "http_server: out of memory (realloc %zu B)\n", n);
        abort();
    }
    return q;
}

static char *xstrdup_(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = xmalloc_(n);
    memcpy(p, s, n);
    return p;
}

typedef struct {
    char *p;
    size_t len;
    size_t cap;
} sbuf;

static void sbuf_append(sbuf *b, const void *data, size_t n)
{
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 256;
        while (cap < b->len + n + 1) cap *= 2;
        b->p = xrealloc_(b->p, cap);
        b->cap = cap;
    }
    memcpy(b->p + b->len, data, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void sbuf_free(sbuf *b)
{
    free(b->p);
    memset(b, 0, sizeof(*b));
}

/* ------------------------------------------------------------------ */
/* Server state.                                                       */
/* ------------------------------------------------------------------ */
#define MAX_ROUTES 16

struct route {
    char *path;
    /* Either `handler` or `stream_handler` is non-NULL; both may be set
     * (a buffered fallback alongside a streaming handler). At dispatch
     * time the streaming handler wins if present. */
    ds4cuda_http_handler_fn        handler;
    ds4cuda_http_stream_handler_fn stream_handler;
    void *user_data;
    void *stream_user_data;
};

struct ds4cuda_http_server {
    int listen_fd;
    int port;
    volatile int stopping;
    struct route routes[MAX_ROUTES];
    int n_routes;
};

/* Streaming context handed to a stream handler. Lives on the listener
 * stack inside handle_one(); the handler must NOT retain the pointer
 * beyond its own return. */
struct ds4cuda_http_stream_ctx {
    int    fd;
    int    began;     /* set once stream_begin has emitted headers */
    int    ended;     /* set once stream_end has emitted terminator */
    int    failed;    /* sticky error so write_chunk after broken send no-ops */
};

int ds4cuda_http_server_port(const struct ds4cuda_http_server *s)
{
    return s ? s->port : -1;
}

/* ------------------------------------------------------------------ */
/* Listener setup. localhost-only; we don't want random LAN access to a   */
/* minimal HTTP server.                                                   */
/* ------------------------------------------------------------------ */
int ds4cuda_http_server_create(struct ds4cuda_http_server **out, int port)
{
    if (!out) return -EINVAL;
    *out = NULL;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -errno;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port   = htons((uint16_t)port);
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        int e = errno;
        close(fd);
        return -e;
    }
    if (listen(fd, 16) != 0) {
        int e = errno;
        close(fd);
        return -e;
    }
    /* Read back the actual port (handles port=0 ephemeral assignment). */
    socklen_t slen = sizeof(sa);
    if (getsockname(fd, (struct sockaddr *)&sa, &slen) == 0) {
        port = ntohs(sa.sin_port);
    }

    struct ds4cuda_http_server *s = xmalloc_(sizeof(*s));
    memset(s, 0, sizeof(*s));
    s->listen_fd = fd;
    s->port = port;
    *out = s;
    return 0;
}

void ds4cuda_http_server_destroy(struct ds4cuda_http_server *s)
{
    if (!s) return;
    if (s->listen_fd >= 0) close(s->listen_fd);
    for (int i = 0; i < s->n_routes; i++) free(s->routes[i].path);
    free(s);
}

int ds4cuda_http_server_register(
        struct ds4cuda_http_server *s,
        const char *path,
        ds4cuda_http_handler_fn handler,
        void *user_data)
{
    if (!s || !path || !handler) return -EINVAL;
    /* If a streaming handler already exists for this path, attach the
     * buffered handler to the same route slot rather than 409-ing. This
     * lets a caller install both halves under one path. */
    for (int i = 0; i < s->n_routes; i++) {
        if (!strcmp(s->routes[i].path, path)) {
            if (s->routes[i].handler) return -EEXIST;
            s->routes[i].handler = handler;
            s->routes[i].user_data = user_data;
            return 0;
        }
    }
    if (s->n_routes >= MAX_ROUTES) return -ENOSPC;
    struct route *r = &s->routes[s->n_routes++];
    memset(r, 0, sizeof(*r));
    r->path = xstrdup_(path);
    r->handler = handler;
    r->user_data = user_data;
    return 0;
}

int ds4cuda_http_server_register_stream(
        struct ds4cuda_http_server *s,
        const char *path,
        ds4cuda_http_stream_handler_fn handler,
        void *user_data)
{
    if (!s || !path || !handler) return -EINVAL;
    for (int i = 0; i < s->n_routes; i++) {
        if (!strcmp(s->routes[i].path, path)) {
            if (s->routes[i].stream_handler) return -EEXIST;
            s->routes[i].stream_handler = handler;
            s->routes[i].stream_user_data = user_data;
            return 0;
        }
    }
    if (s->n_routes >= MAX_ROUTES) return -ENOSPC;
    struct route *r = &s->routes[s->n_routes++];
    memset(r, 0, sizeof(*r));
    r->path = xstrdup_(path);
    r->stream_handler = handler;
    r->stream_user_data = user_data;
    return 0;
}

void ds4cuda_http_server_stop(struct ds4cuda_http_server *s)
{
    if (!s) return;
    s->stopping = 1;
    if (s->listen_fd >= 0) {
        shutdown(s->listen_fd, SHUT_RDWR);
        close(s->listen_fd);
        s->listen_fd = -1;
    }
}

/* ------------------------------------------------------------------ */
/* Wire I/O.                                                           */
/* ------------------------------------------------------------------ */
static bool send_all(int fd, const void *data, size_t n)
{
    const char *p = data;
    while (n) {
        ssize_t w = send(fd, p, n, 0);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        p += w; n -= (size_t)w;
    }
    return true;
}

static const char *status_reason(int code)
{
    switch (code) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default:  return "Error";
    }
}

static bool send_response(int fd,
                          int code,
                          const char *content_type,
                          const char *body, size_t body_len)
{
    if (!content_type) content_type = "application/json";
    if (!body) { body = ""; body_len = 0; }
    char hdr[256];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, status_reason(code), content_type, body_len);
    if (n < 0 || (size_t)n >= sizeof(hdr)) return false;
    if (!send_all(fd, hdr, (size_t)n)) return false;
    if (body_len) return send_all(fd, body, body_len);
    return true;
}

static bool send_error(int fd, int code, const char *message)
{
    char body[256];
    /* Inline tiny json — the chat endpoint emits richer error JSON via
     * cjson_min, this path is only for transport-layer failures. */
    int n = snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}\n",
        message ? message : "request failed");
    if (n < 0) n = 0;
    if ((size_t)n >= sizeof(body)) n = sizeof(body) - 1;
    return send_response(fd, code, "application/json", body, (size_t)n);
}

/* ------------------------------------------------------------------ */
/* Streaming (chunked HTTP/1.1) public API.                              */
/* ------------------------------------------------------------------ */
int ds4cuda_http_stream_begin(struct ds4cuda_http_stream_ctx *ctx,
                              int status, const char *content_type)
{
    if (!ctx || ctx->began || ctx->ended) return -EINVAL;
    if (!content_type) content_type = "text/event-stream";
    char hdr[384];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n",
        status, status_reason(status), content_type);
    if (n < 0 || (size_t)n >= sizeof(hdr)) {
        ctx->failed = 1;
        return -EINVAL;
    }
    if (!send_all(ctx->fd, hdr, (size_t)n)) {
        ctx->failed = 1;
        return -EIO;
    }
    ctx->began = 1;
    return 0;
}

int ds4cuda_http_stream_write_chunk(struct ds4cuda_http_stream_ctx *ctx,
                                    const char *data, size_t len)
{
    if (!ctx || !ctx->began || ctx->ended) return -EINVAL;
    if (ctx->failed) return -EIO;
    /* Drop empties — the spec reserves "0\r\n\r\n" exclusively for the
     * terminator, so a 0-length write here would prematurely end the
     * body. The handler is allowed to call us with len==0 and we just
     * ignore it. */
    if (len == 0 || !data) return 0;

    char size_line[32];
    int sn = snprintf(size_line, sizeof(size_line), "%zx\r\n", len);
    if (sn < 0 || (size_t)sn >= sizeof(size_line)) {
        ctx->failed = 1;
        return -EINVAL;
    }
    if (!send_all(ctx->fd, size_line, (size_t)sn)) goto io_fail;
    if (!send_all(ctx->fd, data, len))             goto io_fail;
    if (!send_all(ctx->fd, "\r\n", 2))             goto io_fail;
    return 0;
io_fail:
    ctx->failed = 1;
    return -EIO;
}

int ds4cuda_http_stream_end(struct ds4cuda_http_stream_ctx *ctx)
{
    if (!ctx || !ctx->began || ctx->ended) return -EINVAL;
    ctx->ended = 1;
    if (ctx->failed) return -EIO;
    if (!send_all(ctx->fd, "0\r\n\r\n", 5)) {
        ctx->failed = 1;
        return -EIO;
    }
    return 0;
}

int ds4cuda_http_stream_send_buffered(struct ds4cuda_http_stream_ctx *ctx,
                                      int status,
                                      const char *content_type,
                                      const char *body, size_t body_len)
{
    if (!ctx) return -EINVAL;
    if (ctx->began || ctx->ended) return -EINVAL;
    /* Mark the context as fully spent regardless of send outcome so
     * handle_one() doesn't try to slap a chunked terminator on top. */
    ctx->began = 1;
    ctx->ended = 1;
    if (ctx->failed) return -EIO;
    if (!send_response(ctx->fd, status, content_type, body, body_len)) {
        ctx->failed = 1;
        return -EIO;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Request parsing.                                                    */
/* ------------------------------------------------------------------ */
static ssize_t find_header_end(const char *buf, size_t n)
{
    /* Look for "\r\n\r\n" or "\n\n". */
    if (n < 2) return -1;
    for (size_t i = 3; i < n; i++) {
        if (buf[i - 3] == '\r' && buf[i - 2] == '\n' &&
            buf[i - 1] == '\r' && buf[i]     == '\n') return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (buf[i - 1] == '\n' && buf[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

static long header_content_length(const char *h, size_t n)
{
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            return strtol(v, NULL, 10);
        }
        if (p < end) p++;
    }
    return 0;
}

typedef struct {
    char method[16];
    char path[512];
    char *body;
    size_t body_len;
} parsed_request;

static void parsed_request_free(parsed_request *r)
{
    if (!r) return;
    free(r->body);
    memset(r, 0, sizeof(*r));
}

static int read_request(int fd, parsed_request *out)
{
    sbuf b = {0};
    ssize_t hend = -1;
    const size_t MAX_HEADER = 64 * 1024;
    const size_t MAX_BODY   = 16 * 1024 * 1024;

    while (hend < 0 && b.len < MAX_HEADER) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        sbuf_append(&b, tmp, (size_t)n);
        hend = find_header_end(b.p, b.len);
    }
    if (hend < 0) goto fail;

    /* request line */
    char line[768];
    size_t i = 0;
    while (i < b.len && b.p[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.p[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%15s %511s", out->method, out->path) != 2) goto fail;
    /* strip query string */
    char *q = strchr(out->path, '?');
    if (q) *q = '\0';

    /* body */
    long clen = header_content_length(b.p, (size_t)hend);
    if (clen < 0 || (size_t)clen > MAX_BODY) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        sbuf_append(&b, tmp, (size_t)n);
    }
    out->body_len = (size_t)clen;
    out->body = xmalloc_(out->body_len + 1);
    memcpy(out->body, b.p + hend, out->body_len);
    out->body[out->body_len] = '\0';
    sbuf_free(&b);
    return 0;
fail:
    sbuf_free(&b);
    return -1;
}

/* ------------------------------------------------------------------ */
/* Dispatch.                                                           */
/* ------------------------------------------------------------------ */
static const struct route *find_route(const struct ds4cuda_http_server *s,
                                      const char *path)
{
    for (int i = 0; i < s->n_routes; i++) {
        if (!strcmp(s->routes[i].path, path)) return &s->routes[i];
    }
    return NULL;
}

int ds4cuda_http_server_dispatch_in_process(
        struct ds4cuda_http_server *s,
        const char *method,
        const char *path,
        const char *body,
        size_t body_len,
        int *out_status,
        char **out_body,
        size_t *out_body_len,
        char **out_content_type)
{
    if (!s || !method || !path || !out_status || !out_body || !out_body_len)
        return -EINVAL;
    *out_status = 0;
    *out_body = NULL;
    *out_body_len = 0;
    if (out_content_type) *out_content_type = NULL;

    const struct route *r = find_route(s, path);
    if (!r || !r->handler) {
        /* In-process dispatch is buffered-only; a route registered with
         * only a streaming handler is invisible to this path (callers
         * exercising stream paths drive sockets directly in tests). */
        *out_status = 404;
        const char *msg = "{\"error\":{\"message\":\"unknown endpoint\",\"type\":\"invalid_request_error\"}}\n";
        *out_body = xstrdup_(msg);
        *out_body_len = strlen(msg);
        if (out_content_type) *out_content_type = xstrdup_("application/json");
        return 0;
    }
    const char *bb = body ? body : "";
    int rc = r->handler(method, path, bb, body_len,
                        out_body, out_body_len, out_status,
                        out_content_type, r->user_data);
    if (rc != 0) {
        free(*out_body); *out_body = NULL; *out_body_len = 0;
        if (out_content_type) { free(*out_content_type); *out_content_type = NULL; }
        const char *msg = "{\"error\":{\"message\":\"handler error\",\"type\":\"server_error\"}}\n";
        *out_status = 500;
        *out_body = xstrdup_(msg);
        *out_body_len = strlen(msg);
        if (out_content_type) *out_content_type = xstrdup_("application/json");
        return 0;
    }
    if (*out_status == 0) *out_status = 200;
    if (*out_body && *out_body_len == 0) *out_body_len = strlen(*out_body);
    if (out_content_type && !*out_content_type)
        *out_content_type = xstrdup_("application/json");
    return 0;
}

static void handle_one(struct ds4cuda_http_server *s, int fd)
{
    parsed_request req;
    memset(&req, 0, sizeof(req));
    if (read_request(fd, &req) != 0) {
        send_error(fd, 400, "bad HTTP request");
        return;
    }
    const struct route *r = find_route(s, req.path);
    if (!r) {
        send_error(fd, 404, "unknown endpoint");
        parsed_request_free(&req);
        return;
    }
    /* Streaming handler wins if both are registered; the streaming
     * handler itself is responsible for inspecting the body and
     * choosing buffered-vs-stream behavior. If only a buffered handler
     * is wired, fall back to it. */
    if (r->stream_handler) {
        struct ds4cuda_http_stream_ctx ctx = { .fd = fd };
        int rc = r->stream_handler(req.method, req.path,
                                   req.body, req.body_len,
                                   &ctx, r->stream_user_data);
        if (rc != 0 && !ctx.began) {
            /* Handler refused before sending anything — give the client
             * a real error response. */
            send_error(fd, 500, "handler error");
        } else if (ctx.began && !ctx.ended) {
            /* Handler forgot to terminate — try to close the chunked
             * body cleanly so well-behaved clients see end-of-stream. */
            send_all(fd, "0\r\n\r\n", 5);
        }
        parsed_request_free(&req);
        return;
    }
    if (!r->handler) {
        send_error(fd, 500, "route has no handler");
        parsed_request_free(&req);
        return;
    }
    char *body = NULL;
    size_t body_len = 0;
    int status = 0;
    char *ctype = NULL;
    int rc = r->handler(req.method, req.path, req.body, req.body_len,
                        &body, &body_len, &status, &ctype, r->user_data);
    if (rc != 0) {
        free(body); free(ctype);
        send_error(fd, 500, "handler error");
        parsed_request_free(&req);
        return;
    }
    if (status == 0) status = 200;
    if (body && body_len == 0) body_len = strlen(body);
    send_response(fd, status, ctype ? ctype : "application/json",
                  body ? body : "", body_len);
    free(body);
    free(ctype);
    parsed_request_free(&req);
}

/* ------------------------------------------------------------------ */
/* Listener loop.                                                      */
/* ------------------------------------------------------------------ */
int ds4cuda_http_server_run(struct ds4cuda_http_server *s)
{
    if (!s || s->listen_fd < 0) return -EINVAL;
    /* Avoid SIGPIPE crashing the process when a client disconnects. */
    signal(SIGPIPE, SIG_IGN);
    while (!s->stopping) {
        int cfd = accept(s->listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (s->stopping) break;
            if (errno == EINTR) continue;
            if (errno == EBADF) break;
            return -errno;
        }
        struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
        setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        handle_one(s, cfd);
        close(cfd);
    }
    return 0;
}
