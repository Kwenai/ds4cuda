/*
 * main_server.c — standalone resident HTTP server for ds4cuda.
 *
 * The inference engine loads the 81 GB managed-weight region once during
 * process startup, then every OpenAI / Anthropic request reuses the same
 * engine and session worker.  This is intentionally thin glue around the
 * server/engine APIs; it does not change the CUDA forward path.
 */
#define _GNU_SOURCE

#include "server/anthropic_endpoint.h"
#include "server/http_server.h"
#include "server/inference_engine.h"
#include "server/openai_endpoint.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct ds4cuda_http_server *g_server;

static void on_signal(int signo)
{
    (void)signo;
    if (g_server) ds4cuda_http_server_stop(g_server);
}

static void usage(const char *argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("options:\n");
    printf("  --gguf PATH          GGUF model path (default: $DS4CUDA_GGUF)\n");
    printf("  --port N             listen port on 127.0.0.1 (default: 8080)\n");
    printf("  --max-context N      KV/cache context capacity (default: 262144)\n");
    printf("  --max-tokens N       default response token limit (default: 128)\n");
    printf("  --queue-depth N      pending request FIFO depth (default: 4)\n");
    printf("  --restore PATH       load disk KV/session cache before serving\n");
    printf("  --quiet-load         suppress managed-weight chunk progress\n");
    printf("  --model-id NAME      model id returned by OpenAI/Anthropic endpoints\n");
    printf("  env DS4CUDA_WEIGHT_BACKEND=managed|mmap_direct (default: managed)\n");
    printf("  -h, --help           show this help without loading the model\n");
}

static int parse_int_arg(const char *flag, const char *value, int min_value)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(value, &end, 10);
    if (errno || !end || *end || v < min_value || v > 2147483647L) {
        fprintf(stderr, "invalid %s value: %s\n", flag, value);
        return -1;
    }
    return (int)v;
}

int main(int argc, char **argv)
{
    /* Default GGUF comes from $DS4CUDA_GGUF; --gguf overrides. */
    const char *gguf_path = getenv("DS4CUDA_GGUF");
    const char *restore_path = NULL;
    const char *model_id = "deepseek-v4-flash";
    int port = 8080;
    int max_context = 262144;
    int max_tokens = 128;
    int queue_depth = 4;
    int verbose_load = 1;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else if (!strcmp(a, "--gguf") && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if (!strcmp(a, "--restore") && i + 1 < argc) {
            restore_path = argv[++i];
        } else if (!strcmp(a, "--model-id") && i + 1 < argc) {
            model_id = argv[++i];
        } else if (!strcmp(a, "--port") && i + 1 < argc) {
            port = parse_int_arg(a, argv[++i], 0);
            if (port < 0) return 2;
        } else if (!strcmp(a, "--max-context") && i + 1 < argc) {
            max_context = parse_int_arg(a, argv[++i], 1);
            if (max_context < 0) return 2;
        } else if (!strcmp(a, "--max-tokens") && i + 1 < argc) {
            max_tokens = parse_int_arg(a, argv[++i], 1);
            if (max_tokens < 0) return 2;
        } else if (!strcmp(a, "--queue-depth") && i + 1 < argc) {
            queue_depth = parse_int_arg(a, argv[++i], 1);
            if (queue_depth < 0) return 2;
        } else if (!strcmp(a, "--quiet-load")) {
            verbose_load = 0;
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", a);
            usage(argv[0]);
            return 2;
        }
    }

    if (!gguf_path || !gguf_path[0]) {
        fprintf(stderr,
                "ERROR: GGUF path not provided. Pass --gguf PATH or set\n"
                "  DS4CUDA_GGUF in the environment. See CONTRIBUTING.md\n"
                "  \"Environment Setup\".\n");
        return 2;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    fprintf(stderr, "[ds4cuda-server] loading engine\n");
    /* Only log the basename so the full local filesystem layout doesn't
     * leak into logs / bug reports. The directory portion is rarely useful
     * for debugging anyway. */
    {
        const char *gguf_base = strrchr(gguf_path, '/');
        gguf_base = gguf_base ? gguf_base + 1 : gguf_path;
        fprintf(stderr, "[ds4cuda-server] gguf=%s\n", gguf_base);
    }
    fprintf(stderr, "[ds4cuda-server] max_context=%d max_tokens=%d queue_depth=%d\n",
            max_context, max_tokens, queue_depth);
    fprintf(stderr, "[ds4cuda-server] DS4CUDA_WEIGHT_BACKEND=%s\n",
            getenv("DS4CUDA_WEIGHT_BACKEND") ?
                getenv("DS4CUDA_WEIGHT_BACKEND") : "managed");

    struct ds4cuda_inference_engine_options engine_opts;
    memset(&engine_opts, 0, sizeof(engine_opts));
    engine_opts.gguf_path = gguf_path;
    engine_opts.max_context = max_context;
    engine_opts.default_max_new_tokens = max_tokens;
    engine_opts.queue_depth = queue_depth;
    engine_opts.verbose_load = verbose_load;
    engine_opts.restore_path = restore_path;

    struct ds4cuda_inference_engine *engine = NULL;
    int rc = ds4cuda_inference_engine_create(&engine, &engine_opts);
    if (rc != 0 || !engine) {
        fprintf(stderr, "[ds4cuda-server] engine_create failed rc=%d\n", rc);
        return 1;
    }

    struct ds4cuda_http_server *srv = NULL;
    rc = ds4cuda_http_server_create(&srv, port);
    if (rc != 0 || !srv) {
        fprintf(stderr, "[ds4cuda-server] http_server_create(port=%d) failed rc=%d\n",
                port, rc);
        ds4cuda_inference_engine_destroy(engine);
        return 1;
    }
    g_server = srv;

    struct ds4cuda_chat_endpoint_options openai_opts;
    memset(&openai_opts, 0, sizeof(openai_opts));
    openai_opts.model_id = model_id;
    openai_opts.default_max_tokens = max_tokens;
    openai_opts.generator = ds4cuda_real_buffered_generator;
    openai_opts.generator_user_data = engine;
    openai_opts.stream_generator = ds4cuda_real_stream_generator;
    openai_opts.stream_generator_user_data = engine;
    rc = ds4cuda_openai_endpoint_install(srv, &openai_opts);
    if (rc != 0) {
        fprintf(stderr, "[ds4cuda-server] openai_endpoint_install failed rc=%d\n", rc);
        ds4cuda_http_server_destroy(srv);
        ds4cuda_inference_engine_destroy(engine);
        return 1;
    }

    struct ds4cuda_anthropic_endpoint_options anthropic_opts;
    memset(&anthropic_opts, 0, sizeof(anthropic_opts));
    anthropic_opts.model_id = model_id;
    anthropic_opts.default_max_tokens = max_tokens;
    anthropic_opts.generator = ds4cuda_real_anthropic_generator;
    anthropic_opts.generator_user_data = engine;
    rc = ds4cuda_anthropic_endpoint_install(srv, &anthropic_opts);
    if (rc != 0) {
        fprintf(stderr, "[ds4cuda-server] anthropic_endpoint_install failed rc=%d\n", rc);
        ds4cuda_http_server_destroy(srv);
        ds4cuda_inference_engine_destroy(engine);
        return 1;
    }

    fprintf(stderr, "[ds4cuda-server] listening on http://127.0.0.1:%d\n",
            ds4cuda_http_server_port(srv));
    fprintf(stderr, "[ds4cuda-server] model is resident until this process exits\n");

    rc = ds4cuda_http_server_run(srv);
    if (rc != 0) {
        fprintf(stderr, "[ds4cuda-server] http_server_run returned rc=%d\n", rc);
    }

    g_server = NULL;
    ds4cuda_http_server_destroy(srv);
    ds4cuda_inference_engine_destroy(engine);
    return rc == 0 ? 0 : 1;
}
