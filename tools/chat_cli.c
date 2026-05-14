/*
 * chat_cli.c — minimal one-shot CLI driver for ds4cuda inference engine.
 *
 * Usage:
 *   chat_cli "<prompt>" [max_new_tokens]
 *
 * Loads the engine (this is the cold/warm 81 GB GGUF load), renders a
 * single-user-turn DSML prompt, runs ds4cuda_inference_engine_generate_sync,
 * tokenizes the output back to count generated tokens, and prints:
 *
 *   load_ms      — engine create wall time (cold ≈ 280 s, warm ≈ 70 s)
 *   gen_wall_ms  — generate_sync wall time (prefill + decode + tokenize)
 *   prompt_tok   — tokens in the rendered DSML prompt
 *   out_bytes    — bytes of UTF-8 emitted
 *   out_tok      — tokens in the output (decoded back via tokenizer)
 *   tok_per_sec  — out_tok / (gen_wall_ms / 1000)
 *   forward_per_sec — (prompt_tok + out_tok) / gen_wall_ms — full forward
 *                    passes/sec (decode + prefill cost combined; useful
 *                    for comparing throughput across prompts of different
 *                    lengths since prefill is per-token forward)
 */
#define _GNU_SOURCE

#include "ds4cuda.h"
#include "server/inference_engine.h"
#include "server/chat_template.h"
#include "tokenizer/tokenizer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s \"<prompt>\" [max_new_tokens]\n", argv[0]);
        return 2;
    }
    const char *prompt_text = argv[1];
    int max_new_tokens = (argc >= 3) ? atoi(argv[2]) : 128;
    if (max_new_tokens <= 0) max_new_tokens = 128;

    /* max_context can be overridden via env DS4CUDA_MAX_CONTEXT for ad-hoc
     * sweeps. Default 262144 (256K) — session_state ~10 GB at this size,
     * total RSS ~98 GB on Spark, well under the 113 GB cap. */
    int max_context = 262144;
    const char *env_ctx = getenv("DS4CUDA_MAX_CONTEXT");
    if (env_ctx && env_ctx[0]) {
        int v = atoi(env_ctx);
        if (v > 0) max_context = v;
    }

    /* GGUF path comes from DS4CUDA_GGUF; required so we don't ship a
     * personal path in the OSS source. */
    const char *gguf_path = getenv("DS4CUDA_GGUF");
    if (!gguf_path || !gguf_path[0]) {
        fprintf(stderr,
                "ERROR: DS4CUDA_GGUF env var not set.\n"
                "  Set it to the path of the production GGUF, e.g.\n"
                "    export DS4CUDA_GGUF=/path/to/DeepSeek-V4-Flash-...gguf\n"
                "  See CONTRIBUTING.md \"Environment Setup\".\n");
        return 2;
    }

    /* Init details (prompt text, gguf path) are gated behind DS4CUDA_VERBOSE=1
     * so the default stderr stream doesn't leak the user's prompt or their
     * local filesystem layout. The model's response on stdout is the normal
     * useful output. */
    const char *verbose_env = getenv("DS4CUDA_VERBOSE");
    int verbose = (verbose_env && verbose_env[0] && verbose_env[0] != '0');
    if (verbose) {
        fprintf(stderr, "[chat_cli] prompt = %s\n", prompt_text);
        fprintf(stderr, "[chat_cli] max_new_tokens = %d\n", max_new_tokens);
        fprintf(stderr, "[chat_cli] max_context   = %d\n", max_context);
        fprintf(stderr, "[chat_cli] gguf_path     = %s\n", gguf_path);
        fprintf(stderr, "[chat_cli] DS4CUDA_WEIGHT_BACKEND = %s\n",
                getenv("DS4CUDA_WEIGHT_BACKEND") ?
                    getenv("DS4CUDA_WEIGHT_BACKEND") : "managed");
    }

    /* ---- load engine ---- */
    struct ds4cuda_inference_engine_options opts = {0};
    opts.gguf_path = gguf_path;
    opts.max_context = max_context;
    opts.default_max_new_tokens = max_new_tokens;
    opts.queue_depth = 4;
    opts.verbose_load = 1;

    double t0 = now_ms();
    struct ds4cuda_inference_engine *e = NULL;
    int rc = ds4cuda_inference_engine_create(&e, &opts);
    double t_load = now_ms() - t0;
    if (rc != 0 || !e) {
        fprintf(stderr, "[chat_cli] engine_create rc=%d\n", rc);
        return 1;
    }
    fprintf(stderr, "[chat_cli] engine loaded in %.2f s\n", t_load / 1000.0);

    /* ---- render DSML prompt ---- */
    struct ds4cuda_chat_message msgs[1];
    msgs[0].role = "user";
    msgs[0].content = prompt_text;
    char *dsml_prompt = ds4cuda_render_chat_prompt_alloc(msgs, 1, /*enable_think=*/false);
    if (!dsml_prompt) {
        fprintf(stderr, "[chat_cli] render_chat_prompt failed\n");
        ds4cuda_inference_engine_destroy(e);
        return 1;
    }
    size_t dsml_bytes = strlen(dsml_prompt);
    fprintf(stderr, "[chat_cli] DSML prompt %zu bytes\n", dsml_bytes);

    /* ---- count prompt tokens via tokenizer (separate handle, cheap) ---- */
    struct ds4_model m = {0};
    if (ds4_model_open(&m, gguf_path) != 0) {
        fprintf(stderr, "[chat_cli] ds4_model_open for tokenizer failed\n");
        free(dsml_prompt);
        ds4cuda_inference_engine_destroy(e);
        return 1;
    }
    struct ds4cuda_tokenizer *tok = NULL;
    if (ds4cuda_tokenizer_init(&tok, &m) != 0 || !tok) {
        fprintf(stderr, "[chat_cli] tokenizer_init failed\n");
        ds4_model_close(&m);
        free(dsml_prompt);
        ds4cuda_inference_engine_destroy(e);
        return 1;
    }
    int prompt_tok_buf[4096];
    int prompt_tok = ds4cuda_tokenize(tok, dsml_prompt, prompt_tok_buf, 4096);
    if (prompt_tok < 0) {
        fprintf(stderr, "[chat_cli] tokenize prompt failed rc=%d\n", prompt_tok);
        ds4cuda_tokenizer_free(tok);
        ds4_model_close(&m);
        free(dsml_prompt);
        ds4cuda_inference_engine_destroy(e);
        return 1;
    }
    fprintf(stderr, "[chat_cli] prompt tokens = %d\n", prompt_tok);

    /* ---- generate ---- */
    char *out_text = NULL;
    double t1 = now_ms();
    rc = ds4cuda_inference_engine_generate_sync(e, dsml_prompt, max_new_tokens, &out_text);
    double t_gen = now_ms() - t1;
    if (rc != 0 || !out_text) {
        fprintf(stderr, "[chat_cli] generate_sync rc=%d\n", rc);
        ds4cuda_tokenizer_free(tok);
        ds4_model_close(&m);
        free(dsml_prompt);
        ds4cuda_inference_engine_destroy(e);
        return 1;
    }

    /* ---- count output tokens ---- */
    size_t out_bytes = strlen(out_text);
    int out_tok_buf[8192];
    int out_tok = ds4cuda_tokenize_raw(tok, out_text, out_tok_buf, 8192);
    if (out_tok < 0) out_tok = 0;

    /* ---- report ---- */
    printf("\n");
    printf("prompt:    %s\n", prompt_text);
    printf("output:    %s\n", out_text);
    printf("\n---- timing ----\n");
    printf("  load_ms          = %12.2f  (%.2f s)\n", t_load, t_load / 1000.0);
    printf("  gen_wall_ms      = %12.2f  (%.2f s)\n", t_gen,  t_gen  / 1000.0);
    printf("  prompt_tok       = %12d\n", prompt_tok);
    printf("  out_tok          = %12d\n", out_tok);
    printf("  out_bytes        = %12zu\n", out_bytes);
    if (t_gen > 0) {
        double secs = t_gen / 1000.0;
        printf("  out_tok_per_sec  = %12.3f   (decode-only estimate, conflates prefill)\n",
               (double)out_tok / secs);
        printf("  forward_per_sec  = %12.3f   (prompt+out_tok / wall — true per-pass throughput)\n",
               (double)(prompt_tok + out_tok) / secs);
        printf("  ms_per_forward   = %12.2f\n",
               t_gen / (double)(prompt_tok + out_tok));
    }

    free(out_text);
    free(dsml_prompt);
    ds4cuda_tokenizer_free(tok);
    ds4_model_close(&m);
    ds4cuda_inference_engine_destroy(e);
    return 0;
}
