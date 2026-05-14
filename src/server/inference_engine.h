/*
 * inference_engine.h — real inference engine that drives
 * ds4_forward_token against the loaded 81 GB DeepSeek-V4-Flash model.
 *
 * Architecture (single-session FIFO, single GPU, single worker thread):
 *
 *   ds4cuda_inference_engine_create
 *       loads the GGUF (mmap), runs ds4_model_load_to_managed (~70 s
 *       warm cache), allocates a single ds4_session_state for
 *       max_context tokens, and spawns one worker thread that pops
 *       jobs from a bounded FIFO and feeds them into ds4_forward_token.
 *
 *   ds4cuda_real_buffered_generator
 *   ds4cuda_real_stream_generator
 *       Match the existing function-pointer types in openai_endpoint.h
 *       (ds4cuda_chat_generator_fn / ds4cuda_chat_stream_generator_fn).
 *       Both enqueue a job and either wait for the buffer (buffered) or
 *       pump the per-token emit callback as the worker generates.
 *
 *   ds4cuda_real_anthropic_generator
 *       Adapter for the Anthropic /v1/messages endpoint
 *       (ds4cuda_anthropic_generator_fn). Same buffered behavior; the
 *       tools_text argument is currently ignored (tool-call generation
 *       is out of scope for this release — single-session FIFO
 *       inference is the operating model).
 *
 *   ds4cuda_inference_engine_destroy
 *       Stops the worker, frees session_state, frees managed weights,
 *       closes model.
 *
 * Concurrency: the FIFO is bounded (default 4 slots). When full, the
 * generator returns -EBUSY which the endpoint maps to HTTP 503. We do
 * NOT support multiple concurrent prefills — the worker is strictly
 * one-job-at-a-time, FIFO. Multi-session is out of scope; prefix-reuse
 * is handled per-session by the prompt-prefix sync path.
 *
 * RSS budget: 81 GB managed weights + ~92 MB session state +
 * ~16 MB activation arena + small server + thread stacks. Well under
 * the 113 GB DGX Spark cap (81 + ~92 MB + ~16 MB).
 */
#ifndef DS4CUDA_INFERENCE_ENGINE_H
#define DS4CUDA_INFERENCE_ENGINE_H

#include <stddef.h>

#include "openai_endpoint.h"
#include "anthropic_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ds4cuda_inference_engine;

struct ds4cuda_inference_engine_options {
    /* Path to the DeepSeek-V4-Flash GGUF. Required. */
    const char *gguf_path;
    /* KV-cache context capacity in tokens. Drives session_state arena
     * size. 64 is enough for "Hello"-class prompts; 4096 covers most
     * single-turn chat. Default 256 if 0. */
    int max_context;
    /* Default ceiling for per-request max_tokens when the request omits
     * it. Default 64 if 0. */
    int default_max_new_tokens;
    /* FIFO queue depth. Default 4 if 0. */
    int queue_depth;
    /* If non-zero, ds4_model_load_to_managed prints chunk progress. */
    int verbose_load;
    /* Optional: path to a previously saved KV+token-cache to restore on
     * engine creation (prefix-sync feature).  When non-NULL, after the model loads
     * and the session is allocated, the engine calls
     * ds4_session_load_from_disk(<path>) and then loads <path>.tokens to
     * populate the cached prompt-prefix token sequence so the next job
     * can hit the prefix-sync fast path.  If load fails, engine_create
     * returns the load error (we treat a bad restore as a hard error so
     * callers know the cache wasn't honored). NULL = no restore. */
    const char *restore_path;
};

/* Construct + load the engine. Blocks ~70 s warm / ~250 s cold while
 * the 81 GB GGUF streams into managed memory. Returns 0 on success
 * with *out populated; <0 on failure (bad GGUF, alloc failure, CUDA
 * error). On failure *out is left NULL. */
int  ds4cuda_inference_engine_create(
        struct ds4cuda_inference_engine **out,
        const struct ds4cuda_inference_engine_options *opts);

/* Stop the worker thread, free everything. Safe on NULL. */
void ds4cuda_inference_engine_destroy(struct ds4cuda_inference_engine *e);

/* OpenAI buffered chat generator. Matches ds4cuda_chat_generator_fn.
 * `user_data` MUST be a non-NULL ds4cuda_inference_engine pointer.
 *
 * Enqueues a job with the given (already DSML-rendered) prompt and
 * waits on a per-job condvar until the worker finishes (or the queue
 * is full -> returns -EBUSY).  On success *out_text is malloc()'d
 * and ownership transfers to the caller. */
int  ds4cuda_real_buffered_generator(
        const char *prompt,
        int max_new_tokens,
        void *user_data,
        char **out_text);

/* OpenAI streaming chat generator. Matches
 * ds4cuda_chat_stream_generator_fn. `user_data` MUST be a non-NULL
 * ds4cuda_inference_engine pointer. The worker invokes `emit` per
 * generated token (already decoded to UTF-8 bytes); this function
 * drives the worker on the calling thread (the listener thread) by
 * pumping per-token outputs through the emit channel as the worker
 * produces them. */
int  ds4cuda_real_stream_generator(
        const char *prompt,
        int max_new_tokens,
        void *user_data,
        ds4cuda_chat_stream_emit_fn emit,
        void *emit_user_data);

/* Anthropic buffered generator (tools_text currently ignored — see
 * comment at top). Matches ds4cuda_anthropic_generator_fn. */
int  ds4cuda_real_anthropic_generator(
        const char *dsml_prompt,
        const char *tools_text,
        int max_new_tokens,
        void *user_data,
        char **out_text);

/* ----- Disk KV save/load wrappers (prefix-sync feature) ---------------- */

/* Save the engine's session_state to disk, plus a sidecar `<path>.tokens`
 * file capturing the prompt-prefix token-id sequence the engine has cached
 * (== prompt + generated of the last completed job; used by
 * load_session_from_disk to seed the prefix-sync compare).
 *
 * Quiesces the worker by enqueuing a synchronization point — saves block
 * the worker thread for the duration of the DtoH dump (tens of ms at
 * max_ctx=64; ~hundreds of ms at max_ctx=4096).  If no completed prompt
 * is cached, the .tokens sidecar is still written with count=0 and the
 * KV blob reflects whatever state the session is in (after engine_create
 * with no prior job, that's an empty-reset state).
 *
 * Returns 0 on success, <0 on I/O / CUDA error. Safe to call concurrently
 * with new jobs (the worker mutex serializes), but the caller is expected
 * to coordinate (this is an admin op, not a hot-path API). */
int  ds4cuda_inference_engine_save_session_to_disk(
        struct ds4cuda_inference_engine *e,
        const char *path);

/* Load a previously saved session into the engine's session_state +
 * token cache.  Same constraints as ds4_session_load_from_disk: the
 * engine's max_context MUST match the file's header.max_context.
 *
 * Returns 0 on success, <0 on error.  On success the next job whose
 * prompt extends the saved token sequence will hit the prefix-sync
 * fast path.  Note: engine_create exposes a `restore_path` option that
 * calls this internally after session alloc; callers using that field
 * should NOT call this function again. */
int  ds4cuda_inference_engine_load_session_from_disk(
        struct ds4cuda_inference_engine *e,
        const char *path);

/* ----- Synchronous in-process generation ------------------------------- */

/* Run a single buffered generation against a pre-rendered DSML prompt,
 * blocking until done or error. Drives the engine without going through
 * the HTTP layer. Returns 0 on success, <0 on error. */
int  ds4cuda_inference_engine_generate_sync(
        struct ds4cuda_inference_engine *e,
        const char *prompt,
        int max_new_tokens,
        char **out_text);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_INFERENCE_ENGINE_H */
