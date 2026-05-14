/*
 * chat_template.h — DeepSeek-V4 / DSML chat template renderer.
 *
 * Renders an OpenAI-style {role, content}* message array into the canonical
 * DSML prompt string consumed by ds4_tokenize_rendered_chat (cite ds4.c
 * around line 14524). The DSML format uses three special tokens as turn
 * separators:
 *
 *   <｜begin▁of▁sentence｜>      BOS
 *   <｜User｜>                    user / tool turn marker
 *   <｜Assistant｜>               assistant turn marker
 *   <think>  /  </think>          reasoning fences
 *
 * The rendering rules — distilled from ds4_encode_chat_prompt (ds4.c:14549)
 * + ds4_chat_append_message (ds4.c:14562):
 *
 *   1. Always begin with BOS.
 *   2. system / developer messages: emit body text verbatim, no role marker.
 *   3. user messages: emit "<｜User｜>" then body text. role=="tool"|"function"
 *      adds a literal "Tool: " prefix to the body.
 *   4. assistant messages: emit "<｜Assistant｜>". If the body does not already
 *      start with "<think>" or "</think>", insert "</think>" first (ds4
 *      ships final assistant text already with </think> — we replicate
 *      that). Then the body.
 *   5. Final "assistant prefix" trailer: emit "<｜Assistant｜>" plus either
 *      "<think>" (think mode) or "</think>" (no-think). Caller picks via
 *      `enable_think`.
 *
 * The output is plain UTF-8 text, NUL-terminated; the tokenizer step (not
 * part of this module — this just renders to text) tokenizes via
 * ds4_tokenize_rendered_chat in the integrated build.
 */
#ifndef DS4CUDA_CHAT_TEMPLATE_H
#define DS4CUDA_CHAT_TEMPLATE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ds4cuda_chat_message {
    const char *role;     /* "system" | "user" | "assistant" | "tool" | "developer" | "function" */
    const char *content;  /* may be NULL (treated as ""). */
};

/* Render `n_messages` messages plus a trailing assistant prefix into `out`.
 *   out, out_size : caller-supplied buffer; result is NUL-terminated.
 *   enable_think  : true -> trailer ends with "<think>", false -> "</think>".
 *
 * Returns the number of bytes written *excluding* the NUL terminator. If
 * the buffer was too small, returns -1 and `out[0]` is set to '\0'. The
 * function never aborts; out_size==0 returns -1.
 *
 * NULL `messages` with n_messages==0 is allowed and produces only the BOS
 * + assistant prefix (a "fresh start" preamble).
 */
long ds4cuda_render_chat_prompt(
        const struct ds4cuda_chat_message *messages,
        int n_messages,
        bool enable_think,
        char *out, size_t out_size);

/* Same shape as ds4cuda_render_chat_prompt but returns a malloc()'d string
 * sized exactly to the result. NULL on internal error. Convenience for the
 * OpenAI endpoint glue — saves writing two-pass length probing. */
char *ds4cuda_render_chat_prompt_alloc(
        const struct ds4cuda_chat_message *messages,
        int n_messages,
        bool enable_think);

/* Render a chat prompt with an explicit system text prepended to the
 * `messages` array (Anthropic /v1/messages style — the "system" field is
 * a top-level sibling of "messages" rather than a role inside the array).
 *
 *   system_text   : optional UTF-8 string, NULL or "" for "no system text".
 *                   Treated identically to a leading {role:"system"} message
 *                   so the rest of the renderer code stays single-source.
 *   messages      : the user/assistant turn array (may be empty).
 *   n_messages    : count of messages.
 *   enable_think  : passes through to ds4cuda_render_chat_prompt.
 *
 * Returns malloc()'d string; NULL on alloc failure.  This thin wrapper
 * exists so the Anthropic endpoint does not need to maintain its own
 * synthetic system message buffer per request.
 */
char *ds4cuda_render_chat_prompt_with_system_alloc(
        const char *system_text,
        const struct ds4cuda_chat_message *messages,
        int n_messages,
        bool enable_think);

/* Special token marker constants — exported so tests can verify exact byte
 * sequence and so the eventual tokenizer step can re-detect them. */
extern const char *const DS4CUDA_TOK_BOS;        /* "<｜begin▁of▁sentence｜>" */
extern const char *const DS4CUDA_TOK_USER;       /* "<｜User｜>"               */
extern const char *const DS4CUDA_TOK_ASSISTANT;  /* "<｜Assistant｜>"          */
extern const char *const DS4CUDA_TOK_THINK_OPEN; /* "<think>"                  */
extern const char *const DS4CUDA_TOK_THINK_CLOSE;/* "</think>"                 */

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_CHAT_TEMPLATE_H */
