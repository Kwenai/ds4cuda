/*
 * sse_writer.h — Server-Sent Events frame helpers.
 *
 * SSE is a textual sub-protocol layered on a chunked HTTP/1.1 response
 * (Content-Type: text/event-stream). Each "event" is encoded as one or
 * more "field:value\n" lines followed by an empty line ("\n\n"
 * separator). The OpenAI streaming API uses only the `data:` field plus
 * an explicit "[DONE]" sentinel:
 *
 *   data: {"id":"...","choices":[{"delta":{"content":"Hello"}}, ...]}
 *
 *   data: [DONE]
 *
 * This module knows nothing about HTTP transport; it only formats the
 * SSE envelope around a JSON payload. The caller (openai_endpoint.c)
 * threads the rendered bytes through ds4cuda_http_stream_write_chunk so
 * each SSE frame becomes a single chunked-transfer fragment, satisfying
 * the "one frame per chunk" expectation of OpenAI clients.
 *
 * Helpers also build the per-token chat.completion.chunk JSON delta
 * objects so openai_endpoint.c stays parsing-only.
 */
#ifndef DS4CUDA_SSE_WRITER_H
#define DS4CUDA_SSE_WRITER_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wrap an arbitrary UTF-8 payload as a single SSE "data:" frame. The
 * returned malloc()'d string is "data: <payload>\n\n". Caller frees. */
char *ds4cuda_sse_format_data(const char *payload);

/* Convenience: same as above but spelled "data: [DONE]\n\n". */
char *ds4cuda_sse_format_done(void);

/* Build a chat.completion.chunk JSON string with a content delta.
 *   id     : "chatcmpl-..." (echoed across all chunks of a stream)
 *   model  : model id echoed back
 *   role   : if non-NULL, included in delta (only the first chunk does this)
 *   content: if non-NULL, included as delta.content; pass NULL on the
 *            final chunk where finish_reason carries the meaning.
 *   finish_reason: NULL (mid-stream) or "stop"/"length" on the final
 *            non-DONE chunk.
 *
 * Returns malloc()'d JSON string, caller frees. */
char *ds4cuda_sse_build_chunk_json(const char *id,
                                   const char *model,
                                   long created,
                                   const char *role,
                                   const char *content,
                                   const char *finish_reason);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_SSE_WRITER_H */
