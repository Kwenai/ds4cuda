/*
 * tokenizer.h — DeepSeek V4 Flash GPT-2 byte-level BPE tokenizer.
 *
 * Port of the JoyAI/DeepSeek BPE tokenizer in ds4/ds4.c (lines 13919-14588).
 * Reads "tokenizer.ggml.tokens" + "tokenizer.ggml.merges" from the GGUF
 * metadata KV table (parsed by ds4_model_open / ds4_gguf_parse) and
 * provides:
 *
 *   - encode (string -> token ids), supporting both raw text and chat-
 *     template-rendered text with embedded special-token markers
 *     (<｜begin▁of▁sentence｜>, <｜User｜>, ..., <think>, </think>, ｜DSML｜).
 *   - decode (ids -> string).
 *
 * Pure host C; no CUDA / GPU dependency. Memory: two open-addressed hash
 * tables (token_to_id ~ 129280 entries, merge_rank ~ 250k entries) on
 * top of GGUF mmap. ~6-8 MiB malloc total.
 */
#ifndef DS4CUDA_TOKENIZER_H
#define DS4CUDA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ds4_model;            /* fwd decl from <ds4cuda.h> */
struct ds4cuda_tokenizer;

/* Build a tokenizer from the GGUF metadata in `m`. Returns 0 on success
 * and stores a heap-allocated handle in *out (caller frees with
 * ds4cuda_tokenizer_free). On any error returns < 0 and *out is NULL.
 * Aborts only on out-of-memory (consistent with rest of ds4cuda). */
int  ds4cuda_tokenizer_init(struct ds4cuda_tokenizer **out,
                            const struct ds4_model *m);
void ds4cuda_tokenizer_free(struct ds4cuda_tokenizer *t);

/* Encode UTF-8 text -> array of int32 token ids. Recognizes embedded
 * DSML special-token UTF-8 byte sequences emitted by the chat template
 * (<｜begin▁of▁sentence｜>, <｜User｜>, <｜Assistant｜>, <｜end▁of▁sentence｜>,
 * <think>, </think>, ｜DSML｜) and substitutes their token id directly
 * (mirror of ds4_tokenize_rendered_chat, ds4.c:14524).
 *
 *   text     : NUL-terminated UTF-8 string (NULL is treated as "").
 *   out_ids  : caller-supplied buffer of length max_ids.
 *   max_ids  : capacity of out_ids in elements.
 *
 * Returns the number of token ids written. If the prompt would produce
 * more than max_ids tokens, returns -E2BIG (-7). out_ids may be NULL iff
 * max_ids == 0 (length-probe mode).
 */
int  ds4cuda_tokenize(const struct ds4cuda_tokenizer *t,
                      const char *text,
                      int *out_ids, int max_ids);

/* Same shape as ds4cuda_tokenize but encodes raw text only (ignores any
 * occurrences of special-token markers in the body — they go through the
 * BPE byte path). Mirrors ds4_tokenize_text (ds4.c:14486). */
int  ds4cuda_tokenize_raw(const struct ds4cuda_tokenizer *t,
                          const char *text,
                          int *out_ids, int max_ids);

/* Decode token ids -> UTF-8 text. out_text is NUL-terminated on success.
 * Returns the number of bytes written *excluding* the trailing NUL. If
 * max_size is too small, returns -E2BIG (-7) and out_text[0] is set to
 * '\0' (provided max_size >= 1). Returns -EINVAL (-22) on an out-of-range
 * token id. Performs the inverse of GPT-2 byte-level BPE encoding (each
 * Unicode codepoint maps back to one byte). Special-token strings are
 * emitted verbatim (e.g. "<｜User｜>"). */
int  ds4cuda_detokenize(const struct ds4cuda_tokenizer *t,
                        const int *ids, int n_ids,
                        char *out_text, size_t max_size);

/* Public (read-only) accessors for the cached special-token ids. Useful
 * for callers that want to splice in BOS / Assistant markers without
 * rendering the chat template through ds4cuda_tokenize. Returns < 0 if
 * the token is not present in the vocab (should never happen on a valid
 * V4 GGUF — vocab_load aborts in that case). */
int  ds4cuda_tokenizer_bos_id(const struct ds4cuda_tokenizer *t);
int  ds4cuda_tokenizer_eos_id(const struct ds4cuda_tokenizer *t);
int  ds4cuda_tokenizer_user_id(const struct ds4cuda_tokenizer *t);
int  ds4cuda_tokenizer_assistant_id(const struct ds4cuda_tokenizer *t);
int  ds4cuda_tokenizer_think_open_id(const struct ds4cuda_tokenizer *t);
int  ds4cuda_tokenizer_think_close_id(const struct ds4cuda_tokenizer *t);
int  ds4cuda_tokenizer_dsml_id(const struct ds4cuda_tokenizer *t);

/* Vocabulary size (= n_vocab from the GGUF). */
int  ds4cuda_tokenizer_n_vocab(const struct ds4cuda_tokenizer *t);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_TOKENIZER_H */
