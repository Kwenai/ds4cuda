/*
 * ds4cuda_internal.h — non-public helpers shared between
 * gguf_parser.c / model_open.c / tensor_table.c / main_tools.c.
 * Not installed as part of the public API.
 */
#ifndef DS4CUDA_INTERNAL_H
#define DS4CUDA_INTERNAL_H

#include "ds4cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* fatal abort, prints fmt + args to stderr then abort() */
__attribute__((noreturn, format(printf, 1, 2)))
void ds4_die(const char *fmt, ...);

/* Step-1 GGUF parser. mmap must already be in place. Fills m->kv,
 * m->tensors, m->tensor_data_pos, m->alignment. Aborts on error. */
int  ds4_gguf_parse(struct ds4_model *m);

/* Step-2 config validation. Reads keys from m->kv and stamps them into
 * m->n_layer / m->n_embd / etc. Aborts if any required key is missing
 * or has a value mismatching the expected DS4 geometry. */
void ds4_config_validate_model(struct ds4_model *m);

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_INTERNAL_H */
