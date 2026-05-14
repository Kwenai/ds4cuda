/*
 * tensor_table.c — reserved translation unit for tensor-name index
 * helpers.
 *
 * ds4_model_find_tensor / ds4_model_find_kv currently live in
 * gguf_parser.c as a linear strcmp scan (~µs over 1328 tensors). If a
 * per-name hash table is ever needed, it belongs here; the TU exists so
 * the build graph and link order are stable when that lands.
 */
#include "ds4cuda.h"
#include "ds4cuda_internal.h"
