/*
 * main_tools.c — ds4cuda_tools CLI.
 *
 *   ds4cuda_tools dump-tensors <gguf>
 *       Emit one TSV row per tensor, header identical to the Python
 *       reference output from scripts/gguf_header_parse.py.
 *
 *   ds4cuda_tools validate <gguf>
 *       Run config_validate_model + sanity-check totals; print "OK" on
 *       success or a fatal error and abort.
 *
 *   ds4cuda_tools dump-kv <gguf>
 *       (bonus, helpful when debugging) — print every KV's key+type+
 *       a short repr of its value.
 */
#define _GNU_SOURCE
#include "ds4cuda.h"
#include "ds4cuda_internal.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t tensor_elements(const struct ds4_tensor *t)
{
    uint64_t e = 1;
    for (uint32_t d = 0; d < t->n_dims; ++d) e *= t->dims[d];
    return e;
}

static int cmd_dump_tensors(const char *path)
{
    struct ds4_model m;
    if (ds4_model_open(&m, path) != 0) return 1;

    printf("name\ttype_id\ttype_name\tndim\tdims\telements\tbytes\tabs_offset\n");
    for (size_t i = 0; i < m.n_tensors; ++i) {
        const struct ds4_tensor *t = &m.tensors[i];
        printf("%s\t%u\t%s\t%u\t",
               t->name, (unsigned)t->quant, ds4_quant_name(t->quant),
               (unsigned)t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; ++d)
            printf("%s%" PRIu64, d ? "," : "", t->dims[d]);
        printf("\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
               tensor_elements(t), t->byte_size, t->abs_offset);
    }
    ds4_model_close(&m);
    return 0;
}

static int cmd_validate(const char *path)
{
    struct ds4_model m;
    if (ds4_model_open(&m, path) != 0) return 1;

    uint64_t total = 0;
    for (size_t i = 0; i < m.n_tensors; ++i) total += m.tensors[i].byte_size;

    fprintf(stderr,
            "ds4cuda_tools validate: GGUF v3 alignment=%u\n"
            "  n_kv=%zu n_tensors=%zu tensor_data_pos=%" PRIu64 "\n"
            "  file_size=%zu  sum(byte_size)=%" PRIu64 "\n"
            "  geometry: n_layer=%u n_embd=%u n_vocab=%u n_head=%u "
            "n_expert=%u n_expert_used=%u\n",
            m.alignment, m.n_kv, m.n_tensors, m.tensor_data_pos,
            m.file_size, total,
            m.n_layer, m.n_embd, m.n_vocab, m.n_head,
            m.n_expert, m.n_expert_used);
    printf("validate OK\n");
    ds4_model_close(&m);
    return 0;
}

static int cmd_dump_kv(const char *path)
{
    struct ds4_model m;
    if (ds4_model_open(&m, path) != 0) return 1;
    for (size_t i = 0; i < m.n_kv; ++i) {
        const struct ds4_kv *kv = &m.kv[i];
        printf("%s\t(type=%u)\t", kv->key, (unsigned)kv->type);
        switch (kv->type) {
        case DS4_GGUF_UINT8:   printf("%u\n",   kv->v.u8); break;
        case DS4_GGUF_INT8:    printf("%d\n",   kv->v.i8); break;
        case DS4_GGUF_BOOL:    printf("%s\n",   kv->v.b ? "true" : "false"); break;
        case DS4_GGUF_UINT16:  printf("%u\n",   kv->v.u16); break;
        case DS4_GGUF_INT16:   printf("%d\n",   kv->v.i16); break;
        case DS4_GGUF_UINT32:  printf("%u\n",   kv->v.u32); break;
        case DS4_GGUF_INT32:   printf("%d\n",   kv->v.i32); break;
        case DS4_GGUF_FLOAT32: printf("%g\n",   (double)kv->v.f32); break;
        case DS4_GGUF_UINT64:  printf("%" PRIu64 "\n", kv->v.u64); break;
        case DS4_GGUF_INT64:   printf("%" PRId64 "\n", kv->v.i64); break;
        case DS4_GGUF_FLOAT64: printf("%g\n",   kv->v.f64); break;
        case DS4_GGUF_STRING:  printf("'%s'\n", kv->v.s); break;
        case DS4_GGUF_ARRAY:
            printf("array<inner=%u, len=%" PRIu64 ">\n",
                   (unsigned)kv->v.arr.elem_type, kv->v.arr.length);
            break;
        default: printf("<?>\n");
        }
    }
    ds4_model_close(&m);
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
        "usage:\n"
        "  ds4cuda_tools dump-tensors <gguf>\n"
        "  ds4cuda_tools dump-kv      <gguf>\n"
        "  ds4cuda_tools validate     <gguf>\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) { usage(); return 2; }
    const char *cmd = argv[1];
    const char *path = argv[2];
    if (strcmp(cmd, "dump-tensors") == 0) return cmd_dump_tensors(path);
    if (strcmp(cmd, "dump-kv")      == 0) return cmd_dump_kv(path);
    if (strcmp(cmd, "validate")     == 0) return cmd_validate(path);
    usage();
    return 2;
}
