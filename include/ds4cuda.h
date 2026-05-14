/*
 * ds4cuda — minimal C API for the DeepSeek-V4-Flash CUDA inference engine.
 *
 * Covers the GGUF parser + tensor metadata table + config validation, the
 * managed-memory model loader, per-session decode state, and the CUDA forward
 * launcher entry points.
 */
#ifndef DS4CUDA_H
#define DS4CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* GGUF v3 quant types we care about. Numeric IDs match the on-disk    */
/* enum (ds4.c gguf_types[] table, ds4cuda/scripts/gguf_header_parse.py).
 * This enum lists only the quant types referenced by the runtime
 * forward path. The parser (src/gguf_parser.c) tolerates additional
 * GGUF v3 types — they are accepted for size/offset accounting so
 * tensors we never read still parse cleanly. The abort happens later
 * if a forward-required tensor turns out to use an unsupported type
 * (config_validate_model + the per-kernel dispatch). */
/* ------------------------------------------------------------------ */
enum ds4_quant_type {
    DS4_QUANT_F32      = 0,
    DS4_QUANT_F16      = 1,
    DS4_QUANT_Q8_0     = 8,
    DS4_QUANT_Q2_K     = 10,
    DS4_QUANT_IQ2_XXS  = 16,
    DS4_QUANT_I32      = 26,
};

/* GGUF metadata value types (subset; full set in gguf_parser.c). */
enum ds4_gguf_value_type {
    DS4_GGUF_UINT8   = 0,
    DS4_GGUF_INT8    = 1,
    DS4_GGUF_UINT16  = 2,
    DS4_GGUF_INT16   = 3,
    DS4_GGUF_UINT32  = 4,
    DS4_GGUF_INT32   = 5,
    DS4_GGUF_FLOAT32 = 6,
    DS4_GGUF_BOOL    = 7,
    DS4_GGUF_STRING  = 8,
    DS4_GGUF_ARRAY   = 9,
    DS4_GGUF_UINT64  = 10,
    DS4_GGUF_INT64   = 11,
    DS4_GGUF_FLOAT64 = 12,
};

/* CUDA-visible weight backing mode.  The default managed backend mirrors
 * the full GGUF into cudaMallocManaged memory.  mmap_direct is opt-in and
 * keeps tensor bytes in the GGUF mmap, relying on GB10 pageable memory
 * access for CUDA kernels. */
enum ds4_weight_backend {
    DS4_WEIGHT_BACKEND_MANAGED = 0,
    DS4_WEIGHT_BACKEND_MMAP_DIRECT = 1,
};

/* ------------------------------------------------------------------ */
/* Tensor metadata. One entry per GGUF tensor.                         */
/* ------------------------------------------------------------------ */
#define DS4_MAX_DIMS 4

struct ds4_tensor {
    const char *name;              /* points into model->name_pool, NUL-terminated */
    enum ds4_quant_type quant;
    uint32_t n_dims;
    uint64_t dims[DS4_MAX_DIMS];
    uint64_t abs_offset;           /* byte offset from start of mmap */
    uint64_t byte_size;            /* total bytes for this tensor */
};

/* ------------------------------------------------------------------ */
/* Lightweight metadata-KV record (parsed strictly enough to satisfy   */
/* config_validate_model in src/model_open.c). Array values are stored */
/* as a pointer into the mmap region: caller is expected to know the   */
/* element type and length (see ds4_kv_array.*).                       */
/* ------------------------------------------------------------------ */
struct ds4_kv_array {
    enum ds4_gguf_value_type elem_type;
    uint64_t length;
    const uint8_t *raw;            /* points into mmap */
};

struct ds4_kv {
    const char *key;               /* into name pool */
    enum ds4_gguf_value_type type;
    /* For scalar value types we eagerly decode into one of these slots.
     * For DS4_GGUF_ARRAY the array struct is filled. For DS4_GGUF_STRING
     * the string is into name pool. */
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        uint8_t  b;                /* bool */
        const char *s;             /* string into name pool */
        struct ds4_kv_array arr;
    } v;
};

/* ------------------------------------------------------------------ */
/* The model handle: one open GGUF file.                               */
/* ------------------------------------------------------------------ */
struct ds4_model {
    /* mmap region */
    int fd;
    void *mmap_ptr;
    size_t file_size;
    uint32_t alignment;            /* GGUF general.alignment, default 32 */
    uint64_t tensor_data_pos;      /* aligned start of tensor data section */

    /* parsed deepseek4 geometry (ds4.c:2343–2425). populated by
     * config_validate_model. */
    uint32_t n_layer;              /* deepseek4.block_count            = 43 */
    uint32_t n_embd;               /* deepseek4.embedding_length       = 4096 */
    uint32_t n_vocab;              /* deepseek4.vocab_size             = 129280 */
    uint32_t n_head;               /* deepseek4.attention.head_count   = 64 */
    uint32_t head_kv;              /* deepseek4.attention.head_count_kv= 1 */
    uint32_t key_len;              /* deepseek4.attention.key_length   = 512 */
    uint32_t val_len;              /* deepseek4.attention.value_length = 512 */
    uint32_t rope_dim;             /* deepseek4.rope.dimension_count   = 64 */
    uint32_t out_group;            /* deepseek4.attention.output_group_count = 8 */
    uint32_t q_lora_rank;          /* = 1024 */
    uint32_t out_lora_rank;        /* = 1024 */
    uint32_t n_expert;             /* = 256 */
    uint32_t n_expert_used;        /* = 6 */
    uint32_t expert_ff;            /* = 2048 */
    uint32_t n_shared_expert;      /* = 1 */
    uint32_t hash_layer_count;     /* = 3 */
    uint32_t expert_group_count;   /* optional, expected 0 */
    uint32_t expert_group_used;    /* optional, expected 0 */
    uint32_t sliding_window;       /* = 128 */
    uint32_t indexer_head;         /* = 64 */
    uint32_t indexer_key_len;      /* = 128 */
    uint32_t indexer_top_k;        /* = 512 */
    uint32_t hc_count;             /* hyper_connection.count = 4 */
    uint32_t hc_sinkhorn_iter;     /* = 20 */

    /* tensor table */
    struct ds4_tensor *tensors;
    size_t n_tensors;

    /* metadata KV table */
    struct ds4_kv *kv;
    size_t n_kv;

    /* one big arena for all NUL-terminated names (tensor + KV + string
     * values). Single allocation; freed on close. */
    char *name_pool;
    size_t name_pool_size;
    size_t name_pool_used;

    /* Optional cudaMallocManaged region holding the entire GGUF file
     * after ds4_model_load_to_managed. Layout mirrors the on-disk file:
     * `weights_managed_base + tensor->abs_offset` returns the device
     * pointer for that tensor's bytes. NULL if load_to_managed was not
     * called. weights_managed_size == file_size when populated.
     *
     * Lifecycle: caller MUST call ds4_model_managed_free BEFORE
     * ds4_model_close (the latter is pure C and cannot call cudaFree).
     * After ds4_model_managed_free both fields are reset to 0/NULL. */
    void  *weights_managed_base;
    size_t weights_managed_size;

    enum ds4_weight_backend weight_backend;

    /* Byte ranges within the mmap, populated by ds4_gguf_parse for tooling.
     * kv_byte_range_end          = start of first tensor info record (after
     *                              the KV records section, byte offset from
     *                              start of file).
     * tensor_info_byte_range_end = start of alignment padding before the
     *                              tensor data blob (i.e. just past the last
     *                              tensor info record).
     * Both equal a file offset (not relative to anything else). Used by the
     * offline repack tool to copy KV + tensor-info sections byte-for-byte
     * without re-encoding them.
     *
     * Appended at the end of the struct so a partial rebuild does not
     * silently corrupt earlier fields if some object files were compiled
     * against the prior header.  Treat header changes as ABI-breaking and
     * `make clean` if in doubt. */
    uint64_t kv_byte_range_end;
    uint64_t tensor_info_byte_range_end;
};

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

/* Open and parse a GGUF file. Returns 0 on success, -1 on error.
 * On success the caller MUST call ds4_model_close to release mmap +
 * malloc'd metadata. On any structural / spec violation the function
 * aborts (matching ds4 model_open behavior — invalid GGUFs cannot run,
 * fail-fast over silent fallback). */
int  ds4_model_open(struct ds4_model *out, const char *gguf_path);
void ds4_model_close(struct ds4_model *m);

/* Lookup tensor by GGUF name. NULL if not found. O(N) linear scan;
 * 1328 tensors × strcmp is ~µs scale. */
const struct ds4_tensor *ds4_model_find_tensor(const struct ds4_model *m,
                                               const char *name);

/* Lookup KV entry by key. NULL if absent. */
const struct ds4_kv *ds4_model_find_kv(const struct ds4_model *m,
                                       const char *key);

/* Block element / byte counts for a quant type. Returns 1 for non-block
 * types (F32/F16/I32). Aborts on unsupported types. */
size_t ds4_quant_block_elems(enum ds4_quant_type t);
size_t ds4_quant_block_bytes(enum ds4_quant_type t);
const char *ds4_quant_name(enum ds4_quant_type t);

/* ------------------------------------------------------------------ */
/* Full-weight load into cudaMallocManaged region.                     */
/*                                                                     */
/* Pipeline (method A):                                                */
/*   1. cudaMallocManaged(file_size) — lazy alloc, no physical RAM yet.*/
/*   2. for each chunk_size_bytes window in [0, file_size):            */
/*        memcpy from mmap_ptr+off to managed+off (managed RSS +chunk).*/
/*        madvise(mmap_ptr+off, chunk, MADV_DONTNEED) (mmap RSS -chunk)*/
/*      Peak RSS during the loop = (loaded so far) + 2*chunk           */
/*                              <= file_size + 2*chunk.                */
/*   3. madvise the entire mmap MADV_DONTNEED to clear residue.        */
/*                                                                     */
/* RSS guard: every chunk re-reads /proc/meminfo MemAvailable. If it   */
/* drops below 16 GB the function aborts(2) (1 GB headroom over the    */
/* 15 GB OS reserve). Setup time is roughly file_size / 0.7 GB/s on    */
/* GB10 (≈ 56 s for 80 GB).                                            */
/*                                                                     */
/* Default chunk size when 0 is passed: 4 GiB. Returns 0 on success    */
/* and a negative errno on a non-fatal failure; aborts on red-line     */
/* breach or CUDA error (consistent with model_open's fail-fast).      */
int ds4_model_load_to_managed(struct ds4_model *m,
                              size_t chunk_size_bytes,
                              int verbose);

/* Free the cudaMallocManaged region populated by ds4_model_load_to_managed.
 * Must be called BEFORE ds4_model_close (the close path is pure C and
 * cannot link against cudart). No-op if base is NULL. */
void ds4_model_managed_free(struct ds4_model *m);

/* Device pointer for a tensor's raw bytes inside the managed region.
 * Returns NULL if load_to_managed was not called or t is NULL.
 * Equivalent to (uint8_t*)m->weights_managed_base + t->abs_offset. */
const void *ds4_tensor_managed_ptr(const struct ds4_model *m,
                                   const struct ds4_tensor *t);

/* CUDA-visible pointer for the active weight backend:
 *   managed     -> weights_managed_base + tensor->abs_offset
 *   mmap_direct -> mmap_ptr + tensor->abs_offset
 *
 * mmap_direct is experimental and should be enabled only via explicit
 * server/CLI environment plumbing while validating bandwidth and page-fault
 * behavior. */
const void *ds4_tensor_device_ptr(const struct ds4_model *m,
                                  const struct ds4_tensor *t);

void ds4_model_set_weight_backend(struct ds4_model *m,
                                  enum ds4_weight_backend backend);
enum ds4_weight_backend ds4_model_weight_backend(const struct ds4_model *m);
const char *ds4_weight_backend_name(enum ds4_weight_backend backend);

/* ------------------------------------------------------------------ */
/* Per-session decode/prefill state — KV cache + compressor state      */
/* + cross-layer HC residual + activation arena. Designed as a flat    */
/* C struct so the graph executor can pass &session->layers[il] into   */
/* per-layer launchers without an extra dereference layer.             */
/*                                                                     */
/* Geometry (hardcoded against DeepSeek-V4-Flash; see ds4/ds4.c:407   */
/* ds4_layer_compress_ratio):                                          */
/*   N_LAYER = 43                                                      */
/*   layer 0,1     : ratio 0  (dense raw SWA only, no compressor)      */
/*   layer >=2 even: ratio 4  (attn compressor + indexer compressor)   */
/*   layer >=2 odd : ratio 128 (attn compressor only, no indexer)      */
/*                                                                     */
/* Storage dtype rationale: raw_kv / comp_kv are fp32 device buffers   */
/* matching cuda/flash_attn.cu's launch_flash_attn_decode_*_f32 ABI    */
/* (the kernel applies the f16 round-trip on the read side; producers  */
/* also round-trip on insert per ds4.c:6353-6363). The fp32 footprint  */
/* is 2x the fp16 lower bound; budget at max_context=1M still fits in  */
/* the 32 GB post-weights arena (raw 5.4 MB + comp ratio-4 11 GB +     */
/* comp ratio-128 168 MB + indexer 2.7 GB ~= 14 GB).                   */
/* ------------------------------------------------------------------ */

#define DS4_N_LAYER          43u
#define DS4_N_EMBD          4096u
#define DS4_N_HEAD            64u
#define DS4_N_HEAD_DIM       512u
#define DS4_N_SWA            128u
#define DS4_N_INDEXER_HEAD    64u
#define DS4_N_INDEXER_HEAD_DIM 128u
#define DS4_N_INDEXER_TOP_K  512u
#define DS4_N_HC               4u

/* Per-layer KV state. All `*` pointers are device (cudaMalloc'd into
 * a single arena owned by ds4_session_state). When a layer doesn't use
 * a particular buffer (e.g. layers 0,1 have no compressor; ratio-128
 * layers have no indexer) the corresponding pointer is NULL. */
struct ds4_layer_state {
    int il;                           /* layer index 0..42 */
    int compress_ratio;               /* 0 (il<2) | 4 (even>=2) | 128 (odd) */
    int has_indexer;                  /* 1 iff compress_ratio == 4 */

    /* Raw SWA ring (all layers). Contiguous fp32 [n_raw * head_dim].
     * Producer: kv_cache_push_raw equivalent (memmove sliding once
     * n_raw == cap_raw, see ds4.c:6351-6363). cap_raw = N_SWA = 128. */
    float *raw_kv;                    /* device fp32 [cap_raw * HEAD_DIM] */
    int cap_raw;
    int n_raw;

    /* Compressed attention KV (layers >=2 only). Capacity sized for
     * the configured max_context: ceil(max_ctx/ratio) + 2 emit slots
     * (matches ds4.c:6310). NULL for layers 0,1. */
    float *comp_kv;                   /* device fp32 [cap_comp * HEAD_DIM] */
    int cap_comp;
    int n_comp;

    /* Streaming compressor state (layers >=2 only).  Two parallel
     * buffers (kv + score), sized [coff*ratio rows, coff*head_dim cols]
     * fp32. coff = 2 for ratio=4, coff = 1 for ratio=128.
     * - ratio=4  (attn) : 8 rows x 1024 cols = 8192 fp32 = 32 KB each.
     * - ratio=128 (attn): 128 rows x 512 cols = 65536 fp32 = 256 KB each.
     * NULL for layers 0,1. score buffer must be filled with -INF on
     * reset (matches ds4.c:6318-6320).
     * Reference: commit 6c6fdf8 + cuda/compressor.cuh contract. */
    float *attn_state_kv;             /* device fp32 */
    float *attn_state_score;          /* device fp32 */

    /* Indexer compressor (ratio==4 layers only — even layers >=2).
     * Same coff/state shape as attn but with INDEXER_HEAD_DIM = 128:
     *   state buffers: [coff*ratio, coff*128] = [8, 256] = 2048 fp32 = 8 KB each.
     *   index_comp_kv: [cap_comp, INDEXER_HEAD_DIM] fp32.
     * NULL for ratio-128 layers (and layers 0,1). */
    float *index_comp_kv;             /* device fp32 [cap_comp * INDEXER_HEAD_DIM] */
    int n_index_comp;
    float *index_state_kv;            /* device fp32 */
    float *index_state_score;         /* device fp32 */

    /* Optional model/session-resident warp-native SoA mirror for this
     * layer's ffn_down_exps Q2_K tensor. These pointers are NULL unless
     * an opt-in load/init path builds the layout:
     *   scales [n_experts][out_dim][16][n_blocks_in]
     *   qs     [n_experts][out_dim][64][n_blocks_in]
     *   d      [n_experts][out_dim][n_blocks_in]
     *   dmin   [n_experts][out_dim][n_blocks_in]
     */
    uint8_t  *ffn_down_soa_scales;    /* device uint8 */
    uint8_t  *ffn_down_soa_qs;        /* device uint8 */
    uint16_t *ffn_down_soa_d;         /* device fp16 bits */
    uint16_t *ffn_down_soa_dmin;      /* device fp16 bits */

    /* Optional model-resident SoA v2 mirror for this layer's IQ2_XXS
     * ffn_gate_exps + ffn_up_exps tensors. NULL unless the loaded GGUF
     * contains blk.<il>.ffn_{gate,up}_exps_soa_v2.weight,
     * in which case inference_engine.cu slices these pointers from the
     * tensor blob via ds4_iq2_xxs_soa_v2_layout(). Shapes:
     *   qs : uint16 [n_experts][out_dim=FF_EXP][32][in_dim/256]
     *   d  : uint16 [n_experts][out_dim=FF_EXP][in_dim/256]
     */
    uint16_t *ffn_gate_soa_qs;
    uint16_t *ffn_gate_soa_d;
    uint16_t *ffn_up_soa_qs;
    uint16_t *ffn_up_soa_d;
};

/* Top-level session state.  Holds all device buffers needed by the
 * decode-graph executor for a single inference session.  Owns one
 * cudaMalloc'd arena (state_arena_base / state_arena_bytes) carved
 * into the per-layer pointers + cross-layer HC residual + activation
 * scratch. Single allocation = no fragmentation, single cudaFree. */
struct ds4_session_state {
    int max_context;                  /* configured ctx capacity (tokens) */
    int pos;                          /* next-token position (0-based) */
    int n_tokens_processed;           /* tokens consumed since reset */

    struct ds4_layer_state layers[DS4_N_LAYER];

    /* Cross-layer HC residual ping/pong buffers: [N_HC=4, N_EMBD=4096]
     * fp32 each. Maintained across layers within one token (carries the
     * hyper-connection stream from layer to layer; reset to zero per new
     * sequence). */
    float *residual_hc;               /* device fp32 [N_HC * N_EMBD] = 16384 */
    float *residual_hc_scratch;       /* device fp32 [N_HC * N_EMBD] = 16384 */

    /* Activation arena: scratch for per-token kernel intermediates
     * (Q proj fan-out, attn heads, RMSNorm temp, MoE mid, etc.). The
     * graph executor sub-allocates this as a bump pointer per layer
     * — caller is expected to know the per-layer max usage. We size
     * it to cover the largest single kernel scratch in the forward path
     * (routed_mid_all + heads + comp scratch ≈ 128 KB peak); bump
     * to 16 MB to give comfortable headroom for FFN expert chunks
     * and future fused kernels. */
    float *activation_arena;          /* device fp32 [arena_size/4 floats] */
    size_t arena_size;                /* bytes */

    /* Single-allocation backing storage for everything above (except
     * the C-side struct itself).  All `*` pointers in ds4_layer_state /
     * residual_hc / residual_hc_scratch / activation_arena point inside
     * this arena. One cudaMalloc, one cudaFree on tear-down. */
    void  *state_arena_base;
    size_t state_arena_bytes;
};

/* Allocate a session state for `max_context` tokens. Performs one
 * cudaMalloc, partitions the buffer across all 43 layers + cross-layer
 * state + activation arena, zero-fills KV/HC buffers and -INF-fills
 * compressor score state (matching ds4.c:6318-6320 init). Returns 0 on
 * success, negative on failure. On success `out->state_arena_base`
 * holds the device-managed allocation; caller must call
 * ds4_session_state_free to release. */
int  ds4_session_state_alloc(struct ds4_session_state *out,
                             int max_context);

/* Free the device arena owned by a session state. Safe on a zeroed
 * struct (no-op if base == NULL). After return, `*s` is zeroed. */
void ds4_session_state_free(struct ds4_session_state *s);

/* Free optional per-layer resident MoE-down SoA buffers and NULL their
 * pointers. Safe on zeroed state and on layers where the pointers are
 * already NULL. ds4_session_state_free calls this automatically. */
void ds4_session_state_free_resident_moe_down_soa(struct ds4_session_state *s);

/* Reset session counters and re-initialize KV/state buffers for a new
 * sequence WITHOUT freeing the arena. Zeros raw/comp KV cursors and
 * residual_hc; -INF-fills score buffers; resets pos / n_tokens. The
 * arena pointer set is preserved. O(state size on device). */
void ds4_session_state_reset(struct ds4_session_state *s);

/* Total device bytes a freshly-allocated state of the given context
 * would consume. Pure size calculator (no allocation, no CUDA call);
 * useful for the residency budget check before deciding max_context. */
size_t ds4_session_state_arena_bytes(int max_context);

/* ------------------------------------------------------------------ */
/* Quant block layout (declarations only; dequant lives in cuda/).    */
/* ------------------------------------------------------------------ */

#define DS4_QK_K 256

/* Q8_0: 32 elements, 34 bytes. ds4_metal.m:1172–1175, dense.metal:759. */
struct ds4_block_q8_0 {
    uint16_t d;          /* fp16 scale, offset 0 */
    int8_t   qs[32];     /* quants, offset 2 */
};                       /* total 34 bytes */

/* Q2_K: 256 elements, 84 bytes. ds4.c:128–133 (CPU), moe.metal:94–99. */
struct ds4_block_q2_K {
    uint8_t  scales[16]; /* low 4b = scale, high 4b = min, offset 0 */
    uint8_t  qs[64];     /* 2-bit quants × 256, offset 16 */
    uint16_t d;          /* fp16 super-scale, offset 80 */
    uint16_t dmin;       /* fp16 super-min,   offset 82 */
};                       /* total 84 bytes */

/* IQ2_XXS: 256 elements, 66 bytes. ds4.c:148–151, moe.metal:108–111. */
struct ds4_block_iq2_xxs {
    uint16_t d;          /* fp16 super-scale, offset 0 */
    uint16_t qs[32];     /* 8 ib32 sub-blocks × (4 × u16), offset 2 */
};                       /* total 66 bytes */

/* Q8_K: 256 elements, 292 bytes. activation-only (NOT a weight format).
 * Listed for completeness; only used at runtime when quantizing
 * activations for IQ2_XXS / Q2_K dot product. ds4.c:142–146. */
struct ds4_block_q8_K {
    float    d;
    int8_t   qs[256];
    int16_t  bsums[16];
};                       /* total 292 bytes */

#ifdef __cplusplus
}
#endif

#endif /* DS4CUDA_H */
