// forward_layer.cu — streaming forward for one DeepSeek-V4 layer.
//
// Dispatches all 43 layers (il=0..42):
//
//   - layer 0, 1 (ratio-0): identical 31-stage pipeline, just with
//                           blk.<il>.* weight names.  Hash-router branch
//                           in the FFN half (il<3, DS4_N_HASH_LAYER=3).
//
//   - layer 2 (ratio-4):    the attention half adds a 3-step bracket
//                           between Step 11 (KVcur) and Step 13 (flash):
//
//       12a. attn_compressor_decode_step    (cite ds4.c:7093)
//             input = attn_norm; weights blk.2.attn_compressor_*.
//             writes state row; on emit ((pos+1)%4==0) produces a 512-fp32
//             KVcompress that we push to comp_kv ring.
//       12b. indexer_compressor_decode_step (cite ds4.c:7111)
//             same shape with INDEXER_HEAD_DIM=128, weights
//             blk.2.indexer_compressor_*.  On emit pushes to
//             index_comp_kv ring.
//       12c. indexer_allowed_short_circuit  (cite ds4.c:6926)
//             n_index_comp <= 512 -> all-1 mask.  For 10-token "Hello"
//             prompt n_index_comp peaks at 2.
//
//      Step 13 then becomes launch_flash_attn_decode_mixed_f32 instead of
//      decode_raw, with raw_kv + comp_kv + comp_allowed.
//
//   - layer 3..42:          two sub-cases by parity (ds4.c:407-411):
//       odd  il (ratio=128): step 12a only (attn compressor, coff=1,
//             out=512, ratio=128); no indexer (cite ds4.c:7109 guard).
//             For "Hello" 10-token prompt, no emit happens (10 < 128) so
//             n_comp=0 and decode_mixed degenerates into pure raw SWA.
//       even il (ratio=4):  same 12a+12b+12c bracket as layer 2, just with
//             top-k router weights instead of hash router.
//
//      FFN half: layers 0,1,2 use the hash router (ffn_gate_tid2eid look-up,
//      cite ds4.c:5119).  Layers 3..42 use the biased top-k router
//      (probs + exp_probs_b → argsort → top 6, cite ds4.c:5217).
//
// All other steps (1..11, 14..31) are identical across all 43 layers.
//
// Memory:
//   - All small intermediates ( <= 1 MiB ) live inside the session
//     activation_arena (16 MiB).  We bump-allocate from a local cursor.
//     Layer 2 adds a few small carve-outs:
//       - emit_kv          [HEAD_DIM=512]                   = 2 KiB
//       - indexer_emit_kv  [INDEXER_HEAD_DIM=128]           = 0.5 KiB
//       - comp_allowed     [cap_comp <= 64 at max_ctx=64]   = 256 B
//
// Synchronization: this function enqueues work on the caller-provided
// stream and returns without a per-layer stream synchronize.  Callers that
// need host-visible results must synchronize at the token/test boundary.

#include "forward_layer.cuh"

#include "common.cuh"
#include "hc_sinkhorn.cuh"
#include "norm.cuh"
#include "dense_q8.cuh"
#include "rope.cuh"
#include "fp8_kv.cuh"
#include "flash_attn.cuh"
#include "attn_out.cuh"
#include "glu.cuh"
#include "elementwise.cuh"
#include "router.cuh"
#include "moe_iq2_pair.cuh"
#include "moe_q2k_sum6.cuh"
#include "compressor.cuh"
#include "indexer_allowed.cuh"
#include "perf_timeline.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>

namespace ds4cuda {
namespace {

#define FCK(stmt) do {                                                     \
    cudaError_t _e = (stmt);                                               \
    if (_e != cudaSuccess) {                                               \
        std::fprintf(stderr, "forward_layer: CUDA error %s (%s) at %s:%d\n", \
                     cudaGetErrorName(_e), cudaGetErrorString(_e),         \
                     __FILE__, __LINE__);                                  \
        return -1;                                                         \
    }                                                                      \
} while (0)

#define PCK(stmt) do {                                                     \
    int _rc = (stmt);                                                       \
    if (_rc != 0) {                                                         \
        std::fprintf(stderr, "forward_layer: perf timeline error at %s:%d\n", \
                     __FILE__, __LINE__);                                  \
        return -1;                                                          \
    }                                                                       \
} while (0)

#define PERF_STAGE_BEGIN(var, name, cat, in_s, out_s, weight_s, wb, ib, ob, sb, nk, note_s) \
    ds4cuda::ds4_perf_marker var = {};                                      \
    do {                                                                    \
        if (ds4cuda::ds4_perf_timeline_is_enabled()) {                      \
            ds4cuda::ds4_perf_stage _perf_stage = {};                       \
            _perf_stage.token = pos;                                        \
            _perf_stage.layer = il;                                         \
            _perf_stage.stage = (name);                                     \
            _perf_stage.category = (cat);                                   \
            _perf_stage.input_shape = (in_s);                               \
            _perf_stage.output_shape = (out_s);                             \
            _perf_stage.weight_shape = (weight_s);                          \
            _perf_stage.weight_bytes = (uint64_t)(wb);                      \
            _perf_stage.input_bytes = (uint64_t)(ib);                       \
            _perf_stage.output_bytes = (uint64_t)(ob);                      \
            _perf_stage.scratch_bytes = (uint64_t)(sb);                     \
            _perf_stage.kernels = (nk);                                     \
            _perf_stage.notes = (note_s);                                   \
            PCK(ds4cuda::ds4_perf_timeline_stage_begin(&_perf_stage,        \
                                                       stream, &var));      \
        }                                                                   \
    } while (0)

#define PERF_STAGE_END(var) do {                                            \
    if (ds4cuda::ds4_perf_timeline_is_enabled()) {                          \
        PCK(ds4cuda::ds4_perf_timeline_stage_end(&var, stream));            \
    }                                                                       \
} while (0)

// ---------------------------------------------------------------------
// SWA eviction support kernels.
//
// When the raw_kv ring is full (n_raw == cap_raw == DS4_N_SWA == 128),
// ds4.c:6385 memmove-shifts entries [1, cap_raw) to [0, cap_raw-1) and
// drops the oldest, then writes the new entry at slot (cap_raw - 1).
// We mirror that here.
//
// K rows in raw_kv are POST-RoPE (see ds4.c:7085-7088 and the CUDA
// forward path Step 10 -> Step 11 -> Step 12 ordering in this file:
// tail-RoPE was applied at the row's original absolute pos at the time
// of insertion). After eviction the surviving rows still encode RoPE for
// their original absolute positions; the new Q for the current token
// applies RoPE at its absolute pos. attn_sinks is a per-head bias
// tensor (NOT a row of K/V) and is read from the GGUF unchanged on every
// call -- not affected by eviction.
// ---------------------------------------------------------------------

// Forward-shift n_dst_rows of `head_dim` floats: dst row r := src row (r+1)
// for r in [0, n_dst_rows).  Applies f16 round-trip on the read so the
// value bit-pattern in raw_kv stays an f16 lane after the shift (matches
// ds4.c:6380 / 6389: rows are stored as f16-roundtripped fp32).
//
// Single-block sequential implementation: __syncthreads() between rows
// guarantees row r is fully written before any thread starts reading
// row r+1, which is the canonical ds4 memmove direction (low->high) and
// correct under in-place overlap.  blockDim.x threads stride through
// head_dim columns within each row.  cap_raw == 128 -> 127 dst rows
// × 512 cols = 65024 fp32 ops; trivially memory-bandwidth bound in one
// SM block.
__global__ void k_swa_shift_f16_round(float *base, int head_dim,
                                      int n_dst_rows) {
    const int tid = threadIdx.x;
    for (int r = 0; r < n_dst_rows; ++r) {
        const float *src = base + (size_t)(r + 1) * head_dim;
        float       *dst = base + (size_t)r       * head_dim;
        for (int j = tid; j < head_dim; j += blockDim.x) {
            // f16 round-trip mirrors ds4.c:6380. The source already
            // passed through f16 round-trip on its prior insert and
            // f16(f16(x))==f16(x) is idempotent, so this is functionally
            // equivalent to a plain copy, but we apply the round-trip
            // anyway to make the in-cache invariant explicit and to
            // mirror ds4.c kv_cache_push_raw semantics 1:1.
            const float v = src[j];
            dst[j] = __half2float(__float2half(v));
        }
        __syncthreads();   // make row r writes visible before row r+1 reads
    }
}

// Write a single new row at `base + slot * head_dim` from `src`, with
// f16 round-trip on each element (matches ds4.c:6380 / 6389). One block,
// blockDim.x threads strided over head_dim columns.
__global__ void k_swa_push_f16_round(float *base, int slot,
                                     const float *src, int head_dim) {
    const int tid = threadIdx.x;
    float *dst = base + (size_t)slot * head_dim;
    for (int j = tid; j < head_dim; j += blockDim.x) {
        const float v = src[j];
        dst[j] = __half2float(__float2half(v));
    }
}

// Round up `n` to a 256-byte boundary (matches session_state.cu's
// alignment so all sub-buffers stay 256-aligned).
static inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

static inline size_t fp32_bytes(size_t n) { return n * sizeof(float); }
static inline size_t i32_bytes(size_t n) { return n * sizeof(int32_t); }
static inline size_t q8_weight_bytes(int n_rows, int n_cols) {
    return (size_t)n_rows * (size_t)(n_cols / 32) * sizeof(block_q8_0);
}
static inline size_t q8_scratch_bytes(int n_cols) {
    return (size_t)n_cols * sizeof(int8_t) + (size_t)(n_cols / 32) * sizeof(float);
}
static inline size_t q8k_scratch_bytes(int n_vec, int n_cols) {
    const int blocks = (n_cols / 256) * n_vec;
    return (size_t)blocks * 256 * sizeof(int8_t)
         + (size_t)blocks * sizeof(float)
         + (size_t)blocks * 16 * sizeof(int16_t);
}
static inline size_t iq2_selected_weight_bytes(int n_used, int out_dim, int in_dim) {
    return (size_t)n_used * (size_t)out_dim * (size_t)(in_dim / 256)
         * sizeof(block_iq2_xxs);
}
static inline size_t q2k_selected_weight_bytes(int n_used, int out_dim, int in_dim) {
    return (size_t)n_used * (size_t)out_dim * (size_t)(in_dim / 256)
         * sizeof(block_q2_K);
}

// Bump-pointer helper: hand out `bytes` from the activation arena, return
// a typed pointer, and advance the cursor (in bytes from the arena base).
// Returns NULL if the arena would overflow.
static float *arena_carve(uint8_t *base, size_t total_bytes,
                          size_t *cursor, size_t bytes,
                          const char *tag) {
    size_t pos = *cursor;
    size_t need = align256(bytes);
    if (pos + need > total_bytes) {
        std::fprintf(stderr,
                     "forward_layer: activation arena overflow at '%s' "
                     "(have=%zu need=%zu used=%zu cap=%zu)\n",
                     tag, bytes, need, pos, total_bytes);
        return nullptr;
    }
    *cursor = pos + need;
    return reinterpret_cast<float *>(base + pos);
}

// Find a tensor by name; abort with a clear message if not found.
static const struct ds4_tensor *find_required(const struct ds4_model *m,
                                              const char *name) {
    const struct ds4_tensor *t = ds4_model_find_tensor(m, name);
    if (!t) {
        std::fprintf(stderr, "forward_layer: required tensor '%s' not found\n",
                     name);
    }
    return t;
}

// Find a tensor by name; return NULL silently if not present. Use this
// for tensors that the runtime can substitute with an alternate path
// (e.g. ffn_down_exps.weight is absent in moe_down SoA v2 replace-mode GGUFs;
// the SoA-direct path uses L->ffn_down_soa_* instead).
static const struct ds4_tensor *find_optional(const struct ds4_model *m,
                                              const char *name) {
    return ds4_model_find_tensor(m, name);
}

// Read tensor's CUDA-visible pointer for the active weight backend.
static const void *managed_ptr(const struct ds4_model *m,
                               const struct ds4_tensor *t) {
    const void *p = ds4_tensor_device_ptr(m, t);
    if (!p) {
        std::fprintf(stderr,
                     "forward_layer: tensor '%s' has no device ptr "
                     "(backend=%s)\n",
                     t->name,
                     ds4_weight_backend_name(ds4_model_weight_backend(m)));
    }
    return p;
}

// Geometry constants from include/ds4cuda.h.
static constexpr int kN_EMBD     = (int)DS4_N_EMBD;       // 4096
static constexpr int kN_HC       = (int)DS4_N_HC;         // 4
static constexpr int kHC_DIM     = kN_HC * kN_EMBD;       // 16384
static constexpr int kN_HEAD     = (int)DS4_N_HEAD;       // 64
static constexpr int kHEAD_DIM   = (int)DS4_N_HEAD_DIM;   // 512
static constexpr int kN_ROT      = 64;
static constexpr int kQ_DIM      = kN_HEAD * kHEAD_DIM;   // 32768
static constexpr int kKV_DIM     = kHEAD_DIM;             // 1 KV head, mla
static constexpr int kQ_LORA_RANK = 1024;
static constexpr int kN_OUT_GROUP = 8;
static constexpr int kN_LORA_O    = 1024;
static constexpr int kATTN_LOW_OUT = kN_OUT_GROUP * kN_LORA_O;   // 8192
static constexpr int kFF_EXP    = 2048;
static constexpr int kN_EXPERT  = 256;
static constexpr int kN_USED    = 6;
static constexpr int kN_SINKHORN_ITER = 20;
static constexpr float kRMS_EPS = 1.0e-6f;
static constexpr float kSWIGLU_CLAMP = 10.0f;
static constexpr float kEXPERT_WEIGHT_SCALE = 1.5f;
static constexpr float kEXPERT_SUM_FLOOR = 6.103515625e-5f;

// Layer compressor / indexer geometry.  cite include/ds4cuda.h
// + cuda/compressor.cuh.
//
// ratio-4 (il=2,4,...,42): coff=2, comp_width=1024, state=[8 rows,1024 cols]=8192
// ratio-128 (il=3,5,...,41): coff=1, comp_width=512, state=[128 rows,512 cols]=65536
static constexpr int kCOMPRESS_RATIO_L2  = 4;
static constexpr int kCOMPRESS_RATIO_L3  = 128;
static constexpr int kCOFF_L2            = 2;                                 // ratio==4 -> coff=2
static constexpr int kCOFF_L3            = 1;                                 // ratio==128 -> coff=1
static constexpr int kATTN_COMP_WIDTH_L2 = kCOFF_L2 * kHEAD_DIM;              // 1024
static constexpr int kATTN_COMP_STATE_L2 = kCOFF_L2 * kCOMPRESS_RATIO_L2 * kATTN_COMP_WIDTH_L2; // 8192
static constexpr int kATTN_COMP_WIDTH_L3 = kCOFF_L3 * kHEAD_DIM;              // 512
static constexpr int kATTN_COMP_STATE_L3 = kCOFF_L3 * kCOMPRESS_RATIO_L3 * kATTN_COMP_WIDTH_L3; // 65536
static constexpr int kIDX_HEAD_DIM       = (int)DS4_N_INDEXER_HEAD_DIM;       // 128
static constexpr int kIDX_COMP_WIDTH     = kCOFF_L2 * kIDX_HEAD_DIM;          // 256
static constexpr int kIDX_COMP_STATE     = kCOFF_L2 * kCOMPRESS_RATIO_L2 * kIDX_COMP_WIDTH;  // 2048
static constexpr int kINDEXER_TOP_K      = (int)DS4_N_INDEXER_TOP_K;          // 512
static constexpr int kN_INDEXER_HEAD     = (int)DS4_N_INDEXER_HEAD;           // 64
static constexpr int kIDX_Q_DIM          = kN_INDEXER_HEAD * kIDX_HEAD_DIM;   // 8192
static constexpr int kQ8_SCRATCH_ELEMS   = kN_OUT_GROUP * (kHEAD_DIM * (kN_HEAD / kN_OUT_GROUP)); // 32768
static constexpr int kQ8_SCRATCH_BLOCKS  = kQ8_SCRATCH_ELEMS / 32;
static constexpr int kQ8K_SCRATCH_BLOCKS = kN_USED * (kFF_EXP / 256);         // MoE down max: 6 * 8
static constexpr int kCOMP_SCRATCH_WIDTH = kATTN_COMP_WIDTH_L2;              // max(coff*head_dim) = 1024

// ds4.c:407-411 ds4_layer_compress_ratio.
__host__ static inline int layer_compress_ratio_il(int il) {
    if (il < 2) return 0;
    return (il & 1) == 0 ? 4 : 128;
}

// Format a "blk.<il>.<suffix>" tensor name into `buf`.  buf must be at
// least 64 bytes.
static void layer_tensor_name(char *buf, size_t buflen, int il, const char *suffix) {
    std::snprintf(buf, buflen, "blk.%d.%s", il, suffix);
}

} // namespace

// ---------------------------------------------------------------------
//  Public entry point: generic per-layer forward.
// ---------------------------------------------------------------------

int ds4_forward_layer(const struct ds4_model      *m,
                      struct ds4_session_state    *s,
                      int                          il,
                      int                          token_id,
                      int                          pos,
                      const float                 *input_residual_hc,
                      float                       *output_residual_hc,
                      cudaStream_t                 stream) {
    if (!m || !s || !input_residual_hc || !output_residual_hc) {
        std::fprintf(stderr, "forward_layer: NULL arg\n");
        return -1;
    }
    if (il < 0 || il >= (int)DS4_N_LAYER) {
        std::fprintf(stderr, "forward_layer: il=%d not in [0,%d)\n",
                     il, (int)DS4_N_LAYER);
        return -1;
    }
    struct ds4_layer_state *L = &s->layers[il];
    const int expected_ratio = layer_compress_ratio_il(il);
    if (L->compress_ratio != expected_ratio) {
        std::fprintf(stderr,
                     "forward_layer: layer %d expected ratio=%d, got %d\n",
                     il, expected_ratio, L->compress_ratio);
        return -1;
    }
    // raw_kv ring is bounded to cap_raw == DS4_N_SWA == 128.  Once full,
    // Step 12 below performs the SWA eviction (drop oldest, shift, append
    // new at last slot) — matches ds4.c:6385 kv_cache_push_raw memmove
    // path.  No early-abort here: SWA eviction is the contract.
    if (L->cap_raw <= 0) {
        std::fprintf(stderr,
                     "forward_layer: layer %d invalid cap_raw=%d\n",
                     il, L->cap_raw);
        return -1;
    }
    // Per-layer dispatch flags.
    const bool has_compressor   = (L->compress_ratio != 0);     // il>=2
    const bool has_indexer      = (L->has_indexer != 0);        // il>=2 && ratio==4
    const bool use_topk_router  = (il >= 3);                    // DS4_N_HASH_LAYER=3
    const int  comp_ratio       = L->compress_ratio;
    if (has_indexer && comp_ratio != kCOMPRESS_RATIO_L2) {
        std::fprintf(stderr,
                     "forward_layer: il=%d has_indexer=1 but ratio=%d (expect 4)\n",
                     il, comp_ratio);
        return -1;
    }
    // ----- 1) Look up all weight tensors --------------------------------
    char nm[64];
    layer_tensor_name(nm, sizeof(nm), il, "hc_attn_fn.weight");
    const struct ds4_tensor *t_hc_attn_fn    = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "hc_attn_scale.weight");
    const struct ds4_tensor *t_hc_attn_scale = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "hc_attn_base.weight");
    const struct ds4_tensor *t_hc_attn_base  = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_norm.weight");
    const struct ds4_tensor *t_attn_norm     = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_q_a.weight");
    const struct ds4_tensor *t_attn_q_a      = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_q_a_norm.weight");
    const struct ds4_tensor *t_attn_q_a_norm = find_required(m, nm);
    // NOTE: GGUF naming for the KV LoRA-down weight is `attn_kv.weight`
    // (not `attn_kv_a.weight`).  Confirmed via `ds4cuda_tools dump-tensors`
    // — only the *norm* tensor uses the `_a_` infix.
    layer_tensor_name(nm, sizeof(nm), il, "attn_kv.weight");
    const struct ds4_tensor *t_attn_kv_a     = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_kv_a_norm.weight");
    const struct ds4_tensor *t_attn_kv_a_norm= find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_q_b.weight");
    const struct ds4_tensor *t_attn_q_b      = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_sinks.weight");
    const struct ds4_tensor *t_attn_sinks    = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_output_a.weight");
    const struct ds4_tensor *t_attn_output_a = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "attn_output_b.weight");
    const struct ds4_tensor *t_attn_output_b = find_required(m, nm);

    layer_tensor_name(nm, sizeof(nm), il, "hc_ffn_fn.weight");
    const struct ds4_tensor *t_hc_ffn_fn     = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "hc_ffn_scale.weight");
    const struct ds4_tensor *t_hc_ffn_scale  = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "hc_ffn_base.weight");
    const struct ds4_tensor *t_hc_ffn_base   = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "ffn_norm.weight");
    const struct ds4_tensor *t_ffn_norm      = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "ffn_gate_shexp.weight");
    const struct ds4_tensor *t_ffn_gate_shexp= find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "ffn_up_shexp.weight");
    const struct ds4_tensor *t_ffn_up_shexp  = find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "ffn_down_shexp.weight");
    const struct ds4_tensor *t_ffn_down_shexp= find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "ffn_gate_inp.weight");
    const struct ds4_tensor *t_ffn_gate_inp  = find_required(m, nm);
    // Hash router (il<3) uses ffn_gate_tid2eid; top-k (il>=3) uses
    // exp_probs_b.bias.  The other tensor doesn't exist on disk for that
    // layer, so we conditionally fetch.
    const struct ds4_tensor *t_ffn_gate_tid2eid = nullptr;
    const struct ds4_tensor *t_ffn_exp_probs_b = nullptr;
    if (use_topk_router) {
        layer_tensor_name(nm, sizeof(nm), il, "exp_probs_b.bias");
        t_ffn_exp_probs_b = find_required(m, nm);
    } else {
        layer_tensor_name(nm, sizeof(nm), il, "ffn_gate_tid2eid.weight");
        t_ffn_gate_tid2eid = find_required(m, nm);
    }
    /* When the GGUF was repacked in replace-mode (gate+up SoA v2 / moe_down SoA v2),
     * the original blk.<il>.ffn_{down,gate,up}_exps.weight tensors are
     * absent; only the SoA v2 variants are present. In that case the
     * matching L->ffn_*_soa_* pointers are populated (wired at engine
     * init in src/server/inference_engine.cu) and the SoA launcher
     * path doesn't need AoS pointers. Make the AoS lookups optional then. */
    const bool soa_active =
        L->ffn_down_soa_scales && L->ffn_down_soa_qs &&
        L->ffn_down_soa_d      && L->ffn_down_soa_dmin;
    const bool soa_gate_up_active =
        L->ffn_gate_soa_qs && L->ffn_gate_soa_d &&
        L->ffn_up_soa_qs   && L->ffn_up_soa_d;
    layer_tensor_name(nm, sizeof(nm), il, "ffn_gate_exps.weight");
    const struct ds4_tensor *t_ffn_gate_exps =
        soa_gate_up_active ? find_optional(m, nm) : find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "ffn_up_exps.weight");
    const struct ds4_tensor *t_ffn_up_exps =
        soa_gate_up_active ? find_optional(m, nm) : find_required(m, nm);
    layer_tensor_name(nm, sizeof(nm), il, "ffn_down_exps.weight");
    const struct ds4_tensor *t_ffn_down_exps =
        soa_active ? find_optional(m, nm) : find_required(m, nm);

    // Compressor (il>=2) and indexer (ratio==4 only) weights.
    const struct ds4_tensor *t_attn_comp_kv    = nullptr;
    const struct ds4_tensor *t_attn_comp_gate  = nullptr;
    const struct ds4_tensor *t_attn_comp_ape   = nullptr;
    const struct ds4_tensor *t_attn_comp_norm  = nullptr;
    const struct ds4_tensor *t_idx_comp_kv     = nullptr;
    const struct ds4_tensor *t_idx_comp_gate   = nullptr;
    const struct ds4_tensor *t_idx_comp_ape    = nullptr;
    const struct ds4_tensor *t_idx_comp_norm   = nullptr;
    // Long-prompt indexer scoring weights.
    const struct ds4_tensor *t_idx_attn_q_b    = nullptr;
    const struct ds4_tensor *t_idx_proj        = nullptr;
    if (has_compressor) {
        layer_tensor_name(nm, sizeof(nm), il, "attn_compressor_kv.weight");
        t_attn_comp_kv   = find_required(m, nm);
        layer_tensor_name(nm, sizeof(nm), il, "attn_compressor_gate.weight");
        t_attn_comp_gate = find_required(m, nm);
        layer_tensor_name(nm, sizeof(nm), il, "attn_compressor_ape.weight");
        t_attn_comp_ape  = find_required(m, nm);
        layer_tensor_name(nm, sizeof(nm), il, "attn_compressor_norm.weight");
        t_attn_comp_norm = find_required(m, nm);
    }
    if (has_indexer) {
        layer_tensor_name(nm, sizeof(nm), il, "indexer_compressor_kv.weight");
        t_idx_comp_kv    = find_required(m, nm);
        layer_tensor_name(nm, sizeof(nm), il, "indexer_compressor_gate.weight");
        t_idx_comp_gate  = find_required(m, nm);
        layer_tensor_name(nm, sizeof(nm), il, "indexer_compressor_ape.weight");
        t_idx_comp_ape   = find_required(m, nm);
        layer_tensor_name(nm, sizeof(nm), il, "indexer_compressor_norm.weight");
        t_idx_comp_norm  = find_required(m, nm);
        // Long-prompt scoring weights (F16). Tensor names match
        // ds4.c:2595-2596 (blk.<il>.indexer.attn_q_b.weight,
        // blk.<il>.indexer.proj.weight).
        layer_tensor_name(nm, sizeof(nm), il, "indexer.attn_q_b.weight");
        t_idx_attn_q_b = find_required(m, nm);
        layer_tensor_name(nm, sizeof(nm), il, "indexer.proj.weight");
        t_idx_proj     = find_required(m, nm);
    }

    if (!t_hc_attn_fn || !t_hc_attn_scale || !t_hc_attn_base ||
        !t_attn_norm  || !t_attn_q_a || !t_attn_q_a_norm ||
        !t_attn_kv_a || !t_attn_kv_a_norm || !t_attn_q_b ||
        !t_attn_sinks || !t_attn_output_a || !t_attn_output_b ||
        !t_hc_ffn_fn || !t_hc_ffn_scale || !t_hc_ffn_base ||
        !t_ffn_norm || !t_ffn_gate_shexp || !t_ffn_up_shexp ||
        !t_ffn_down_shexp || !t_ffn_gate_inp ||
        (!t_ffn_gate_exps && !soa_gate_up_active) ||
        (!t_ffn_up_exps && !soa_gate_up_active) ||
        (!t_ffn_down_exps && !soa_active)) {
        return -1;
    }
    if (use_topk_router && !t_ffn_exp_probs_b) return -1;
    if (!use_topk_router && !t_ffn_gate_tid2eid) return -1;
    if (has_compressor && (!t_attn_comp_kv || !t_attn_comp_gate ||
                           !t_attn_comp_ape || !t_attn_comp_norm)) {
        return -1;
    }
    if (has_indexer && (!t_idx_comp_kv || !t_idx_comp_gate ||
                        !t_idx_comp_ape || !t_idx_comp_norm ||
                        !t_idx_attn_q_b || !t_idx_proj)) {
        return -1;
    }

    // Pull out the device pointers via the managed region.
    const uint16_t *p_hc_attn_fn    = (const uint16_t *)managed_ptr(m, t_hc_attn_fn);
    const float    *p_hc_attn_scale = (const float    *)managed_ptr(m, t_hc_attn_scale);
    const float    *p_hc_attn_base  = (const float    *)managed_ptr(m, t_hc_attn_base);
    const float    *p_attn_norm_w   = (const float    *)managed_ptr(m, t_attn_norm);
    const block_q8_0 *p_attn_q_a    = (const block_q8_0 *)managed_ptr(m, t_attn_q_a);
    const float    *p_attn_q_a_norm = (const float    *)managed_ptr(m, t_attn_q_a_norm);
    const block_q8_0 *p_attn_kv_a   = (const block_q8_0 *)managed_ptr(m, t_attn_kv_a);
    const float    *p_attn_kv_a_norm= (const float    *)managed_ptr(m, t_attn_kv_a_norm);
    const block_q8_0 *p_attn_q_b    = (const block_q8_0 *)managed_ptr(m, t_attn_q_b);
    const float    *p_attn_sinks    = (const float    *)managed_ptr(m, t_attn_sinks);
    const block_q8_0 *p_attn_output_a = (const block_q8_0 *)managed_ptr(m, t_attn_output_a);
    const block_q8_0 *p_attn_output_b = (const block_q8_0 *)managed_ptr(m, t_attn_output_b);
    const uint16_t *p_hc_ffn_fn     = (const uint16_t *)managed_ptr(m, t_hc_ffn_fn);
    const float    *p_hc_ffn_scale  = (const float    *)managed_ptr(m, t_hc_ffn_scale);
    const float    *p_hc_ffn_base   = (const float    *)managed_ptr(m, t_hc_ffn_base);
    const float    *p_ffn_norm_w    = (const float    *)managed_ptr(m, t_ffn_norm);
    const block_q8_0 *p_ffn_gate_shexp = (const block_q8_0 *)managed_ptr(m, t_ffn_gate_shexp);
    const block_q8_0 *p_ffn_up_shexp   = (const block_q8_0 *)managed_ptr(m, t_ffn_up_shexp);
    const block_q8_0 *p_ffn_down_shexp = (const block_q8_0 *)managed_ptr(m, t_ffn_down_shexp);
    const uint16_t *p_ffn_gate_inp  = (const uint16_t *)managed_ptr(m, t_ffn_gate_inp);
    const int32_t  *p_ffn_tid2eid   = use_topk_router ? nullptr
        : (const int32_t  *)managed_ptr(m, t_ffn_gate_tid2eid);
    const float    *p_ffn_exp_probs_b = use_topk_router
        ? (const float *)managed_ptr(m, t_ffn_exp_probs_b)
        : nullptr;
    /* AoS pointers are only consumed by the AoS launcher branches below;
     * when soa_gate_up_active / soa_active are true the corresponding
     * tensors are absent and these stay nullptr. */
    const block_iq2_xxs *p_ffn_gate_exps = t_ffn_gate_exps
        ? (const block_iq2_xxs *)managed_ptr(m, t_ffn_gate_exps)
        : nullptr;
    const block_iq2_xxs *p_ffn_up_exps = t_ffn_up_exps
        ? (const block_iq2_xxs *)managed_ptr(m, t_ffn_up_exps)
        : nullptr;
    const block_q2_K *p_ffn_down_exps = t_ffn_down_exps
        ? (const block_q2_K *)managed_ptr(m, t_ffn_down_exps)
        : nullptr;

    // Compressor / indexer managed pointers (NULL when unused).
    const uint16_t *p_attn_comp_kv   = nullptr;
    const uint16_t *p_attn_comp_gate = nullptr;
    const uint16_t *p_attn_comp_ape  = nullptr;
    const float    *p_attn_comp_norm = nullptr;
    const uint16_t *p_idx_comp_kv    = nullptr;
    const uint16_t *p_idx_comp_gate  = nullptr;
    const uint16_t *p_idx_comp_ape   = nullptr;
    const float    *p_idx_comp_norm  = nullptr;
    // Long-prompt indexer weights (F16, only on has_indexer layers).
    const uint16_t *p_idx_attn_q_b   = nullptr;
    const uint16_t *p_idx_proj       = nullptr;
    if (has_compressor) {
        p_attn_comp_kv   = (const uint16_t *)managed_ptr(m, t_attn_comp_kv);
        p_attn_comp_gate = (const uint16_t *)managed_ptr(m, t_attn_comp_gate);
        p_attn_comp_ape  = (const uint16_t *)managed_ptr(m, t_attn_comp_ape);
        p_attn_comp_norm = (const float    *)managed_ptr(m, t_attn_comp_norm);
    }
    if (has_indexer) {
        p_idx_comp_kv    = (const uint16_t *)managed_ptr(m, t_idx_comp_kv);
        p_idx_comp_gate  = (const uint16_t *)managed_ptr(m, t_idx_comp_gate);
        p_idx_comp_ape   = (const uint16_t *)managed_ptr(m, t_idx_comp_ape);
        p_idx_comp_norm  = (const float    *)managed_ptr(m, t_idx_comp_norm);
        p_idx_attn_q_b   = (const uint16_t *)managed_ptr(m, t_idx_attn_q_b);
        p_idx_proj       = (const uint16_t *)managed_ptr(m, t_idx_proj);
    }

    if (!p_hc_attn_fn || !p_hc_attn_scale || !p_hc_attn_base ||
        !p_attn_norm_w || !p_attn_q_a || !p_attn_q_a_norm ||
        !p_attn_kv_a || !p_attn_kv_a_norm || !p_attn_q_b ||
        !p_attn_sinks || !p_attn_output_a || !p_attn_output_b ||
        !p_hc_ffn_fn || !p_hc_ffn_scale || !p_hc_ffn_base ||
        !p_ffn_norm_w || !p_ffn_gate_shexp || !p_ffn_up_shexp ||
        !p_ffn_down_shexp || !p_ffn_gate_inp ||
        (!p_ffn_gate_exps && !soa_gate_up_active) ||
        (!p_ffn_up_exps && !soa_gate_up_active) ||
        (!p_ffn_down_exps && !soa_active)) {
        return -1;
    }
    if (use_topk_router && !p_ffn_exp_probs_b) return -1;
    if (!use_topk_router && !p_ffn_tid2eid) return -1;
    if (has_compressor && (!p_attn_comp_kv || !p_attn_comp_gate ||
                           !p_attn_comp_ape || !p_attn_comp_norm)) {
        return -1;
    }
    if (has_indexer && (!p_idx_comp_kv || !p_idx_comp_gate ||
                        !p_idx_comp_ape || !p_idx_comp_norm ||
                        !p_idx_attn_q_b || !p_idx_proj)) {
        return -1;
    }

    // ----- 2) Carve activation arena ------------------------------------
    uint8_t *arena_base = (uint8_t *)s->activation_arena;
    size_t   arena_cap  = s->arena_size;
    size_t   cur        = 0;

    float *d_hc_attn_post_w = arena_carve(arena_base, arena_cap, &cur,
                                          kN_HC * sizeof(float),
                                          "hc_attn_post_w");
    float *d_hc_attn_comb   = arena_carve(arena_base, arena_cap, &cur,
                                          kN_HC * kN_HC * sizeof(float),
                                          "hc_attn_comb");
    float *d_hc_attn_pre    = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "hc_attn_pre");
    // HC sinkhorn multi-CTA: scratches for launch_hc_attn_pre_v2_f32. The
    // RMSNorm scale handoff is a single fp32; the mix handoff is
    // mix_len = n_hc*(2+n_hc) = 24 fp32. Tiny (100 bytes) — arena cost
    // negligible.
    float *d_hc_attn_rms    = arena_carve(arena_base, arena_cap, &cur,
                                          sizeof(float), "hc_attn_rms");
    float *d_hc_attn_mix    = arena_carve(arena_base, arena_cap, &cur,
                                          kN_HC * (2 + kN_HC) * sizeof(float),
                                          "hc_attn_mix");
    float *d_attn_norm      = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "attn_norm");
    float *d_q_lora         = arena_carve(arena_base, arena_cap, &cur,
                                          kQ_LORA_RANK * sizeof(float), "q_lora");
    float *d_q_lora_norm    = arena_carve(arena_base, arena_cap, &cur,
                                          kQ_LORA_RANK * sizeof(float), "q_lora_norm");
    float *d_kv_raw         = arena_carve(arena_base, arena_cap, &cur,
                                          kKV_DIM * sizeof(float), "kv_raw");
    float *d_kv_norm        = arena_carve(arena_base, arena_cap, &cur,
                                          kKV_DIM * sizeof(float), "kv_norm");
    float *d_kv_rope        = arena_carve(arena_base, arena_cap, &cur,
                                          kKV_DIM * sizeof(float), "kv_rope");
    float *d_kv_cur         = arena_carve(arena_base, arena_cap, &cur,
                                          kKV_DIM * sizeof(float), "kv_cur");
    float *d_q_raw          = arena_carve(arena_base, arena_cap, &cur,
                                          kQ_DIM * sizeof(float), "q_raw");
    float *d_q_norm         = arena_carve(arena_base, arena_cap, &cur,
                                          kQ_DIM * sizeof(float), "q_norm");
    float *d_q_cur          = arena_carve(arena_base, arena_cap, &cur,
                                          kQ_DIM * sizeof(float), "q_cur");
    float *d_kqv_out        = arena_carve(arena_base, arena_cap, &cur,
                                          kQ_DIM * sizeof(float), "kqv_out");
    float *d_kqv_back       = arena_carve(arena_base, arena_cap, &cur,
                                          kQ_DIM * sizeof(float), "kqv_back");
    float *d_attn_low       = arena_carve(arena_base, arena_cap, &cur,
                                          kATTN_LOW_OUT * sizeof(float), "attn_low");
    float *d_attn_out       = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "attn_out");
    float *d_resid_hc_attn  = arena_carve(arena_base, arena_cap, &cur,
                                          kHC_DIM * sizeof(float),
                                          "residual_hc_after_attn");

    // Compressor / indexer carve-outs (only when needed).
    float *d_attn_emit_kv = nullptr;
    float *d_idx_emit_kv  = nullptr;
    int32_t *d_comp_allowed = nullptr;
    // Scratch for the long-prompt indexer scoring path. These
    // are unused (left null) on the short path; the long path is fired
    // only when L->n_index_comp > kINDEXER_TOP_K.
    float *d_indexer_q       = nullptr;   // [kIDX_Q_DIM=8192] fp32
    float *d_indexer_weights = nullptr;   // [kN_INDEXER_HEAD=64] fp32
    float *d_indexer_scores  = nullptr;   // [cap_comp] fp32
    if (has_compressor) {
        d_attn_emit_kv = arena_carve(arena_base, arena_cap, &cur,
                                     kHEAD_DIM * sizeof(float), "attn_emit_kv");
    }
    if (has_indexer) {
        d_idx_emit_kv  = arena_carve(arena_base, arena_cap, &cur,
                                     kIDX_HEAD_DIM * sizeof(float), "idx_emit_kv");
        // n_index_comp post-push <= cap_comp ; reserve cap_comp int32s.
        const int alloc_n = L->cap_comp > 0 ? L->cap_comp : 1;
        d_comp_allowed = (int32_t *)arena_carve(arena_base, arena_cap, &cur,
                                                (size_t)alloc_n * sizeof(int32_t),
                                                "comp_allowed");
        // Long-prompt scratch. Only consumed when n_index_comp > top_k.
        d_indexer_q       = arena_carve(arena_base, arena_cap, &cur,
                                        kIDX_Q_DIM * sizeof(float),
                                        "indexer_q");
        d_indexer_weights = arena_carve(arena_base, arena_cap, &cur,
                                        kN_INDEXER_HEAD * sizeof(float),
                                        "indexer_weights");
        d_indexer_scores  = arena_carve(arena_base, arena_cap, &cur,
                                        (size_t)alloc_n * sizeof(float),
                                        "indexer_scores");
    }

    float *d_hc_ffn_post_w  = arena_carve(arena_base, arena_cap, &cur,
                                          kN_HC * sizeof(float), "hc_ffn_post_w");
    float *d_hc_ffn_comb    = arena_carve(arena_base, arena_cap, &cur,
                                          kN_HC * kN_HC * sizeof(float),
                                          "hc_ffn_comb");
    float *d_hc_ffn_pre     = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "hc_ffn_pre");
    // HC sinkhorn multi-CTA: FFN-side scratches for launch_hc_ffn_pre_v2_f32.
    // Identical sizing to the attn-side scratches above.
    float *d_hc_ffn_rms     = arena_carve(arena_base, arena_cap, &cur,
                                          sizeof(float), "hc_ffn_rms");
    float *d_hc_ffn_mix     = arena_carve(arena_base, arena_cap, &cur,
                                          kN_HC * (2 + kN_HC) * sizeof(float),
                                          "hc_ffn_mix");
    float *d_ffn_norm       = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "ffn_norm");
    float *d_shared_gate    = arena_carve(arena_base, arena_cap, &cur,
                                          kFF_EXP * sizeof(float), "shared_gate");
    float *d_shared_up      = arena_carve(arena_base, arena_cap, &cur,
                                          kFF_EXP * sizeof(float), "shared_up");
    float *d_shared_silu    = arena_carve(arena_base, arena_cap, &cur,
                                          kFF_EXP * sizeof(float), "shared_silu");
    float *d_ffn_shexp      = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "ffn_shexp");
    float *d_router_logits  = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EXPERT * sizeof(float),
                                          "router_logits");
    float *d_router_probs   = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EXPERT * sizeof(float),
                                          "router_probs");
    int32_t *d_router_topk_ids = (int32_t *)arena_carve(arena_base, arena_cap, &cur,
                                                        kN_USED * sizeof(int32_t),
                                                        "router_topk_ids");
    float *d_router_topk_w  = arena_carve(arena_base, arena_cap, &cur,
                                          kN_USED * sizeof(float),
                                          "router_topk_w");
    float *d_routed_mid     = arena_carve(arena_base, arena_cap, &cur,
                                          kN_USED * kFF_EXP * sizeof(float),
                                          "routed_mid");
    float *d_ffn_moe_out    = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "ffn_moe_out");
    float *d_ffn_out        = arena_carve(arena_base, arena_cap, &cur,
                                          kN_EMBD * sizeof(float), "ffn_out");
    int8_t *d_q8_xq_scratch = (int8_t *)arena_carve(arena_base, arena_cap, &cur,
                                                    kQ8_SCRATCH_ELEMS * sizeof(int8_t),
                                                    "q8_xq_scratch");
    float *d_q8_xscale_scratch = arena_carve(arena_base, arena_cap, &cur,
                                             kQ8_SCRATCH_BLOCKS * sizeof(float),
                                             "q8_xscale_scratch");
    int8_t *d_q8k_xq_scratch = (int8_t *)arena_carve(arena_base, arena_cap, &cur,
                                                     kQ8K_SCRATCH_BLOCKS * 256 * sizeof(int8_t),
                                                     "q8k_xq_scratch");
    float *d_q8k_xscale_scratch = arena_carve(arena_base, arena_cap, &cur,
                                              kQ8K_SCRATCH_BLOCKS * sizeof(float),
                                              "q8k_xscale_scratch");
    int16_t *d_q8k_xbsums_scratch = (int16_t *)arena_carve(arena_base, arena_cap, &cur,
                                                           kQ8K_SCRATCH_BLOCKS * 16 * sizeof(int16_t),
                                                           "q8k_xbsums_scratch");
    int8_t *d_q8k_xq_soa_scratch = (int8_t *)arena_carve(arena_base, arena_cap, &cur,
                                                         kQ8K_SCRATCH_BLOCKS * 256 * sizeof(int8_t),
                                                         "q8k_xq_soa_scratch");
    int16_t *d_q8k_xbsums_soa_scratch = (int16_t *)arena_carve(arena_base, arena_cap, &cur,
                                                               kQ8K_SCRATCH_BLOCKS * 16 * sizeof(int16_t),
                                                               "q8k_xbsums_soa_scratch");
    float *d_comp_kv_cur_scratch = arena_carve(arena_base, arena_cap, &cur,
                                               kCOMP_SCRATCH_WIDTH * sizeof(float),
                                               "comp_kv_cur_scratch");
    float *d_comp_sc_cur_scratch = arena_carve(arena_base, arena_cap, &cur,
                                               kCOMP_SCRATCH_WIDTH * sizeof(float),
                                               "comp_sc_cur_scratch");

    if (!d_hc_attn_post_w || !d_hc_attn_comb || !d_hc_attn_pre ||
        !d_hc_attn_rms || !d_hc_attn_mix ||
        !d_attn_norm || !d_q_lora || !d_q_lora_norm ||
        !d_kv_raw || !d_kv_norm || !d_kv_rope || !d_kv_cur ||
        !d_q_raw || !d_q_norm || !d_q_cur ||
        !d_kqv_out || !d_kqv_back || !d_attn_low || !d_attn_out ||
        !d_resid_hc_attn ||
        !d_hc_ffn_post_w || !d_hc_ffn_comb || !d_hc_ffn_pre ||
        !d_hc_ffn_rms || !d_hc_ffn_mix ||
        !d_ffn_norm || !d_shared_gate || !d_shared_up ||
        !d_shared_silu || !d_ffn_shexp || !d_router_logits ||
        !d_router_probs || !d_router_topk_ids || !d_router_topk_w ||
        !d_routed_mid || !d_ffn_moe_out || !d_ffn_out ||
        !d_q8_xq_scratch || !d_q8_xscale_scratch ||
        !d_q8k_xq_scratch || !d_q8k_xscale_scratch ||
        !d_q8k_xbsums_scratch || !d_q8k_xq_soa_scratch ||
        !d_q8k_xbsums_soa_scratch || !d_comp_kv_cur_scratch ||
        !d_comp_sc_cur_scratch) {
        return -1;
    }
    if (has_compressor && !d_attn_emit_kv) return -1;
    if (has_indexer && (!d_idx_emit_kv || !d_comp_allowed)) return -1;

    // ===== Attention half ==============================================

    // Step 1: hc_attn_pre (HC RMSNorm + matvec + sinkhorn + weighted-sum)
    PERF_STAGE_BEGIN(perf_hc_attn_pre, "hc_attn_pre", "hc",
                     "hc=4x4096", "4096", "hc_attn_fn+scale+base",
                     t_hc_attn_fn->byte_size + t_hc_attn_scale->byte_size +
                         t_hc_attn_base->byte_size,
                     fp32_bytes(kHC_DIM), fp32_bytes(kN_EMBD),
                     fp32_bytes(kN_HC + kN_HC * kN_HC), 1,
                     "attn half");
    launch_hc_attn_pre_v2_f32(input_residual_hc, p_hc_attn_fn,
                              p_hc_attn_scale, p_hc_attn_base,
                              d_hc_attn_post_w, d_hc_attn_comb, d_hc_attn_pre,
                              kN_EMBD, kN_HC, kN_SINKHORN_ITER, kRMS_EPS,
                              d_hc_attn_rms, d_hc_attn_mix,
                              stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_hc_attn_pre);

    // Step 2: attn_norm = RMSNorm(hc_attn_pre, attn_norm.weight, eps)
    PERF_STAGE_BEGIN(perf_attn_norm, "attn_norm", "norm",
                     "4096", "4096", "4096",
                     t_attn_norm->byte_size, fp32_bytes(kN_EMBD),
                     fp32_bytes(kN_EMBD), 0, 1, "");
    launch_rms_norm_f32(d_hc_attn_pre, p_attn_norm_w, d_attn_norm,
                        kN_EMBD, kRMS_EPS, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_attn_norm);

    // Step 3: q_lora = matvec_q8_0(attn_q_a, attn_norm)
    PERF_STAGE_BEGIN(perf_q_lora, "q_lora", "q8",
                     "4096", "1024", "1024x4096:q8_0",
                     q8_weight_bytes(kQ_LORA_RANK, kN_EMBD),
                     fp32_bytes(kN_EMBD), fp32_bytes(kQ_LORA_RANK),
                     q8_scratch_bytes(kN_EMBD), 2, "");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        p_attn_q_a, d_attn_norm, d_q_lora,
        /*n_rows=*/kQ_LORA_RANK, /*n_cols=*/kN_EMBD,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_q_lora);

    // Step 4: q_lora_norm = RMSNorm(q_lora, attn_q_a_norm.weight)
    PERF_STAGE_BEGIN(perf_q_lora_norm, "q_lora_norm", "norm",
                     "1024", "1024", "1024",
                     t_attn_q_a_norm->byte_size, fp32_bytes(kQ_LORA_RANK),
                     fp32_bytes(kQ_LORA_RANK), 0, 1, "");
    launch_rms_norm_f32(d_q_lora, p_attn_q_a_norm, d_q_lora_norm,
                        kQ_LORA_RANK, kRMS_EPS, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_q_lora_norm);

    // Step 5: KVraw = matvec_q8_0(attn_kv_a, attn_norm)
    PERF_STAGE_BEGIN(perf_kv_raw, "kv_raw", "q8",
                     "4096", "512", "512x4096:q8_0",
                     q8_weight_bytes(kKV_DIM, kN_EMBD),
                     fp32_bytes(kN_EMBD), fp32_bytes(kKV_DIM),
                     q8_scratch_bytes(kN_EMBD), 2, "");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        p_attn_kv_a, d_attn_norm, d_kv_raw,
        /*n_rows=*/kKV_DIM, /*n_cols=*/kN_EMBD,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_kv_raw);

    // Step 6: KVnorm = RMSNorm(KVraw, attn_kv_a_norm.weight)
    PERF_STAGE_BEGIN(perf_kv_norm, "kv_norm", "norm",
                     "512", "512", "512",
                     t_attn_kv_a_norm->byte_size, fp32_bytes(kKV_DIM),
                     fp32_bytes(kKV_DIM), 0, 1, "");
    launch_rms_norm_f32(d_kv_raw, p_attn_kv_a_norm, d_kv_norm,
                        kKV_DIM, kRMS_EPS, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_kv_norm);

    // Step 7: Qraw = matvec_q8_0(attn_q_b, q_lora_norm)
    PERF_STAGE_BEGIN(perf_q_raw, "q_raw", "q8",
                     "1024", "64x512", "32768x1024:q8_0",
                     q8_weight_bytes(kQ_DIM, kQ_LORA_RANK),
                     fp32_bytes(kQ_LORA_RANK), fp32_bytes(kQ_DIM),
                     q8_scratch_bytes(kQ_LORA_RANK), 2, "");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        p_attn_q_b, d_q_lora_norm, d_q_raw,
        /*n_rows=*/kQ_DIM, /*n_cols=*/kQ_LORA_RANK,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_q_raw);

    // Step 8: Qnorm = batch RMSNorm per head (no weight)
    PERF_STAGE_BEGIN(perf_q_norm, "q_norm", "norm",
                     "64x512", "64x512", "none",
                     0, fp32_bytes(kQ_DIM), fp32_bytes(kQ_DIM), 0, 1, "");
    launch_rms_norm_batch_f32(d_q_raw, /*w=*/nullptr, /*weight_dim=*/0,
                              d_q_norm,
                              /*n_rows=*/kN_HEAD, /*n_per_row=*/kHEAD_DIM,
                              kRMS_EPS, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_q_norm);

    // Step 9: Qcur = tail-RoPE(Qnorm, n_heads=64, head_dim=512, n_rot=64, pos, il)
    PERF_STAGE_BEGIN(perf_q_cur, "q_cur", "rope",
                     "64x512", "64x512", "none",
                     0, fp32_bytes(kQ_DIM), fp32_bytes(kQ_DIM), 0, 1, "");
    launch_tail_rope_yarn_f32(d_q_norm, d_q_cur, kN_HEAD, kHEAD_DIM,
                              kN_ROT, pos, il, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_q_cur);

    // Step 10: KVrope = tail-RoPE(KVnorm, n_heads=1, head_dim=512, n_rot=64, pos, il)
    PERF_STAGE_BEGIN(perf_kv_rope, "kv_rope", "rope",
                     "512", "512", "none",
                     0, fp32_bytes(kKV_DIM), fp32_bytes(kKV_DIM), 0, 1, "");
    launch_tail_rope_yarn_f32(d_kv_norm, d_kv_rope, /*n_heads=*/1,
                              kHEAD_DIM, kN_ROT, pos, il, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_kv_rope);

    // Step 11: KVcur = FP8 round-trip(KVrope) — non-RoPE prefix in 64-elem
    //          blocks, RoPE tail copied through.
    PERF_STAGE_BEGIN(perf_kv_cur, "kv_cur", "kv",
                     "512", "512", "none",
                     0, fp32_bytes(kKV_DIM), fp32_bytes(kKV_DIM), 0, 1,
                     "fp8 round trip");
    launch_fp8_kv_quantize_round_trip_f32(d_kv_rope, d_kv_cur,
                                          /*n_rows=*/1, kHEAD_DIM, kN_ROT,
                                          stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_kv_cur);

    // Step 12: push raw_kv ring with f16 round-trip on insert.  ds4.c does
    //          the f16 round-trip on push (kv_cache_push_raw, ds4.c:6380);
    //          flash_attn kernel applies an *additional* f16 round-trip
    //          on read but double-rounding f16(f16(x))==f16(x) so it's
    //          idempotent.
    //
    //          SWA eviction: when n_raw == cap_raw (== DS4_N_SWA == 128),
    //          shift rows [1..cap_raw) down to [0..cap_raw-1) — dropping
    //          the oldest row — then write the new row at slot cap_raw-1.
    //          n_raw stays at cap_raw.  Mirrors ds4.c:6385-6389
    //          kv_cache_push_raw memmove path.  K rows are POST-RoPE so
    //          shifted rows remain RoPE-correct for their original
    //          absolute positions; Q for the current token has just been
    //          RoPE'd at its absolute `pos` (Step 9 above).
    {
        PERF_STAGE_BEGIN(perf_raw_kv_push, "raw_kv_push", "kv",
                         "512", "raw_kv", "none",
                         0, fp32_bytes(kKV_DIM), fp32_bytes(kKV_DIM),
                         0, (L->n_raw < L->cap_raw) ? 1 : 2,
                         (L->n_raw < L->cap_raw) ? "append" : "swa_evict");
        if (L->n_raw < L->cap_raw) {
            // Append slot.
            const int slot = L->n_raw;
            k_swa_push_f16_round<<<1, 256, 0, stream>>>(L->raw_kv, slot,
                                                       d_kv_cur, kHEAD_DIM);
            FCK(cudaGetLastError());
            L->n_raw = slot + 1;
        } else {
            // SWA eviction: shift in-place rows[1..cap_raw) -> rows[0..cap_raw-1),
            // then append at last slot.  Single-block sequential shift
            // (canonical ds4 memmove direction, low->high) with
            // __syncthreads between rows so the in-place overlap is safe.
            const int n_shift = L->cap_raw - 1;   // 127 dst rows
            if (n_shift > 0) {
                k_swa_shift_f16_round<<<1, 256, 0, stream>>>(
                    L->raw_kv, kHEAD_DIM, n_shift);
                FCK(cudaGetLastError());
            }
            const int slot = L->cap_raw - 1;
            k_swa_push_f16_round<<<1, 256, 0, stream>>>(L->raw_kv, slot,
                                                       d_kv_cur, kHEAD_DIM);
            FCK(cudaGetLastError());
            // n_raw stays at cap_raw.
        }
        PERF_STAGE_END(perf_raw_kv_push);
    }

    // ----- Compressor + indexer + allowed mask (il>=2) -------------------
    int attn_emit_host = 0;
    int idx_emit_host  = 0;
    if (has_compressor) {
        // Step 12a: attn compressor decode step (input = attn_norm).
        // cite ds4.c:7093.  ratio == 4 (coff=2) for il=2,4,...,42; ratio == 128
        // (coff=1) for il=3,5,...,41.
        const int attn_state_floats = (comp_ratio == kCOMPRESS_RATIO_L2)
            ? kATTN_COMP_STATE_L2 : kATTN_COMP_STATE_L3;
        const int attn_comp_width = (comp_ratio == kCOMPRESS_RATIO_L2)
            ? kATTN_COMP_WIDTH_L2 : kATTN_COMP_WIDTH_L3;
        const int attn_comp_kernels = (((pos + 1) % comp_ratio) == 0)
            ? ((comp_ratio == kCOMPRESS_RATIO_L2) ? 7 : 6)
            : 3;

        PERF_STAGE_BEGIN(perf_attn_compressor, "attn_compressor", "compressor",
                         "4096", "state+optional_512", "kv/gate/ape/norm",
                         t_attn_comp_kv->byte_size + t_attn_comp_gate->byte_size +
                             t_attn_comp_ape->byte_size + t_attn_comp_norm->byte_size,
                         fp32_bytes(kN_EMBD),
                         fp32_bytes(attn_state_floats) * 2 + fp32_bytes(kHEAD_DIM),
                         fp32_bytes(attn_comp_width) * 2,
                         attn_comp_kernels, comp_ratio == 4 ? "ratio4" : "ratio128");
        launch_compressor_decode_step_f32_prealloc(
            /*is_attn=*/true, d_attn_norm,
            p_attn_comp_kv, p_attn_comp_gate, p_attn_comp_ape, p_attn_comp_norm,
            kN_EMBD, kHEAD_DIM, comp_ratio, pos, il,
            L->attn_state_kv, L->attn_state_score,
            d_attn_emit_kv, &attn_emit_host,
            d_comp_kv_cur_scratch, d_comp_sc_cur_scratch, stream);
        FCK(cudaGetLastError());
        PERF_STAGE_END(perf_attn_compressor);

        if (attn_emit_host) {
            // Push to comp_kv ring at slot n_comp.  ds4.c:6395 applies
            // fp16 round-trip on push; flash_attn decode_mixed kernel
            // applies the same fp16 round-trip on the read side (matches
            // the ratio-4 layer-2 kqv-out contract).
            //
            // The CPU reference ds4_die's on compressed-cache overflow
            // (ds4.c:6393); we mirror the same fail-fast behavior here.
            // Long-context capacity is bounded by the configured
            // `max_context` — cap_comp grows with it via
            // compute_cap_comp = max_context/ratio + 2
            // (session_state.cu:137). The compressed cache has no
            // eviction path; once full, the only remedy is to raise
            // max_context.
            if (L->n_comp >= L->cap_comp) {
                std::fprintf(stderr,
                             "forward_layer: layer %d comp_kv full (n_comp=%d cap=%d)\n",
                             il, L->n_comp, L->cap_comp);
                return -1;
            }
            const int slot = L->n_comp;
            float *dst = L->comp_kv + (size_t)slot * kHEAD_DIM;
            FCK(cudaMemcpyAsync(dst, d_attn_emit_kv,
                                (size_t)kHEAD_DIM * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
            L->n_comp = slot + 1;
        }
    }
    if (has_indexer) {
        // Step 12b: indexer compressor decode step (input = attn_norm,
        // shared with attn compressor; cite ds4.c:7116).  Only ratio-4
        // layers have indexer (cite ds4.c:7109 guard).
        const int idx_emit = (((pos + 1) % comp_ratio) == 0) ? 1 : 0;
        PERF_STAGE_BEGIN(perf_indexer_compressor, "indexer_compressor", "compressor",
                         "4096", "state+optional_128", "kv/gate/ape/norm",
                         t_idx_comp_kv->byte_size + t_idx_comp_gate->byte_size +
                             t_idx_comp_ape->byte_size + t_idx_comp_norm->byte_size,
                         fp32_bytes(kN_EMBD),
                         fp32_bytes(kIDX_COMP_STATE) * 2 + fp32_bytes(kIDX_HEAD_DIM),
                         fp32_bytes(kIDX_COMP_WIDTH) * 2,
                         idx_emit ? 7 : 3, "ratio4_indexer");
        launch_compressor_decode_step_f32_prealloc(
            /*is_attn=*/false, d_attn_norm,
            p_idx_comp_kv, p_idx_comp_gate, p_idx_comp_ape, p_idx_comp_norm,
            kN_EMBD, kIDX_HEAD_DIM, comp_ratio, pos, il,
            L->index_state_kv, L->index_state_score,
            d_idx_emit_kv, &idx_emit_host,
            d_comp_kv_cur_scratch, d_comp_sc_cur_scratch, stream);
        FCK(cudaGetLastError());
        PERF_STAGE_END(perf_indexer_compressor);

        if (idx_emit_host) {
            if (L->n_index_comp >= L->cap_comp) {
                std::fprintf(stderr,
                             "forward_layer: layer %d index_comp_kv full "
                             "(n=%d cap=%d)\n",
                             il, L->n_index_comp, L->cap_comp);
                return -1;
            }
            const int slot = L->n_index_comp;
            float *dst = L->index_comp_kv + (size_t)slot * kIDX_HEAD_DIM;
            FCK(cudaMemcpyAsync(dst, d_idx_emit_kv,
                                (size_t)kIDX_HEAD_DIM * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
            L->n_index_comp = slot + 1;
        }

        // Step 12c: indexer_allowed.  cite ds4.c:6926.
        //   - n_index_comp <= TOP_K: short-circuit fill (all 1s).
        //   - n_index_comp >  TOP_K: long-prompt path — score
        //     every compressed row and mark the top-K. Mirrors ds4.c:6957-6979.
        if (L->n_index_comp > 0) {
            const bool long_path = (L->n_index_comp > kINDEXER_TOP_K);
            PERF_STAGE_BEGIN(perf_indexer_allowed, "indexer_allowed", "router",
                             "n_index_comp", "allowed_mask", "none",
                             0, 0, i32_bytes((size_t)L->n_index_comp), 0, 1,
                             long_path ? "long_score_topk" : "short_circuit");
            if (!long_path) {
                launch_indexer_allowed_short_circuit_i32(
                    d_comp_allowed, L->n_index_comp, kINDEXER_TOP_K, stream);
            } else {
                // 1. F16 matvec: qr_norm[1024] -> q[8192]
                //    Weight layout: ds4.c stores indexer_attn_q_b with
                //    dim[0]=N_LORA_Q (in), dim[1]=index_q_dim (out). Our
                //    matvec expects W[n_rows, n_cols] = [out, in].
                launch_mul_mv_f16_f32(
                    p_idx_attn_q_b, d_q_lora_norm, d_indexer_q,
                    /*n_rows=*/kIDX_Q_DIM,
                    /*n_cols=*/kQ_LORA_RANK,
                    stream);
                FCK(cudaGetLastError());

                // 2. tail RoPE on q (n_head=64, head_dim=128, n_rot=64).
                //    ds4.c:6951 rope_tail_layer_inplace(q, 64, 128, 64, pos, il, false).
                //    Our launcher writes in-place when x_in == x_out.
                launch_tail_rope_yarn_f32(
                    d_indexer_q, d_indexer_q,
                    /*n_heads=*/kN_INDEXER_HEAD,
                    /*head_dim=*/kIDX_HEAD_DIM,
                    /*n_rot=*/kN_ROT,
                    pos, il, stream);
                FCK(cudaGetLastError());

                // 3. F16 matvec: attn_norm[4096] -> head_weights[64].
                //    ds4.c:6953 matvec_any(weights, model, layer->indexer_proj, cur);
                //    indexer_proj layout: dim[0]=N_EMBD (in), dim[1]=N_INDEXER_HEAD (out).
                launch_mul_mv_f16_f32(
                    p_idx_proj, d_attn_norm, d_indexer_weights,
                    /*n_rows=*/kN_INDEXER_HEAD,
                    /*n_cols=*/kN_EMBD,
                    stream);
                FCK(cudaGetLastError());

                // 4. Scale weights by 1/sqrt(head_dim * n_head). ds4.c:6954.
                const float idx_scale = 1.0f /
                    sqrtf((float)(kIDX_HEAD_DIM * kN_INDEXER_HEAD));
                launch_scale_inplace_f32(
                    d_indexer_weights, kN_INDEXER_HEAD, idx_scale, stream);
                FCK(cudaGetLastError());

                // 5. Score + top-K. Our session-state indexer cache
                //    stores compressed rows as fp32 (cuda/session_state.cu
                //    L->index_comp_kv carve), so we call the fp32-KV
                //    variant of the score kernel. Pattern mirrors
                //    ds4.c:6957-6979.
                launch_indexer_score_topk_f32_i32(
                    d_indexer_q, d_indexer_weights,
                    L->index_comp_kv,
                    L->n_index_comp,
                    kN_INDEXER_HEAD, kIDX_HEAD_DIM, kINDEXER_TOP_K,
                    d_comp_allowed, d_indexer_scores, stream);
            }
            FCK(cudaGetLastError());
            PERF_STAGE_END(perf_indexer_allowed);
        }
    }

    // Step 13: flash_attn — n_kv = post-push n_raw for layer 0/1; mixed
    //          (raw + comp + allowed) for layer 2..42.  For ratio-128
    //          layers ("Hello" 10-token prompt has n_comp=0), decode_mixed
    //          collapses to the raw-SWA loop (n_comp=0 mass loop = 0 iters).
    PERF_STAGE_BEGIN(perf_flash_attn, "flash_attn", "attention",
                     "q=64x512,kv", "64x512", "attn_sinks",
                     t_attn_sinks->byte_size,
                     fp32_bytes(kQ_DIM) +
                         fp32_bytes((size_t)L->n_raw * kHEAD_DIM) +
                         fp32_bytes((size_t)L->n_comp * kHEAD_DIM),
                     fp32_bytes(kQ_DIM), 0, 1,
                     has_compressor ? "mixed_raw_comp" : "raw_swa");
    if (has_compressor) {
        launch_flash_attn_decode_mixed_f32(
            d_q_cur,
            L->raw_kv, L->comp_kv,
            (has_indexer && L->n_index_comp > 0) ? d_comp_allowed : nullptr,
            p_attn_sinks, d_kqv_out,
            kN_HEAD, /*n_raw=*/L->n_raw,
            /*n_comp=*/L->n_comp, kHEAD_DIM,
            stream);
    } else {
        launch_flash_attn_decode_raw_f32(d_q_cur,
                                         L->raw_kv, L->raw_kv,
                                         p_attn_sinks, d_kqv_out,
                                         kN_HEAD, /*n_kv=*/L->n_raw, kHEAD_DIM,
                                         stream);
    }
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_flash_attn);

    // Step 14: kqv_back = inverse tail-RoPE(kqv_out)
    PERF_STAGE_BEGIN(perf_kqv_back, "kqv_back", "rope",
                     "64x512", "64x512", "none",
                     0, fp32_bytes(kQ_DIM), fp32_bytes(kQ_DIM), 0, 1, "");
    launch_tail_rope_yarn_inverse_f32(d_kqv_out, d_kqv_back, kN_HEAD,
                                      kHEAD_DIM, kN_ROT, pos, il, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_kqv_back);

    // Step 15: attn_low = grouped Q8_0 matvec(attn_output_a, kqv_back)
    PERF_STAGE_BEGIN(perf_attn_low, "attn_low", "q8",
                     "8x4096", "8x1024", "8x1024x4096:q8_0",
                     t_attn_output_a->byte_size, fp32_bytes(kQ_DIM),
                     fp32_bytes(kATTN_LOW_OUT), q8_scratch_bytes(kQ_DIM),
                     2, "grouped");
    launch_mul_mv_q8_0_q8_0_grouped_f32_prealloc(
        p_attn_output_a, d_kqv_back, d_attn_low,
        /*n_groups=*/kN_OUT_GROUP,
        /*group_dim=*/kHEAD_DIM * (kN_HEAD / kN_OUT_GROUP),
        /*rank=*/kN_LORA_O,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_attn_low);

    // Step 16: attn_out = matvec_q8_0(attn_output_b, attn_low)
    PERF_STAGE_BEGIN(perf_attn_out, "attn_out", "q8",
                     "8192", "4096", "4096x8192:q8_0",
                     q8_weight_bytes(kN_EMBD, kATTN_LOW_OUT),
                     fp32_bytes(kATTN_LOW_OUT), fp32_bytes(kN_EMBD),
                     q8_scratch_bytes(kATTN_LOW_OUT), 2, "");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        p_attn_output_b, d_attn_low, d_attn_out,
        /*n_rows=*/kN_EMBD, /*n_cols=*/kATTN_LOW_OUT,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_attn_out);

    // Step 17: hc_post_attn = HC expand+add+split using hc_attn_pre's post/comb.
    PERF_STAGE_BEGIN(perf_hc_post_attn, "hc_post_attn", "hc",
                     "4096+hc_resid", "4x4096", "none",
                     0, fp32_bytes(kN_EMBD + kHC_DIM),
                     fp32_bytes(kHC_DIM), 0, 1, "");
    launch_hc_post_f32(d_attn_out, input_residual_hc,
                       d_hc_attn_post_w, d_hc_attn_comb,
                       d_resid_hc_attn,
                       kN_HC, kN_EMBD, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_hc_post_attn);

    // ===== FFN half =====================================================

    // Step 18: hc_ffn_pre — same fused kernel as hc_attn_pre but with
    //          hc_ffn_* weights and residual_hc_after_attn input.
    PERF_STAGE_BEGIN(perf_hc_ffn_pre, "hc_ffn_pre", "hc",
                     "hc=4x4096", "4096", "hc_ffn_fn+scale+base",
                     t_hc_ffn_fn->byte_size + t_hc_ffn_scale->byte_size +
                         t_hc_ffn_base->byte_size,
                     fp32_bytes(kHC_DIM), fp32_bytes(kN_EMBD),
                     fp32_bytes(kN_HC + kN_HC * kN_HC), 1,
                     "ffn half");
    launch_hc_ffn_pre_v2_f32(d_resid_hc_attn, p_hc_ffn_fn,
                             p_hc_ffn_scale, p_hc_ffn_base,
                             d_hc_ffn_post_w, d_hc_ffn_comb, d_hc_ffn_pre,
                             kN_EMBD, kN_HC, kN_SINKHORN_ITER, kRMS_EPS,
                             d_hc_ffn_rms, d_hc_ffn_mix,
                             stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_hc_ffn_pre);

    // Step 19: ffn_norm = RMSNorm(hc_ffn_pre, ffn_norm.weight)
    PERF_STAGE_BEGIN(perf_ffn_norm, "ffn_norm", "norm",
                     "4096", "4096", "4096",
                     t_ffn_norm->byte_size, fp32_bytes(kN_EMBD),
                     fp32_bytes(kN_EMBD), 0, 1, "");
    launch_rms_norm_f32(d_hc_ffn_pre, p_ffn_norm_w, d_ffn_norm,
                        kN_EMBD, kRMS_EPS, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_ffn_norm);

    // Step 20: shared_gate = matvec_q8_0(ffn_gate_shexp, ffn_norm)
    PERF_STAGE_BEGIN(perf_shared_gate, "shared_gate", "q8",
                     "4096", "2048", "2048x4096:q8_0",
                     q8_weight_bytes(kFF_EXP, kN_EMBD),
                     fp32_bytes(kN_EMBD), fp32_bytes(kFF_EXP),
                     q8_scratch_bytes(kN_EMBD), 2, "");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        p_ffn_gate_shexp, d_ffn_norm, d_shared_gate,
        /*n_rows=*/kFF_EXP, /*n_cols=*/kN_EMBD,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_shared_gate);

    // Step 21: shared_up = matvec_q8_0(ffn_up_shexp, ffn_norm)
    PERF_STAGE_BEGIN(perf_shared_up, "shared_up", "q8",
                     "4096", "2048", "2048x4096:q8_0",
                     q8_weight_bytes(kFF_EXP, kN_EMBD),
                     fp32_bytes(kN_EMBD), fp32_bytes(kFF_EXP),
                     q8_scratch_bytes(kN_EMBD), 2, "");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        p_ffn_up_shexp, d_ffn_norm, d_shared_up,
        /*n_rows=*/kFF_EXP, /*n_cols=*/kN_EMBD,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_shared_up);

    // Step 22: shared_silu_mul = silu(shared_gate) * shared_up
    PERF_STAGE_BEGIN(perf_shared_silu, "shared_silu", "elementwise",
                     "2048+2048", "2048", "none",
                     0, fp32_bytes(kFF_EXP * 2), fp32_bytes(kFF_EXP),
                     0, 1, "");
    launch_silu_mul_f32(d_shared_gate, d_shared_up, d_shared_silu,
                        kFF_EXP, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_shared_silu);

    // Step 23: ffn_shexp = matvec_q8_0(ffn_down_shexp, shared_silu_mul)
    PERF_STAGE_BEGIN(perf_ffn_shexp, "ffn_shexp", "q8",
                     "2048", "4096", "4096x2048:q8_0",
                     q8_weight_bytes(kN_EMBD, kFF_EXP),
                     fp32_bytes(kFF_EXP), fp32_bytes(kN_EMBD),
                     q8_scratch_bytes(kFF_EXP), 2, "");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        p_ffn_down_shexp, d_shared_silu, d_ffn_shexp,
        /*n_rows=*/kN_EMBD, /*n_cols=*/kFF_EXP,
        d_q8_xq_scratch, d_q8_xscale_scratch, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_ffn_shexp);

    // Step 24: router_logits = matvec_f16(ffn_gate_inp, ffn_norm)
    PERF_STAGE_BEGIN(perf_router_logits, "router_logits", "router",
                     "4096", "256", "256x4096:f16",
                     t_ffn_gate_inp->byte_size, fp32_bytes(kN_EMBD),
                     fp32_bytes(kN_EXPERT), 0, 1, "");
    launch_mul_mv_f16_f32(p_ffn_gate_inp, d_ffn_norm, d_router_logits,
                          /*out_dim=*/kN_EXPERT, /*in_dim=*/kN_EMBD, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_router_logits);

    // Step 25: router_probs = sqrt(softplus_stable(router_logits))
    PERF_STAGE_BEGIN(perf_router_probs, "router_probs", "router",
                     "256", "256", "none",
                     0, fp32_bytes(kN_EXPERT), fp32_bytes(kN_EXPERT), 0, 1, "");
    launch_sqrt_softplus_f32(d_router_logits, d_router_probs,
                             kN_EXPERT, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_router_probs);

    // Step 26+27: router_topk_ids + router_topk_w.  Two paths:
    //   - il < 3 (layer 0/1/2): hash router from ffn_gate_tid2eid table
    //     (cite ds4.c:5119 layer_hash_selected_experts).  Two separate
    //     launchers because the topk_ids depends only on token_id, while
    //     topk_w needs router_probs + selected ids.
    //   - il >= 3 (layer 3..42): biased top-k router
    //     (cite ds4.c:5217 layer_topk_selected_experts_from_probs).
    //     Selection score = probs + exp_probs_b; weighting uses unbiased
    //     probs.  Single fused launcher computes both ids + weights.
    PERF_STAGE_BEGIN(perf_router_topk, "router_topk", "router",
                     "256", "top6 ids+w", use_topk_router ? "exp_probs_b" : "tid2eid",
                     use_topk_router ? t_ffn_exp_probs_b->byte_size :
                         (size_t)kN_USED * sizeof(int32_t),
                     fp32_bytes(kN_EXPERT), i32_bytes(kN_USED) + fp32_bytes(kN_USED),
                     0, use_topk_router ? 1 : 2,
                     use_topk_router ? "biased_topk" : "hash_router");
    if (use_topk_router) {
        launch_topk_selected_experts_f32(
            d_router_probs, p_ffn_exp_probs_b,
            d_router_topk_ids, d_router_topk_w,
            kN_EXPERT, kN_USED,
            kEXPERT_WEIGHT_SCALE, kEXPERT_SUM_FLOOR,
            stream);
        FCK(cudaGetLastError());
    } else {
        launch_hash_router_topk_ids_i32(p_ffn_tid2eid, token_id,
                                        d_router_topk_ids, kN_USED, stream);
        FCK(cudaGetLastError());
        launch_hash_router_topk_w_f32(d_router_probs, d_router_topk_ids,
                                      d_router_topk_w, kN_USED,
                                      kEXPERT_WEIGHT_SCALE, kEXPERT_SUM_FLOOR,
                                      stream);
        FCK(cudaGetLastError());
    }
    PERF_STAGE_END(perf_router_topk);

    // Step 28: routed_expert_mid.  The direct launcher consumes full
    // [n_expert][row][block] tensors and indexes topk_ids on device,
    // avoiding per-layer/per-token selected-expert D2D packing.
    PERF_STAGE_BEGIN(perf_moe_gate_up, "moe_gate_up", "moe",
                     "4096+top6", "6x2048",
                     soa_gate_up_active ?
                         "resident_top6x2x2048x4096:iq2_xxs_soa_v2" :
                         "top6x2x2048x4096:iq2_xxs",
                     2 * iq2_selected_weight_bytes(kN_USED, kFF_EXP, kN_EMBD),
                     fp32_bytes(kN_EMBD) + i32_bytes(kN_USED) + fp32_bytes(kN_USED),
                     fp32_bytes(kN_USED * kFF_EXP),
                     q8k_scratch_bytes(1, kN_EMBD), 2,
                     soa_gate_up_active ?
                         "resident_soa_v2_full_tensor_top6" :
                         "direct_full_tensor_top6");
    if (soa_gate_up_active) {
        launch_routed_moe_pair_swiglu_resident_soa_v2_f32_prealloc(
            L->ffn_gate_soa_qs, L->ffn_gate_soa_d,
            L->ffn_up_soa_qs,   L->ffn_up_soa_d,
            d_ffn_norm, d_router_topk_ids, d_router_topk_w,
            d_routed_mid,
            kN_USED, kFF_EXP, kN_EMBD, kN_EXPERT,
            kSWIGLU_CLAMP,
            d_q8k_xq_scratch, d_q8k_xscale_scratch,
            d_q8k_xbsums_scratch, stream);
    } else {
        launch_routed_moe_pair_swiglu_full_f32_prealloc(
            p_ffn_gate_exps,
            p_ffn_up_exps,
            d_ffn_norm, d_router_topk_ids, d_router_topk_w,
            d_routed_mid,
            kN_USED, kFF_EXP, kN_EMBD, kN_EXPERT,
            kSWIGLU_CLAMP,
            d_q8k_xq_scratch, d_q8k_xscale_scratch,
            d_q8k_xbsums_scratch, stream);
    }
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_moe_gate_up);

    // Step 29: ffn_moe_out = sum-6 Q2_K down(routed_mid)
    const bool use_resident_moe_down =
        L->ffn_down_soa_scales && L->ffn_down_soa_qs &&
        L->ffn_down_soa_d && L->ffn_down_soa_dmin;
    PERF_STAGE_BEGIN(perf_moe_down, "moe_down", "moe",
                     "6x2048+top6", "4096",
                     use_resident_moe_down ? "resident_top6x4096x2048:q2_k_soa_v2" :
                         "top6x4096x2048:q2_k",
                     q2k_selected_weight_bytes(kN_USED, kN_EMBD, kFF_EXP),
                     fp32_bytes(kN_USED * kFF_EXP) + i32_bytes(kN_USED),
                     fp32_bytes(kN_EMBD),
                     use_resident_moe_down ?
                         q8k_scratch_bytes(kN_USED, kFF_EXP) * 2 :
                         q8k_scratch_bytes(kN_USED, kFF_EXP),
                     use_resident_moe_down ? 3 : 2,
                     use_resident_moe_down ?
                         "resident_soa_v2_full_tensor_top6_sum6" :
                         "direct_full_tensor_top6_sum6");
    if (use_resident_moe_down) {
        launch_routed_moe_q2k_sum6_resident_soa_v2_f32_prealloc(
            L->ffn_down_soa_scales,
            L->ffn_down_soa_qs,
            L->ffn_down_soa_d,
            L->ffn_down_soa_dmin,
            d_routed_mid, d_router_topk_ids,
            d_ffn_moe_out,
            kN_USED, /*out_dim=*/kN_EMBD,
            /*in_dim=*/kFF_EXP, kN_EXPERT,
            d_q8k_xq_scratch, d_q8k_xscale_scratch,
            d_q8k_xbsums_scratch,
            d_q8k_xq_soa_scratch, d_q8k_xbsums_soa_scratch,
            stream);
    } else {
        launch_routed_moe_q2k_sum6_full_f32_prealloc(
            p_ffn_down_exps,
            d_routed_mid, d_router_topk_ids,
            d_ffn_moe_out,
            kN_USED, /*out_dim=*/kN_EMBD,
            /*in_dim=*/kFF_EXP, kN_EXPERT,
            d_q8k_xq_scratch, d_q8k_xscale_scratch,
            d_q8k_xbsums_scratch, stream);
    }
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_moe_down);

    // Step 30: ffn_out = ffn_shexp + ffn_moe_out
    PERF_STAGE_BEGIN(perf_ffn_out, "ffn_out", "elementwise",
                     "4096+4096", "4096", "none",
                     0, fp32_bytes(kN_EMBD * 2), fp32_bytes(kN_EMBD),
                     0, 1, "");
    launch_add_f32(d_ffn_shexp, d_ffn_moe_out, d_ffn_out, kN_EMBD, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_ffn_out);

    // Step 31: hc_post_ffn = HC expand+add+split using hc_ffn_pre's post/comb.
    PERF_STAGE_BEGIN(perf_hc_post_ffn, "hc_post_ffn", "hc",
                     "4096+hc_resid", "4x4096", "none",
                     0, fp32_bytes(kN_EMBD + kHC_DIM),
                     fp32_bytes(kHC_DIM), 0, 1, "");
    launch_hc_post_f32(d_ffn_out, d_resid_hc_attn,
                       d_hc_ffn_post_w, d_hc_ffn_comb,
                       output_residual_hc,
                       kN_HC, kN_EMBD, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_hc_post_ffn);

    return 0;
}

} // namespace ds4cuda
