// forward_token.cu — streaming forward for one ds4 token end-to-end.
//
// Implements the wrapper around ds4_forward_layer (il in {0..42}) that adds:
//
//   1.  embed_token row gather (cite ds4.c:7833 forward_token line)
//   2.  4-replica HC residual (cite ds4.c:7836 hc_from_plain_embedding)
//   3.  per-layer ds4_forward_layer dispatch loop (cite ds4.c:7843)
//   4.  output_logits_one_decode_scratch tail
//        - launch_output_hc_head_f32  (cite ds4.c:8154)
//        - launch_rms_norm_f32 with output_norm.weight (cite ds4.c:8158)
//        - launch_mul_mv_q8_0_q8_0_f32 with output.weight (cite ds4.c:8162)
//
// The activation_arena (16 MiB inside session_state) is a bump pointer
// pool; we rewind cur=0 at the start of each forward_token call so
// per-stage scratch memory is reused across tokens.  Each
// ds4_forward_layer call also bump-allocates from the same arena, so
// our top-level allocations are placed AFTER the per-layer scratch
// will need (~9 MiB peak inside layer 2 at max_ctx=64).  In practice
// we allocate only ~520 KiB at the top level (4 residual_hc copies +
// embedding row + final-stage scratch), well within the 16 MiB budget.
//
// Footprint (per call):
//   embed_buf            :     16 KiB  (N_EMBD fp32)
//   residual_hc_a / _b   :  2 * 64 KiB (HC_DIM fp32; ping-pong layer io)
//   collapsed            :     16 KiB
//   norm_out             :     16 KiB
//   logits               :    505 KiB  (N_VOCAB fp32, 129280 * 4)
//   --------                --------
//   total                :    632 KiB
//
// The per-layer forward consumes its scratch from `cur` AFTER our top
// allocations, so the arena cursor at layer-call time is ~632 KiB and
// each layer's ~3 MiB carve fits cleanly in the remaining 15.4 MiB.

#include "forward_token.cuh"

#include "common.cuh"
#include "embedding.cuh"
#include "forward_layer.cuh"
#include "output_head.cuh"
#include "norm.cuh"
#include "dense_q8.cuh"
#include "perf_timeline.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

namespace ds4cuda {
namespace {

#define FCK(stmt) do {                                                     \
    cudaError_t _e = (stmt);                                               \
    if (_e != cudaSuccess) {                                               \
        std::fprintf(stderr, "forward_token: CUDA error %s (%s) at %s:%d\n", \
                     cudaGetErrorName(_e), cudaGetErrorString(_e),         \
                     __FILE__, __LINE__);                                  \
        return -1;                                                         \
    }                                                                      \
} while (0)

#define PCK(stmt) do {                                                     \
    int _rc = (stmt);                                                       \
    if (_rc != 0) {                                                         \
        std::fprintf(stderr, "forward_token: perf timeline error at %s:%d\n", \
                     __FILE__, __LINE__);                                  \
        return -1;                                                          \
    }                                                                       \
} while (0)

#define PERF_STAGE_BEGIN(var, name, cat, in_s, out_s, weight_s, wb, ib, ob, sb, nk, note_s) \
    ds4cuda::ds4_perf_marker var = {};                                      \
    do {                                                                    \
        if (ds4cuda::ds4_perf_timeline_is_enabled()) {                      \
            ds4cuda::ds4_perf_stage _perf_stage = {};                       \
            _perf_stage.token = perf_token;                                 \
            _perf_stage.layer = -1;                                         \
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

// 256-byte alignment matches forward_layer.cu's bump arena.  Keeping
// downstream allocations 256-aligned helps cudaMemcpy / kernels that
// like coalesced reads.
static inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }
static inline size_t fp32_bytes(size_t n) { return n * sizeof(float); }
static inline size_t q8_weight_bytes(int n_rows, int n_cols) {
    return (size_t)n_rows * (size_t)(n_cols / 32) * sizeof(block_q8_0);
}
static inline size_t q8_scratch_bytes(int n_cols) {
    return (size_t)n_cols * sizeof(int8_t) + (size_t)(n_cols / 32) * sizeof(float);
}

static float *arena_carve(uint8_t *base, size_t total_bytes,
                          size_t *cursor, size_t bytes,
                          const char *tag) {
    size_t pos = *cursor;
    size_t need = align256(bytes);
    if (pos + need > total_bytes) {
        std::fprintf(stderr,
                     "forward_token: activation arena overflow at '%s' "
                     "(have=%zu need=%zu used=%zu cap=%zu)\n",
                     tag, bytes, need, pos, total_bytes);
        return nullptr;
    }
    *cursor = pos + need;
    return reinterpret_cast<float *>(base + pos);
}

static const struct ds4_tensor *find_required(const struct ds4_model *m,
                                              const char *name) {
    const struct ds4_tensor *t = ds4_model_find_tensor(m, name);
    if (!t) {
        std::fprintf(stderr, "forward_token: required tensor '%s' not found\n",
                     name);
    }
    return t;
}

static const void *managed_ptr(const struct ds4_model *m,
                               const struct ds4_tensor *t) {
    const void *p = ds4_tensor_device_ptr(m, t);
    if (!p) {
        std::fprintf(stderr,
                     "forward_token: tensor '%s' has no device ptr "
                     "(backend=%s)\n",
                     t->name,
                     ds4_weight_backend_name(ds4_model_weight_backend(m)));
    }
    return p;
}

// Geometry constants, mirroring forward_layer.cu.
static constexpr int   kN_EMBD  = (int)DS4_N_EMBD;       // 4096
static constexpr int   kN_HC    = (int)DS4_N_HC;         // 4
static constexpr int   kN_LAYER = (int)DS4_N_LAYER;      // 43
static constexpr float kRMS_EPS = 1.0e-6f;

} // namespace

// ---------------------------------------------------------------------
//  Final-stage chain: hc_head + output_norm + output projection.
// ---------------------------------------------------------------------

int ds4_forward_token_final_stage(
        const float           *residual_hc,
        const uint16_t        *output_hc_fn,
        const float           *output_hc_scale,
        const float           *output_hc_base,
        const float           *output_norm_w,
        const block_q8_0      *output_w_q8_0,
        int                    n_embd,
        int                    n_hc,
        int                    n_vocab,
        float                  rms_eps,
        float                 *scratch_collapsed,
        float                 *scratch_norm,
        float                 *logits_out,
        int8_t                *scratch_q8_xq,
        float                 *scratch_q8_scale,
        cudaStream_t           stream,
        int                    perf_token) {
    if (!residual_hc || !output_hc_fn || !output_hc_scale ||
        !output_hc_base || !output_norm_w || !output_w_q8_0 ||
        !scratch_collapsed || !scratch_norm || !logits_out ||
        !scratch_q8_xq || !scratch_q8_scale) {
        std::fprintf(stderr, "forward_token_final_stage: NULL arg\n");
        return -1;
    }

    // Step 5: output_hc_head_one — RMSNorm-no-weight + matvec_f16 +
    // sigmoid+eps + hc_weighted_sum.  cite ds4.c:8154.
    PERF_STAGE_BEGIN(perf_output_hc_head, "output_hc_head", "output",
                     "4x4096", "4096", "output_hc_fn+scale+base",
                     (size_t)n_hc * (size_t)n_embd * (size_t)n_hc * sizeof(uint16_t) +
                         sizeof(float) + (size_t)n_hc * sizeof(float),
                     fp32_bytes((size_t)n_hc * n_embd), fp32_bytes(n_embd),
                     0, 1, "");
    launch_output_hc_head_f32(residual_hc, output_hc_fn,
                              output_hc_scale, output_hc_base,
                              scratch_collapsed,
                              n_embd, n_hc, rms_eps, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_output_hc_head);

    // Step 6: output_norm = rms_norm_weight(output_collapsed,
    // output_norm.weight).  cite ds4.c:8158.
    PERF_STAGE_BEGIN(perf_output_norm, "output_norm", "norm",
                     "4096", "4096", "4096",
                     fp32_bytes(n_embd), fp32_bytes(n_embd),
                     fp32_bytes(n_embd), 0, 1, "");
    launch_rms_norm_f32(scratch_collapsed, output_norm_w, scratch_norm,
                        n_embd, rms_eps, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_output_norm);

    // Step 7: logits = matvec_q8_0(output.weight, output_norm).
    // cite ds4.c:8162.
    PERF_STAGE_BEGIN(perf_output_projection, "output_projection", "q8",
                     "4096", "129280", "129280x4096:q8_0",
                     q8_weight_bytes(n_vocab, n_embd),
                     fp32_bytes(n_embd), fp32_bytes(n_vocab),
                     q8_scratch_bytes(n_embd), 2, "lm_head");
    launch_mul_mv_q8_0_q8_0_f32_prealloc(
        output_w_q8_0, scratch_norm, logits_out,
        /*n_rows=*/n_vocab, /*n_cols=*/n_embd,
        scratch_q8_xq, scratch_q8_scale, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_output_projection);

    return 0;
}

// ---------------------------------------------------------------------
//  ds4_forward_token: full per-token pipeline.
// ---------------------------------------------------------------------

int ds4_forward_token(const struct ds4_model      *m,
                      struct ds4_session_state    *s,
                      int                          token_id,
                      int                          pos,
                      const float                **logits_out,
                      cudaStream_t                 stream) {
    const int perf_token = pos;
    if (!m || !s || !logits_out) {
        std::fprintf(stderr, "forward_token: NULL arg\n");
        return -1;
    }

    // ----- 1) Look up and validate global tensors ----------------------
    const struct ds4_tensor *t_token_embd = find_required(m, "token_embd.weight");
    const struct ds4_tensor *t_hc_fn      = find_required(m, "output_hc_fn.weight");
    const struct ds4_tensor *t_hc_scale   = find_required(m, "output_hc_scale.weight");
    const struct ds4_tensor *t_hc_base    = find_required(m, "output_hc_base.weight");
    const struct ds4_tensor *t_out_norm   = find_required(m, "output_norm.weight");
    const struct ds4_tensor *t_out        = find_required(m, "output.weight");
    if (!t_token_embd || !t_hc_fn || !t_hc_scale || !t_hc_base ||
        !t_out_norm || !t_out) {
        return -1;
    }
    if (t_token_embd->quant != DS4_QUANT_F16 ||
        t_hc_fn->quant      != DS4_QUANT_F16 ||
        t_hc_scale->quant   != DS4_QUANT_F32 ||
        t_hc_base->quant    != DS4_QUANT_F32 ||
        t_out_norm->quant   != DS4_QUANT_F32 ||
        t_out->quant        != DS4_QUANT_Q8_0) {
        std::fprintf(stderr, "forward_token: unexpected weight dtype "
                     "(token_embd=%d hc_fn=%d hc_scale=%d hc_base=%d "
                     "out_norm=%d out=%d)\n",
                     (int)t_token_embd->quant, (int)t_hc_fn->quant,
                     (int)t_hc_scale->quant,   (int)t_hc_base->quant,
                     (int)t_out_norm->quant,   (int)t_out->quant);
        return -1;
    }
    const int n_vocab_t = (int)t_token_embd->dims[1];
    if (n_vocab_t <= 0 || token_id < 0 || token_id >= n_vocab_t) {
        std::fprintf(stderr, "forward_token: token_id %d out of vocab=%d\n",
                     token_id, n_vocab_t);
        return -1;
    }

    const uint16_t *p_token_embd = (const uint16_t *)managed_ptr(m, t_token_embd);
    const uint16_t *p_hc_fn      = (const uint16_t *)managed_ptr(m, t_hc_fn);
    const float    *p_hc_scale   = (const float    *)managed_ptr(m, t_hc_scale);
    const float    *p_hc_base    = (const float    *)managed_ptr(m, t_hc_base);
    const float    *p_out_norm   = (const float    *)managed_ptr(m, t_out_norm);
    const block_q8_0 *p_out_w    = (const block_q8_0 *)managed_ptr(m, t_out);
    if (!p_token_embd || !p_hc_fn || !p_hc_scale || !p_hc_base ||
        !p_out_norm || !p_out_w) {
        return -1;
    }

    // ----- 2) Carve top-level scratch from the activation arena --------
    uint8_t *arena_base = (uint8_t *)s->activation_arena;
    size_t   arena_cap  = s->arena_size;
    size_t   cur        = 0;

    // The per-layer ds4_forward_layer call internally bump-allocates
    // from cur=0 of the arena (it does NOT see our cursor).  To avoid
    // collision we therefore CANNOT keep our own state inside the
    // arena across layer calls.  Instead we copy the residual_hc into
    // s->residual_hc (which is OUTSIDE the activation arena, sized
    // for exactly HC_DIM fp32) and use that as both the layer input
    // pong buffer and the final-stage input.
    //
    // The layer ping buffer lives in session_state as residual_hc_scratch,
    // avoiding per-token cudaMalloc/cudaFree and the cudaFree-side sync.

    if (!s->residual_hc || !s->residual_hc_scratch) {
        std::fprintf(stderr,
                     "forward_token: session_state residual ping/pong is NULL\n");
        return -1;
    }
    if (arena_cap < (size_t)kN_EMBD * sizeof(float)) {
        std::fprintf(stderr, "forward_token: arena too small (cap=%zu)\n", arena_cap);
        return -1;
    }

    // Embedding buffer is the only top-level arena allocation we keep
    // — once we copy out into s->residual_hc the embed_buf can be
    // overwritten by per-layer scratch on the first ds4_forward_layer
    // call.  Carving here is a no-op in terms of arena state seen by
    // the per-layer call (we set cur back to 0 below).
    float *embed_buf = arena_carve(arena_base, arena_cap, &cur,
                                    (size_t)kN_EMBD * sizeof(float),
                                    "embed_buf");
    if (!embed_buf) return -1;

    // ----- 3) Token embedding lookup -----------------------------------
    // cite ds4.c:7833 forward_token line / ds4.c:2655 embed_token_f16.
    PERF_STAGE_BEGIN(perf_embed_token, "embed_token", "embedding",
                     "token_id", "4096", "token_embd.row:f16",
                     (size_t)kN_EMBD * sizeof(uint16_t),
                     0, fp32_bytes(kN_EMBD), 0, 1, "");
    launch_embed_token_f16_to_f32(p_token_embd, token_id, kN_EMBD,
                                  embed_buf, stream);
    FCK(cudaGetLastError());
    PERF_STAGE_END(perf_embed_token);

    // ----- 4) 4-replica residual_hc construction ----------------------
    // cite ds4.c:7836 hc_from_plain_embedding — copy embed_buf into
    // each of N_HC=4 residual_hc rows.  Use session-owned ping/pong
    // buffers so this path has no per-token device allocation.
    float *resid_a = s->residual_hc;     // 64 KiB session-owned, outside arena
    float *resid_b = s->residual_hc_scratch;

    PERF_STAGE_BEGIN(perf_residual_init, "residual_init", "embedding",
                     "4096", "4x4096", "none",
                     0, fp32_bytes(kN_EMBD), fp32_bytes((size_t)kN_HC * kN_EMBD),
                     0, kN_HC, "4 device copies");
    for (int h = 0; h < kN_HC; ++h) {
        FCK(cudaMemcpyAsync(resid_a + (size_t)h * kN_EMBD, embed_buf,
                            (size_t)kN_EMBD * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
    }
    PERF_STAGE_END(perf_residual_init);
    // embed_buf is no longer needed after this point — its arena slot
    // gets reused by the per-layer call below.
    cur = 0;  // rewind so ds4_forward_layer sees a fresh arena.

    // ----- 5) For il in 0..42: ds4_forward_layer ----------------------
    // cite ds4.c:7843 layer forward dispatch.  The per-layer launcher
    // bump-allocates from the activation arena starting at cur=0;
    // we keep ping=resid_a / pong=resid_b OUTSIDE the arena so layer
    // scratch never overwrites residual state.
    float *in_resid  = resid_a;
    float *out_resid = resid_b;
    int rc = 0;
    for (int il = 0; il < kN_LAYER; ++il) {
        rc = ds4_forward_layer(m, s, il, token_id, pos,
                               in_resid, out_resid, stream);
        if (rc != 0) {
            std::fprintf(stderr, "forward_token: ds4_forward_layer(il=%d) failed (rc=%d)\n",
                         il, rc);
            return rc;
        }
        // Swap ping/pong for the next layer.
        std::swap(in_resid, out_resid);
    }
    // After the loop the FINAL output is in `in_resid` (we swapped at end
    // of the last iteration so the result of the last layer is now in
    // in_resid).

    // ----- 6) Final stage: hc_head + output_norm + output projection ---
    // After the layer loop the activation arena is again at cur=0 (the
    // last ds4_forward_layer call rewound implicitly via its own scope).
    // Carve the final-stage scratch + logits buffer from the arena.
    float *collapsed = arena_carve(arena_base, arena_cap, &cur,
                                    (size_t)kN_EMBD * sizeof(float),
                                    "output_collapsed");
    float *norm_out  = arena_carve(arena_base, arena_cap, &cur,
                                    (size_t)kN_EMBD * sizeof(float),
                                    "output_norm_out");
    float *logits    = arena_carve(arena_base, arena_cap, &cur,
                                    (size_t)n_vocab_t * sizeof(float),
                                    "logits");
    int8_t *q8_xq = (int8_t *)arena_carve(arena_base, arena_cap, &cur,
                                          (size_t)kN_EMBD * sizeof(int8_t),
                                          "output_q8_xq");
    float *q8_scale = arena_carve(arena_base, arena_cap, &cur,
                                  (size_t)(kN_EMBD / 32) * sizeof(float),
                                  "output_q8_scale");
    if (!collapsed || !norm_out || !logits || !q8_xq || !q8_scale) {
        return -1;
    }

    rc = ds4_forward_token_final_stage(in_resid,
                                       p_hc_fn, p_hc_scale, p_hc_base,
                                       p_out_norm, p_out_w,
                                       kN_EMBD, kN_HC, n_vocab_t,
                                       kRMS_EPS,
                                       collapsed, norm_out, logits,
                                       q8_xq, q8_scale,
                                       stream, pos);
    if (rc != 0) {
        return rc;
    }

    *logits_out = logits;
    return 0;
}

} // namespace ds4cuda
