// forward_batch.cu — layer-major chunked prefill (minimal).
//
// "chunked prefill 是 layer-major + 绝对 chunk 边界对齐 ... 每层权重
//  只读一次 ... KV/comp/indexer 状态行也按相同 emit 顺序前进 ...
//  chunk size = 2048 token".
//
// Cite ds4.c:7910 prefill_layer_major_cpu — the upstream CPU prefill
// path that this file mirrors structurally.  In ds4 the loop is:
//
//      for il in 0..N_LAYER:
//          layer_attention_raw_swa_batch(...)   // batched attn
//          layer_ffn_shared_batch(...)          // batched FFN
//
// Our minimal version uses the same outer-layer / inner-token
// nesting but the inner loop is still a per-token call into
// ds4_forward_layer (which re-uses already-aligned per-token kernels).
// Future steps will replace the inner loop with batched matvec
// kernels (launch_mul_mv_q8_0_q8_0_batched / launch_flash_attn_prefill
// / launch_routed_moe_batched) without changing this file's public
// contract.
//
// The numerical contract is: argmax(logits_last) MUST equal the
// streaming forward_token chain's argmax for the same input
// (== 2581 for "Hello", per the terminal argmax baseline).  The kernels
// are identical between streaming and chunked, so any drift is solely
// from fp32 reduction-order differences in the chunked compression
// path — bounded by the chain integration tolerance of
// ref_max_abs * 1e-2.

#include "forward_batch.cuh"

#include "common.cuh"
#include "embedding.cuh"
#include "forward_layer.cuh"
#include "forward_token.cuh"
#include "output_head.cuh"
#include "norm.cuh"
#include "dense_q8.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

namespace ds4cuda {
namespace {

#define BCK(stmt) do {                                                     \
    cudaError_t _e = (stmt);                                               \
    if (_e != cudaSuccess) {                                               \
        std::fprintf(stderr, "forward_batch: CUDA error %s (%s) at %s:%d\n", \
                     cudaGetErrorName(_e), cudaGetErrorString(_e),         \
                     __FILE__, __LINE__);                                  \
        return -1;                                                         \
    }                                                                      \
} while (0)

static inline size_t align256(size_t n) { return (n + 255u) & ~size_t(255u); }

static const struct ds4_tensor *find_required(const struct ds4_model *m,
                                              const char *name) {
    const struct ds4_tensor *t = ds4_model_find_tensor(m, name);
    if (!t) {
        std::fprintf(stderr, "forward_batch: required tensor '%s' not found\n",
                     name);
    }
    return t;
}

static const void *managed_ptr(const struct ds4_model *m,
                               const struct ds4_tensor *t) {
    const void *p = ds4_tensor_device_ptr(m, t);
    if (!p) {
        std::fprintf(stderr,
                     "forward_batch: tensor '%s' has no device ptr "
                     "(backend=%s)\n",
                     t->name,
                     ds4_weight_backend_name(ds4_model_weight_backend(m)));
    }
    return p;
}

// Geometry constants, mirroring forward_token.cu / forward_layer.cu.
static constexpr int   kN_EMBD  = (int)DS4_N_EMBD;       // 4096
static constexpr int   kN_HC    = (int)DS4_N_HC;         // 4
static constexpr int   kHC_DIM  = kN_HC * kN_EMBD;       // 16384
static constexpr int   kN_LAYER = (int)DS4_N_LAYER;      // 43
static constexpr float kRMS_EPS = 1.0e-6f;

} // namespace

// ---------------------------------------------------------------------
//  ds4_forward_chunk: layer-major chunked prefill (minimal).
// ---------------------------------------------------------------------

int ds4_forward_chunk(const struct ds4_model       *m,
                      struct ds4_session_state     *s,
                      const int                    *token_ids,
                      int                           chunk_start_pos,
                      int                           n_tokens,
                      float                        *out_logits_last,
                      cudaStream_t                  stream) {
    if (!m || !s || !token_ids || !out_logits_last) {
        std::fprintf(stderr, "forward_chunk: NULL arg\n");
        return -1;
    }
    if (n_tokens <= 0 || n_tokens > DS4_CHUNK_SIZE_MAX) {
        std::fprintf(stderr,
                     "forward_chunk: n_tokens=%d out of [1, %d]\n",
                     n_tokens, DS4_CHUNK_SIZE_MAX);
        return -1;
    }
    if (chunk_start_pos < 0) {
        std::fprintf(stderr, "forward_chunk: chunk_start_pos=%d < 0\n",
                     chunk_start_pos);
        return -1;
    }

    // ----- 1) Look up the global tensors used by the head + embedding. -
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
        std::fprintf(stderr, "forward_chunk: unexpected weight dtype\n");
        return -1;
    }
    const int n_vocab_t = (int)t_token_embd->dims[1];
    if (n_vocab_t <= 0) {
        std::fprintf(stderr, "forward_chunk: bad n_vocab=%d\n", n_vocab_t);
        return -1;
    }

    // Validate token_ids in range.
    for (int t = 0; t < n_tokens; ++t) {
        if (token_ids[t] < 0 || token_ids[t] >= n_vocab_t) {
            std::fprintf(stderr,
                         "forward_chunk: token_ids[%d]=%d out of vocab=%d\n",
                         t, token_ids[t], n_vocab_t);
            return -1;
        }
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

    // ----- 2) Allocate per-token residual_hc ping/pong buffers ---------
    //
    // Sizing: 2 * n_tokens * HC_DIM * 4 B = 2 * n_tokens * 64 KiB.
    // For "Hello" n_tokens=10 -> 1.28 MiB.  For chunk_size=2048 ->
    // 256 MiB total — still well within the 32 GB activation budget.
    //
    // We allocate this OUTSIDE session_state.activation_arena so the
    // per-layer ds4_forward_layer call sees a fresh (cur=0) arena every
    // time it is invoked.  Per-token state simply lives in our local
    // pair of pings/pongs.
    //
    // For tiny chunks (< 32 tokens) we'd love to live inside the 16 MiB
    // session arena, but the per-layer call's own scratch already eats
    // ~3 MiB so we keep ours separate to avoid any chance of collision.
    const size_t per_token_hc_bytes = (size_t)kHC_DIM * sizeof(float);
    const size_t resid_buf_bytes    = (size_t)n_tokens * per_token_hc_bytes;
    float *d_resid_a = nullptr;
    float *d_resid_b = nullptr;
    BCK(cudaMalloc(&d_resid_a, resid_buf_bytes));
    cudaError_t emb_alloc_err = cudaMalloc(&d_resid_b, resid_buf_bytes);
    if (emb_alloc_err != cudaSuccess) {
        cudaFree(d_resid_a);
        std::fprintf(stderr,
                     "forward_chunk: cudaMalloc(d_resid_b, %zu) failed: %s\n",
                     resid_buf_bytes, cudaGetErrorString(emb_alloc_err));
        return -1;
    }

    // Scratch for the embedding step.  We reuse session_state's
    // activation_arena top — it's safe at this point because no
    // per-layer call has run yet.  Carve from the arena base directly
    // (cur=0) and rewind once we copy out into the residual buffer.
    uint8_t *arena_base = (uint8_t *)s->activation_arena;
    size_t   arena_cap  = s->arena_size;
    if (arena_cap < (size_t)kN_EMBD * sizeof(float)) {
        std::fprintf(stderr, "forward_chunk: arena too small (cap=%zu)\n",
                     arena_cap);
        cudaFree(d_resid_a); cudaFree(d_resid_b);
        return -1;
    }
    float *embed_buf = reinterpret_cast<float *>(arena_base);

    // ----- 3) Embed all N tokens + 4-replica HC build ------------------
    //
    // For each token:
    //   embed_buf = token_embd[token_id] (f16 -> f32)
    //   for h in 0..N_HC:
    //       d_resid_a[t * HC_DIM + h*N_EMBD .. +N_EMBD] = embed_buf
    //
    // We reuse one embed_buf scratch across all N tokens (single fp32
    // [N_EMBD] live region).  This costs N_TOKENS * N_HC = 4*N memcopies
    // but each is tiny (16 KB) so it's bandwidth-trivial.
    //
    // cite ds4.c:7836-7939 hc_from_plain_embedding loop in
    // prefill_layer_major_cpu.
    for (int t = 0; t < n_tokens; ++t) {
        const int tok_id = token_ids[t];
        launch_embed_token_f16_to_f32(p_token_embd, tok_id, kN_EMBD,
                                      embed_buf, stream);
        BCK(cudaGetLastError());
        for (int h = 0; h < kN_HC; ++h) {
            float *dst = d_resid_a + (size_t)t * kHC_DIM + (size_t)h * kN_EMBD;
            BCK(cudaMemcpyAsync(dst, embed_buf,
                                (size_t)kN_EMBD * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
        }
    }
    // Make sure the embedding writes complete before the layer loop
    // starts overwriting embed_buf via ds4_forward_layer's bump arena
    // (which begins at the same arena_base).
    BCK(cudaStreamSynchronize(stream));

    // ----- 4) Layer-major chunk loop -----------------------------------
    //
    // Outer: il in 0..43.  Inner: t in 0..N_TOKENS.  Within each inner
    // step we run the ALREADY-ALIGNED ds4_forward_layer per-token
    // launcher pipeline.  KV/compressor/indexer rings advance once per
    // (il, t) pair — this is the same emit semantics as ds4's
    // prefill_layer_major_cpu (cite ds4.c:7910).
    //
    // Per-token absolute pos = chunk_start_pos + t.  This is the value
    // that drives RoPE angles + compressor emit ((pos+1)%ratio==0
    // boundary).  For the first chunk of a fresh session,
    // chunk_start_pos == 0.
    //
    // After processing all N tokens at layer il:
    //   - swap d_resid_a / d_resid_b  (current layer's outputs become
    //     next layer's inputs).
    //
    // The per-layer ds4_forward_layer call internally bump-allocates
    // from session_state.activation_arena starting at cur=0.  Our
    // d_resid_a/b live OUTSIDE that arena so layer scratch never
    // overwrites residual state.

    float *in_resid  = d_resid_a;
    float *out_resid = d_resid_b;
    int rc = 0;

    for (int il = 0; il < kN_LAYER; ++il) {
        for (int t = 0; t < n_tokens; ++t) {
            const int tok_id = token_ids[t];
            const int abs_pos = chunk_start_pos + t;
            const float *t_in  = in_resid  + (size_t)t * kHC_DIM;
            float       *t_out = out_resid + (size_t)t * kHC_DIM;
            rc = ds4_forward_layer(m, s, il, tok_id, abs_pos,
                                   t_in, t_out, stream);
            if (rc != 0) {
                std::fprintf(stderr,
                             "forward_chunk: ds4_forward_layer(il=%d, t=%d, "
                             "pos=%d) failed (rc=%d)\n",
                             il, t, abs_pos, rc);
                cudaFree(d_resid_a); cudaFree(d_resid_b);
                return rc;
            }
        }
        // After this layer's inner-token loop, swap ping/pong so next
        // layer reads from `out_resid` (the just-written outputs).
        std::swap(in_resid, out_resid);
    }
    // After the loop the FINAL output of layer 42 lives in `in_resid`
    // (because the swap at end of il=42 moved out_resid -> in_resid).

    // ----- 5) Final stage on last token only ---------------------------
    //
    // Only the LAST token's logits matter for prefill (next-token
    // prediction = sample from logits at position chunk_end-1).
    // Discard residual_hc[0..N-2].
    //
    // The final-stage chain runs through the session activation arena.
    // At this point arena cur=0 (last layer call rewound it).  We
    // carve scratch + the logits buffer directly.
    const size_t need_collapsed = (size_t)kN_EMBD     * sizeof(float);
    const size_t need_norm      = (size_t)kN_EMBD     * sizeof(float);
    const size_t need_logits    = (size_t)n_vocab_t   * sizeof(float);
    const size_t need_q8_xq     = (size_t)kN_EMBD     * sizeof(int8_t);
    const size_t need_q8_scale  = (size_t)(kN_EMBD / 32) * sizeof(float);
    const size_t need_total     = align256(need_collapsed)
                                + align256(need_norm)
                                + align256(need_logits)
                                + align256(need_q8_xq)
                                + align256(need_q8_scale);
    if (need_total > arena_cap) {
        std::fprintf(stderr,
                     "forward_chunk: arena too small for final stage "
                     "(need=%zu cap=%zu)\n", need_total, arena_cap);
        cudaFree(d_resid_a); cudaFree(d_resid_b);
        return -1;
    }
    size_t cur = 0;
    float *d_collapsed = reinterpret_cast<float *>(arena_base + cur);
    cur += align256(need_collapsed);
    float *d_norm = reinterpret_cast<float *>(arena_base + cur);
    cur += align256(need_norm);
    float *d_logits = reinterpret_cast<float *>(arena_base + cur);
    cur += align256(need_logits);
    int8_t *d_q8_xq = reinterpret_cast<int8_t *>(arena_base + cur);
    cur += align256(need_q8_xq);
    float *d_q8_scale = reinterpret_cast<float *>(arena_base + cur);
    cur += align256(need_q8_scale);

    const float *last_resid = in_resid + (size_t)(n_tokens - 1) * kHC_DIM;
    rc = ds4_forward_token_final_stage(last_resid,
                                       p_hc_fn, p_hc_scale, p_hc_base,
                                       p_out_norm, p_out_w,
                                       kN_EMBD, kN_HC, n_vocab_t,
                                       kRMS_EPS,
                                       d_collapsed, d_norm, d_logits,
                                       d_q8_xq, d_q8_scale,
                                       stream);
    if (rc != 0) {
        std::fprintf(stderr, "forward_chunk: final stage failed rc=%d\n", rc);
        cudaFree(d_resid_a); cudaFree(d_resid_b);
        return rc;
    }

    // ----- 6) Sync + D2H copy of last-token logits ---------------------
    BCK(cudaStreamSynchronize(stream));
    BCK(cudaMemcpy(out_logits_last, d_logits,
                   (size_t)n_vocab_t * sizeof(float),
                   cudaMemcpyDeviceToHost));

    // ----- 7) Cleanup --------------------------------------------------
    cudaFree(d_resid_a);
    cudaFree(d_resid_b);

    return 0;
}

} // namespace ds4cuda
