// compressor.cuh — host-callable launchers for the DeepSeek-V4 streaming
// compressor decode-one step (used by both attn and indexer paths).
//
// Implementation: cuda/compressor.cu.
// Reference: ds4/ds4.c:6420-6471 (compressor_pool_decode_state) and
// ds4/ds4.c:6475-6568 (compressor_decode_one).
//
// What this kernel does
// ---------------------
// Streaming compressor for one token at one layer.  For ratio=4:
//   - coff = 2, width = 2*head_dim, state buffer is [2*4 rows, 2*head_dim cols]
//     = 8 rows of 2*head_dim floats.
//   - On every token, project x (RMSNorm-ed hidden state) through two F16
//     pair matmuls (W_kv, W_gate) to produce kv_cur[width], sc_cur[width],
//     add APE bias `ape[pos_mod * width + j]` to sc_cur,
//     write to state row `compress_ratio + pos_mod` (row 4..7 for ratio=4).
//   - When (pos+1) % compress_ratio == 0, pool 8 rows -> head_dim, RMSNorm
//     with `norm` weight, then tail-RoPE YaRN at comp_pos=pos+1-ratio.
//     Attn path additionally runs E4M3FN fp8 round-trip on the non-RoPE
//     prefix.  The state ring is then shifted (rows 4..7 -> 0..3, then
//     duplicated back to 4..7) per ds4.c:6547-6563.
//
// Differences attn vs indexer (ratio==4, layer 2):
//   - attn  : head_dim=512 (DS4_N_HEAD_DIM), W_kv/W_gate F16 [4096, 1024],
//             ape F16 [1024, 4], norm F32 [512].  fp8 round-trip on emit.
//   - indexer: head_dim=128 (DS4_N_INDEXER_HEAD_DIM), W_kv/W_gate F16
//              [4096, 256], ape F16 [256, 4], norm F32 [128].  No fp8.
//
// CPU dump tags exercised by the ratio-4 streaming compressor stage tests:
//     compressor_state_kv          fp32 [coff*head_dim * coff*ratio]
//     compressor_state_score       fp32 [coff*head_dim * coff*ratio]
//     KVcompress                   fp32 [head_dim] on emit
//     indexer_state_kv             fp32 [coff*INDEXER_HEAD_DIM * coff*ratio]
//     indexer_state_score          fp32 [coff*INDEXER_HEAD_DIM * coff*ratio]
//     indexer_KVcompress           fp32 [INDEXER_HEAD_DIM] on emit
// (cite: commit 6c6fdf8 message.)

#ifndef DS4CUDA_COMPRESSOR_CUH
#define DS4CUDA_COMPRESSOR_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ds4cuda {

// One compressor decode-step.  Buffers are device pointers.
//
// Inputs:
//   x_f32        : fp32 [in_dim]   = attn_norm of current token (RMSNorm-ed
//                                    hidden state, in_dim = DS4_N_EMBD = 4096).
//   w_kv_f16     : f16  [out_dim * in_dim]   = compressor_kv weight, where
//                  out_dim = coff * head_dim_out (=1024 attn, =256 indexer).
//   w_gate_f16   : f16  [out_dim * in_dim]   = compressor_gate weight.
//   w_ape_f16    : f16  [compress_ratio * out_dim] = APE bias.  Indexed as
//                  ape[pos_mod * out_dim + j] (matches ds4.c tensor_2d_value
//                  with dim[0]=out_dim, dim[1]=compress_ratio).
//   w_norm_f32   : fp32 [head_dim_out]       = RMSNorm gain (used on emit).
//
// State buffers (read-modify-write each step):
//   state_kv     : fp32 [coff*ratio * coff*head_dim_out]
//   state_score  : fp32 [coff*ratio * coff*head_dim_out]
//
// Output (only touched on emit):
//   emit_out     : fp32 [head_dim_out]  (= KVcompress or indexer_KVcompress).
//
// Other:
//   head_dim_out  : 512 attn / 128 indexer.
//   in_dim        : 4096 (DS4_N_EMBD).
//   compress_ratio: 4.
//   pos           : current token position (0-based).
//   il            : layer index (used for RoPE freq-base/scale).
//   is_attn       : true → fp8 E4M3FN round-trip on emit's non-RoPE prefix.
//   emitted       : optional out-pointer (host int).  Set to 1 if emit ran,
//                   0 otherwise.  May be NULL.
//
// The launcher serializes a small chain of kernels + the existing
// launch_mul_mv_f16_f32 launcher.  All work runs on `stream`.
void launch_compressor_decode_step_f32(
    bool             is_attn,
    const float     *x_f32,
    const uint16_t  *w_kv_f16,
    const uint16_t  *w_gate_f16,
    const uint16_t  *w_ape_f16,
    const float     *w_norm_f32,
    int              in_dim,
    int              head_dim_out,
    int              compress_ratio,
    int              pos,
    int              il,
    float           *state_kv,
    float           *state_score,
    float           *emit_out,
    int             *emitted_out_host,
    cudaStream_t     stream = 0);

// Same compressor decode-step contract as above, with caller-owned
// projection scratch:
//   scratch_kv_cur : float [coff * head_dim_out]
//   scratch_sc_cur : float [coff * head_dim_out]
// where coff = 2 for compress_ratio==4, otherwise 1.
// Performs no device allocation/free and does not synchronize.
void launch_compressor_decode_step_f32_prealloc(
    bool             is_attn,
    const float     *x_f32,
    const uint16_t  *w_kv_f16,
    const uint16_t  *w_gate_f16,
    const uint16_t  *w_ape_f16,
    const float     *w_norm_f32,
    int              in_dim,
    int              head_dim_out,
    int              compress_ratio,
    int              pos,
    int              il,
    float           *state_kv,
    float           *state_score,
    float           *emit_out,
    int             *emitted_out_host,
    float           *scratch_kv_cur,
    float           *scratch_sc_cur,
    cudaStream_t     stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_COMPRESSOR_CUH
