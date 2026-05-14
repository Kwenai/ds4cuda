// embedding.cuh — host-callable launcher for the token-embedding row
// lookup kernel.
//
// Implementation: cuda/embedding.cu.
//
// Pipeline (matches ds4/ds4.c:2655 embed_token_f16):
//
//     ds4_tensor *te = w->token_embd;       // F16 [n_embd, vocab=129280]
//     const uint16_t *base = tensor_data(m, te);
//     const uint64_t stride = te->dim[0];   // = N_EMBD = 4096
//     const uint16_t *row = base + (uint64_t)token * stride;
//     for (uint64_t i = 0; i < stride; i++)
//         out[i] = f16_to_f32(row[i]);
//
// Shape:
//   - token_embd_weight : F16 row-major, row `t` at offset t*n_embd, length
//                         n_embd. The on-disk GGUF dim layout is
//                         dim[0]=n_embd, dim[1]=vocab — verified at
//                         ds4/ds4.c:2657 (`token >= te->dim[1]`).
//   - out               : fp32 [n_embd], the f16->f32 row.
//
// Numerical contract: __half2float (IEEE 754 binary16->binary32) per
// element, identical to ds4cuda::fp16_bits_to_fp32 used elsewhere and to
// ds4.c:1485 f16_to_f32 fallback.  Bit-equal with the CPU reference for
// all finite + subnormal inputs.
//
// One block per token (here always batch=1).  Each thread handles
// (n_embd / blockDim.x) elements via grid-stride loop, no reduction.

#ifndef DS4CUDA_EMBEDDING_CUH
#define DS4CUDA_EMBEDDING_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ds4cuda {

// One-token embedding lookup: gather row `token_id` from token_embd_weight
// (F16 row-major [vocab, n_embd] in linear-stride form, where row t lives
// at offset t * n_embd) and write a single fp32 [n_embd] vector to `out`.
//
// Pre-conditions:
//   - 0 <= token_id < vocab.  The launcher does NOT bound-check at runtime;
//     callers must guarantee this (e.g. tokenizer enforces).
//   - All buffers are device pointers.
void launch_embed_token_f16_to_f32(const uint16_t *token_embd_weight,
                                   int             token_id,
                                   int             n_embd,
                                   float          *out,
                                   cudaStream_t    stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_EMBEDDING_CUH
