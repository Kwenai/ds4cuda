// embedding.cu — token-embedding row gather kernel.
//
// Mirrors ds4/ds4.c:2655 embed_token_f16:
//
//     for (uint64_t i = 0; i < n_embd; i++) out[i] = f16_to_f32(row[i]);
//
// The DS4 tensor layout is F16 row-major with dim[0]=n_embd, dim[1]=vocab,
// and on-disk row `t` starts at base + t*n_embd*sizeof(uint16_t) — see
// ds4.c:2661-2666 (the same row layout the HC residual mix code uses
// host-side at model-load time).  This kernel performs exactly that row
// gather + IEEE binary16->binary32 cast in a single block.
//
// Numerical: bit-equal with ds4.c f16_to_f32 (uses fp16_bits_to_fp32, which
// goes through __half2float — see common.cuh).  No reduction, so no
// fp32-order drift.

#include <cstdint>
#include <cuda_runtime.h>

#include "common.cuh"
#include "embedding.cuh"

namespace ds4cuda {

namespace {

__global__ void embed_token_f16_to_f32_kernel(const uint16_t *__restrict__ token_embd_weight,
                                              int             token_id,
                                              int             n_embd,
                                              float          *__restrict__ out)
{
    const uint16_t *row = token_embd_weight + (size_t)token_id * (size_t)n_embd;
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         i < n_embd;
         i += (int)(gridDim.x * blockDim.x))
    {
        out[i] = fp16_bits_to_fp32(row[i]);
    }
}

} // namespace

void launch_embed_token_f16_to_f32(const uint16_t *token_embd_weight,
                                   int             token_id,
                                   int             n_embd,
                                   float          *out,
                                   cudaStream_t    stream)
{
    constexpr int BS = 256;
    // n_embd = 4096 in DS4; one block of 256 threads = 16 elems per thread.
    // Use a small grid so the fast path (n_embd <= ~16 * 1024) stays
    // single-launch with low overhead.
    int grid = (n_embd + BS - 1) / BS;
    if (grid < 1) grid = 1;
    embed_token_f16_to_f32_kernel<<<grid, BS, 0, stream>>>(token_embd_weight,
                                                           token_id,
                                                           n_embd,
                                                           out);
}

} // namespace ds4cuda
