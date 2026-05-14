// argmax.cuh — stream-ordered CUDA argmax helper.
//
// Used by the inference engine to avoid copying the full vocab logits
// to host at every token boundary. The launcher writes one int32 index
// to device memory; callers decide when/how to synchronize.

#ifndef DS4CUDA_ARGMAX_CUH
#define DS4CUDA_ARGMAX_CUH

#include <cuda_runtime.h>

namespace ds4cuda {

// Launch an fp32 argmax over x[0..n). Ties select the lowest index,
// matching the old host loop that only updated on strict `>`.
//
// Preconditions:
//   - x points to n fp32 values in device-accessible memory.
//   - out_idx points to one device int.
//   - n may be non-power-of-two; n <= 0 writes -1.
//
// The launcher queues work on `stream` and does not synchronize.
void launch_argmax_f32_to_i32(const float *x,
                              int *out_idx,
                              int n,
                              cudaStream_t stream = 0);

} // namespace ds4cuda

#endif // DS4CUDA_ARGMAX_CUH
