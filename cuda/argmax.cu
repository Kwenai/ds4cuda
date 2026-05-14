// argmax.cu — fp32 logits argmax reduction.

#include <cmath>
#include <cuda_runtime.h>

#include "argmax.cuh"

namespace ds4cuda {

struct ArgmaxPair {
    float value;
    int index;
};

__device__ __forceinline__ ArgmaxPair better_pair(ArgmaxPair a,
                                                  ArgmaxPair b) {
    if (b.value > a.value) return b;
    if (b.value == a.value && b.index < a.index) return b;
    return a;
}

__global__ void argmax_kernel(const float *__restrict__ x,
                              int *__restrict__ out_idx,
                              int n) {
    const int tid = threadIdx.x;

    ArgmaxPair best;
    best.value = -INFINITY;
    best.index = n;

    for (int i = tid; i < n; i += blockDim.x) {
        ArgmaxPair cur;
        cur.value = x[i];
        cur.index = i;
        best = better_pair(best, cur);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        ArgmaxPair other;
        other.value = __shfl_down_sync(0xffffffff, best.value, offset);
        other.index = __shfl_down_sync(0xffffffff, best.index, offset);
        best = better_pair(best, other);
    }

    __shared__ float warp_values[8];
    __shared__ int warp_indices[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_values[warp] = best.value;
        warp_indices[warp] = best.index;
    }
    __syncthreads();

    if (warp == 0) {
        ArgmaxPair block_best;
        if (lane < (blockDim.x >> 5)) {
            block_best.value = warp_values[lane];
            block_best.index = warp_indices[lane];
        } else {
            block_best.value = -INFINITY;
            block_best.index = n;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            ArgmaxPair other;
            other.value = __shfl_down_sync(0xffffffff, block_best.value, offset);
            other.index = __shfl_down_sync(0xffffffff, block_best.index, offset);
            block_best = better_pair(block_best, other);
        }
        if (lane == 0) *out_idx = block_best.index;
    }
}

__global__ void argmax_invalid_kernel(int *out_idx) {
    *out_idx = -1;
}

void launch_argmax_f32_to_i32(const float *x,
                              int *out_idx,
                              int n,
                              cudaStream_t stream) {
    if (n <= 0) {
        argmax_invalid_kernel<<<1, 1, 0, stream>>>(out_idx);
        return;
    }

    constexpr int TPB = 256;
    argmax_kernel<<<1, TPB, 0, stream>>>(x, out_idx, n);
}

} // namespace ds4cuda
