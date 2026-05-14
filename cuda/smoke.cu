/* CUDA smoke test for ds4cuda on DGX Spark.
 *
 * Verifies:
 *   - nvcc -arch=sm_120 compiles and links against cudart
 *   - cudaMalloc / cudaMemcpy / kernel launch / cudaFree round-trip works
 *   - Reports device properties relevant to ds4cuda (compute cap, SM count,
 *     shared-mem-per-block, warp size, max threads, unified-mem flags)
 *
 * Run via `make smoke`. Successful exit code 0 + visual inspection of the
 * device-prop dump is the CUDA-toolchain gate; this is intentionally
 * trivial (no quantization, no real kernels).
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(stmt) do { \
    cudaError_t _e = (stmt); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__, \
                     cudaGetErrorName(_e), cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

__global__ void smoke_double(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;
}

int main(void) {
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    std::printf("cudaGetDeviceCount: %d\n", n_dev);
    if (n_dev <= 0) {
        std::fprintf(stderr, "no CUDA devices\n");
        return 1;
    }

    int dev = 0;
    CK(cudaSetDevice(dev));

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    std::printf("Device %d: %s\n", dev, p.name);
    std::printf("  compute cap          : %d.%d\n", p.major, p.minor);
    std::printf("  SM count             : %d\n", p.multiProcessorCount);
    std::printf("  warp size            : %d\n", p.warpSize);
    std::printf("  shared/block default : %zu bytes\n", (size_t)p.sharedMemPerBlock);
    std::printf("  shared/block opt-in  : %zu bytes\n", (size_t)p.sharedMemPerBlockOptin);
    std::printf("  shared/SM            : %zu bytes\n", (size_t)p.sharedMemPerMultiprocessor);
    std::printf("  regs/block           : %d\n", p.regsPerBlock);
    std::printf("  regs/SM              : %d\n", p.regsPerMultiprocessor);
    std::printf("  max threads/blk      : %d\n", p.maxThreadsPerBlock);
    std::printf("  max threads/SM       : %d\n", p.maxThreadsPerMultiProcessor);
    std::printf("  total global         : %llu MiB\n",
                (unsigned long long)(p.totalGlobalMem >> 20));
    std::printf("  L2 cache             : %d KiB\n", p.l2CacheSize >> 10);
    std::printf("  memory bus width     : %d bits\n", p.memoryBusWidth);
    std::printf("  unified addressing   : %d\n", p.unifiedAddressing);
    std::printf("  managed memory       : %d\n", p.managedMemory);
    std::printf("  pageable mem access  : %d\n", p.pageableMemoryAccess);
    std::printf("  host native atomics  : %d\n", p.hostNativeAtomicSupported);
    std::printf("  concurrent kernels   : %d\n", p.concurrentKernels);
    std::printf("  ECC                  : %d\n", p.ECCEnabled);

    /* Round-trip: 1 MiB float array, double each element on device, verify. */
    const int N = 1 << 18; /* 262144 floats = 1 MiB */
    size_t bytes = (size_t)N * sizeof(float);

    float *h = (float *)std::malloc(bytes);
    for (int i = 0; i < N; ++i) h[i] = (float)i * 0.125f;

    float *d = nullptr;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));

    int tpb = 256;
    int bpg = (N + tpb - 1) / tpb;
    smoke_double<<<bpg, tpb>>>(d, N);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    float *r = (float *)std::malloc(bytes);
    CK(cudaMemcpy(r, d, bytes, cudaMemcpyDeviceToHost));
    CK(cudaFree(d));

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expv = h[i] * 2.0f;
        if (r[i] != expv) {
            if (errors < 3) {
                std::fprintf(stderr, "mismatch at %d: got %g expected %g\n",
                             i, r[i], expv);
            }
            ++errors;
        }
    }
    std::free(h);
    std::free(r);

    if (errors) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", errors);
        return 1;
    }
    std::printf("smoke_double round-trip OK (%d elements)\n", N);
    return 0;
}
