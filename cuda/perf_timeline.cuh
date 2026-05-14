// perf_timeline.cuh — opt-in CUDA event timeline for ds4cuda forward stages.
//
// The hot path calls are intentionally no-ops unless a timeline is active.
// Stage durations are measured with CUDA events on the caller's stream; CSV
// emission is deferred until ds4_perf_timeline_end() so stage_end does not
// introduce a per-stage CPU/GPU synchronization.

#ifndef DS4CUDA_PERF_TIMELINE_CUH
#define DS4CUDA_PERF_TIMELINE_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace ds4cuda {

struct ds4_perf_stage {
    int token;
    int layer;
    const char *stage;
    const char *category;
    const char *input_shape;
    const char *output_shape;
    const char *weight_shape;
    uint64_t weight_bytes;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t scratch_bytes;
    int kernels;
    const char *notes;
};

struct ds4_perf_marker {
    int record_index;
};

int ds4_perf_timeline_begin(const char *csv_path);
int ds4_perf_timeline_end();
bool ds4_perf_timeline_is_enabled();

int ds4_perf_timeline_stage_begin(const ds4_perf_stage *stage,
                                  cudaStream_t stream,
                                  ds4_perf_marker *marker);
int ds4_perf_timeline_stage_end(ds4_perf_marker *marker, cudaStream_t stream);

} // namespace ds4cuda

#endif // DS4CUDA_PERF_TIMELINE_CUH
