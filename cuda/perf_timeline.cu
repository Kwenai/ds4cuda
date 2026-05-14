#include "perf_timeline.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ds4cuda {
namespace {

struct TimelineRecord {
    ds4_perf_stage stage;
    std::string stage_s;
    std::string category_s;
    std::string input_shape_s;
    std::string output_shape_s;
    std::string weight_shape_s;
    std::string notes_s;
    cudaEvent_t start;
    cudaEvent_t stop;
};

struct TimelineState {
    bool enabled = false;
    std::string path;
    std::vector<TimelineRecord> records;
};

TimelineState g_timeline;

static const char *safe_cstr(const char *s)
{
    return s ? s : "";
}

static void stage_copy_strings(TimelineRecord *r, const ds4_perf_stage *s)
{
    r->stage = *s;
    r->stage_s = safe_cstr(s->stage);
    r->category_s = safe_cstr(s->category);
    r->input_shape_s = safe_cstr(s->input_shape);
    r->output_shape_s = safe_cstr(s->output_shape);
    r->weight_shape_s = safe_cstr(s->weight_shape);
    r->notes_s = safe_cstr(s->notes);
}

static void csv_write_escaped(FILE *f, const std::string &s)
{
    bool quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            quote = true;
            break;
        }
    }
    if (!quote) {
        std::fputs(s.c_str(), f);
        return;
    }
    std::fputc('"', f);
    for (char c : s) {
        if (c == '"') std::fputc('"', f);
        std::fputc(c, f);
    }
    std::fputc('"', f);
}

static int cuda_fail(const char *op, cudaError_t err)
{
    if (err == cudaSuccess) return 0;
    std::fprintf(stderr, "perf_timeline: %s failed: %s (%s)\n",
                 op, cudaGetErrorName(err), cudaGetErrorString(err));
    return -1;
}

} // namespace

int ds4_perf_timeline_begin(const char *csv_path)
{
    if (!csv_path || csv_path[0] == '\0') {
        std::fprintf(stderr, "perf_timeline: empty csv path\n");
        return -1;
    }
    if (g_timeline.enabled) {
        std::fprintf(stderr, "perf_timeline: timeline already enabled\n");
        return -1;
    }
    g_timeline.enabled = true;
    g_timeline.path = csv_path;
    g_timeline.records.clear();
    g_timeline.records.reserve(4096);
    return 0;
}

bool ds4_perf_timeline_is_enabled()
{
    return g_timeline.enabled;
}

int ds4_perf_timeline_stage_begin(const ds4_perf_stage *stage,
                                  cudaStream_t stream,
                                  ds4_perf_marker *marker)
{
    if (!g_timeline.enabled) {
        if (marker) marker->record_index = -1;
        return 0;
    }
    if (!stage || !marker) {
        std::fprintf(stderr, "perf_timeline: NULL stage or marker\n");
        return -1;
    }

    TimelineRecord r = {};
    stage_copy_strings(&r, stage);
    if (cuda_fail("cudaEventCreate(start)", cudaEventCreate(&r.start)) != 0) return -1;
    if (cuda_fail("cudaEventCreate(stop)", cudaEventCreate(&r.stop)) != 0) {
        cudaEventDestroy(r.start);
        return -1;
    }
    if (cuda_fail("cudaEventRecord(start)", cudaEventRecord(r.start, stream)) != 0) {
        cudaEventDestroy(r.stop);
        cudaEventDestroy(r.start);
        return -1;
    }

    g_timeline.records.push_back(std::move(r));
    marker->record_index = (int)g_timeline.records.size() - 1;
    return 0;
}

int ds4_perf_timeline_stage_end(ds4_perf_marker *marker, cudaStream_t stream)
{
    if (!g_timeline.enabled) return 0;
    if (!marker || marker->record_index < 0 ||
        marker->record_index >= (int)g_timeline.records.size()) {
        std::fprintf(stderr, "perf_timeline: invalid marker\n");
        return -1;
    }
    TimelineRecord &r = g_timeline.records[(size_t)marker->record_index];
    return cuda_fail("cudaEventRecord(stop)", cudaEventRecord(r.stop, stream));
}

int ds4_perf_timeline_end()
{
    if (!g_timeline.enabled) return 0;

    FILE *f = std::fopen(g_timeline.path.c_str(), "w");
    if (!f) {
        std::perror("perf_timeline: fopen");
        return -1;
    }

    std::fprintf(f,
                 "token,layer,stage,category,input_shape,output_shape,weight_shape,"
                 "weight_bytes,input_bytes,output_bytes,scratch_bytes,kernels,ms,cum_ms,notes\n");

    float cum_ms = 0.0f;
    int rc = 0;
    for (TimelineRecord &r : g_timeline.records) {
        cudaError_t e = cudaEventSynchronize(r.stop);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "perf_timeline: cudaEventSynchronize failed: %s (%s)\n",
                         cudaGetErrorName(e), cudaGetErrorString(e));
            rc = -1;
            break;
        }
        float ms = 0.0f;
        e = cudaEventElapsedTime(&ms, r.start, r.stop);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "perf_timeline: cudaEventElapsedTime failed: %s (%s)\n",
                         cudaGetErrorName(e), cudaGetErrorString(e));
            rc = -1;
            break;
        }
        cum_ms += ms;

        const ds4_perf_stage &s = r.stage;
        std::fprintf(f, "%d,%d,", s.token, s.layer);
        csv_write_escaped(f, r.stage_s);
        std::fputc(',', f);
        csv_write_escaped(f, r.category_s);
        std::fputc(',', f);
        csv_write_escaped(f, r.input_shape_s);
        std::fputc(',', f);
        csv_write_escaped(f, r.output_shape_s);
        std::fputc(',', f);
        csv_write_escaped(f, r.weight_shape_s);
        std::fprintf(f, ",%llu,%llu,%llu,%llu,%d,%.6f,%.6f,",
                     (unsigned long long)s.weight_bytes,
                     (unsigned long long)s.input_bytes,
                     (unsigned long long)s.output_bytes,
                     (unsigned long long)s.scratch_bytes,
                     s.kernels, ms, cum_ms);
        csv_write_escaped(f, r.notes_s);
        std::fputc('\n', f);
    }

    if (std::fclose(f) != 0) {
        std::perror("perf_timeline: fclose");
        rc = -1;
    }

    for (TimelineRecord &r : g_timeline.records) {
        cudaEventDestroy(r.stop);
        cudaEventDestroy(r.start);
    }
    g_timeline.records.clear();
    g_timeline.path.clear();
    g_timeline.enabled = false;
    return rc;
}

} // namespace ds4cuda
