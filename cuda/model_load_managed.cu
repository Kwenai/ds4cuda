/* model_load_managed.cu — full 81 GB GGUF load into cudaMallocManaged.
 *
 * Implements ds4_model_load_to_managed and ds4_model_managed_free, both
 * declared in include/ds4cuda.h with extern "C" linkage so pure-C callers
 * (src/main_tools.c, the test binaries) link cleanly.
 *
 * Strategy: ~120 GB/s steady-state read bandwidth, ~0.7 s/GB setup.
 *
 *   void *managed = NULL;
 *   cudaMallocManaged(&managed, file_size);
 *   for (off = 0; off < file_size; off += chunk) {
 *       memcpy(managed+off, mmap+off, this_chunk);
 *       madvise(mmap+off, this_chunk, MADV_DONTNEED);
 *   }
 *
 * Peak RSS during the loop = (loaded so far) + chunk (managed mid-write)
 *                          + chunk (mmap mid-read; madvise releases just
 *                          after each memcpy). For a 4 GiB chunk this
 *                          tops out at file_size + 8 GiB (= 89 GiB for
 *                          80 GiB weights, well under the 113 GiB OS red
 *                          line set by feedback_os_memory_reserve.md).
 *
 * Red-line guard: every chunk we re-read /proc/meminfo MemAvailable; on
 * MemAvailable < 16 GiB we abort(2) immediately (1 GiB headroom over the
 * 15 GiB OS reserve). The abort() signal lets harnesses distinguish an
 * OOM-near-miss from a CUDA error (which CK() also signals via abort).
 *
 * The header (first ~5 MB) is included in the managed alloc to keep
 * tensor->abs_offset directly indexable: the on-disk file layout is
 * mirrored 1:1 in managed memory, so tensors sit at managed+abs_offset.
 * Wasting 5 MB on header bytes is cheap relative to 80 GB of weights.
 *
 * Lifecycle: ds4_model_managed_free MUST be called before ds4_model_close
 * (the latter is pure C in src/model_open.c — it cannot link cudaFree).
 * The src side stores weights_managed_base/size purely as bookkeeping;
 * close zeroes them defensively but does not call cudaFree itself.
 */

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "ds4cuda.h"

/* OS keeps 15 GB reserved (see feedback_os_memory_reserve.md); we leave
 * 1 GB more as headroom before tripping the abort. */
static const long DS4_RED_LINE_AVAIL_KB = 16L * 1024L * 1024L;  /* 16 GiB in kB */

#define CK(stmt)                                                          \
    do {                                                                  \
        cudaError_t _e = (stmt);                                          \
        if (_e != cudaSuccess) {                                          \
            std::fprintf(stderr,                                          \
                         "CUDA error at %s:%d: %s (%s)\n",                \
                         __FILE__, __LINE__,                              \
                         cudaGetErrorName(_e),                            \
                         cudaGetErrorString(_e));                         \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

static long read_rss_kb(void) {
    FILE *f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long rss = -1;
    while (std::fgets(line, sizeof line, f)) {
        if (std::sscanf(line, "VmRSS: %ld kB", &rss) == 1) break;
    }
    std::fclose(f);
    return rss;
}

static long read_meminfo_kb(const char *key) {
    FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long val = -1;
    size_t klen = std::strlen(key);
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, key, klen) == 0) {
            std::sscanf(line + klen, ": %ld kB", &val);
            break;
        }
    }
    std::fclose(f);
    return val;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Check the OS red line; abort(2) on breach so the parent harness can
 * distinguish from CUDA errors (which CK above signals via abort). */
static void enforce_red_line(const char *tag, int verbose) {
    long avail_kb = read_meminfo_kb("MemAvailable");
    if (avail_kb < 0) return;     /* /proc unreadable: skip rather than block */
    if (avail_kb < DS4_RED_LINE_AVAIL_KB) {
        std::fprintf(stderr,
                     "*** RED LINE *** at %s: MemAvailable %ld MB < %ld MB; aborting\n",
                     tag, avail_kb / 1024, DS4_RED_LINE_AVAIL_KB / 1024);
        std::abort();   /* exit code 134; abort() (vs exit) leaves a core
                           and a clear stack to diagnose how we drifted. */
    }
    if (verbose >= 2) {
        long rss_kb = read_rss_kb();
        std::fprintf(stderr,
                     "[mem %-22s] RSS=%5ld MB  available=%6ld MB\n",
                     tag, rss_kb / 1024, avail_kb / 1024);
    }
}

extern "C" int ds4_model_load_to_managed(struct ds4_model *m,
                                         size_t chunk_size_bytes,
                                         int verbose)
{
    if (!m) return -EINVAL;
    if (!m->mmap_ptr || m->file_size == 0) return -EINVAL;
    if (m->weights_managed_base != NULL) return -EALREADY;

    if (chunk_size_bytes == 0) {
        chunk_size_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;  /* 4 GiB default */
    }

    /* Pre-flight: require MemAvailable >= file_size + 16 GB. Setup peak
     * is file_size + 2*chunk (= ~8 GB for the default 4 GB chunk), so 16
     * GB headroom is generous; the red-line guard inside the loop
     * provides the hard backstop. */
    long avail_kb_pre = read_meminfo_kb("MemAvailable");
    if (avail_kb_pre >= 0) {
        long need_kb = (long)(m->file_size / 1024) + (16L * 1024L * 1024L);
        if (avail_kb_pre < need_kb) {
            std::fprintf(stderr,
                         "ds4_model_load_to_managed: MemAvailable %ld MB < required %ld MB\n"
                         "  hint: drop_caches and retry\n",
                         avail_kb_pre / 1024, need_kb / 1024);
            return -ENOMEM;
        }
    }

    if (verbose) {
        std::fprintf(stderr,
                     "ds4_model_load_to_managed: file_size=%.2f GiB, chunk=%.2f GiB\n",
                     (double)m->file_size / (1024.0*1024.0*1024.0),
                     (double)chunk_size_bytes / (1024.0*1024.0*1024.0));
    }

    enforce_red_line("00 pre alloc", verbose);

    void *managed = NULL;
    double t_alloc0 = now_sec();
    CK(cudaMallocManaged(&managed, m->file_size));
    if (verbose) {
        std::fprintf(stderr,
                     "[load] cudaMallocManaged(%.2f GiB) = %.3f s (lazy, RSS unchanged)\n",
                     (double)m->file_size / (1024.0*1024.0*1024.0),
                     now_sec() - t_alloc0);
    }
    enforce_red_line("01 post alloc", verbose);

    const uint8_t *src_base = (const uint8_t *)m->mmap_ptr;
    uint8_t       *dst_base = (uint8_t *)managed;

    const double total_gib = (double)m->file_size / (1024.0*1024.0*1024.0);
    double t_loop0 = now_sec();
    double last_log = t_loop0;

    for (size_t off = 0; off < m->file_size; off += chunk_size_bytes) {
        size_t this_chunk = chunk_size_bytes;
        if (off + this_chunk > m->file_size) this_chunk = m->file_size - off;

        /* memcpy: pages-in mmap chunk + writes managed chunk. */
        std::memcpy(dst_base + off, src_base + off, this_chunk);

        /* madvise the just-copied mmap window to release physical pages.
         * MADV_DONTNEED requires page-aligned ranges. The mmap region is
         * page-aligned (mmap returns one), so off-aligned + len-aligned
         * relative to off=0 is the natural rule. The first chunk (off=0)
         * always satisfies alignment; subsequent chunks are 4 GiB-aligned
         * and therefore page-aligned. We still compute the safe aligned
         * window in case the caller passes a non-page-multiple chunk. */
        const size_t pg = 4096;
        size_t a_start = off & ~(pg - 1);          /* round down */
        size_t a_end   = ((off + this_chunk) + pg - 1) & ~(pg - 1); /* round up */
        if (a_end > m->file_size)
            a_end = m->file_size;                  /* don't exceed file len */
        size_t a_len = a_end - a_start;
        if (::madvise((void *)((uint8_t *)m->mmap_ptr + a_start), a_len,
                      MADV_DONTNEED) != 0) {
            /* Non-fatal — on some kernels MADV_DONTNEED may EAGAIN under
             * contention. Continue and let the final whole-mmap madvise
             * mop up. The red-line guard would catch any catastrophic
             * RSS overshoot before damage. */
            if (verbose) {
                std::fprintf(stderr,
                             "[load] madvise DONTNEED at off=%zu len=%zu: %s (continuing)\n",
                             a_start, a_len, std::strerror(errno));
            }
        }

        /* Per-chunk red-line check + progress log. */
        char tag[64];
        std::snprintf(tag, sizeof tag, "chunk %.0f/%.0f GiB",
                      (double)(off + this_chunk) / (1024.0*1024.0*1024.0),
                      total_gib);
        enforce_red_line(tag, verbose);

        if (verbose) {
            double now = now_sec();
            /* Log at most every 1 second, plus the final chunk. */
            int is_final = (off + this_chunk == m->file_size);
            if (is_final || (now - last_log) >= 1.0) {
                long rss_kb = read_rss_kb();
                long avail_kb = read_meminfo_kb("MemAvailable");
                double gb_done = (double)(off + this_chunk) / (1024.0*1024.0*1024.0);
                double rate = gb_done / (now - t_loop0 + 1e-9);
                std::fprintf(stderr,
                             "[load] %5.1f / %5.1f GiB  (%5.1f GiB/s avg)  RSS=%5ld MB  avail=%6ld MB\n",
                             gb_done, total_gib, rate,
                             rss_kb / 1024, avail_kb / 1024);
                last_log = now;
            }
        }
    }

    /* Final sweep: drop any residual mmap pages (the kernel may keep
     * recently-faulted pages in cache even after MADV_DONTNEED). */
    if (::madvise(m->mmap_ptr, m->file_size, MADV_DONTNEED) != 0) {
        if (verbose) {
            std::fprintf(stderr,
                         "[load] final madvise: %s (continuing)\n",
                         std::strerror(errno));
        }
    }

    double t_total = now_sec() - t_loop0;
    if (verbose) {
        std::fprintf(stderr,
                     "[load] done: %.2f GiB in %.2f s = %.1f GiB/s avg\n",
                     total_gib, t_total, total_gib / t_total);
    }

    /* Stamp the model handle. */
    m->weights_managed_base = managed;
    m->weights_managed_size = m->file_size;
    m->weight_backend = DS4_WEIGHT_BACKEND_MANAGED;

    enforce_red_line("99 post load", verbose);
    return 0;
}

extern "C" void ds4_model_managed_free(struct ds4_model *m)
{
    if (!m || !m->weights_managed_base) return;
    cudaError_t e = cudaFree(m->weights_managed_base);
    if (e != cudaSuccess) {
        std::fprintf(stderr,
                     "ds4_model_managed_free: cudaFree failed: %s\n",
                     cudaGetErrorString(e));
        /* Don't abort — the caller is on the close path. */
    }
    m->weights_managed_base = NULL;
    m->weights_managed_size = 0;
}

extern "C" const void *ds4_tensor_managed_ptr(const struct ds4_model *m,
                                              const struct ds4_tensor *t)
{
    if (!m || !t || !m->weights_managed_base) return NULL;
    if (t->abs_offset + t->byte_size > m->weights_managed_size) return NULL;
    return (const void *)((const uint8_t *)m->weights_managed_base + t->abs_offset);
}

extern "C" const void *ds4_tensor_device_ptr(const struct ds4_model *m,
                                             const struct ds4_tensor *t)
{
    if (!m || !t) return NULL;
    switch (m->weight_backend) {
    case DS4_WEIGHT_BACKEND_MMAP_DIRECT:
        if (!m->mmap_ptr) return NULL;
        if (t->abs_offset + t->byte_size > m->file_size) return NULL;
        return (const void *)((const uint8_t *)m->mmap_ptr + t->abs_offset);
    case DS4_WEIGHT_BACKEND_MANAGED:
    default:
        return ds4_tensor_managed_ptr(m, t);
    }
}

extern "C" void ds4_model_set_weight_backend(struct ds4_model *m,
                                             enum ds4_weight_backend backend)
{
    if (!m) return;
    m->weight_backend = backend;
}

extern "C" enum ds4_weight_backend ds4_model_weight_backend(
        const struct ds4_model *m)
{
    return m ? m->weight_backend : DS4_WEIGHT_BACKEND_MANAGED;
}

extern "C" const char *ds4_weight_backend_name(
        enum ds4_weight_backend backend)
{
    switch (backend) {
    case DS4_WEIGHT_BACKEND_MMAP_DIRECT:
        return "mmap_direct";
    case DS4_WEIGHT_BACKEND_MANAGED:
        return "managed";
    default:
        return "unknown";
    }
}
