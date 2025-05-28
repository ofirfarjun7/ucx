#ifndef LIBPERF_CUDA_H_
#define LIBPERF_CUDA_H_

#include <stddef.h>  // For size_t
#include <stdint.h>  // For uint64_t

#define TIMING_QUEUE_SIZE    2048

typedef struct uct_gdaki_packed_batch {
    void* qp;
    uint64_t batch_length;
    uint8_t** src_buf;
    uint64_t* sizes;
    uint32_t* src_mkey;
    uint8_t** dst_buf;
    uint32_t* dst_mkey;
} uct_gdaki_packed_batch_t;

/**
 * Describes a performance test.
 */
typedef struct ucx_perf_params_cuda {
    uct_gdaki_packed_batch_t* batch;
    unsigned                  max_outstanding; /* Maximal number of outstanding sends */
    uint64_t        warmup_iter;     /* Number of warm-up iterations */
    double                    warmup_time;     /* Approximately how long to warm-up */
    uint64_t        max_iter;        /* Iterations limit, 0 - unlimited */
    double                    max_time;        /* Time limit (seconds), 0 - unlimited */
    double                    report_interval; /* Interval at which to call the report callback */
    double                    percentile_rank; /* The percentile rank of the percentile reported
                                               in latency tests */
} ucx_perf_params_cuda_t;

typedef struct ucx_perf_context_cuda {
    ucx_perf_params_cuda_t       params;

    /* Measurements */
    double                       start_time_acc;  /* accurate start time */
    unsigned long                   end_time;        /* inaccurate end time (upper bound) */
    unsigned long                   prev_time;       /* time of previous iteration */
    unsigned long                   report_interval; /* interval of showing report */
    uint64_t           last_report;     /* last report to CPU */
    uint64_t           max_iter;

    /* Measurements of current/previous **report** */
    struct {
        uint64_t       msgs;    /* number of messages */
        uint64_t       bytes;   /* number of bytes */
        uint64_t       iters;   /* number of iterations */
        unsigned long               time;    /* inaccurate time (for median and report interval) */
        double                   time_acc; /* accurate time (for avg latency/bw/msgrate) */
    } current, prev;

    unsigned long                   timing_queue[TIMING_QUEUE_SIZE];
    unsigned                     timing_queue_head;
} ucx_perf_context_cuda_t;

#ifdef __CUDA_ARCH__
__device__
static inline void ucx_perf_update(ucx_perf_context_cuda_t *perf,
                                   uint64_t iters,
                                   size_t bytes)
{
    perf->current.time   = 0; // TODO: capture time
    perf->current.iters += iters;
    perf->current.bytes += bytes;
    perf->current.msgs  += 1;

    perf->timing_queue[perf->timing_queue_head] =
                    perf->current.time - perf->prev_time;
    ++perf->timing_queue_head;
    if (perf->timing_queue_head == TIMING_QUEUE_SIZE) {
        perf->timing_queue_head = 0;
    }

    perf->prev_time = perf->current.time;

    if ((perf->current.time - perf->prev.time) >= perf->report_interval) {
        // TODO: report to CPU
    }
}
#endif

#endif // LIBPERF_CUDA_H_
