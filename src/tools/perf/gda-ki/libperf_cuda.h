#ifndef LIBPERF_CUDA_H_
#define LIBPERF_CUDA_H_

#include <stddef.h>  // For size_t
#include <stdint.h>  // For uint64_t

typedef unsigned long long ucx_perf_cuda_time_t;

//TODO: Replace with real packed batch API
typedef struct uct_gdaki_packed_batch {
    void* qp;
    uint64_t batch_length;
    uint64_t batch_total_size;
    uint8_t** src_buf;
    uint64_t* sizes;
    uint32_t* src_mkey;
    uint8_t** dst_buf;
    uint32_t* dst_mkey;
} uct_gdaki_packed_batch_t;

//TODO: Improve code resue by resusing CPU perftest code.
typedef struct ucx_perf_cuda_result {
    uint64_t           iters;
    unsigned long long elapsed_time;
    uint64_t           bytes;
    struct {
        double         percentile;
        double         moment_average; /* Average since last report */
        double         total_average;  /* Average of the whole test */
    }
    latency, bandwidth, msgrate;
} ucx_perf_cuda_result_t;

/**
 * Describes a performance test.
 */
typedef struct ucx_perf_params_cuda {
    uct_gdaki_packed_batch_t* batch;
    unsigned                  max_outstanding; /* Maximal number of outstanding sends */
    unsigned                  m_sends_outstanding;
    uint64_t                  warmup_iter;     /* Number of warm-up iterations */
    double                    warmup_time;     /* Approximately how long to warm-up */
    uint64_t                  max_iter;        /* Iterations limit, 0 - unlimited */
    ucx_perf_cuda_time_t      max_time;        /* Time limit (seconds), 0 - unlimited */
    ucx_perf_cuda_time_t      report_interval; /* Interval at which to call the report callback */

} ucx_perf_params_cuda_t;

typedef struct ucx_perf_context_cuda {
    ucx_perf_params_cuda_t   params;

    /* Measurements */
    ucx_perf_cuda_time_t     start_time;      /* inaccurate end time (upper bound) */
    ucx_perf_cuda_time_t     end_time;        /* inaccurate end time (upper bound) */
    ucx_perf_cuda_time_t     prev_time;       /* time of previous iteration */
    ucx_perf_cuda_time_t     report_interval; /* interval of showing report */
    uint64_t                 last_report;     /* last report to CPU */
    uint64_t                 max_iter;
    volatile int             test_completed;    // Signal test completion

    int                      active_buffer;
    /* Measurements of current/previous **report** */
    struct {
        uint64_t             msgs;    /* number of messages */
        uint64_t             bytes;   /* number of bytes */
        uint64_t             iters;   /* number of iterations */
        ucx_perf_cuda_time_t time;    /* inaccurate time (for median and report interval) */
    } current[2], prev[2];

    volatile int             results_ready;     // Signal CPU to calculate and print
} ucx_perf_context_cuda_t;

void inline ucx_perf_calc_cuda_result(ucx_perf_context_cuda_t *perf, int read_buf, ucx_perf_cuda_result_t *result)
{
    result->latency.moment_average =
        (perf->current[read_buf].time - perf->prev[read_buf].time)
        / (perf->current[read_buf].iters - perf->prev[read_buf].iters);
    
    result->latency.total_average =
        (perf->current[read_buf].time - perf->start_time)
        / perf->current[read_buf].iters;
    
    result->bandwidth.moment_average =
        (perf->current[read_buf].bytes - perf->prev[read_buf].bytes) /
        (perf->current[read_buf].time - perf->prev[read_buf].time);
    
    result->bandwidth.total_average =
        perf->current[read_buf].bytes /
        (perf->current[read_buf].time - perf->start_time);
    
    result->msgrate.moment_average =
        (perf->current[read_buf].msgs - perf->prev[read_buf].msgs) /
        (perf->current[read_buf].time - perf->prev[read_buf].time);

    result->msgrate.total_average =
        perf->current[read_buf].msgs /
        (perf->current[read_buf].time - perf->start_time);
}

void inline ucx_perf_cuda_report(ucx_perf_cuda_result_t *result)
{
    printf("Latency: %.3f ns\n", result->latency.moment_average);
    printf("Bandwidth: %.2f Gbps\n", result->bandwidth.moment_average);
    printf("Message rate: %.2f Mps\n", result->msgrate.moment_average);
}

#ifdef __CUDACC__
#define GDAKI_DEVICE_GET_TIME_NS(globaltimer) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer))

__device__ 
static inline unsigned long long gdaki_get_time_ns(void) {
    unsigned long long globaltimer;
    GDAKI_DEVICE_GET_TIME_NS(globaltimer);
    return globaltimer;
}

__device__
static inline void ucx_perf_cuda_update(ucx_perf_context_cuda_t *perf,
                                        uint64_t iters,
                                        size_t bytes)
{
    perf->current[perf->active_buffer].time   = gdaki_get_time_ns(); // TODO: capture time
    perf->current[perf->active_buffer].iters += iters;
    perf->current[perf->active_buffer].bytes += bytes;
    perf->current[perf->active_buffer].msgs  += 1;

    perf->prev_time = perf->current[perf->active_buffer].time;
}

#endif // __CUDACC__

#endif // LIBPERF_CUDA_H_
