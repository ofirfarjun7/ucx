#include <stdio.h>
#include <stdint.h>

#include "libperf_cuda.h"

typedef unsigned long   ucs_time_t;
typedef uint64_t ucx_perf_counter_t;

struct gdaki_perf_context {
    /* Measurements */
    double                       start_time_acc;  /* accurate start time */
    ucs_time_t                   end_time;        /* inaccurate end time (upper bound) */
    ucs_time_t                   prev_time;       /* time of previous iteration */
    ucs_time_t                   report_interval; /* interval of showing report */
    ucx_perf_counter_t           max_iter;

    /* Measurements of current/previous **report** */
    struct {
        ucx_perf_counter_t       msgs;    /* number of messages */
        ucx_perf_counter_t       bytes;   /* number of bytes */
        ucx_perf_counter_t       iters;   /* number of iterations */
        ucs_time_t               time;    /* inaccurate time (for median and report interval) */
        double                   time_acc; /* accurate time (for avg latency/bw/msgrate) */
    } current, prev;
};

// Implementation of device function
__device__ void uct_post_batch(uct_gdaki_packed_batch_t *batch) {
    printf("GDAKI post batch dummy on block %d, thread %d!\n", 
           blockIdx.x, threadIdx.x);
}

// TODO: Add qp and cq etc
__global__ void run_bw_test(ucx_perf_context_cuda_t *ctx)
{

    // TODO: Initialize metrices
    uint32_t m_sends_outstanding = 0;

	for (uint32_t idx = 0; idx < ctx->params.max_iter; idx ++) {

		while (m_sends_outstanding > ctx->params.max_outstanding) {
			// TODO: Progress and wait for completion of outstanding batches
		}

		// Call to new gdaki kernel
		uct_post_batch(ctx->params.batch);

        if (threadIdx.x == 0) {
		    // TODO: update m_sends_outstanding, metrices and notify CPU
        }
	}

    __syncthreads();
}

// C wrapper function implementation
extern "C" void launch_bw_test(ucx_perf_context_cuda_t *ctx) {

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        return;
    }

    run_bw_test<<<1, 1>>>(ctx);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", 
                cudaGetErrorString(err));
        return;
    }
}
