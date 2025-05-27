#include <stdio.h>
#include <stdint.h>

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
__device__ void uct_post_batch(
    void *qp,
    uint32_t batch_length,
    uint8_t **src_buf,
    uint32_t *size,
    uint32_t *src_mkey,
    uint8_t **dst_buf,
    uint32_t *dst_mkey) {
    printf("GDAKI post batch dummy on block %d, thread %d!\n", 
           blockIdx.x, threadIdx.x);
}

// TODO: Add qp and cq etc
__global__ void run_bw_test(
    void *qp,
	uint32_t send_window_size,
	uint32_t num_iters,
	uint32_t batch_length,
	uint8_t **src_buf,
	uint32_t *size,
	uint32_t *src_mkey,
	uint8_t **dst_buf,
	uint32_t *dst_mkey)
{

    // TODO: Initialize metrices
    struct gdaki_perf_context *ctx = NULL;
    uint32_t m_sends_outstanding = 0;

	for (uint32_t idx = 0; idx < num_iters; idx ++) {

		while (m_sends_outstanding > send_window_size) {
			// TODO: Progress and wait for completion of outstanding batches
		}

		// Call to new gdaki kernel
		uct_post_batch(qp, batch_length, src_buf, size, src_mkey, dst_buf, dst_mkey);

		// TODO: update m_sends_outstanding and update metrices
        if (threadIdx.x == 0) {
            if (ctx == NULL) {
            }
        }
	}

    __syncthreads();
}

// C wrapper function implementation
extern "C" void launch_bw_test(
    void *qp,
    uint32_t send_window_size,
    uint32_t num_iters,
    uint32_t batch_length,
    uint8_t **src_buf,
    uint32_t *size,
    uint32_t *src_mkey,
    uint8_t **dst_buf,
    uint32_t *dst_mkey) {

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        return;
    }

    run_bw_test<<<1, 1>>>(qp, send_window_size, num_iters, batch_length, src_buf, size, src_mkey, dst_buf, dst_mkey);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", 
                cudaGetErrorString(err));
        return;
    }
}
