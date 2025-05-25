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
__device__ void bw_test() {
    printf("Testing BW on block %d, thread %d!\n", 
           blockIdx.x, threadIdx.x);
}

// TODO: Add qp and cq etc
__global__ void run_bw_test(
    struct gdaki_perf_context *perf_ctx,
	uint32_t num_iters,
    unsigned int *m_sends_outstanding,
    unsigned int send_window_size,
	uint32_t *size,
	uint8_t *src_buf,
	uint32_t *src_mkey,
	uint8_t *dst_buf,
	uint32_t *dst_mkey)
{

	#if KERNEL_DEBUG_TIMES == 1
		unsigned long long step1 = 0, step2 = 0, step3 = 0;
	#endif

	for (uint32_t idx = threadIdx.x; idx < num_iters; idx += blockDim.x) {
		#if KERNEL_DEBUG_TIMES == 1
			DOCA_GPUNETIO_DEVICE_GET_TIME(step1);
		#endif

        // TODO: Wait send window and advance sends outstanding counter
        while (!atomicInc(m_sends_outstanding, send_window_size + 1)) {
            // TODO: Progress and wait for completion of outstanding sends
        }
        // TODO: Call ucx gdaki put batch warp.
        bw_test();

		#if KERNEL_DEBUG_TIMES == 1
			DOCA_GPUNETIO_DEVICE_GET_TIME(step2);
		#endif

		#if KERNEL_DEBUG_TIMES == 1
			DOCA_GPUNETIO_DEVICE_GET_TIME(step3);
		#endif

		#if KERNEL_DEBUG_TIMES == 1
		if (threadIdx.x == 0)
			printf("iteration %d src_buf %lx size %d dst_buf %lx put %ld ns, poll %ld ns\n",
				idx, src_buf, size, dst_buf, step2-step1, step3-step2);
		#endif
	}

    // TODO: Poll for completion of all CQEs
    __syncthreads();

    // TODO: Update the context with the new measurements
}

// C wrapper function implementation
extern "C" void launch_bw_test() {
    run_bw_test<<<1, 1>>>(NULL, 1, NULL, 1, NULL, NULL, NULL, NULL, NULL);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", 
                cudaGetErrorString(err));
        return;
    }
}
