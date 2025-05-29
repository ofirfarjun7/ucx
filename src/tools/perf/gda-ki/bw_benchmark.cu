#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <gdrapi.h>

#include "libperf_cuda.h"
#include "gdaki_mem_handle.h"

// Implementation of device function
__device__ void uct_post_batch(uct_gdaki_packed_batch_t *batch) {
    printf("GDAKI post batch dummy on block %d, thread %d!\n", 
           blockIdx.x, threadIdx.x);
}

// TODO: Add qp and cq etc
__global__ void run_bw_test(ucx_perf_context_cuda_t *ctx)
{
    ucx_perf_cuda_time_t next_report_time;

    if (threadIdx.x == 0) {
        ctx->start_time = gdaki_get_time_ns();
        ctx->test_completed = 0;
        ctx->results_ready = 0;
        ctx->active_buffer = 0;
        memset((void*)&ctx->current[0], 0, sizeof(ctx->current[0]));
        memset((void*)&ctx->current[1], 0, sizeof(ctx->current[1]));
        memset((void*)&ctx->prev[0], 0, sizeof(ctx->prev[0]));
        memset((void*)&ctx->prev[1], 0, sizeof(ctx->prev[1]));
        next_report_time = gdaki_get_time_ns() + ctx->params.report_interval;
    }
    __syncthreads();

    for (uint32_t idx = 0; idx < ctx->max_iter; idx++) {
        while (ctx->params.m_sends_outstanding > ctx->params.max_outstanding) {
            // TODO: Progress and wait for completion of outstanding batches
        }

        // Call to new gdaki kernel
        uct_post_batch(ctx->params.batch);

        if (threadIdx.x == 0) {
            ctx->params.m_sends_outstanding++;
            // Update current buffer metrics
            ucx_perf_cuda_update(ctx, 1, ctx->params.batch->batch_total_size);

            // Check if it's time to report
            ucx_perf_cuda_time_t current_time = gdaki_get_time_ns();
            if (current_time >= next_report_time) {
                // Switch to other buffer
                ctx->active_buffer ^= 1;
                // Signal CPU to calculate and print
                ctx->results_ready = 1;
                // Set next report time
                next_report_time = current_time + ctx->params.report_interval;
            }
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ctx->end_time = gdaki_get_time_ns();
        ctx->test_completed = 1;
    }
}

// C wrapper function implementation
// TODO: pass struct with test parameters.
extern "C" void launch_bw_test() {
    // TODO: Initialize measurement metrics
    ucx_perf_cuda_result_t result = {0};
    gdaki_mem_handle_t mem_handle = NULL;
    ucx_perf_context_cuda_t *gpu_ctx = NULL;
    ucx_perf_context_cuda_t *cpu_ctx;
    ucx_perf_cuda_time_t poll_interval;

    mem_handle = gdaki_mem_create(NULL, sizeof(ucx_perf_context_cuda_t));
    if (!mem_handle) {
        printf("Failed to create GDRcopy memory handle using managed memory\n");
        cudaMallocManaged(&gpu_ctx, sizeof(ucx_perf_context_cuda_t));
        cpu_ctx = gpu_ctx;
    } else {
        cpu_ctx = (ucx_perf_context_cuda_t*)gdaki_mem_get_ptr(mem_handle);
        gpu_ctx = (ucx_perf_context_cuda_t*)gdaki_mem_get_gpu_ptr(mem_handle);
    }

    //TODO: Initialize context test parameters.

    // Launch kernel asynchronously
    run_bw_test<<<1, 1>>>(gpu_ctx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    poll_interval = cpu_ctx->params.report_interval / 10;
    while (!cpu_ctx->test_completed) {
        // Check if GPU has new results ready
        if (cpu_ctx->results_ready) {
            // Read from inactive buffer (opposite of GPU's active buffer)
            int read_buf = 1 - cpu_ctx->active_buffer;
            ucx_perf_calc_cuda_result(cpu_ctx, read_buf, &result);
            ucx_perf_cuda_report(&result);
            // Clear the ready flag after processing
            cpu_ctx->results_ready = 0;
        }
        usleep(poll_interval); // Small sleep to prevent busy-waiting
    }
    
    printf("\nFinal Results:\n");
    printf("Total messages: %lu\n", cpu_ctx->max_iter);
    printf("Total bytes: %lu\n", result.bytes);
    printf("Total time: %.3f seconds\n", result.elapsed_time);
    printf("Average bandwidth: %.2f Gbps\n", result.bandwidth.total_average);
    printf("Average latency: %.3f ns\n", result.latency.total_average);
    printf("Average message rate: %.2f Mps\n", result.msgrate.total_average);

cleanup:
    if (mem_handle) {
        gdaki_mem_destroy(mem_handle);
    } else if (gpu_ctx) {
        cudaFree(gpu_ctx);
    }
}
