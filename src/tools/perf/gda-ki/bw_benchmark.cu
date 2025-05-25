#include <stdio.h>

__device__ void bw_test() {
    printf("Testing BW on block %d, thread %d!\n", 
           blockIdx.x, threadIdx.x);
}

__global__ void run_bw_test() {
    bw_test();
}

// C wrapper function implementation
extern "C" void launch_bw_test() {
    run_bw_test<<<1, 1>>>();
    
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
