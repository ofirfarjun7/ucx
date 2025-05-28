#include <stdio.h>

__device__ void lat_test() {
    printf("Testing LAT on block %d, thread %d!\n", 
           blockIdx.x, threadIdx.x);
}

__global__ void run_lat_test() {
    lat_test();
}

extern "C" void launch_lat_test() {
    run_lat_test<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        return;
    }
    
    while (0) {
        // TODO: wait for test completion and report metrics.
    }
}
