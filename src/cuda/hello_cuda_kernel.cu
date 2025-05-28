#include <stdio.h>
#include "hello_cuda.h"

__device__ void hello_kernel() {
    printf("Hello World from block %d, thread %d!\n", 
           blockIdx.x, threadIdx.x);
}

__global__ void run_hello() {
    hello_kernel();
}

// C wrapper function implementation
extern "C" void launch_hello() {
    run_hello<<<1, 1>>>();
    
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
