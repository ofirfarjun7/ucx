#include "gdaki_mem_handle.h"
#include <gdrapi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define PAGE_ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

struct gdaki_mem_handle {
    void*    gpu_ptr;
    void*    cpu_ptr;
    size_t   size;
    gdr_t    gdr;
    gdr_mh_t mh;
    bool     owns_gpu_mem;  // True if we allocated the GPU memory
};

gdaki_mem_handle_t gdaki_mem_create(void* gpu_ptr, size_t size) {
    gdaki_mem_handle_t handle = (gdaki_mem_handle_t)calloc(1, sizeof(*handle));
    int ret;

    if (!handle) {
        return NULL;
    }

    handle->size = PAGE_ROUND_UP(size, GPU_PAGE_SIZE);

    // Allocate GPU memory if not provided
    if (gpu_ptr == NULL) {
        cudaError_t cuda_err = cudaMalloc(&handle->gpu_ptr, handle->size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate GPU memory: %s\n", 
                    cudaGetErrorString(cuda_err));
            goto free;
        }
        handle->owns_gpu_mem = true;
        cuda_err = cudaMemset(handle->gpu_ptr, 0, handle->size);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Failed to initialize GPU memory: %s\n", 
                    cudaGetErrorString(cuda_err));
            goto free_cuda;
        }
    } else {
        assert(size % GPU_PAGE_SIZE == 0);
        handle->gpu_ptr = gpu_ptr;
        handle->owns_gpu_mem = false;
    }

    // Initialize GDRcopy
    handle->gdr = gdr_open();
    if (!handle->gdr) {
        fprintf(stderr, "GDRcopy initialization failed\n");
        goto free_cuda;
    }

    // Pin and map the buffer
    ret = gdr_pin_buffer(handle->gdr, (unsigned long)handle->gpu_ptr, handle->size, 0, 0, &handle->mh);
    if (ret) {
        fprintf(stderr, "GDRcopy pin buffer failed\n");
        goto close;
    }

    ret = gdr_map(handle->gdr, handle->mh, &handle->cpu_ptr, handle->size);
    if (ret) {
        fprintf(stderr, "GDRcopy map failed\n");
        goto unpin;
    }

    return handle;

unpin:
    gdr_unpin_buffer(handle->gdr, handle->mh);
close:
    gdr_close(handle->gdr);
free_cuda:
    if (handle->owns_gpu_mem) {
        cudaFree(handle->gpu_ptr);
    }
free:
    free(handle);
    return NULL;
}

void* gdaki_mem_get_ptr(gdaki_mem_handle_t handle) {
    return handle ? handle->cpu_ptr : NULL;
}

void* gdaki_mem_get_gpu_ptr(gdaki_mem_handle_t handle) {
    return handle ? handle->gpu_ptr : NULL;
}

void gdaki_mem_destroy(gdaki_mem_handle_t handle) {
    if (!handle) {
        return;
    }

    gdr_unmap(handle->gdr, handle->mh, handle->cpu_ptr, handle->size);
    gdr_unpin_buffer(handle->gdr, handle->mh);
    gdr_close(handle->gdr);
    
    if (handle->owns_gpu_mem) {
        cudaFree(handle->gpu_ptr);
    }
    
    free(handle);
} 