#ifndef HELLO_CUDA_H
#define HELLO_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// C wrapper function to launch the kernel
void launch_hello(void);

#ifdef __cplusplus
}
#endif

#endif // HELLO_CUDA_H
