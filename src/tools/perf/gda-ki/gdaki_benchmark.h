#ifndef GDAKI_BENCHMARK_H
#define GDAKI_BENCHMARK_H

#include "libperf_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

// C wrapper function to launch the kernel
void launch_bw_test();
void launch_lat_test();
#ifdef __cplusplus
}
#endif

#endif // GDAKI_BENCHMARK_H
