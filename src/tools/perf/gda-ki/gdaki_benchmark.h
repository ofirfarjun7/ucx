#ifndef GDAKI_BENCHMARK_H
#define GDAKI_BENCHMARK_H

#ifdef __cplusplus
extern "C" {
#endif

// C wrapper function to launch the kernel
void launch_bw_test(void);
void launch_lat_test(void);
#ifdef __cplusplus
}
#endif

#endif // GDAKI_BENCHMARK_H
