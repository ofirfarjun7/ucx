#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2024. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_NVCC

# Build shared library
lib_LTLIBRARIES = libperf_gdaki.la

# Library sources
libperf_gdaki_la_SOURCES = \
    gdaki_benchmark.h \
    bw_benchmark.cu \
    lat_benchmark.cu

# Add include directory
gdakidir = $(includedir)/gda-ki
gdaki_HEADERS = gdaki_benchmark.h

# Compile CUDA files with NVCC wrapper
SUFFIXES = .cu
NVCC_WRAP = $(srcdir)/nvcc_wrap.sh
.cu.lo:
	$(LIBTOOL) --tag=CXX --mode=compile $(NVCC_WRAP) -c $< -o $@

# Library version info (major:minor:revision)
libperf_gdaki_la_LDFLAGS = -version-info 0:0:0

# Include paths
libperf_gdaki_la_CPPFLAGS = \
    -I$(top_srcdir)/src

# Link with CUDA runtime and C++ standard library
libperf_gdaki_la_LIBADD = -lcuda -lcudart -lstdc++

# Make sure wrapper script is executable
EXTRA_DIST = gdaki_benchmark.h bw_benchmark.cu lat_benchmark.cu nvcc_wrap.sh

endif # HAVE_NVCC 