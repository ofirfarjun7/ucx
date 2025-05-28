#!/bin/bash

output=""
src=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            output="$2"
            shift 2
            ;;
        -c)
            shift
            ;;
        -*)
            # Skip other options
            shift
            ;;
        *)
            src="$1"
            shift
            ;;
    esac
done

if [ -z "$output" ] || [ -z "$src" ]; then
    echo "Error: Missing output or source file"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $(dirname "$output")

# Call NVCC
/hpc/local/oss/cuda12.4.0/redhat8/bin/nvcc -Xcompiler "-fPIC" $NVCC_FLAGS -c "$src" -o "$output"
