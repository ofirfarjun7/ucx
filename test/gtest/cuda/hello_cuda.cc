//This file is used to test the hello_cuda.cu file

#include <common/test.h>
#include <common/test_helpers.h>
#include <cuda/hello_cuda.h>  // Use the installed header from src/cuda

class cuda_hello : public ucs::test {
protected:
    virtual void SetUp() {
        ucs::test::SetUp();  // Call parent's SetUp
    }
};

UCS_TEST_F(cuda_hello, basic) {
    launch_hello();  // Call the function from libucx_cuda
}