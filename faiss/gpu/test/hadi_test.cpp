

#include <faiss/gpu/utils/PtxUtils.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <sys/time.h>

#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

struct LoadCode32 {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 1;
            
        asm volatile(
            "s_load_dwordx2 s[2:3], %1, 0x0;"
            "s_mov_b32 %0, s2;"
            : "=s"(code32[0])
            : "s"(p)
            : "s2", "s3", "memory"
        );
    }
};

__global__ void loadCode32Kernel(unsigned int* result, uint8_t* p, int offset) {
    LoadCode32::load(result, p, offset);
}

TEST(LoadCode32Test, LoadsCorrectCode) {
    unsigned int code32[1] = {0};
    uint8_t memory[8] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    unsigned int* d_code32;
    uint8_t* d_memory;
    hipMalloc(&d_code32, sizeof(unsigned int));
    hipMalloc(&d_memory, sizeof(memory));
    hipMemcpy(d_memory, memory, sizeof(memory), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(loadCode32Kernel, dim3(1), dim3(1), 0, 0, d_code32, d_memory, 0);
    hipMemcpy(code32, d_code32, sizeof(unsigned int), hipMemcpyDeviceToHost);
    EXPECT_EQ(code32[0], 0x78563412);

    hipLaunchKernelGGL(loadCode32Kernel, dim3(1), dim3(1), 0, 0, d_code32, d_memory, 1);
    hipMemcpy(code32, d_code32, sizeof(unsigned int), hipMemcpyDeviceToHost);
    EXPECT_EQ(code32[0], 0xF0DEBC9A);

    hipFree(d_code32);
    hipFree(d_memory);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
