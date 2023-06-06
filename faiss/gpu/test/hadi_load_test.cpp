

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

// struct LoadCode32 {
//     static inline  __device__ void load(
//             unsigned int code32[1],
//             uint8_t* p,
//             int offset) {
//         // p += offset * 1;

//         asm volatile("s_load_dword %0, %1, 0x0"
//                  : "=s"(code32[0])
//                  : "s"(reinterpret_cast<uint32_t*>(p)));

//         // asm volatile("s_load_dword s2, %0, 0x0;" : : "s"(reinterpret_cast<unsigned int*>(p)) : "s2", "memory");
//         // // asm volatile("s_load_dword s2, %0, 0x0;" : : "s"(p) : "s2", "memory");
//         // asm volatile("s_mov_b32 %0, s2;" : "=s"(code32[0]) : : "s2", "memory");
//     }
// };

struct LoadCode32 {
    static inline  __device__ void load(
            unsigned int code32[1],
            uint32_t* p,
            int offset) {
        // p += offset * 1;

        // asm volatile("s_load_dword %0, %1, 0x0"
        //          : "=s"(code32[0])
        //          : "s"(p));

        asm volatile("s_load_dword s2, %0, 0x0;" : : "s"(reinterpret_cast<unsigned int*>(p)) : "s2", "memory");
        // asm volatile("s_load_dword s2, %0, 0x0;" : : "s"(p) : "s2", "memory");
        asm volatile("s_mov_b32 %0, s2;" : "=s"(code32[0]) : : "s2", "memory");
    }
};

// Kernel function to call the load function
__global__ void loadCode32Kernel(unsigned int* result, uint32_t* p, int offset) {
    LoadCode32::load(result, p, offset);
}

// Test case for load function
TEST(LoadTest, LoadFunctionTest)
{
    unsigned int code32[1] = {0};
    // uint8_t p[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    uint32_t p = 0x1;
    // for (int i = 0; i < 4; i++) {
    //     std::cout << i << " >> " << (int)p[i] << std::endl;
    // }

    int offset = 0;

    // Allocate GPU memory for input and output
    unsigned int* d_code32;
    // uint8_t* d_p;
    uint32_t* d_p;
    hipMalloc(&d_code32, sizeof(unsigned int));
    // hipMalloc(&d_p, 8 * sizeof(uint8_t));
    hipMalloc(&d_p, sizeof(uint32_t));

    // Copy input data from host to GPU memory
    hipMemcpy(d_code32, code32, sizeof(unsigned int), hipMemcpyHostToDevice);
    // hipMemcpy(d_p, p, 8 * sizeof(uint8_t), hipMemcpyHostToDevice);
    hipMemcpy(d_p, &p, sizeof(uint32_t), hipMemcpyHostToDevice);

    // Launch the kernel
    dim3 grid(1);
    dim3 block(1);
    // lihLaunch<<<grid, block>>>(LoadCode32(), d_code32, d_p, offset);
    hipLaunchKernelGGL(loadCode32Kernel, grid, block, 0, 0, d_code32, d_p, offset);

    // Copy the result from GPU memory to host
    hipMemcpy(code32, d_code32, sizeof(unsigned int), hipMemcpyDeviceToHost);

    // Clean up GPU memory
    hipFree(d_code32);
    hipFree(d_p);

    // Assert the expected result
    // Modify this according to your expected result
    std::cout << "val = " << *static_cast<uint32_t*>(code32) << std::endl;


    EXPECT_EQ(code32[0], 0);
}

// Run the tests
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
