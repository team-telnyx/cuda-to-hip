#include <hip/hip_runtime.h>
#include <stdio.h>
#include <iostream>

__device__ void load(unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 1;

/*
        asm volatile("s_load_dword s2, %0, 0x0;" : : "s"(reinterpret_cast<unsigned int*>(p)) : "s2", "memory");
        asm volatile("s_mov_b32 %0, s2;" : "=s"(code32[0]) : : "s2", "memory");
        uint64_t address;
        asm volatile(
            "s_mov_b64 %0, %1"
            : "=s"(address)
            : "s"(p)
        );
*/

        uint32_t alignedAddress = reinterpret_cast<uintptr_t>(p) ; // & ~0x03
/*
        unsigned int result;
        unsigned int* buffer = nullptr;  // Assume buffer points to a valid buffer resource
        unsigned int offset2 = 0;  // Offset for buffer access
        asm volatile(
            "s_buffer_load_dword %0, s[4:7] offset:0" :"=s"(result):: "s4", "s5", "s6", "s7", "memory"
        );
*/        
        asm volatile(
            "s_load_dword %0, %1 offset:0 glc"
            : "=s"(code32[0])
            : "s"(reinterpret_cast<const uint32_t*>(p))
            : "memory"
        );

    }

__global__ void testKernel(unsigned int code32[1], uint8_t* data, size_t size) {
    
    load(code32, data, 1);
}

int main() {
    const size_t size = 100;
    uint8_t data[size] = {    1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    unsigned int code32[1] = {0};

    // Allocate memory on the device
    unsigned int* d_code32;
    hipMalloc(&d_code32, sizeof(unsigned int));
    // Copy input data from host to GPU memory
    hipMemcpy(d_code32, code32, sizeof(unsigned int), hipMemcpyHostToDevice);

    uint8_t* deviceData;
    hipMalloc(&deviceData, sizeof(uint8_t) * size);

    // Copy the input data to the device
    hipMemcpy(deviceData, data, sizeof(uint8_t) * size, hipMemcpyHostToDevice);

    // Launch the test kernel
    hipLaunchKernelGGL(testKernel, dim3((size + 255) / 256), dim3(256), 0, 0, d_code32, deviceData, size);

    // Copy the result from GPU memory to host
    hipMemcpy(code32, d_code32, sizeof(unsigned int), hipMemcpyDeviceToHost);
    
    std::cout << "val = " << *static_cast<uint32_t*>(code32) << std::endl;

    // Free the device memory
    hipFree(deviceData);

    return 0;
}

