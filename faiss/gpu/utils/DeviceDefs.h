/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>

namespace faiss {
namespace gpu {

// We require at least CUDA 8.0 for compilation
// #if CUDA_VERSION < 8000
// #error "CUDA >= 8.0 is required"
// #endif

// We validate this against the actual architecture in device initialization
constexpr int kWarpSize = 64;

// This is a memory barrier for intra-warp writes to shared memory.
__inline__ __device__ void warpFence() {
// #if defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
// #else
    // For the time being, assume synchronicity.
    //  __threadfence_block();
// #endif
}

#if defined(__HIP_PLATFORM_NVCC__) && (__CUDA_ARCH__ > 9000)
// Based on the CUDA version (we assume what version of nvcc/ptxas we were
// compiled with), the register allocation algorithm is much better, so only
// enable the 2048 selection code if we are above 9.0 (9.2 seems to be ok)
#define GPU_MAX_SELECTION_K 2048
#else
// TODO: HADI warp 1024 > 2048
#define GPU_MAX_SELECTION_K 1024
#endif

} // namespace gpu
} // namespace faiss
