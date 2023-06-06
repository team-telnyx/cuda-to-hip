/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Timer.h>
#include <faiss/impl/FaissAssert.h>
#include <chrono>

namespace faiss {
namespace gpu {

KernelTimer::KernelTimer(hipStream_t stream)
        : startEvent_(0), stopEvent_(0), stream_(stream), valid_(true) {
    HIP_VERIFY(hipEventCreate(&startEvent_));
    HIP_VERIFY(hipEventCreate(&stopEvent_));

    HIP_VERIFY(hipEventRecord(startEvent_, stream_));
}

KernelTimer::~KernelTimer() {
    HIP_VERIFY(hipEventDestroy(startEvent_));
    HIP_VERIFY(hipEventDestroy(stopEvent_));
}

float KernelTimer::elapsedMilliseconds() {
    FAISS_ASSERT(valid_);

    HIP_VERIFY(hipEventRecord(stopEvent_, stream_));
    HIP_VERIFY(hipEventSynchronize(stopEvent_));

    auto time = 0.0f;
    HIP_VERIFY(hipEventElapsedTime(&time, startEvent_, stopEvent_));
    valid_ = false;

    return time;
}

CpuTimer::CpuTimer() {
    start_ = std::chrono::steady_clock::now();
}

float CpuTimer::elapsedMilliseconds() {
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start_;

    return duration.count();
}

} // namespace gpu
} // namespace faiss
