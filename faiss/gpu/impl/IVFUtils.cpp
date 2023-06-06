#include "hip/hip_runtime.h"
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/scan.h>
#include <faiss/gpu/impl/IVFUtils.h>
#include <faiss/gpu/utils/Tensor.h>
#include <faiss/gpu/utils/ThrustUtils.h>

#include <algorithm>

namespace faiss {
namespace gpu {

// Calculates the total number of intermediate distances to consider
// for all queries
__global__ void getResultLengths(
        Tensor<idx_t, 2, true> ivfListIds,
        idx_t* listLengths,
        idx_t totalSize,
        Tensor<idx_t, 2, true> length) {
    idx_t linearThreadId = idx_t(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;
    if (linearThreadId >= totalSize) {
        return;
    }

    auto nprobe = ivfListIds.getSize(1);
    auto queryId = linearThreadId / nprobe;
    auto listId = linearThreadId % nprobe;

    idx_t centroidId = ivfListIds[queryId][listId];

    // Safety guard in case NaNs in input cause no list ID to be generated
    length[queryId][listId] = (centroidId != -1) ? listLengths[centroidId] : 0;
}

void runCalcListOffsets(
        GpuResources* res,
        Tensor<idx_t, 2, true>& ivfListIds,
        DeviceVector<idx_t>& listLengths,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<char, 1, true>& thrustMem,
        hipStream_t stream) {
    FAISS_ASSERT(ivfListIds.getSize(0) == prefixSumOffsets.getSize(0));
    FAISS_ASSERT(ivfListIds.getSize(1) == prefixSumOffsets.getSize(1));

    idx_t totalSize = ivfListIds.numElements();

    idx_t numThreads = std::min(totalSize, (idx_t)getMaxThreadsCurrentDevice());
    idx_t numBlocks = utils::divUp(totalSize, numThreads);

    auto grid = dim3(numBlocks);
    auto block = dim3(numThreads);

    hipLaunchKernelGGL(getResultLengths, grid, block, 0, stream, 
            ivfListIds, listLengths.data(), totalSize, prefixSumOffsets);
    HIP_TEST_ERROR();

    // Prefix sum of the indices, so we know where the intermediate
    // results should be maintained
    // Thrust wants a place for its temporary allocations, so provide
    // one, so it won't call hipMalloc/Free if we size it sufficiently
    ThrustAllocator alloc(
            res, stream, thrustMem.data(), thrustMem.getSizeInBytes());

    thrust::inclusive_scan(
            thrust::hip::par(alloc).on(stream),
            prefixSumOffsets.data(),
            prefixSumOffsets.data() + totalSize,
            prefixSumOffsets.data());
    HIP_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
