#include "hip/hip_runtime.h"
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/DeviceDefs.h>
#include <faiss/gpu/utils/DeviceTensor.h>
#include <faiss/gpu/utils/Select.h>

namespace faiss {
namespace gpu {

// Number of warps that the kernel is instantiated with
constexpr int kWarps = 8;
constexpr int kLanes = kWarpSize;

constexpr int kMaxDistance = std::numeric_limits<int>::max();

// Performs a binary matrix multiplication, returning the lowest k results in
// `vecs` for each `query` in terms of Hamming distance (a fused kernel)
// Each warp calculates distance for a single query
template <int NumWarpQ, int NumThreadQ, typename BinaryType>
__launch_bounds__(kWarps* kLanes) __global__ void binaryDistanceAnySize(
        const Tensor<BinaryType, 2, true> vecs,
        const Tensor<BinaryType, 2, true> query,
        Tensor<int, 2, true> outK,
        Tensor<idx_t, 2, true> outV,
        int k) {
    // A matrix tile (query, k)
    __shared__ BinaryType queryTile[kWarps][kLanes + 1]; // avoid bank conflict

    // B matrix tile (vec, k)
    __shared__ BinaryType vecTile[kLanes][kLanes + 1]; // avoid bank conflict

    WarpSelect<
            int,
            idx_t,
            false,
            Comparator<int>,
            NumWarpQ,
            NumThreadQ,
            kWarps * kLanes>
            heap(kMaxDistance, -1, k);

    int warpId = hipThreadIdx_y;
    int laneId = hipThreadIdx_x;

    // Each warp handles a single query
    idx_t warpQuery = idx_t(hipBlockIdx_x) * kWarps + warpId;
    bool queryInBounds = warpQuery < query.getSize(0);

    // Each warp loops through the entire chunk of vectors
    for (idx_t blockVec = 0; blockVec < vecs.getSize(0); blockVec += kLanes) {
        int threadDistance = 0;

        // Reduction dimension
        for (idx_t blockK = 0; blockK < vecs.getSize(1); blockK += kLanes) {
            idx_t laneK = blockK + laneId;
            bool kInBounds = laneK < vecs.getSize(1);

            queryTile[warpId][laneId] =
                    queryInBounds && kInBounds ? query[warpQuery][laneK] : 0;

            // kWarps warps are responsible for loading 32 vecs
#pragma unroll
            for (int i = 0; i < kLanes / kWarps; ++i) {
                int warpVec = i * kWarps + warpId;
                idx_t vec = blockVec + warpVec;
                bool vecInBounds = vec < vecs.getSize(0);

                vecTile[warpVec][laneId] =
                        vecInBounds && kInBounds ? vecs[vec][laneK] : 0;
            }

            __syncthreads();

            // Compare distances
#pragma unroll
            for (int i = 0; i < kLanes; ++i) {
                threadDistance +=
                        __popc(queryTile[warpId][i] ^ vecTile[laneId][i]);
            }

            __syncthreads();
        }

        // Lanes within a warp are different vec results against the same query
        // Only submit distances which represent real (query, vec) pairs
        bool valInBounds =
                queryInBounds && (blockVec + laneId < vecs.getSize(0));
        threadDistance = valInBounds ? threadDistance : kMaxDistance;
        idx_t id = valInBounds ? blockVec + laneId : idx_t(-1);

        heap.add(threadDistance, id);
    }

    heap.reduce();

    if (warpQuery < query.getSize(0)) {
        heap.writeOut(outK[warpQuery].data(), outV[warpQuery].data(), k);
    }
}

// Version of the kernel that avoids a loop over the reduction dimension, and
// thus avoids reloading the query vectors
template <
        int NumWarpQ,
        int NumThreadQ,
        typename BinaryType,
        int ReductionLimit = kLanes>
__global__ void __launch_bounds__(kWarps* kLanes) binaryDistanceLimitSize(
        const Tensor<BinaryType, 2, true> vecs,
        const Tensor<BinaryType, 2, true> query,
        Tensor<int, 2, true> outK,
        Tensor<idx_t, 2, true> outV,
        int k) {
    // A matrix tile (query, k)
    __shared__ BinaryType queryTile[kWarps][kLanes + 1]; // avoid bank conflict

    // B matrix tile (vec, k)
    __shared__ BinaryType vecTile[kLanes][kLanes + 1]; // avoid bank conflict

    WarpSelect<
            int,
            idx_t,
            false,
            Comparator<int>,
            NumWarpQ,
            NumThreadQ,
            kWarps * kLanes>
            heap(kMaxDistance, -1, k);

    int warpId = hipThreadIdx_y;
    int laneId = hipThreadIdx_x;

    // Each warp handles a single query
    int laneK = laneId;
    idx_t warpQuery = idx_t(hipBlockIdx_x) * kWarps + warpId;
    bool kInBounds = laneK < vecs.getSize(1);
    bool queryInBounds = warpQuery < query.getSize(0);

    queryTile[warpId][laneId] =
            queryInBounds && kInBounds ? query[warpQuery][laneK] : 0;

    // Each warp loops through the entire chunk of vectors
    for (idx_t blockVec = 0; blockVec < vecs.getSize(0); blockVec += kLanes) {
        int threadDistance = 0;

        // kWarps warps are responsible for loading 32 vecs
#pragma unroll
        for (int i = 0; i < kLanes / kWarps; ++i) {
            int warpVec = i * kWarps + warpId;
            idx_t vec = blockVec + warpVec;
            bool vecInBounds = vec < vecs.getSize(0);

            vecTile[warpVec][laneId] =
                    vecInBounds && kInBounds ? vecs[vec][laneK] : 0;
        }

        __syncthreads();

        // Compare distances
#pragma unroll
        for (int i = 0; i < ReductionLimit; ++i) {
            threadDistance += __popc(queryTile[warpId][i] ^ vecTile[laneId][i]);
        }

        __syncthreads();

        // Lanes within a warp are different vec results against the same query
        // Only submit distances which represent real (query, vec) pairs
        bool valInBounds =
                queryInBounds && (blockVec + laneId < vecs.getSize(0));
        threadDistance = valInBounds ? threadDistance : kMaxDistance;
        idx_t id = valInBounds ? blockVec + laneId : idx_t(-1);

        heap.add(threadDistance, id);
    }

    heap.reduce();

    if (warpQuery < query.getSize(0)) {
        heap.writeOut(outK[warpQuery].data(), outV[warpQuery].data(), k);
    }
}

template <typename BinaryType>
void runBinaryDistanceAnySize(
        Tensor<BinaryType, 2, true>& vecs,
        Tensor<BinaryType, 2, true>& query,
        Tensor<int, 2, true>& outK,
        Tensor<idx_t, 2, true>& outV,
        int k,
        hipStream_t stream) {
    dim3 grid(utils::divUp(query.getSize(0), kWarps));
    dim3 block(kLanes, kWarps);

    if (k == 1) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<1, 1, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<32, 2, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<64, 3, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 128) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<128, 3, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 256) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<256, 4, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 512) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<512, 8, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 1024) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<1024, 8, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    }
#if GPU_MAX_SELECTION_K >= 2048
    else if (k <= 2048) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceAnySize<2048, 8, BinaryType>), grid, block, 0, stream, vecs, query, outK, outV, k);
    }
#endif
}

template <typename BinaryType, int ReductionLimit>
void runBinaryDistanceLimitSize(
        Tensor<BinaryType, 2, true>& vecs,
        Tensor<BinaryType, 2, true>& query,
        Tensor<int, 2, true>& outK,
        Tensor<idx_t, 2, true>& outV,
        int k,
        hipStream_t stream) {
    dim3 grid(utils::divUp(query.getSize(0), kWarps));
    dim3 block(kLanes, kWarps);

    if (k == 1) {
        // TODO: HADI
        std::cout << "K = " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<1, 1, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 32) {
        // TODO: HADI
        std::cout << "K = " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<32, 2, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 64) {
        // TODO: HADI
        std::cout << "K = " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<64, 3, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 128) {
        // TODO: HADI
        std::cout << "K = " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<128, 3, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 256) {
        // TODO: HADI
        std::cout << "K = " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<256, 4, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 512) {
        // TODO: HADI
        std::cout << "K = " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<512, 8, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    } else if (k <= 1024) {
        // TODO: HADI
        std::cout << "K = " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<1024, 8, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    }
#if GPU_MAX_SELECTION_K >= 2048
    else if (k <= 2048) {
        // TODO: HADI
        std::cout << "K <= 2048 " << k << std::endl;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(binaryDistanceLimitSize<2048, 8, BinaryType, ReductionLimit>), grid, block, 0, stream, vecs, query, outK, outV, k);
    }
#endif
}

void runBinaryDistance(
        Tensor<unsigned char, 2, true>& vecs,
        Tensor<unsigned char, 2, true>& query,
        Tensor<int, 2, true>& outK,
        Tensor<idx_t, 2, true>& outV,
        int k,
        hipStream_t stream) {
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
    FAISS_ASSERT(vecs.getSize(1) == query.getSize(1));

    FAISS_ASSERT(outK.getSize(1) == k);
    FAISS_ASSERT(outV.getSize(1) == k);

    // For the optimized uint32 kernel, we handle 32 * 8 = 256 max dims
    constexpr int kReductionLimit32 = 8;

    // For the optimized uint8 kernel, we handle 8 * 16 = 128 max dims
    constexpr int kReductionLimit8 = 16;

    // All other cases (large or small) go through the general kernel

    if (vecs.getSize(1) % sizeof(unsigned int) == 0 &&
        (vecs.getSize(1) / sizeof(unsigned int)) <= kReductionLimit32) {
        auto vecs32 = vecs.castResize<unsigned int>();
        auto query32 = query.castResize<unsigned int>();

        // Optimize for vectors with dimensions a multiple of 32 that are less
        // than 32 * kReductionLimit (256) dimensions in size
        // TODO: HADI
        std::cout << "runBinaryDistanceLimitedSize 32" << std::endl;
        runBinaryDistanceLimitSize<unsigned int, kReductionLimit32>(
                vecs32, query32, outK, outV, k, stream);

    } else if (vecs.getSize(1) <= kReductionLimit8) {
        // Optimize for vectors with dimensions a multiple of 32 that are less
        // than 32 * kReductionLimit (256) dimensions in size
        // TODO: HADI
        std::cout << "runBinaryDistanceLimitedSize 8" << std::endl;
        runBinaryDistanceLimitSize<unsigned char, kReductionLimit8>(
                vecs, query, outK, outV, k, stream);
    } else {
        // Arbitrary size kernel
        runBinaryDistanceAnySize<unsigned char>(
                vecs, query, outK, outV, k, stream);
    }
}

} // namespace gpu
} // namespace faiss
