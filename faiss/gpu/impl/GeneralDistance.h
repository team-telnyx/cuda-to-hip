#include "hip/hip_runtime.h"
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/MetricType.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/gpu/impl/DistanceUtils.h>
#include <faiss/gpu/utils/BlockSelectKernel.h>
#include <faiss/gpu/utils/ConversionOperators.h>
#include <faiss/gpu/utils/DeviceDefs.h>
#include <faiss/gpu/utils/DeviceTensor.h>
#include <faiss/gpu/utils/Select.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <iostream>
#include <memory>

//
// Kernels for non-L2 / inner product distances
//

namespace faiss {
namespace gpu {

// Reduction tree operator
template <typename DistanceOp, int N>
struct ReduceDistanceOp {
    __device__ static DistanceOp reduce(DistanceOp ops[N]) {
        DistanceOp vals[N / 2];
#pragma unroll
        for (int i = 0; i < N / 2; ++i) {
            vals[i] = ops[i * 2];
            vals[i].combine(ops[i * 2 + 1]);
        }

        return ReduceDistanceOp<DistanceOp, N / 2>::reduce(vals);
    }
};

template <typename DistanceOp>
struct ReduceDistanceOp<DistanceOp, 1> {
    __device__ static DistanceOp reduce(DistanceOp ops[1]) {
        return ops[0];
    }
};

// Implements a pairwise reduction tree
template <typename T, int Unroll, int DimMultiple, typename DistanceOp>
inline __device__ DistanceOp
reduce(const DistanceOp& in,
       const T queryTile[kWarpSize][DimMultiple * kWarpSize + 1],
       const T vecTile[kWarpSize][DimMultiple * kWarpSize + 1]) {
    DistanceOp accs[Unroll];
#pragma unroll
    for (int i = 0; i < Unroll; ++i) {
        accs[i] = in.zero();
    }

    auto vecTileBase = vecTile[hipThreadIdx_x];
    auto queryTileBase = queryTile[hipThreadIdx_y];

#pragma unroll
    for (int i = 0; i < Unroll; ++i) {
#pragma unroll
        for (int j = 0; j < (kWarpSize * DimMultiple / Unroll); ++j) {
            int idx = i * (kWarpSize * DimMultiple / Unroll) + j;
            accs[i].handle(
                    ConvertTo<float>::to(queryTileBase[idx]),
                    ConvertTo<float>::to(vecTileBase[idx]));
        }
    }

    return ReduceDistanceOp<DistanceOp, Unroll>::reduce(accs);
}

// Our general distance matrix "multiplication" kernel
template <typename T, typename DistanceOp, bool InnerContig>
__launch_bounds__(kWarpSize* kWarpSize) __global__ void generalDistance(
        Tensor<T, 2, InnerContig> query, // m x k
        Tensor<T, 2, InnerContig> vec,   // n x k
        DistanceOp op,
        Tensor<float, 2, true> out) { // m x n
    constexpr int kDimMultiple = 1;

    __shared__ T queryTile[kWarpSize][kWarpSize * kDimMultiple + 1];
    __shared__ T vecTile[kWarpSize][kWarpSize * kDimMultiple + 1];

    // block y -> query
    // block x -> vector

    idx_t queryBlock = idx_t(hipBlockIdx_y) * kWarpSize;
    idx_t queryThread = queryBlock + hipThreadIdx_y;

    idx_t vecBlock = idx_t(hipBlockIdx_x) * kWarpSize;
    idx_t vecThreadLoad = vecBlock + hipThreadIdx_y;
    idx_t vecThreadSave = vecBlock + hipThreadIdx_x;

    DistanceOp acc = op.zero();

    auto queryTileBase = queryTile[hipThreadIdx_y];
    auto vecTileBase = vecTile[hipThreadIdx_y];

    auto queryBase = query[queryThread];
    auto vecBase = vec[vecThreadLoad];

    if ((hipBlockIdx_x != (hipGridDim_x - 1)) && (hipBlockIdx_y != (hipGridDim_y - 1))) {
        //
        // Interior tile
        //
        idx_t limit =
                utils::roundDown(query.getSize(1), kWarpSize * kDimMultiple);

        for (idx_t k = hipThreadIdx_x; k < limit; k += kWarpSize * kDimMultiple) {
            // Load query tile
#pragma unroll
            for (int i = 0; i < kDimMultiple; ++i) {
                queryTileBase[hipThreadIdx_x + i * kWarpSize] =
                        queryBase[k + i * kWarpSize];
                vecTileBase[hipThreadIdx_x + i * kWarpSize] =
                        vecBase[k + i * kWarpSize];
            }

            __syncthreads();

            // thread (y, x) does (query y, vec x)
            acc.combine(reduce<T, 8, kDimMultiple, DistanceOp>(
                    op, queryTile, vecTile));

            __syncthreads();
        }

        // Handle remainder
        if (limit < query.getSize(1)) {
#pragma unroll
            for (int i = 0; i < kDimMultiple; ++i) {
                idx_t k = limit + hipThreadIdx_x + i * kWarpSize;
                bool kInBounds = k < query.getSize(1);

                queryTileBase[hipThreadIdx_x + i * kWarpSize] =
                        kInBounds ? queryBase[k] : ConvertTo<T>::to(0);

                vecTileBase[hipThreadIdx_x + i * kWarpSize] =
                        kInBounds ? vecBase[k] : ConvertTo<T>::to(0);
            }

            __syncthreads();

            idx_t remainder = query.getSize(1) - limit;

            // thread (y, x) does (query y, vec x)
#pragma unroll
            for (idx_t i = 0; i < remainder; ++i) {
                acc.handle(
                        ConvertTo<float>::to(queryTileBase[i]),
                        ConvertTo<float>::to(vecTile[hipThreadIdx_x][i]));
            }
        }

        // Write out results
        out[queryThread][vecThreadSave] = acc.reduce();
    } else {
        //
        // Otherwise, we're an exterior tile
        //

        bool queryThreadInBounds = queryThread < query.getSize(0);
        bool vecThreadInBoundsLoad = vecThreadLoad < vec.getSize(0);
        bool vecThreadInBoundsSave = vecThreadSave < vec.getSize(0);
        idx_t limit = utils::roundDown(query.getSize(1), kWarpSize);

        for (idx_t k = hipThreadIdx_x; k < limit; k += kWarpSize) {
            // Load query tile
            queryTileBase[hipThreadIdx_x] =
                    queryThreadInBounds ? queryBase[k] : ConvertTo<T>::to(0);

            vecTileBase[hipThreadIdx_x] =
                    vecThreadInBoundsLoad ? vecBase[k] : ConvertTo<T>::to(0);

            __syncthreads();

            // thread (y, x) does (query y, vec x)
#pragma unroll
            for (int i = 0; i < kWarpSize; ++i) {
                acc.handle(
                        ConvertTo<float>::to(queryTileBase[i]),
                        ConvertTo<float>::to(vecTile[hipThreadIdx_x][i]));
            }

            __syncthreads();
        }

        // Handle remainder
        if (limit < query.getSize(1)) {
            idx_t k = limit + hipThreadIdx_x;
            bool kInBounds = k < query.getSize(1);

            // Load query tile
            queryTileBase[hipThreadIdx_x] = queryThreadInBounds && kInBounds
                    ? queryBase[k]
                    : ConvertTo<T>::to(0);

            vecTileBase[hipThreadIdx_x] = vecThreadInBoundsLoad && kInBounds
                    ? vecBase[k]
                    : ConvertTo<T>::to(0);

            __syncthreads();

            idx_t remainder = query.getSize(1) - limit;

            // thread (y, x) does (query y, vec x)
            for (int i = 0; i < remainder; ++i) {
                acc.handle(
                        ConvertTo<float>::to(queryTileBase[i]),
                        ConvertTo<float>::to(vecTile[hipThreadIdx_x][i]));
            }
        }

        // Write out results
        if (queryThreadInBounds && vecThreadInBoundsSave) {
            out[queryThread][vecThreadSave] = acc.reduce();
        }
    }
}

template <typename T, typename DistanceOp, bool InnerContig>
void runGeneralDistanceKernel(
        Tensor<T, 2, InnerContig>& vecs,
        Tensor<T, 2, InnerContig>& query,
        Tensor<float, 2, true>& out,
        const DistanceOp& op,
        hipStream_t stream) {
    FAISS_ASSERT(vecs.getSize(1) == query.getSize(1));
    FAISS_ASSERT(out.getSize(0) == query.getSize(0));
    FAISS_ASSERT(out.getSize(1) == vecs.getSize(0));

    dim3 grid(
            utils::divUp(vecs.getSize(0), kWarpSize),
            utils::divUp(query.getSize(0), kWarpSize));
    FAISS_ASSERT(grid.y <= getMaxGridCurrentDevice().y);
    dim3 block(kWarpSize, kWarpSize);

    hipLaunchKernelGGL(generalDistance, grid, block, 0, stream, query, vecs, op, out);
}

template <typename T, typename DistanceOp, bool InnerContig>
void runGeneralDistance(
        GpuResources* res,
        hipStream_t stream,
        Tensor<T, 2, InnerContig>& centroids,
        Tensor<T, 2, InnerContig>& queries,
        int k,
        const DistanceOp& op,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices) {
    // The # of centroids in `centroids` based on memory layout
    auto numCentroids = centroids.getSize(0);

    // The # of queries in `queries` based on memory layout
    auto numQueries = queries.getSize(0);

    // The dimensions of the vectors to consider
    auto dim = queries.getSize(1);
    FAISS_ASSERT(
            (numQueries == 0 || numCentroids == 0) ||
            dim == centroids.getSize(1));

    FAISS_ASSERT(outDistances.getSize(0) == numQueries);
    FAISS_ASSERT(outIndices.getSize(0) == numQueries);
    FAISS_ASSERT(outDistances.getSize(1) == k);
    FAISS_ASSERT(outIndices.getSize(1) == k);

    // If we're quering against a 0 sized set, just return empty results
    if (centroids.numElements() == 0) {
        thrust::fill(
                thrust::hip::par.on(stream),
                outDistances.data(),
                outDistances.end(),
                Limits<float>::getMax());

        thrust::fill(
                thrust::hip::par.on(stream),
                outIndices.data(),
                outIndices.end(),
                -1);

        return;
    }

    // By default, aim to use up to 512 MB of memory for the processing, with
    // both number of queries and number of centroids being at least 512.
    idx_t tileRows = 0;
    idx_t tileCols = 0;
    chooseTileSize(
            numQueries,
            numCentroids,
            dim,
            sizeof(T),
            res->getTempMemoryAvailableCurrentDevice(),
            tileRows,
            tileCols);

    auto numColTiles = utils::divUp(numCentroids, tileCols);

    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation

    // Temporary output memory space we'll use
    DeviceTensor<float, 2, true> distanceBuf1(
            res, makeTempAlloc(AllocType::Other, stream), {tileRows, tileCols});
    DeviceTensor<float, 2, true> distanceBuf2(
            res, makeTempAlloc(AllocType::Other, stream), {tileRows, tileCols});
    DeviceTensor<float, 2, true>* distanceBufs[2] = {
            &distanceBuf1, &distanceBuf2};

    DeviceTensor<float, 2, true> outDistanceBuf1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<float, 2, true> outDistanceBuf2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<float, 2, true>* outDistanceBufs[2] = {
            &outDistanceBuf1, &outDistanceBuf2};

    DeviceTensor<idx_t, 2, true> outIndexBuf1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<idx_t, 2, true> outIndexBuf2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {tileRows, numColTiles * k});
    DeviceTensor<idx_t, 2, true>* outIndexBufs[2] = {
            &outIndexBuf1, &outIndexBuf2};

    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;
    bool interrupt = false;

    // Tile over the input queries
    for (idx_t i = 0; i < numQueries; i += tileRows) {
        if (interrupt || InterruptCallback::is_interrupted()) {
            interrupt = true;
            break;
        }

        auto curQuerySize = std::min(tileRows, numQueries - i);

        auto outDistanceView = outDistances.narrow(0, i, curQuerySize);
        auto outIndexView = outIndices.narrow(0, i, curQuerySize);

        auto queryView = queries.narrow(0, i, curQuerySize);

        auto outDistanceBufRowView =
                outDistanceBufs[curStream]->narrow(0, 0, curQuerySize);
        auto outIndexBufRowView =
                outIndexBufs[curStream]->narrow(0, 0, curQuerySize);

        // Tile over the centroids
        for (idx_t j = 0; j < numCentroids; j += tileCols) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }

            auto curCentroidSize = std::min(tileCols, numCentroids - j);
            auto curColTile = j / tileCols;

            auto centroidsView =
                    sliceCentroids(centroids, true, j, curCentroidSize);

            auto distanceBufView = distanceBufs[curStream]
                                           ->narrow(0, 0, curQuerySize)
                                           .narrow(1, 0, curCentroidSize);

            auto outDistanceBufColView =
                    outDistanceBufRowView.narrow(1, k * curColTile, k);
            auto outIndexBufColView =
                    outIndexBufRowView.narrow(1, k * curColTile, k);

            runGeneralDistanceKernel(
                    centroidsView,
                    queryView,
                    distanceBufView,
                    op,
                    streams[curStream]);

            // For IP, just k-select the output for this tile
            if (tileCols == numCentroids) {
                // Write into the final output
                runBlockSelect(
                        distanceBufView,
                        outDistanceView,
                        outIndexView,
                        DistanceOp::kDirection,
                        k,
                        streams[curStream]);
            } else {
                // Write into the intermediate output
                runBlockSelect(
                        distanceBufView,
                        outDistanceBufColView,
                        outIndexBufColView,
                        DistanceOp::kDirection,
                        k,
                        streams[curStream]);
            }
        }

        // As we're finished with processing a full set of centroids, perform
        // the final k-selection
        if (tileCols != numCentroids) {
            // The indices are tile-relative; for each tile of k, we need to add
            // tileCols to the index
            runIncrementIndex(
                    outIndexBufRowView, k, tileCols, streams[curStream]);

            runBlockSelectPair(
                    outDistanceBufRowView,
                    outIndexBufRowView,
                    outDistanceView,
                    outIndexView,
                    DistanceOp::kDirection,
                    k,
                    streams[curStream]);
        }

        curStream = (curStream + 1) % 2;
    }

    // Have the desired ordering stream wait on the multi-stream
    streamWait({stream}, streams);

    if (interrupt) {
        FAISS_THROW_MSG("interrupted");
    }

    HIP_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss