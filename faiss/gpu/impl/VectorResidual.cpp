#include "hip/hip_runtime.h"
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <hip/hip_math_constants.h> // in CUDA SDK, for CUDART_NAN_F
#include <faiss/gpu/impl/VectorResidual.h>
#include <faiss/gpu/utils/ConversionOperators.h>
#include <faiss/gpu/utils/Tensor.h>

#include <algorithm>

namespace faiss {
namespace gpu {

template <typename CentroidT, bool LargeDim>
__global__ void calcResidual(
        Tensor<float, 2, true> vecs,
        Tensor<CentroidT, 2, true> centroids,
        Tensor<idx_t, 1, true> vecToCentroid,
        Tensor<float, 2, true> residuals) {
    auto vec = vecs[hipBlockIdx_x];
    auto residual = residuals[hipBlockIdx_x];
    auto centroidId = vecToCentroid[hipBlockIdx_x];
    float CUDART_NAN_F = std::nanf("");
    // Vector could be invalid (containing NaNs), so -1 was the
    // classified centroid
    if (centroidId == -1) {
        if (LargeDim) {
            for (idx_t i = hipThreadIdx_x; i < vecs.getSize(1); i += hipBlockDim_x) {
                residual[i] = CUDART_NAN_F;
            }
        } else {
            residual[hipThreadIdx_x] = CUDART_NAN_F;
        }

        return;
    }

    auto centroid = centroids[centroidId];

    if (LargeDim) {
        for (idx_t i = hipThreadIdx_x; i < vecs.getSize(1); i += hipBlockDim_x) {
            residual[i] = vec[i] - ConvertTo<float>::to(centroid[i]);
        }
    } else {
        residual[hipThreadIdx_x] =
                vec[hipThreadIdx_x] - ConvertTo<float>::to(centroid[hipThreadIdx_x]);
    }
}

template <typename CentroidT>
void calcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<CentroidT, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
        hipStream_t stream) {
    FAISS_ASSERT(vecs.getSize(1) == centroids.getSize(1));
    FAISS_ASSERT(vecs.getSize(1) == residuals.getSize(1));
    FAISS_ASSERT(vecs.getSize(0) == vecToCentroid.getSize(0));
    FAISS_ASSERT(vecs.getSize(0) == residuals.getSize(0));

    dim3 grid(vecs.getSize(0));

    idx_t maxThreads = getMaxThreadsCurrentDevice();
    bool largeDim = vecs.getSize(1) > maxThreads;
    dim3 block(std::min(vecs.getSize(1), maxThreads));

    if (largeDim) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(calcResidual<CentroidT, true>), grid, block, 0, stream, 
                vecs, centroids, vecToCentroid, residuals);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(calcResidual<CentroidT, false>), grid, block, 0, stream, 
                vecs, centroids, vecToCentroid, residuals);
    }

    HIP_TEST_ERROR();
}

void runCalcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
        hipStream_t stream) {
    calcResidual<float>(vecs, centroids, vecToCentroid, residuals, stream);
}

void runCalcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<half, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
        hipStream_t stream) {
    calcResidual<half>(vecs, centroids, vecToCentroid, residuals, stream);
}

template <typename T>
__global__ void gatherReconstructByIds(
        Tensor<idx_t, 1, true> ids,
        Tensor<T, 2, true> vecs,
        Tensor<float, 2, true> out) {
    auto id = ids[hipBlockIdx_x];
    auto vec = vecs[id];
    auto outVec = out[hipBlockIdx_x];

    Convert<T, float> conv;

    for (idx_t i = hipThreadIdx_x; i < vecs.getSize(1); i += hipBlockDim_x) {
        outVec[i] = id == idx_t(-1) ? 0.0f : conv(vec[i]);
    }
}

template <typename T>
__global__ void gatherReconstructByRange(
        idx_t start,
        idx_t num,
        Tensor<T, 2, true> vecs,
        Tensor<float, 2, true> out) {
    auto id = start + hipBlockIdx_x;
    auto vec = vecs[id];
    auto outVec = out[hipBlockIdx_x];

    Convert<T, float> conv;

    for (idx_t i = hipThreadIdx_x; i < vecs.getSize(1); i += hipBlockDim_x) {
        outVec[i] = id == idx_t(-1) ? 0.0f : conv(vec[i]);
    }
}

template <typename T>
void gatherReconstructByIds(
        Tensor<idx_t, 1, true>& ids,
        Tensor<T, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    FAISS_ASSERT(ids.getSize(0) == out.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == out.getSize(1));

    dim3 grid(ids.getSize(0));

    idx_t maxThreads = getMaxThreadsCurrentDevice();
    dim3 block(std::min(vecs.getSize(1), maxThreads));

    hipLaunchKernelGGL(gatherReconstructByIds<T>, grid, block, 0, stream, ids, vecs, out);

    HIP_TEST_ERROR();
}

template <typename T>
void gatherReconstructByRange(
        idx_t start,
        idx_t num,
        Tensor<T, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    FAISS_ASSERT(num > 0);
    FAISS_ASSERT(num == out.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == out.getSize(1));
    FAISS_ASSERT(start + num <= vecs.getSize(0));

    dim3 grid(num);

    idx_t maxThreads = getMaxThreadsCurrentDevice();
    dim3 block(std::min(vecs.getSize(1), maxThreads));

    hipLaunchKernelGGL(gatherReconstructByRange<T>, grid, block, 0, stream, start, num, vecs, out);

    HIP_TEST_ERROR();
}

void runReconstruct(
        Tensor<idx_t, 1, true>& ids,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    gatherReconstructByIds<float>(ids, vecs, out, stream);
}

void runReconstruct(
        Tensor<idx_t, 1, true>& ids,
        Tensor<half, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    gatherReconstructByIds<half>(ids, vecs, out, stream);
}

void runReconstruct(
        idx_t start,
        idx_t num,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    gatherReconstructByRange<float>(start, num, vecs, out, stream);
}

void runReconstruct(
        idx_t start,
        idx_t num,
        Tensor<half, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    gatherReconstructByRange<half>(start, num, vecs, out, stream);
}

} // namespace gpu
} // namespace faiss
