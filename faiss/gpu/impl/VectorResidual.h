/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/utils/Tensor.h>
#include <hip/hip_fp16.h>
namespace faiss {
namespace gpu {

// Calculates residual v_i - c_j for all v_i in vecs where j = vecToCentroid[i]
void runCalcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
        hipStream_t stream);

void runCalcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<half, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
        hipStream_t stream);

// Gather vectors
void runReconstruct(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream);

void runReconstruct(
        Tensor<idx_t, 1, true>& listIds,
        Tensor<half, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream);

void runReconstruct(
        idx_t start,
        idx_t num,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream);

void runReconstruct(
        idx_t start,
        idx_t num,
        Tensor<half, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream);

} // namespace gpu
} // namespace faiss
