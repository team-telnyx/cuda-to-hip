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
#include <faiss/gpu/impl/L2Norm.h>
#include <faiss/gpu/utils/ConversionOperators.h>
#include <faiss/gpu/utils/DeviceDefs.h>
#include <faiss/gpu/utils/MathOperators.h>
#include <faiss/gpu/utils/PtxUtils.h>
#include <faiss/gpu/utils/Reductions.h>

#include <algorithm>

namespace faiss {
namespace gpu {

// Input: (batch x dim)
// Output: (batch norm)
// Done under the presumption that the dimension size is not too large
// (<10k or so), since there wouldn't be enough parallelism applying a
// single block to the problem. Also that each vector is large enough
// (>64), since a single block works on multiple rows' norms at the
// same time.
// T: the type we are doing the math in (e.g., float, half)
// TVec: the potentially vectorized type we are loading in (e.g.,
// float4, half2)
template <
        typename T,
        typename TVec,
        int RowTileSize,
        bool NormLoop,
        bool NormSquared>
__global__ void l2NormRowMajor(
        Tensor<TVec, 2, true> input,
        Tensor<float, 1, true> output) {
    extern __shared__ char smemByte[]; // #warps * RowTileSize elements
    float* smem = (float*)smemByte;

    // these are fine to be int (just based on block dimensions)
    int numWarps = utils::divUp(hipBlockDim_x, kWarpSize);
    int laneId = hipThreadIdx_x % kWarpSize;
    int warpId = hipThreadIdx_x / kWarpSize;

    bool lastRowTile = (hipBlockIdx_x == (hipGridDim_x - 1));
    idx_t rowStart = idx_t(hipBlockIdx_x) * RowTileSize;
    // accumulate in f32
    float rowNorm[RowTileSize];

    if (lastRowTile) {
        // We are handling the very end of the input matrix rows
        for (idx_t row = 0; row < input.getSize(0) - rowStart; ++row) {
            if (NormLoop) {
                rowNorm[0] = 0;

                for (idx_t col = hipThreadIdx_x; col < input.getSize(1);
                     col += hipBlockDim_x) {
                    TVec val = input[rowStart + row][col];
                    val = Math<TVec>::mul(val, val);
                    rowNorm[0] = rowNorm[0] + Math<TVec>::reduceAdd(val);
                }
            } else {
                TVec val = input[rowStart + row][hipThreadIdx_x];
                val = Math<TVec>::mul(val, val);
                rowNorm[0] = Math<TVec>::reduceAdd(val);
            }

            rowNorm[0] = warpReduceAllSum(rowNorm[0]);
            if (laneId == 0) {
                smem[row * numWarps + warpId] = rowNorm[0];
            }
        }
    } else {
        // We are guaranteed that all RowTileSize rows are available in
        // [rowStart, rowStart + RowTileSize)

        if (NormLoop) {
            // A single block of threads is not big enough to span each
            // vector
            TVec tmp[RowTileSize];

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                rowNorm[row] = 0;
            }

            for (idx_t col = hipThreadIdx_x; col < input.getSize(1);
                 col += hipBlockDim_x) {
#pragma unroll
                for (int row = 0; row < RowTileSize; ++row) {
                    tmp[row] = input[rowStart + row][col];
                }

#pragma unroll
                for (int row = 0; row < RowTileSize; ++row) {
                    tmp[row] = Math<TVec>::mul(tmp[row], tmp[row]);
                }

#pragma unroll
                for (int row = 0; row < RowTileSize; ++row) {
                    rowNorm[row] =
                            rowNorm[row] + Math<TVec>::reduceAdd(tmp[row]);
                }
            }
        } else {
            TVec tmp[RowTileSize];

            // A block of threads is the exact size of the vector
#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                tmp[row] = input[rowStart + row][hipThreadIdx_x];
            }

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                tmp[row] = Math<TVec>::mul(tmp[row], tmp[row]);
            }

#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                rowNorm[row] = Math<TVec>::reduceAdd(tmp[row]);
            }
        }

        // Sum up all parts in each warp
#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
            rowNorm[row] = warpReduceAllSum(rowNorm[row]);
        }

        if (laneId == 0) {
#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                smem[row * numWarps + warpId] = rowNorm[row];
            }
        }
    }

    __syncthreads();

    // Sum across warps
    if (warpId == 0) {
#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
            rowNorm[row] =
                    laneId < numWarps ? smem[row * numWarps + laneId] : 0;
        }

#pragma unroll
        for (int row = 0; row < RowTileSize; ++row) {
            rowNorm[row] = warpReduceAllSum(rowNorm[row]);
        }

        // Write out answer
        if (laneId == 0) {
#pragma unroll
            for (int row = 0; row < RowTileSize; ++row) {
                int outCol = rowStart + row;

                if (lastRowTile) {
                    if (outCol < output.getSize(0)) {
                        output[outCol] = NormSquared
                                ? ConvertTo<float>::to(rowNorm[row])
                                : sqrtf(ConvertTo<float>::to(rowNorm[row]));
                    }
                } else {
                    output[outCol] = NormSquared
                            ? ConvertTo<float>::to(rowNorm[row])
                            : sqrtf(ConvertTo<float>::to(rowNorm[row]));
                }
            }
        }
    }
}

// Input: (dim x batch)
// Output: (batch norm)
// Handles the case where `input` is column major. A single thread calculates
// the norm of each vector instead of a block-wide reduction.
template <typename T, bool NormSquared>
__global__ void l2NormColMajor(
        Tensor<T, 2, true> input,
        Tensor<float, 1, true> output) {
    // grid-stride loop to handle all batch elements
    for (idx_t batch = idx_t(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;
         batch < input.getSize(1);
         batch += hipGridDim_x * hipBlockDim_x) {
        float sum = 0;

        // This is still a coalesced load from the memory
        for (idx_t dim = 0; dim < input.getSize(0); ++dim) {
            // Just do the math in float32, even if the input is float16
            float v = ConvertTo<float>::to(input[dim][batch]);
            sum += v * v;
        }

        if (!NormSquared) {
            sum = sqrtf(sum);
        }

        output[batch] = ConvertTo<float>::to(sum);
    }
}

template <typename T, typename TVec>
void runL2Norm(
        Tensor<T, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        hipStream_t stream) {
    idx_t maxThreads = (idx_t)getMaxThreadsCurrentDevice();
    constexpr int rowTileSize = 8;

#define RUN_L2_ROW_MAJOR(TYPE_T, TYPE_TVEC, INPUT)                           \
    do {                                                                     \
        if (normLoop) {                                                      \
            if (normSquared) {                                               \
                hipLaunchKernelGGL(HIP_KERNEL_NAME(l2NormRowMajor<TYPE_T, TYPE_TVEC, rowTileSize, true, true>), \
                             grid, block, smem, stream, INPUT, output);      \
            } else {                                                         \
                hipLaunchKernelGGL(HIP_KERNEL_NAME(l2NormRowMajor<TYPE_T, TYPE_TVEC, rowTileSize, true, false>), \
                             grid, block, smem, stream, INPUT, output);      \
            }                                                                \
        } else {                                                             \
            if (normSquared) {                                               \
                hipLaunchKernelGGL(HIP_KERNEL_NAME(l2NormRowMajor<TYPE_T, TYPE_TVEC, rowTileSize, false, true>),  \
                             grid, block, smem, stream, INPUT, output);      \
            } else {                                                         \
                hipLaunchKernelGGL(HIP_KERNEL_NAME(l2NormRowMajor<TYPE_T, TYPE_TVEC, rowTileSize, false, false>), \
                             grid, block, smem, stream, INPUT, output);      \
            }                                                                \
        }                                                                    \
    } while (0)

    if (inputRowMajor) {
        //
        // Row-major kernel
        ///

        if (input.template canCastResize<TVec>()) {
            // Can load using the vectorized type
            auto inputV = input.template castResize<TVec>();

            auto dim = inputV.getSize(1);
            bool normLoop = dim > maxThreads;
            auto numThreads = std::min(dim, maxThreads);

            auto grid = dim3(utils::divUp(inputV.getSize(0), rowTileSize));
            auto block = dim3(numThreads);

            auto smem = sizeof(float) * rowTileSize *
                    utils::divUp(numThreads, kWarpSize);

            RUN_L2_ROW_MAJOR(T, TVec, inputV);
        } else {
            // Can't load using the vectorized type

            auto dim = input.getSize(1);
            bool normLoop = dim > maxThreads;
            auto numThreads = std::min(dim, maxThreads);

            auto grid = dim3(utils::divUp(input.getSize(0), rowTileSize));
            auto block = dim3(numThreads);

            auto smem = sizeof(float) * rowTileSize *
                    utils::divUp(numThreads, kWarpSize);

            RUN_L2_ROW_MAJOR(T, T, input);
        }
    } else {
        //
        // Column-major kernel
        //

        // Just use a fixed-sized block, since the kernel threads are fully
        // independent
        auto block = 128;

        // Cap the grid size at 2^16 since there is a grid-stride loop to handle
        // processing everything
        auto grid = (int)std::min(
                utils::divUp(input.getSize(1), (idx_t)block), (idx_t)65536);

        if (normSquared) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(l2NormColMajor<T, true>), dim3(grid), dim3(block), 0, stream, input, output);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(l2NormColMajor<T, false>), dim3(grid), dim3(block), 0, stream, input, output);
        }
    }

#undef RUN_L2

    HIP_TEST_ERROR();
}

void runL2Norm(
        Tensor<float, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        hipStream_t stream) {
    runL2Norm<float, float4>(input, inputRowMajor, output, normSquared, stream);
}

void runL2Norm(
        Tensor<half, 2, true>& input,
        bool inputRowMajor,
        Tensor<float, 1, true>& output,
        bool normSquared,
        hipStream_t stream) {
    runL2Norm<half, half2>(input, inputRowMajor, output, normSquared, stream);
}

} // namespace gpu
} // namespace faiss
