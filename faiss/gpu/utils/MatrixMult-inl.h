/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hipblas/hipblas.h>
#include <faiss/gpu/utils/DeviceTensor.h>
#include <faiss/gpu/utils/Float16.h>
#include <faiss/gpu/utils/HostTensor.h>
#include <faiss/gpu/utils/Tensor.h>
#include <limits>
#include <iostream>

namespace faiss {
namespace gpu {

template <typename T>
struct GetHipType;

template <>
struct GetHipType<float> {
    static constexpr hipblasDatatype_t Type = HIPBLAS_R_32F;
};

template <>
struct GetHipType<half> {
    static constexpr hipblasDatatype_t Type = HIPBLAS_R_16F;
};

template <typename AT, typename BT>
hipblasStatus_t rawGemm(
        hipblasHandle_t handle,
        hipblasOperation_t transa,
        hipblasOperation_t transb,
        int m,
        int n,
        int k,
        const float fAlpha,
        const void* A,
        int lda,
        const void* B,
        int ldb,
        const float fBeta,
        float* C,
        int ldc) {
    auto cAT = GetHipType<AT>::Type;
    auto cBT = GetHipType<BT>::Type;
// std::cout << "HADI: Paased 0" << std::endl;
    // Always accumulate in f32
    return hipblasGemmEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            &fAlpha,
            A,
            cAT,
            lda,
            B,
            cBT,
            ldb,
            &fBeta,
            C,
            HIPBLAS_R_32F,
            ldc,
            HIPBLAS_R_32F,
            HIPBLAS_GEMM_DEFAULT);
}

template <typename AT, typename BT>
hipblasStatus_t rawBatchGemm(
        hipblasHandle_t handle,
        hipblasOperation_t transa,
        hipblasOperation_t transb,
        int m,
        int n,
        int k,
        const float fAlpha,
        const void* A,
        int lda,
        long long int strideA,
        const void* B,
        int ldb,
        long long int strideB,
        const float fBeta,
        float* C,
        int ldc,
        long long int strideC,
        int batchCount) {
    auto cAT = GetHipType<AT>::Type;
    auto cBT = GetHipType<BT>::Type;

    // Always accumulate in f32
    return hipblasGemmStridedBatchedEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            &fAlpha,
            A,
            cAT,
            lda,
            strideA,
            B,
            cBT,
            ldb,
            strideB,
            &fBeta,
            C,
            HIPBLAS_R_32F,
            ldc,
            strideC,
            batchCount,
            HIPBLAS_R_32F,
            HIPBLAS_GEMM_DEFAULT);
}

template <typename AT, typename BT>
void runMatrixMult(
        Tensor<float, 2, true>& c,
        bool transC,
        Tensor<AT, 2, true>& a,
        bool transA,
        Tensor<BT, 2, true>& b,
        bool transB,
        float alpha,
        float beta,
        hipblasHandle_t handle,
        hipStream_t stream) {
    // All sizes must be within int bounds
    FAISS_ASSERT(c.getSize(0) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(c.getSize(1) <= std::numeric_limits<int>::max());

    FAISS_ASSERT(a.getSize(0) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(a.getSize(1) <= std::numeric_limits<int>::max());

    FAISS_ASSERT(a.getSize(0) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(a.getSize(1) <= std::numeric_limits<int>::max());

// std::cout << "HADI: Paased 1" << std::endl;
    hipblasSetStream(handle, stream);

// std::cout << "HADI: Paased 2" << std::endl;
    // Check that we have (m x k) * (k x n) = (m x n)
    // using the input row-major layout
    int aM = transA ? a.getSize(1) : a.getSize(0);
    int aK = transA ? a.getSize(0) : a.getSize(1);

    int bK = transB ? b.getSize(1) : b.getSize(0);
    int bN = transB ? b.getSize(0) : b.getSize(1);

    int cM = transC ? c.getSize(1) : c.getSize(0);
    int cN = transC ? c.getSize(0) : c.getSize(1);

    FAISS_ASSERT(aM == cM);
    FAISS_ASSERT(aK == bK);
    FAISS_ASSERT(bN == cN);

    FAISS_ASSERT(a.getStride(1) == 1);
    FAISS_ASSERT(b.getStride(1) == 1);
    FAISS_ASSERT(c.getStride(1) == 1);

    // Now, we have to represent the matrix multiplication in
    // column-major layout
    float* pC = c.data();

    int m = c.getSize(1); // stride 1 size
    int n = c.getSize(0); // other size
    int k = transA ? a.getSize(0) : a.getSize(1);

    int lda = transC ? a.getStride(0) : b.getStride(0);
    int ldb = transC ? b.getStride(0) : a.getStride(0);
    int ldc = c.getStride(0);

    auto gemmTrA = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    auto gemmTrB = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    if (transC) {
        gemmTrA = transA ? HIPBLAS_OP_N : HIPBLAS_OP_T;
        gemmTrB = transB ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    }

    hipblasStatus_t err;

// std::cout << "HADI: Paased 3" << std::endl;
    if (transC) {
        err = rawGemm<AT, BT>(
                handle,
                gemmTrA,
                gemmTrB,
                m,
                n,
                k,
                alpha,
                a.data(),
                lda,
                b.data(),
                ldb,
                beta,
                pC,
                ldc);
    } else {
        err = rawGemm<AT, BT>(
                handle,
                gemmTrA,
                gemmTrB,
                m,
                n,
                k,
                alpha,
                b.data(),
                lda,
                a.data(),
                ldb,
                beta,
                pC,
                ldc);
    }
// std::cout << "HADI: Paased 4" << std::endl;
    FAISS_ASSERT_FMT(
            err == HIPBLAS_STATUS_SUCCESS,
            "hipblas failed (%d): "
            "(%ld, %ld)%s x (%ld, %ld)%s = (%ld, %ld)%s "
            "gemm params m %d n %d k %d trA %s trB %s lda %d ldb %d ldc %d",
            (int)err,
            a.getSize(0),
            a.getSize(1),
            transA ? "'" : "",
            b.getSize(0),
            b.getSize(1),
            transB ? "'" : "",
            c.getSize(0),
            c.getSize(1),
            transC ? "'" : "",
            m,
            n,
            k,
            gemmTrA == HIPBLAS_OP_T ? "T" : "N",
            gemmTrB == HIPBLAS_OP_T ? "T" : "N",
            lda,
            ldb,
            ldc);
            
// std::cout << "HADI: Paased 5" << std::endl;
    HIP_TEST_ERROR();
}

template <typename AT, typename BT>
void runBatchMatrixMult(
        Tensor<float, 3, true>& c,
        bool transC,
        Tensor<AT, 3, true>& a,
        bool transA,
        Tensor<BT, 3, true>& b,
        bool transB,
        float alpha,
        float beta,
        hipblasHandle_t handle,
        hipStream_t stream) {
    // All sizes must be within int bounds
    FAISS_ASSERT(c.getSize(0) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(c.getSize(1) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(c.getSize(2) <= std::numeric_limits<int>::max());

    FAISS_ASSERT(a.getSize(0) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(a.getSize(1) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(a.getSize(2) <= std::numeric_limits<int>::max());

    FAISS_ASSERT(a.getSize(0) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(a.getSize(1) <= std::numeric_limits<int>::max());
    FAISS_ASSERT(a.getSize(2) <= std::numeric_limits<int>::max());

    FAISS_ASSERT(c.getSize(0) == a.getSize(0));
    FAISS_ASSERT(a.getSize(0) == b.getSize(0));

    // This uses the strided batch MM, which assumes a uniform stride
    FAISS_ASSERT(a.getStride(0) == a.getSize(1) * a.getSize(2));
    FAISS_ASSERT(b.getStride(0) == b.getSize(1) * b.getSize(2));
    FAISS_ASSERT(c.getStride(0) == c.getSize(1) * c.getSize(2));

    hipblasSetStream(handle, stream);

    // Check that we have (m x k) * (k x n) = (m x n)
    // using the input row-major layout
    int aM = transA ? a.getSize(2) : a.getSize(1);
    int aK = transA ? a.getSize(1) : a.getSize(2);

    int bK = transB ? b.getSize(2) : b.getSize(1);
    int bN = transB ? b.getSize(1) : b.getSize(2);

    int cM = transC ? c.getSize(2) : c.getSize(1);
    int cN = transC ? c.getSize(1) : c.getSize(2);

    FAISS_ASSERT(aM == cM);
    FAISS_ASSERT(aK == bK);
    FAISS_ASSERT(bN == cN);

    // Now, we have to represent the matrix multiplication in
    // column-major layout
    void* pA = transC ? (void*)a.data() : (void*)b.data();
    void* pB = transC ? (void*)b.data() : (void*)a.data();
    float* pC = c.data();

    int m = c.getSize(2); // stride 1 size
    int n = c.getSize(1); // other size
    int k = transA ? a.getSize(1) : a.getSize(2);

    int lda = transC ? a.getStride(1) : b.getStride(1);
    int ldb = transC ? b.getStride(1) : a.getStride(1);
    int ldc = c.getStride(1);

    auto gemmTrA = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    auto gemmTrB = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    if (transC) {
        gemmTrA = transA ? HIPBLAS_OP_N : HIPBLAS_OP_T;
        gemmTrB = transB ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    }

    long long int gemmStrideA = transC ? a.getStride(0) : b.getStride(0);
    long long int gemmStrideB = transC ? b.getStride(0) : a.getStride(0);
    long long int gemmStrideC = c.getStride(0);

    auto err = rawBatchGemm<AT, BT>(
            handle,
            gemmTrA,
            gemmTrB,
            m,
            n,
            k,
            alpha,
            pA,
            lda,
            gemmStrideA,
            pB,
            ldb,
            gemmStrideB,
            beta,
            pC,
            ldc,
            gemmStrideC,
            a.getSize(0));

    FAISS_ASSERT_FMT(
            err == HIPBLAS_STATUS_SUCCESS,
            "hipblasGemmStridedBatchedEx failed (%d)",
            (int)err);
    HIP_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
