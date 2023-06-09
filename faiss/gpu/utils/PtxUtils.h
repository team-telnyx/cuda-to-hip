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

// defines to simplify the SASS assembly structure file/line in the profiler
#define GET_BITFIELD_U32(OUT, VAL, POS, LEN) \
    asm("v_bfe_u32 %0, %1, %2, %3;" : "=v"(OUT) : "v"(VAL), "v"(POS), "v"(LEN));

#define GET_BITFIELD_U64(OUT, VAL, POS, LEN) \
    asm("v_bfe_u64 %0, %1, %2, %3;" : "=v"(OUT) : "v"(VAL), "v"(POS), "v"(LEN));

__device__ __forceinline__ unsigned int getBitfield(
        unsigned int val,
        int pos,
        int len) {
    unsigned int ret;
    asm("v_bfe_u32 %0, %1, %2, %3;" : "=v"(ret) : "v"(val), "v"(pos), "v"(len));
    return ret;
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
    uint64_t ret;
    asm("v_bfe_u64 %0, %1, %2, %3;" : "=v"(ret) : "v"(val), "v"(pos), "v"(len));
    return ret;
}

__device__ __forceinline__ unsigned int setBitfield(
        unsigned int val,
        unsigned int toInsert,
        int pos,
        int len) {
    unsigned int ret;
    asm("v_bfi_b32 %0, %1, %2, %3, %4;"
        : "=v"(ret)
        : "v"(toInsert), "v"(val), "v"(pos), "v"(len));
    return ret;
}

__device__ __inline__ int getLaneId() {
    unsigned int laneID;
    asm volatile(
        "s_mov_b32 s0, m0\n"
        "s_and_b32 %0, s0, 0x3F"
        : "=r" (laneID)
        :
        : "s0"
    );
    return laneID;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm volatile ("s_mov_b32 s[0], 0xFFFFFFFF\n\t"
              "s_mov_b32 s[1], s[0]\n\t"
              "s_waive_all_vgprs\n\t"
              "v_mov_b32 v[0], 0xFFFFFFFF\n\t"
              "v_cmp_lt_u32 v[1], vcc, v[0]\n\t"
              "v_readlane_b32 %0, v[1]\n\t"
              : "=v"(mask)
              :
              : "memory", "v0", "v1", "s0", "s1");

    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
    unsigned mask;
    asm("v_mov_b32 %%lanemask_le, %0;" : "=v"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
    unsigned mask;
    asm("v_mov_b32 %%lanemask_gt, %0;" : "=v"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
    unsigned mask;
    asm("v_mov_b32 %%lanemask_ge, %0;" : "=v"(mask));
    return mask;
}

__device__ __forceinline__ void namedBarrierWait(int name, int numThreads) {
    asm volatile("s_barrier %0, %1;"
        :
        : "s"(name), "s"(numThreads)
        : "memory");
}

__device__ __forceinline__ void namedBarrierArrived(int name, int numThreads) {
    asm volatile("s_sendmsg 0x3, %1, 0, %0;"
    :
    : "r"(name), "r"(numThreads)
    : "memory");
}

} // namespace gpu
} // namespace faiss
