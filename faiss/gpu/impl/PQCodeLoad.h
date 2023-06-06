/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <faiss/gpu/utils/PtxUtils.h>

namespace faiss {
namespace gpu {

// TODO: HADI
// Replace all this instruction inplace:
/*
#if __HIP_DEVICE_COMPILE__ >= 350
// Use the CC 3.5+ read-only texture cache (nc)
#define LD_NC_V1 "llvm.amdgcn.buffer.load.dword"
#define LD_NC_V2 "llvm.amdgcn.buffer.load.dwordx2"
#define LD_NC_V4 "llvm.amdgcn.buffer.load.dwordx4"
#else
// Read normally
#define LD_NC_V1 "llvm.amdgcn.buffer.load.dword"
#define LD_NC_V2 "llvm.amdgcn.buffer.load.dwordx2"
#define LD_NC_V4 "llvm.amdgcn.buffer.load.dwordx4"
#endif // __HIP_DEVICE_COMPILE__
*/

///
/// This file contains loader functions for PQ codes of various byte
/// length.
///

// Type-specific wrappers around the PTX bfe.* instruction, for
// quantization code extraction
inline __device__ unsigned int getByte(unsigned char v, int pos, int width) {
    return v;
}

inline __device__ unsigned int getByte(unsigned short v, int pos, int width) {
    return getBitfield((unsigned int)v, pos, width);
}

inline __device__ unsigned int getByte(unsigned int v, int pos, int width) {
    return getBitfield(v, pos, width);
}

inline __device__ unsigned int getByte(uint64_t v, int pos, int width) {
    return getBitfield(v, pos, width);
}

template <int NumSubQuantizers>
struct LoadCode32 {};

template <>
struct LoadCode32<1> {
    static inline  __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 1;

        // TODO: HADI
        auto a = *reinterpret_cast<uint8_t*>(p);
        code32[0] = a;
        // asm volatile(
        //     "s_load_dword %0, %1 offset:0 glc"
        //     : "=s"(code32[0])
        //     : "s"(reinterpret_cast<const uint64_t*>(p))
        //     : "memory"
        // );
    }
};

template <>
struct LoadCode32<2> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
        p += offset * 2;
        // TODO: HADI
      auto a = *reinterpret_cast<uint16_t*>(p);
      code32[0] = a;
    }
};

template <>
struct LoadCode32<3> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
            p += offset * 3;

            // Load individual bytes using ds_read_u8
            // TODO: HADI
            unsigned int a = *p;
            unsigned int b = *(p + 1);
            unsigned int c = *(p + 2);

            // Combine the bytes to form the 32-bit value
            code32[0] = (c << 16) | (b << 8) | a;
      }     
};

template <>
struct LoadCode32<4> {
    static inline __device__ void load(
            unsigned int code32[1],
            uint8_t* p,
            int offset) {
            p += offset * 4;
            // TODO: HADI
            // asm volatile("flat_load_dwordx2 %0, v[3:4] glc;" : "=v"(temp) : "v"(p) : "memory", "v3", "v4");
            code32[0] = *reinterpret_cast<unsigned int*>(p);
    }
};

template <>
struct LoadCode32<8> {
    static inline __device__ void load(
            unsigned int code32[2],
            uint8_t* p,
            int offset) {
        p += offset * 8;
        // TODO: HADI
      code32[0] = *reinterpret_cast<unsigned int*>(p);
      code32[1] = *reinterpret_cast<unsigned int*>(p + 1);
    }
};

template <>
struct LoadCode32<12> {
    static inline __device__ void load(
            unsigned int code32[3],
            uint8_t* p,
            int offset) {
        p += offset * 12;
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
        
        // TODO: HADI
      code32[0] = *reinterpret_cast<unsigned int*>(p);
      code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
      code32[2] = *reinterpret_cast<unsigned int*>(p + 8);

    }
};

template <>
struct LoadCode32<16> {
    static inline __device__ void load(
            unsigned int code32[4],
            uint8_t* p,
            int offset) {
        p += offset * 16;

        // TODO: HADI
      code32[0] = *reinterpret_cast<unsigned int*>(p + 0);
      code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
      code32[2] = *reinterpret_cast<unsigned int*>(p + 8);
      code32[3] = *reinterpret_cast<unsigned int*>(p + 12);
    }
};

template <>
struct LoadCode32<20> {
    static inline __device__ void load(
            unsigned int code32[5],
            uint8_t* p,
            int offset) {
        p += offset * 20;
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *reinterpret_cast<unsigned int*>(p);
      code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
      code32[2] = *reinterpret_cast<unsigned int*>(p + 8);
      code32[3] = *reinterpret_cast<unsigned int*>(p + 12);
      code32[4] = *reinterpret_cast<unsigned int*>(p + 16);
    }
};

template <>
struct LoadCode32<24> {
    static inline __device__ void load(
            unsigned int code32[6],
            uint8_t* p,
            int offset) {
        p += offset * 24;
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *reinterpret_cast<unsigned int*>(p + 0);
      code32[1] = *reinterpret_cast<unsigned int*>(p + 4);
      code32[2] = *reinterpret_cast<unsigned int*>(p + 8);
      code32[3] = *reinterpret_cast<unsigned int*>(p + 12);
      code32[4] = *reinterpret_cast<unsigned int*>(p + 16);
      code32[5] = *reinterpret_cast<unsigned int*>(p + 20);
    }
};

template <>
struct LoadCode32<28> {
    static inline __device__ void load(
            unsigned int code32[7],
            uint8_t* p,
            int offset) {
        p += offset * 28;
        // FIXME: this is a non-coalesced, unaligned, non-vectorized load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
      code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
      code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
      code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
      code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
      code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
      code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
    }
};

template <>
struct LoadCode32<32> {
    static inline __device__ void load(
            unsigned int code32[8],
            uint8_t* p,
            int offset) {
        p += offset * 32;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
      code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
      code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
      code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
      code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
      code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
      code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
      code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
    }
};

template <>
struct LoadCode32<40> {
    static inline __device__ void load(
            unsigned int code32[10],
            uint8_t* p,
            int offset) {
        p += offset * 40;
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
      code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
      code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
      code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
      code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
      code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
      code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
      code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
      code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
      code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
    }
};

template <>
struct LoadCode32<48> {
    static inline __device__ void load(
            unsigned int code32[12],
            uint8_t* p,
            int offset) {
        p += offset * 48;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
        // Load code32 values from memory using regular memory operations
      code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
      code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
      code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
      code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
      code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
      code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
      code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
      code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
      code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
      code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
      code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
      code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));

    }
};

template <>
struct LoadCode32<56> {
    static inline __device__ void load(
            unsigned int code32[14],
            uint8_t* p,
            int offset) {
        p += offset * 56;
        // FIXME: this is a non-coalesced, unaligned, 2-vectorized load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
      code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
      code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
      code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
      code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
      code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
      code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
      code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
      code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
      code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
      code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
      code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));
      code32[12] = *(reinterpret_cast<unsigned int*>(p + 48));
      code32[13] = *(reinterpret_cast<unsigned int*>(p + 52));
    }
};

template <>
struct LoadCode32<64> {
    static inline __device__ void load(
            unsigned int code32[16],
            uint8_t* p,
            int offset) {
        p += offset * 64;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
      code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
      code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
      code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
      code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
      code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
      code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
      code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
      code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
      code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
      code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
      code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));
      code32[12] = *(reinterpret_cast<unsigned int*>(p + 48));
      code32[13] = *(reinterpret_cast<unsigned int*>(p + 52));  
      code32[14] = *(reinterpret_cast<unsigned int*>(p + 56));
      code32[15] = *(reinterpret_cast<unsigned int*>(p + 60));  
    }
};

template <>
struct LoadCode32<96> {
    static inline __device__ void load(
            unsigned int code32[24],
            uint8_t* p,
            int offset) {
        p += offset * 96;
        // FIXME: this is a non-coalesced load
        // unfortunately need to reorganize memory layout by warp
        // TODO: HADI
      code32[0] = *(reinterpret_cast<unsigned int*>(p + 0));
      code32[1] = *(reinterpret_cast<unsigned int*>(p + 4));
      code32[2] = *(reinterpret_cast<unsigned int*>(p + 8));
      code32[3] = *(reinterpret_cast<unsigned int*>(p + 12));
      code32[4] = *(reinterpret_cast<unsigned int*>(p + 16));
      code32[5] = *(reinterpret_cast<unsigned int*>(p + 20));
      code32[6] = *(reinterpret_cast<unsigned int*>(p + 24));
      code32[7] = *(reinterpret_cast<unsigned int*>(p + 28));
      code32[8] = *(reinterpret_cast<unsigned int*>(p + 32));
      code32[9] = *(reinterpret_cast<unsigned int*>(p + 36));
      code32[10] = *(reinterpret_cast<unsigned int*>(p + 40));
      code32[11] = *(reinterpret_cast<unsigned int*>(p + 44));
      code32[12] = *(reinterpret_cast<unsigned int*>(p + 48));
      code32[13] = *(reinterpret_cast<unsigned int*>(p + 52));  
      code32[14] = *(reinterpret_cast<unsigned int*>(p + 56));
      code32[15] = *(reinterpret_cast<unsigned int*>(p + 60));  
      code32[16] = *(reinterpret_cast<unsigned int*>(p + 64));
      code32[17] = *(reinterpret_cast<unsigned int*>(p + 68));
      code32[18] = *(reinterpret_cast<unsigned int*>(p + 72));
      code32[19] = *(reinterpret_cast<unsigned int*>(p + 76));
      code32[20] = *(reinterpret_cast<unsigned int*>(p + 80));
      code32[21] = *(reinterpret_cast<unsigned int*>(p + 84));  
      code32[22] = *(reinterpret_cast<unsigned int*>(p + 88));
      code32[23] = *(reinterpret_cast<unsigned int*>(p + 92)); 
    }
};

} // namespace gpu
} // namespace faiss
