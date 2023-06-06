/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Float16.h>

#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

//
// Templated wrappers to express load/store for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace faiss {
namespace gpu {

template <typename T>
struct LoadStore {
    static inline __device__ T load(void* p) {
        return *((T*)p);
    }

    static inline __device__ void store(void* p, const T& v) {
        *((T*)p) = v;
    }
};

template <>
struct LoadStore<Half4> {
    static inline __device__ Half4 load(void* p) {
        Half4 out;
// TODO: HADI 
        out.a.x = *(reinterpret_cast<uint32_t*>(p));
        out.b.x = *(reinterpret_cast<uint32_t*>(p) + 4);
        // asm volatile ("s_load_dwordx2 v[2:3], %1 glc\n\t"
        //       "v_pack_b32_f16 v[4], v[2]\n\t"
        //       "v_cvt_f32_u32 v[5], v[4]\n\t"
        //       "v_pack_b32_f16 v[6], v[3]\n\t"
        //       "v_cvt_f32_u32 v[7], v[6]\n\t"
        //       "v_pack_b32_f16 v[8], v[5]\n\t"
        //       "v_pack_b32_f16 v[9], v[7]\n\t"
        //       "v_cvt_u32_f16 v[10], v[8]\n\t"
        //       "v_cvt_u32_f16 v[11], v[9]\n\t"
        //       "v_mov_b32 v2, v[10]\n\t"
        //       "v_mov_b32 v3, v[11]"
        //       : "=v"(out.a.x), "=v"(out.b.x)
        //       : "v"(p)
        //       : "memory", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");

        return out;
    }

    static inline __device__ void store(void* p, Half4& v) {
        // asm volatile ("v_cvt_f32_f16 v[2], %1\n\t"
        //       "v_cvt_f32_f16 v[3], %2\n\t"
        //       "v_cvt_f32_f16 v[4], %3\n\t"
        //       "v_cvt_f32_f16 v[5], %4\n\t"
        //       "s_buffer_store_dwordx4 v[2:5], offset:0, %0 glc\n\t"
        //       :
        //       : "v"(p), "v"(v.a.x), "v"(v.a.y), "v"(v.b.x), "v"(v.b.y)
        //       : "memory", "v2", "v3", "v4", "v5");
        *reinterpret_cast<uint32_t*>(p) = v.a.x;
        *(reinterpret_cast<uint32_t*>(p) + 4) = v.b.x;
    }
};

template <>
struct LoadStore<Half8> {
    static inline __device__ Half8 load(void* p) {
        Half8 out;
        // asm volatile("ds_read_b32 %0, %1 offset:0\n"
        //      "ds_read_b32 %1, %1 offset:4 \n"
        //      "ds_read_b32 %2, %1 offset:8 \n"
        //      "ds_read_b32 %3, %1 offset:12 \n"
        //      : "=v"(out.a.a.x), "=v"(out.a.b.x), "=v"(out.b.a.x), "=v"(out.b.b.x)
        //      : "v"(p));
        out.a.a.x = *(reinterpret_cast<uint32_t*>(p));
        out.a.b.x = *(reinterpret_cast<uint32_t*>(p) + 4);
        out.b.a.x = *(reinterpret_cast<uint32_t*>(p) + 8);
        out.b.b.x = *(reinterpret_cast<uint32_t*>(p) + 12);
        
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {

    // asm("buffer_store_dwordx4 %0, {%1, %2, %3, %4}, offset:0 offen glc\n"
    //         : "=v"(p)
    //         :  "v"(v.a.a.x), "v"(v.a.b.x), "v"(v.b.a.x), "v"(v.b.b.x));

        *reinterpret_cast<uint32_t*>(p) = v.a.a.x;
        *(reinterpret_cast<uint32_t*>(p) + 4) = v.a.b.x;
        *(reinterpret_cast<uint32_t*>(p) + 8) = v.b.a.x;
        *(reinterpret_cast<uint32_t*>(p) + 12) = v.b.b.x;

    }
};

} // namespace gpu
} // namespace faiss
