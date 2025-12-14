#pragma once
#ifdef _WIN32
#include <windows.h>
#define aligned_alloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#define NOMINMAX
#include <cstdint>
#include "stdlib.h"

#ifndef _WIN32

#define aligned_alloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)

#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <cmath>
#include <unordered_map>
#include <map>
#include <iostream>
inline float16x8_t fp16_exp(float16x8_t &a) {

    float16_t* ptr = (float16_t*)(&a);
    float16x8_t result;
    float16_t* result_ptr = (float16_t*)(&result);
    
    for(int i = 0; i < 8; i++) {
        result_ptr[i] = (float16_t)(expf((float)ptr[i]));
    }

    return result;
}

inline float32x4_t fp32_exp(float32x4_t a) {

    float32_t* ptr = (float32_t*)(&a);
    float32x4_t result;
    float32_t* result_ptr = (float32_t*)(&result);
    
    for(int i = 0; i < 4; i++) {
        result_ptr[i] = expf(ptr[i]);
    }

    return result;
}

inline  float16_t fp16_sum(float16x8_t vec) {
    float16_t* ptr = (float16_t*)&vec;
    float16_t sum = 0.0f;
    sum = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
    return sum;
}

inline  float fp32_sum(float16x8_t vec) {
    float16_t* ptr = (float16_t*)&vec;
    float sum = 0.0f;
    for(int i = 0;i<8 ;i++){
        sum += ptr[i];
    }
    return sum;
}


inline void print_float16x8_t(float16x8_t a ){

    float16_t* ptr = (float16_t *)(&a);
    for(int k = 0;k<8;k++)std::cout<<ptr[k]<<" ";
    std::cout<<std::endl;

}