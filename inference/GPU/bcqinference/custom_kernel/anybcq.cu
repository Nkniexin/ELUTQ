#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "anybcq.h"
#include "typetraits.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>

#define K_TILE_SIZE 64
#define NUM_THREADS 256
#define M_TILE_SIZE 1024

#define K_TILE_SIZE_DEQUANT 4
#define NUM_THREADS_DEQUANT 64
#define M_TILE_SIZE_DEQUANT 64

// #define max_num_bits 4


__global__ void nqmv_bias(
    const uint32_t* q_weight, // quantized weights, W[kSize/32][nb][mSize]
    const __half* alpha, // alpha[num_groups][nb][mSize]
    const __half* q_bias, // q_bias[num_groups][mSize]
    const __half* input, // input[kSize]
    __half* output, // output[mSize]
    const int M, // mSize
    const int K, // kSize
    const int precision, // nb
    const int max_num_bits, // nb
    const int group_size // group_size
) {
    // Optimize shared memory layout to avoid bank conflicts
    __shared__ __half lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
 
    const int lut_y = threadIdx.x / lut_x_size;
    const int lut_x = threadIdx.x % lut_x_size;
 
    // Use vectorized load for better memory throughput
    const __half *_inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];
    
    // Use float4 vectorized loads when possible for better memory coalescing
    float4 inp_vec;
    inp_vec = ((float4*)_inp)[0];
    const __half2 *inp_half2 = (const __half2*)&inp_vec;
    
    __half base = __float2half((2 * ((lut_x>>0) & 1) - 1)) * inp_half2[0].x
                    + __float2half((2 * ((lut_x>>1) & 1) - 1)) * inp_half2[0].y
                    + __float2half((2 * ((lut_x>>2) & 1) - 1)) * inp_half2[1].x
                    + __float2half((2 * ((lut_x>>3) & 1) - 1)) * inp_half2[1].y
                    + __float2half((2 * ((lut_x>>4) & 1) - 1)) * inp_half2[2].x
                    + __float2half((2 * ((lut_x>>5) & 1) - 1)) * inp_half2[2].y
                    + __float2half((2 * ((lut_x>>6) & 1) - 1)) * inp_half2[3].x
                    + __float2half((2 * ((lut_x>>7) & 1) - 1)) * inp_half2[3].y;
    lut[lut_y][lut_x] = base;
 
    // Calculate starting point for LUT construction
    const int s = (lut_x_size==1)  ?0:
                  (lut_x_size==2)  ?1:
                  (lut_x_size==4)  ?2:
                  (lut_x_size==8)  ?3:
                  (lut_x_size==16) ?4:
                  (lut_x_size==32) ?5:
                  (lut_x_size==64) ?6: 
                  (lut_x_size==128)?7: 8;
 
    // Unroll loop for better performance when possible
    #pragma unroll
    for(int s_iter = s; s_iter < 8; s_iter++){
        const __half iValue = __float2half(2) * _inp[s_iter];
        #pragma unroll
        for (int i = (1 << s_iter); i < (1 << (s_iter + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] = lut[lut_y][i + lut_x - (1 << s_iter)] + iValue;
        }
    }
     __syncthreads();
 
    const int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x * 2;
    const int m_end = min((blockIdx.x + 1) * M_TILE_SIZE, M);
    const int m_step = blockDim.x * 2;

    const uint32_t* __restrict__ bW = &q_weight[blockIdx.y * K_TILE_SIZE / 32 * max_num_bits * M];
    const int group_idx = (blockIdx.y * K_TILE_SIZE) / group_size;
    for (int m = m_start; m < m_end; m += m_step) {
        __half2 acc = __halves2half2(0,0);

        {
            // Use vectorized loads for bias values
            const __half2 qb = ((const __half2*)&q_bias[group_idx*M + m])[0];
            __half2 t = __halves2half2(0, 0);
            
            #pragma unroll
            for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                // Optimize reduction with better instruction scheduling
                const __half t0 = __hadd(__hadd(lut[kt*4+0][255], lut[kt*4+1][255]),
                                        __hadd(lut[kt*4+2][255], lut[kt*4+3][255]));
                const __half2 tt = __halves2half2(t0, t0);
                t = __hadd2(t, tt);
            }
            acc = __hfma2(qb, t, acc);
        }

        #pragma unroll
        for (int b = 0; b < precision; ++b) {
            __half2 t = __halves2half2(0, 0);
            
            #pragma unroll
            for (int kt = 0; kt < K_TILE_SIZE/32; ++kt) {
                // Use vectorized loads for weights
                const uint64_t w_pair = ((const uint64_t*)&bW[kt*max_num_bits*M + b*M + m])[0];
                const uint32_t w0 = (uint32_t)w_pair;
                const uint32_t w1 = (uint32_t)(w_pair >> 32);
                
                const uchar4 by0 = *reinterpret_cast<const uchar4*>(&w0);
                const uchar4 by1 = *reinterpret_cast<const uchar4*>(&w1);
                
                // Optimize lookup with better instruction scheduling
                const __half t00 = __hadd(__hadd(lut[kt*4+0][by0.x], lut[kt*4+1][by0.y]),
                                         __hadd(lut[kt*4+2][by0.z], lut[kt*4+3][by0.w]));
                const __half t11 = __hadd(__hadd(lut[kt*4+0][by1.x], lut[kt*4+1][by1.y]),
                                         __hadd(lut[kt*4+2][by1.z], lut[kt*4+3][by1.w]));
                t = __hadd2(t, __halves2half2(t00, t11));
            }
            
            // Use vectorized load for alpha values
            const __half2 a = ((const __half2*)&alpha[group_idx*precision*M + b*M + m])[0];
            acc = __hfma2(a, t, acc);
        }
        // Use vectorized atomic add for better performance
        atomicAdd((__half2*)&output[m], acc);
     }
 }

__global__ void dequantize_t(
    uint32_t* q_weight, // quantized weights, bW[kSize/32][nb][mSize]
    __half* alpha, // alpha[num_groups][nb][mSize]
    __half* q_bias, // q_bias[num_groups][mSize]
    __half* output, // dequantized weights,[kSize][mSize]
    int M, // mSize
    int K, // kSize
    int precision, // nb
    int max_num_bits, // nb
    int group_size // group_size
){
    int m_step = blockDim.y;

    int m_start = blockIdx.y * M_TILE_SIZE_DEQUANT + threadIdx.y;
    int m_end = (blockIdx.y + 1) * M_TILE_SIZE_DEQUANT;
    m_end = (m_end < M) ? m_end : M;

    int k     = blockIdx.x * K_TILE_SIZE_DEQUANT + threadIdx.x;
    int tk = k/32;
    int t  = k%32;
    int k_end = (blockIdx.x + 1) * K_TILE_SIZE_DEQUANT;
    k_end = (k_end < K) ? k_end : K;

    int g_idx = (blockIdx.x * K_TILE_SIZE_DEQUANT/group_size);

    for(int m = m_start;m<m_end;m += m_step){
        if(k < k_end){
            __half r = 0;
            for(int b = 0;b<precision;b++){
                if((q_weight[tk * max_num_bits * M + b * M + m] >> t) & 1) r += alpha[g_idx * precision*M + b * M + m];
                else                                             r -= alpha[g_idx * precision*M + b * M + m];
            }
            output[k * M + m] = r + q_bias[g_idx * M + m];
        }
    }
}