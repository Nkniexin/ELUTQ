#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "lutgemm.h"
#include "typetraits.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>

#define K_TILE_SIZE 64
#define NUM_THREADS 256
#define M_TILE_SIZE 2048

#define K_TILE_SIZE_DEQUANT 4
#define NUM_THREADS_DEQUANT 64
#define M_TILE_SIZE_DEQUANT 64

// #define max_num_bits 4


__global__ void nqmv_bias(
    uint32_t* q_weight, // quantized weights, W[kSize/32][nb][mSize]
    __half* alpha, // alpha[num_groups][nb][mSize]
    __half* q_bias, // q_bias[num_groups][mSize]
    __half* input, // input[kSize]
    __half* output, // output[mSize]
    int M, // mSize
    int K, // kSize
    int precision, // nb
    int max_num_bits, // nb
    int group_size // group_size
) {
    __shared__ __half lut[K_TILE_SIZE / 8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE / 8);

    int lut_y = threadIdx.x / lut_x_size;
    int lut_x = threadIdx.x % lut_x_size;

    __half* _inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];

    __half base =    + __float2half((2 * ((lut_x>>0) & 1) - 1)) * _inp[0]
                     + __float2half((2 * ((lut_x>>1) & 1) - 1)) * _inp[1]
                     + __float2half((2 * ((lut_x>>2) & 1) - 1)) * _inp[2]
                     + __float2half((2 * ((lut_x>>3) & 1) - 1)) * _inp[3]
                     + __float2half((2 * ((lut_x>>4) & 1) - 1)) * _inp[4]
                     + __float2half((2 * ((lut_x>>5) & 1) - 1)) * _inp[5]
                     + __float2half((2 * ((lut_x>>6) & 1) - 1)) * _inp[6]
                     + __float2half((2 * ((lut_x>>7) & 1) - 1)) * _inp[7];

    lut[lut_y][lut_x] = base;

    int s = (lut_x_size == 1)
                ? 0
                : (lut_x_size == 2)
                ? 1
                : (lut_x_size == 4)
                ? 2
                : (lut_x_size == 8)
                ? 3
                : (lut_x_size == 16)
                ? 4
                : (lut_x_size == 32)
                ? 5
                : (lut_x_size == 64)
                ? 6
                : (lut_x_size == 128)
                ? 7
                : 8;

    for (; s < 8; s++) {
        __half iValue = __float2half(2) * _inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] = lut[lut_y][i + lut_x - (1 << s)] + iValue;
        }
    }
    __syncthreads();

    int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x * 2;
    int m_end = (blockIdx.x + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = blockDim.x * 2;

    uint32_t* bW = &q_weight[blockIdx.y * K_TILE_SIZE / 32 * max_num_bits * M];
    int group_idx = (blockIdx.y * K_TILE_SIZE) / group_size;
    for (int m = m_start; m < m_end; m += m_step) {
        __half2 acc = __halves2half2(0,0);

        {
            __half2 qb = __halves2half2(q_bias[group_idx*M + m + 0],
                                        q_bias[group_idx*M + m + 1]);
            __half2 t  = __halves2half2(0,0);
            #pragma unroll
            for (int kt=0; kt < K_TILE_SIZE/32; ++kt) {
                __half t0 = __hadd(__hadd(lut[kt*4+0][255], lut[kt*4+1][255]),
                                __hadd(lut[kt*4+2][255], lut[kt*4+3][255]));
                __half2 tt = __halves2half2(t0, t0);
                t = __hadd2(t, tt);
            }
            acc = __hfma2(qb, t, acc);
        }

        #pragma unroll
        for (int b=0; b<precision; ++b) {
            __half2 t = __halves2half2(0,0);
            #pragma unroll
            for (int kt=0; kt < K_TILE_SIZE/32; ++kt) {
                // m
                uint32_t w0 = bW[kt*max_num_bits*M + b*M + m + 0];
                uchar4   by0 = *reinterpret_cast<uchar4*>(&w0);
                __half t00 = __hadd(__hadd(lut[kt*4+0][by0.x], lut[kt*4+1][by0.y]),
                                    __hadd(lut[kt*4+2][by0.z], lut[kt*4+3][by0.w]));
                // m+1
                uint32_t w1 = bW[kt*max_num_bits*M + b*M + m + 1];
                uchar4   by1 = *reinterpret_cast<uchar4*>(&w1);
                __half t11 = __hadd(__hadd(lut[kt*4+0][by1.x], lut[kt*4+1][by1.y]),
                                    __hadd(lut[kt*4+2][by1.z], lut[kt*4+3][by1.w]));
                t = __hadd2(t, __halves2half2(t00, t11));
            }
            __half2 a = __halves2half2(alpha[group_idx*precision*M + b*M + m + 0],
                                    alpha[group_idx*precision*M + b*M + m + 1]);
            acc = __hfma2(a, t, acc);

            // // a *= 2
            // a = __hmul2(a, __half2half2(two));
        }
	    atomicAdd((half2*)&output[m], acc);
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