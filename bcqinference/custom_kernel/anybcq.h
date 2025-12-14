#ifndef ANYBCQ_CUH
#define ANYBCQ_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "datatype.h"
#include "typetraits.h"

#include <torch/extension.h>
#include <cuda_runtime.h>

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
);

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
);

#endif

