// Originally from https://github.com/snu-mllab/GuidedQuant/blob/main/inference/ap_gemv/gemv.h

#ifndef GEMV_CUH
#define GEMV_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>

#include <torch/extension.h>
#include <cuda_runtime.h>

void anyprec_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

void anybcq_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int max_num_bits,
    int group_size
);

torch::Tensor anyprec_dequant(
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

torch::Tensor anybcq_dequant(
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int max_num_bits,
    int group_size
);

#endif 
