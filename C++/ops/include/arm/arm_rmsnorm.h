#pragma once
#include<arm_device.h>


void launch_rms_norm(
    float16_t* output,
    float16_t* vals,
    float16_t* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    int threadnum,
    AliveThreadPool *pool
);

void launch_pre_rms_norm(
    float16_t* output,
    float16_t* res_output,
    float16_t* vals,
    float16_t* residual,
    float16_t* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    int threadnum,
    AliveThreadPool *pool
);

