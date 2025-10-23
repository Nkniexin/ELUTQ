#pragma once
#include <arm_device.h>
#include <arm_t_mac_kernal.h>


void arm_gemv(
    int m,
    int k,
    int n, 
    float_type*input,
    uint8_t *weight,
    float_type *w_scale,
    float_type* w_zero, 
    int8_t* lut,
    float_type * lut_scale,
    float_type* output,
    int Actk,
    int bit
);