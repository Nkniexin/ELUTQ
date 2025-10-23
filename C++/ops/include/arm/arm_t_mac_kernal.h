#pragma once
#include <stdint.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <arm_neon.h>
#include <arm_device.h>
#include <arm_nnstructures.h>


void t_mac_g4_groupsize32_calculate_lut_impl(int in_channel,const float_type * activation, int8_t * lut , float_type*lut_scale,int Actk);

void t_mac_g4_groupsize32_calculate_float_lut_impl(int in_channel,const float_type * activation,uint8_t * lut,int Actk);

void t_mac_g4_float_update_impl(
    float_type* out,
    uint8_t* weight,
    int in_channel,
    int out_channel,
    uint8_t* lut,
    float_type* w_scale,
    int weight_order,
    int ActK
);

void t_mac_g4_int8_update_impl(
    float_type* out,
    uint8_t* weight,
    int in_channel,
    int out_channel,
    int8_t* lut,
    float_type* lut_scale,
    float_type* w_scale,
    int weight_order,
    int ActK
);

void t_mac_g4_float_update_sparse_impl(
    float_type* out,
    uint8_t* weight,
    int in_channel,
    int out_channel,
    uint8_t* lut,
    float_type* w_scale,
    const sparse_mask &mask,
    int weight_order,
    int ActK
);

bool t_mac_g4_rearrange_weight_impl(weight_tensor* weight,uint8_t* workplace,int n_thread);