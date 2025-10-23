#pragma once
#include <arm_device.h>


struct sparse_mask
{
    uint32_t* data;
    uint32_t len;
};

class weight_tensor{
public:
    void *weight = nullptr; //权重meta 数据
    int in_channel = -1; 
    int out_channel = -1;
    float_type* w_scale = nullptr;
    float_type* w_zero = nullptr;
    float_type* bias = nullptr;
    int groupsize = -1;
    int bit = -1;
    int extra_bit = -1;
    bool use_sparse = false;
    uint32_t* sparse_mask_data = nullptr; //sparse_mask_data
    uint32_t sparse_mask_len = -1; //mask_len

    weight_tensor(){

    }

    ~weight_tensor(){
        if(weight)aligned_free(weight);
        if(w_scale)aligned_free(w_scale);
        if(w_zero)aligned_free(w_zero);
        if(bias)aligned_free(bias);
        if(sparse_mask_data)aligned_free(sparse_mask_data);
    }
};
