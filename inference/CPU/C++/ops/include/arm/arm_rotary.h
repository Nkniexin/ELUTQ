#pragma once
#include <arm_device.h>


void launch_llama_qkv_apply_rotary(
    float_type* q,
    float_type* k,
    float_type* v,
    float_type* x,
    const int* pid,  // if we need to change to uint64_t ?
    int num_tokens,
    int head_q,
    int head_kv,
    int dhead,
    int apply_logn,
    float rotary_base 
);