#pragma once
#include<arm_device.h>

void arm_cache(
    float16_t* k,
    float16_t* v,
    float16_t* kcache,
    float16_t* vcache,
    int num_tokens,
    int head_kv,
    int dhead,
    int* cache_positions
);