#pragma once
#include <arm_device.h>

void arm_embedding(
    float_type* out,
    int* input_ids,
    float_type* embedding,
    int num_tokens,
    int hidden_size,
    void* unused
);