#pragma once
#include <arm_device.h>

void arm_argmax_last(
    int* output,
    const float_type* input,
    int* max_index_temp,
    float_type* max_value_temp,
    int ntokens,  //we actually only care about the prediction of last token
    int dimension
);
