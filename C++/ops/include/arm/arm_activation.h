#pragma once
#include <arm_device.h>


void gated_silu(
    float_type* output,
    float_type* gate,
    float_type* up,
    int ntokens,
    int dimension,
    int threadnum,
    AliveThreadPool*pool
);