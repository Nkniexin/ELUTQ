#include <arm_device.h>
#include <arm_t_mac_kernal.h>

void arm_gemm(
    int m,
    int k,
    int n, 
    float_type*input,
    float_type* output,
    void* temp_workplace,
    weight_tensor*weight,
    int threadnum,
    AliveThreadPool*pool
);