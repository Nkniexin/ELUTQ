#include <arm_gemv.h>


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
){

    for(int row = 0 ; row < m ; row++){

        t_mac_g4_groupsize32_calculate_lut_impl(
                k,input + row*k,lut,lut_scale,Actk
            );

        for(int i = 0 ; i < bit ;i++){
            int weight_bias = i * k * n / 8;
            t_mac_g4_int8_update_impl(
                output + row*n,
                weight + weight_bias,
                k,n,lut,lut_scale,w_scale,i,Actk);
        }
    }
    
}