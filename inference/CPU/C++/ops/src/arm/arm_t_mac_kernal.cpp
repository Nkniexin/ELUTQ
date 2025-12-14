
#include "arm_t_mac_kernal.h"
#ifndef ARM_T_MAC_KERNAL_H
#define ARM_T_MAC_KERNAL_H
#endif

#include <arm_neon.h>
typedef float16_t float_type;
/**
 * 对于一个通道的激活，默认groupsize = 32, g=4的t_mac查找表构建
 * @param activation:输入激活值，默认为fp32
 * 
 * 
 * */
void t_mac_g4_groupsize32_calculate_lut_impl(int in_channel,const float_type * activation, int8_t * lut , float_type*lut_scale,int Actk){ 
    
    float_type mul_max_per4 = 0;
    float_type temp[1024] = {0};
#pragma unroll
    for(int i = 0 ;i< in_channel; i += Actk){
        memset(temp,0,sizeof(temp));
        mul_max_per4 = 0;
#pragma unroll
        for(int j  = 0; j < Actk;j +=4){

#pragma unroll
            for(int g = 0 ; g < 16 ;g++){
                
                if(g & 0b0001){
                    temp[j*4 + g] += activation[i + j];
                }
                if(g & 0b0010){
                    temp[j*4 + g] += activation[i + j  + 1];
                }
                if(g & 0b0100){
                    temp[j*4 + g] += activation[i + j  + 2];
                }
                if(g & 0b1000){
                    temp[j*4 + g] += activation[i + j  + 3];
                }
                mul_max_per4 = std::max((float_type)std::abs(temp[j*4 + g]),mul_max_per4);
            }
        }

        float_type scale = mul_max_per4/127.0f;
        lut_scale[i/Actk] = scale;
        const float16x8_t scale_vec = vdupq_n_f16(scale);
        int8_t * out = lut + i*4;

#pragma unroll
        for (int k = 0; k < Actk; k += 16) {
            float16x8_t f0 = vld1q_f16(temp + k);
            float16x8_t f1 = vld1q_f16(temp + k + 8);

            f0 = vdivq_f16(f0, scale_vec);
            f1 = vdivq_f16(f1, scale_vec);

            float32x4_t f0_l = vcvt_f32_f16(vget_low_f16(f0));
            float32x4_t f0_h = vcvt_f32_f16(vget_high_f16(f0));
            float32x4_t f1_l = vcvt_f32_f16(vget_low_f16(f1));
            float32x4_t f1_h = vcvt_f32_f16(vget_high_f16(f1));

            // Step 2: 四舍五入
            f0_l = vrndnq_f32(f0_l);
            f0_h = vrndnq_f32(f0_h);
            f1_l = vrndnq_f32(f1_l);
            f1_h = vrndnq_f32(f1_h);

            // Step 3: float32 → int32
            int32x4_t i0_l = vcvtq_s32_f32(f0_l);
            int32x4_t i0_h = vcvtq_s32_f32(f0_h);
            int32x4_t i1_l = vcvtq_s32_f32(f1_l);
            int32x4_t i1_h = vcvtq_s32_f32(f1_h);

            // Step 4: int32 → int16
            int16x4_t i0_l_16 = vqmovn_s32(i0_l);
            int16x4_t i0_h_16 = vqmovn_s32(i0_h);
            int16x4_t i1_l_16 = vqmovn_s32(i1_l);
            int16x4_t i1_h_16 = vqmovn_s32(i1_h);

            int16x8_t i0_16 = vcombine_s16(i0_l_16, i0_h_16);
            int16x8_t i1_16 = vcombine_s16(i1_l_16, i1_h_16);

            // Step 5: int16 → int8
            int8x8_t i0_8 = vqmovn_s16(i0_16);
            int8x8_t i1_8 = vqmovn_s16(i1_16);

            // Step 6: combine to int8x16_t and store
            int8x16_t result = vcombine_s8(i0_8, i1_8);
            vst1q_s8(out + k, result);
        }
    }
}

void t_mac_g4_groupsize32_calculate_float_lut_impl(
    int in_channel,
    const float_type * activation,
    uint8_t * lut,
    int Actk
)

{

    float_type temp[16] = {0};
#pragma unroll
    for(int i = 0 ;i<in_channel;i+=4){

        memset(temp,0,sizeof(temp));

#pragma unroll
        for(int g = 0 ;g < 16 ; g++){

            if(g & 0b0001){
                    temp[ g] += activation[i ];
                }
            if(g & 0b0010){
                temp[ g] += activation[i + 1];
            }
            if(g & 0b0100){
                temp[ g] += activation[i   + 2];
            }
            if(g & 0b1000){
                temp[g] += activation[i  + 3];
            }
            
        }

        uint8x16_t a= vld1q_u8((uint8_t*)temp);

        uint8x16_t b = vld1q_u8((uint8_t*)temp + 16);

            
        vst1q_u8(lut+i/4*32,a);
        
        vst1q_u8(lut+i/4*32 + 16,b);

    }
}


template <int N>
struct SignedHalvingAdder {
    SignedHalvingAdder<N / 2> adder;
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k < N / 2) {
            adder.push(v, k);
            if (k == N / 2 - 1) {
                lhs = adder.get();
            }
        } else {
            adder.push(v, k - N / 2);
            if (k == N - 1) {
                lhs = vrhaddq_s8(lhs, adder.get());
            }
        }
    }

    inline int8x16_t get() {
        return lhs;
    }

    inline int16x8_t get_low() {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high() {
        return vmovl_high_s8(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = vrhaddq_s8(lhs, v);
        }
    }

    inline int8x16_t get() {
        return lhs;
    }

    inline int16x8_t get_low() {
        return vmovl_s8(vget_low_s8(lhs));
    }

    inline int16x8_t get_high() {
        return vmovl_high_s8(lhs);
    }
};

struct SignedLongAdder {
    int16x8_t lhs_low;
    int16x8_t lhs_high;
    int8x16_t lhs;

    inline void push(int8x16_t v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs_low = vaddl_s8(vget_low_s8(lhs), vget_low_s8(v));
            lhs_high = vaddl_high_s8(lhs, v);
        }
    }

    inline int16x8_t get_low() {
        return lhs_low;
    }

    inline int16x8_t get_high() {
        return lhs_high;
    }
};

template <int N>
struct SignedWideningAdder {
    SignedLongAdder adder;
    int16x8_t lhs_low;
    int16x8_t lhs_high;

    inline void push(int8x16_t v, int k) {
        if (k % 2 == 0) {
            adder.push(v, 0);
        } else {
            adder.push(v, 1);
            if (k == 1) {
                lhs_low = adder.get_low();
                lhs_high = adder.get_high();
            } else {
                lhs_low += adder.get_low();
                lhs_high += adder.get_high();
            }
        }
    }

    inline int16x8_t get_low() {
        return lhs_low;
    }

    inline int16x8_t get_high() {
        return lhs_high;
    }
};


/**
 * 执行bit-wise矩阵乘法
 * @param activation:输入激活值，默认为fp32
 * @param weight:输入权重值，这里的权重是bit值，即每个权重均只有一个bit
 * 
 * */
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
){
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);

    int K = in_channel/4;
    int limit = int(((float)(ActK))*0.8f);
    int8x16_t vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }

    // SignedWideningAdder<32>adder_bot,adder_top;
    int16x8_t addr1 ,addr2 ,addr3, addr4;
#pragma unroll
    for(int i = 0 ;i< out_channel ;i += 32){

        float16x8_t vec_c0,vec_c1,vec_c2,vec_c3;
#pragma unroll
        for(int j = 0 ; j< in_channel ;j += ActK){

            addr1 = vdupq_n_s16(0);
            addr2 = vdupq_n_s16(0);
            addr3 = vdupq_n_s16(0);
            addr4 = vdupq_n_s16(0);
#pragma unroll
            for(int l = 0 ; l< ActK ; l += 8){
                
                uint8x16_t vec_as_1 = vld1q_u8(weight + (i*in_channel)/8 + (j + l)*4);
                uint8x16_t vec_as_2 = vld1q_u8(weight + (i*in_channel)/8 + (j + l)*4 + 16);
                uint8x16_t vec_a_top_1 = vshrq_n_u8(vec_as_1, 4);
                uint8x16_t vec_a_bot_1 = vandq_u8(vec_as_1, vec_mask);
                uint8x16_t vec_a_top_2 = vshrq_n_u8(vec_as_2, 4);
                uint8x16_t vec_a_bot_2 = vandq_u8(vec_as_2, vec_mask);

                int8x16_t vec_v_bot_tmp_1 = vqtbl1q_s8(vec_lut[(j+l) / 4], vec_a_bot_1);
                int8x16_t vec_v_bot_tmp_2 = vqtbl1q_s8(vec_lut[(j+l) / 4], vec_a_bot_2);
                int8x16_t vec_v_top_tmp_1 = vqtbl1q_s8(vec_lut[(j+l) / 4 + 1], vec_a_top_1);
                int8x16_t vec_v_top_tmp_2 = vqtbl1q_s8(vec_lut[(j+l) / 4 + 1], vec_a_top_2);


                addr1 = vaddq_s16(addr1,vmovl_s8(vget_low_s8(vec_v_bot_tmp_1)));
                addr1 = vaddq_s16(addr1,vmovl_s8(vget_low_s8(vec_v_bot_tmp_2)));
                addr2 = vaddq_s16(addr2,vmovl_s8(vget_high_s8(vec_v_bot_tmp_1)));
                addr2 = vaddq_s16(addr2,vmovl_s8(vget_high_s8(vec_v_bot_tmp_2)));
                addr3 = vaddq_s16(addr3,vmovl_s8(vget_low_s8(vec_v_top_tmp_1)));
                addr3 = vaddq_s16(addr3,vmovl_s8(vget_low_s8(vec_v_top_tmp_2)));
                addr4 = vaddq_s16(addr4,vmovl_s8(vget_high_s8(vec_v_top_tmp_1)));
                addr4 = vaddq_s16(addr4,vmovl_s8(vget_high_s8(vec_v_top_tmp_2)));
                

            }

            float16x8_t vec_v_bot_low  = vcvtq_f16_s16(addr1);
            float16x8_t vec_v_bot_high = vcvtq_f16_s16(addr2);
            float16x8_t vec_v_top_low  = vcvtq_f16_s16(addr3);
            float16x8_t vec_v_top_high = vcvtq_f16_s16(addr4);
            
            float16x8_t vec0_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 0);
            float16x8_t vec1_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 8);
            float16x8_t vec2_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 16);
            float16x8_t vec3_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 24);
            
            float16x8_t lut_s = vdupq_n_f16(lut_scale[j/ActK]);

            if(j == 0){
                vec_c0 = vec_v_bot_low * lut_s * vec0_w_scale ;
                vec_c1 = vec_v_bot_high * lut_s * vec1_w_scale;
                vec_c2 = vec_v_top_low * lut_s * vec2_w_scale;
                vec_c3 = vec_v_top_high * lut_s * vec3_w_scale;
            }
            else{
                vec_c0 += vec_v_bot_low * lut_s*vec0_w_scale;
                vec_c1 += vec_v_bot_high * lut_s*vec1_w_scale;
                vec_c2 += vec_v_top_low * lut_s*vec2_w_scale;
                vec_c3 += vec_v_top_high * lut_s*vec3_w_scale;
            }
        }


        float16x8_t weight_pow = vdupq_n_f16((1)<<weight_order);
        if(weight_order == 0){
            vst1q_f16(out + i,vec_c0);
            vst1q_f16(out + i + 8,vec_c1);
            vst1q_f16(out + i + 16,vec_c2);
            vst1q_f16(out + i + 24,vec_c3);
        }
        else{
            vst1q_f16(out + i,vaddq_f16(vld1q_f16(out + i),vmulq_f16(vec_c0,weight_pow)));
            vst1q_f16(out + i + 8,vaddq_f16(vld1q_f16(out + i + 8),vmulq_f16(vec_c1,weight_pow)));
            vst1q_f16(out + i + 16,vaddq_f16(vld1q_f16(out + i + 16),vmulq_f16(vec_c2,weight_pow)));
            vst1q_f16(out + i + 24,vaddq_f16(vld1q_f16(out + i + 24),vmulq_f16(vec_c3,weight_pow)));
        }
    }
}

void t_mac_g4_float_update_impl(
    float_type* out,
    uint8_t* weight,
    int in_channel,
    int out_channel,
    uint8_t* lut,
    float_type* w_scale,
    int weight_order,
    int ActK
){
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);

    int K = in_channel/4;
    int limit = int(((float)(ActK))*0.8f);

    uint8x16x2_t vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld2q_u8(lut + k * 32);
    }

    float16x8_t vec_c0, vec_c1, vec_c2, vec_c3;
    float16x8_t addr1, addr2, addr3, addr4;

#pragma unroll
    for(int i = 0 ; i < out_channel ;i += 32){
#pragma unroll
        for(int j = 0;j < in_channel ; j += ActK){

            addr1 = vdupq_n_f16(0.0f);
            addr2 = vdupq_n_f16(0.0f);
            addr3 = vdupq_n_f16(0.0f);
            addr4 = vdupq_n_f16(0.0f);
#pragma unroll          
            for(int l = 0 ; l< ActK ;l += 8){

                uint8x16_t vec_as_1 = vld1q_u8(weight + (i*in_channel)/8 + (j + l)*4); //权重1
                uint8x16_t vec_as_2 = vld1q_u8(weight + (i*in_channel)/8 + (j + l)*4 + 16); //权重2

                uint8x16_t vec_a_top_1 = vshrq_n_u8(vec_as_1, 4);
                uint8x16_t vec_a_bot_1 = vandq_u8(vec_as_1, vec_mask);
                // print_uint8x16_t(vec_a_bot_1);
                uint8x16_t vec_a_top_2 = vshrq_n_u8(vec_as_2, 4);
                uint8x16_t vec_a_bot_2 = vandq_u8(vec_as_2, vec_mask);


                uint8x16_t vec_v_bot_low_1 = vqtbl1q_u8(vec_lut[(j+l)/4].val[0], vec_a_bot_1);
                uint8x16_t vec_v_bot_low_2 = vqtbl1q_u8(vec_lut[(j+l)/4].val[1], vec_a_bot_1);
                uint8x16x2_t vec_v_bot_1 = vzipq_u8(vec_v_bot_low_1, vec_v_bot_low_2);

                uint8x16_t vec_v_bot_top_1 = vqtbl1q_u8(vec_lut[(j+l)/4].val[0], vec_a_bot_2);
                uint8x16_t vec_v_bot_top_2 = vqtbl1q_u8(vec_lut[(j+l)/4].val[1], vec_a_bot_2);
                uint8x16x2_t vec_v_bot_2 = vzipq_u8(vec_v_bot_top_1, vec_v_bot_top_2);


                uint8x16_t vec_v_top_low_1 = vqtbl1q_u8(vec_lut[(j+l)/4 + 1].val[0], vec_a_top_1);
                uint8x16_t vec_v_top_low_2 = vqtbl1q_u8(vec_lut[(j+l)/4 + 1].val[1], vec_a_top_1);
                uint8x16x2_t vec_v_top_1 = vzipq_u8(vec_v_top_low_1, vec_v_top_low_2);

                uint8x16_t vec_v_top_top_1 = vqtbl1q_u8(vec_lut[(j+l)/4 + 1].val[0], vec_a_top_2);
                uint8x16_t vec_v_top_top_2 = vqtbl1q_u8(vec_lut[(j+l)/4 + 1].val[1], vec_a_top_2);
                uint8x16x2_t vec_v_top_2 = vzipq_u8(vec_v_top_top_1, vec_v_top_top_2);
                

                addr1 += vreinterpretq_f16_u8(vec_v_bot_1.val[0]);
                addr1 += vreinterpretq_f16_u8(vec_v_top_1.val[0]);

                addr2 += vreinterpretq_f16_u8(vec_v_bot_1.val[1]);
                addr2 += vreinterpretq_f16_u8(vec_v_top_1.val[1]);

                addr3 += vreinterpretq_f16_u8(vec_v_bot_2.val[0]);
                addr3 += vreinterpretq_f16_u8(vec_v_top_2.val[0]);

                addr4 += vreinterpretq_f16_u8(vec_v_bot_2.val[1]);
                addr4 += vreinterpretq_f16_u8(vec_v_top_2.val[1]);


            }

            float16x8_t vec0_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 0);
            float16x8_t vec1_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 8);
            float16x8_t vec2_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 16);
            float16x8_t vec3_w_scale = vld1q_f16(w_scale + i*in_channel/ActK + 32*j/ActK + 24);

            if(j == 0){
                vec_c0 = addr1 * vec0_w_scale ;
                vec_c1 = addr2 * vec1_w_scale;
                vec_c2 = addr3 * vec2_w_scale;
                vec_c3 = addr4 * vec3_w_scale;
            }
            else{
                vec_c0 += addr1 * vec0_w_scale;
                vec_c1 += addr2 * vec1_w_scale;
                vec_c2 += addr3 * vec2_w_scale;
                vec_c3 += addr4 * vec3_w_scale;
            }
        }

        vst1q_f16(out + i,vaddq_f16(vld1q_f16(out + i),vec_c0));
        vst1q_f16(out + i + 8,vaddq_f16(vld1q_f16(out + i + 8),vec_c1));
        vst1q_f16(out + i + 16,vaddq_f16(vld1q_f16(out + i + 16),vec_c2));
        vst1q_f16(out + i + 24,vaddq_f16(vld1q_f16(out + i + 24),vec_c3));
    }

}


bool t_mac_g4_rearrange_weight_normal_impl(int out_channel, int in_channel, uint8_t* weight, 
    float_type*w_scale ,float_type* w_zero,uint8_t* workplace,int bits,int Actk,int n_thread)
{
    //每32 x 8 个作为一组
    int index = 0;
    int chunk_size = (out_channel + n_thread -1 )/n_thread;
    for(int thread = 0; thread < n_thread ;thread ++){
        int start = thread*chunk_size;
        int end = MIN(start + chunk_size,out_channel);
        for(int bit = 0 ;bit < bits ; bit ++){
            uint8_t* weight_start = weight + bit * out_channel*in_channel/8;
            for(int i = start; i< end ; i+=32){
                for(int j = 0 ;j < in_channel/8 ;j++){
                    for(int k = 0; k < 32 ;k++){
                        workplace[index++] = weight_start[(i+k)*in_channel/8 + j];
                    }
                }
            }
        }  
    }


    memcpy(weight,workplace,in_channel*out_channel*bits/8);

    if(index != in_channel * out_channel*bits/ 8){
        std::cout<<"index != in_channel * out_channel"<<std::endl;
        return 0;
    }

    index = 0;
    float_type* temp = (float_type*)workplace; 

    for(int thread = 0;  thread < n_thread ;thread++){
        int start = thread*chunk_size;
        int end = MIN(start + chunk_size,out_channel);
        for(int bit = 0 ;bit < bits ; bit ++){

            float_type* w_scale_start = w_scale + bit * out_channel*in_channel/Actk;
            for(int i = start;i<end ;i+=32){

                for(int j = 0; j< in_channel ;j+=Actk){

                    for(int k = 0; k < 32 ;k++){

                        temp[index++] = w_scale_start[(i+k)*in_channel/Actk + j/Actk];
                    }
                }
            }
        }
    }

    memcpy(w_scale,temp,bits*in_channel*out_channel/Actk*sizeof(float_type));

    if(index != bits*in_channel*out_channel/Actk){
        std::cout<<"index != in_channel * out_channel/Ack"<<std::endl;
        return 0;
    }

    index = 0;
    temp = (float_type*)workplace;
    for(int i = 0;i<out_channel ;i+=32){

        for(int j = 0; j< in_channel ;j+=Actk){

            for(int k = 0; k < 32 ;k++){

                temp[index++] = w_zero[(i+k)*in_channel/Actk + j/Actk];
            }
        }
    }

    memcpy(w_zero,workplace,in_channel*out_channel/Actk*sizeof(float_type));
    
    if(index != in_channel*out_channel/Actk){
        std::cout<<"index != in_channel * out_channel/Ack"<<std::endl;
        return 0;
    }

    return 1;

}

bool t_mac_g4_rearrange_weight_sparse_impl(int out_channel, int in_channel, uint8_t* weight, 
    float_type*w_scale ,float_type* w_zero,uint8_t* workplace,int bits,int Actk,uint32_t*mask_data,uint32_t mask_len)
{

    int base_bit = bits - 1;

    int index = 0;
    uint8_t* weight_start;
    for(int bit = 0 ;bit < base_bit ; bit ++){

         weight_start = weight + bit * out_channel*in_channel/8;
        
        for(int i = 0 ; i < out_channel ; i += 32){

            for(int j = 0 ; j < in_channel/8 ; j ++){

                for(int k = 0; k < 32 ;k++){
                    workplace[index++] = weight_start[(i+k)*in_channel/8 + j];
                }
            }
        }
    }

    int len = mask_len;
    weight_start = weight + base_bit * out_channel*in_channel/8;
    for(int i = 0 ;i < len ;i+=2){

        for(int j = 0 ;j < Actk/8 ;j ++){

            for(int k = 0;k<32;k++){
                workplace[index++] = weight_start[i*2*Actk + k * Actk/8 +j ];
            }
        }

    }

    memcpy(weight,workplace,index);

    if(index != in_channel*out_channel*base_bit/8 + len*2*Actk){
        std::cout<<"index != in_channel*out_channel*base_bit/8 + len*2*Actk"<<std::endl;
        return 0;
    }

    

    index = 0;
    float_type* temp = (float_type*)workplace; 

    for(int bit = 0 ;bit < base_bit ; bit ++){

        float_type* w_scale_start = w_scale + bit * out_channel*in_channel/Actk;
        for(int i = 0;i<out_channel ;i+=32){

            for(int j = 0; j< in_channel ;j+=Actk){

                for(int k = 0; k < 32 ;k++){

                    temp[index++] = w_scale_start[(i+k)*in_channel/Actk + j/Actk];
                }
            }
        }
    }

    memcpy(w_scale,temp,base_bit*in_channel*out_channel/Actk*sizeof(float_type));

    if(index != base_bit*in_channel*out_channel/Actk){
        std::cout<<"index != in_channel * out_channel/Ack"<<std::endl;
        return 0;
    }

    index = 0;
    temp = (float_type*)workplace;
    for(int i = 0;i<out_channel ;i+=32){

        for(int j = 0; j< in_channel ;j+=Actk){

            for(int k = 0; k < 32 ;k++){

                temp[index++] = w_zero[(i+k)*in_channel/Actk + j/Actk];
            }
        }
    }

    memcpy(w_zero,workplace,in_channel*out_channel/Actk*sizeof(float_type));
    
    if(index != in_channel*out_channel/Actk){
        std::cout<<"index != in_channel * out_channel/Ack"<<std::endl;
        return 0;
    }

    return 1;

}

/**
 * @brief: 重新排列权重，wbit = 4的情况
 * @TODO:多线程+SIMD优化
 */
bool t_mac_g4_rearrange_weight_impl(weight_tensor*weight,uint8_t* workplace,int n_thread){


    if(weight->use_sparse){

        return t_mac_g4_rearrange_weight_sparse_impl(weight->out_channel,weight->in_channel,
            (uint8_t*)weight->weight,weight->w_scale,
            weight->w_zero,workplace,weight->bit,weight->groupsize,
            weight->sparse_mask_data,weight->sparse_mask_len);

    }
    else{ 


        return t_mac_g4_rearrange_weight_normal_impl(weight->out_channel,weight->in_channel,
            (uint8_t*)weight->weight,weight->w_scale,
            weight->w_zero,workplace,weight->bit,weight->groupsize,n_thread);
    }



    return 1;
}



void t_mac_g4_float_update_sparse_impl(
    float_type* out,
    uint8_t* weight,
    int in_channel,
    int out_channel,
    uint8_t* lut,
    float_type* w_scale,
    const sparse_mask& mask,
    int weight_order,
    int ActK
){
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);

    int K = in_channel/4;

    uint8x16x2_t vec_lut[K];

#pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = vld2q_u8(lut + k * 32);
    }

    float16x8_t vec_c0, vec_c1, vec_c2, vec_c3;
    float16x8_t addr1, addr2, addr3, addr4;


    int len = mask.len;
    uint32_t* mask_data = mask.data;
#pragma unroll
    for(int i = 0 ; i < len ; i += 2){

        int x = mask_data[i];
        int y = mask_data[i+1];

        addr1 = vdupq_n_f16(0.0f);
        addr2 = vdupq_n_f16(0.0f);
        addr3 = vdupq_n_f16(0.0f);
        addr4 = vdupq_n_f16(0.0f);
#pragma unroll
        for(int l = 0 ; l< ActK ;l += 8){

            uint8x16_t vec_as_1 = vld1q_u8(weight + i*2*ActK + 4*l); //权重1
            uint8x16_t vec_as_2 = vld1q_u8(weight + i*2*ActK + 4*l + 16); //权重2

            uint8x16_t vec_a_top_1 = vshrq_n_u8(vec_as_1, 4);
            uint8x16_t vec_a_bot_1 = vandq_u8(vec_as_1, vec_mask);

            uint8x16_t vec_a_top_2 = vshrq_n_u8(vec_as_2, 4);
            uint8x16_t vec_a_bot_2 = vandq_u8(vec_as_2, vec_mask);


            uint8x16_t vec_v_bot_low_1 = vqtbl1q_u8(vec_lut[(y+l)/4].val[0], vec_a_bot_1);
            uint8x16_t vec_v_bot_low_2 = vqtbl1q_u8(vec_lut[(y+l)/4].val[1], vec_a_bot_1);
            uint8x16x2_t vec_v_bot_1 = vzipq_u8(vec_v_bot_low_1, vec_v_bot_low_2);

            uint8x16_t vec_v_bot_top_1 = vqtbl1q_u8(vec_lut[(y+l)/4].val[0], vec_a_bot_2);
            uint8x16_t vec_v_bot_top_2 = vqtbl1q_u8(vec_lut[(y+l)/4].val[1], vec_a_bot_2);
            uint8x16x2_t vec_v_bot_2 = vzipq_u8(vec_v_bot_top_1, vec_v_bot_top_2);


            uint8x16_t vec_v_top_low_1 = vqtbl1q_u8(vec_lut[(y+l)/4 + 1].val[0], vec_a_top_1);
            uint8x16_t vec_v_top_low_2 = vqtbl1q_u8(vec_lut[(y+l)/4 + 1].val[1], vec_a_top_1);
            uint8x16x2_t vec_v_top_1 = vzipq_u8(vec_v_top_low_1, vec_v_top_low_2);

            uint8x16_t vec_v_top_top_1 = vqtbl1q_u8(vec_lut[(y+l)/4 + 1].val[0], vec_a_top_2);
            uint8x16_t vec_v_top_top_2 = vqtbl1q_u8(vec_lut[(y+l)/4 + 1].val[1], vec_a_top_2);
            uint8x16x2_t vec_v_top_2 = vzipq_u8(vec_v_top_top_1, vec_v_top_top_2);
            

            addr1 += vreinterpretq_f16_u8(vec_v_bot_1.val[0]);
            addr1 += vreinterpretq_f16_u8(vec_v_top_1.val[0]);

            addr2 += vreinterpretq_f16_u8(vec_v_bot_1.val[1]);
            addr2 += vreinterpretq_f16_u8(vec_v_top_1.val[1]);

            addr3 += vreinterpretq_f16_u8(vec_v_bot_2.val[0]);
            addr3 += vreinterpretq_f16_u8(vec_v_top_2.val[0]);

            addr4 += vreinterpretq_f16_u8(vec_v_bot_2.val[1]);
            addr4 += vreinterpretq_f16_u8(vec_v_top_2.val[1]);

        }

        float16x8_t vec0_w_scale = vld1q_f16(w_scale + i/2 * 32 + 0);
        float16x8_t vec1_w_scale = vld1q_f16(w_scale + i/2 * 32 + 8);
        float16x8_t vec2_w_scale = vld1q_f16(w_scale + i/2 * 32 + 16);
        float16x8_t vec3_w_scale = vld1q_f16(w_scale + i/2 * 32 + 24);

        if(i == 0 || x != mask_data[i-2]){
            vec_c0 = addr1 * vec0_w_scale ;
            vec_c1 = addr2 * vec1_w_scale;
            vec_c2 = addr3 * vec2_w_scale;
            vec_c3 = addr4 * vec3_w_scale;
        }
        else{
            vec_c0 += addr1 * vec0_w_scale;
            vec_c1 += addr2 * vec1_w_scale;
            vec_c2 += addr3 * vec2_w_scale;
            vec_c3 += addr4 * vec3_w_scale;
        }

        if(i==len-2 || x != mask_data[i+2]){

            vst1q_f16(out + x,vaddq_f16(vld1q_f16(out + x),vec_c0));
            vst1q_f16(out + x + 8,vaddq_f16(vld1q_f16(out + x + 8),vec_c1));
            vst1q_f16(out + x + 16,vaddq_f16(vld1q_f16(out + x + 16),vec_c2));
            vst1q_f16(out + x + 24,vaddq_f16(vld1q_f16(out + x + 24),vec_c3));
        }
    }
        
}
