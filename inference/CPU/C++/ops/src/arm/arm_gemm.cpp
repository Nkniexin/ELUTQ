
#include <arm_gemm.h>


struct MultiThreadGemmFP : MultiThreadBaseOp{
    int m;
    int k;
    int n; 
    float_type*input;
    float_type *weight;
    float_type *bias;
    float_type* output;

    MultiThreadGemmFP(int m,
                    int k,
                    int n, 
                    float_type*input,
                    float_type *weight,
                    float_type *bias,
                    float_type* output)
                : m(m),k(k),n(n),input(input),weight(weight),bias(bias),output(output){}

    ~MultiThreadGemmFP(){}

    void Run(){

        for(int i = 0;i < m;i++){

            for(int j = 0;j < n;j ++){

                float16x8_t sum = vdupq_n_f16(0.0f);
                for(int l = 0;l < k;l += 8){
                    sum = vfmaq_f16(
                        sum,
                        vld1q_f16(input + i*k + l),
                        vld1q_f16(weight + j*k + l)
                    );
                }

                output[i*n + j] = fp16_sum(sum);
                

            }
        }

        if(bias!=nullptr){
            for(int i = 0 ;i < m;i++){

                for(int j = 0;j<n;j++){
                    output[i*n+j] += bias[j];
                }
            }
        }   

    }

};


void MultiThread_arm_gemm_fp16_impl(
    int m,
    int k,
    int n, 
    float_type*input,
    float_type *weight,
    float_type* bias,
    float_type* output,
    int threadnum,
    AliveThreadPool*pool
){

    int chunk_size = (m + threadnum -1 ) / threadnum;

    std::vector<MultiThreadGemmFP*>ops;
    for(int i = 0;i<threadnum ;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size,m);

        ops.push_back(new MultiThreadGemmFP(end - start,k,n,input + start*k,weight,bias,output+start*n));

    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i = 0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }
}


struct MultiThreadGemmW_lowbit_Activation_FP : MultiThreadBaseOp{
    int m;
    int k;
    int n; 
    float_type*input;
    uint8_t *weight;
    float_type *w_scale;
    float_type* w_zero;
    float_type*bias;
    float_type* output;
    int Actk;
    int bit;

    float_type *input_sum = nullptr;
    uint8_t*lut = nullptr;

    MultiThreadGemmW_lowbit_Activation_FP(int m,
                                        int k,
                                        int n, 
                                        float_type*input,
                                        uint8_t *weight,
                                        float_type *w_scale,
                                        float_type* w_zero,
                                        float_type*bias,
                                        float_type* output,
                                        int Actk,
                                        int bit
                                    ):
                            m(m),k(k),n(n),input(input),
                            weight(weight),w_scale(w_scale),
                            w_zero(w_zero),bias(bias),output(output),Actk(Actk),bit(bit){
                                input_sum = (float_type*)malloc(k/Actk*sizeof(float_type)); //TODO:优化，不使用malloc,直接从malloc分配
                                lut = (uint8_t*)malloc(k*8*sizeof(uint8_t));
                            }

    ~MultiThreadGemmW_lowbit_Activation_FP(){
        free(lut);
        free(input_sum);
    }


    void Run(){

        for(int row = 0; row < m ;row ++ ){

            for(int i = 0;i < k;i += Actk){
                input_sum[i/Actk] = 0.0f;
                for(int j = 0;j < Actk;j ++){
                    input_sum[i/Actk]+=input[row*k + i + j];
                }
            }

            int cnt_per_row = k/Actk;

            float16x8_t out1,out2,out3,out4;

            for(int i = 0;i < n ;i += 32){
                out1 = vdupq_n_f16(0.0f);
                out2 = vdupq_n_f16(0.0f);
                out3 = vdupq_n_f16(0.0f);
                out4 = vdupq_n_f16(0.0f);
                for(int j = 0;j < cnt_per_row ;j++){

                    float16x8_t temp_input_sum = vdupq_n_f16(input_sum[j]);


                    float16x8_t w_zero1 = vld1q_f16(w_zero + i*k/Actk + 32*j);
                    float16x8_t w_zero2 = vld1q_f16(w_zero + i*k/Actk + 32*j + 8);
                    float16x8_t w_zero3 = vld1q_f16(w_zero + i*k/Actk + 32*j + 16);
                    float16x8_t w_zero4 = vld1q_f16(w_zero + i*k/Actk + 32*j + 24);

                    out1 += temp_input_sum*w_zero1;
                    out2 += temp_input_sum*w_zero2;
                    out3 += temp_input_sum*w_zero3;
                    out4 += temp_input_sum*w_zero4;
                }

                vst1q_f16(output + row*n + i,out1);
                vst1q_f16(output + row*n + i + 8,out2);
                vst1q_f16(output + row*n + i + 16,out3);
                vst1q_f16(output + row*n + i + 24,out4);

            }
        }

        for(int row = 0; row < m ;row ++ ){

            t_mac_g4_groupsize32_calculate_float_lut_impl(
                k,input + row*k,(uint8_t*)lut,Actk
            );

            for(int i = 0 ;i< bit ;i++){
                int weight_bias = i * k * n / 8;
                int w_scale_bias = i * k * n / Actk;
                t_mac_g4_float_update_impl(
                    output + row*n,
                    (uint8_t*)weight + weight_bias,
                    k,n,(uint8_t*)lut,w_scale + w_scale_bias,i,Actk
                );
            }
        }

        if(bias!=nullptr){

            for(int i = 0 ;i<m;i++){
                for(int j = 0;j<n;j++){
                    output[i*n+j] += bias[j];
                }
            }
        }
    }
    
};

struct MultiThreadGemvW_lowbit_Activation_FP : MultiThreadBaseOp{
    int m;
    int k;
    int n; 
    float_type*input;
    uint8_t *weight;
    float_type *w_scale;
    float_type *w_zero;
    float_type*bias;
    float_type* output;
    int Actk;
    int bit;

    float_type *input_sum = nullptr;
    uint8_t*lut = nullptr;

    MultiThreadGemvW_lowbit_Activation_FP(int m,
                                        int k,
                                        int n, 
                                        float_type*input,
                                        uint8_t *weight,
                                        float_type *w_scale,
                                        float_type* w_zero,
                                        float_type*bias,
                                        float_type* output,
                                        int Actk,
                                        int bit,
                                        float_type* input_sum,
                                        uint8_t*lut
                                    ):
                            m(m),k(k),n(n),input(input),
                            weight(weight),w_scale(w_scale),
                            w_zero(w_zero),bias(bias),output(output),Actk(Actk),bit(bit)
                            ,input_sum(input_sum),lut(lut){
                            }

    ~MultiThreadGemvW_lowbit_Activation_FP(){}


    void Run(){

        for(int i = 0;i < k;i += Actk){
            input_sum[i/Actk] = 0.0f;
            for(int j = 0;j < Actk;j ++){
                input_sum[i/Actk]+=input[i + j];
            }
        }

        int cnt_per_row = k/Actk;

        float16x8_t out1,out2,out3,out4;

        for(int i = 0;i < n ;i += 32){
            out1 = vdupq_n_f16(0.0f);
            out2 = vdupq_n_f16(0.0f);
            out3 = vdupq_n_f16(0.0f);
            out4 = vdupq_n_f16(0.0f);
            for(int j = 0;j < cnt_per_row ;j++){

                float16x8_t temp_input_sum = vdupq_n_f16(input_sum[j]);

                float16x8_t w_zero1 = vld1q_f16(w_zero + i*k/Actk + 32*j);
                float16x8_t w_zero2 = vld1q_f16(w_zero + i*k/Actk + 32*j + 8);
                float16x8_t w_zero3 = vld1q_f16(w_zero + i*k/Actk + 32*j + 16);
                float16x8_t w_zero4 = vld1q_f16(w_zero + i*k/Actk + 32*j + 24);

                out1 += temp_input_sum*w_zero1;
                out2 += temp_input_sum*w_zero2;
                out3 += temp_input_sum*w_zero3;
                out4 += temp_input_sum*w_zero4;
            }

            vst1q_f16(output  + i,out1);
            vst1q_f16(output  + i + 8,out2);
            vst1q_f16(output  + i + 16,out3);
            vst1q_f16(output  + i + 24,out4);

        }
        
        t_mac_g4_groupsize32_calculate_float_lut_impl(
            k,input ,(uint8_t*)lut,Actk
        );

        for(int i = 0 ;i< bit ;i++){
            int weight_bias = i * k * n / 8;
            int w_scale_bias = i * k *n / Actk;
            t_mac_g4_float_update_impl(
                output,
                (uint8_t*)weight + weight_bias,
                k,n,(uint8_t*)lut,w_scale + w_scale_bias,i,Actk
            );
        }
        

        if(bias!=nullptr){
            for(int j = 0;j<n;j++){
                output[j] += bias[j];
            }
        }
    }
    
};

void MultiThread_arm_gemv_w_lowbit_activation_fp_impl(
    int m,
    int k,
    int n, 
    float_type*input,
    uint8_t *weight,
    float_type *w_scale,
    float_type* w_zero, 
    float_type*bias,
    float_type* output,
    void* temp_workplace,
    int Actk,
    int bit,
    int threadnum,
    AliveThreadPool*pool
){

    int chunk_size = (n + threadnum -1 ) / threadnum;

    std::vector<MultiThreadGemvW_lowbit_Activation_FP*>ops;

    float_type* input_sum_workplace = (float_type*)temp_workplace;
    uint8_t *lut_workplace = (uint8_t*)(temp_workplace + threadnum*k/Actk*sizeof(float_type));

    for(int i = 0;i<threadnum ;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size ,n);

        uint8_t* weight_start = weight + bit*start*k/8;
        float_type* w_scale_start = w_scale + bit*start*k/Actk;
        float_type* w_zero_start = w_zero + start*k/Actk;
        float_type* bias_start = nullptr;
        if(bias !=nullptr)bias_start = bias + start;

        ops.push_back(new MultiThreadGemvW_lowbit_Activation_FP(
            m,k,end - start,input,weight_start,w_scale_start,w_zero_start,bias_start,output + start,Actk,bit,input_sum_workplace + i*k/Actk, lut_workplace + k*8
        ));
    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i = 0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }
}

void MultiThread_arm_gemm_w_lowbit_activation_fp_impl(
    int m,
    int k,
    int n, 
    float_type*input,
    uint8_t *weight,
    float_type *w_scale,
    float_type* w_zero, 
    float_type*bias,
    float_type* output,
    void* temp_workplace,
    int Actk,
    int bit,
    int threadnum,
    AliveThreadPool*pool
){

   
    for(int i = 0;i < m ;i++){
        MultiThread_arm_gemv_w_lowbit_activation_fp_impl(1,k,n,input + i*k , (uint8_t*)weight,
                                                    w_scale,w_zero,bias,output + i*n,
                                                    temp_workplace,
                                                    Actk,bit,threadnum,pool);
    }

}


struct MultiThreadGemmW_lowbit_Activation_FP_Sparse : MultiThreadBaseOp{
    int m;
    int k;
    int n; 
    float_type*input;
    uint8_t *weight;
    float_type *w_scale;
    float_type* w_zero;
    float_type*bias;
    float_type* output;
    sparse_mask mask;
    int Actk;
    int bit;


    float_type *input_sum = nullptr;
    uint8_t*lut = nullptr;

    MultiThreadGemmW_lowbit_Activation_FP_Sparse(int m,
                                        int k,
                                        int n, 
                                        float_type*input,
                                        uint8_t *weight,
                                        float_type *w_scale,
                                        float_type* w_zero,
                                        float_type*bias,
                                        float_type* output,
                                        int Actk,
                                        int bit,
                                        sparse_mask mask
                                    ):
                            m(m),k(k),n(n),input(input),
                            weight(weight),w_scale(w_scale),
                            w_zero(w_zero),bias(bias),output(output),Actk(Actk),bit(bit),mask(mask){
                                input_sum = (float_type*)malloc(k/Actk*sizeof(float_type)); //TODO:优化，不使用malloc,直接从malloc分配
                                lut = (uint8_t*)malloc(k*8*sizeof(uint8_t));
                            }

    ~MultiThreadGemmW_lowbit_Activation_FP_Sparse(){
        free(lut);
        free(input_sum);
    }


    void Run(){

        for(int row = 0; row < m ;row ++ ){

            for(int i = 0;i < k;i += Actk){
                input_sum[i/Actk] = 0.0f;
                for(int j = 0;j < Actk;j ++){
                    input_sum[i/Actk]+=input[row*k + i + j];
                }
            }

            int cnt_per_row = k/Actk;

            float16x8_t out1,out2,out3,out4;

            for(int i = 0;i < n ;i += 32){
                out1 = vdupq_n_f16(0.0f);
                out2 = vdupq_n_f16(0.0f);
                out3 = vdupq_n_f16(0.0f);
                out4 = vdupq_n_f16(0.0f);
                for(int j = 0;j < cnt_per_row ;j++){

                    float16x8_t temp_input_sum = vdupq_n_f16(input_sum[j]);


                    float16x8_t w_zero1 = vld1q_f16(w_zero + i*k/Actk + 32*j);
                    float16x8_t w_zero2 = vld1q_f16(w_zero + i*k/Actk + 32*j + 8);
                    float16x8_t w_zero3 = vld1q_f16(w_zero + i*k/Actk + 32*j + 16);
                    float16x8_t w_zero4 = vld1q_f16(w_zero + i*k/Actk + 32*j + 24);

                    out1 += temp_input_sum*w_zero1;
                    out2 += temp_input_sum*w_zero2;
                    out3 += temp_input_sum*w_zero3;
                    out4 += temp_input_sum*w_zero4;
                }

                vst1q_f16(output + row*n + i,out1);
                vst1q_f16(output + row*n + i + 8,out2);
                vst1q_f16(output + row*n + i + 16,out3);
                vst1q_f16(output + row*n + i + 24,out4);

            }
        }

        for(int row = 0; row < m ;row ++ ){

            t_mac_g4_groupsize32_calculate_float_lut_impl(
                k,input + row*k,(uint8_t*)lut,Actk
            );

            for(int i = 0 ;i< bit ;i++){
                int weight_bias = i * k * n / 8;
                int w_scale_bias = i * k * n / Actk;
                if(i==bit - 1){
                    
                    t_mac_g4_float_update_sparse_impl(
                        output + row*n,
                        weight + weight_bias,
                        k,n,(uint8_t*)lut,w_scale + w_scale_bias,mask,i,Actk
                    );
                }
                else{
                    t_mac_g4_float_update_impl(
                        output + row*n,
                        weight + weight_bias,
                        k,n,(uint8_t*)lut,w_scale + w_scale_bias,i,Actk
                    );
                }
            }
        }

        if(bias!=nullptr){

            for(int i = 0 ;i<m;i++){
                for(int j = 0;j<n;j++){
                    output[i*n+j] += bias[j];
                }
            }
        }
    }
    
};


void MultiThread_arm_gemm_w_lowbit_activation_fp_Sparse_impl(
    int m,
    int k,
    int n, 
    float_type*input,
    uint8_t *weight,
    float_type *w_scale,
    float_type* w_zero, 
    float_type*bias,
    uint32_t* mask_data,
    uint32_t mask_len,
    float_type* output,
    void* temp_workplace,
    int Actk,
    int bit,
    int threadnum,
    AliveThreadPool*pool
){
    int chunk_size = (m + threadnum -1 ) / threadnum;

    std::vector<MultiThreadGemmW_lowbit_Activation_FP_Sparse*>ops;

    sparse_mask mask;
    mask.data = mask_data;
    mask.len = mask_len;

    for(int i = 0;i<threadnum ;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size ,m);

        ops.push_back(new MultiThreadGemmW_lowbit_Activation_FP_Sparse(
            end-start,k,n,input+start*k,weight,w_scale,w_zero,bias,output + start*n,Actk,bit,mask
        ));
    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i = 0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }

}


struct MultiThreadGemvFP : MultiThreadBaseOp{
    int m;
    int k;
    int n; 
    float_type*input;
    float_type *weight;
    float_type *bias;
    float_type* output;

    
    MultiThreadGemvFP(int m,
                        int k,
                        int n, 
                        float_type*input,
                        float_type *weight,
                        float_type *bias,
                        float_type* output
                        ):m(m),k(k),n(n),input(input),weight(weight),bias(bias),output(output){}

    ~MultiThreadGemvFP(){}


    void Run(){

        for(int j = 0;j < n;j ++){

            float16x8_t sum1 = vdupq_n_f16(0.0f);
            // float16x8_t sum2 = vdupq_n_f16(0.0f);
            // float16x8_t sum3 = vdupq_n_f16(0.0f);
            // float16x8_t sum4 = vdupq_n_f16(0.0f);
            for(int l = 0;l < k;l += 32){
                sum1 = vfmaq_f16(
                    sum1,
                    vld1q_f16(input + l),
                    vld1q_f16(weight + j*k + l)
                );

                sum1 = vfmaq_f16(
                    sum1,
                    vld1q_f16(input + l + 8),
                    vld1q_f16(weight + j*k + l + 8)
                );

                sum1 = vfmaq_f16(
                    sum1,
                    vld1q_f16(input + l + 16),
                    vld1q_f16(weight + j*k + l + 16)
                );

                sum1 = vfmaq_f16(
                    sum1,
                    vld1q_f16(input + l + 24),
                    vld1q_f16(weight + j*k + l + 24)
                );
            }

            output[j] = fp16_sum(sum1);

            if(bias != nullptr )output[j]+=bias[j];
        }
    }
    
};

void MultiThread_arm_gemv_fp16_impl(
    int m,
    int k,
    int n, 
    float_type*input,
    float_type *weight,
    float_type* bias,
    float_type* output,
    int threadnum,
    AliveThreadPool*pool
){

    int chunk_size = (n + threadnum -1 ) / threadnum;

    

    std::vector<MultiThreadGemvFP*>ops;
    for(int i = 0;i<threadnum ;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size,n);

        ops.push_back(new MultiThreadGemvFP(m,k,end - start,input,weight + start*k,bias==nullptr?nullptr:bias + start,output+start));

    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i = 0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }


}







struct MultiThreadGemvW_lowbit_Activation_FP_Sparse : MultiThreadBaseOp{
    int m;
    int k;
    int n; 
    float_type*input;
    uint8_t *weight;
    float_type *w_scale;
    float_type* w_zero;
    float_type*bias;
    float_type* output;

    sparse_mask mask;
    int Actk;
    int bit;

    int start;
    int end;
    int mask_start;
    int mask_end;

    float_type *input_sum = nullptr;
    uint8_t*lut = nullptr;

    MultiThreadGemvW_lowbit_Activation_FP_Sparse(int m,
                                        int k,
                                        int n, 
                                        int start,
                                        int end,
                                        int mask_start,
                                        int mask_end,
                                        float_type*input,
                                        uint8_t *weight,
                                        float_type *w_scale,
                                        float_type* w_zero,
                                        float_type*bias,
                                        sparse_mask mask,
                                        float_type* output,
                                        int Actk,
                                        int bit,
                                        float_type* input_sum,
                                        uint8_t*lut
                                    ):
                            m(m),k(k),n(n),start(start),end(end),input(input),
                            weight(weight),w_scale(w_scale),
                            w_zero(w_zero),bias(bias),output(output),Actk(Actk),bit(bit)
                            ,input_sum(input_sum),lut(lut),mask(mask),mask_start(mask_start),mask_end(mask_end){
                            }

    ~MultiThreadGemvW_lowbit_Activation_FP_Sparse(){}


    void Run(){

        

        for(int i = 0;i < k;i += Actk){
            input_sum[i/Actk] = 0.0f;
            for(int j = 0;j < Actk;j ++){
                input_sum[i/Actk]+=input[i + j];
            }
        }

        int cnt_per_row = k/Actk;

        float16x8_t out1,out2,out3,out4;

        for(int i = start;i < end ;i += 32){
            out1 = vdupq_n_f16(0.0f);
            out2 = vdupq_n_f16(0.0f);
            out3 = vdupq_n_f16(0.0f);
            out4 = vdupq_n_f16(0.0f);
            for(int j = 0;j < cnt_per_row ;j++){

                float16x8_t temp_input_sum = vdupq_n_f16(input_sum[j]);


                float16x8_t w_zero1 = vld1q_f16(w_zero + i*k/Actk + 32*j);
                float16x8_t w_zero2 = vld1q_f16(w_zero + i*k/Actk + 32*j + 8);
                float16x8_t w_zero3 = vld1q_f16(w_zero + i*k/Actk + 32*j + 16);
                float16x8_t w_zero4 = vld1q_f16(w_zero + i*k/Actk + 32*j + 24);

                out1 += temp_input_sum*w_zero1;
                out2 += temp_input_sum*w_zero2;
                out3 += temp_input_sum*w_zero3;
                out4 += temp_input_sum*w_zero4;
            }

            vst1q_f16(output  + i,out1);
            vst1q_f16(output  + i + 8,out2);
            vst1q_f16(output  + i + 16,out3);
            vst1q_f16(output  + i + 24,out4);

        }
        

        

        t_mac_g4_groupsize32_calculate_float_lut_impl(
            k,input ,(uint8_t*)lut,Actk
        );

        for(int i = 0 ;i< bit ;i++){
            
            

            if(i==bit - 1){
                sparse_mask mask_temp;
                mask_temp.len = mask_end - mask_start;
                mask_temp.data = mask.data + mask_start; 
                int weight_bias = i * k * n / 8 + 2*mask_start*Actk;
                int w_scale_bias = i * k * n / Actk + mask_start*16;
                t_mac_g4_float_update_sparse_impl(
                        output,
                        weight + weight_bias ,
                        k,end - start,(uint8_t*)lut,w_scale + w_scale_bias,mask_temp,i,Actk
                    );
            }
            else{
                int weight_bias = i * k * n / 8 + start*k/8;
                int w_scale_bias = i * k * n / Actk + start*k/Actk;
                t_mac_g4_float_update_impl(
                    output + start,
                    (uint8_t*)weight + weight_bias,
                    k,end - start,(uint8_t*)lut,w_scale + w_scale_bias,i,Actk
                );
            }
        }
        

        if(bias!=nullptr){
            for(int j = start;j<end;j++){
                output[j] += bias[j];
            }
        }
    }
    
};


void MultiThread_arm_gemv_w_lowbit_activation_fp_Sparse_impl(
    int m,
    int k,
    int n, 
    float_type*input,
    uint8_t *weight,
    float_type *w_scale,
    float_type* w_zero, 
    float_type*bias,
    uint32_t* mask_data,
    uint32_t mask_len,
    float_type* output,
    void* temp_workplace,
    int Actk,
    int bit,
    int threadnum,
    AliveThreadPool*pool
){
    int chunk_size = (n + threadnum -1 ) / threadnum;

    std::vector<MultiThreadGemvW_lowbit_Activation_FP_Sparse*>ops;

    float_type* input_sum_workplace = (float_type*)temp_workplace;
    uint8_t *lut_workplace = (uint8_t*)(temp_workplace + threadnum*k/Actk*sizeof(float_type));

    sparse_mask mask;
    mask.data = mask_data;
    mask.len =mask_len; 
    for(int i = 0;i<threadnum ;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size ,n);

        uint32_t* mask_data_thread_start = mask_data;
        uint32_t* mask_data_thread_end = mask_data;
        
        while(*mask_data_thread_start < start && mask_data_thread_start < mask_data + mask_len)mask_data_thread_start+=2; //TODO:优化设计
        while(*mask_data_thread_end <end && mask_data_thread_end < mask_data + mask_len)mask_data_thread_end+=2;
        
        int mask_start = mask_data_thread_start - mask_data;
        int mask_end = mask_data_thread_end - mask_data;
        ops.push_back(new MultiThreadGemvW_lowbit_Activation_FP_Sparse(
            m,k,n,start,end,mask_start,mask_end,input,weight,w_scale,w_zero,bias,mask,output,Actk,bit,input_sum_workplace + i*k/Actk, lut_workplace + k*8
        ));
    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i = 0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }

}


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
){


    if(weight->bit == 16){
        if(m == 1)MultiThread_arm_gemv_fp16_impl(m,k,n,input,(float_type*)weight->weight,weight->bias,output,threadnum,pool);
        else MultiThread_arm_gemm_fp16_impl(m,k,n,input,(float_type*)weight->weight,weight->bias,output,threadnum,pool);
    }else if(weight->bit == 8){

        //TODO
    }else if(weight->bit <=4){

        if(m == 1){

            if(!weight->use_sparse){
                MultiThread_arm_gemv_w_lowbit_activation_fp_impl(m,k,n,input,(uint8_t*)weight->weight,
                                                    weight->w_scale,weight->w_zero,weight->bias,output,
                                                    temp_workplace,
                                                    weight->groupsize,weight->bit,threadnum,pool);
            }
            else{
                MultiThread_arm_gemv_w_lowbit_activation_fp_Sparse_impl(m,k,n,input,(uint8_t*)weight->weight,
                                                    weight->w_scale,weight->w_zero,weight->bias,weight->sparse_mask_data,
                                                    weight->sparse_mask_len,output,
                                                    temp_workplace,
                                                    weight->groupsize,weight->bit,threadnum,pool);
            }
        }
        else{
            if(!weight->use_sparse){
                MultiThread_arm_gemm_w_lowbit_activation_fp_impl(m,k,n,input,(uint8_t*)weight->weight,
                                                    weight->w_scale,weight->w_zero,weight->bias,output,
                                                    temp_workplace,
                                                    weight->groupsize,weight->bit,threadnum,pool);
            }
            else{
                MultiThread_arm_gemm_w_lowbit_activation_fp_Sparse_impl(m,k,n,input,(uint8_t*)weight->weight,
                                                    weight->w_scale,weight->w_zero,weight->bias,weight->sparse_mask_data,
                                                    weight->sparse_mask_len,output,
                                                    temp_workplace,
                                                    weight->groupsize,weight->bit,threadnum,pool);
            }
        }
    }
}

