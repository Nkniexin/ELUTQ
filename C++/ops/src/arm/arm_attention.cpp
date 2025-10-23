#include<arm_attention.h>


struct MultiThreadAttention : MultiThreadBaseOp{
    float16_t* o;
    float16_t* q;
    float16_t* k;
    float16_t* v;
    float* softmax;
    int start;
    int end;
    int ntokens;
    int nkv_tokens;
    int head_q;
    int head_kv;
    int dhead;
    float scale;

    MultiThreadAttention(
                        float16_t* o,
                        float16_t* q,
                        float16_t* k,
                        float16_t* v,
                        int start,
                        int end,
                        int ntokens,
                        int nkv_tokens,
                        int head_q,
                        int head_kv,
                        int dhead,
                        float16_t scale):
                        o(o),q(q),k(k),v(v),start(start),end(end),
                        ntokens(ntokens),nkv_tokens(nkv_tokens),head_q(head_q),
                        head_kv(head_kv),dhead(dhead),scale(scale){
                            softmax = (float*)malloc(nkv_tokens*sizeof(float));
                        }

    ~MultiThreadAttention(){
        free(softmax);
    }

    void Run() {

        for (int i = start; i < end; i++) {

            int valid_nkv_tokens = nkv_tokens - ntokens + i + 1;

            // head_q loop
            for (int h = 0; h < head_q; h++) {
                int h_kv = h * head_kv / head_q;

                // q * k
                for(int token = 0; token < valid_nkv_tokens; token++) {
                    
                    int j = 0; 
                    float16x8_t c_vec = vdupq_n_f16(0.0f);
                    for(; j <= dhead-8; j += 8) {         // 256/32 = 8
                        float16x8_t a_vec = vld1q_f16(q + i * head_q * dhead + h * dhead + j);
                        float16x8_t b_vec = vld1q_f16(k + token * head_kv * dhead + h_kv * dhead + j);
                        c_vec = vfmaq_f16(c_vec, a_vec, b_vec);
                    }

                    float res_sum = fp32_sum(c_vec);
                    for(; j < dhead ; j++){
                        res_sum += q[i*head_q*dhead+h*dhead+j]*k[token*head_kv*dhead+h_kv*dhead+j];
                    }
                    softmax[token] = (res_sum * scale);
                }

                // softmax over softmax_tmp
                int j = 0;
                float softmax_max_val = softmax[0];
                for(int token =  0 ;token < valid_nkv_tokens ;token ++){
                    softmax_max_val = MAX(softmax_max_val , softmax[token]);
                }


                float max_val = softmax_max_val;
                float res_sum =  0.0f;
                for(;j<valid_nkv_tokens;j++){
                    softmax[j] = expf(softmax[j] - max_val);
                    res_sum += (softmax[j]);
                }

                j = 0;
                float32x4_t a_vec = vdupq_n_f32(res_sum);
                for (; j <= valid_nkv_tokens-4; j+=4) {
                    vst1q_f32(softmax + j,vdivq_f32(vld1q_f32(softmax+j),a_vec));
                }

                for( ;j < valid_nkv_tokens; j++ ){
                    softmax[j] = (softmax[j]/res_sum);
                }

                for(int token = 0 ; token < valid_nkv_tokens; token ++){

                    float16x8_t a_vec = vdupq_n_f16(softmax[token]);

                    int j = 0;
                    float16x8_t *out_temp = (float16x8_t*)(o + i * head_q * dhead + h * dhead);
                    for(;j <= dhead - 8 ;j += 8){
                        if(token == 0){
                            out_temp[j/8]  = vdupq_n_f16(0.0f);
                        }
                        
                        out_temp[j/8]  = vfmaq_f16(out_temp[j/8],vld1q_f16(v+token*head_kv*dhead+h_kv*dhead+j),a_vec);
                        
                    }

                    for(;j < dhead ;j ++){
                        if(token == 0){
                            o[i * head_q * dhead + h * dhead + j] = 0;
                        }  
                        o[i * head_q * dhead + h * dhead + j] += (v[token * head_kv * dhead + h_kv * dhead + j]) * (softmax[token]);
                    }
                }
            }
        }
        
    }
};


void MultiThread_attention_impl( //TOOD:长文本时，attention任务不应该是按ntokens切分，因为靠前的token的计算量是很小的
    float16_t* o,   
    float16_t* q,   
    float16_t* k,   
    float16_t* v,   
    float16_t* temp_workplace,
    int ntokens,
    int nkv_tokens,
    int head_q,
    int head_kv,
    int dhead,
    float16_t scale,
    int threadnum,
    AliveThreadPool*pool
){
    int chunk_size = (ntokens + threadnum -1 ) / threadnum;

    std::vector<MultiThreadAttention*>ops;

    for(int i = 0;i<threadnum ;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size,ntokens);

        ops.push_back(new MultiThreadAttention(o,q,k,v,start,end,ntokens,nkv_tokens,head_q,head_kv,dhead,scale));

    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i = 0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }
}

struct MultiThreadAttentionSingle : MultiThreadBaseOp{
    float16_t* o;
    float16_t* q;
    float16_t* k;
    float16_t* v;
    float* softmax;
    int start;
    int end;
    int ntokens;
    int nkv_tokens;
    int head_q;
    int head_kv;
    int dhead;
    float16_t scale;

    MultiThreadAttentionSingle(float16_t* o,
                        float16_t* q,
                        float16_t* k,
                        float16_t* v,
                        int start,
                        int end,
                        int ntokens,
                        int nkv_tokens,
                        int head_q,
                        int head_kv,
                        int dhead,
                        float16_t scale):
                        o(o),q(q),k(k),v(v),start(start),end(end),
                        ntokens(ntokens),nkv_tokens(nkv_tokens),head_q(head_q),
                        head_kv(head_kv),dhead(dhead),scale(scale){
                            softmax = (float*)malloc(nkv_tokens*sizeof(float));
                        }

    ~MultiThreadAttentionSingle(){
        free(softmax);
    }

    void Run(){

        for(int h = start ;h  < end; h++){

            int h_kv =  h * head_kv/head_q;
            //q*k;
            for(int token = 0 ; token < nkv_tokens ;token ++){

                int j = 0;
                float16x8_t c_vec = vdupq_n_f16(0.0f);
                for(;j <= dhead - 8; j+=8){
                    float16x8_t a_vec = vld1q_f16(q + h * dhead + j);
                    float16x8_t b_vec = vld1q_f16(k + token * head_kv * dhead + h_kv * dhead + j);
                    c_vec = vfmaq_f16(c_vec, a_vec, b_vec);
                }

                softmax[token] = fp32_sum(c_vec);

                for(;j < dhead ;j ++){ 
                    softmax[token]  += q[h*dhead+j]*k[token*head_kv*dhead + h_kv * dhead + j];
                }

                softmax[token] *= scale;
                
            }

            int j = 0;
            float softmax_max_val = softmax[0];
            for(int token =  0 ;token < nkv_tokens ;token ++){
                softmax_max_val = std::max(softmax_max_val , softmax[token]);
            }
            float res_sum = 0;

            float32x4_t max_val = vdupq_n_f32(softmax_max_val);
            float32x4_t acc = vdupq_n_f32(0.0f);
            for(;j<nkv_tokens;j++){
                softmax[j] = (expf((softmax[j]) - softmax_max_val));
                res_sum += (softmax[j]);
            }

            j = 0;
            float32x4_t a_vec = vdupq_n_f32(res_sum);
            for (; j <= nkv_tokens-4; j+=4) {
                vst1q_f32(softmax + j,vdivq_f32(vld1q_f32(softmax+j),a_vec));
            }

            for( ;j < nkv_tokens; j++ ){
                softmax[j] = (softmax[j]/res_sum);
            }
            
            for(int token = 0 ; token < nkv_tokens; token ++){

                    float16x8_t a_vec = vdupq_n_f16(softmax[token]);

                    int j = 0;
                    float16x8_t *out_temp = (float16x8_t*)(o +  h * dhead);
                    for(;j <= dhead - 8 ;j += 8){
                        if(token == 0){
                            out_temp[j/8]  = vdupq_n_f16(0.0f);
                        }
                        
                        out_temp[j/8]  = vfmaq_f16(out_temp[j/8],vld1q_f16(v+token*head_kv*dhead+h_kv*dhead+j),a_vec);
                        
                    }

                    for(;j < dhead ;j ++){
                        if(token == 0){
                            o[ h * dhead + j] = 0;
                        }  
                        o[h * dhead + j] += (v[token * head_kv * dhead + h_kv * dhead + j]) * (softmax[token]);
                    }
                }
            
        }

    }


};

void MultiThread_attention_single_impl(
    float16_t* o,   
    float16_t* q,   
    float16_t* k,   
    float16_t* v,   
    float16_t* temp_workplace,
    int ntokens,
    int nkv_tokens,
    int head_q,
    int head_kv,
    int dhead,
    float16_t scale,
    int threadnum,
    AliveThreadPool*pool
){

    int chunk_size = (head_q + threadnum - 1) / threadnum;

    std::vector<MultiThreadAttentionSingle*>ops;
    for(int i = 0;i<threadnum;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size,head_q);

        ops.push_back(new MultiThreadAttentionSingle(o,q,k,v,start,end,ntokens,nkv_tokens,head_q,head_kv,dhead,scale));

    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i = 0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }

}


void arm_attention(
    float16_t* o,   // ntokens, nhead, dhead
    float16_t* q,   // ntokens, nhead, dhead
    float16_t* k,   // nkv_tokens, nhead, dhead
    float16_t* v,   // nkv_tokens, nhead, dhead
    float16_t* temp_workplace,
    int ntokens,
    int nkv_tokens,
    int head_q,
    int head_kv,
    int dhead,
    float scale,
    int threadnum,
    AliveThreadPool*pool
) {
    
    if(ntokens == 1) MultiThread_attention_single_impl(o,q,k,v,temp_workplace,ntokens,nkv_tokens,head_q,head_kv,dhead,scale,threadnum,pool);
    else MultiThread_attention_impl(o,q,k,v,temp_workplace,ntokens,nkv_tokens,head_q,head_kv,dhead,scale,threadnum,pool);

}