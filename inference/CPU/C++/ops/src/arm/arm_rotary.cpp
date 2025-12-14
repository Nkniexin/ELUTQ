#include <arm_rotary.h>


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
){

    int start = 0;
    int end = num_tokens;

    float_type* q_start = x;
    // std::cout<<"head_q : "<<head_q<<std::endl;
    // std::cout<<"dhead : "<<dhead<<std::endl;
    // std::cout<<"q_start :"<<x[6 * (head_q ) * dhead]<<std::endl;
    float_type* k_start = x + num_tokens*head_q*dhead;
    float_type* v_start = x + num_tokens*head_q*dhead + num_tokens*head_kv*dhead;
    for (int i = start; i < end; i++) {
        for(int h = 0; h < head_q; h++) {
            // q
            float_type* qpointer = q_start + i * (head_q ) * dhead + h * dhead;
            for(int j = 0; j < dhead / 2; j++) {
                float inv_freq = pid[i] * powf(rotary_base, - (float)(j * 2) / (float)dhead);
                float q1 = float(qpointer[j]);
                // if(std::isnan(q1)){
                //     std::cout<<i<<" "<<h<<" "<<j<<std::endl; 
                //     std::cout<<"111111"<<std::endl;
                //     exit(-1);
                // }
                float q2 = float(qpointer[j + dhead / 2]);

                float new_q1 = (q1 * cosf(inv_freq) - q2 * sinf(inv_freq));
                float new_q2 = (q1 * sinf(inv_freq) + q2 * cosf(inv_freq));
                float_type* q_out = reinterpret_cast<float_type*>(q);
                q_out[i * head_q * dhead + h * dhead + j] = (float_type)new_q1;
                
                q_out[i * head_q * dhead + h * dhead + j + dhead / 2] = new_q2;
                // if(std::isnan(q2)){
                //     std::cout<<"111111"<<std::endl;
                // }
            }
        }
    }

    for(int i = start ;i < end ;i++){
        for(int h = 0; h < head_kv; h++) {
            // k
            float_type* kpointer = (float_type*)k_start + i * (head_kv) * dhead + h * dhead;
            for(int j = 0; j < dhead / 2; j++) {
                float inv_freq = pid[i] * powf(rotary_base, - (float)(j * 2) / (float)dhead);
                float k1 = float(kpointer[j]);
                float k2 = float(kpointer[j + dhead / 2]);
                float new_k1 = (k1 * cosf(inv_freq) - k2 * sinf(inv_freq));
                float new_k2 = (k1 * sinf(inv_freq) + k2 * cosf(inv_freq));
                float_type* k_out = reinterpret_cast<float_type*>(k);
                k_out[i * head_kv * dhead + h * dhead + j] = new_k1;
                // if(std::isnan(k1)){
                //     std::cout<<"111111"<<std::endl;
                // }
                k_out[i * head_kv * dhead + h * dhead + j + dhead / 2] = new_k2;
                // if(std::isnan(k2)){
                //     std::cout<<"111111"<<std::endl;
                // }
            }
        }
    }

    for (int i = start ; i < end ;i++){
        for(int h = 0; h < head_kv; h++) {
            // v
            for(int j = 0; j < dhead; j++)
            {
                v[i * head_kv * dhead + h * dhead + j] = \
                (
                    (v_start[i * head_kv * dhead + h * dhead + j])
                );
            }
        }
    }
}