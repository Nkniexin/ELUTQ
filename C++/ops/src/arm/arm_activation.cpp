#include <arm_activation.h>

struct MultiThreadGatedSilu : MultiThreadBaseOp{
    float_type* output;
    float_type* gate;
    float_type* up;
    int ntokens;
    int dimension;

    MultiThreadGatedSilu(float_type* output,
                         float_type* gate,
                         float_type* up,
                        int ntokens,
                        int dimension
                    )
        : output(output),gate(gate),up(up),
        ntokens(ntokens),dimension(dimension){}


    ~MultiThreadGatedSilu() {}

    void Run(){

        for (int i = 0; i < ntokens; i++) {
            for (int j = 0; j < dimension; j += 4) {

                // float16x8_t gate_x_vec = vld1q_f16(gate + i * dimension  + j);
                // float16x8_t up_x_vec = vld1q_f16(up + i * dimension + j );
                // float16x8_t neg_gate_x_vec = vnegq_f16(gate_x_vec);
                // float16x8_t exp_vec = fp16_exp(neg_gate_x_vec);
                // float16x8_t denom_vec = vaddq_f16(vdupq_n_f16(1.0f), exp_vec);
                // float16x8_t sigmoid_val_vec = vdivq_f16(gate_x_vec, denom_vec);
                // float16x8_t result_vec = vmulq_f16(up_x_vec, sigmoid_val_vec);            
                // vst1q_f16(output + i * dimension + j, result_vec);

                float16x4_t gate_f16 = vld1_f16(gate + i * dimension + j);
                float16x4_t up_f16   = vld1_f16(up + i * dimension + j);

                float32x4_t gate_f32 = vcvt_f32_f16(gate_f16);
                float32x4_t up_f32   = vcvt_f32_f16(up_f16);

                float32x4_t neg_gate = vnegq_f32(gate_f32);
                float32x4_t exp_val  = fp32_exp(neg_gate);  
                float32x4_t denom    = vaddq_f32(vdupq_n_f32(1.0f), exp_val);
                float32x4_t sigmoid  = vdivq_f32(gate_f32, denom);
                float32x4_t result   = vmulq_f32(up_f32, sigmoid);

                float16x4_t result_f16 = vcvt_f16_f32(result);
                vst1_f16(output + i * dimension + j, result_f16);

            }
        }
    }
};


void Multithread_gated_silu_impl(
    float_type* output,
    float_type* gate,
    float_type* up,
    int ntokens,
    int dimension,
    int threadnum,
    AliveThreadPool*pool
){

    int chunk_size = (ntokens + threadnum -1) / threadnum;
    std::vector<MultiThreadGatedSilu*>ops;

    for(int i = 0 ;i < threadnum ;i++){

        int start = chunk_size*i;
        int end = std::min(start + chunk_size,ntokens);

        ops.push_back(new MultiThreadGatedSilu(output + start*dimension,gate + start*dimension,
                                            up+start*dimension,end-start,dimension
                                            )
        );

    }

    for(int i = 0;i<threadnum;i++){
        pool->PushOp(i,ops[i]);
    }

    for(int i=0;i<threadnum;i++){
        pool->Wait(i);
        delete ops[i];
    }
}

void gated_silu(
    float_type* output,
    float_type* gate,
    float_type* up,
    int ntokens,
    int dimension,
    int threadnum,
    AliveThreadPool*pool
){
    int start = 0;
    int end = ntokens;

    Multithread_gated_silu_impl(output,gate,up,ntokens,dimension,threadnum,pool);

    
}