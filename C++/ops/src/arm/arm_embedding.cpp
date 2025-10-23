#include <arm_embedding.h>


void arm_embedding(
    float_type* out,
    int* input_ids,
    float_type* embedding,
    int num_tokens,
    int hidden_size,
    void* unused
) {
    
    int start = 0;
    int end = num_tokens;

    for(int i = start ; i < end ;i ++){
        memcpy(out + i*hidden_size , embedding + input_ids[i] * hidden_size , hidden_size * sizeof(float_type));
    }

}