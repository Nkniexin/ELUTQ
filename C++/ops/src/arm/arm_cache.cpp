#include<arm_cache.h>

void arm_cache(
    float16_t* k,
    float16_t* v,
    float16_t* kcache,
    float16_t* vcache,
    int num_tokens,
    int head_kv,
    int dhead,
    int* cache_positions
){

    int start = 0;
    int end = num_tokens;

    for (int i = start; i < end; i++) {
        for(int h = 0; h < head_kv; h++) {    
                memcpy(kcache + cache_positions[i] * head_kv * dhead + h * dhead ,k + i * head_kv * dhead + h * dhead , (dhead)*sizeof(float16_t));
                memcpy(vcache + cache_positions[i] * head_kv * dhead + h * dhead ,v + i * head_kv * dhead + h * dhead , (dhead)*sizeof(float16_t));
        }
    }
}