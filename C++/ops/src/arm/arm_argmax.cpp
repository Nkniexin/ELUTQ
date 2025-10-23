#include <arm_argmax.h>

void arm_argmax_last(
    int* output,
    const float_type* input,
    int* max_index_temp,
    float_type* max_value_temp,
    int ntokens,  //we actually care about the prediction of last token
    int dimension
){


    //easy version,simply get the index of max_value
    int start = 0;

    int end = dimension;

    int max_index = 0;
    float_type max_value = input[0];

    for(int i = start ;i < end ;i ++){
        if(input[i] > max_value){
            max_index = i;
            max_value = input[(ntokens-1)*dimension+i];
        }
    }


    *output  = max_index;
}