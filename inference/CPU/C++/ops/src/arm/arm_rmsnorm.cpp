#include<arm_rmsnorm.h>


struct MultiThreadRMSNorm : MultiThreadBaseOp{
    float16_t* output;
    float16_t* vals;
    float16_t* gamma;
    float epsilon;
    int rows;
    int elems_per_row;

    MultiThreadRMSNorm(float16_t* output,
                        float16_t* vals,
                        float16_t* gamma,
                        float epsilon,
                        int rows,
                        int elems_per_row
                     ):output(output),vals(vals)
                     ,gamma(gamma),epsilon(epsilon),rows(rows),elems_per_row(elems_per_row)
                     {}


    ~MultiThreadRMSNorm(){}

    void Run(){

        for (int i = 0; i < rows ;i++) {
            float sum = 0.0;
            for (int j = 0; j < elems_per_row; j++) { 
                float val = (vals[i * elems_per_row + j]);
                sum += val * val;
            }
            float rms = sqrtf(sum / elems_per_row + epsilon);
            for (int j = 0; j < elems_per_row; j++) {
                float val = (vals[i * elems_per_row + j]);
                float normed = val * float(gamma[j]) / rms ;
                output[i * elems_per_row + j] = float16_t(normed);  
            }
        }
    }
};

struct MultiThreadPreRMSNorm : MultiThreadBaseOp{
    float16_t* output;
    float16_t* res_output;
    const float16_t* vals;
    const float16_t* residual;
    const float16_t* gamma;
    float epsilon;
    int rows;
    int elems_per_row;

    MultiThreadPreRMSNorm(float16_t* output,
                        float16_t* res_output,
                        const float16_t* vals,
                        const float16_t* residual,
                        const float16_t* gamma,
                        float epsilon,
                        int rows,
                        int elems_per_row
                     ):output(output),res_output(res_output),vals(vals),residual(residual)
                     ,gamma(gamma),epsilon(epsilon),rows(rows),elems_per_row(elems_per_row)
                     {}


    ~MultiThreadPreRMSNorm(){}

    void Run(){

        for (int i = 0; i < rows; i++) {
            float sum = 0.0;
            for (int j = 0; j < elems_per_row; j++) {
                float val = float(vals[i * elems_per_row + j]) + float(residual[i * elems_per_row + j]);
                sum += val * val;
                res_output[i * elems_per_row + j] = float_type(val);
            }
            float rms = sqrtf(sum / elems_per_row + epsilon);
            for (int j = 0; j < elems_per_row; j++) {
                float val = (res_output[i * elems_per_row + j]);
                float normed = val * float(gamma[j]) / rms ;
                output[i * elems_per_row + j] = float_type(normed);
            }
        }   
    }
};

void Multithread_RMSNorm_impl(
    float16_t* output,
    float16_t* vals,
    float16_t* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    int threadnum,
    AliveThreadPool*pool
)
{

    int chunk_size = (rows + threadnum -1) / threadnum;

    std::vector<MultiThreadRMSNorm*>ops;

    for(int i=0;i<threadnum;i++){

        int start = chunk_size*i;
        int end = std::min(start+chunk_size,rows);


        ops.push_back(new MultiThreadRMSNorm(output + start*elems_per_row,vals + start*elems_per_row,
                                            gamma,epsilon,end - start,elems_per_row           
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

void Multithread_Pre_RMSNorm_impl(
    float16_t* output,
    float16_t* res_output,
    float16_t* vals,
    float16_t* residual,
    float16_t* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    int threadnum,
    AliveThreadPool*pool
){

    int chunk_size = (rows + threadnum -1) / threadnum;

    std::vector<MultiThreadPreRMSNorm*>ops;

    for(int i=0;i<threadnum;i++){

        int start = chunk_size*i;
        int end = std::min(start+chunk_size,rows);

        ops.push_back(new MultiThreadPreRMSNorm(output + start*elems_per_row,res_output + start*elems_per_row,vals + start*elems_per_row,
                                            residual + start*elems_per_row,gamma,epsilon,end - start,elems_per_row           
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


void launch_rms_norm(
    float16_t* output,
    float16_t* vals,
    float16_t* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    int threadnum,
    AliveThreadPool*pool
){

    int start = 0;
    int end = rows;

    Multithread_RMSNorm_impl(output,vals,gamma,epsilon,rows,elems_per_row,
                            threadnum,pool);

}




void launch_pre_rms_norm(
    float16_t* output,
    float16_t* res_output,
    float16_t* vals,
    float16_t* residual,
    float16_t* gamma,
    float epsilon,
    int rows,
    int elems_per_row,
    int threadnum,
    AliveThreadPool *pool
){

    Multithread_Pre_RMSNorm_impl(
        output,
        res_output,
        vals,
        residual,
        gamma,
        epsilon,
        rows,
        elems_per_row,
        threadnum,
        pool
    );
    
}