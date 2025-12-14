#include <parameters.h>
#include "../../../common/include/Allocate.h"
#include "../../../common/include/fileutils.h"

weight_tensor* load_Parameter(const std::string & path,Parameter_type type ,size_t size,uint8_t* workplace,int n_thread){

    weight_tensor* weight = new weight_tensor();
    if(type == Parameter_type::PARAM_TYPE_LINEAR){


        std::string config = path + "/weight_config.json";

        json11::Json json = Get_json(config);

        

        int groupsize       = json["groupsize"].int_value();
        int weight_data_len = json["weight_data_len"].int_value();
        bool use_sparse     = json["use_sparse"].bool_value();
        int base_bit        = json["base_bit"].int_value();
        int sparse_mask_len = -1;
        if (use_sparse){
            sparse_mask_len = json["sparse_mask_len"].int_value();
        }
        int in_channel      = json["in_channel"].int_value();
        int out_channel     = json["out_channel"].int_value();
        bool has_bias       = json["has_bias"].bool_value();

        weight->in_channel = in_channel;
        weight->out_channel = out_channel;
        weight->groupsize = groupsize;
        weight->bit = base_bit;
        weight->use_sparse = use_sparse;

        

        weight->weight = Allocate_Memory(weight_data_len);
        read_from_file(weight->weight,weight_data_len,(path + "/w_quant.bin").c_str());

        if(!use_sparse){
            weight->w_scale = (float_type*)Allocate_Memory(base_bit*in_channel*out_channel/groupsize*sizeof(float_type));
            read_from_file(weight->w_scale,base_bit*in_channel*out_channel/groupsize*sizeof(float_type),(path + "/w_scale.bin").c_str());
        }
        else{
            int w_sacle_len = base_bit * in_channel * out_channel / groupsize + sparse_mask_len/2 * 32 ; //TODO,优化，不仅仅只支持32
            weight->w_scale = (float_type*)Allocate_Memory(w_sacle_len*sizeof(float_type));
            read_from_file(weight->w_scale,w_sacle_len*sizeof(float_type),(path + "/w_scale.bin").c_str());
        }
        

        weight->w_zero = (float_type*)Allocate_Memory(in_channel*out_channel/groupsize*sizeof(float_type));
        read_from_file(weight->w_zero,in_channel*out_channel/groupsize*sizeof(float_type),(path + "/w_zero.bin").c_str());

        if(has_bias){
            weight->bias = (float_type*)Allocate_Memory(out_channel*sizeof(float_type));
            read_from_file(weight->bias,out_channel*sizeof(float_type),(path + "/bias.bin").c_str());
        }

        if(use_sparse){
            weight->bit = base_bit + 1;
            weight->sparse_mask_len = sparse_mask_len;
            weight->sparse_mask_data = (uint32_t*)Allocate_Memory(sparse_mask_len*sizeof(uint32_t));
            read_from_file(weight->sparse_mask_data,sparse_mask_len*sizeof(uint32_t),(path + "/weight_mask.bin").c_str());

        }

        
        if(!t_mac_g4_rearrange_weight_impl(weight,workplace,n_thread)){exit(-1);return nullptr;}
        
    }

    else if(type == Parameter_type::PARAM_TYPE_EMBEDDING){

        weight->weight= Allocate_Memory(size);
        weight->bit = 16; //TODO:从config读取
        read_from_file(weight->weight, size, (path).c_str());

    }

    else if(type == Parameter_type::PARAM_TYPE_RMS){


        weight->weight = Allocate_Memory(size);
        
        read_from_file(weight->weight,size , (path).c_str());


    }

    else if(type == Parameter_type::PARAM_TYPE_LM_HEAD){

        weight->weight = Allocate_Memory(size);
        weight->bit = 16; //TODO:从config读取
        read_from_file(weight->weight,size , (path).c_str());
    }   

    return weight;

}
