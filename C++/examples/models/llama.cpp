#include <llama.h>

// #define DEBUG_INTERMEDIATE
// #define OPS_Testing
#define USE_TMAC
// #define All_LayerTesting

llama::llama(const char* model_path, int n_threads){

    
    init_model(n_threads);

    std::string config_path = std::string(model_path) + "/config.json";
    json11::Json config_json = Get_json(config_path);

    hidden_size = config_json["hidden_size"].int_value();
    head_q = config_json["num_attention_heads"].int_value();
    head_kv = config_json["num_key_value_heads"].int_value();
    dhead = hidden_size/head_q;
    num_layers = config_json["num_hidden_layers"].int_value();
    epsilon = static_cast<float>(config_json["rms_norm_eps"].number_value());
    rope_base = static_cast<float>(config_json["rope_theta"].number_value());
    max_seqlen = 2048;
    vocab_size = config_json["vocab_size"].int_value();
    ffn_dim = config_json["intermediate_size"].int_value();
    tie_word_embeddings = config_json["tie_word_embeddings"].bool_value();

    q_dim = head_q * dhead;
    k_dim = head_kv * dhead;
    v_dim = head_kv * dhead;

    load_model(model_path);


}

llama::~llama(){
    aligned_free(workspace);
    aligned_free(residual);
    aligned_free(temp_workspace);


    for(int i = 0 ; i < kcache.size(); i++){
        aligned_free(kcache[i]);
    }

    for(int i = 0; i < vcache.size() ;i ++){
        aligned_free(vcache[i]);
    }

    for(auto key : model_params) {
        delete key.second;
    }

    delete this->pool;

}

void llama::load_model(const char* model_path){

    std::string param_path = std::string(model_path);

    workspace = Allocate_Memory(max_seqlen * hidden_size * 20 * sizeof(float_type)); //TODO:精细化控制
    temp_workspace = Allocate_Memory(hidden_size * 100);  // TODO:把这个和workplace合在一起
    residual = Allocate_Memory(max_seqlen * hidden_size * sizeof(float_type));  

    model_params["embedding"] = load_Parameter(param_path + "/embed_tokens.bin",Parameter_type::PARAM_TYPE_EMBEDDING,vocab_size * hidden_size * sizeof(float_type),nullptr,n_threads);
    model_params["final.rmsnorm.gamma"] = load_Parameter(param_path + "/final_rmsnorm_weight.bin",Parameter_type::PARAM_TYPE_RMS,hidden_size * sizeof(float_type),nullptr,n_threads);

    std::string lm_head_path = param_path + "/lm_head";
    if(std::filesystem::exists(lm_head_path)){
        model_params["final.weight"] = load_Parameter(param_path + "/lm_head/lm_head",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);
    }
    else{
        if (tie_word_embeddings) {
            model_params["final.weight"] = model_params["embedding"];
        } else {
            model_params["final.weight"] = load_Parameter(param_path + "/lm_head.bin",Parameter_type::PARAM_TYPE_LM_HEAD,hidden_size*vocab_size*sizeof(float_type),nullptr,n_threads);
        }
    }   
    
    for(int layerid = 0; layerid < num_layers; layerid++) {
        kcache.push_back(Allocate_Memory(max_seqlen * head_kv * dhead * sizeof(float_type)));
        vcache.push_back(Allocate_Memory(max_seqlen * head_kv * dhead * sizeof(float_type)));

        model_params["model." + std::to_string(layerid) + ".norm1.gamma"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/input_layernorm_weight.bin",Parameter_type::PARAM_TYPE_RMS,hidden_size * sizeof(float_type),nullptr,n_threads);

        model_params["model." + std::to_string(layerid) + ".norm2.gamma"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/post_attention_layernorm_weight.bin",Parameter_type::PARAM_TYPE_RMS,hidden_size * sizeof(float_type),nullptr,n_threads);

        model_params["model." + std::to_string(layerid) + ".attn.q.weight"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/self_attn.q_proj",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);

        model_params["model." + std::to_string(layerid) + ".attn.k.weight"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/self_attn.k_proj",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);
        
        model_params["model." + std::to_string(layerid) + ".attn.v.weight"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/self_attn.v_proj",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);
        
        model_params["model." + std::to_string(layerid) + ".attn.proj.weight"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/self_attn.o_proj",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);

        model_params["model." + std::to_string(layerid) + ".mlp.up.weight"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/mlp.up_proj",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);
        
        model_params["model." + std::to_string(layerid) + ".mlp.gate.weight"] =
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/mlp.gate_proj",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);

        model_params["model." + std::to_string(layerid) + ".mlp.fc2.weight"] = 
        load_Parameter(param_path + "/" + std::to_string(layerid) + "/mlp.down_proj",Parameter_type::PARAM_TYPE_LINEAR,-1,(uint8_t*)workspace,n_threads);
    
    }
    std::cout<< "load model done" << std::endl;
}

void llama::init_model(int n_threads){

    this->context_len = 0;
    this->input_ids.clear();
    this->position_ids.clear();
    this->res.clear();

    this->kcache.clear();
    this->vcache.clear();

    this->n_threads = n_threads;
    this->pool = new AliveThreadPool(n_threads);

}

void llama::set_generate_num(int generate_num){

    this->generate_num = generate_num; 
}

void llama::reset(){

    this->context_len = 0;
    this->input_ids.clear();
    this->position_ids.clear();
    this->res.clear();

}

int llama::get_response(){
    return res.back();
}

void llama::forward(std::vector<int> input_tokens, std::vector<int> position_ids){

    this->input_ids = input_tokens;
    this->position_ids = position_ids;

    
    this->context_len += input_tokens.size();

    arm_embedding(
        (float_type*)workspace + input_ids.size() * hidden_size,
        input_ids.data(),
        (float_type*)model_params["embedding"]->weight,
        (int)input_ids.size(),
        hidden_size,
        nullptr
    );

// float_type* now = (float_type*)workspace + input_ids.size() * hidden_size;

// for(int i =0;i<10;i++)std::cout<<now[i]<<" ";

// exit(0);
#ifdef DEBUG_INTERMEDIATE
    std::cout << "embedding done" << std::endl;
    write_to_file(
        (float_type*)workspace + input_ids.size() * hidden_size,
        (int)input_ids.size() * hidden_size * sizeof(float_type),
        "output/output_embed.bin"
    );
#endif
    int qkv_dim = (head_q + 2 * head_kv) * dhead;

#ifdef All_LayerTesting
        auto alllayer_start = std::chrono::steady_clock::now();
#endif
    for(int layerid = 0; layerid < num_layers; layerid++) {

#ifdef OPS_Testing
        auto rmsnorm_start = std::chrono::steady_clock::now();
#endif
        if(layerid == 0) {
            memcpy((float_type*)residual, (float_type*)workspace + input_ids.size() * hidden_size, input_ids.size() * hidden_size * sizeof(float_type));
    
            launch_rms_norm(
                (float_type*)workspace,
                (float_type*)workspace + input_ids.size() * hidden_size,
                (float_type*)model_params["model." + std::to_string(layerid) + ".norm1.gamma"]->weight,
                epsilon,
                (int)input_ids.size(),
                hidden_size,
                this->n_threads,
                this->pool
            );
        } else {
            launch_pre_rms_norm(
                (float_type*)workspace,
                (float_type*)residual,
                (float_type*)workspace,
                (float_type*)residual,
                (float_type*)model_params["model." + std::to_string(layerid) + ".norm1.gamma"]->weight,
                epsilon,
                (int)input_ids.size(),
                hidden_size,
                this->n_threads,
                this->pool
            );
        }
// float_type* now = (float_type*)workspace;

// for(int i =0;i<10;i++)std::cout<<now[i]<<" ";

// exit(0);

// read_from_file(workspace,1536*2,std::string("/home/elf/Sparse_T_MAC/rms1.bin").c_str());

#ifdef OPS_Testing
        auto rmsnorm_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_rmsnorm = rmsnorm_end - rmsnorm_start;
        std::cout << "layer:" << layerid <<"rmsnorm Elapsed time: " << elapsed_rmsnorm.count() << " seconds" << "\n" <<std::endl;
#endif

#ifdef DEBUG_INTERMEDIATE
        std::cout << "rmsnorm1 done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace,
            (int)input_ids.size() * hidden_size * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_rms1.bin").c_str()
        );
#endif

#ifdef OPS_Testing
auto q_start = std::chrono::steady_clock::now();
#endif
        //q
        arm_gemm(
            (int)input_ids.size(),
            (int)hidden_size,
            (int)q_dim,
            (float_type*)workspace, //in
            (float_type*)workspace + input_ids.size() * hidden_size,  //out
            temp_workspace,
            model_params["model." + std::to_string(layerid) + ".attn.q.weight"],
            this->n_threads,
            this->pool
        );
#ifdef OPS_Testing
                auto q_end =std::chrono::steady_clock::now();
                std::chrono::duration<double> q_elapsed = q_end - q_start;
                std::cout << "layer" << layerid << "q Elapsed time: " <<q_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif
#ifdef DEBUG_INTERMEDIATE
        std::cout << "gemm q done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace + input_ids.size() * hidden_size,
            (int)input_ids.size() * q_dim * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_q.bin").c_str()
        );
#endif

// float_type* now = (float_type*)workspace + input_ids.size() * hidden_size;

// for(int i =0;i<10;i++)std::cout<<now[i]<<" ";

// exit(0);

// if(layerid == 27){
    
//     write_to_file(
//         (float_type*)workspace + input_ids.size() * hidden_size,
//         (int)input_ids.size() * q_dim * sizeof(float_type),
//         (std::to_string(layerid) + "_output_q.bin").c_str()
//     );

// }

// if(layerid == 0){
    
//     float_type* now = (float_type*)workspace + input_ids.size() * hidden_size;  //out

//     for(int j = 0;j<input_ids.size()*q_dim ;j++){
//         if(std::isnan(now[j])){
//             std::cout<<"nan is ok!"<<std::endl;
//             break;
//         }
//     }

//     std::cout<<now[6*hidden_size]<<std::endl;
//     write_to_file(
//         (float_type*)workspace + input_ids.size() * hidden_size,
//         (int)input_ids.size() * q_dim * sizeof(float_type),
//         (std::to_string(layerid) + "_output_q.bin").c_str()
//     );

// }


// std::cout<<std::endl;
// for(int i =0;i<10;i++)std::cout<<now[i + hidden_size]<<" ";

// std::cout<<std::endl;
// exit(0);

#ifdef OPS_Testing
        auto k_start = std::chrono::steady_clock::now();
#endif
        //k
        arm_gemm(
            (int)input_ids.size(),
            (int)hidden_size,
            (int)k_dim,
            (float_type*)workspace, //in
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * q_dim,  //out
            temp_workspace,
            model_params["model." + std::to_string(layerid) + ".attn.k.weight"],
            this->n_threads,
            this->pool
        );

// if(layerid == 0){
    
//     float_type* now = (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * q_dim;

//     for(int j = 0;j<input_ids.size()*k_dim ;j++){
//         if(std::isnan(now[j])){
//             std::cout<<"nan is ok!"<<std::endl;
//             break;
//         }
//     }

//     std::cout<<now[6*hidden_size]<<std::endl;
//     write_to_file(
//         (float_type*)workspace + input_ids.size() * hidden_size,
//         (int)input_ids.size() * q_dim * sizeof(float_type),
//         (std::to_string(layerid) + "_output_q.bin").c_str()
//     );

// }

#ifdef OPS_Testing
        auto k_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> k_elapsed = k_end - k_start;
        std::cout << "layer" << layerid << "k Elapsed time: " <<k_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif
#ifdef DEBUG_INTERMEDIATE
        std::cout << "gemm k done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size()*q_dim,
            (int)input_ids.size() * k_dim * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_k.bin").c_str()
        );
#endif

#ifdef OPS_Testing
        auto v_start = std::chrono::steady_clock::now();
#endif
        //v
        arm_gemm(
            (int)input_ids.size(),
            (int)hidden_size,
            (int)v_dim,
            (float_type*)workspace, //in
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * q_dim + input_ids.size() * k_dim,   //out
            temp_workspace,
            model_params["model." + std::to_string(layerid) + ".attn.v.weight"], //weight
            this->n_threads,
            this->pool
        );

#ifdef OPS_Testing
        auto v_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> v_elapsed = v_end - v_start;
        std::cout << "layer" << layerid << "v Elapsed time: " <<v_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif

#ifdef DEBUG_INTERMEDIATE
        std::cout << "gemm v done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size()*(q_dim + k_dim),
            (int)input_ids.size() * v_dim * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_v.bin").c_str()
        );
#endif

#ifdef OPS_Testing
        auto rotary_start = std::chrono::steady_clock::now();
#endif   
        launch_llama_qkv_apply_rotary(
            (float_type*)workspace  ,//q_out
            (float_type*)workspace + input_ids.size() * (hidden_size + qkv_dim), //k_out
            (float_type*)workspace + input_ids.size() * (hidden_size + qkv_dim + head_kv * dhead), //v_out
            (float_type*)workspace + input_ids.size() * hidden_size,
            position_ids.data(),
            (int)input_ids.size(),
            head_q,
            head_kv,
            dhead,
            1,
            rope_base
        );

// if(layerid == 0){
    
//     float_type* now = (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * q_dim;

//     for(int j = 0;j<input_ids.size()*k_dim ;j++){
//         if(std::isnan(now[j])){
//             std::cout<<"nan is ok!"<<std::endl;
//             break;
//         }
//     }

//     std::cout<<now[6*hidden_size]<<std::endl;
//     write_to_file(
//         (float_type*)workspace + input_ids.size() * hidden_size,
//         (int)input_ids.size() * q_dim * sizeof(float_type),
//         (std::to_string(layerid) + "_output_q.bin").c_str()
//     );

// }



            
#ifdef OPS_Testing
        auto rotary_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> rotary_elapsed = rotary_end - rotary_start;
        std::cout << "layer" << layerid << "rotary Elapsed time: " <<rotary_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif  

#ifdef OPS_Testing
        auto cache_start = std::chrono::steady_clock::now();
#endif   
            
        arm_cache(
            (float_type*)workspace + input_ids.size() * (hidden_size + qkv_dim),
            (float_type*)workspace + input_ids.size() * (hidden_size + qkv_dim + head_kv * dhead),
            (float_type*)kcache[layerid],
            (float_type*)vcache[layerid],
            (int)input_ids.size(),
            head_kv,
            dhead,
            position_ids.data()
        );
            
#ifdef OPS_Testing
        auto cache_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> cache_elapsed = cache_end - cache_start;
        std::cout << "layer" << layerid << "cache Elapsed time: " <<cache_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif       

#ifdef OPS_Testing
        auto attention_start = std::chrono::steady_clock::now();
#endif  
        // attention
        arm_attention(
            (float_type*)workspace + input_ids.size() * hidden_size,
            (float_type*)workspace,
            (float_type*)kcache[layerid],
            (float_type*)vcache[layerid],
            (float_type*)workspace + input_ids.size() * hidden_size * 2,
            (int)input_ids.size(),
            context_len,
            head_q,
            head_kv,
            dhead,
            sqrt(1.0 / dhead),
            this->n_threads,
            this->pool
        );
// if(layerid == 0){
    
//     float_type* now = (float_type*)workspace + input_ids.size() * hidden_size;

//     for(int j = 0;j<input_ids.size()*hidden_size;j++){
//         if(std::isnan(now[j])){
//             std::cout<<"nan is ok!"<<std::endl;
//             break;
//         }
//     }

//     std::cout<<now[6*hidden_size]<<std::endl;
//     write_to_file(
//         (float_type*)workspace + input_ids.size() * hidden_size,
//         (int)input_ids.size() * q_dim * sizeof(float_type),
//         (std::to_string(layerid) + "_output_q.bin").c_str()
//     );

// }


#ifdef OPS_Testing
        auto attention_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> attention_elapsed = attention_end - attention_start;
        std::cout << "layer" << layerid << "attention Elapsed time: " <<attention_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif  

#ifdef DEBUG_INTERMEDIATE
        std::cout << "attention done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace + input_ids.size() * hidden_size,
            (int)input_ids.size() * hidden_size * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_attention.bin").c_str()
        );
#endif

#ifdef OPS_Testing
        auto proj_start = std::chrono::steady_clock::now();
#endif  
        arm_gemm(
            (int)input_ids.size(),
            (int)hidden_size,
            (int)hidden_size,
            (float_type*)workspace + input_ids.size() * hidden_size, //in
            (float_type*)workspace,  //out
            temp_workspace,
            model_params["model." + std::to_string(layerid) + ".attn.proj.weight"],
            this->n_threads,
            this->pool
        );



// float_type* now = (float_type*)workspace;

// for(int i =0;i<10;i++)std::cout<<now[i]<<" ";

// exit(0);
#ifdef OPS_Testing
        auto proj_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> proj_elapsed = proj_end - proj_start;
        std::cout << "layer" << layerid << "proj Elapsed time: " <<proj_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif        
#ifdef DEBUG_INTERMEDIATE
        std::cout << "attention out done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace,
            (int)input_ids.size() * hidden_size * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_attenout.bin").c_str()
        );
#endif
#ifdef OPS_Testing
        auto rms2_start = std::chrono::steady_clock::now();
#endif  
        // rms norm            
        launch_pre_rms_norm(
            (float_type*)workspace,
            (float_type*)residual,
            (float_type*)workspace,
            (float_type*)residual,
            (float_type*)model_params["model." + std::to_string(layerid) + ".norm2.gamma"]->weight,
            epsilon,
            (int)input_ids.size(),
            hidden_size,
            this->n_threads,
            this->pool
        );



#ifdef OPS_Testing
        auto rms2_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> rms2_elapsed = rms2_end - rms2_start;
        std::cout << "layer" << layerid << "rms2 Elapsed time: " <<rms2_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif   
#ifdef DEBUG_INTERMEDIATE
        std::cout << "rms2 done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace,
            (int)input_ids.size() * hidden_size * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_rms2.bin").c_str()
        );
#endif
#ifdef OPS_Testing
    auto gate_start = std::chrono::steady_clock::now();
#endif  
        // gate
        arm_gemm(
            (int)input_ids.size(),
            (int)hidden_size,
            (int)ffn_dim ,
            (float_type*)workspace , //in
            (float_type*)workspace + input_ids.size() * hidden_size,  //out
            temp_workspace,
            model_params["model." + std::to_string(layerid) + ".mlp.gate.weight"], //weight
            this->n_threads,
            this->pool
        );

    
#ifdef OPS_Testing
        auto gate_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> gate_elapsed = gate_end - gate_start;
        std::cout << "layer" << layerid << "gate Elapsed time: " <<gate_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif   

#ifdef DEBUG_INTERMEDIATE
        std::cout << "gate done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace + input_ids.size() * hidden_size,
            (int)input_ids.size() *ffn_dim * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_gate.bin").c_str()
        );
#endif
#ifdef OPS_Testing
    auto up_start = std::chrono::steady_clock::now();
#endif  
        // up
        arm_gemm(
            (int)input_ids.size(),
            (int)hidden_size,
            (int)ffn_dim ,
            (float_type*)workspace , //in
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * ffn_dim ,  //out
            temp_workspace,
            model_params["model." + std::to_string(layerid) + ".mlp.up.weight"], //weight
            this->n_threads,
            this->pool
        );
#ifdef OPS_Testing
        auto up_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> up_elapsed = up_end - up_start;
        std::cout << "layer" << layerid << "up Elapsed time: " <<up_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif   
#ifdef DEBUG_INTERMEDIATE
        std::cout << "up done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * ffn_dim,
            (int)input_ids.size() *ffn_dim * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_up.bin").c_str()
        );
#endif

#ifdef OPS_Testing
    auto silu_start = std::chrono::steady_clock::now();
#endif           

        gated_silu(
            (float_type*)workspace + input_ids.size() * hidden_size+input_ids.size()*ffn_dim*2,
            (float_type*)workspace + input_ids.size() * hidden_size,
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * ffn_dim,
            (int)input_ids.size(),
            ffn_dim,
            this->n_threads,
            this->pool
        );


#ifdef OPS_Testing
        auto silu_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> silu_elapsed = silu_end - silu_start;
        std::cout << "layer" << layerid << "silu Elapsed time: " <<silu_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif   

#ifdef DEBUG_INTERMEDIATE
        std::cout << "gate_silu done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace + input_ids.size() * hidden_size+input_ids.size()*ffn_dim*2,
            (int)input_ids.size() *ffn_dim * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_silu.bin").c_str()
        );
#endif
#ifdef OPS_Testing
    auto ffn_start = std::chrono::steady_clock::now();
#endif     
        // ffn out
        arm_gemm(
            (int)input_ids.size(),
            (int)ffn_dim ,
            (int)hidden_size,
            (float_type*)workspace + input_ids.size() * hidden_size + input_ids.size() * ffn_dim * 2, //in
            (float_type*)workspace,  //out
            temp_workspace,
            model_params["model." + std::to_string(layerid) + ".mlp.fc2.weight"], //weight
            this->n_threads,
            this->pool
        );



#ifdef OPS_Testing
        auto ffn_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> ffn_elapsed = ffn_end - ffn_start;
        std::cout << "layer" << layerid << "ffn Elapsed time: " <<ffn_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif   
        
#ifdef DEBUG_INTERMEDIATE
        std::cout << "ffn2 done for layer " << layerid << std::endl;
        write_to_file(
            (float_type*)workspace,
            (int)input_ids.size() * hidden_size * sizeof(float_type),
            ("output/" + std::to_string(layerid) + "/output_ffn2.bin").c_str()
        );
#endif
    }




#ifdef All_LayerTesting
        auto alllayer_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> alllayer_elapsed = alllayer_end - alllayer_start;
        std::cout <<"alllayer Elapsed time: " << alllayer_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif
#ifdef OPS_Testing
    auto finalrms_start = std::chrono::steady_clock::now();
#endif     
    // final rmsnorm
    launch_pre_rms_norm(
        (float_type*)workspace,
        (float_type*)residual,
        (float_type*)workspace,
        (float_type*)residual,
        (float_type*)model_params["final.rmsnorm.gamma"]->weight,
        epsilon,
        (int)input_ids.size(),
        hidden_size,
        this->n_threads,
        this->pool
    );
#ifdef OPS_Testing
        auto finalrms_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> finalrms_elapsed = finalrms_end - finalrms_start;
        std::cout << "finalrms Elapsed time: " <<finalrms_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif  


#ifdef OPS_Testing
    auto lmhead_start = std::chrono::steady_clock::now();
#endif   
    arm_gemm(
        (int)1,
        (int)hidden_size,
        (int)vocab_size,
        (float_type*)workspace + (input_ids.size() - 1) * hidden_size, //in
        (float_type*)workspace + input_ids.size() * hidden_size, //out
        temp_workspace,
        model_params["final.weight"], //weight
        this->n_threads,
        this->pool
    );
#ifdef OPS_Testing
        auto lmhead_end =std::chrono::steady_clock::now();
        std::chrono::duration<double> lmhead_elapsed = lmhead_end - lmhead_start;
        std::cout << "lmhead Elapsed time: " <<lmhead_elapsed.count() << " seconds" << "\n" <<std::endl;
#endif  

// float_type* now = (float_type*)workspace + input_ids.size() * hidden_size;

// for(int i =0;i<10;i++)std::cout<<now[i]<<" ";

// std::cout<<std::endl;

// exit(0);
    arm_argmax_last(
        (int *)workspace,
        (const float_type*)workspace + input_ids.size() * hidden_size,
        (int*)workspace + input_ids.size()*hidden_size+vocab_size,
        (float_type*)workspace + input_ids.size()*hidden_size+vocab_size+n_threads,
        1,
        (int)vocab_size
    );

        
    res.push_back(*((int*)workspace));
}


