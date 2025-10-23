#pragma once
#include <ops.h>
#include <common.h>
#include <fileutils.h>
#include <json11.hpp>
#include <Allocate.h>
#include <filesystem>


class llama
{
	public:
		llama(const char* model_path, int n_threads);
		~llama();

		void forward(std::vector<int> input_tokens, std::vector<int> position_ids);
		void set_generate_num(int generate_num);
		void reset();
        void load_model(const char* model_path);
        void init_model(int n_threads);
		int get_response();

	private:
		int n_threads;
		std::map<std::string, weight_tensor*> model_params;

		void* embedding;
		void* final_rms_gamma;
		void* lm_head_weight;

		void* workspace;
		void* residual;
		void* temp_workspace;
        float* w_scale; 
		std::vector<void*> kcache;
		std::vector<void*> vcache;
		std::vector<int> input_ids;
		std::vector<int> position_ids;
		std::vector<int> res;


		AliveThreadPool* pool;

		// the hyperparameters
		int hidden_size;
		int head_q;
		int head_kv;
		int dhead;
		int num_layers;
		int max_seqlen;
		int vocab_size;
        int q_dim,k_dim,v_dim;
		float epsilon;
		float rope_base;
		int context_len;
		int ffn_dim;
		int generate_num = 0;
		std::string weight_qtype;
		std::string kv_cache_qtype;
        int weight_groupsize = 128;
		std::vector<std::string>weight_qtypes  = {"16bit","4bit" , "3bit", "2bit" };

		bool tie_word_embeddings = false;
        
};