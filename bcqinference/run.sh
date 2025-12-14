CUDA_VISIBLE_DEVICES=1 python generate.py --compile 2 --num_samples 5 \
  --model_name meta-llama/Llama-3.1-8B  --bitwidth 3 --group_size 128 --dtype "float16" \
  --backend bcq --max_new_tokens 100 --checkpoint_path /home/nku509/codes/SSD/ELUTQ/Efficient_Finetuning/output/bcq_model/Llama-3-8b-w3g128-1024-c4 --print_result --prompt "please introduce beijing, "