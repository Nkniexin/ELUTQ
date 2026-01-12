CUDA_VISIBLE_DEVICES=1 python generate.py --compile 2 --num_samples 5 \
  --model_name meta-llama/Llama-3.1-8B  --bitwidth 2 --group_size 128 --dtype "float16" \
  --backend bcq --max_new_tokens 128 --checkpoint_path ./Llama-3.1-8b-w2g128