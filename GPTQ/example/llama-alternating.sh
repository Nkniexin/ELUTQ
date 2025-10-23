CUDA_VISIBLE_DEVICES=1 python llama.py \
--model path/to/llama3_8b_hf  \
--dataset c4 \
--skip_lmhead \
--wbits 2 \
--groupsize 128 \
--alternating_optimization \
--export llama3_8b_w2g128_C++