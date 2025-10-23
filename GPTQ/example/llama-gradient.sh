CUDA_VISIBLE_DEVICES=1 python llama.py \
--model path/to/llama3_8b_hf  \
--dataset c4 \
--skip_lmhead \
--wbits 2 \
--groupsize 128 \
--iters 100 \
--lr 0.001 \
--export llama3_8b_w2g128_C++