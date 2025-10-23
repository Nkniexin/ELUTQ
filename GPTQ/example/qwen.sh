python qwen.py \
--model Qwen3-8B   \
--dataset c4 \
--skip_lmhead \
--wbits 2 \
--groupsize 128 \
--lr 0.001 \
--iters 100 \
--export qwen3_8b_w2g128_C++