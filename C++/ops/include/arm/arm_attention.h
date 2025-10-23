#pragma once
#include <arm_device.h>

void arm_attention(
    float16_t* o,   // ntokens, nhead, dhead
    float16_t* q,   // ntokens, nhead, dhead
    float16_t* k,   // nkv_tokens, nhead, dhead
    float16_t* v,   // nkv_tokens, nhead, dhead
    float16_t* softmax,
    int ntokens,
    int nkv_tokens,
    int head_q,
    int head_kv,
    int dhead,
    float scale,
    int threadnum,
    AliveThreadPool*pool
);