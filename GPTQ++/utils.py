import torch
from math import inf
import logging
from termcolor import colored
import sys
import os
import time
import json

from safetensors.torch import safe_open
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RMSNorm


model_transofomer_block_dict = {
    'llama': LlamaDecoderLayer,
    'qwen3': Qwen3DecoderLayer,
}

model_norm_dict = {
    'llama': LlamaRMSNorm,
    'qwen3': Qwen3RMSNorm,
}


def update_dataset(layer, dataset, dev, attention_mask, position_ids,position_embeddings):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)

                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids,position_embeddings = position_embeddings).to('cpu')
                dataset.update_data(index,new_data)

def offload_get_embedtokens(model_path_or_name):
    index_path = f"{model_path_or_name}/model.safetensors.index.json"
    config = AutoConfig.from_pretrained(model_path_or_name)
    with open(index_path, "r") as f:
        index = json.load(f)["weight_map"]

    embed_prefix = "model.embed_tokens."
    embed_tensors = {k: v for k, v in index.items() if k.startswith(embed_prefix)}

    embed_state = {}
    for name, shard in embed_tensors.items():
        with safe_open(f"{model_path_or_name}/{shard}", framework="pt", device="cpu") as f:
            embed_state[name[len(embed_prefix):]] = f.get_tensor(name)

    embed_layer = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    embed_layer.load_state_dict(embed_state)

    return embed_layer

def offload_get_block(model_path_or_name, layer_idx:int = 0, model_type:str='llama'):
    index_path = f"{model_path_or_name}/model.safetensors.index.json"
    config = AutoConfig.from_pretrained(model_path_or_name)
    with open(index_path, "r") as f:
        index = json.load(f)["weight_map"]
    config._attn_implementation = "sdpa"
    LayerClass = model_transofomer_block_dict[model_type]
    layer = LayerClass(config,layer_idx=layer_idx)
    prefix = f"model.layers.{layer_idx}."
    layer_tensors = {k: v for k, v in index.items() if k.startswith(prefix)}

    state_dict = {}
    for name, shard in layer_tensors.items():
        with safe_open(f"{model_path_or_name}/{shard}", framework="pt", device="cpu") as f:
            state_dict[name] = f.get_tensor(name)

    state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    layer.load_state_dict(state_dict, strict=False)

    return layer

def offload_get_lmhead(model_path_or_name):

    index_path = f"{model_path_or_name}/model.safetensors.index.json"
    config = AutoConfig.from_pretrained(model_path_or_name)
    with open(index_path, "r") as f:
        index = json.load(f)["weight_map"]
    lm_prefix = "lm_head."
    lm_tensors = {k: v for k, v in index.items() if k.startswith(lm_prefix)}

    lm_state = {}
    for name, shard in lm_tensors.items():
        with safe_open(f"{model_path_or_name}/{shard}", framework="pt", device="cpu") as f:
            lm_state[name[len(lm_prefix):]] = f.get_tensor(name)

    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    lm_head.load_state_dict(lm_state)

    return lm_head

def offload_get_norm(model_path_or_name,model_type:str='llama'):
    index_path = f"{model_path_or_name}/model.safetensors.index.json"
    config = AutoConfig.from_pretrained(model_path_or_name)
    with open(index_path, "r") as f:
        index = json.load(f)["weight_map"]
    norm_prefix = "model.norm."
    norm_tensors = {k: v for k, v in index.items() if k.startswith(norm_prefix)}

    norm_state = {}
    for name, shard in norm_tensors.items():
        with safe_open(f"{model_path_or_name}/{shard}", framework="pt", device="cpu") as f:
            norm_state[name[len(norm_prefix):]] = f.get_tensor(name)
    normClass = model_norm_dict[model_type]
    norm_layer = normClass(config.hidden_size, eps=config.rms_norm_eps)
    norm_layer.load_state_dict(norm_state)
    return norm_layer


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger