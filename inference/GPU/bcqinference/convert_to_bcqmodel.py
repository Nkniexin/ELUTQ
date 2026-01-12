"""
change AutoModelForCausalLM to bcq's transformer model
"""

import os
import json
import shutil
from safetensors.torch import load_file, save_file
import torch

model_path = 'path/to/bcq_model/Llama-3-8b-w3g128-1024'
output_dir = 'path/to/new_model_safetensors'


def convert_model(model_path:str = None, output_dir:str = None) :
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    MAX_SHARD_BYTES = 4 * 1024 ** 3

    os.makedirs(output_dir, exist_ok=True)

    print("Copying model directory...")

    for root, dirs, files in os.walk(model_path):
        rel = os.path.relpath(root, model_path)
        target_root = os.path.join(output_dir, rel)
        os.makedirs(target_root, exist_ok=True)

        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(target_root, f)
            shutil.copy2(src, dst)

    print("Model directory copied to:", output_dir)

    index_path = os.path.join(output_dir, "model.safetensors.index.json")

    state_dict = {}

    if os.path.exists(index_path):
        print("Loading multi-shard safetensors...")
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_cache = {}

        for key, shard_file in weight_map.items():
            shard_path = os.path.join(output_dir, shard_file)
            if shard_path not in shard_cache:
                shard_cache[shard_path] = load_file(shard_path)
            state_dict[key] = shard_cache[shard_path][key]

        print(f"Loaded {len(state_dict)} tensors from {len(shard_cache)} shards")

    else:
        print("Loading single model.safetensors...")
        sf_path = os.path.join(output_dir, "model.safetensors")
        if not os.path.exists(sf_path):
            raise FileNotFoundError("No safetensors found in output_dir after copy!")
        state_dict = load_file(sf_path)
        print(f"Loaded {len(state_dict)} tensors.")

    layer_keys = [k for k in state_dict.keys() if k.startswith("model.layers.")]
    layer_ids = set(int(k.split(".")[2]) for k in layer_keys)
    num_layers = max(layer_ids) + 1

    print("Detected layers:", num_layers)

    key_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "lm_head.weight": "output.weight",
        "model.norm.weight": "norm.weight",
    }

    # 动态添加每层映射
    for i in range(num_layers):
        key_map.update({
            f"model.layers.{i}.self_attn.q_proj.qweight": f"layers.{i}.attention.wq.qweight",
            f"model.layers.{i}.self_attn.q_proj.alpha": f"layers.{i}.attention.wq.alpha",
            f"model.layers.{i}.self_attn.q_proj.beta": f"layers.{i}.attention.wq.beta",

            f"model.layers.{i}.self_attn.k_proj.qweight": f"layers.{i}.attention.wk.qweight",
            f"model.layers.{i}.self_attn.k_proj.alpha": f"layers.{i}.attention.wk.alpha",
            f"model.layers.{i}.self_attn.k_proj.beta": f"layers.{i}.attention.wk.beta",

            f"model.layers.{i}.self_attn.v_proj.qweight": f"layers.{i}.attention.wv.qweight",
            f"model.layers.{i}.self_attn.v_proj.alpha": f"layers.{i}.attention.wv.alpha",
            f"model.layers.{i}.self_attn.v_proj.beta": f"layers.{i}.attention.wv.beta",

            f"model.layers.{i}.self_attn.o_proj.qweight": f"layers.{i}.attention.wo.qweight",
            f"model.layers.{i}.self_attn.o_proj.alpha": f"layers.{i}.attention.wo.alpha",
            f"model.layers.{i}.self_attn.o_proj.beta": f"layers.{i}.attention.wo.beta",

            f"model.layers.{i}.mlp.gate_proj.qweight": f"layers.{i}.feed_forward.w1.qweight",
            f"model.layers.{i}.mlp.gate_proj.alpha": f"layers.{i}.feed_forward.w1.alpha",
            f"model.layers.{i}.mlp.gate_proj.beta": f"layers.{i}.feed_forward.w1.beta",

            f"model.layers.{i}.mlp.up_proj.qweight": f"layers.{i}.feed_forward.w3.qweight",
            f"model.layers.{i}.mlp.up_proj.alpha": f"layers.{i}.feed_forward.w3.alpha",
            f"model.layers.{i}.mlp.up_proj.beta": f"layers.{i}.feed_forward.w3.beta",

            f"model.layers.{i}.mlp.down_proj.qweight": f"layers.{i}.feed_forward.w2.qweight",
            f"model.layers.{i}.mlp.down_proj.alpha": f"layers.{i}.feed_forward.w2.alpha",
            f"model.layers.{i}.mlp.down_proj.beta": f"layers.{i}.feed_forward.w2.beta",

            f"model.layers.{i}.input_layernorm.weight": f"layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight": f"layers.{i}.post_attention_layernorm.weight",
        })

    new_state = {}
    missing = []

    for old, new in key_map.items():
        if old in state_dict:
            new_state[new] = state_dict[old]
        else:
            missing.append(old)

    print(f"Mapped {len(new_state)} tensors.")
    if missing:
        print("Missing example keys:", missing[:5])

    print("Removing old safetensors and index.json...")

    for f in os.listdir(output_dir):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(output_dir, f))

    print("Old files removed.")

    def tensor_bytes(t):
        return t.numel() * t.element_size()

    shards = []
    cur = {}
    cur_size = 0

    for k, v in new_state.items():
        size = tensor_bytes(v)
        if cur_size + size > MAX_SHARD_BYTES and cur:
            shards.append(cur)
            cur = {}
            cur_size = 0
        cur[k] = v
        cur_size += size

    if cur:
        shards.append(cur)

    num_shards = len(shards)
    print("Saving", num_shards, "shards...")

    weight_map = {}
    total_size = 0

    for i, sd in enumerate(shards):
        name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
        path = os.path.join(output_dir, name)
        save_file(sd, path)

        for key, tensor in sd.items():
            weight_map[key] = name
            total_size += tensor_bytes(tensor)

        print("Saved:", name, f"({len(sd)} tensors)")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print("New index.json saved.")


if __name__ == '__main__' :


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    convert_model(args.model_path, args.output_dir)


