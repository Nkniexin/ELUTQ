
import torch
import torch.nn as nn
from tqdm import tqdm
import gc

from plugin import *

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, load_checkpoint_and_dispatch, dispatch_model

from utils import *

class BCQLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        wbits, 
        group_size,
        bias=False, 
        dtype=torch.half):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = wbits
        if group_size == -1:
            self.group_size = in_features
        else:
            self.group_size = group_size
        self.dtype = dtype

        self.register_buffer(
            'qweight',
            torch.empty(
                (in_features//32, self.bits, out_features), 
                dtype=torch.int32)
        )
        buf_name = f"alpha"
        self.register_buffer(
            buf_name,
            torch.empty(
                (in_features // self.group_size, self.bits, out_features), 
                dtype=dtype)
        )
        buf_name = f"beta"
        self.register_buffer(
            buf_name,
            torch.empty(
                (in_features // self.group_size, out_features), 
                dtype=dtype)
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype)
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, out_features), dtype=self.dtype, device='cuda')

    def _gemm(self, x, w_bits, alpha, beta):
        """
        x : (B, T, in_features)
        w_bits: precision
        alpha: (num_groups, w_bits, out_features)
        return -> (B, T, out_features)
        """
        B, T, _ = x.shape
        weight = anybcq_dequant(
            self.qweight, alpha, beta, w_bits, w_bits, self.group_size
        )

        x_flat = x.reshape(-1, self.in_features)
        y_flat = torch.matmul(x_flat, weight)

        return y_flat.reshape(B, T, self.out_features)

    def forward(self, x, **kwargs):
        """
        x : (B, T, in_features)
        precision : None → self.precision / int 
        """
        
        if x.numel() // self.in_features == 1:
            self.output.zero_()
            anybcq_gemv(
                x, self.output,
                self.qweight, self.alpha, self.beta,
                self.bits, self.bits, self.group_size  # To be compatible with the operators of anybcq
            )
            out = self.output
        else:
            out = self._gemm(x, self.bits, self.alpha, self.beta)

        if self.bias is not None:
            out += self.bias
        return out
    
def load_bcq_model(model_path, wbits, groupsize) :
    print(f"Loading BCQ model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            bcq_linear = BCQLinear(module.in_features, module.out_features, wbits, groupsize, not module.bias is None)
            bcq_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, bcq_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed BCQ weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed BCQ weights Successfully")

    return model, tokenizer