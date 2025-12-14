import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from quantize.triton_utils.kernels import dequant_dim0, dequant_dim1
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, load_checkpoint_and_dispatch, dispatch_model
from tqdm import tqdm
import gc  
from quantize.utils import get_named_linears,set_op_by_name

import json
import os

logger = getLogger(__name__)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass

def Pack_T_MAC_Weight(w_quant:torch.Tensor,bits:int = 4):

        assert w_quant.ndim == 3, "w_quant must be a 2D tensor"
        assert w_quant.shape[2] % 8 == 0, "w_quant's width must be divisible by 32 // bit"
        assert w_quant.shape[0]  == bits, "w_quant's first dimension must match the number of bits"
        packed = torch.zeros((w_quant.shape[0], w_quant.shape[1], w_quant.shape[2] // 8), dtype=torch.uint8, device=w_quant.device)

        for bit in range(bits) :
            for i in range(w_quant.shape[2] // 8):
                for j in range(8):
                    packed[bit,:, i] |= (w_quant[bit,:, i * 8 + j])  << j

        return packed


def export(quantized_w:torch.Tensor, w_scale:torch.Tensor, w_zero:torch.Tensor, 
               groupsize = 128,output_dir = None,bit = None):
        
        assert quantized_w.ndim == 3, "quantized_w must be a 3D tensor"
        assert w_scale.ndim == 3, "scales must be a 3D tensor"
        assert w_zero.ndim == 3, "zeros must be a 3D tensor"
        
        import os
        os.makedirs(output_dir,exist_ok=True)
        
        w_zero = w_zero.to(torch.float16)
        w_zero = w_zero.contiguous().cpu().numpy()
        with open(f'{output_dir}/w_zero.bin', 'wb') as f:
            f.write(w_zero.tobytes())

        w_scale = w_scale.to(torch.float16)
        w_scale = w_scale.permute(2, 0, 1)
        w_scale = w_scale.contiguous().detach().cpu().numpy()

        with open(f'{output_dir}/w_scale.bin', 'wb') as f:
            f.write(w_scale.tobytes())

        quantized_w = quantized_w.to(torch.uint8).permute(2, 0, 1)
        quantized_w = Pack_T_MAC_Weight(quantized_w,bit)

        quantized_w = quantized_w.contiguous().cpu().numpy()

        with open(f'{output_dir}/w_quant.bin', 'wb') as f:
            f.write(quantized_w.tobytes())

class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        use_fake=False,
        **kwargs
    ):
        super().__init__()
        # if bits not in [2, 4, 8]:
        #     raise NotImplementedError("Only 2,4,8 bits are supported.")
        # if infeatures % 32 != 0 or outfeatures % 32 != 0:
        #     raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer(
            'qweight',
            torch.zeros((math.ceil(infeatures / (32 // self.bits)), outfeatures), dtype=torch.int32)
        )
        self.register_parameter(
            'scales',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures,self.bits), dtype=torch.float16))
        )
        self.register_buffer(
            'qzeros',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))  # maybe not use nn.Parameter
        )

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        # self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.use_fake = use_fake

    def post_init(self):
        pass


    def use_fake_quantization(self, del_quant=False,transpose=False):
        # use fake quantization for faster training but consume more memory
        weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        dim0, dim1, dim2 = weight.shape

        weight =(weight.view(-1, self.group_size, dim1,dim2) * self.scales.view(-1, 1, dim1,dim2))
        weight = torch.sum(weight,dim=-1)
        weight = weight + self.qzeros.view(-1,1,dim1)
        weight = weight.reshape(dim0,dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1).contiguous()
        self.register_buffer(
            'weight',
            weight
        )
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.qzeros

        
    def pack(self, linear, intweight, scales, zeros):
        hlq_scales = scales
        self.scales = nn.Parameter(hlq_scales.half())
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = intweight.to(torch.uint8)

        intweight = intweight.t().contiguous()

        intweight = intweight.cpu().numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((math.ceil(intweight.shape[0]/(32//self.bits)), intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        qweight = qweight.astype(np.int32)

        self.qweight = torch.from_numpy(qweight)
        self.qzeros = nn.Parameter(zeros.half())

    def unpack_bits(self, packed: torch.Tensor, dim3: int) -> torch.Tensor:
        unpacked = ((packed.unsqueeze(-1).to(torch.int8) >> torch.arange(dim3, device=packed.device)) & 1)
        return unpacked.to(torch.float16)

    def forward(self, x):
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)

            weight = weight.to(x.device)

            dim0, dim1, dim2 = weight.shape

            weight =(weight.view(-1, self.group_size, dim1,dim2) * self.scales.view(-1, 1, dim1,dim2))
            weight = torch.sum(weight,dim=-1)
            weight = weight + self.qzeros.view(-1,1,dim1)
            weight = weight.reshape(dim0,dim1)

        out = torch.matmul(x, weight.to(x.dtype))
        out = out + self.bias if self.bias is not None else out
        return out
    
    def export_cpp(self, file_prefix):

        quantized_w = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        quantized_w = quantized_w.transpose(0,1)

        w_scale = self.scales
        w_scale = w_scale.transpose(0,1)

        w_zero = self.qzeros
        w_zero = w_zero.transpose(0,1)
        
        if w_zero.ndim == 2 :
            w_zero = w_zero.unsqueeze(-1)

        logger.info(f"exporting quantized weights to {file_prefix}")

        weight_config = {}
        weight_config['weight_data_len'] = int(self.infeatures * self.outfeatures * int(self.bits)/8)
        weight_config['base_bit'] = int(self.bits)
        weight_config['use_sparse'] = False
        weight_config['in_channel'] = self.infeatures
        weight_config['out_channel'] = self.outfeatures

        if self.bias is not None :
            weight_config['has_bias'] = True 
            bias = self.bias.data.clone()
            bias = bias.to(torch.float16)
            bias = bias.contiguous().cpu().numpy()
            with open(f'{file_prefix}/bias.bin', 'wb') as f:
                f.write(bias.tobytes())
        else :
            weight_config['has_bias'] = False
        export(quantized_w, w_scale, w_zero, self.group_size, file_prefix ,int(self.bits))

        with open(f'{file_prefix}/weight_config.json', 'w') as f:
            json.dump(weight_config, f,indent=4)
        
def load_quantized_model(model_path, wbits, group_size):
    print(f"Loading quantized model from {model_path}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer

def load_quantized_model_cpu(model_path, wbits, group_size):
    print(f"Loading quantized model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    model.tie_weights()
    device_map ={'':'cpu'}
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")
    return model, tokenizer


decoderlayer_name_dict = {
    "qwen3": "Qwen3DecoderLayer",
    'llama': "LlamaDecoderLayer",
}


def load_quantized_model_multi_devices(model_path, wbits, group_size,model_type:str="llama") :
    print(f"Loading quantized model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Replace linears with QuantLinear"):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features, module.out_features, module.bias is not None)
            set_op_by_name(layer, name, q_linear)

    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is None:
        print("CUDA_VISIBLE_DEVICES not set; accelerate will detect available GPUs.")
    else:
        print(f"CUDA_VISIBLE_DEVICES = {visible_devices}")

    max_memory = {0: "10GB", 1: "10GB"}   # Modify this based on your GPU setup

    no_split_module_classes = []
    no_split_module_classes.append(decoderlayer_name_dict[model_type])

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes
    )

    print("Dispatching & loading checkpoint across devices (device_map='auto') ...")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_path,
        device_map=device_map,
        offload_folder=None,          
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,  
        dtype=torch.float16,         
        # offload_state_dict=True,   
    )
    print("Loading pre-computed quantized weights Successfully")

    model = dispatch_model(model, device_map=device_map)

    return model, tokenizer


def load_quantized_to_fp16(model_path, wbits, group_size):
    print(f"Loading quantized model from {model_path}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    with torch.no_grad() :
        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            named_linears = get_named_linears(layer, QuantLinear)
            for name, module in named_linears.items():
                fp_linear = torch.nn.Linear(module.infeatures,module.outfeatures,not module.bias is None,dtype=torch.float16)
                module.use_fake_quantization(del_quant=True,transpose=True)
                fp_linear.weight.copy_(module.weight)
                if module.bias is not None :
                    fp_linear.bias.copy_(module.bias)
                set_op_by_name(layer, name, fp_linear)
                torch.cuda.empty_cache()
                gc.collect()
                    
    return model, tokenizer


def _save_weight_to_bin(
    tensor: torch.Tensor, 
    file_path: str, 
    save_metadata: bool
) -> None:

    tensor = tensor.to(torch.float16) 
    tensor = tensor.contiguous().cpu().numpy()
    with open(file_path, 'wb') as f:
        f.write(tensor.tobytes())

def export_llama_weights(
    model,
    output_dir, 
    config,
    save_metadata: bool = True
) -> None:

    import os
    os.makedirs(output_dir, exist_ok=True)

    if config.tie_word_embeddings:
        embed_weight = model.model.embed_tokens.weight.data
        _save_weight_to_bin(embed_weight, os.path.join(output_dir, "shared_embed_lm_head.bin"), save_metadata)
    else :
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_weight = model.model.embed_tokens.weight.data
            _save_weight_to_bin(embed_weight, os.path.join(output_dir, "embed_tokens.bin"), save_metadata)

        if hasattr(model, 'lm_head'):
            lm_head_weight = model.lm_head.weight.data
            _save_weight_to_bin(lm_head_weight, os.path.join(output_dir, "lm_head.bin"), save_metadata)
        
    for name, module in model.named_modules():
        if "input_layernorm" in name.lower() or "post_attention_layernorm" in name.lower():

            out_name = name.split('.')[-1]
            layer_idx = None
            parts = name.split('.')
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break

            if layer_idx is not None:
                layer_dir = os.path.join(output_dir, f"{layer_idx}")
                os.makedirs(layer_dir, exist_ok=True)

                weight = module.weight.data
                _save_weight_to_bin(
                    weight,
                    os.path.join(layer_dir, f"{out_name}_weight.bin"),
                    save_metadata
                )

                if hasattr(module, 'bias') and module.bias is not None:
                    bias = module.bias.data
                    _save_weight_to_bin(
                        bias,
                        os.path.join(layer_dir, "rmsnorm_bias.bin"),
                        save_metadata
                    )

    final_rmsnorm = model.model.norm.weight.data
    _save_weight_to_bin(
        final_rmsnorm, 
        os.path.join(output_dir, "final_rmsnorm_weight.bin"), 
        save_metadata
    )

def export_model_cpp(model, export_dir):

    export_llama_weights(model, export_dir, model.config)

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            layer_idx = None
            parts = name.split('.')
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break
            layer_name = name.split('.')[-1]
            if layer_idx is not None:
                layer_dir = os.path.join(export_dir, f"{layer_idx}", layer_name)
                os.makedirs(layer_dir, exist_ok=True)
                module.export_cpp(layer_dir)
    

__all__ = ["QuantLinear","load_omniq_quantized"]
