"""
This script converts a given model (int_linear_real.QuantLinear) into the BCQ (Binarization-code Quantization ) format.
It loads the model, applies the required quantization procedures, and exports the final BCQ
weights and metadata.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from quantize.int_linear_real import QuantLinear,load_quantized_model
from quantize.utils import *
from quantize.triton_utils.kernels import dequant_dim0


def pack_bits_to_int32(x: torch.Tensor):
    """
    x: tensor of shape [dim0, dim1, dim2], values ∈ {0,1}
    Returns a tensor of shape [dim0//32, dim1, dim2], dtype=int32.
    Each output element is a packed 32-bit integer.
    """
    assert x.shape[0] % 32 == 0, "dim0 must be divisible by 32"

    x_int = x.to(torch.int32)
    b, h, w = x.shape
    x_int = x_int.view(b // 32, 32, h, w)   

    shifts = torch.arange(32, device=x.device, dtype=torch.int32).view(32, 1, 1)

    masks = 1 << shifts   

    packed = torch.sum(x_int * masks, dim=1)

    return packed.to(torch.int32)

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

    def pack_from_hlq(self, HLQLinear) :
        
        qweight = HLQLinear.qweight
        alpha = HLQLinear.scales
        beta = HLQLinear.qzeros

        # print('origin : ', alpha.shape)

        if HLQLinear.bias is not None:
            bias = HLQLinear.bias
        else :
            bias = None

        # need to rearrange qweight from (infeatures / (32 // self.bits),outfeatures) to (infeatures // 32, self.bits, outfeatures)
        qweight = dequant_dim0(
            qweight,
            self.bits,
            (1 << self.bits) - 1,
            self.in_features,
            self.out_features
        ) 


        qweight = qweight.permute(0,2,1).contiguous()

        qweight = pack_bits_to_int32(qweight)

        alpha = alpha.permute(0,2,1).contiguous()

        sum_alpha = alpha.sum(dim=1)

        beta = beta + sum_alpha / 2

        alpha = alpha / 2

        self.qweight = qweight.clone()

        self.alpha = alpha.clone().to(self.dtype)
        self.beta = beta.clone().to(self.dtype)

        if bias is not None :
            self.bias = bias.clone().to(self.dtype)
        else :
            self.bias = None


@torch.no_grad()
def export_bcq(model) :

    layers = model.model.layers
    for i in tqdm(range(len(layers))) :
        layer = layers[i]
        linear_type = type(layer.self_attn.q_proj)
        named_linears = get_named_linears(layer, linear_type)
        for name, module in named_linears.items() :
                
                bcq_linear = BCQLinear(module.infeatures, module.outfeatures,module.bits, module.group_size, True if module.bias is not None else False,dtype=torch.float16)

                bcq_linear.pack_from_hlq(module)

                set_op_by_name(layer,name,bcq_linear)

    return model


if __name__ == '__main__' :

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_quant", type=str, help="Quantized model path")
    parser.add_argument("--wbits", type=int, default=4, help="weights quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--save_path", type=str,help='bcq model save path')

    args = parser.parse_args()


    model,tokenizer = load_quantized_model(args.resume_quant, args.wbits, args.group_size)
    model.eval()


    model = export_bcq(model)

    model = model.half()
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)




    




        
        









        

