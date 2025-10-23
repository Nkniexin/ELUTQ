import math
import time
import json
import os

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F

from quant import *
DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from logging import getLogger
logger = getLogger(__name__)

class HLQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.scaler_row = torch.ones(self.columns,device=self.dev)

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


    def fasterquant(
        self, average_bit = None,groupsize=-1,layerid = None,layer_name = None,
        output_dir = None, percdamp=.01,actorder = False,iters = 100,lr = 0.001, use_alternating_optimization = False
    ):

        assert average_bit - int(average_bit) < 1e-9,"GPTQ only support int wbit"

        get_zp = False
        if output_dir is not None :
            output_dir = f'{output_dir}/{layerid}/{layer_name}'
            os.makedirs(output_dir,exist_ok=True)
            get_zp = True

        if layer_name == 'lm_head' :  
            groupsize = 128 # lmhead's groupsize default = 128
            average_bit = 4 # lmhead's average_bit default = 4.0
        
        if groupsize == -1 :
            groupsize = self.columns

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        tick = time.time()

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        quantized = []
        scales = []
        zero_points = []

        for i1 in range(0, self.columns, groupsize):
            i2 = min(i1 + groupsize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if get_zp :
                Q1, quantized1, scales1, zero_points1 = pseudo_quantize_tensor(W1,int(average_bit),True,groupsize,False,get_zp,iters = iters,
                                                                               lr = lr,use_alternating_optimization = use_alternating_optimization)
                quantized.append(quantized1)
                scales.append(scales1)
                zero_points.append(zero_points1)
            else :
                Q1 = pseudo_quantize_tensor(W1,int(average_bit),True,groupsize,False,get_zp,iters = iters,
                                            lr = lr,use_alternating_optimization = use_alternating_optimization)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                q = Q1[:,i]
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        logger.info(f'Quantization time for layer {layer_name} : {time.time() - tick:.2f} seconds')

        loss = torch.sum(Losses).item()
        
        logger.info(f'error, {loss}')

        if actorder:
            Q = Q[:, invperm]

        torch.cuda.empty_cache()  

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if output_dir is not None :
            
            weight_config = {}
            weight_config['weight_data_len'] = int(W.shape[1] * W.shape[0] * int(average_bit)/8)
            weight_config['base_bit'] = int(average_bit)
            weight_config['use_sparse'] = False
            weight_config['in_channel'] = self.columns
            weight_config['out_channel'] = self.rows

            if self.layer.bias is not None :
                weight_config['has_bias'] = True 
                bias = self.layer.bias.data.clone()
                bias = bias.to(torch.float16)
                bias = bias.contiguous().cpu().numpy()
                if output_dir is not None :
                    with open(f'{output_dir}/bias.bin', 'wb') as f:
                        f.write(bias.tobytes())
            else :
                weight_config['has_bias'] = False

            quantized_w = torch.cat(quantized, dim=1)
            w_scale = torch.cat(scales, dim=1)
            w_zero = torch.cat(zero_points, dim=1)

            self.export(quantized_w, w_scale, w_zero, groupsize, output_dir ,int(average_bit))

            with open(f'{output_dir}/weight_config.json', 'w') as f:
                json.dump(weight_config, f,indent=4)

        
        torch.cuda.empty_cache()

    def export(self,quantized_w:torch.Tensor, w_scale:torch.Tensor, w_zero:torch.Tensor, 
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
        w_scale = w_scale.contiguous().cpu().numpy()

        with open(f'{output_dir}/w_scale.bin', 'wb') as f:
            f.write(w_scale.tobytes())

        quantized_w = quantized_w.to(torch.uint8).permute(2, 0, 1)
        quantized_w = self.Pack_T_MAC_Weight(quantized_w,bit)

        quantized_w = quantized_w.contiguous().cpu().numpy()

        with open(f'{output_dir}/w_quant.bin', 'wb') as f:
            f.write(quantized_w.tobytes())

        
    def Pack_T_MAC_Weight(self, w_quant:torch.Tensor,bits:int = 4):

        assert w_quant.ndim == 3, "w_quant must be a 2D tensor"
        assert w_quant.shape[2] % 8 == 0, "w_quant's width must be divisible by 32 // bit"
        assert w_quant.shape[0]  == bits, "w_quant's first dimension must match the number of bits"
        packed = torch.zeros((w_quant.shape[0], w_quant.shape[1], w_quant.shape[2] // 8), dtype=torch.uint8, device=w_quant.device)

        for bit in range(bits) :
            for i in range(w_quant.shape[2] // 8):
                for j in range(8):
                    packed[bit,:, i] |= (w_quant[bit,:, i * 8 + j])  << j

        return packed

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
