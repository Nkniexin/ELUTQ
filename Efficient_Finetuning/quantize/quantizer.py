import torch
import torch.nn as nn
import gc

import pdb
from itertools import product

import torch.optim as optim

CLIPMIN = 1e-4

def generate_bit_combinations(bits):

    return torch.tensor(
        list(product([0, 1], repeat=int(bits))),
        dtype=torch.int8
    )


def find_optimal_params_gradient(tensor, bits=3, max_iters=100, tol=1e-7, lr=0.001,min_iters = 40):
    zero_points_dynamic_percent = 1.0 / (2**bits - 1)
    device = tensor.device
    dtype = torch.float32
    tensor = tensor.to(dtype)
    n_blocks, block_size = tensor.shape

    original_mins = tensor.min(dim=1).values
    original_maxs = tensor.max(dim=1).values
    dynamic_ranges = original_maxs - original_mins

    bit_combinations = generate_bit_combinations(bits).to(device).to(dtype)
    
    scales = torch.zeros((n_blocks, bits), device=device, requires_grad=True,dtype=dtype)
    
    scale_init = (dynamic_ranges / (2**bits - 1)).unsqueeze(1) * torch.pow(2, torch.arange(0, bits, device=device)).to(dtype)
    
    scales.data.copy_(scale_init)

    zero_points = original_mins.clone().detach().to(device).to(dtype).requires_grad_(True)

    optimizer = optim.Adam([scales,zero_points], lr=lr)
    
    prev_loss = float('inf')
    best_loss = float('inf')
    best_scales = scales.clone().detach()
    best_zero_points = zero_points.clone().detach()
    
    for iteration in range(max_iters):
        optimizer.zero_grad()
        
        with torch.no_grad():
            lower_bound = original_mins - 0.05 * dynamic_ranges
            upper_bound = original_mins + zero_points_dynamic_percent * dynamic_ranges
            zero_points.data = torch.clamp(zero_points.data, lower_bound, upper_bound)
            
        adjusted_tensor = tensor - zero_points.unsqueeze(1)
        
        comb_vals = torch.matmul(scales, bit_combinations.t())
        
        diff = comb_vals.unsqueeze(1) - adjusted_tensor.unsqueeze(2)
        abs_diff = torch.abs(diff)
        indices = torch.argmin(abs_diff, dim=2)
        
        comb_index = indices.flatten()
        
        best_combinations = bit_combinations[comb_index].reshape(n_blocks, block_size, bits)
        
        approx = torch.sum(best_combinations * scales.unsqueeze(1), dim=2)
        
        reconstructed = approx + zero_points.unsqueeze(1)
        
        loss = torch.mean((tensor - reconstructed) ** 2)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():

            scales.data = torch.clamp(scales.data, min=1e-8)
        
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_scales = scales.clone().detach()
            best_zero_points = zero_points.clone().detach()
        
        if abs(prev_loss - current_loss) < tol:
            if iteration > min_iters:  
                break
        
        prev_loss = current_loss

    torch.cuda.empty_cache()
        
    return best_scales, best_zero_points


def generate_augmented_bit_combinations(bits, device=None, dtype=torch.float32):

    device = device if device is not None else torch.device('cpu')
    base = torch.tensor([[((i >> j) & 1) for j in range(bits)] for i in range(2**bits)],
                        device=device, dtype=dtype)  # [2^bits, bits]
    ones = torch.ones((base.shape[0], 1), device=device, dtype=dtype)  # [2^bits, 1]
    return torch.cat([base, ones], dim=1)  # [2^bits, bits+1]

def optimize_s_batch_with_constant(tensor, best_bits):

    device = tensor.device
    dtype = tensor.dtype
    n_blocks, block_size, bits_plus_1 = best_bits.shape

    Bt = best_bits.transpose(1, 2)                       # [n_blocks, bits+1, block_size]
    BtB = torch.bmm(Bt, best_bits)                       # [n_blocks, bits+1, bits+1]
    BtB_inv = torch.linalg.pinv(BtB)                     # batched pseudo-inverse
    BtY = torch.bmm(Bt, tensor.unsqueeze(2))             # [n_blocks, bits+1, 1]

    s_opt = torch.bmm(BtB_inv, BtY).squeeze(2)           # [n_blocks, bits+1]
    return s_opt

@torch.no_grad()
def find_optimal_params_alternating(tensor, bits=3, max_iters=20, device=None):
    device = device if device is not None else tensor.device
    dtype = torch.float32
    tensor = tensor.to(device).to(dtype)
    n_blocks, block_size = tensor.shape

    original_mins = tensor.min(dim=1).values
    original_maxs = tensor.max(dim=1).values
    dynamic_ranges = original_maxs - original_mins

    s_init_nonconst = (dynamic_ranges / (2**bits - 1)).unsqueeze(1) * torch.pow(
        2, torch.arange(bits, device=device, dtype=dtype)
    )  # [n_blocks, bits]
    const_init = original_mins.unsqueeze(1)               
    scales = torch.cat([s_init_nonconst, const_init], dim=1)  

    bit_combinations = generate_augmented_bit_combinations(bits, device=device, dtype=dtype)  

    best_bits = torch.zeros((n_blocks, block_size, bits + 1), device=device, dtype=dtype)

    for iteration in range(max_iters):

        comb_vals = torch.matmul(scales, bit_combinations.t())        

        diff = (tensor.unsqueeze(2) - comb_vals.unsqueeze(1))        # [n_blocks, block_size, 2^bits]
        abs_diff = torch.abs(diff)
        indices = torch.argmin(abs_diff, dim=2)                      # [n_blocks, block_size]
        best_bits = bit_combinations[indices]                        # [n_blocks, block_size, bits+1]

        s_opt = optimize_s_batch_with_constant(tensor, best_bits)    # [n_blocks, bits+1]
        scales = s_opt  

        reconstructed = torch.bmm(best_bits, scales.unsqueeze(2)).squeeze(2)  # [n_blocks, block_size]

        # del diff, comb_vals, indices, best_bits
        # torch.cuda.empty_cache()
        # gc.collect()

        mse = torch.mean((tensor - reconstructed) ** 2)
         
    return  scales[:,:bits],scales[:,bits]



def quantize_with_scales(tensor, scales, zero_points, bits=3):

    device = tensor.device
    dtype = torch.float32
    n_blocks, block_size = tensor.shape
    
    bit_combinations = generate_bit_combinations(bits).to(device).to(dtype)
    
    adjusted_tensor = tensor - zero_points.unsqueeze(1).to(device)

    comb_vals = torch.matmul(scales.to(dtype).to(device), bit_combinations.t())
    
    diff = comb_vals.unsqueeze(1).to(device) - adjusted_tensor.unsqueeze(2).to(device)
    abs_diff = torch.abs(diff)
    indices = torch.argmin(abs_diff, dim=2)
    
    comb_index = indices.flatten()
    quantized = bit_combinations[comb_index].reshape(n_blocks, block_size, bits).byte()
    
    return quantized

def pack_bits(bit_combinations: torch.Tensor) -> torch.Tensor:
    dim3 = bit_combinations.shape[-1]
    assert dim3 <= 8, "Only up to 8 bits packing is supported."

    weights = (2 ** torch.arange(dim3, device=bit_combinations.device, dtype=torch.int8))
    packed = torch.sum(bit_combinations.to(torch.int8) * weights, dim=-1)
    return packed

def unpack_bits(packed: torch.Tensor, dim3: int) -> torch.Tensor:
    unpacked = ((packed.unsqueeze(-1).to(torch.int32) >> torch.arange(dim3, device=packed.device)) & 1)
    return unpacked.to(torch.int8)


def train_quantization_softmin(tensor, scales, zero_points, bits=3, tau = 1.0):

    device = tensor.device
    dtype = torch.float32
    n_blocks, block_size = tensor.shape
    
    bit_combinations = generate_bit_combinations(bits).to(device).to(dtype)
    
    adjusted_tensor = tensor - zero_points.unsqueeze(1).to(device)

    comb_vals = torch.matmul(scales.to(dtype).to(device), bit_combinations.t())
    
    diff = comb_vals.unsqueeze(1).to(device) - adjusted_tensor.unsqueeze(2).to(device)
    abs_diff = torch.abs(diff)
    p = torch.softmax(-abs_diff / tau, dim=2) 

    approx = torch.sum(p * comb_vals.unsqueeze(1), dim=2)
    reconstructed = approx + zero_points.unsqueeze(1)

    return reconstructed
    

class HLQQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        use_alternating = True,
        use_tile = False,
    ):
        super().__init__()
        assert 2 <= n_bits <= 4, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        
        self.dim1, self.dim2 = weight.shape
        self.dtype = weight.dtype
        # init scale and zero point through HLQ qunatization
        self.use_alternating = use_alternating
        self.use_tile = True
        self.HLQ_init(weight, self.use_alternating, self.use_tile)

    def HLQ_init(self, weight, use_alternating, use_tile):
        
        dim1,dim2 = weight.shape

        if use_tile :
            scales_list = []
            zeros_list = []
            weight_reshaped = weight.clone().detach().reshape(-1,self.group_size)
            chunk_size = weight_reshaped.shape[0] // 8
            for i in range(0, weight_reshaped.shape[0], int(chunk_size)):
                end = min(i + chunk_size,weight_reshaped.shape[0])
                weight_tile = weight_reshaped[i:end,:]

                if use_alternating :
                    scales, zeros = find_optimal_params_alternating(weight_tile,bits = self.n_bits)
                else :
                    scales, zeros = find_optimal_params_gradient(weight_tile, bits=self.n_bits)
                
                scales_list.append(scales)
                zeros_list.append(zeros)

                torch.cuda.empty_cache()
                gc.collect()
            scales = torch.cat(scales_list,dim=0)
            zeros = torch.cat(zeros_list,dim=0)


        else :
            if use_alternating :
                scales, zeros = find_optimal_params_alternating(weight.clone().detach().reshape(-1,self.group_size),bits = self.n_bits)
            else :
                scales, zeros = find_optimal_params_gradient(weight.clone().detach().reshape(-1,self.group_size), bits=self.n_bits)
        
        torch.cuda.empty_cache()
        gc.collect()

        self.scale = nn.Parameter(scales.to(weight.dtype))
        self.zero_point = nn.Parameter(zeros.to(weight.dtype))

        if use_tile :
            weight_reshaped = weight.clone().detach().reshape(-1,self.group_size)
            bit_combinations_list = []
            chunk_size = weight_reshaped.shape[0] // 8
            for i in range(0, weight_reshaped.shape[0], int(chunk_size)):
                end = min(i + chunk_size,weight_reshaped.shape[0])
                weight_tile = weight_reshaped[i:end,:]
                scales_tile = scales[i:end,:]
                zeros_tile = zeros[i:end]

                bit_combinations_tile = quantize_with_scales(weight_tile, scales_tile, zeros_tile, bits=self.n_bits).to(torch.int8).to(weight.device)
                bit_combinations_list.append(bit_combinations_tile)

            self.bit_combinations = torch.cat(bit_combinations_list,dim=0)
        else :
            self.bit_combinations = quantize_with_scales(weight.clone().detach().reshape(-1,self.group_size), scales, zeros, bits=self.n_bits).to(torch.int8).to(weight.device)

        self.bit_combinations = pack_bits(self.bit_combinations)

        self.bit_combinations = self.bit_combinations.reshape(dim1,dim2)

        torch.cuda.empty_cache()
        gc.collect()

        
    def get_packed_bit_combinations(self, x):

        dim1,dim2 = x.shape
        self.bit_combinations = quantize_with_scales(x.clone().detach().reshape(-1,self.group_size), self.scale, self.zero_point, bits=self.n_bits).to(torch.int8).to(x.device)

        self.bit_combinations = pack_bits(self.bit_combinations)

        self.bit_combinations = self.bit_combinations.reshape(dim1,dim2)


    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self):

        bit_combinations = self.bit_combinations.reshape(-1,self.group_size)

        bit_combinations = unpack_bits(bit_combinations, self.n_bits).to(self.dtype)

        approx = torch.sum(bit_combinations * self.scale.unsqueeze(1), dim=2)
        
        dequantized = approx + self.zero_point.unsqueeze(1)

        dequantized = dequantized.reshape(self.dim1,self.dim2)

        return dequantized
            
    def forward(self):

        x_dequant = self.fake_quant()
        return x_dequant
