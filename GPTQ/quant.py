import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from tqdm import tqdm

from itertools import product


def generate_bit_combinations(bits,dtype):

    return torch.tensor(
        list(product([0, 1], repeat=int(bits))),
        dtype=dtype
    )
    
@torch.enable_grad()
def find_optimal_params_gradient(tensor, bits=3, max_iters=100, tol=1e-6, lr=0.001, min_iters = 40): 
    zero_points_dynamic_percent = 1.0 / (2**bits - 1)
    device = tensor.device
    dtype = torch.float32
    tensor = tensor.to(dtype)
    n_blocks, block_size = tensor.shape

    original_mins = tensor.min(dim=1).values
    original_maxs = tensor.max(dim=1).values
    dynamic_ranges = original_maxs - original_mins

    bit_combinations = generate_bit_combinations(bits,dtype).to(device)
    
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

def find_optimal_params_alternating(tensor, bits=3, max_iters=10, device=None):
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
        mse = torch.mean((tensor - reconstructed) ** 2)
         
    return  scales[:,:bits],scales[:,bits]
def quantize_with_scales(tensor, scales, zero_points, bits=3):

    device = tensor.device
    dtype = tensor.dtype
    n_blocks, block_size = tensor.shape
    
    bit_combinations = generate_bit_combinations(bits, dtype).to(device)
    
    adjusted_tensor = tensor - zero_points.unsqueeze(1).to(device)

    comb_vals = torch.matmul(scales.to(device), bit_combinations.t())
    
    diff = comb_vals.unsqueeze(1).to(device) - adjusted_tensor.unsqueeze(2).to(device)
    abs_diff = torch.abs(diff)
    indices = torch.argmin(abs_diff, dim=2)
    

    comb_index = indices.flatten()
    quantized = bit_combinations[comb_index].reshape(n_blocks, block_size, bits).byte()
    
    return quantized

def dequantize_with_scales(quantized, scales, zero_points):

    device = quantized.device
    approx = torch.sum(quantized.float() * (scales.unsqueeze(1)).to(device), dim=2)
    
    dequantized = approx + zero_points.unsqueeze(1).to(device)
    
    return dequantized

def pseudo_quantize_tensor(        
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,iters = 100,lr = 0.001,use_alternating_optimization=False
):  
    
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    
    if use_alternating_optimization :
        scale_list, zero_point = find_optimal_params_alternating(w.clone().detach(),bits = int(n_bit))
    else :
        scale_list, zero_point = find_optimal_params_gradient(w.clone().detach(),bits = int(n_bit),max_iters = iters, lr =lr)

    quantized = quantize_with_scales(w.clone().detach(), scale_list, zero_point,int(n_bit))

    reconstructed = dequantize_with_scales(quantized, scale_list, zero_point)

    w = reconstructed.reshape(org_w_shape).to(w.dtype)

    if get_scale_zp :
        quantized = quantized.reshape(org_w_shape[0],org_w_shape[1],int(n_bit))
        scale_list = scale_list.reshape(org_w_shape[0],org_w_shape[1] // q_group_size,int(n_bit))
        zero_point = zero_point.reshape(org_w_shape[0],org_w_shape[1] // q_group_size,1)

        return w,quantized, scale_list, zero_point
    else :
        return w


