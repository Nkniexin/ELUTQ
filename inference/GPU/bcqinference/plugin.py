import torch
import custom_kernel

@torch.library.custom_op("plugin::anybcq_gemv", mutates_args={"output"})
def anybcq_gemv(
    x: torch.Tensor, 
    output: torch.Tensor, 
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    q_bias: torch.Tensor, 
    bitwidth: int,
    max_num_bits: int, group_size: int) -> None:
    custom_kernel.anybcq_gemv(
        x, output, q_weight, alpha, q_bias, bitwidth, max_num_bits, group_size)

def anybcq_dequant(
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    q_bias: torch.Tensor, 
    bitwidth: int, 
    max_num_bits: int, group_size: int) -> None:
    weight = custom_kernel.anybcq_dequant(
        q_weight, alpha, q_bias, bitwidth, max_num_bits, group_size)
    return weight

@anybcq_gemv.register_fake
def _(x, output, q_weight, alpha, q_bias, bitwidth, max_num_bits, group_size):
    return None
