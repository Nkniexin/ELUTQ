import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os
from typing import Callable, TypeVar
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from quantize.block_ap_offload import *
from tqdm import tqdm
from quantize.int_linear_real import load_quantized_model,QuantLinear
from torch.utils.checkpoint import checkpoint

T = TypeVar("T")

def maybe_checkpoint(func: Callable[[...], T], *inputs, checkpoint_enabled: bool, **checkpoint_kwargs) -> T:
    """Execute function normally or with checkpointing, depending on checkpoint_enabled. Forward **checkpoint_kwargs"""
    return checkpoint(func, *inputs,**checkpoint_kwargs) if checkpoint_enabled else func(*inputs)

# def maybe_checkpoint(func: Callable[[...], T], *inputs, checkpoint_enabled: bool, **checkpoint_kwargs) -> T:
#     """Execute function normally or with checkpointing, depending on checkpoint_enabled. Forward **checkpoint_kwargs"""
#     return func(*inputs) if checkpoint_enabled else checkpoint(func, *inputs, **checkpoint_kwargs)

def compute_kl_divergence_loss_values(
    *,
    student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
    max_tokens_per_chunk: int = 256,
    checkpoint_last_chunk: bool = True,
    **checkpoint_kwargs,
) -> torch.Tensor:
    """
    Compute token-wise KL divergence loss without materializing all logits/logprobs simultaneously
    :param student_hidden_states: input hidden states for student head, [batch_size, sequence_length, student_dim]
    :param student_lm_head: a token-wise layer (e.g. nn.Linear) mapping from student_dim to logits [vocabulary_size]
    :param teacher_hidden_states: input hidden states for teacher head, [batch_size, sequence_length, teacher_dim]
    :param teacher_lm_head: a token-wise layer (e.g. nn.Linear) mapping from teacher_dim to logits [vocabulary_size]
    :note: teacher is applied to hidden states without no_grad. If required, set requires_grad=False on teacher manually
    :param max_tokens_per_chunk: materialize logits logprobs for at most this many tokens at a time
    :param checkpoint_kwargs: additional arguments passed to checkpoint (e.g. use_reentrant or determinism_check)
    :param checkpoint_last_chunk: if False, do not apply gradient checkpointing to the very last chunk of inputs
        since they are the first ones to be re-materialized anyway. Useful if loss is backpropagated immediately.
    :returns: token-wise KL loss values of shape [batch_size, sequence_length]
    """
    assert student_hidden_states.requires_grad or teacher_hidden_states.requires_grad or not torch.is_grad_enabled()
    assert teacher_hidden_states.shape[:-1] == student_hidden_states.shape[:-1]
    flat_student_hidden_states = student_hidden_states.flatten(0, -2)
    flat_teacher_hidden_states = teacher_hidden_states.flatten(0, -2)
    total_tokens = flat_teacher_hidden_states.shape[0]

    loss_values_by_chunk = []
    for chunk_start in range(0, total_tokens, max_tokens_per_chunk):
        is_last_chunk = chunk_start + max_tokens_per_chunk >= total_tokens
        loss_values_by_chunk.append(
            maybe_checkpoint(
                _compute_kl_div_from_flat_hidden_states,
                flat_student_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                student_lm_head,
                flat_teacher_hidden_states[chunk_start : chunk_start + max_tokens_per_chunk],
                teacher_lm_head,
                checkpoint_enabled=torch.is_grad_enabled() and (checkpoint_last_chunk or not is_last_chunk),
                **checkpoint_kwargs,
            )
        )
    return torch.cat(loss_values_by_chunk).reshape(*student_hidden_states.shape[:2])



def _compute_kl_div_from_flat_hidden_states(
    flat_student_hidden_states: torch.Tensor,
    student_lm_head: nn.Module,
    flat_teacher_hidden_states: torch.Tensor,
    teacher_lm_head: nn.Module,
) -> torch.Tensor:
    student_logprobs = F.log_softmax(student_lm_head(flat_student_hidden_states), dim=-1)
    with torch.no_grad() :
        teacher_logprobs = F.log_softmax(teacher_lm_head(flat_teacher_hidden_states), dim=-1)
    return F.kl_div(input=student_logprobs, target=teacher_logprobs, log_target=True, reduction="none").sum(-1)


def Get_teacher_model(teacher_model_path,teacher_model_type):

    config = AutoConfig.from_pretrained(teacher_model_path)
    config.num_hidden_layers = 1 # for block_ap quantization, we only need to load 1 layer each time
    model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16)
    model.half()
    config._attn_implementation = 'sdpa'

    model.model.embed_tokens = offload_get_embedtokens(teacher_model_path)
    model.model.layers[0] = offload_get_block(teacher_model_path, layer_idx=0, model_type = teacher_model_type)
    model.model.norm = offload_get_norm(teacher_model_path, model_type = teacher_model_type)
    model.lm_head = offload_get_lmhead(teacher_model_path)

    for param in model.parameters():
        param.requires_grad = False
    return model


def e2e_kd(
    args,
    trainloader,
    valloader,
    logger=None
):
    
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    # get teacher model 
    teacher_model = Get_teacher_model(args.teacher_model,args.model_type)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.config.use_cache = False

    layers = teacher_model.model.layers
    teacher_model.model.embed_tokens = teacher_model.model.embed_tokens.to(dev)
    teacher_model.model.norm = teacher_model.model.norm.to(dev)
    if hasattr(teacher_model.model, 'rotary_emb'):
        # for llama-3.1
        teacher_model.model.rotary_emb = teacher_model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    assert args.model_name is not None, 'model name should given for caching dataset'
    flag = args.model_name
    initialized = False
    if args.off_load_to_disk: 
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                initialized = True
                break
                # shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None

    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                teacher_model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)

    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                teacher_model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    # step 3: catch the input of the first layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None
            self.position_embeddings = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]

            if self.position_embeddings is None :
                self.position_embeddings = kwargs['position_embeddings']

            raise ValueError
        
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
    
    if initialized == False :
        layers[0] = Catcher(layers[0],fp_train_inps)
        iters = len(trainloader)//args.batch_size
        with torch.no_grad():
            for i in range(iters):
                data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
                try:
                    teacher_model(data.to(dev))
                except ValueError:
                    pass
    else :
        layers[0] = Catcher(layers[0],fp_val_inps)
        iters = len(valloader)//args.batch_size
        with torch.no_grad():
            for i in range(iters):
                data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
                try:
                    teacher_model(data.to(dev))
                except ValueError:
                    pass
    
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    position_embeddings = layers[0].position_embeddings

    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    teacher_model.model.embed_tokens = teacher_model.model.embed_tokens.cpu()
    teacher_model.model.norm = teacher_model.model.norm.cpu()
    if hasattr(teacher_model.model, 'rotary_emb'):
        # for llama-3.1
        teacher_model.model.rotary_emb = teacher_model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                teacher_model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    if initialized == False :
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)

        # get fp_train_inps by teacher model
        config = AutoConfig.from_pretrained(args.teacher_model)
        with torch.no_grad():
            for block_index in tqdm(
                range(config.num_hidden_layers),
                desc="Processing teacher model layers",
                dynamic_ncols=True
            ):
                qlayer = offload_get_block(model_path_or_name=args.teacher_model, layer_idx=block_index, model_type = args.model_type).to(dev)

                update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids,position_embeddings)
                
                del qlayer
                torch.cuda.empty_cache()
                gc.collect()


    teacher_model_lmhead = teacher_model.lm_head.to(dev).float()
    for param in teacher_model_lmhead.parameters():
            param.requires_grad = False

    student_model,tokenizer = load_quantized_model(args.student_model,args.wbits, args.group_size)
    logger.info(f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")

    student_model.config.use_cache = False
    student_model = student_model.float()

    for name, param in student_model.named_parameters():
        param.requires_grad = False

    loss_func = torch.nn.MSELoss()
    if args.epochs > 0:
        param = []
        assert args.quant_lr > 0 
        param_group_index = 0
        total_training_iteration = args.epochs * args.train_size / args.batch_size 
        if args.quant_lr > 0:
            param = []
            for name, module in student_model.named_modules():
                # if isinstance(module, LoraLayer):
                if isinstance(module, QuantLinear) and not 'head' in name:
                    module.scales.requires_grad = True
            param.append({'params': [p for n, p in student_model.named_parameters() if 'scale' in n], 'weight_decay': 0.0, 'lr': args.quant_lr})
            quant_index = param_group_index
        optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
        quant_scheduler = CosineAnnealingLR(optimizer, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
        trainable_number = trainable_parameters_num(student_model)
        print(f"trainable parameter number: {trainable_number/1e6}M")

        student_model.gradient_checkpointing_enable()
        for epoch in range(args.epochs):
            # step: 6.4 training
            loss_list = []
            start_time = time.time()
            acc_step = args.gradient_accumulation_steps

            pbar = tqdm(
                enumerate(zip(quant_train_inps, fp_train_inps)),
                total=len(quant_train_inps),
                desc=f"Epoch {epoch+1}/{args.epochs}",
                dynamic_ncols=True
            )

            for index, (quant_inps, fp_inps) in pbar :
                optimizer.zero_grad()
                # obtain output of quantization model
                student_hiddenstates = quant_inps.to(dev).float().requires_grad_(True)
                teacher_hiddenstates = fp_inps.to(dev).float()
                for qlayer in student_model.model.layers[: student_model.config.num_hidden_layers]:
                    # student_hiddenstates = checkpoint(
                    #     qlayer,
                    #     student_hiddenstates,
                    #     attention_mask_batch,
                    #     position_ids,
                    #     None,
                    #     False,
                    #     None,
                    #     position_embeddings,
                    #     use_reentrant=False,
                    # )
                    student_hiddenstates = qlayer(
                        student_hiddenstates,
                        attention_mask = attention_mask_batch,
                        position_ids = position_ids,
                        position_embeddings = position_embeddings,
                    )
        
                kl_loss_values = compute_kl_divergence_loss_values(
                    student_hidden_states=student_hiddenstates,
                    student_lm_head=student_model.lm_head,
                    teacher_hidden_states=teacher_hiddenstates,
                    teacher_lm_head=teacher_model_lmhead,
                    max_tokens_per_chunk=4096,
                )

                loss = kl_loss_values.mean() / acc_step
                loss.backward()

                # loss = loss_func(teacher_hiddenstates,student_hiddenstates)
                # loss = loss / acc_step
                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                loss_list.append(loss.detach().cpu())
                if (index + 1) % acc_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]

            train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            logger.info(f"epoch {epoch} recon_loss:{loss_mean} quant_lr:{quant_scheduler.get_lr()[0]}  max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
        optimizer.zero_grad()
        del optimizer

    del teacher_model
    torch.cuda.empty_cache()
    gc.collect()   

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    student_model.config.use_cache = True    
    student_model.half()


    student_model.save_pretrained(args.save_quant_dir)
    tokenizer.save_pretrained(args.save_quant_dir)
    logger.info(f"saved quantized model to {args.save_quant_dir}")

    return student_model,tokenizer

    





    





    

    