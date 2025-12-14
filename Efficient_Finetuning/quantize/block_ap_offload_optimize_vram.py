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
    set_quant_state,quant_inplace,set_quant_parameters,block_set_quant_parameters,block_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os
import psutil
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

def block_ap_offload(
    model,
    config,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk: 
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk,disk_data_block_size=args.off_load_batch_size)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk,disk_data_block_size=args.off_load_batch_size)
    
    # step 3: catch the input of thefirst layer 
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
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
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
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    del layers[0]
    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer

        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk,disk_data_block_size=args.off_load_batch_size)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk,disk_data_block_size=args.off_load_batch_size)
                                    
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)
        
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk,disk_data_block_size=args.off_load_batch_size)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk,disk_data_block_size=args.off_load_batch_size)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)
    # step 6: start training    
    loss_func = torch.nn.MSELoss()

    layers = nn.ModuleList()
    for block_index in range(config.num_hidden_layers):
        logger.info(f"=== Start quantize blocks {block_index}===")
        # step 6.1: replace torch.nn.Linear with QuantLinear 
        qlayer = offload_get_block(model_path_or_name=args.model, layer_idx=block_index, model_type = args.model_type).to(dev)
        # qlayer = copy.deepcopy(layer)

        if args.epochs > 0:
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids,position_embeddings)
            update_dataset(qlayer,fp_val_inps,dev,attention_mask,position_ids,position_embeddings)

        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
                scales = quantlinear.weight_quantizer.scale.detach()
                zeros = quantlinear.weight_quantizer.zero_point.detach().cpu()
                group_size = quantlinear.weight_quantizer.group_size
                dim0, dim1 = quantlinear.out_features,quantlinear.in_features
                scale_dim0,scale_dim2 = scales.shape
                scales = scales.view(dim0,scale_dim0 // dim0, args.wbits).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(quantlinear.cpu(),  scales.float().cpu(), zeros.float().cpu())
                q_linear.to(dev)
                set_op_by_name(qlayer, name, q_linear)
                del module
                del quantlinear
        torch.cuda.empty_cache()
        gc.collect()
        qlayer.to(dev)
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is required for AMP training
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size 
            if args.quant_lr > 0:
                block_set_quant_parameters(qlayer,True)
                param.append({"params":block_quant_parameters(qlayer),"lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                block_set_quant_parameters(qlayer,False)
                
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        input = quant_inps.to(dev)
                        
                        label = fp_inps.to(dev)
                        quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids,position_embeddings = position_embeddings)
                        reconstruction_loss = loss_func(label, quant_out)
                        loss =  reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]

                # step 6.5: calculate validation loss
                val_loss_list = []
                
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids,position_embeddings = position_embeddings)
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        qlayer.half()

        # step 6.7: update inputs of quantization model
        if args.epochs>0:
            for index,data in enumerate(fp_train_inps):
                quant_train_inps.update_data(index, data)
            for index,data in enumerate(fp_val_inps):
                quant_val_inps.update_data(index, data)

        layers.append(qlayer.to("cpu"))
        torch.cuda.empty_cache()
    
    pid = os.getpid() 
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    logger.info(f'Final CPU memory used : {mem_info.rss / 1024**3:.2f} GiB')


    model.model.layers = layers
    model.config.num_hidden_layers = len(layers)

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    model.half()
    return model

