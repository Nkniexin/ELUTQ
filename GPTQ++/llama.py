import time

import torch
import torch.nn as nn

from modelutils import *
from quant import *

import utils
from utils import *
from pathlib import Path

from HLQ import *
import os
from quantize.int_linear_real import QuantLinear,load_quantized_model_cpu,load_quantized_model
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM

from tqdm import tqdm
import gc
seqlen = 2048

def cpu_memory_allocated() :

    import psutil
    pid = os.getpid() 
    process = psutil.Process(pid)
    mem_info = process.memory_info()

    return mem_info.rss / 1024**3
    
def get_llama(model):
    config = AutoConfig.from_pretrained(model)
    orginal_num_layers = config.num_hidden_layers
    config.num_hidden_layers = 1 # for block_ap quantization, we only need to load 1 layer each time
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,legacy=False)
    model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16)
    model.half()
    config.num_hidden_layers = orginal_num_layers
    config._attn_implementation = 'sdpa'

    model.model.embed_tokens = offload_get_embedtokens(args.model)
    model.model.layers[0] = offload_get_block(args.model, layer_idx=0, model_type = args.model_type)
    model.model.norm = offload_get_norm(args.model, model_type = args.model_type)
    model.lm_head = offload_get_lmhead(args.model)

    model.seqlen = 2048

    for param in model.parameters():
        param.requires_grad = False

    return model

@torch.no_grad()
def llama_sequential(model, args, dataloader, dev,logger):
    logger.info('Starting ...')

    config = AutoConfig.from_pretrained(args.model)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings', None)
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    inps = inps.to('cpu')
    # outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    model.model.embed_tokens = None
    model.lm_head = None
    
    logger.info(f'CPU memory used : {cpu_memory_allocated():.2f} GiB')
    logger.info('Ready.')   
    layers = nn.ModuleList()
    model.model.layers = None
    gc.collect()
    for i in range(config.num_hidden_layers):
        layer = offload_get_block(args.model,i,args.model_type).to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = HLQ(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,position_embeddings = position_embeddings).to('cpu')
                layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,position_embeddings = position_embeddings)
            for h in handles:
                h.remove()

            for name in subset:
                logger.info(f'{i}, {name}')
                logger.info('Quantizing ...')
                intweight,scales,zeros = gptq[name].fasterquant(
                    average_bit = args.wbits, groupsize=args.groupsize, 
                    layerid = i,layer_name=name,output_dir=args.export,actorder = args.actorder,
                    percdamp = args.percdamp, iters = args.alternate_iters if args.alternating_optimization else args.gradient_iters,
                    lr = args.lr,use_alternating_optimization = args.alternating_optimization
                )

                group_size = args.groupsize
                dim0,dim1 = gptq[name].rows, gptq[name].columns
                scales = scales.view(dim0,-1,args.wbits).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                
                q_linear = QuantLinear(args.wbits, group_size, gptq[name].columns,gptq[name].rows,not gptq[name].layer.bias is None)
                q_linear.pack(gptq[name].layer.cpu(), intweight.cpu() ,scales.float().cpu(), zeros.float().cpu())
                q_linear.to(dev)
                set_op_by_name(layer, name, q_linear) 
                gptq[name].free()  
                torch.cuda.empty_cache()
                logger.info(f"pack quantized {name} finished")
        
        for j in range(args.nsamples):
            # outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,position_embeddings = position_embeddings).to('cpu')
            inps[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask, position_ids=position_ids,position_embeddings = position_embeddings).to('cpu')

        layers.append(layer.cpu())
        del layer
        del gptq 
        torch.cuda.empty_cache()

        # inps, outs = outs, inps
    
    if args.skip_lmhead == False :

        model.model.norm = model.model.norm.to(dev)
        inps =  model.model.norm(inps)

        subset = {}
        subset['lm_head'] = model.lm_head.to(dev)
        gptq = {}

        for name in subset:
            gptq[name] = HLQ(subset[name])

        inps = inps.to(dev)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        layer = model.lm_head.to(dev)

        for j in range(args.nsamples):

            _ = layer(inps[j].unsqueeze(0))

        for h in handles:
            h.remove()

        for name in subset:
            logger.info('lm_head')
            logger.info('Quantizing ...')
            gptq[name].fasterquant(
                average_bit = args.wbits,groupsize=args.groupsize,layerid = 'lm_head',layer_name='lm_head',
                output_dir=args.export,actorder = args.actorder,
                percdamp = args.percdamp,iters = args.alternate_iters if args.alternating_optimization else args.gradient_iters,
                lr = args.lr,use_alternating_optimization = args.alternating_optimization
            )
            gptq[name].free()
        
    logger.info(f'Final CPU memory used : {cpu_memory_allocated():.2f} GiB')
    gc.collect()
    model.model.embed_tokens = offload_get_embedtokens(args.model)
    model.model.layers = layers
    model.lm_head = offload_get_lmhead(args.model)
    model.config.num_hidden_layers = len(layers)
    model.config.use_cache = use_cache
    model.half()

@torch.no_grad()
def llama_eval(model, testenc, dev):
    logger.info('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.to('cpu')
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings', None)
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
            
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    for i in tqdm(range(len(layers)), desc="Processing layers", ncols=100):
        logger.info(f'{i}')
        layer = layers[i].to(dev)
        
        batch_size = 16
        for j in range(0, nsamples, batch_size):
            batch = inps[j:j+batch_size].to(dev)
            out_batch = layer(batch, attention_mask=attention_mask, position_ids=position_ids, position_embeddings = position_embeddings).to('cpu')
            inps[j:j+batch_size] = out_batch
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0).to(dev)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * seqlen):((i + 1) * seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    model.config.use_cache = use_cache

    return ppl.item()


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
    args, 
    config,
    save_metadata: bool = True
) -> None:

    output_dir = args.export
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

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )

    parser.add_argument(
        '--resume_quant', type=str, default=None,
        help='Path to quantized model checkpoint to resume from.'
    )

    parser.add_argument(
        '--log_dir', type=str, default='./logs',
    )

    parser.add_argument(
        '--alternating-optimization', action='store_true', help='Whether to use alternating optimization.'
    )

    parser.add_argument(
        '--model_type', type=str, choices=['llama','qwen3'], default='llama',
    )

    parser.add_argument(
        '--skip_lmhead', action='store_true',
        help='Whether to skip quantizing the lm_head.'
    )

    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )

    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--export',type=str,default=None,
        help='export model for c++ inference '
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--actorder', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )

    parser.add_argument(
        '--alternate_iters', type=int,default=20,
    )

    parser.add_argument(
        '--gradient_iters', type=int,default=100,
    )

    parser.add_argument(
        '--lr', type=float,default=0.001,
    )

    args = parser.parse_args()

    
    if args.resume_quant is not None :
        model,tokenizer = load_quantized_model_cpu(args.resume_quant,args.wbits,args.groupsize)
    else :
        model = get_llama(args.model)
        model.eval()

    #init logger
    if args.log_dir is not None :
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    logger = utils.create_logger(args.log_dir)
    logger.info(args)

    if args.export is not None :

        config = AutoConfig.from_pretrained(args.model)
        generation_config = model.generation_config
        tokenzier = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        tokenzier.save_pretrained(args.export)
        config.save_pretrained(args.export)
        generation_config.save_pretrained(args.export)
        export_llama_weights(model,args,config)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=seqlen
    )


    if args.wbits < 16  and  args.resume_quant is None:
        tick = time.time()
        llama_sequential(model, args,dataloader, DEV,logger)
        logger.info(f'Quantization time: {time.time() - tick:.2f} seconds')

    if args.save:
        model.save_pretrained(args.save)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.save)

    # datasets = ['wikitext2', 'ptb', 'c4'] 
    datasets = ['wikitext2','c4']
    # datasets = ['c4']
    # datasets = ['wikitext2']
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=seqlen
        )
        ppl = llama_eval(model, testloader, DEV)
        logger.info(f'{dataset} perplexity: {ppl:.2f}')

    

