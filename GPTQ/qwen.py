import time

import torch
import torch.nn as nn

from modelutils import *
from quant import *

import utils
from pathlib import Path

from HLQ import *
import os

from transformers import AutoConfig,AutoTokenizer


def get_qwen(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def qwen_sequential(model, dataloader, dev,logger):
    logger.info('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    logger.info('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)
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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,position_embeddings = position_embeddings)
            for h in handles:
                h.remove()

            for name in subset:
                logger.info(f'{i}, {name}')
                logger.info('Quantizing ...')
                gptq[name].fasterquant(
                    average_bit = args.wbits, groupsize=args.groupsize, 
                    layerid = i,layer_name=name,output_dir=args.export,actorder = args.actorder,
                    percdamp = args.percdamp,iters = args.iters,lr = args.lr
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,position_embeddings = position_embeddings)

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    
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
                percdamp = args.percdamp,iters = args.iters,lr = args.lr,logger = logger
            )
            gptq[name].free()
    

    model.config.use_cache = use_cache


@torch.no_grad()
def qwen_eval(model, testenc, dev):
    logger.info('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    for i in range(len(layers)):
        logger.info(f'{i}')
        layer = layers[i].to(dev)
        
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings = position_embeddings)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

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

def export_qwen_weights(
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
        help='qwen model to load; pass location of hugginface converted checkpoint.'
    )

    parser.add_argument(
        '--log_dir', type=str, default='./logs',
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
        '--wbits', type=float, default=16,
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
        '--iters', type=int,default=100,
    )

    parser.add_argument(
        '--lr', type=float,default=0.001,
    )


    args = parser.parse_args()

    model = get_qwen(args.model)
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
        export_qwen_weights(model,args,config)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )


    if args.wbits < 16 :
        tick = time.time()
        qwen_sequential(model, dataloader, DEV,logger)
        logger.info(f'Quantization time: {time.time() - tick:.2f} seconds')


    # datasets = ['wikitext2', 'ptb', 'c4'] 
    # datasets = ['wikitext2','c4']
    datasets = ['c4']
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        ppl = qwen_eval(model, testloader, DEV)
        logger.info(f'{dataset} perplexity: {ppl:.2f}')

    if args.save:
        model.save_pretrained(args.save)

