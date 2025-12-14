import os
import sys
import random
import numpy as np
import torch
import time
from datautils_block import get_loaders, test_ppl
import torch.nn as nn
from quantize.block_ap_lazyload import block_ap_lazyload, offload_get_block, offload_get_norm, offload_get_lmhead,offload_get_embedtokens
from tqdm import tqdm
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from quantize.int_linear_real import load_quantized_model,export_model_cpp
from accelerate import infer_auto_device_map, dispatch_model

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    '''
    Note: evaluation simply move model to single GPU. 
    Therefor, to evaluate large model such as Llama-2-70B on single A100-80GB,
    please activate '--real_quant'.
    '''
    # import pdb;pdb.set_trace()
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    model.eval()
    results = {}

    if args.eval_ppl:
        # datasets = ["wikitext2", "c4"]
        datasets = ["c4"]
        # datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        from lm_eval.api import task 
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        task_manager=task_manager,
        cache_requests=True,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="direction of cached dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None,  help="model path of resumed quantized model")
    parser.add_argument("--calib_dataset",type=str,default="redpajama",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=4096, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="input sequence length for evaluating perplexity")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--num_fewshot", type=int,default=0 , help="num_fewshot for lm_eval ")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--wbits", type=int, default=4, help="weights quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--quant_lr", type=float, default=1e-4, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="lr of full-precision weights")
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0,help="weight decay")
    parser.add_argument("--net", type=str, default=None,help="model (family) name, for the easier saving of data cache")
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False, help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--off_load_batch_size", type=int , default=256, help="off load batch size when using --off_load_to_disk")
    parser.add_argument("--model_type", type=str, default="llama", choices=['llama','qwen3'], help="model type, e.g., Llama, Qwen3")
    parser.add_argument("--export", type=str, default=None, help="export the quantized model for c++ inference, specify the output file prefix")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

        
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    if args.net is None:
        args.net = args.model.split('/')[-1]
        logger.info(f"net is None, setting as {args.net}")
    if args.resume_quant:
        # directly load quantized model for evaluation
        model, tokenizer = load_quantized_model(args.resume_quant,args.wbits, args.group_size)
        logger.info(f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")
    else:
        # load fp quantized model
        config = AutoConfig.from_pretrained(args.model)
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


        for param in model.parameters():
            param.requires_grad = False

        # quantization
        if args.wbits < 16:
            logger.info("=== start quantization ===")
            tick = time.time()     
            # load calibration dataset
            cache_trainloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)    
                torch.save(valloader, cache_valloader)    
            block_ap_lazyload(
                model,
                config,
                args,
                trainloader,
                valloader,
                logger,
            )
            logger.info(time.time() - tick)
    torch.cuda.empty_cache()
    if args.save_quant_dir:
        logger.info("start saving model")
        model.save_pretrained(args.save_quant_dir)  
        tokenizer.save_pretrained(args.save_quant_dir) 
        logger.info("save model success")

    if args.export is not None and args.wbits < 16:
        logger.info("start exporting quantized model for c++ inference")
        export_path = args.export

        config = model.config
        generation_config = model.generation_config

        tokenizer.save_pretrained(args.export)
        config.save_pretrained(args.export)
        generation_config.save_pretrained(args.export)

        export_model_cpp(model, export_path)
        
        
        
    evaluate(model, tokenizer, args,logger)



if __name__ == "__main__":
    print(sys.argv)
    main()
