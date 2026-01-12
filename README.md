# ELUTQ
ELUTQ is an efficient quantization framework designed for deploying large language models on edge devices. It aligns closely with the computation pipeline of Bit-serial LUT-based GEMM, ensuring both accuracy and efficiency in low-bit inference.

*Paper* : [https://arxiv.org/abs/2510.19482](https://arxiv.org/abs/2510.19482)

## 🚀 Key Features

- **Hierarchical Linear Quantization (HLQ)**  
  A novel quantization method that better captures the weight distribution compared to traditional uniform quantization.

- **Seamless Integration**  
  Compatible with existing quantization techniques, including **post-training quantization (PTQ)** and **Efficienet Finetuning**.

- **Inference Framework**  
  Provides end-to-end inference for fast and accurate on-device inference, including CPUs and GPUs


## Supported Models

| Model Family | Example Models         | Notes                           |
|--------------|------------------------|----------------------------------|
| LLaMA        | LLaMA-7B, LLaMA-13B     | Supports LLaMA, LLaMA2 and LlaMA3       |
| Qwen         | Qwen2-7B, Qwen-8B      | Qwen   and other variants      |

## 📊 Perplexity Comparison on C4 Dataset

A comparison of perplexity (↓) between weight-only quantization methods on the **C4** dataset with a context length of 2048. **Wbits** denotes the bit-width of weights, while **BPW** represents the average number of bits per weight.Scale and zero-point are assumed to be stored in `fp16` format.  

| **Method** | **#W** | **#G** | **BPW**  | **LLaMA3.1-8** | **Qwen3-8** |
|-------------|:------:|:------:|:--------:|:-------------:|:------------:|
| **Baseline** | 16 | – | 16  | 8.89 | 13.30 |
| **HLQ-GPTQ** | 2 | 128 | 2.37  | _27.14_ | _24.60_ |
| **HLQ-Finetuning** | 2 | 128 | 2.37  | **15.08** | **19.46** |
| **HLQ-GPTQ** | 2 | 64 | 2.75  | 20.52 | _20.82_ |
| **HLQ-Finetuning** | 2 | 64 | 2.75  | **14.43** | **17.68** |
| **HLQ-GPTQ** | 3 | 128 | 3.5  | _10.85_ | _14.15_ |
| **HLQ-Finetuning** | 3 | 128 | 3.5 | **10.47** | **13.54** |

## 📦 Installation
```
git clone --recurse-submodules https://github.com/Nkniexin/ELUTQ.git
conda create -n ELUTQ python=3.11
conda activate ELUTQ
pip install -r requirements.txt
```

## ⚡Quick start

Taking LLaMA3.1-8B as an example.

### HLQ-GPTQ
1. Use Alternating optimization
- w2g128 :
```bash
cd GPTQ
CUDA_VISIBLE_DEVICES=0 python llama.py --model path/to/llama3.1_8b_hf  --dataset c4  --skip_lmhead --wbits 2 --groupsize 128 --alternating-optimization
```
2. Use Gradient-based optimization
- w2g128
```bash
cd GPTQ
CUDA_VISIBLE_DEVICES=0 python llama.py --model path/to/llama3.1_8b_hf  --dataset c4  --skip_lmhead --wbits 2 --groupsize 128 --iters 100 --lr 0.001 
```

### HLQ-Finetuning
HLQ-Finetuning has two stage: **Block-Reconstruction** and **End-to-End Tuning**.

1. Blcok-Reconstruction
```bash
cd Efficient_Finetuning
bash examples/block_ap/llama3.1-8b/w2g128-c4.sh

```
2. End-to-End Finetuning
```bash
bash examples/e2e_qp/llama3.1-8b/w2g128-c4.sh
```

## Inference 
stay tuned...


## Third-Party Resources

This project uses the following open-source tools and assets:
- [EfficientQAT](https://github.com/OpenGVLab/EfficientQAT) 
- [GPTQ](https://github.com/ist-daslab/gptq)



## Citation
If you found this work useful, please consider citing:
```bash
@misc{nie2025elutqefficientlutawarequantization,
      title={ELUTQ: Efficient LUT-Aware Quantization for Deploying Large Language Models on Edge Devices}, 
      author={Xin Nie and Liang Dong and HaiCheng Zhang and JiaWang Xiao and G. Sun},
      year={2025},
      eprint={2510.19482},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.19482}, 
}
```



