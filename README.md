# ELUTQ
ELUTQ is an efficient quantization framework designed for deploying large language models on edge devices. It aligns closely with the computation pipeline of Bit-serial LUT-based GEMM, ensuring both accuracy and efficiency in low-bit inference.

## 🚀 Key Features

- **Hierarchical Linear Quantization (HLQ)**  
  A novel quantization method that better captures the weight distribution compared to traditional uniform quantization.

- **Seamless Integration**  
  Compatible with existing quantization techniques, including **post-training quantization (PTQ)** and **Efficienet Finetuning**.

- **C++ Inference Framework**  
  Provides a pure **C++ runtime** for fast and accurate on-device inference, optimized for edge hardware.


## Supported Models

| Model Family | Example Models         | Notes                           |
|--------------|------------------------|----------------------------------|
| LLaMA        | LLaMA-7B, LLaMA-13B     | Supports LLaMA, LLaMA2 and LlaMA3       |
| Qwen         | Qwen2-7B, Qwen-8B      | Qwen   and other variants      |

## 📊 Perplexity Comparison on C4 Dataset

A comparison of perplexity (↓) between weight-only quantization methods on the **C4** dataset with a context length of 2048. **Wbits** denotes the bit-width of weights, while **BPW** represents the average number of bits per weight.Scale and zero-point are assumed to be stored in `fp16` format.  

| **Method** | **#W** | **#G** | **BPW** | **LLaMA2-7** | **LLaMA2-13** | **LLaMA3-8** | **Qwen3-8** |
|-------------|:------:|:------:|:--------:|:-------------:|:--------------:|:-------------:|:------------:|
| **Baseline** | 16 | – | 16 | 6.97 | 6.47 | 8.89 | 13.30 |
| **GPTQ** | 2 | 128 | 2.25 | 33.70 | 20.97 | 181.82 | 35.57 |
| **AWQ** | 2 | 128 | 2.25 | 1.72e5 | 9.41e4 | 2.14e6 | 2.52e6 |
| **OmniQuant** | 2 | 128 | 2.25 | 15.02 | 11.05 | 35.73 | – |
| **PB-LLM\*** | – | – | 2.2 | 20.60 | 15.32 | 57.33 | – |
| 🩶 **HLQ-GPTQ** | 2 | 128 | 2.37 | _14.89_ | _9.75_ | _27.14_ | _24.60_ |
| 🩶 **HLQ-Finetuning** | 2 | 128 | 2.37 | **11.21** | **9.43** | **19.08** | **22.14** |
| **DB-LLM** | 2 | 64 | – | _9.62_ | _8.38_ | _19.20_ | – |
| 🩶 **HLQ-GPTQ** | 2 | 64 | 2.75 | 13.27 | 9.24 | 20.52 | _20.82_ |
| 🩶 **HLQ-Finetuning** | 2 | 64 | 2.75 | **9.12** | **8.02** | **17.08** | **18.24** |
| **GPTQ** | 3 | 128 | 3.25 | 7.89 | 7.00 | 11.66 | 14.39 |
| **AWQ** | 3 | 128 | 3.25 | 7.84 | 6.94 | 11.49 | 18.51 |
| **OmniQ** | 3 | 128 | 3.25 | 7.75 | 6.98 | 11.66 | – |
| 🩶 **HLQ-GPTQ** | 3 | 128 | 3.5 | _7.66_ | _6.90_ | _10.85_ | _14.15_ |
| 🩶 **HLQ-Finetuning** | 3 | 128 | 3.5 | **7.55** | **6.83** | **10.74** | **13.54** |

## 📦 Installation
```
git clone --recurse-submodules https://github.com/Nkniexin/ELUTQ.git
conda create -n ELUTQ python=3.11
conda activate ELUTQ
pip install -r requirements.txt
```

## ⚡Quick start

Taking LLaMA3-8B as an example.

### HLQ-GPTQ
1. Use Alternating optimization
- w2g128 :
```bash
cd GPTQ
CUDA_VISIBLE_DEVICES=0 python llama.py --model path/to/llama3_8b_hf  --dataset c4  --skip_lmhead --wbits 2 --groupsize 128 --alternating-optimization
```
2. Use Gradient-based optimization
- w2g128
```bash
cd GPTQ
CUDA_VISIBLE_DEVICES=0 python llama.py --model path/to/llama3_8b_hf  --dataset c4  --skip_lmhead --wbits 2 --groupsize 128 --iters 100 --lr 0.001 
```

3. You can add `--export` to export model for C++ inference.
- w2g128
```bash
cd GPTQ
CUDA_VISIBLE_DEVICES=0 python llama.py --model path/to/llama3_8b_hf  --dataset c4  --skip_lmhead --wbits 2 --groupsize 128 --alternating-optimization --export llama3_8b_w2g128_C++
```

### HLQ-Finetuning
HLQ-Finetuning has two stage: **Block-Reconstruction** and **End-to-End Tuning**.

1. Blcok-Reconstruction
```bash
cd Efficient_Finetuning
bash examples/block_ap/llama3-8b/w2g128-c4-trainsize1024.sh

```
2. End-to-End Finetuning
```bash
bash examples/e2e_qp/llama3-8b/w2g128-c4-1024.sh
```

3. You can add `--export` to export model for C++ inference.


## C++ Inference 
See `C++/README.md` for more details

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



