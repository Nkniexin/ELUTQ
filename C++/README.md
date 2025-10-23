# C++
We provide a high-performance C++ inference backend tailored for ELUTQ


## 🌟 Key Features

1. **Inheritance of T-MAC Kernel**  
   Builds upon the proven T-MAC kernel architecture to provide optimized matrix multiplication performance.

2. **Support for Hierarchical Linear Quantization (HLQ)**  
   Fully compatible with HLQ-quantized weights, enabling arbitrary precision and mixed-bit quantization schemes to maximize accuracy and compression.

3. **Sparse LUT-Based Kernel**  
   Incorporates sparse lookup-table based kernels to accelerate inference.

## 🚀 Quick start
### 1.Build
```bash
cmake -B build
cmake --build build --config Release
```

### 2.Test Speed
```bash
./build/bin/main /path/to/model/
```
You will obtain the following output in the terminal.
```bash
load model done
prefill speed: 27.361 tokens/s

decode speed: 14.699 tokens/s
```

## 📘 Developer Guide

### 🔍 Key Kernels Location

   - **GEMM kernel**

      Location : `ops/src/arm/arm_t_mac_kernal.cpp`

