from safetensors.torch import load_file

# # 读取文件的元信息（不加载到显存，只获取形状和dtype）
# filename = "/home/nku509/codes/SSD/ELUTQ/Efficient_Finetuning/output/bcq_model/Llama-3-8b-w2g128-2048-c4/model-00001-of-00002.safetensors"
# metadata = load_file(filename, device="cpu")  # device 可选，cpu即可

# # metadata 是一个字典，key 是参数名，value 是 tensor
# for name, tensor in metadata.items():
#     print(name, tensor.shape, tensor.dtype)


b = "01101111101100100001100011110101"

print(len(b))
val = int(b, 2)
print(val)