import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# --- 关键修改 ---
# 1. 直接使用Hugging Face Hub上的模型ID
#    代码会自动处理下载和缓存，第一次运行会下载，之后会直接从本地加载。
#    这里以 HunyuanDi-v1.1 为例，你可以换成你需要的其他模型ID。
model_id = "Tencent-Hunyuan/HunyuanDi-v1.1" 

# 2. 配置4-bit量化，大幅减少模型大小和内存占用，加速下载和加载
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 定义模型加载参数
kwargs = dict(
    quantization_config=quantization_config, # 应用量化配置
    attn_implementation="sdpa",              # 使用 "flash_attention_2" 如果已安装
    trust_remote_code=True,
    torch_dtype="auto",                      # torch_dtype设为auto，配合量化使用
    device_map="auto",                       # 自动将模型分配到可用设备（如GPU）
    moe_impl="eager",                        # 使用 "flashinfer" 如果已安装
)

print(f"正在从 '{model_id}' 加载模型，并应用4-bit量化...")

# 使用修改后的配置加载模型
model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)

print("模型加载完成。")

# 生成图片
prompt = "一只棕白相间的狗在草地上奔跑"
# 注意：原代码中的 stream=True 在 generate_image 中可能不存在，这里移除了
# 如果你的模型支持流式生成，可以再加回来
image = model.generate_image(prompt=prompt) 
image.save("dog_on_grass.png")

print("图片已生成并保存为 dog_on_grass.png")
