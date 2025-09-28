import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 不需要在这里定义 hf_token 了

model_id = "Tencent-Hunyuan/HunyuanDi-v1.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

kwargs = dict(
    quantization_config=quantization_config,
    attn_implementation="sdpa",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="eager",
)

print(f"正在从 '{model_id}' 加载模型（通过环境变量进行身份验证）...")

# library 会自动找到 HUGGING_FACE_HUB_TOKEN，无需手动传入 token 参数
model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
model.load_tokenizer(model_id)

print("模型加载完成。")

# ... 后续代码不变 ...
prompt = "一只棕白相间的狗在草地上奔跑"
image = model.generate_image(prompt=prompt)
image.save("dog_on_grass.png")
print("图片已生成并保存为 dog_on_grass.png")
