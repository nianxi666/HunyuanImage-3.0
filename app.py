from transformers import AutoModelForCausalLM
import torch

# --- 诊断信息 ---
print("--- 正在运行新的诊断脚本 V2 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# --- 诊断信息结束 ---

# Load the model
model_id = "./HunyuanImage-3"

kwargs = dict(
    attn_implementation="sdpa",
    trust_remote_code=True,
    torch_dtype="auto",
    # 移除了 device_map="auto" 来进行手动设备控制
    moe_impl="eager",
)

print("\n1. 正在从磁盘加载模型到 CPU...")
# 首先，将模型完整加载到 CPU
model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
print("   模型加载完成。")

print("\n2. 正在将整个模型移动到 GPU (cuda)...")
# 然后，用 .to("cuda") 强制将模型的每一个部分都移动到 GPU
model = model.to("cuda")
print("   模型移动完成。")


# --- 验证步骤 ---
# 检查模型的一个参数，确认它是否真的在 GPU 上
param_device = next(model.parameters()).device
print(f"\n3. 验证模型设备: 一个模型参数所在的设备是 -> {param_device}")
# --- 验证结束 ---


print("\n4. 正在加载分词器...")
model.load_tokenizer(model_id)
print("   分词器加载完成。")


print("\n5. 正在生成图片...")
# generate the image
prompt = "A brown and white dog is running on the grass"
image = model.generate_image(prompt=prompt, stream=True)
image.save("image.png")

print("\n6. 图片已成功保存为 image.png!")
