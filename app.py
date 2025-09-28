from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
import os

# Download only the necessary small files for tokenizer and config (weights will be downloaded during model loading)
local_tokenizer_dir = "./HunyuanImage-3-tokenizer"
if not os.path.exists(local_tokenizer_dir):
    snapshot_download(
        repo_id="tencent/HunyuanImage-3.0",
        local_dir=local_tokenizer_dir,
        ignore_patterns=["*.safetensors", "*.bin"]  # Ignore large weight files
    )

# Load the model from remote (downloads weights on the fly while loading)
model_id_remote = "tencent/HunyuanImage-3.0"
kwargs = dict(
    attn_implementation="sdpa",     # Use "flash_attention_2" if FlashAttention is installed
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="eager",   # Use "flashinfer" if FlashInfer is installed
)

model = AutoModelForCausalLM.from_pretrained(model_id_remote, **kwargs)

# Load tokenizer from the local directory with small files
model.load_tokenizer(local_tokenizer_dir)

# Generate the image
prompt = "A brown and white dog is running on the grass"
image = model.generate_image(prompt=prompt, stream=True)
image.save("image.png")
