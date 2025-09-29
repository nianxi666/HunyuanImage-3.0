from transformers import AutoModelForCausalLM
import torch

# Load the model
model_id = "./HunyuanImage-3"

kwargs = dict(
    attn_implementation="sdpa",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    moe_impl="eager",
)

# Load the model AND explicitly move all its components to the GPU
model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to("cuda")
model.load_tokenizer(model_id)

# Generate the image
prompt = "A brown and white dog is running on the grass"
image = model.generate_image(prompt=prompt, stream=True)
image.save("image.png")

print("Image saved successfully to image.png!")
