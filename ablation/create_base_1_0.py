from transformers import Qwen2VLForConditionalGeneration
from model_1_0 import CustomQwen2VLForConditionalGeneration

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="cpu"
)

model = CustomQwen2VLForConditionalGeneration(base_model.config)
model = model.to(base_model.dtype).to(base_model.device)
model.load_state_dict(base_model.state_dict(), strict=False)
model.save_pretrained("saves/base_1_0")
