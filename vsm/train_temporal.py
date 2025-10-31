import torch
from dataset import SSv2
from transformers import Qwen2VLForConditionalGeneration
from custom_model import CustomQwen2VLForConditionalGeneration

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


dataset = SSv2()

training_args = SFTConfig(
    output_dir="saves",
    overwrite_output_dir=True,
    do_train=True, bf16=True,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    # warmup_ratio=0.1,
    remove_unused_columns=False,
    dataset_kwargs={
        "skip_prepare_dataset": True
    },
    save_strategy="no",
    # save_steps=1,
    log_level="debug",
    log_level_replica="debug",
    logging_steps=32
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "saves/base_2B", torch_dtype="auto"
)



model = CustomQwen2VLForConditionalGeneration(base_model.config)
model = model.to(base_model.dtype).to(base_model.device)
model.load_state_dict(base_model.state_dict(), strict=False)
del base_model
torch.cuda.empty_cache()




lora_config_all = LoraConfig(
    r=32,
    lora_alpha=32,
    bias="none",
    target_modules=[
        "attn.qkv", "attn.proj",
        "mlp.fc1", "mlp.fc2",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    rank_pattern={
        "attn.qkv": 128
    },
    alpha_pattern={
        "attn.qkv": 128
    }
)

model = get_peft_model(model, lora_config_all)

for name, param in model.named_parameters():
    if "temp_attn" in name or "norm3" in name:
        param.requires_grad = True

training_args.output_dir = "saves/temporal_2B"
training_args.num_train_epochs = 3

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=dataset.collate,
    tokenizer=dataset.preprocessor.tokenizer,
)

trainer.train()
model = trainer.model.merge_and_unload()
model.save_pretrained(training_args.output_dir)
torch.cuda.empty_cache()
