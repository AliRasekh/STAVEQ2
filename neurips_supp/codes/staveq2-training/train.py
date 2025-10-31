import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from model import CustomQwen2VLForConditionalGeneration
from dataset import WebvidQA


dataset = WebvidQA()
accelerator = Accelerator()
device = accelerator.device

training_args = SFTConfig(
    output_dir="saves/joint",
    overwrite_output_dir=True,
    do_train=True, bf16=True,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    warmup_ratio=0.02,
    remove_unused_columns=False,
    dataset_kwargs={
        "skip_prepare_dataset": True
    },
    dataloader_num_workers=8, 
    save_strategy="steps",
    save_steps=1000,
    log_level="debug",
    log_level_replica="debug",
    logging_steps=32,
)

################################################################################

model = CustomQwen2VLForConditionalGeneration.from_pretrained(
    "saves/base", torch_dtype="auto", device_map=device
)

lora_config_all = LoraConfig(
    r=16,
    lora_alpha=16,
    bias="none",
    target_modules=[
        "attn.qkv", "attn.proj",
        "mlp.fc1", "mlp.fc2",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    rank_pattern={
        "attn.qkv": 32
    },
    alpha_pattern={
        "attn.qkv": 32
    },
    modules_to_save=["temp_attn", "norm3"]
)

model = get_peft_model(model, lora_config_all)

if accelerator.is_main_process:
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=dataset.collate,
    tokenizer=dataset.preprocessor.tokenizer,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)

if accelerator.is_main_process:
    model = trainer.model.merge_and_unload()
    model.save_pretrained("saves/final")

################################################################################

accelerator.end_training()
torch.cuda.empty_cache()
