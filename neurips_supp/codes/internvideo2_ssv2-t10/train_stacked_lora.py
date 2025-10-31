import torch
from modeling_videochat2 import InternVideo2_VideoChat2
from model_config import VideoChat2Config
from dataset import SSv2
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="saves/internvideo2_temporal/lora",
    overwrite_output_dir="True",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    do_train=True,
    bf16=True,
    optim="adamw_torch_fused",
    log_level="debug",
    log_level_replica="debug",
    logging_steps=20,
    save_strategy="no",
    # save_steps=1,
    learning_rate=1e-4,
    # max_grad_norm=0.5,
    # warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    push_to_hub=False,
    gradient_checkpointing=False,
    ddp_find_unused_parameters=True
)

# LoRA configuration
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["qkv", "fc1", "fc2", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", \
                    "crossattention.self.query", "crossattention.self.key", "crossattention.self.value"],
)

training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

dataset = SSv2()


model = InternVideo2_VideoChat2.from_pretrained(
    'saves/internvideo2_temporal', torch_dtype="auto"
)



tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=dataset.collate,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model()

torch.cuda.empty_cache()