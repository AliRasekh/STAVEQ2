import torch
from accelerate import Accelerator

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from transformers import AutoTokenizer
from dataset import SSv2
from modeling_videochat2 import InternVideo2_VideoChat2


training_args = SFTConfig(
    output_dir="saves/internvideo2",
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
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    push_to_hub=False,
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False
)


training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}


dataset = SSv2()


model = InternVideo2_VideoChat2.from_pretrained(
    'base_model',
    torch_dtype="auto"
)

tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

for param in model.parameters():
    param.requires_grad = False


for idx, block in enumerate(model.vision_encoder.temporal_blocks):
    block.apply(model.vision_encoder._init_weights)

for idx, block in enumerate(model.vision_encoder.temporal_blocks):
    print(f"Unfreezing temporal block {idx}")
    for param in block.parameters():
        param.requires_grad = True

def set_inputs_require_grad(m):
    if hasattr(m, 'requires_grad_'):
        m.requires_grad_(True)

model.vision_encoder.patch_embed.apply(set_inputs_require_grad)



if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_reentrant = False
    model.enable_input_require_grads()


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=dataset.collate,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model()

torch.cuda.empty_cache()
