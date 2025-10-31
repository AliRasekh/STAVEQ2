import torch
from accelerate import Accelerator

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from transformers import AutoTokenizer
from dataset import SSv2
from modeling_videochat2 import InternVideo2_VideoChat2


training_args = SFTConfig(
    output_dir="saves/internvideo2_temporal_stacked",
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


for idx in range(39):
    block = model.vision_encoder.blocks[idx]

    if hasattr(block, 'ls3'):
        block.ls3.requires_grad_(True)

    if hasattr(block, 'norm3') and hasattr(block.norm3, 'weight'):
        block.norm3.weight.requires_grad_(True)

    if hasattr(block, 'temporal_attn'):
        ta = block.temporal_attn
        
        if hasattr(ta.proj, 'weight'):
            ta.proj.weight.requires_grad_(True)
        if hasattr(ta.proj, 'bias'):
            ta.proj.bias.requires_grad_(True)

        if hasattr(ta.qkv, 'weight'):
            ta.qkv.weight.requires_grad_(True)
        if hasattr(ta.qkv, 'bias'):
            ta.qkv.bias.requires_grad_(True)

    print(f"Reinitializing block {idx}")

    if hasattr(block, 'ls3'):
        model.vision_encoder._init_weights(block.ls3)

    if hasattr(block, 'norm3'):
        model.vision_encoder._init_weights(block.norm3)

    if hasattr(block.temporal_attn, 'proj'):
        model.vision_encoder._init_weights(block.temporal_attn.proj)

    if hasattr(block.temporal_attn, 'qkv'):
        model.vision_encoder._init_weights(block.temporal_attn.qkv)


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
