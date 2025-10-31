import torch
from dataset import SSv2
from transformers import Qwen2VLForConditionalGeneration

from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer


dataset = SSv2()

training_args = SFTConfig(
    output_dir="saves/base",
    overwrite_output_dir=True,
    do_train=True, bf16=True,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
 
    remove_unused_columns=False,
    dataset_kwargs={
        "skip_prepare_dataset": True
    },
    save_strategy="no",

    log_level="debug",
    log_level_replica="debug",
    logging_steps=32
)


base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto"
)




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
        "attn.qkv": 64
    },
    alpha_pattern={
        "attn.qkv": 64
    }
)

base_model = get_peft_model(base_model, lora_config_all)

training_args.output_dir = "saves/base_2B"
training_args.num_train_epochs = 3

trainer = SFTTrainer(
    model=base_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=dataset.collate,
    tokenizer=dataset.preprocessor.tokenizer,
)

trainer.train()
base_model = trainer.model.merge_and_unload()
base_model.save_pretrained(training_args.output_dir)
torch.cuda.empty_cache()

