import torch
from dataset import SSv2
from model_1_0 import CustomQwen2VLForConditionalGeneration
from evaluate import calculate_accuracy

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

dataset = SSv2()

training_args = SFTConfig(
    output_dir="saves/model_1_0",
    overwrite_output_dir=True,
    do_train=True, bf16=True,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    warmup_ratio=0.05,
    remove_unused_columns=False,
    dataset_kwargs={
        "skip_prepare_dataset": True
    },
    dataloader_num_workers=2,
    save_strategy="steps",
    save_steps=1000,
    log_level="debug",
    log_level_replica="debug",
    logging_steps=16,
)

################################################################################

model = CustomQwen2VLForConditionalGeneration.from_pretrained(
    "saves/base_1_0", torch_dtype="auto", device_map="auto"
)

lora_config_all = LoraConfig(
    r=16,
    lora_alpha=16,
    bias="none",
    target_modules=[
        "attn.qkv", "attn.proj",
        "mlp.fc1", "mlp.fc2",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "merger.mlp.0", "merger.mlp.2"
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

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=dataset.collate,
    tokenizer=dataset.preprocessor.tokenizer,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)

model = trainer.model.merge_and_unload()
model.save_pretrained("saves/model_1_0_final")

################################################################################

dataset = SSv2(test=True)

answers = []
model.eval()

for i in range(len(dataset)):

    inp = dataset.collate([dataset[i]]).to(model.device)
    ans = model.generate(**inp, max_new_tokens=32)
    ans = dataset.preprocessor.tokenizer.decode(ans[0][len(inp['input_ids'][0]):])
    answers.append(f"--------------------------------------------------\n")
    answers.append(f"LLM: {ans}\n")
    answers.append(f"GT: {dataset[i][1]['content'][0]['text']}\n")

with open('model_1_0_result.txt', 'w') as file:
    file.writelines(answers)

calculate_accuracy('model_1_0_result.txt')

################################################################################

torch.cuda.empty_cache()
