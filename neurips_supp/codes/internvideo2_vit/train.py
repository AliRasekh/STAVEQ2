from argparse import Namespace
from datetime import timedelta
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from safetensors.torch import save_file
from modeling_videochat2 import InternVideo2_VideoChat2
from custom_vit_stacked import PretrainVisionTransformer_clean
from custom_model import Internvideo2VisionCustom
from model_config import VideoChat2Config
from transformers import AutoTokenizer, AutoConfig, AutoModel
import json, torch
from easydict import EasyDict
from dataset import Internvideo2Dataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.optim.lr_scheduler import CosineAnnealingLR


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    videos, labels = zip(*batch)
    videos = torch.stack(videos)
    labels = torch.tensor(labels)
    return videos, labels



with open("base_model/1B_ft_ssv2_f8.pth", "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")

state_dict = checkpoint['module']

new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace(".gamma", "")
    new_state_dict[new_key] = v



vision_encoder = PretrainVisionTransformer_clean()
vision_encoder.load_state_dict(new_state_dict, strict=False)
model = Internvideo2VisionCustom(vision_encoder)


accelerator = Accelerator(mixed_precision="bf16")

training_args = Namespace(
    gradient_checkpointing=False,
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_train_epochs=5
)



for param in model.parameters():
    param.requires_grad = False

for idx in range(40):
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



for param in model.classifier.parameters():
    param.requires_grad = True


if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_reentrant = False
    model.enable_input_require_grads()



train_dataset = Internvideo2Dataset()
train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)


trainable_params = filter(lambda p: p.requires_grad, model.parameters())

optimizer = optim.AdamW(trainable_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

criterion = nn.CrossEntropyLoss()

torch.cuda.empty_cache()
train_loader, model, optimizer = accelerator.prepare(
    train_loader, model, optimizer
)

checkpoint_path = "checkpoint.pth"

num_epochs = 1

total_steps = num_epochs * len(train_loader)

scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        if data is None:
            continue
        inputs, labels = data

        optimizer.zero_grad()

        inputs = inputs[0]

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        accelerator.backward(loss)


        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if i % 20 == 1:
            last_loss = running_loss / i
            print('batch {} loss: {}'.format(i - 1, last_loss))
        
        if i % 800 == 1:
            torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

    accelerator.wait_for_everyone()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

final_model_path = 'saves/full_train_balanced/custom_vit.safetensors'
save_file({k: v.cpu() for k, v in model.state_dict().items()}, final_model_path)
print(f"Training complete! Final model saved to {final_model_path}")
