import torch
import os, json, sys
from dataset import Internvideo2Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from safetensors.torch import load_file

backbone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


sys.path.append(backbone_path)

from custom_model import Internvideo2VisionCustom
from custom_vit_stacked import PretrainVisionTransformer_clean

from accelerate import Accelerator

import torch
from tqdm import tqdm

def evaluate(model, dataloader):
    model.eval() 
    total_samples = 0
    correct_predictions = 0


    with torch.no_grad():  
        for batch in tqdm(dataloader, desc="Evaluating"):
            videos, labels = batch

            videos = videos.to('cuda')

      
            logits = model(videos) 

         
            preds = torch.argmax(logits, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_samples += 1

    accuracy = correct_predictions / total_samples

    print(f"Accuracy: {accuracy*100:.2f}%")

    return accuracy


dics = torch.load('../checkpoint.pth', map_location='cpu')

model_dics = dics["model_state_dict"]

model_dics = {
            k.replace('module.', ''): v
            for k, v in model_dics.items() if k.startswith('module.')
        }

vision_encoder = PretrainVisionTransformer_clean()
model = Internvideo2VisionCustom(vision_encoder)
model.load_state_dict(model_dics, strict=True)
model = model.to('cuda')



test_dataset = Internvideo2Dataset()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)


evaluate(model, test_dataset)

