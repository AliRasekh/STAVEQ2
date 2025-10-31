import torch
import os, json, sys
from dataset import Internvideo2Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from custom_vit_stacked_vis import PretrainVisionTransformer_clean
from custom_model_vis import Internvideo2VisionCustom
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from torchvision.transforms import ToPILImage
from PIL import Image

import torch

import os
import numpy as np
import cv2
import torch
from torchvision.transforms.functional import to_pil_image



def compute_rollout_from_spatial(attentions_dict_list):
    result = torch.eye(attentions_dict_list[0]['spatial_attn'].size(-1)).to(attentions_dict_list[0]['spatial_attn'].device)
    for attn_dict in attentions_dict_list:
        spatial_attn = attn_dict['spatial_attn']
        attn_heads_fused = spatial_attn.mean(dim=1)
        attn_with_residual = attn_heads_fused + torch.eye(attn_heads_fused.size(-1)).to(attn_heads_fused.device)
        attn_with_residual /= attn_with_residual.sum(dim=-1, keepdim=True)
        result = attn_with_residual @ result
    return result


def visualize_video_attention(video_tensor, attentions_dict_list, output_dir="attn_visualization"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "original_frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "attention_maps"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)
    T, C, H, W = frames.shape
    L = (H // 14) * (W // 14)
    N = 1 + T * L

    rollout = compute_rollout_from_spatial(attentions_dict_list)
    cls_to_patch_attention = rollout[:, 0, 1:]

    att_per_frame = cls_to_patch_attention.view(1, T, L).mean(dim=1)
    att_map = att_per_frame[0].view(16, 16).cpu().numpy()


    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
    att_map_resized = cv2.resize(att_map, (224, 224))

    to_pil = ToPILImage()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)



    def denormalize(tensor):
        return (tensor * std) + mean

    for t in range(T):
        
        frame = video_tensor[t]
        mean = mean.to(frame.device)
        std = std.to(frame.device)
        frame = denormalize(frame)
        frame = torch.clamp(frame, 0, 1)
        frame_img = to_pil(frame)

        frame_np = np.array(frame_img)
        heatmap = cv2.applyColorMap(np.uint8(255 * att_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(0.6 * frame_np + 0.4 * heatmap)

        frame_img.save(os.path.join(output_dir, "original_frames", f"frame_{t}.png"))
        plt.imsave(os.path.join(output_dir, "attention_maps", f"attn_{t}.png"), att_map_resized, cmap='jet')
        Image.fromarray(overlay).save(os.path.join(output_dir, "overlay", f"overlay_{t}.png"))





dics = torch.load('../saves/checkpoint_head_22.pth', map_location='cpu')

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


model.eval() 

with torch.no_grad():  
    for batch in test_dataset:
        videos, labels = batch

        videos = videos.to('cuda')
  
        logits, attentions = model(videos, return_attn=True)
        break

frames = videos[0].permute(1, 0, 2, 3)

rollout = compute_rollout_from_spatial(attentions)
visualize_video_attention(frames, attentions, output_dir="vis_spatial")

