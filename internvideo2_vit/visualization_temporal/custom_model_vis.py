import os
import torch
import torch.nn as nn
import json
from safetensors import safe_open
from custom_vit_stacked_vis import PretrainVisionTransformer_clean
from safetensors.torch import load_file

class Internvideo2VisionCustom(nn.Module):
    def __init__(self, vision_encoder: nn.Module):
        super().__init__()
        self.vision_encoder = vision_encoder
        embed_dim = 768
        self.classifier = nn.Linear(embed_dim, 174)

    @classmethod
    def from_pretrained(cls, safetensor_folder: str, index_file: str = "model.safetensors.index.json"):
        index_path = os.path.join(safetensor_folder, index_file)

        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data['weight_map']

        tensor_file_to_keys = {}
        for key, filename in weight_map.items():
            tensor_file_to_keys.setdefault(filename, []).append(key)

        all_tensors = {}
        for shard_filename, keys in tensor_file_to_keys.items():
            shard_path = os.path.join(safetensor_folder, shard_filename)
            print(f"Loading shard: {shard_path}")
            
            with safe_open(shard_path, framework="pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    all_tensors[key] = tensor

        vision_encoder = PretrainVisionTransformer_clean()
        vision_encoder_state_dict = {
            k.replace('vision_encoder.', ''): v
            for k, v in all_tensors.items() if k.startswith('vision_encoder.')
        }
        vision_encoder.load_state_dict(vision_encoder_state_dict, strict=False)
        print("Loaded vision_encoder weights.")


        model = cls(vision_encoder=vision_encoder)

        return model

    def forward(self, x, return_attn=False):
        if return_attn:
            x_vis, x_pool_vis, _, _, maps = self.vision_encoder(x, return_attn=return_attn)
        else:
            x_vis, x_pool_vis, _, _ = self.vision_encoder(x)
        logits = self.classifier(x_pool_vis)
        if return_attn:
            return logits, maps
        else:
            return logits
