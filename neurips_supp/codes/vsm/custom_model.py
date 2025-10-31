from transformers import (
    Qwen2VLPreTrainedModel,
    Qwen2VLModel,
    Qwen2VLForConditionalGeneration,
)

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    VisionRotaryEmbedding, apply_rotary_pos_emb_vision,
    PatchEmbed, LayerNorm, VisionMlp, PatchMerger,
    QWEN2_VL_VISION_ATTENTION_CLASSES
)

import torch
from torch import nn
import torch.nn.functional as F
import math


class VisionTemporalAttention(nn.Module):

    temparal_dim_scale = 0.25

    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.hidden_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        self.num_heads = max(1, round(config.num_heads * self.temparal_dim_scale))
        self.neck_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(self.hidden_dim, self.neck_dim * 3, bias=True)
        self.proj = nn.Linear(self.neck_dim, self.hidden_dim)


    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> torch.Tensor:


        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states) \
            .reshape(-1, cu_seqlens[1], 3, self.num_heads, self.head_dim) \
            .permute(2, 1, 0, 3, 4).unbind(0) 


        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output) 
        return attn_output


class Qwen2VLVisionBlock(nn.Module):

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm3 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = QWEN2_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.embed_dim, num_heads=config.num_heads
        )
        self.temp_attn = VisionTemporalAttention(config)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)


    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, temporal_pos_emb) -> torch.Tensor:

        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.temp_attn(
            self.norm3(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=temporal_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.temporal_pos_emb = VisionRotaryEmbedding(head_dim)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)


        temporal_pos_emb = self.temporal_pos_emb(len(cu_seqlens)-1)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                temporal_pos_emb=temporal_pos_emb
            )

        return self.merger(hidden_states)


class CustomQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):

    def __init__(self, config):
        super(Qwen2VLForConditionalGeneration, self).__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        self.post_init()
