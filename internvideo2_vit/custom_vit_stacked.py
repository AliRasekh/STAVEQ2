import math
from copy import deepcopy
import logging
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from torch.nn import ParameterDict, ModuleDict
import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange


logger = logging.getLogger(__name__)

try:
    from .flash_attention_class import FlashAttention
except:
    logger.warn(f'flash_attn is not installed, you can install it by `pip install flash_attn` ')
try:
    from flash_attn.modules.mlp import FusedMLP
except:
    logger.warn(f'FusedMLP of flash_attn is not installed!!!')

try:
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except:
    logger.warn(f'DropoutAddRMSNorm of flash_attn is not installed!!!')

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# MVD: https://github.com/ruiwang2021/mvd/blob/main/modeling_finetune.py
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(
        embed_dim_spatial, grid
    )

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed_internvideo2(checkpoint_model, model, orig_t_size = 8):
    # interpolate position embedding
    for pos_name in ['pos_embed', 'clip_pos_embed']:
        if pos_name in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_name]
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # we use 8 frames for pretraining
            # new_t_size = args.num_frames * args.num_segments // model.patch_embed.tubelet_size
            new_t_size = model.num_frames // model.tubelet_size
            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (new_t_size))** 0.5)
            
            # class_token and dist_token are kept unchanged
            if orig_t_size != new_t_size:
                logger.info(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
                pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
                pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
                pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model[pos_name] = new_pos_embed
                pos_embed_checkpoint = new_pos_embed

            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model[pos_name] = new_pos_embed
    
    
    if 'pos_embed_spatial' in checkpoint_model or 'pos_embed_temporal' in checkpoint_model:
        raise NotImplementedError

def interpolate_pos_embed_internvideo2_new(checkpoint_model, model, orig_t_size = 8):
    pos_names = []
    for k in checkpoint_model.keys():
        if ('pos_embed' in k or 'clip_pos_embed' in k) and 'img_pos_embed' not in k: # NOTE 暂时不插值img_pos，高分辨率时可能需要再加
            pos_names.append(k)
    
    logger.info(f"pos names list for interpolating: {pos_names}")

    assert len(pos_names) > 0, checkpoint_model.keys()

    if 'pos_embed_spatial' in checkpoint_model.keys() or 'pos_embed_temporal' in checkpoint_model.keys():
        raise NotImplementedError
    
    # interpolate position embedding
    for pos_name in pos_names:

        pos_embed_checkpoint = checkpoint_model[pos_name]
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # we use 8 frames for pretraining
        # new_t_size = args.num_frames * args.num_segments // model.patch_embed.tubelet_size
        new_t_size = model.num_frames // model.tubelet_size
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (new_t_size))** 0.5)
        
        # class_token and dist_token are kept unchanged
        if orig_t_size != new_t_size:
            logger.info(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
            pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
            pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
            pos_embed_checkpoint = new_pos_embed

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
    
    

def interpolate_pos_embed(checkpoint_model, model, orig_t_size=4, pos_name='vision_encoder.pos_embed'):
    if pos_name in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_name]
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # we use 4 frames for pretraining
        new_t_size = model.T
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (new_t_size))** 0.5)
        
        # class_token and dist_token are kept unchanged
        if orig_t_size != new_t_size:
            print(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
            pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
            pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
            pos_embed_checkpoint = new_pos_embed

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_name] = new_pos_embed
    else:
        raise NotImplementedError



class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
        
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AttentiveBlock(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()
        
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        
        return x


class AttentionPoolingBlock(AttentiveBlock):
    
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, use_fused_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)
        
        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def _naive_attn(self, x):
        B, N, C = x.shape
        # print(x.shape)
        # print(x.shape, torch.cuda.memory_allocated(), torch.cuda.memory_allocated())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(qkv.shape)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        # torch.cuda.empty_cache()
        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        attn = self.attn_drop(attn)
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_allocated())
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print(f"\033[31m这{x.device}是{self.proj.weight.device} {self.proj.bias.device}\033[0m")
        # print(f"\033[31m类型{x.dtype}是{self.proj.weight.dtype} {self.proj.bias.dtype}\033[0m")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        
        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)
        
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        outs = self.proj_drop(outs)
        return outs
    
    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        # print("joint attention shape: ",x.shape)
        return x


class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, T, L, C = x.shape
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * L, T, C)

        qkv = self.qkv(x).reshape(B * L, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)
        out = out.transpose(1, 2).reshape(B * L, T, C)

        out = self.proj(out)
        out = self.proj_drop(out)

 
        out = out.reshape(B, L, T, C).permute(0, 2, 1, 3)
        
        return out




class StackedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=1e-5,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_flash_attn=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        with_cp=False,
        qk_normalization=False,
        layerscale_no_force_fp32=False,
        use_fused_rmsnorm=False
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=11,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_attn=use_flash_attn,
            causal=False,
            norm_layer=norm_layer,
            qk_normalization=qk_normalization,
            use_fused_rmsnorm=use_fused_rmsnorm
        )
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.temporal_attn = TemporalAttentionBlock(
            dim=dim,
            num_heads=22,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.ls3 = nn.Parameter(init_values * torch.ones(dim))
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm



    def forward(self, x, residual=None):
        def _inner_forward(x, residual=None):
            x = x + self.drop_path1(self.ls1 * self.attn(self.norm1(x)))

            cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]

            B, N_patch, C = patch_tokens.shape
            T = 8
            L = N_patch // T

            assert T * L == N_patch, f"Invalid reshape dims: T={T}, L={L}, N_patch={N_patch}"

            x_temp = patch_tokens.view(B, T, L, C)

            x_temp = x_temp + self.drop_path3(self.ls3 * self.temporal_attn(self.norm3(x_temp)))

            x_temp = x_temp.view(B, N_patch, C)

            x = torch.cat([cls_token, x_temp], dim=1)

            x = x + self.drop_path2(self.ls2 * self.mlp(self.norm2(x)))

            return x
        
        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., rank=16, lora_dropout=0.05):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        
        self.lora_dropout = ModuleDict({"default": nn.Dropout(lora_dropout)})

    
    def forward(self, x):
        base_out1 = self.fc1(x)
        x = base_out1
        x = self.act(x)
        x = self.drop1(x)
        
        base_out2 = self.fc2(x)
        x = base_out2
        x = self.drop2(x)
        return x



class Block(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_fused_rmsnorm=False):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
                              qk_normalization=qk_normalization,
                              use_fused_rmsnorm=use_fused_rmsnorm)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def forward(self, x, residual=None):
        
        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1 * self.attn(x) )
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2 * self.mlp(x) )
                return x, residual
            else:
                assert residual is None
                # print(x.shape)
                x = x + self.drop_path1(self.ls1 * self.attn(self.norm1(x)))
                x = x + self.drop_path2(self.ls2 * self.mlp(self.norm2(x)))
                return x
        
        if self.with_cp:
            # print(f"\033[31m use_checkpoint [0m")
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
            num_frames=8, tubelet_size=1, norm_layer=None
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            num_frames // tubelet_size, 
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        ) # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.num_img_patches = self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim, 
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]), 
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)  # B x C x T x HW => B x T x HW x C
        x = self.norm(x)
        return x

class PretrainVisionTransformer_clean(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False, # follow internvl_clip to set False
            drop_path_rate: float = 0.25, # may need ablation
            embed_dim: int = 1408,
            num_heads: int = 16,
            mlp_ratio: float = 48/11,
            init_values: float = 1e-5, # may need ablation
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = False,
            use_fused_rmsnorm: bool = False,
            use_fused_mlp: bool = False,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = False, # whether True for training?
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            sep_image_video_pos_embed: bool = True,
            use_checkpoint: bool = False,
            checkpoint_num: int = 48,
            # for unmasked teacher
            x_vis_return_idx=-1,
            x_vis_only=False
        ):
        super().__init__()
        print("MODE STACKED")
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, 'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent'
        
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim

        logger.info(f"Origin depth: {depth}")
        depth = depth + x_vis_return_idx + 1
        logger.info(f"New depth: {depth}")
        self.depth = depth

        self.x_vis_only = x_vis_only

        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        num_img_patches = self.patch_embed.num_img_patches
        # print(f"num_patches: {num_patches}, num_img_patches: {num_img_patches}")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        self.sep_image_video_pos_embed = sep_image_video_pos_embed
        if sep_pos_embed:
            raise NotImplementedError
        else:
            if sep_image_video_pos_embed:
                logger.info("Use joint position embedding, for image and video we use different pos_embed.")
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
            else:
                logger.info("Use joint position embedding, for image and video we use same pos_embed.")
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.patch_embed.num_patches // num_frames, embed_dim))

        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        logger.info(f"Droppath rate: {dpr}")
        logger.info(f"Checkpoint list: {with_cp_list}")
        

        self.blocks = nn.ModuleList([
            StackedBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])




        if not self.x_vis_only:
            self.clip_projector = AttentionPoolingBlock(
                dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)
            self.fc_norm = nn.LayerNorm(clip_embed_dim)
        

        
        self.init_pos_embed()
        self.init_pos_embed_4d()
        # trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)
        # self.fix_init_weight()

    def init_pos_embed(self):
        logger.info("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
                self.patch_embed.grid_size[0], # t_size
                cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            if self.sep_image_video_pos_embed:
                img_pos_embed = get_3d_sincos_pos_embed(
                    self.pos_embed.shape[-1], 
                    self.patch_embed.grid_size[1], # height & weight
                    1,
                    cls_token=True
                )
                self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

    def init_pos_embed_4d(self):
        logger.info("Init temporal + spatial pos_embed with sincos")

        temporal_pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim=self.temporal_pos_embed.shape[-1],
            pos=np.arange(self.num_frames)
        )
        temporal_pos_embed = torch.from_numpy(temporal_pos_embed).float().unsqueeze(0).unsqueeze(2)  # [1, T, 1, C]
        self.temporal_pos_embed.data.copy_(temporal_pos_embed)

        spatial_size = int((self.patch_embed.num_patches // self.num_frames) ** 0.5)
        # spatial
        grid_h = np.arange(spatial_size, dtype=np.float32)
        grid_w = np.arange(spatial_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        spatial_pos_embed = get_2d_sincos_pos_embed_from_grid(
            embed_dim=self.spatial_pos_embed.shape[-1],
            grid=grid
        )
        spatial_pos_embed = torch.from_numpy(spatial_pos_embed).float().unsqueeze(0).unsqueeze(1)  # [1, 1, L, C]
        self.spatial_pos_embed.data.copy_(spatial_pos_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 
            'pos_embed_spatial', 
            'pos_embed_temporal', 
            'pos_embed_cls',
            'img_pos_embed',
            'cls_token'
        }
    
    def expand_pos_embed(self, pos_embed, new_t_size, L, use_vitar_fuzzing=False):
        '''
        @param: 
            pos_embed: original pos_embed, (1, T*L + 1, embed_dim)
            T: frames
            L: w * h
            method: interpolation method
        '''
        pos_embed_checkpoint = pos_embed
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1
        
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(self.num_frames / self.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(L ** 0.5)
        
        # class_token and dist_token are kept unchanged
        if self.num_frames != new_t_size:
            logger.info(f"Temporal interpolate from {self.num_frames} to {new_t_size} ")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
            pos_tokens = pos_tokens.view(1, self.num_frames, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, self.num_frames)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens.cpu(), size=new_t_size, mode='linear').cuda()
            pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            pos_embed_checkpoint = new_pos_embed
        
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens.cpu(), size=(new_size, new_size), mode='bicubic', align_corners=False).cuda()
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        
        if use_vitar_fuzzing:
            ...
        
        return new_pos_embed
    
    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, mask=None, use_image=False):
        # print(x.shape)
        x = self.patch_embed(x.type(self.dtype))
        # print(x.shape)
        # print(f"x.shape: {x.shape} x.dtype: {x.dtype}, model.dtype: {self.dtype}")
        B, T, L, C = x.shape  # T: temporal; L: spatial
        x_joint = x.view([B, T * L, C])
        # print(x.shape)

        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_joint = torch.cat((cls_tokens, x_joint), dim=1)
        # print('posp: ',self.pos_embed.shape)
        # add pos_embed
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            if use_image:
                if self.sep_image_video_pos_embed:
                    pos_embed = self.img_pos_embed
                else:
                    # (1, num_img_patches + 1, embed_dim)
                    # print('origin pos_embed.shape:', self.pos_embed.shape)
                    cls_pos_embed = self.pos_embed[:, 0:1, :]
                    # print('cls_pos_embed.shape:', cls_pos_embed.shape)

                    img_pos_embed = self.pos_embed[:, 1:, :].view(1, self.num_frames, self.patch_embed.num_patches // self.num_frames, self.embed_dim).mean(dim=1)
                    # print('img_pos_embed.shape:', img_pos_embed.shape)

                    pos_embed = torch.cat([cls_pos_embed, img_pos_embed], dim=1)
                    # print('final img_pos_embed.shape:', pos_embed.shape)
            else:
                pos_embed = self.pos_embed
        # print("pos: ",pos_embed.shape)
        if pos_embed[0].shape != x_joint[0].shape:
            # print(f'pos embed shape {pos_embed.shape} does not match x[0].shape {x[0].shape}')
            pos_embed = self.expand_pos_embed(pos_embed, T, L) # can accelerate here
        assert pos_embed[0].shape == x_joint[0].shape, f'pos embed shape: {pos_embed.shape} not match x[0].shape {x_joint[0].shape}'
        # print("pos_embed.shape:", pos_embed.shape)
        # print('x shape:', x.shape)
        x_joint = x_joint + pos_embed

        # x = x + self.temporal_pos_embed + self.spatial_pos_embed

        # mask tokens, ~mask means visible
        if mask is not None:
            x_joint = x_joint[~mask].reshape(B, -1, C) 
        else:
            x_joint = x_joint.reshape(B, -1, C) 

        residual = None

        for idx, blk in enumerate(self.blocks):
            if isinstance(x_joint, tuple) and len(x) == 2:
                x_joint, residual = x_joint
            x_joint = blk(x_joint, residual=residual)

        x = x_joint

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
        
        x_vis = x
        if self.x_vis_only:
            return x_vis
        else:
            x_pool_vis = self.clip_projector(x_vis)
            x_pool_vis = self.fc_norm(x_pool_vis)
            return x_vis, x_pool_vis, None, None
    

def pretrain_internvideo2_giant_patch14_224_clean(config):
    model = PretrainVisionTransformer_clean(
        in_chans=3, img_size=224, patch_size=14,
        embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        attn_pool_num_heads=16, qkv_bias=False,
        drop_path_rate=0.25,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=config.vision_encoder.get('use_flash_attn', False),
        use_fused_rmsnorm=config.vision_encoder.get('use_fused_rmsnorm', False),
        use_fused_mlp=config.vision_encoder.get('use_fused_mlp', False),
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=True,
        num_frames=config.vision_encoder.num_frames,
        tubelet_size=config.vision_encoder.tubelet_size,
        sep_pos_embed=False,
        sep_image_video_pos_embed=config.vision_encoder.sep_image_video_pos_embed,
        use_checkpoint=config.vision_encoder.use_checkpoint,
        checkpoint_num=config.vision_encoder.checkpoint_num,
        x_vis_return_idx=config.vision_encoder.x_vis_return_idx,
        x_vis_only=config.vision_encoder.x_vis_only,
    )

    if config.vision_encoder.pretrained is not None:
        logger.info(f"Loading pretrained weights from {config.vision_encoder.pretrained}")
        state_dict = torch.load(config.vision_encoder.pretrained, map_location='cpu')
        interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=4) # NOTE 8f for stage1
        message = model.load_state_dict(state_dict, strict=False)
        logger.info(message)
    else:
        logger.info("No pretrained weights!!!")
    return model



def pretrain_internvideo2_6b_patch14_224_clean(config):
    model = PretrainVisionTransformer_clean(
        in_chans=3, img_size=224, patch_size=14,
        embed_dim=3200, depth=48, num_heads=25, mlp_ratio=4,
        clip_embed_dim=config.vision_encoder.clip_embed_dim,
        attn_pool_num_heads=16, qkv_bias=False,
        drop_path_rate=0.3,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=config.vision_encoder.get('use_flash_attn', True),
        use_fused_rmsnorm=config.vision_encoder.get('use_fused_rmsnorm', True),
        use_fused_mlp=config.vision_encoder.get('use_fused_mlp', True),
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=True,
        num_frames=config.vision_encoder.num_frames,
        tubelet_size=config.vision_encoder.tubelet_size,
        sep_pos_embed=False,
        sep_image_video_pos_embed=config.vision_encoder.sep_image_video_pos_embed,
        use_checkpoint=config.vision_encoder.use_checkpoint,
        checkpoint_num=config.vision_encoder.checkpoint_num,
        x_vis_return_idx=config.vision_encoder.x_vis_return_idx,
        x_vis_only=config.vision_encoder.x_vis_only
    )

    if config.vision_encoder.pretrained is not None:
        logger.info(f"Loading pretrained weights from {config.vision_encoder.pretrained}")
        state_dict = torch.load(config.vision_encoder.pretrained, map_location='cpu')
        interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=8) # NOTE 8f for stage1
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg)
    else:
        logger.info("No pretrained weights!!!")
    return model
