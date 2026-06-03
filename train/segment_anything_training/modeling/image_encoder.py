# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
import numpy as np
import cv2

from .ChannelViT.image_encoder_withChannel import PatchEmbedPerChannelV2_5
from .ChannelViT.channel_patch_embed_legacy import (
    PatchEmbedPerChannel,
    PatchEmbedPerChannelV1_1,
    PatchEmbedPerChannelV2,
    PatchEmbedPerChannelV2_1,
    PatchEmbedPerChannelV2_2,
    PatchEmbedPerChannelV2_3,
    PatchEmbedPerChannelV2_4,
    PatchEmbedPerChannelV2_6,
    PatchEmbedPerChannelV2_7,
    PatchEmbedPerChannelV3,
)
from .feature_vis import *
from functools import partial
from typing import Optional, Tuple, Type
import loralib
# from loralib.layers import Linear
# from segment_anything_training.modeling.common import MLPBlock, LayerNorm2d
from .common import LayerNorm2d, MLPBlock
import os


class _LegacyPatchEmbedV1Adapter(nn.Module):
    """Normalize legacy V1 output to the newer patch-embed interface."""

    def __init__(self, module: nn.Module, in_chans: int) -> None:
        super().__init__()
        self.module = module
        self.in_chans = in_chans

    def forward(self, x: torch.Tensor, train_model: bool):
        x = self.module(x)
        ortho_loss = torch.tensor(0.0, device=x.device)
        return x, self.in_chans, ortho_loss


def _build_v2_5_patch_embed(**kwargs) -> nn.Module:
    path_embed_version = kwargs.pop("path_embed_version")
    shared_kwargs = dict(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=False,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    )
    version_overrides = MAINLINE_CHANNEL_PATCH_EMBED_CONFIGS[path_embed_version]
    shared_kwargs.update(version_overrides)
    return PatchEmbedPerChannelV2_5(**shared_kwargs)


MAINLINE_CHANNEL_PATCH_EMBED_CONFIGS = {
    "2_5": {"ortho_loss_lambda": 0.1, "Attention_Upsample_V": 1},
    "2_5_2": {"ortho_loss_lambda": 0.0, "Attention_Upsample_V": 2},
    "2_5_3": {"ortho_loss_lambda": 0.0, "Attention_Upsample_V": 3},
    "2_5_4": {"ortho_loss_lambda": 0.0, "Attention_Upsample_V": 4},
}


LEGACY_CHANNEL_PATCH_EMBED_BUILDERS = {
    "1_1": lambda **kwargs: _LegacyPatchEmbedV1Adapter(
        PatchEmbedPerChannelV1_1(
            in_chans=kwargs["in_chans"],
            embed_dim=kwargs["embed_dim"],
        ),
        in_chans=kwargs["in_chans"],
    ),
    "2": lambda **kwargs: PatchEmbedPerChannelV2(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=False,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    ),
    "2_1": lambda **kwargs: PatchEmbedPerChannelV2_1(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=False,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    ),
    "2_2": lambda **kwargs: PatchEmbedPerChannelV2_2(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=False,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    ),
    "2_3": lambda **kwargs: PatchEmbedPerChannelV2_3(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=False,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    ),
    "2_4": lambda **kwargs: PatchEmbedPerChannelV2_4(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=False,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    ),
    "2_6": lambda **kwargs: PatchEmbedPerChannelV2_6(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=False,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    ),
    "2_7": lambda **kwargs: PatchEmbedPerChannelV2_7(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=True,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
    ),
    "2_7_2": lambda **kwargs: PatchEmbedPerChannelV2_7(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=True,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
        Attention_Upsample_V=2,
    ),
    "2_7_3": lambda **kwargs: PatchEmbedPerChannelV2_7(
        img_size=kwargs["img_size"],
        patch_size=kwargs["patch_size"],
        in_chans=kwargs["in_chans"],
        embed_dim=kwargs["embed_dim"],
        enable_sample=True,
        ortho_loss_lambda=0.1,
        proxy_loss_lambda=0.1,
        orthogonal_init=True,
        use_orth_loss=kwargs["use_orth_loss"],
        use_channelToken=kwargs["use_channelToken"],
        Attention_Upsample_V=3,
    ),
}


def build_channel_patch_embed(
    *,
    path_embed_version: str,
    img_size: int,
    patch_size: int,
    in_chans: int,
    embed_dim: int,
    use_orth_loss: bool,
    use_channelToken: bool,
) -> nn.Module:
    kwargs = dict(
        path_embed_version=path_embed_version,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        use_orth_loss=use_orth_loss,
        use_channelToken=use_channelToken,
    )
    if path_embed_version in MAINLINE_CHANNEL_PATCH_EMBED_CONFIGS:
        return _build_v2_5_patch_embed(**kwargs)
    if path_embed_version in LEGACY_CHANNEL_PATCH_EMBED_BUILDERS:
        return LEGACY_CHANNEL_PATCH_EMBED_BUILDERS[path_embed_version](**kwargs)
    raise ValueError(
        "Unsupported pathEmbed_v="
        f"{path_embed_version}. Supported mainline versions: "
        f"{sorted(MAINLINE_CHANNEL_PATCH_EMBED_CONFIGS.keys())}; "
        f"supported legacy versions: {sorted(LEGACY_CHANNEL_PATCH_EMBED_BUILDERS.keys())}"
    )

# copy from https://github.com/sustainlab-group/SatMAE.git
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb
def get_2d_sincos_pos_embed(embed_dim, grid_size, batch_size=1):
    """
    Generate 2D sine-cosine positional embeddings in [B, H, W, D] format.

    Args:
        embed_dim: int, embedding dimension (D)
        grid_size: int, number of patches along height and width (H = W = img_size // patch_size)
        batch_size: int, number of batches (B)

    Returns:
        pos_embed: [batch_size, grid_size, grid_size, embed_dim]
    """
    # 1. Generate coordinate grid
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0)  # shape: (2, grid_size, grid_size)

    grid = grid.reshape(2, -1)  # shape: (2, grid_size*grid_size)

    # 2. Get positional encoding
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # shape: (grid_size*grid_size, embed_dim)

    # 3. Reshape to [H, W, D]
    pos_embed = pos_embed.reshape(grid_size, grid_size, embed_dim)  # (H, W, D)

    # 4. Expand to batch size: [B, H, W, D]
    pos_embed = np.tile(pos_embed[None, :, :, :], (batch_size, 1, 1, 1))  # (B, H, W, D)

    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def manually_load_qkv_lora_weights(model, state_dict):
    for name, module in model.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv,  loralib.layers.Linear):
            # 构建完整的原始qkv名称（根据实际结构调整）
            qkv_key = f"{name}.qkv.weight"
            if qkv_key in state_dict:
                full_weight = state_dict[qkv_key]  # [3*embed_dim, in_dim]
                
                
                if hasattr(module.qkv, 'base_layer'):
                    module.qkv.base_layer.weight.data.copy_(full_weight)
                else:
                    module.qkv.weight.data.copy_(full_weight)
                    
                print(f"[OK] Loaded qkv weights into LoRALinear: {qkv_key}")
            else:
                print(f"[WARN] QKV weight not found for: {qkv_key}")


def replace_qkv_with_lora(module, r=8, lora_alpha=16, lora_dropout=0.1):
    for name, child in module.named_children():
        if isinstance(child, Block):
            # 定位到Attention模块中的qkv层
            original_qkv = child.attn.qkv
            
            # 创建LoRA版本的Linear层（继承原始维度参数）
            lora_qkv = loralib.layers.Linear(
                in_features=original_qkv.in_features,
                out_features=original_qkv.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=(original_qkv.bias is not None),
                fan_in_fan_out=False, 
                merge_weights=False
            )
            
            # # 关键步骤：继承原始权重
            # lora_qkv.weight = original_qkv.weight  # 继承原始参数
            # if original_qkv.bias is not None:
            #     lora_qkv.bias = original_qkv.bias
            
            # 替换原始qkv层
            child.attn.qkv = lora_qkv
            
        else:
            # 递归处理子模块
            replace_qkv_with_lora(child, r, lora_alpha, lora_dropout)
class ChannelImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        set_lora: bool = False,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        ) # image to patch embedding

        self.pos_embed: Optional[nn.Parameter] = None

        #TODO: How to add MultiSpectral positional embedding?

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            block = Block( #Transformer block TODO: understand the parameters/Set LoRA
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            if set_lora:
                original_qkv = block.attn.qkv
                # block.attn.qkv = loralib.layers.Linear()
                lora_qkv = loralib.layers.Linear(
                    in_features=original_qkv.in_features,
                    out_features=original_qkv.out_features,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    fan_in_fan_out=False,
                    merge_weights=True,
                    bias=original_qkv.bias is not None,
                    device=original_qkv.weight.device,
                    dtype=original_qkv.weight.dtype,
                )
                # 复制原参数
                lora_qkv.weight = original_qkv.weight  # 继承原始参数
                if original_qkv.bias is not None:
                    lora_qkv.bias = original_qkv.bias
                lora_qkv.load_state_dict(original_qkv.state_dict(), strict=False)  # strict=False忽略LoRA参数
                # 替换
                block.attn.qkv = lora_qkv
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        interm_embeddings=[]
        for blk in self.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        
        return x, interm_embeddings

class GroupFusion(nn.Module):
    def __init__(self, num_groups: int, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        input_dim = num_groups * embed_dim
        hidden_dim = input_dim * 2   
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):  # x: (B, H, W, G*D)
        return self.fusion(x)  # (B, H, W, D)

class DecoderWithSkip(nn.Module):
    def __init__(self, in_main_channels=64, skip_channels=768, out_channels=3):
        super().__init__()
        self.reduce_skip = nn.Conv2d(skip_channels, in_main_channels, kernel_size=1)

        self.up1 = nn.Sequential(  # 64 → 256
            nn.Conv2d(in_main_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 256→512
        )
        self.up2 = nn.Sequential(  # 256 → 1024
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 512→1024
        )
        self.up3 = nn.Sequential(  # 256 → 1024
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),  # 512→1024
        )
        self.up4 = nn.Sequential(  # 256 → 1024
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=2, stride=2),  # 512→1024
        )

    def forward(self, x, skip_feats):  # x: [B, 256, 256, 64]; skip_feats: list of [B, 64, 64, 768]
        # x = x.permute(0, 3, 1, 2)  # → [B, 64, 256, 256]

        # 融合一个或多个中间特征
        for skip in skip_feats:
            skip = skip.permute(0, 3, 1, 2)  # [B, 768, 64, 64]
            skip = self.reduce_skip(skip)   # [B, 64, 64, 64]
            skip_up = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip_up
            

        x = self.up1(x)  # [B, 64, 512, 512]
        x = self.up2(x)  # [B, 3, 1024, 1024]
        x = self.up3(x)  # [B, 3, 2048, 2048]
        x = self.up4(x)
        return x

# 添加了解码module，使其完成解码任务。
class ChannelCompressViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        set_lora: bool = False,
        channel_embed: int = 256,
        use_channel_embedding: bool=False,
        channel_groups=((0,1,2),(3,))
    ) -> None:
        super().__init__()
        path_embed = embed_dim-channel_embed
        self.img_size = img_size
        if use_channel_embedding:
            self.patch_embed = nn.ModuleList([PatchEmbed(kernel_size=(patch_size, patch_size),stride=(patch_size, patch_size), in_chans=len(group), embed_dim=embed_dim)
                                          for group in channel_groups])
            G = len(channel_groups)
            D = embed_dim
            self.group_fusion = GroupFusion(num_groups=G, embed_dim=D)

        else:
            self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim,
            ) # image to patch embedding

        self.pos_embed: Optional[nn.Parameter] = None
        
        #TODO: How to add MultiSpectral positional embedding?
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            if not use_channel_embedding:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
                )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, img_size // patch_size, img_size // patch_size, path_embed)
                )
                # path_num = int(img_size // patch_size ** 2)
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], img_size // patch_size)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.use_channel_embedding = use_channel_embedding
        self.channel_groups = channel_groups
        if use_channel_embedding:
            num_groups = len(channel_groups)
            self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed), requires_grad=True)
            channel_pos = torch.arange(len(self.channel_groups)).numpy()
            channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1],channel_pos)
            self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            block = Block( #Transformer block TODO: understand the parameters/Set LoRA
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        
        self.decoder = DecoderWithSkip(in_main_channels=out_chans, skip_channels=embed_dim, out_channels=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_channel_embedding:
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            interm_embeddings=[]
            for blk in self.blocks:
                x = blk(x)
                if blk.window_size == 0:
                    interm_embeddings.append(x)
            x = self.neck(x.permute(0, 3, 1, 2))
            reconstructed_image = self.decoder(x, interm_embeddings)
            return x, interm_embeddings , reconstructed_image
        else:
            # Step 1: Patch embed for each channel group
            b, c, h, w = x.shape
            x_c_embed = []
            for i, group in enumerate(self.channel_groups):
                x_c = x[:, group, :, :]  # (B, Cg, H, W)
                x_c_embed.append(self.patch_embed[i](x_c))  # → (B, L, D)

            x = torch.stack(x_c_embed, dim=1)  # → (B, G, L, D)
            B, G, H_, W_, D = x.shape
            # Step 2: Add positional + channel embedding
            channel_embed = self.channel_embed.unsqueeze(2).unsqueeze(3)  # (1, G, 1, 1, cD)
            channel_embed = channel_embed.expand(-1, G, H_, W_, -1)

            # Pos embed: assume self.pos_embed shape is (1, H', W', pD)
            pos_embed = self.pos_embed.unsqueeze(1).expand(-1, G, -1, -1, -1)  # (1, G, H', W', pD)

            # Combine
            pos_channel = torch.cat([pos_embed, channel_embed], dim=-1)  # (1, G, H', W', D)
            x = x + pos_channel  # (B, G, H', W', D)

            x = x.permute(0, 2, 3, 1, 4)  # (B, H, W, G, D)
            x = x.reshape(B, H_, W_, G * D)  # (B, H, W, G*D)
            x = self.group_fusion(x)       # (B, H, W, D)
            # Step 3: Flatten group and token dims: (B, G, L, D) → (B, G*L, D)
            # x = x.view(b, G * L, D)
            interm_embeddings=[]
            for blk in self.blocks:
                x = blk(x)
                if blk.window_size == 0:
                    interm_embeddings.append(x)
            x = self.neck(x.permute(0, 3, 1, 2))

            reconstructed_image = self.decoder(x, interm_embeddings)
            # reconstructed_image = reconstructed_image.permute(0, 2, 3, 1)  # (B, H, W, C)
            return x, interm_embeddings , reconstructed_image

class ChannelTokenImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
        pathEmbed_v='2_1',
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.pathEmbed_v = pathEmbed_v
        self.patch_embed = build_channel_patch_embed(
            path_embed_version=pathEmbed_v,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_orth_loss=use_orth_loss,
            use_channelToken=use_channelToken,
        )
        path_embed = embed_dim
        self.pos_embed: Optional[nn.Parameter] = None
        
        #TODO: How to add MultiSpectral positional embedding?

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, path_embed)
            )
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            block = Block( #Transformer block TODO: understand the parameters/Set LoRA
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor, train_model=True) -> torch.Tensor:
        x,cin,ortho_loss = self.patch_embed(x,train_model)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        interm_embeddings=[]
        for blk in self.blocks:
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings,ortho_loss


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        set_lora: bool = False,
        channel_embed: int = 256,
        use_channel_embedding: bool=False,
        channel_groups=((0,1,2),(3,))
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        
        self.img_size = img_size
        if use_channel_embedding:
            self.patch_embed = nn.ModuleList([PatchEmbed(kernel_size=(patch_size, patch_size),stride=(patch_size, patch_size), in_chans=len(group), embed_dim=embed_dim)
                                          for group in channel_groups])
            G = len(channel_groups)
            D = embed_dim
            self.group_fusion = GroupFusion(num_groups=G, embed_dim=D)
            path_embed = embed_dim-channel_embed
            # self.patch_embed = nn.ModuleList([PatchEmbed(
            #     kernel_size=(patch_size, patch_size),
            #     stride=(patch_size, patch_size),
            #     in_chans=in_chans,
            #     embed_dim=embed_dim,
            # )])
        else:
            self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim,
            ) # image to patch embedding
            path_embed = embed_dim
        self.pos_embed: Optional[nn.Parameter] = None
        
        #TODO: How to add MultiSpectral positional embedding?

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, path_embed)
            )
        self.use_channel_embedding = use_channel_embedding
        self.channel_groups = channel_groups
        if use_channel_embedding:
            num_groups = len(channel_groups)
            self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed), requires_grad=False)
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            block = Block( #Transformer block TODO: understand the parameters/Set LoRA
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_channel_embedding:
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            interm_embeddings=[]
            for blk in self.blocks:
                x = blk(x)
                if blk.window_size == 0:
                    interm_embeddings.append(x)
            x = self.neck(x.permute(0, 3, 1, 2))
        else:
            # Step 1: Patch embed for each channel group
            b, c, h, w = x.shape
            x_c_embed = []
            for i, group in enumerate(self.channel_groups):
                x_c = x[:, group, :, :]  # (B, Cg, H, W)
                x_c_embed.append(self.patch_embed[i](x_c))  # → (B, L, D)

            x = torch.stack(x_c_embed, dim=1)  # → (B, G, L, D)
            B, G, H_, W_, D = x.shape
            # Step 2: Add positional + channel embedding
            # Channel embed: (1, G, 1, 1, cD)
            channel_embed = self.channel_embed.unsqueeze(2).unsqueeze(3)  # (1, G, 1, 1, cD)
            channel_embed = channel_embed.expand(-1, G, H_, W_, -1)

            # Pos embed: assume self.pos_embed shape is (1, H', W', pD)
            pos_embed = self.pos_embed.unsqueeze(1).expand(-1, G, -1, -1, -1)  # (1, G, H', W', pD)

            # Combine
            pos_channel = torch.cat([pos_embed, channel_embed], dim=-1)  # (1, G, H', W', D)
            x = x + pos_channel  # (B, G, H', W', D)

            x = x.permute(0, 2, 3, 1, 4)  # (B, H, W, G, D)
            x = x.reshape(B, H_, W_, G * D)  # (B, H, W, G*D)
            x = self.group_fusion(x)       # (B, H, W, D)
            # Step 3: Flatten group and token dims: (B, G, L, D) → (B, G*L, D)
            # x = x.view(b, G * L, D)
            interm_embeddings=[]
            for blk in self.blocks:
                x = blk(x)
                if blk.window_size == 0:
                    interm_embeddings.append(x)
            x = self.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0: # window_size ==0   global attention
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x
        


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
if __name__ == '__main__':  
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    # image_embedding_size = image_size // vit_patch_size
    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]
    vit_channel= ChannelCompressViT( #
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        in_chans =4,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        use_channel_embedding=True,
    )
    vit_channel.to('cuda')
    # param load
    checkpoint_path = "/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_b_01ec64.pth"
    state_dict = torch.load(checkpoint_path, map_location='cuda')
    vit_prefix = 'image_encoder.'  
    vit_state_dict = {
        k[len(vit_prefix):]: v for k, v in state_dict.items() if k.startswith(vit_prefix)
    }
    vit_state_dict = {
        k: v for k, v in vit_state_dict.items()
        if not (k.startswith("patch_embed") or k.startswith("pos_embed") or k.startswith("group_fusion") or k.startswith("decoder"))
    }

    # 加载参数，strict=False 以防有部分不匹配（比如额外添加了 channel_embed）
    missing_keys, unexpected_keys = vit_channel.load_state_dict(vit_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Step 4: 输入一个假图像（或者你自己的图像）
    image_path = "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/train/MultiSpectral/AT_4560_S2_10m_256.tif"
    image = iio.imread(image_path)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).float().to('cuda') * 255
    # dummy_input = torch.randn(1, 4, 1024, 1024).to('cuda')  # 输入 shape 按你的 patch 设置而定

    # Step 5: Forward 推理
    vit_channel.eval()
    with torch.no_grad():
        result = vit_channel(image_tensor)
        if len(result) == 3:
            compress_embed, interm, reconstruct_image  = result
        else:
            compress_embed, interm = result
        # compress_embed, interm, reconstruct_image  = vit_channel(image_tensor)
    
    image_RGB = reconstruct_image.cpu().numpy().squeeze()
    image_RGB = np.clip(image_RGB, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    image_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite("vit_reconstruct_image.png", image_RGB)
    a = 1
