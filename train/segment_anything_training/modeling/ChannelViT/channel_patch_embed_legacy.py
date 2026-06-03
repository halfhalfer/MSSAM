# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
import numpy as np
import cv2
from einops import rearrange, repeat
import random
import math


from functools import partial
from typing import Optional, Tuple, Type
from .vitForChannel import trunc_normal_,Block
from .feature_vis import *
from .decoder import DPTDecoder,DPTSegmentationHead
from ..multispectral_encoder import ConvBlock
import loralib
from enum import Enum
# from loralib.layers import Linear
# from segment_anything_training.modeling.common import MLPBlock, LayerNorm2d
# from ..common import LayerNorm2d, MLPBlock
class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        """
        get all values from Enum
        """
        return list(map(lambda c: c.value, cls))


import matplotlib.pyplot as plt
import seaborn as sns







def ortho_proj_loss_fn_v2(features, labels, gamma_s, gamma_d, reverse_pos_pairs: bool, use_square: bool):
    """
    features: shape (b, num_tokens, d)
    labels: shape (num_tokens)
    gamma_s, gamma_d: lambda_s and lambda_d in E.q (2) and (3) in the paper
    reverse_pos_pairs: If true, we want each token to be orthogonal to all other tokens, regarless of their channels.
    """
    device = features.device
    #  features are normalized
    features = F.normalize(features, p=2, dim=-1)

    labels = labels[None, :, None]  # extend dims

    mask = torch.eq(labels, labels.transpose(-2, -1)).bool().to(device)
    eye = torch.eye(mask.shape[-2], mask.shape[-1]).bool().to(device).unsqueeze(0)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()
    dot_prod = torch.matmul(features, features.transpose(-2, -1))

    mask_pos_sum = mask_pos.sum(dim=(-2, -1)) + 1e-6
    mask_neg_sum = mask_neg.sum(dim=(-2, -1)) + 1e-6

    pos_pairs_mean = (mask_pos * dot_prod).sum(dim=(-2, -1)) / mask_pos_sum
    neg_pairs_mean = (mask_neg * dot_prod).sum(dim=(-2, -1)) / mask_neg_sum

    if use_square:
        neg_pairs_mean = neg_pairs_mean**2

    if reverse_pos_pairs:
        if use_square:
            pos_pairs_mean = pos_pairs_mean**2
        loss = gamma_s * pos_pairs_mean + gamma_d * neg_pairs_mean
    else:
        loss = gamma_s * (1.0 - pos_pairs_mean) + gamma_d * neg_pairs_mean
    return loss.mean()

def ortho_proj_loss_fn_v3(features, labels, gamma_s, gamma_d, reverse_pos_pairs: bool, use_square: bool, sample_size=3072):
    """
    features: shape (b, num_tokens, d)
    labels: shape (num_tokens)
    """
    device = features.device
    B, N, D = features.shape

    # ========== Step 1: flatten & sample ==========
    features = features.view(-1, D)    # (B*N, D)
    total_tokens = features.size(0)

    if total_tokens < sample_size:
        raise ValueError(f"Total tokens {total_tokens} < sample_size {sample_size}")

    # 随机采样 sample_size 个 token
    indices = torch.randperm(total_tokens, device=device)[:sample_size]
    features = features[indices]       # (sample_size, D)
    labels = labels[indices]           # (sample_size,)

    # ========== Step 2: 原始损失计算逻辑 ==========

    features = F.normalize(features, p=2, dim=-1)
    labels = labels[None, :, None]  # (1, N, 1)

    mask = torch.eq(labels, labels.transpose(-2, -1)).bool().to(device)
    eye = torch.eye(mask.shape[-2], device=device).bool().unsqueeze(0)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()
    dot_prod = torch.matmul(features, features.transpose(-2, -1))  # (1, N, N)

    mask_pos_sum = mask_pos.sum(dim=(-2, -1)) + 1e-6
    mask_neg_sum = mask_neg.sum(dim=(-2, -1)) + 1e-6

    pos_pairs_mean = (mask_pos * dot_prod).sum(dim=(-2, -1)) / mask_pos_sum
    neg_pairs_mean = (mask_neg * dot_prod).sum(dim=(-2, -1)) / mask_neg_sum

    if use_square:
        neg_pairs_mean = neg_pairs_mean ** 2

    if reverse_pos_pairs:
        if use_square:
            pos_pairs_mean = pos_pairs_mean ** 2
        loss = gamma_s * pos_pairs_mean + gamma_d * neg_pairs_mean
    else:
        loss = gamma_s * (1.0 - pos_pairs_mean) + gamma_d * neg_pairs_mean

    return loss.mean()


def pairwise_distance_v2(proxies, x, squared=False):
    if squared:
        return (torch.cdist(x, proxies, p=2)) ** 2
    else:
        return torch.cdist(x, proxies, p=2)
def band_entropy_maximization(x_proj, eps=1e-8):
    """
    对多波段特征进行信息熵最大化，鼓励不同波段编码多样信息
    
    Args:
        x_proj: torch.Tensor, shape (B, C, Bands, H, W)
        eps: 防止 log(0)
    
    Returns:
        loss: torch scalar, 越小表示熵越大
    """
    B, C, Bands, H, W = x_proj.shape

    # 展平空间 + 通道，得到每个波段的向量
    # shape: (B, Bands, C*H*W)
    band_vectors = x_proj.permute(0, 2, 1, 3, 4).reshape(B, Bands, -1)

    # 对每个波段向量做 softmax -> 概率分布
    prob = F.softmax(band_vectors, dim=-1)  # (B, Bands, C*H*W)

    # 计算熵
    entropy = -(prob * torch.log(prob + eps)).sum(dim=-1)  # (B, Bands)

    # 平均每个 batch 和每个波段
    loss = -entropy.mean()  # 最大化熵 => 最小化负熵
    return loss
def proxy_loss_fn(proxies, img_emb, gt_imgs, scale: float | nn.Parameter):
    """
    proxies: shape of (num_classes, dim)
    img_emb: shape of (num_imgs, dim)
    gt_imgs: shape of (num_imgs)
    """
    proxies_emb = scale * F.normalize(proxies, p=2, dim=-1)
    img_emb = scale * F.normalize(img_emb, p=2, dim=-1)

    img_dist = pairwise_distance_v2(proxies=proxies_emb, x=img_emb, squared=True)
    img_dist = img_dist * -1.0

    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    img_loss = cross_entropy(img_dist, gt_imgs)
    return img_loss
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
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
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

class FeatureCompressor_linear(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),          # 128 → 64
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1),                      # 通道压缩 → out_channels
        )

    def forward(self, x):
        return self.compressor(x)  # [B, 256, 64, 64]
class FeatureUpsampler(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64 → 128
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128 → 256
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.upsampler(x)  # [B, 256, 256, 256]

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
# Adapted to fit channel-wise embedding; Adapted from  https://github.com/insitro/ChannelViT  and https://github.com/chaudatascience/diverse_channel_vit.
class ImageChannelEncoderViT(nn.Module):
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
        pass
    #     self.img_size = img_size
    #     self.patch_embed = PatchEmbed(
    #         kernel_size=(patch_size, patch_size),
    #         stride=(patch_size, patch_size),
    #         in_chans=in_chans,
    #         embed_dim=embed_dim,
    #     ) # image to patch embedding
    #     path_embed = embed_dim
    #     self.pos_embed: Optional[nn.Parameter] = None
        
    #     if use_abs_pos:
    #         # Initialize absolute positional embedding with pretrain image size.
    #         self.pos_embed = nn.Parameter(
    #             torch.zeros(1, img_size // patch_size, img_size // patch_size, path_embed)
    #         )
    #     self.blocks = nn.ModuleList()
    #     for i in range(depth):
    #         block = Block( #Transformer block TODO: understand the parameters/Set LoRA
    #             dim=embed_dim,
    #             num_heads=num_heads,
    #             mlp_ratio=mlp_ratio,
    #             qkv_bias=qkv_bias,
    #             norm_layer=norm_layer,
    #             act_layer=act_layer,
    #             use_rel_pos=use_rel_pos,
    #             rel_pos_zero_init=rel_pos_zero_init,
    #             window_size=window_size if i not in global_attn_indexes else 0,
    #             input_size=(img_size // patch_size, img_size // patch_size),
    #         )
    #         self.blocks.append(block)

    #     self.neck = nn.Sequential(
    #         nn.Conv2d(
    #             embed_dim,
    #             out_chans,
    #             kernel_size=1,
    #             bias=False,
    #         ),
    #         LayerNorm2d(out_chans),
    #         nn.Conv2d(
    #             out_chans,
    #             out_chans,
    #             kernel_size=3,
    #             padding=1,
    #             bias=False,
    #         ),
    #         LayerNorm2d(out_chans),
    #     )

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.patch_embed(x)
    #     if self.pos_embed is not None:
    #         x = x + self.pos_embed
    #     interm_embeddings=[]
    #     for blk in self.blocks:
    #         x = blk(x)
    #         if blk.window_size == 0:
    #             interm_embeddings.append(x)
    #     x = self.neck(x.permute(0, 3, 1, 2))
        
    #     return x, interm_embeddings




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

class ChannelVisionTransformer(nn.Module):
    """Channel Vision Transformer"""
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        # drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_sample=False,
        use_multi_feature = False,
        **kwargs,
    ):
        super().__init__()
        # self.cfg = config
        drop_path_rate = 0.0
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans

        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            enable_sample=enable_sample,
            ortho_loss_lambda=0.1,
            proxy_loss_lambda=0.1,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # add Seg and Edge Token
        self.seg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.edge_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 3  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        print("----dpr", dpr)
        
        BlockClass = Block
        
        self.blocks = nn.ModuleList(
            [
                BlockClass(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.use_multi_feature = use_multi_feature  
        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.seg_token, std=0.02)
        trunc_normal_(self.edge_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, nc):
        # number of auxilary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embed.shape[1] - num_extra_tokens

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]

        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, nc, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, _, w, h = x.shape
        x, nc, ortho_proxy_loss = self.patch_embed(
            x
        )  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seg_tokens = self.seg_token.expand(B, -1, -1)
        edge_tokens = self.edge_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, seg_tokens, edge_tokens, x), dim=1)

        # x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h, nc)

        ### drop some tokens randomly at the last dim
        ## x: [B CinHW Cout]
        # cinHW = x.shape[1]
        # HW = cinHW // nc
        # cinHW_new = random.randint(1, nc) * HW
        # drops_true = random.sample(range(cinHW), k=cinHW_new)
        # ## make sure the first token ([CLS]) is not dropped
        # drops = [True]
        # for i in range(1, cinHW):
        #     if i in drops_true:
        #         drops.append(True)
        #     else:
        #         drops.append(False)
        # drops = torch.tensor(drops, device=x.device)
        # x = x[:, drops, :]
        # elif self.cfg.dropout_tokens_hcs == "channel" and self.training:
        #     cinHW = x.shape[1]
        #     HW = cinHW // nc
        #     cin_new = random.randint(1, nc)
        #     ## choose cin_new from nc channels
        #     drops_channels = random.sample(range(nc), k=cin_new)
        #     drops = [True]  ## make sure the first token ([CLS]) is not dropped
        #     for i in range(nc):
        #         if i in drops_channels:
        #             tmp = [True] * HW
        #         else:
        #             tmp = [False] * HW
        #         drops.extend(tmp)
        #     drops = torch.tensor(drops, device=x.device)
        #     x = x[:, drops, :]

        # elif self.cfg.dropout_tokens_hcs == "channel_random50" and self.training:
        #     cinHW = x.shape[1]
        #     HW = cinHW // nc
        #     ## get ceil(50% of the channels)
        #     cin_new = int(math.ceil(0.5 * nc))
        #     ## choose cin_new from nc channels
        #     drops_channels = random.sample(range(nc), k=cin_new)
        #     drops = [True]  ## make sure the first token ([CLS]) is not dropped
        #     for i in range(nc):
        #         if i in drops_channels:
        #             tmp = [True] * HW
        #         else:
        #             tmp = [False] * HW
        #         drops.extend(tmp)
        #     drops = torch.tensor(drops, device=x.device)
        #     x = x[:, drops, :]
        # elif self.cfg.dropout_tokens_hcs == "token_random50" and self.training:  ## x: [B CinHW Cout]
        #     cinHW = x.shape[1]
        #     HW = cinHW // nc
        #     cinHW_new = int(math.ceil(0.5 * nc)) * HW
        #     drops_true = random.sample(range(cinHW), k=cinHW_new)
        #     ## make sure the first token ([CLS]) is not dropped
        #     drops = [True]
        #     for i in range(1, cinHW):
        #         if i in drops_true:
        #             drops.append(True)
        #         else:
        #             drops.append(False)
        #     drops = torch.tensor(drops, device=x.device)
        #     x = x[:, drops, :]
        return self.pos_drop(x), ortho_proxy_loss

    def forward(
        self,
        x,
        extra_tokens={},
    ):
        B, _, w, h = x.shape
        x, ortho_proxy_loss = self.prepare_tokens(x)
        nc = x.shape[1] // ((w // self.patch_size) * (h // self.patch_size))

        x_list = []
        num_layers = len(self.blocks)
        num_samples = 4
        #TODO 
        feature_layers = [int(i * num_layers / num_samples) for i in range(num_samples)]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in feature_layers:
                x_list.append(self.norm(x))

        x_list = [item[:,self.num_extra_tokens:] for item in x_list]
        x = self.norm(x)
        if self.use_multi_feature:
            return {'cls_token': x[:,0].clone(),'seg_token': x[:,1].clone(), 'edge_token': x[:,2].clone(), 'patch_tokens': x_list,'ortho_proxy_loss': ortho_proxy_loss}
        else:
            return {'cls_token': x[:,0].clone(),'seg_token': x[:,1].clone(), 'edge_token': x[:,2].clone(), 'patch_tokens': [x[:,self.num_extra_tokens:].clone()],'ortho_proxy_loss': ortho_proxy_loss}

    def get_last_selfattention(self, x, extra_tokens={}, chunk="", layer_idx=-1):
        x, _ = self.prepare_tokens(
            x, chunk=chunk, training_chunks_str=None, new_channel_init=None, extra_tokens=extra_tokens
        )

        for i, blk in enumerate(self.blocks):
            if i == layer_idx:
                return blk(x, return_attention=True)

            x = blk(x)

    def get_intermediate_layers(self, x, extra_tokens={}, n=1):
        x = self.prepare_tokens(x, extra_tokens)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = False,
        freeze_channel_emb: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
    def forward(self, x):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
                
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = 0
        if self.ortho_loss_lambda > 0:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 :
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')
        x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        
        return x_out, Cin, total_loss

# 多通道的2D卷积替代原来的2D卷积
class PatchEmbedPerChannelV1_1(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 4,
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
# V2中的x_out = self.embedding_proj(x_out) 直接使用线性融合通道维度特征，直接破坏了通道语义
# 改进方式1，MLP做通道信息融合
class PatchEmbedPerChannelV2_1(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.embedding_proj = MLP(self.in_chans*embed_dim,embed_dim, embed_dim,3,sigmoid_output=True)


        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken

    def forward(self, x,train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')

        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        x_out = self.embedding_proj(x_out)

        return x_out, Cin, total_loss
class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_chans, embed_dim, hidden_ratio=0.5):
        super().__init__()
        hidden_dim = int(embed_dim * hidden_ratio)

        self.attn_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # 输出注意力权重
            nn.Sigmoid()
        )
        # self.softmax = nn.Softmax(dim=3)  # 对 Cin 维度归一化

    def forward(self, x, Cin):  # x: [B, Hp, Wp, Cin*E]
        B, Hp, Wp, CE = x.shape
        # Cin = self.get_cin(CE)
        E = CE // Cin

        # Reshape: [B, Hp, Wp, Cin*E] -> [B, Hp, Wp, Cin, E]
        x = rearrange(x, 'B Hp Wp (Cin E) -> B Hp Wp Cin E', Cin=Cin)

        # Compute attention scores: [B, Hp, Wp, Cin, 1]
        attn_weight = self.attn_mlp(x)

        # Normalize: [B, Hp, Wp, Cin, 1]
        # attn_weight = self.softmax(attn_score)

        # Weighted sum over Cin: [B, Hp, Wp, E]
        x_fused = (x * attn_weight).sum(dim=3)
        return x_fused  # -> [B, Hp, Wp, E]
    # def get_cin(self, CE):
    #     # 用于自动推断 Cin 和 E
    #     for i in range(1, CE + 1):
    #         if CE % i == 0:
    #             E = CE // i
    #             if E in [128, 256, 384, 512, 768, 1024]:
    #                 return i
    #     raise ValueError(f"Cannot infer Cin from CE={CE}")
# V2中的x_out = self.embedding_proj(x_out) 直接使用线性融合通道维度特征，直接破坏了通道语义
# 改进方式2，使用attention reweighting
class PatchEmbedPerChannelV2_2(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.embedding_proj = ChannelAttentionFusion(self.in_chans*embed_dim, embed_dim)
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken
        
    def forward(self, x,train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')

        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        x_out = self.embedding_proj(x_out,Cin)
        return x_out, Cin, total_loss

class PerPixelChannelAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True, num_channels=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid
        self.num_channels = num_channels

        # QKV projections
        self.q_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)

        # MLP after attention
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # Learnable channel fusion: [Cin, 1]
        self.channel_fuser = nn.Linear(num_channels, 1)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, Hp, Wp, Cin, E]
        returns: [B, Hp, Wp, E]
        """
        B, Hp, Wp, Cin, E = x.shape
        assert Cin == self.num_channels, f"Expected Cin == {self.num_channels}, got {Cin}"

        # Flatten spatial: [B*Hp*Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x  # residual

        # Q, K, V projections
        Q = self.q_proj(x)  # [B*H*W, Cin, d]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Channel attention weights: [B*H*W, Cin, Cin]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores) if self.use_sigmoid else F.softmax(attn_scores, dim=-1)

        # Attention output: [B*H*W, Cin, d]
        attn_out = torch.bmm(attn_weights, V)

        # Output MLP: [B*H*W, Cin, E]
        attn_out = self.out_proj(attn_out)

        # Add residual and normalize
        attn_out = self.norm(attn_out + shortcut)

        # Permute to [B*H*W, E, Cin] for fusion
        attn_out = attn_out.permute(0, 2, 1)  # [B*H*W, E, Cin]

        # Apply channel fusion (linear over Cin)
        fused = self.channel_fuser(attn_out)  # [B*H*W, E, 1]
        fused = fused.squeeze(-1)             # [B*H*W, E]

        # Restore spatial dimensions
        out = fused.view(B, Hp, Wp, E)        # [B, Hp, Wp, E]

        return out
# V2中的x_out = self.embedding_proj(x_out) 直接使用线性融合通道维度特征，直接破坏了通道语义
# 改进方式3，使用Channelt Attention 参考文章:Understanding The Robustness in Vision Transformers
class PatchEmbedPerChannelV2_3(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.embedding_proj = PerPixelChannelAttention(embed_dim)
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken
        
    def forward(self, x,train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')

        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp Cin E')
        x_out = self.embedding_proj(x_out)
        return x_out, Cin, total_loss

class PatchEmbedPerChannelV3(nn.Module):
    """
    1.光谱token由两部分组成，一部分是稳定的正交token，另一部分来自样本的统计数据。
    2.稳定正交token还是要有orth_loss
    3.样本统计token则要一个额外的MLP投影到(C,D)。（如果训练不稳定则再参考Prototype添加一个聚类算法输出一个更稳定的聚类中心)
    4. 关于多样性，目前是整体多样性增加，也就是phy_token加上正交token之后的token要保持多样性
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = False,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.embedding_proj = nn.Linear(self.in_chans*embed_dim, embed_dim)
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken

        # Phycially-informed token ? 是否过于不稳定
        self.spectral_phy_token_proj = MLP(1, embed_dim , embed_dim, 3)
    def forward(self, x,train_model,mask=None):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        x_flat = x.view(B, Cin, -1)

        if mask is not None:
            mask = mask.view(B, 1, -1)  # [B, 1, H*W]
            x_sum = (x_flat * mask).sum(dim=-1)  # [B, C]
            count = mask.sum(dim=-1) + 1e-6  # [B, 1]
            x_mean = x_sum / count  # [B, C]
        else:
            x_mean = x_flat.mean(dim=-1)  # [B, C]
        x_mean = x_mean.squeeze(0).unsqueeze(1)  # [B, 1, C]
        spectral_token = self.spectral_phy_token_proj(x_mean) 
        self.channel_embed.weight.data = self.channel_embed.weight + spectral_token

        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]

                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')

        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        x_out = self.embedding_proj(x_out)
        return x_out, Cin, total_loss

class ChannelFusionDepthwiseConv(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            # 输入: B, C_in, H, W
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),  # depthwise
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 输入: B, H, W, Cin*E → rearrange to B, Cin*E, H, W
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.fusion(x)
        x = rearrange(x, 'B C H W -> B H W C')  # for ViT
        return x

# 使用depthwise卷积处理最后的投影
class PatchEmbedPerChannelV2_4(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.embedding_proj = ChannelFusionDepthwiseConv(self.in_chans * embed_dim, self.embed_dim)

        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken

    def forward(self, x,train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')

        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        x_out = self.embedding_proj(x_out)

        return x_out, Cin, total_loss

class UpsampleProjector(nn.Module):
    def __init__(self, vit_dim, transformer_dim, out_channels):
        super(UpsampleProjector, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),

            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            nn.GELU(),

            nn.ConvTranspose2d(transformer_dim // 2, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.upsample(x)


class PerPixelChannelAttentionWithUpsample(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True, num_channels=4, 
                 out_channels=3, upsample_scale=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid
        self.num_channels = num_channels

        # Channel attention
        self.q_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.channel_fuser = nn.Linear(num_channels, 1)

        # Upsample: from [B, E, Hp, Wp] → [B, 4, H, W]
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, hidden_dim, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 8, kernel_size=2, stride=2)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 16),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 32, kernel_size=2, stride=2)
        )

        self.conv_compress = nn.Conv2d(hidden_dim // 32, num_channels, kernel_size=1)
        self.channel_match_proj = nn.Conv2d(num_channels, 3, kernel_size=1)

        # # Shortcut for skip connection
        # self.shortcut_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        # )

        # self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, image_shortcut):
        """
        x: [B E Cin Hp Wp]
        return: [B, out_channels, H, W]
        """
        B ,E ,Cin ,Hp ,Wp = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, Hp, Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores) if self.use_sigmoid else F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        attn_out = self.out_proj(attn_out)
        attn_out = self.norm(attn_out + shortcut)

        # [B*Hp*Wp, Cin, E] → [B*Hp*Wp, E, Cin] → [B*Hp*Wp, E]
        attn_out = attn_out.permute(0, 2, 1)
        fused = self.channel_fuser(attn_out).squeeze(-1)
        fused = fused.view(B, Hp, Wp, E).permute(0, 3, 1, 2)  # [B, E, Hp, Wp]

        # Upsample and shortcut
        up1 = self.up1(fused)                    # [B, 4, H, W]
        # up1 = up1.permute(0, 2, 3, 1)  + image_shortcut            # [B, H, W, 4] 
        up2 = self.up2(up1)

        reconstructed_image = self.conv_compress(up2)  # [B, 4, H, W]

        reconstructed_image = reconstructed_image + image_shortcut  # [B, 4, H, W]
        out = self.channel_match_proj(reconstructed_image)  # [B, 3, H, W]
        # shortcut_proj = self.shortcut_proj(fused)  # [B, 3, H, W]
        # up2 = self.up2(up1)                      # [B, 3, H, W]
        # out = self.final_conv(up2 + shortcut_proj)

        return out  # [B, 3, H, W]

class PerPixelChannelAttentionWithUpsampleV2(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True, num_channels=4, 
                 out_channels=3, upsample_scale=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid
        self.num_channels = num_channels

        # Channel attention
        self.q_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        # self.channel_fuser = nn.Linear(num_channels, 1)

        # Upsample: from [B, E*Cin, Hp, Wp] → [B, 4, H, W]
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * num_channels, hidden_dim, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 8, kernel_size=2, stride=2)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 16),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 32, kernel_size=2, stride=2)
        )

        self.conv_compress = nn.Conv2d(hidden_dim // 32, num_channels, kernel_size=1)
        self.channel_match_proj = nn.Conv2d(num_channels, 3, kernel_size=1)

        # # Shortcut for skip connection
        # self.shortcut_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        # )

        # self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, image_shortcut):
        """
        x: [B E Cin Hp Wp]
        return: [B, out_channels, H, W]
        """
        # visualize_patch_attention_effect(x,'/mnt/disk3/har/Param/Cropland/S4A/ChannelAttention/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss','2019_31TCJ_patch_15_09',use_sigmoid=True,q_proj=self.q_proj,k_proj=self.k_proj,v_proj=self.v_proj,out_proj=self.out_proj,norm_layer=self.norm,hidden_dim=self.hidden_dim)

        B ,E ,Cin ,Hp ,Wp = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, Hp, Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores) if self.use_sigmoid else F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        attn_out = self.out_proj(attn_out)
        attn_out = self.norm(attn_out + shortcut)

        

        # [B*Hp*Wp, Cin, E] → [B*Hp*Wp, E, Cin] → [B*Hp*Wp, E]
        # attn_out = attn_out.permute(0, 2, 1)
        fused = attn_out.view(B, Hp, Wp, Cin * E).permute(0, 3, 1, 2)  # [B, E*Cin, Hp, Wp]

        # Upsample and shortcut
        up1 = self.up1(fused)                    # [B, 4, H, W]
        # up1 = up1.permute(0, 2, 3, 1)  + image_shortcut            # [B, H, W, 4] 
        up2 = self.up2(up1)

        reconstructed_image = self.conv_compress(up2)  # [B, 4, H, W]

        reconstructed_image = reconstructed_image + image_shortcut  # [B, 4, H, W]
        out = self.channel_match_proj(reconstructed_image)  # [B, 3, H, W]
        # shortcut_proj = self.shortcut_proj(fused)  # [B, 3, H, W]
        # up2 = self.up2(up1)                      # [B, 3, H, W]
        # out = self.final_conv(up2 + shortcut_proj)

        return out  # [B, 3, H, W]

class PerPixelChannelAttentionWithUpsampleV4(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True, num_channels=4, 
                 out_channels=3, upsample_scale=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid
        self.num_channels = num_channels

        # Channel attention
        self.q_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        # self.channel_fuser = nn.Linear(num_channels, 1)

        # Upsample: from [B, E*Cin, Hp, Wp] → [B, 4, H, W]
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * num_channels, hidden_dim, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 8, kernel_size=2, stride=2)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 16),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 32, kernel_size=2, stride=2)
        )

        self.conv_compress = nn.Conv2d(hidden_dim // 32, num_channels, kernel_size=1)
        self.channel_match_proj = nn.Conv2d(num_channels, 3, kernel_size=1)

        # # Shortcut for skip connection
        # self.shortcut_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        # )

        # self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, image_shortcut):
        """
        x: [B E Cin Hp Wp]
        return: [B, out_channels, H, W]
        """
        # visualize_patch_attention_effect(x,'/mnt/disk3/har/Param/Cropland/S4A/ChannelAttention/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss','2019_31TCJ_patch_15_09',use_sigmoid=True,q_proj=self.q_proj,k_proj=self.k_proj,v_proj=self.v_proj,out_proj=self.out_proj,norm_layer=self.norm,hidden_dim=self.hidden_dim)

        B ,E ,Cin ,Hp ,Wp = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, Hp, Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores) if self.use_sigmoid else F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        attn_out = self.out_proj(attn_out)
        attn_out = self.norm(attn_out + shortcut)

        

        # [B*Hp*Wp, Cin, E] → [B*Hp*Wp, E, Cin] → [B*Hp*Wp, E]
        # attn_out = attn_out.permute(0, 2, 1)
        fused = attn_out.view(B, Hp, Wp, Cin * E).permute(0, 3, 1, 2)  # [B, E*Cin, Hp, Wp]

        # Upsample and shortcut
        up1 = self.up1(fused)                    # [B, 4, H, W]
        # up1 = up1.permute(0, 2, 3, 1)  + image_shortcut            # [B, H, W, 4] 
        up2 = self.up2(up1)

        reconstructed_image = self.conv_compress(up2)  # [B, 4, H, W]

        reconstructed_image = reconstructed_image + image_shortcut  # [B, 4, H, W]
        out = self.channel_match_proj(reconstructed_image)  # [B, 3, H, W]
        # shortcut_proj = self.shortcut_proj(fused)  # [B, 3, H, W]
        # up2 = self.up2(up1)                      # [B, 3, H, W]
        # out = self.final_conv(up2 + shortcut_proj)

        return out  # [B, 3, H, W]

class ChannelSelfAttention(nn.Module):
    """
    Channel-wise Self-Attention
    Attention is performed across Cin dimension, with E as embedding dimension.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dim: int = None,
        use_sigmoid: bool = True,
        add_channel_embed: bool = False,
    ):
        """
        Args:
            embed_dim (int): E, embedding dimension per channel
            attn_dim (int): D, attention projection dimension (default = embed_dim)
            use_sigmoid (bool): use sigmoid instead of softmax (not recommended unless you know why)
            add_channel_embed (bool): whether to add learnable channel identity embedding
            num_channels (int): Cin, required if add_channel_embed=True
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.attn_dim = attn_dim or embed_dim
        self.use_sigmoid = use_sigmoid
        self.add_channel_embed = add_channel_embed

        # QKV projections (operate on E)
        self.q_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.attn_dim, embed_dim, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, Cin, E]

        Returns:
            Tensor of shape [B, Cin, E]
        """
        assert x.dim() == 3, f"Expected [B, Cin, E], got {x.shape}"

        B, Cin, E = x.shape
        shortcut = x


        # Q K V
        Q = self.q_proj(x)  # [B, Cin, D]
        K = self.k_proj(x)  # [B, Cin, D]
        V = self.v_proj(x)  # [B, Cin, D]

        # Attention across Cin
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))
        attn_scores = attn_scores / (self.attn_dim ** 0.5)  # [B, Cin, Cin]

        if self.use_sigmoid:
            attn_weights = torch.sigmoid(attn_scores)
        else:
            attn_weights = F.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, V)  # [B, Cin, D]

        # Back to embedding space
        attn_out = self.out_proj(attn_out)  # [B, Cin, E]

        # Residual + Norm
        out = self.norm(attn_out + shortcut)

        return out
class PerPixelChannelAttentionWithUpsampleV3(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True, num_channels=4, 
                 out_channels=3, upsample_scale=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid
        self.num_channels = num_channels

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        # self.channel_fuser = nn.Linear(num_channels, 1)

        # Upsample: from [B, E*Cin, Hp, Wp] → [B, 4, H, W]
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim * num_channels, hidden_dim, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 8, kernel_size=2, stride=2)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 16),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 32, kernel_size=2, stride=2)
        )

        self.conv_compress = nn.Conv2d(hidden_dim // 32, num_channels, kernel_size=1)
        self.channel_match_proj = nn.Conv2d(num_channels, 3, kernel_size=1)

        # # Shortcut for skip connection
        # self.shortcut_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        # )

        # self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, image_shortcut):
        """
        x: [B E Cin Hp Wp]
        return: [B, out_channels, H, W]
        """
        B ,E ,Cin ,Hp ,Wp = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, Hp, Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x


        attn_out = shortcut
        # [B*Hp*Wp, Cin, E] → [B*Hp*Wp, E, Cin] → [B*Hp*Wp, E]
        # attn_out = attn_out.permute(0, 2, 1)
        fused = attn_out.contiguous().view(B, Hp, Wp, Cin * E).permute(0, 3, 1, 2)  # [B, E*Cin, Hp, Wp]

        # Upsample and shortcut
        up1 = self.up1(fused)                    # [B, 4, H, W]
        # up1 = up1.permute(0, 2, 3, 1)  + image_shortcut            # [B, H, W, 4] 
        up2 = self.up2(up1)

        reconstructed_image = self.conv_compress(up2)  # [B, 4, H, W]

        reconstructed_image = reconstructed_image + image_shortcut  # [B, 4, H, W]
        out = self.channel_match_proj(reconstructed_image)  # [B, 3, H, W]
        # shortcut_proj = self.shortcut_proj(fused)  # [B, 3, H, W]
        # up2 = self.up2(up1)                      # [B, 3, H, W]
        # out = self.final_conv(up2 + shortcut_proj)

        return out  # [B, 3, H, W]


class PatchEmbedPerChannelV2_5(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
        Attention_Upsample_V: int = 1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # path embedding from SAM
        self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=3,
                embed_dim=embed_dim,
        )
        
        SAM_path = "/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_b_01ec64.pth" # 加载SAM的权重路径
        
        self.patch_embed_init(SAM_path)
        if Attention_Upsample_V == 1:
            self.embedding_proj = PerPixelChannelAttentionWithUpsample(embed_dim=embed_dim,hidden_dim=embed_dim,num_channels=in_chans)
        elif Attention_Upsample_V == 2:
            self.embedding_proj = PerPixelChannelAttentionWithUpsampleV2(embed_dim=embed_dim,hidden_dim=embed_dim,num_channels=in_chans)
        elif Attention_Upsample_V == 3 or 4:
            self.embedding_proj = PerPixelChannelAttentionWithUpsampleV3(embed_dim=embed_dim,hidden_dim=embed_dim,num_channels=in_chans)
        if Attention_Upsample_V == 4: # 在 channel emb上做attention
            self.ChAttn = ChannelSelfAttention(
                embed_dim=embed_dim,
                attn_dim=embed_dim // 2,
                add_channel_embed=True,
            )
        self.Attention_Upsample_V = Attention_Upsample_V
        # elif Attention_Upsample_V == 3:
        #     self.embedding_proj = PerPixelChannelAttentionWithUpsampleV4(embed_dim=embed_dim,hidden_dim=embed_dim,num_channels=in_chans)
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken

    def patch_embed_init(self, SAM_path):
        # 加载SAM的checkpoint
        state_dict = torch.load(SAM_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # 找出PatchEmbed相关的key
        patch_keys = {k: v for k, v in state_dict.items() if 'patch_embed' in k}

        # 判断是否为空
        if not patch_keys:
            raise ValueError("未在SAM权重中找到'patch_embed'相关的参数。")

        # 重命名key
        renamed_patch_keys = {}
        for k, v in patch_keys.items():
            new_key = k.split('patch_embed.')[-1]  # 去掉前缀
            renamed_patch_keys[new_key] = v

        # 加载权重
        missing, unexpected = self.patch_embed.load_state_dict(renamed_patch_keys, strict=False)

        # print(f"PatchEmbed 权重已加载，缺失项: {missing}, 不期望项: {unexpected}")

    def forward(self, x,train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]

            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        if self.Attention_Upsample_V == 4:
            channel_emb = self.ChAttn(channel_emb.unsqueeze(0) )
            channel_emb = channel_emb.squeeze(0)
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')


        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)') ChannelToken_PEV2_5_2_DecoderV3_2_without_CDL   ChannelToken_PEV2_5_2_DecoderV3_2
        # visualize_band_correlation_v4(x_proj,'/mnt/disk3/har/Param/Cropland/S4A/Token_Similiar/ChannelToken_PEV2_5_2_DecoderV3_2_without_CDL','2019_31TCJ_patch_15_09_img', max_value=3.0)
        # visualize_band_correlation_v4(channel_emb,'/home/huar/Param/MSSAM_review_Exp/S4A/ForReview_Exp/ABL_ANA/Feature_vis','CA_CT_CAForCT_tokenSim', max_value=0.3)
        
        
        x_out = self.embedding_proj(x_proj,x)
        x_out = self.patch_embed(x_out)
        return x_out, Cin, total_loss
    
class SpectralPE(nn.Module):
    def __init__(self, spectral_positions, embed_dim):
        super().__init__()
        # 输入必须为 tensor
        spectral_positions = torch.tensor(spectral_positions, dtype=torch.float32)

        # 归一化波长
        wavelengths = (spectral_positions - spectral_positions.min()) / (spectral_positions.max() - spectral_positions.min())

        # 构造固定编码（sin/cos）
        pe = self.get_spectral_pe(wavelengths, embed_dim)  # [C, D]

        # 注册为可学习参数
        self.spectral_pe = nn.Parameter(pe, requires_grad=True)

    def get_spectral_pe(self, wavelengths, dim):
        pe = []
        for i in range(dim // 2):
            div_term = 10000 ** (2 * i / dim)
            pe.append(torch.sin(wavelengths / div_term))
            pe.append(torch.cos(wavelengths / div_term))
        return torch.stack(pe, dim=-1).view(len(wavelengths), dim)  # [C, D]

    def forward(self):
        return self.spectral_pe  # [C, D]

# 加入波段在光谱位置上的相对位置编码
class PatchEmbedPerChannelV2_6(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
        spectral_positions: list = [450, 550, 665, 842],
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        spectral_encoder = SpectralPE(spectral_positions, embed_dim)
        self.spectral_pe = spectral_encoder()

        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
            # self.spectral_pe.requires_grad = False
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # path embedding from SAM
        self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=3,
                embed_dim=embed_dim,
        )
        SAM_path = "/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_b_01ec64.pth"
        self.patch_embed_init(SAM_path)
        self.embedding_proj = PerPixelChannelAttentionWithUpsample(embed_dim=embed_dim,hidden_dim=embed_dim,num_channels=in_chans)
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken

    def patch_embed_init(self, SAM_path):
        # 加载SAM的checkpoint
        state_dict = torch.load(SAM_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # 找出PatchEmbed相关的key
        patch_keys = {k: v for k, v in state_dict.items() if 'patch_embed' in k}

        # 判断是否为空
        if not patch_keys:
            raise ValueError("未在SAM权重中找到'patch_embed'相关的参数。")

        # 重命名key
        renamed_patch_keys = {}
        for k, v in patch_keys.items():
            new_key = k.split('patch_embed.')[-1]  # 去掉前缀
            renamed_patch_keys[new_key] = v

        # 加载权重
        missing, unexpected = self.patch_embed.load_state_dict(renamed_patch_keys, strict=False)

        # print(f"PatchEmbed 权重已加载，缺失项: {missing}, 不期望项: {unexpected}")

    def forward(self, x,train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')
        spectral_pe_emb = rearrange(self.spectral_pe, 'C D -> 1 D C 1 1')
        if self.use_channelToken:
            x_proj = x_proj + channel_emb + spectral_pe_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        x_out = self.embedding_proj(x_proj,x)
        x_out = self.patch_embed(x_out)
        return x_out, Cin, total_loss

# 带采样的不定通道输入,使用加权求和的方式来合并通道
class PerPixelChannelAttention_Sampling(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid


        # Channel attention
        self.q_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.channel_fuser = nn.Parameter(torch.empty(13, 1))
        # # Shortcut for skip connection
        # self.shortcut_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        # )

        # self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, select_channel):
        """
        x: [B E Cin Hp Wp]
        return: [B, out_channels, H, W]
        """
        B ,E ,Cin ,Hp ,Wp = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, Hp, Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores) if self.use_sigmoid else F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        attn_out = self.out_proj(attn_out)
        
        # [B*Hp*Wp, Cin, E] → [B*Hp*Wp, E, Cin] 
        
        attn_out = self.norm(attn_out + shortcut).permute(0, 2, 1)

        linear_w = self.channel_fuser[select_channel]
        attn_out = torch.matmul(attn_out, linear_w)
        attn_out = attn_out.view(B, Hp, Wp, E)

        # scores = attn_out.mean(dim=3)
        # weights = F.softmax(scores, dim=-1)
        # attn_out = (attn_out * weights.unsqueeze(3)).sum(dim=-1)

        return attn_out  # [B, Hp, Wp, E]

# 带采样的不定通道输入,使用加权求和的方式来合并通道
# 添加上采样，压缩为3通道，以适配SAM的PE
class PerPixelChannelAttentionWithUpSampling_Sampling(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid

        # Channel attention
        self.q_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, hidden_dim, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 8),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 16),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 32, kernel_size=2, stride=2)
        )
        self.channel_match_proj = nn.Conv2d(hidden_dim // 32, 3, kernel_size=1)

        #TODO 
        self.channel_fuser = nn.Parameter(torch.empty(13, 1))
        nn.init.xavier_uniform_(self.channel_fuser)
        # # Shortcut for skip connection
        # self.shortcut_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        # )

        # self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, select_channel):
        """
        x: [B E Cin Hp Wp]
        return: [B, out_channels, H, W]
        """
        B ,E ,Cin ,Hp ,Wp = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, Hp, Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores) if self.use_sigmoid else F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        attn_out = self.out_proj(attn_out)
        
        # [B*Hp*Wp, Cin, E] → [B*Hp*Wp, E, Cin] 
        attn_out = self.norm(attn_out + shortcut).permute(0, 2, 1)
        # attn_out = attn_out.view(B, Hp, Wp, E, Cin)
        attn_out = attn_out.view(B*Hp*Wp, E, Cin)
        # attn_out = self.channel_fuser(attn_out)
        linear_W = self.channel_fuser[select_channel]
        attn_out = torch.matmul(attn_out, linear_W).view(B, Hp, Wp, E)

        up = self.up(attn_out.permute(0, 3, 1, 2))
        channel_match = self.channel_match_proj(up)
        # attn_out = channel_match.permute(0, 2, 3, 1)
        return channel_match  # [B, 3, H, W]

class Dynamic1x1Conv(nn.Module):
    def __init__(self, max_in_channels, out_channels, use_bias=True):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.out_channels = out_channels

        # 初始化最大输入通道数对应的W_all和bias
        self.W_all = nn.Parameter(torch.randn(out_channels, max_in_channels))  # shape: [C_out, C_in_max]
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_all)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    def forward(self, x, cur_channels):
        """
        x: Tensor of shape [B, cur_channels, H, W]
        cur_channels: int, 当前输入图像的实际通道数
        """
        B, C, H, W = x.shape
        assert C == cur_channels, f"Input channel mismatch: got {C}, expected {cur_channels}"

        # 截取当前需要的 W 和 bias
        W = self.W_all[:, cur_channels]  # shape: [C_out, cur_channels]

        # [B, cur_channels, H, W] -> [B, cur_channels, H*W]
        x_flat = x.view(B, cur_channels, -1)

        # 线性乘法：[C_out, cur_channels] x [B, cur_channels, H*W] = [C_out, B, H*W]
        out = torch.matmul(W, x_flat)  # shape: [C_out, B, H*W]
        out = out.permute(1, 0, 2).view(B, self.out_channels, H, W)

        if self.use_bias:
            out = out + self.bias.view(1, -1, 1, 1)

        return out
class Dynamic1x1ConvIO(nn.Module):
    def __init__(self, max_in_channels, max_out_channels, use_bias=True):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.use_bias = use_bias

        # 初始化最大权重和偏置参数
        self.W_all = nn.Parameter(torch.empty(max_out_channels, max_in_channels))  # [C_out_max, C_in_max]
        if use_bias:
            self.bias_all = nn.Parameter(torch.zeros(max_out_channels))
        else:
            self.register_parameter("bias_all", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_all)
        if self.use_bias:
            nn.init.zeros_(self.bias_all)

    def forward(self, x, cur_in_channels=None, cur_out_channels=None):
        """
        x: [B, C_in, H, W]
        cur_in_channels: List[int] or None — 选择输入通道索引（长度应等于 x.shape[1]）
        cur_out_channels: List[int] or None — 选择输出通道索引
        """
        B, C_in, H, W = x.shape

        # 输入通道索引
        if cur_in_channels is None:
            cur_in_indices = torch.arange(self.max_in_channels, device=x.device)
        else:
            cur_in_indices = torch.tensor(cur_in_channels, dtype=torch.long, device=x.device)

        # 输出通道索引
        if cur_out_channels is None:
            cur_out_indices = torch.arange(self.max_out_channels, device=x.device)
        else:
            cur_out_indices = torch.tensor(cur_out_channels, dtype=torch.long, device=x.device)

        # 选择相应参数
        linear_W = self.W_all[cur_out_indices][:, cur_in_indices]  # [C_out_cur, C_in_cur]
        if self.use_bias:
            bias = self.bias_all[cur_out_indices]           # [C_out_cur]

        # 选择输入通道
                  # [B, C_in_cur, H, W]
        x_flat = x.view(B, len(cur_in_indices), -1)  # [B, C_in_cur, H*W]

        # 线性变换：[C_out_cur, C_in_cur] @ [B, C_in_cur, H*W] = [C_out_cur, B, H*W]
        out = torch.matmul(linear_W, x_flat)                      # [C_out_cur, B, H*W]
        out = out.permute(1, 0, 2).view(B, len(cur_out_indices), H, W)

        if self.use_bias:
            out = out + bias.view(1, -1, 1, 1)

        return out
    
class PerPixelChannelAttentionWithUpSampling_SamplingV2(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, use_sigmoid=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.use_sigmoid = use_sigmoid

        # Channel attention
        self.q_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, hidden_dim, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 8),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(hidden_dim // 16),
            nn.GELU(), 
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 32, kernel_size=2, stride=2)
        )
        
        # 动态输入的线性层
        self.short_cut_proj = Dynamic1x1ConvIO(hidden_dim // 32, 13)
        
        self.channel_match_proj = Dynamic1x1ConvIO(13, 3)
        self.channel_fuser = nn.Parameter(torch.empty(13, 1))
        nn.init.xavier_uniform_(self.channel_fuser)
        # # Shortcut for skip connection
        # self.shortcut_proj = nn.Sequential(
        #     nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        #     nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        # )

        # self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, select_channel, image_shortcut):
        """
        x: [B E Cin Hp Wp]
        return: [B, out_channels, H, W]
        """
        B ,E ,Cin ,Hp ,Wp = x.shape
        x = x.permute(0, 3, 4, 2, 1)  # [B, Hp, Wp, Cin, E]
        x = x.view(B * Hp * Wp, Cin, E)
        shortcut = x

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.sigmoid(attn_scores) if self.use_sigmoid else F.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        attn_out = self.out_proj(attn_out)
        
        # [B*Hp*Wp, Cin, E] → [B*Hp*Wp, E, Cin] 
        attn_out = self.norm(attn_out + shortcut).permute(0, 2, 1)
        # attn_out = attn_out.view(B, Hp, Wp, E, Cin)
        attn_out = attn_out.view(B*Hp*Wp, E, Cin)
        # attn_out = self.channel_fuser(attn_out)
        linear_W = self.channel_fuser[select_channel]
        attn_out = torch.matmul(attn_out, linear_W).view(B, Hp, Wp, E)


        up = self.up(attn_out.permute(0, 3, 1, 2)) # [B, hidden_dim // 32, Hp, Wp]
        reconstructed_image = self.short_cut_proj(up, cur_in_channels=None, cur_out_channels=select_channel)
        reconstructed_image = reconstructed_image + image_shortcut
        channel_match = self.channel_match_proj(reconstructed_image, cur_in_channels=select_channel, cur_out_channels=None)
        # attn_out = channel_match.permute(0, 2, 3, 1)
        return channel_match  # [B, 3, H, W]


#尝试波段采样
#版本1:后续不使用SAM的PE，也就不还原为3，H，W形式
#版本2:使用SAM的PE，还原为3，H，W形式
class PatchEmbedPerChannelV2_7(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
        Attention_Upsample_V: int = 1, # 1是不还原 2是还原
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # path embedding from SAM
        self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=3,
                embed_dim=embed_dim,
        )

        # if Attention_Upsample_V == 1:
        #     self.embedding_proj = PerPixelChannelAttentionWithUpsample(embed_dim=embed_dim,hidden_dim=embed_dim,num_channels=in_chans)
        # elif Attention_Upsample_V == 2:
        #     self.embedding_proj = PerPixelChannelAttentionWithUpsampleV2(embed_dim=embed_dim,hidden_dim=embed_dim,num_channels=in_chans)
        if Attention_Upsample_V == 1:
            self.embedding_proj = PerPixelChannelAttention_Sampling(embed_dim=embed_dim,hidden_dim=embed_dim)
        if Attention_Upsample_V == 2:
            self.embedding_proj = PerPixelChannelAttentionWithUpSampling_Sampling(embed_dim=embed_dim,hidden_dim=embed_dim)
            SAM_path = "/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_b_01ec64.pth"
            self.patch_embed_init(SAM_path)
        if Attention_Upsample_V == 3:
            self.embedding_proj = PerPixelChannelAttentionWithUpSampling_SamplingV2(embed_dim=embed_dim,hidden_dim=embed_dim)
            SAM_path = "/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_b_01ec64.pth"
            self.patch_embed_init(SAM_path)
        
        self.Attention_Upsample_V = Attention_Upsample_V   
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken

    def patch_embed_init(self, SAM_path):
        # 加载SAM的checkpoint
        state_dict = torch.load(SAM_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # 找出PatchEmbed相关的key
        patch_keys = {k: v for k, v in state_dict.items() if 'patch_embed' in k}

        # 判断是否为空
        if not patch_keys:
            raise ValueError("未在SAM权重中找到'patch_embed'相关的参数。")

        # 重命名key
        renamed_patch_keys = {}
        for k, v in patch_keys.items():
            new_key = k.split('patch_embed.')[-1]  # 去掉前缀
            renamed_patch_keys[new_key] = v

        # 加载权重
        missing, unexpected = self.patch_embed.load_state_dict(renamed_patch_keys, strict=False)

        # print(f"PatchEmbed 权重已加载，缺失项: {missing}, 不期望项: {unexpected}")

    def forward(self, x, train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample and train_model:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
            cur_channels = list(range(Cin))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(len(cur_channels), device=x.device)
            proxies = self.channel_emb_proxies[cur_channels]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')

        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        if self.Attention_Upsample_V == 1:
            x_out = self.embedding_proj(x_proj,cur_channels)
        elif self.Attention_Upsample_V == 2:
            x_out = self.embedding_proj(x_proj,cur_channels)
            x_out = self.patch_embed(x_out)
        elif self.Attention_Upsample_V == 3:
            x_out = self.embedding_proj(x_proj,cur_channels, x)
            x_out = self.patch_embed(x_out)
        return x_out, Cin, total_loss

class PatchEmbedPerChannelV2(nn.Module):
    """Image to Patch Embedding with Channel Token Encoding"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        # Loss parameters
        ortho_loss_lambda: float = 0.0,
        proxy_loss_lambda: float = 0.0,
        temperature: float = 1.0,
        hcs_sampling_temp: float = 0.1,
        # Additional parameters
        gamma_s: float = 1.0,
        gamma_d: float = 0.5,
        orthogonal_init: bool = True,
        freeze_channel_emb: bool = False,
        use_orth_loss: bool = True,
        use_channelToken: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_sample = enable_sample
        
        # Loss configurations
        self.ortho_loss_lambda = ortho_loss_lambda
        self.proxy_loss_lambda = proxy_loss_lambda
        self.temperature = temperature
        self.hcs_sampling_temp = hcs_sampling_temp
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        
        # Channel embedding
        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        
        if freeze_channel_emb:
            self.channel_embed.weight.requires_grad = False
        
        # Proxy embeddings for proxy loss
        if proxy_loss_lambda > 0:
            self.channel_emb_proxies = nn.Parameter(torch.randn(in_chans, embed_dim) / 8)
        if orthogonal_init:
            nn.init.orthogonal_(self.channel_embed.weight)
            nn.init.orthogonal_(self.channel_emb_proxies)
        # Projection layer
        self.proj = nn.Conv3d(
            1,  # Single channel dimension
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.embedding_proj = nn.Linear(self.in_chans*embed_dim, embed_dim)
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.num_patches = num_patches
        self.use_orth_loss = use_orth_loss
        self.use_channelToken = use_channelToken
        
    def forward(self, x,train_model):
        B, Cin, H, W = x.shape
        cur_channels = list(range(Cin))
        
        # Channel sampling during training
        if self.enable_sample:
            Cin_new = random.randint(1, Cin)
            first_idx = random.randint(0, Cin - 1)
            
            # Compute similarity between channels
            with torch.no_grad():
                weights = self.channel_embed.weight[:Cin]
                weights_norm = F.normalize(weights, p=2, dim=-1)
                similarities = torch.einsum('c d, e d -> c e', weights_norm, weights_norm)
                cos_scores = similarities[first_idx]
            
            # Sample channels based on low similarity
            scores = (1 - cos_scores) / self.hcs_sampling_temp
            probs = F.softmax(scores, dim=-1)
            indices = torch.multinomial(probs, Cin_new, replacement=False)
            indices = indices.tolist()
            if first_idx not in indices:
                indices[-1] = first_idx
                
            x = x[:, indices]
            channel_emb = self.channel_embed(torch.tensor(indices, device=x.device))
            cur_channels = indices
        else:
            channel_emb = self.channel_embed(torch.arange(Cin, device=x.device))
        
        # Apply convolution projection
        x_proj = self.proj(x.unsqueeze(1))  # [B, embed_dim, Cin, H', W']
        _, _, _, Hp, Wp = x_proj.shape
        num_patches = Cin * Hp * Wp
        
        # Compute ortho-projection loss
        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            x_flat = rearrange(x_proj, 'B d c h w -> B (c h w) d')
            token_labels = torch.arange(Cin, device=x.device).repeat_interleave(Hp * Wp)
            if Cin>4:
                ortho_loss = ortho_proj_loss_fn_v3(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
            else:
                ortho_loss = ortho_proj_loss_fn_v2(
                    x_flat,
                    token_labels,
                    gamma_s=self.gamma_s,
                    gamma_d=self.gamma_d,
                    reverse_pos_pairs=False,
                    use_square=False,
                ) * self.ortho_loss_lambda
        
        # Compute proxy loss
        proxy_loss = 0
        if self.proxy_loss_lambda > 0 and self.use_orth_loss and self.use_channelToken and train_model:
            channel_gt = torch.eye(Cin, device=x.device)
            proxies = self.channel_emb_proxies[list(range(Cin))]
            scale = np.sqrt(1.0 / self.temperature)
            proxy_loss = proxy_loss_fn(proxies, channel_emb, channel_gt, scale) * self.proxy_loss_lambda
        
        # Combine losses
        total_loss = ortho_loss + proxy_loss
        
        # Add channel embeddings
        channel_emb = rearrange(channel_emb, 'Cin E -> 1 E Cin 1 1')

        if self.use_channelToken:
            x_proj = x_proj + channel_emb
        
        # Flatten spatial and channel dimensions
        # x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B (Cin Hp Wp) E')
        x_out = rearrange(x_proj, 'B E Cin Hp Wp -> B Hp Wp (Cin E)')
        x_out = self.embedding_proj(x_out)
        return x_out, Cin, total_loss

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, query, key, value):
        # query: [B, 1, D], key/value: [B, N, D]
        out, _ = self.attn(query, key, value)
        return out  # [B, 1, D]

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class ChannelAttSegAndEdge(nn.Module):
    def __init__(self, vit, image_size=256, patch_size=16, feature_dim=768, num_heads=8,in_channels=4):
        super().__init__()
        self.vit = vit
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Multi-token attention（拼接 seg + edge）
        self.attn1 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)

        self.attn2 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.attn3 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(feature_dim)

        total_dim = in_channels * feature_dim

        # 上采样层,用于生成hq_feature，暂时只使用第一层特征,
        if vit.use_multi_feature:
            self.use_multi_feature = True   
            self.up_encoder_feature_first = nn.Sequential(
                nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
            )
            self.up_encoder_feature_last = nn.Sequential(
                nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
            )
        else:
            self.use_multi_feature = False
            
        # 上采样模块,处理和token交互后的图像特征
        self.up = nn.Sequential(
            nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 8),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 16),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
        )

        self.final_channels = total_dim // 32
        self.in_channels = in_channels  

        # 编码上采样后的图像特征
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(self.final_channels, feature_dim // 4, 3, 1, 1),
            LayerNorm2d(feature_dim // 4),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, self.final_channels, 3, 1, 1),
        )
        # Token 投影
        self.seg_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )
        # 输出激活
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. ViT 提取特征
        result = self.vit(x)
        seg_token = result['seg_token']        # [B, D]
        edge_token = result['edge_token']      # [B, D]
        patch_tokens = result['patch_tokens']  # [B, N, D]
        ortho_proxy_loss = result['ortho_proxy_loss']
        
        feature_token_last = patch_tokens[-1]
        B, N, D = feature_token_last.shape
        H = W = self.image_size // self.patch_size
        if self.use_multi_feature:
            feature_token_first = patch_tokens[0]
            feature_token_first = feature_token_first.view(B, self.in_channels, H, W, D)
            feature_token_first = feature_token_first.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)

            feature_token_last = patch_tokens[-1]
            feature_token_last = feature_token_last.view(B, self.in_channels, H, W, D)
            feature_token_last = feature_token_last.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)

            hq_feature_first = self.up_encoder_feature_first(feature_token_first)
            hq_feature_last = self.up_encoder_feature_last(feature_token_last)
            hq_feature = hq_feature_first + hq_feature_last
        
        
        feature_token_last = patch_tokens[-1]

        # 2. 拼接 seg + edge token -> [B, 2, D]
        tokens = torch.stack([seg_token, edge_token], dim=1)

        # 3. token <-> feature attention
        updated_token_1, _ = self.attn1(tokens, feature_token_last, feature_token_last)
        updated_token_1 = self.norm1(updated_token_1 + tokens)

        updated_feature_2, _ = self.attn2(feature_token_last, tokens, tokens)
        updated_feature_2 = self.norm2(updated_feature_2 + feature_token_last)

        updated_token_2, _ = self.attn3(updated_token_1, updated_feature_2, updated_feature_2)
        updated_token_2 = self.norm3(updated_token_2 + updated_token_1)
        
        seg_token, edge_token = updated_token_2[:, 0], updated_token_2[:, 1]  # [B, D], [B, D]

        # 4. 还原 patch_tokens 到空间特征 [B, D, H, W]
        feat_rec = updated_feature_2.view(B, self.in_channels, H, W, D)
        feat = feat_rec.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
        # 5. 上采样到原图尺寸
        up_feat = self.up(feat)  # [B, C', H*4, W*4]

        # 6. 嵌入 mask 特征
        embed_feat = self.embedding_maskfeature(up_feat)  # [B, C, H*4, W*4]
        if self.use_multi_feature:
            embed_feat = embed_feat + hq_feature
    
        # 7. token 投影
        seg_proj = self.seg_proj(seg_token)     # [B, C]
        edge_proj = self.edge_proj(edge_token)  # [B, C]

        # 8. 点积生成 mask
        seg_mask = (embed_feat * seg_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        edge_mask = (embed_feat * edge_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)

        # # 9. 输出 mask
        # seg_mask = self.sigmoid(seg_mask)
        # edge_mask = self.sigmoid(edge_mask)

        return {
            'seg_mask': seg_mask,                  # [B, 1, H, W]
            'edge_mask': edge_mask,                # [B, 1, H, W]
            'seg_token': seg_token,
            'edge_token': edge_token,
            'ortho_proxy_loss': ortho_proxy_loss
        }

# 将seg 和 edge token 设置在ViT之外
class ChannelAttSegAndEdge_V2(nn.Module):
    def __init__(self, vit, image_size=256, patch_size=16, feature_dim=768, num_heads=8,in_channels=4):
        super().__init__()
        self.vit = vit
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.seg_token = nn.Parameter(torch.randn(1, feature_dim))
        self.edge_token = nn.Parameter(torch.randn(1, feature_dim))

        # Multi-token attention（拼接 seg + edge） 
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.attn1 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp1 = MLP(feature_dim, feature_dim, feature_dim, 3)

        self.attn2 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.attn3 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(feature_dim)

        total_dim = in_channels * feature_dim

        # 上采样层,用于生成hq_feature，暂时只使用第一层特征,
        if vit.use_multi_feature:
            self.use_multi_feature = True   
            self.up_encoder_feature_first = nn.Sequential(
                nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
            )
            self.up_encoder_feature_last = nn.Sequential(
                nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
            )
        else:
            self.use_multi_feature = False
            
        # 上采样模块,处理和token交互后的图像特征
        self.up = nn.Sequential(
            nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 8),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 16),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
        )

        self.final_channels = total_dim // 32
        self.in_channels = in_channels  

        # 编码上采样后的图像特征
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(self.final_channels, feature_dim // 4, 3, 1, 1),
            LayerNorm2d(feature_dim // 4),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, self.final_channels, 3, 1, 1),
        )
        # Token 投影
        self.seg_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )

        # self.seg_proj = nn.Sequential(
        #     nn.Linear(feature_dim, feature_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(feature_dim//2, self.final_channels)
        # )
        # self.edge_proj = nn.Sequential(
        #     nn.Linear(feature_dim, feature_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(feature_dim//2, self.final_channels)
        # )

        self.edge_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )
        # 输出激活
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. ViT 提取特征
        result = self.vit(x)
        seg_token = result['seg_token']        # [B, D]
        edge_token = result['edge_token']      # [B, D]
        patch_tokens = result['patch_tokens']  # [B, N, D]
        ortho_proxy_loss = result['ortho_proxy_loss']
        
        feature_token_last = patch_tokens[-1]
        B, N, D = feature_token_last.shape
        H = W = self.image_size // self.patch_size
        if self.use_multi_feature:
            feature_token_first = patch_tokens[0]
            feature_token_first = feature_token_first.view(B, self.in_channels, H, W, D)
            feature_token_first = feature_token_first.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
            # feature_token_first = feature_token_first.permute(0, 4, 1, 2, 3).reshape(B, self.in_channels * D, H, W)

            feature_token_last = patch_tokens[-1]
            feature_token_last = feature_token_last.view(B, self.in_channels, H, W, D)
            feature_token_last = feature_token_last.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
            # feature_token_last = feature_token_last.permute(0, 4, 1, 2, 3).reshape(B, self.in_channels * D, H, W)

            hq_feature_first = self.up_encoder_feature_first(feature_token_first)
            hq_feature_last = self.up_encoder_feature_last(feature_token_last)
            hq_feature = hq_feature_first + hq_feature_last
        
        
        feature_token_last = patch_tokens[-1]

        # 2. 拼接 seg + edge token -> [B, 2, D]
        # tokens = torch.stack([seg_token, edge_token], dim=1)
        tokens = torch.stack([self.seg_token, self.edge_token], dim=1)
        tokens = tokens.expand(B,-1,-1)
        # 3. token <-> feature attention
        tokens = self.self_attn(tokens, tokens, tokens)[0]

        updated_token_1, _ = self.attn1(tokens, feature_token_last, feature_token_last)
        updated_token_1 = self.norm1(updated_token_1 + tokens)
        updated_token_1 = self.mlp1(updated_token_1)

        updated_feature_2, _ = self.attn2(feature_token_last, tokens, tokens)
        updated_feature_2 = self.norm2(updated_feature_2 + feature_token_last)

        updated_token_2, _ = self.attn3(updated_token_1, updated_feature_2, updated_feature_2)
        updated_token_2 = self.norm3(updated_token_2 + updated_token_1)
        
        seg_token, edge_token = updated_token_2[:, 0], updated_token_2[:, 1]  # [B, D], [B, D]

        # 4. 还原 patch_tokens 到空间特征 [B, D, H, W]
        feat_rec = updated_feature_2.view(B, self.in_channels, H, W, D)
        feat = feat_rec.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
        # 5. 上采样到原图尺寸
        up_feat = self.up(feat)  # [B, C', H*4, W*4]

        # 6. 嵌入 mask 特征
        embed_feat = self.embedding_maskfeature(up_feat)  # [B, C, H*4, W*4]
        if self.use_multi_feature:
            embed_feat = embed_feat + hq_feature
    
        # 7. token 投影
        seg_proj = self.seg_proj(seg_token)     # [B, C]
        edge_proj = self.edge_proj(edge_token)  # [B, C]

        # 8. 点积生成 mask
        seg_mask = (embed_feat * seg_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        edge_mask = (embed_feat * edge_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)

        # # 9. 输出 mask
        # seg_mask = self.sigmoid(seg_mask)
        # edge_mask = self.sigmoid(edge_mask)

        return {
            'seg_mask': seg_mask,                  # [B, 1, H, W]
            'edge_mask': edge_mask,                # [B, 1, H, W]
            'seg_token': seg_token,
            'edge_token': edge_token,
            'ortho_proxy_loss': ortho_proxy_loss,
            'patch_tokens': patch_tokens,
        }
    
# 强制让模型输出和SAM的输出对齐，以便使其分布对齐
class ChannelAttSegAndEdge_V3(nn.Module):
    def __init__(self, vit, image_size=256, patch_size=16, feature_dim=768, num_heads=8,in_channels=4):
        super().__init__()
        self.vit = vit
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.seg_token = nn.Parameter(torch.randn(1, feature_dim))
        self.edge_token = nn.Parameter(torch.randn(1, feature_dim))

        # Multi-token attention（拼接 seg + edge） 
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.attn1 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp1 = MLP(feature_dim, feature_dim, feature_dim, 3)

        self.attn2 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.attn3 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(feature_dim)

        total_dim = in_channels * feature_dim
        self.final_channels = 256 # SAM 特征通道
        # 上采样层,用于生成hq_feature，暂时只使用第一层特征,
        if vit.use_multi_feature:
            self.use_multi_feature = True   
            self.up_encoder_feature_first = nn.Sequential(
                nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
                nn.Conv2d(total_dim // 32, self.final_channels, kernel_size=1),
            )

            self.up_encoder_feature_last = nn.Sequential(
                nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),
                nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
                nn.Conv2d(total_dim // 32, self.final_channels, kernel_size=1),
            )
        else:
            self.use_multi_feature = False
            
        # 上采样模块,处理和token交互后的图像特征
        self.up = nn.Sequential(
            nn.ConvTranspose2d(total_dim, total_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 4, total_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 8),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 8, total_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(total_dim // 16),
            nn.GELU(),
            nn.ConvTranspose2d(total_dim // 16, total_dim // 32, kernel_size=2, stride=2),
        )

        

        self.linear_align = FeatureCompressor_linear(total_dim // 32,self.final_channels)
        self.up2 = FeatureUpsampler(self.final_channels,self.final_channels) # 在对齐特征的基础上，再次上采样

        self.in_channels = in_channels  

        # 编码上采样后的图像特征
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(self.final_channels, feature_dim // 4, 3, 1, 1),
            LayerNorm2d(feature_dim // 4),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, self.final_channels, 3, 1, 1),
        )
        # Token 投影
        self.seg_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )
        # 输出激活
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. ViT 提取特征
        result = self.vit(x)
        seg_token = result['seg_token']        # [B, D]
        edge_token = result['edge_token']      # [B, D]
        patch_tokens = result['patch_tokens']  # [B, N, D]
        ortho_proxy_loss = result['ortho_proxy_loss']
        
        feature_token_last = patch_tokens[-1]
        B, N, D = feature_token_last.shape
        H = W = self.image_size // self.patch_size
        if self.use_multi_feature:
            feature_token_first = patch_tokens[0]
            feature_token_first = feature_token_first.view(B, self.in_channels, H, W, D)
            feature_token_first = feature_token_first.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
            # feature_token_first = feature_token_first.permute(0, 4, 1, 2, 3).reshape(B, self.in_channels * D, H, W)

            feature_token_last = patch_tokens[-1]
            feature_token_last = feature_token_last.view(B, self.in_channels, H, W, D)
            feature_token_last = feature_token_last.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
            # feature_token_last = feature_token_last.permute(0, 4, 1, 2, 3).reshape(B, self.in_channels * D, H, W)

            hq_feature_first = self.up_encoder_feature_first(feature_token_first)
            hq_feature_last = self.up_encoder_feature_last(feature_token_last)
            hq_feature = hq_feature_first + hq_feature_last
        
        
        feature_token_last = patch_tokens[-1]

        # 2. 拼接 seg + edge token -> [B, 2, D]
        # tokens = torch.stack([seg_token, edge_token], dim=1)
        tokens = torch.stack([self.seg_token, self.edge_token], dim=1)
        tokens = tokens.expand(B,-1,-1)
        # 3. token <-> feature attention
        tokens = self.self_attn(tokens, tokens, tokens)[0]

        updated_token_1, _ = self.attn1(tokens, feature_token_last, feature_token_last)
        updated_token_1 = self.norm1(updated_token_1 + tokens)
        updated_token_1 = self.mlp1(updated_token_1)

        updated_feature_2, _ = self.attn2(feature_token_last, tokens, tokens)
        updated_feature_2 = self.norm2(updated_feature_2 + feature_token_last)

        updated_token_2, _ = self.attn3(updated_token_1, updated_feature_2, updated_feature_2)
        updated_token_2 = self.norm3(updated_token_2 + updated_token_1)
        
        seg_token, edge_token = updated_token_2[:, 0], updated_token_2[:, 1]  # [B, D], [B, D]

        # 4. 还原 patch_tokens 到空间特征 [B, D, H, W]
        feat_rec = updated_feature_2.view(B, self.in_channels, H, W, D)
        feat = feat_rec.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
        # 5. 上采样到原图尺寸
        up_feat = self.up(feat)  # [B, C', H*4, W*4]
        embed_feat_align = self.linear_align(up_feat)
        embed_feat = self.up2(embed_feat_align)
        # 6. 嵌入 mask 特征
        embed_feat = self.embedding_maskfeature(embed_feat)  # [B, C, H*4, W*4]
        if self.use_multi_feature:
            embed_feat = embed_feat + hq_feature

          # [B, C, H*4, W*4]

        # 7. token 投影
        seg_proj = self.seg_proj(seg_token)     # [B, C]
        edge_proj = self.edge_proj(edge_token)  # [B, C]

        # 8. 点积生成 mask
        seg_mask = (embed_feat * seg_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        edge_mask = (embed_feat * edge_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)

        # # 9. 输出 mask
        # seg_mask = self.sigmoid(seg_mask)
        # edge_mask = self.sigmoid(edge_mask)

        return {
            'seg_mask': seg_mask,                  # [B, 1, H, W]
            'edge_mask': edge_mask,                # [B, 1, H, W]
            'seg_token': seg_token,
            'edge_token': edge_token,
            'ortho_proxy_loss': ortho_proxy_loss,
            'embed_feat_align': embed_feat_align,
        }

# 锯齿输出 优化
class ChannelAttSegAndEdge_V4(nn.Module):
    def __init__(self, vit, image_size=256, patch_size=16, feature_dim=768, num_heads=8,in_channels=4):
        super().__init__()
        self.vit = vit
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.seg_token = nn.Parameter(torch.randn(1, feature_dim))
        self.edge_token = nn.Parameter(torch.randn(1, feature_dim))

        # Multi-token attention（拼接 seg + edge） 
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.attn1 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp1 = MLP(feature_dim, feature_dim, feature_dim, 3)

        self.attn2 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.attn3 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(feature_dim)

        total_dim = in_channels * feature_dim
        self.final_channels = 256 # SAM 特征通道
        # 上采样层,用于生成hq_feature，暂时只使用第一层特征,
        if vit.use_multi_feature:
            self.use_multi_feature = True  
            self.up_encoder_feature_first = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim, total_dim // 4, kernel_size=3, padding=1),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim // 4, total_dim // 8, kernel_size=3, padding=1),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim // 8, total_dim // 16, kernel_size=3, padding=1),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim // 16, self.final_channels, kernel_size=3, padding=1)
            )

            self.up_encoder_feature_last = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim, total_dim // 4, kernel_size=3, padding=1),
                LayerNorm2d(total_dim // 4),
                nn.GELU(),
                
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim // 4, total_dim // 8, kernel_size=3, padding=1),
                LayerNorm2d(total_dim // 8),
                nn.GELU(),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim // 8, total_dim // 16, kernel_size=3, padding=1),
                LayerNorm2d(total_dim // 16),
                nn.GELU(),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(total_dim // 16, self.final_channels, kernel_size=3, padding=1)
            )
        else:
            self.use_multi_feature = False

        # 上采样模块,处理和token交互后的图像特征
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(total_dim, total_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(total_dim // 4),
            nn.GELU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(total_dim // 4, total_dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(total_dim // 8),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(total_dim // 8, total_dim // 16, kernel_size=3, padding=1),
            LayerNorm2d(total_dim // 16),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(total_dim // 16, total_dim // 32, kernel_size=3, padding=1)
        )

        self.linear_align = FeatureCompressor_linear(total_dim // 32,self.final_channels)
        self.up2 = FeatureUpsampler(self.final_channels,self.final_channels) # 在对齐特征的基础上，再次上采样

        self.in_channels = in_channels  

        # 编码上采样后的图像特征
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(self.final_channels, feature_dim // 4, 3, 1, 1),
            LayerNorm2d(feature_dim // 4),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, self.final_channels, 3, 1, 1),
        )
        # Token 投影
        self.seg_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, self.final_channels)
        )
        # 输出激活
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. ViT 提取特征
        result = self.vit(x)
        seg_token = result['seg_token']        # [B, D]
        edge_token = result['edge_token']      # [B, D]
        patch_tokens = result['patch_tokens']  # [B, N, D]
        ortho_proxy_loss = result['ortho_proxy_loss']
        
        feature_token_last = patch_tokens[-1]
        B, N, D = feature_token_last.shape
        H = W = self.image_size // self.patch_size
        if self.use_multi_feature:
            feature_token_first = patch_tokens[0]
            feature_token_first = feature_token_first.view(B, self.in_channels, H, W, D)
            feature_token_first = feature_token_first.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
            # feature_token_first = feature_token_first.permute(0, 4, 1, 2, 3).reshape(B, self.in_channels * D, H, W)

            feature_token_last = patch_tokens[-1]
            feature_token_last = feature_token_last.view(B, self.in_channels, H, W, D)
            feature_token_last = feature_token_last.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
            # feature_token_last = feature_token_last.permute(0, 4, 1, 2, 3).reshape(B, self.in_channels * D, H, W)

            hq_feature_first = self.up_encoder_feature_first(feature_token_first)
            hq_feature_last = self.up_encoder_feature_last(feature_token_last)
            hq_feature = hq_feature_first + hq_feature_last
        
        feature_token_last = patch_tokens[-1]
        # 2. 拼接 seg + edge token -> [B, 2, D]
        tokens = torch.stack([self.seg_token, self.edge_token], dim=1)
        tokens = tokens.expand(B,-1,-1)
        # 3. token <-> feature attention
        tokens = self.self_attn(tokens, tokens, tokens)[0]

        updated_token_1, _ = self.attn1(tokens, feature_token_last, feature_token_last)
        updated_token_1 = self.norm1(updated_token_1 + tokens)
        updated_token_1 = self.mlp1(updated_token_1)

        updated_feature_2, _ = self.attn2(feature_token_last, tokens, tokens)
        updated_feature_2 = self.norm2(updated_feature_2 + feature_token_last)

        updated_token_2, _ = self.attn3(updated_token_1, updated_feature_2, updated_feature_2)
        updated_token_2 = self.norm3(updated_token_2 + updated_token_1)
        
        seg_token, edge_token = updated_token_2[:, 0], updated_token_2[:, 1]  # [B, D], [B, D]

        # 4. 还原 patch_tokens 到空间特征 [B, D, H, W]
        feat_rec = updated_feature_2.view(B, self.in_channels, H, W, D)
        feat = feat_rec.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)

        # 5. 上采样到原图尺寸
        up_feat = self.up(feat)  # [B, C', H*4, W*4]
        embed_feat_align = self.linear_align(up_feat)
        embed_feat = self.up2(embed_feat_align)

        # 6. 嵌入 mask 特征
        embed_feat = self.embedding_maskfeature(embed_feat)  # [B, C, H*4, W*4]
        if self.use_multi_feature:
            embed_feat = embed_feat + hq_feature

        # [B, C, H*4, W*4]

        # 7. token 投影
        seg_proj = self.seg_proj(seg_token)     # [B, C]
        edge_proj = self.edge_proj(edge_token)  # [B, C]

        # 8. 点积生成 mask
        seg_mask = (embed_feat * seg_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        edge_mask = (embed_feat * edge_proj.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)

        return {
            'seg_mask': seg_mask,                  # [B, 1, H, W]
            'edge_mask': edge_mask,                # [B, 1, H, W]
            'seg_token': seg_token,
            'edge_token': edge_token,
            'ortho_proxy_loss': ortho_proxy_loss,
            'embed_feat_align': embed_feat_align,
        }


class ChannelProjection(nn.Module):
    def __init__(self, in_channels=768*4, out_channels=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)
# Unet模式的上采样
class ChannelAttSegAndEdge_V5(nn.Module):
    def __init__(self, vit, image_size=256, patch_size=16, feature_dim=768, num_heads=8,in_channels=4):
        super().__init__()
        self.vit = vit
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.seg_token = nn.Parameter(torch.randn(1, feature_dim))
        self.edge_token = nn.Parameter(torch.randn(1, feature_dim))

        # Multi-token attention（拼接 seg + edge） 
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.attn1 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp1 = MLP(feature_dim, feature_dim, feature_dim, 3)

        self.attn2 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.attn3 = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(feature_dim)

        self.dim_prj_list = nn.ModuleList([
            ChannelProjection(in_channels=768*4, out_channels=768)
            for _ in range(4)  # 举例：构建4个
        ])
        total_dim = in_channels * feature_dim
        self.final_channels = 256 # SAM 特征通道
        # 上采样层,用于生成hq_feature，暂时只使用第一层特征,
        if vit.use_multi_feature:
            self.use_multi_feature = True
            out_channels = [ feature_dim for _ in range(4)]
            output_strides = [16 for _ in range(4)]

            self.multi_scale_decoder = DPTDecoder(encoder_out_channels=out_channels,
            encoder_output_strides=output_strides,
            encoder_has_prefix_tokens=True,
            readout='cat',
            intermediate_channels=(256,512,1024,1024),
            fusion_channels=self.final_channels,)
        else:
            self.use_multi_feature = False
        self.seg_head = DPTSegmentationHead(
            in_channels=self.final_channels,
            out_channels=1,
            activation=None,
            kernel_size=3,
            upsampling=2,
        )
        self.edge_head = DPTSegmentationHead(
            in_channels=self.final_channels,
            out_channels=1,
            activation=None,
            kernel_size=3,
            upsampling=2,
        )
        # 上采样模块,处理和token交互后的图像特征
        self.in_channels = in_channels  

    def forward(self, x):
        # 1. ViT 提取特征
        result = self.vit(x)
        seg_token = result['seg_token']        # [B, D]
        edge_token = result['edge_token']      # [B, D]
        patch_tokens = result['patch_tokens']  # [B, N, D]
        ortho_proxy_loss = result['ortho_proxy_loss']
        
        feature_token_last = patch_tokens[-1]
        B, N, D = feature_token_last.shape
        H = W = self.image_size // self.patch_size

        token_2_image =[]
        for i, item in enumerate(patch_tokens):
            feat_unflatten = patch_tokens[i].view(B, self.in_channels, H, W, D)
            feat = feat_unflatten.permute(0, 1, 4, 2, 3).reshape(B, self.in_channels * D, H, W)
            feat = self.dim_prj_list[i](feat)
            token_2_image.append(feat)  
        tmp_tk = self.seg_token.expand(B,1, -1)
        tmp_tk_list = [tmp_tk for _ in range(len(token_2_image))]
        decoder_output = self.multi_scale_decoder(token_2_image,tmp_tk_list)
        seg_mask = self.seg_head(decoder_output)
        edge_mask = self.edge_head(decoder_output)

        return {
            'seg_mask': seg_mask,                  # [B, 1, H, W]
            'edge_mask': edge_mask,                # [B, 1, H, W]
            'seg_token': seg_token,
            'edge_token': edge_token,
            'ortho_proxy_loss': ortho_proxy_loss,

        }

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_unified_visualization(tensor_list, save_paths, titles=None):
    """
    将多个张量按统一标准可视化并保存
    tensor_list: List of torch.Tensor, 形状均为 [1, 3, 1024, 1024] 或 [1, C, H, W]
    """
    
    processed_images = []
    for tensor in tensor_list:
        # 1. 去掉 batch 维度 -> [3, 1024, 1024]
        img = tensor.squeeze(0).detach().cpu()
        
        # 2. 如果是多通道表征，取通道平均值转为热力图，如果是RGB则跳过此步
        # 这里假设我们要看的是“激活强度”，所以取平均
        img = torch.mean(img, dim=0).numpy() 
        processed_images.append(img)

    # 重点：计算所有图像的全局最大值和最小值，确保颜色标准一致
    global_min = min([img.min() for img in processed_images])
    global_max = max([img.max() for img in processed_images])

    print(f"统一值域范围: [{global_min:.4f}, {global_max:.4f}]")

    for i, img in enumerate(processed_images):
        plt.figure(figsize=(10, 10))
        
        # 使用相同的 vmin 和 vmax
        im = plt.imshow(img, cmap='viridis', vmin=global_min, vmax=global_max)
        
        # 添加颜色条，方便观察数值对应关系
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        if titles:
            plt.title(titles[i])
        
        plt.axis('off')
        
        # 保存到本地
        plt.savefig(save_paths[i], bbox_inches='tight', dpi=300)
        plt.close()
        print(f"已保存: {save_paths[i]}")
def visualize_band_correlation(tensor, output_dir, name_prefix="default", method="pearson"):
    """
    可视化各个波段之间的相关性 (Pearson 或 Cosine)
    
    Args:
        tensor: torch.Tensor, shape (B, C, Bands, H, W)
        output_dir: 输出路径
        name_prefix: 文件名前缀
        method: 'pearson' 或 'cosine'
    """
    os.makedirs(os.path.join(output_dir, "band_correlation"), exist_ok=True)
    B, C, Bands, H, W = tensor.shape
    
    for b in range(B):
        data = tensor[b]  # shape: (C, Bands, H, W)

        # 先把 C, H, W 拉平，得到每个 band 的特征向量
        # shape: (Bands, C*H*W)
        band_vectors = data.permute(1, 0, 2, 3).reshape(Bands, -1).cpu().numpy()

        if method == "pearson":
            # 计算皮尔逊相关系数
            corr_matrix = np.corrcoef(band_vectors)
        elif method == "cosine":
            corr_matrix = cosine_similarity(band_vectors)
        else:
            raise ValueError("method 必须是 'pearson' 或 'cosine'")

        # 可视化
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1 if method=="pearson" else 0, vmax=1,
                    xticklabels=[f"Band {i}" for i in range(Bands)],
                    yticklabels=[f"Band {i}" for i in range(Bands)])
        plt.title(f"Band Correlation Matrix ({method}) - {name_prefix}_B{b}")
        plt.xlabel("Band Index")
        plt.ylabel("Band Index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "band_correlation", f"{name_prefix}_B{b}_{method}.png"))
        plt.close()



if __name__ == '__main__':  
    prompt_embed_dim = 256
    image_size = 256
    vit_patch_size = 16
    channels = 4
    # image_embedding_size = image_size // vit_patch_size
    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]
    vit_channel= ChannelVisionTransformer( #
        img_size=image_size,
        patch_size=vit_patch_size,
        in_chans=channels,
        embed_dim=encoder_embed_dim,
        depth=encoder_depth,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
    )

    SegAndEdgePredicotr = ChannelAttSegAndEdge(vit_channel, image_size=image_size, patch_size=vit_patch_size, feature_dim=encoder_embed_dim, num_heads=encoder_num_heads)
    vit_channel.to('cuda')
    SegAndEdgePredicotr.to('cuda')

    # Step 4: 输入一个假图像（或者你自己的图像）
    image_path = "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/train/MultiSpectral/AT_4560_S2_10m_256.tif"
    image = iio.imread(image_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).float().to('cuda') * 255
    # dummy_input = torch.randn(1, 4, 1024, 1024).to('cuda')  # 输入 shape 按你的 patch 设置而定

    # Step 5: Forward 推理
    vit_channel.eval()
    # with torch.no_grad():
    result = SegAndEdgePredicotr(image_tensor)
    seg_mask , edge_mask , ortho_proxy_loss = result['seg_mask'], result['edge_mask'], result['ortho_proxy_loss']
    # cls_token , seg_token, edge_token, feature_tokens,ortho_proxy_loss = result['cls_token'], result['seg_token'], result['edge_token'], result['patch_tokens'], result['ortho_proxy_loss']
    # B, N, D = feature_tokens.shape
    # H_patch = image_size // vit_patch_size
    # W_patch = image_size // vit_patch_size
    # x = feature_tokens.reshape(B, channels, H_patch, W_patch, D).permute(0, 4, 1, 2, 3)  # B, C, C, H_patch, W_patch
    # x = x.reshape(B, channels*D, H_patch, W_patch )  # 也可以 reshape 为 2D 特征图
        # if len(result) == 3:
        #    x_out, Cin, total_loss  = result
        # else:
        #     compress_embed, interm = result
        # compress_embed, interm, reconstruct_image  = vit_channel(image_tensor)
    
    image_RGB = seg_mask.cpu().numpy().squeeze()
    image_RGB = np.clip(image_RGB, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    image_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite("vit_reconstruct_image.png", image_RGB)
    a = 1




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