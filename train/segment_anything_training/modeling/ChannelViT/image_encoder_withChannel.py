# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Channel-token patch embedding history summary:

- PatchEmbedPerChannel / V1_1:
  Early per-channel patch tokenization experiments.
- V2_1 ~ V2_4:
  Intermediate channel-attention and fusion explorations.
- V2_5:
  Final retained paper path. Per-channel projection is followed by
  attention-based channel fusion and then mapped back into the SAM patch
  embedding space.
- V2_6 / V2_7:
  Spectral positional encoding and channel sampling explorations.
- ChannelVisionTransformer / ChannelAttSegAndEdge*:
  Separate experimental branch, not part of the final paper pipeline.

The full historical implementations are preserved in
`channel_patch_embed_legacy.py` for compatibility with older training and
testing scripts. This file only re-exports the mainline `2_5*` path and the
small set of helpers it depends on.
"""

from .channel_patch_embed_legacy import (
    LayerNorm2d,
    MLP,
    PatchEmbed,
    PatchEmbedPerChannelV2_5,
    PerPixelChannelAttentionWithUpsample,
    PerPixelChannelAttentionWithUpsampleV2,
    PerPixelChannelAttentionWithUpsampleV3,
    ChannelSelfAttention,
    ortho_proj_loss_fn_v2,
    ortho_proj_loss_fn_v3,
    pairwise_distance_v2,
    band_entropy_maximization,
    proxy_loss_fn,
    manually_load_qkv_lora_weights,
    replace_qkv_with_lora,
)

__all__ = [
    "LayerNorm2d",
    "MLP",
    "PatchEmbed",
    "PatchEmbedPerChannelV2_5",
    "PerPixelChannelAttentionWithUpsample",
    "PerPixelChannelAttentionWithUpsampleV2",
    "PerPixelChannelAttentionWithUpsampleV3",
    "ChannelSelfAttention",
    "ortho_proj_loss_fn_v2",
    "ortho_proj_loss_fn_v3",
    "pairwise_distance_v2",
    "band_entropy_maximization",
    "proxy_loss_fn",
    "manually_load_qkv_lora_weights",
    "replace_qkv_with_lora",
]
