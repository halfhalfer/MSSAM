# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Decoder history summary for this repo:

- MaskDecoderHQ:
  Early HQ-SAM adaptation built on top of the original SAM decoder.
- MaskDecoder_HQ_Edge:
  Joint semantic-mask and edge prediction exploration.
- MaskDecoder_Fusion / v2 / v3:
  Early token-fusion and instance-decoding attempts.
- MaskDecoder_Fusion_v3_1 / v3_3 / v3_4 / v3_5:
  Local ablations around structure loss, token fusion, and returned features.
- MaskDecoder_Fusion_v4 ~ v9:
  Later experiments on instance-token generation, dynamic decoding, and
  multi-stage refinement.
- MultiSegToken series:
  Multi-seg-token exploration that was not kept in the final paper pipeline.

Final retained paper path:
- MaskDecoder:
  Base SAM mask decoder used by the original builders.
- MaskDecoder_Fusion_v3_2:
  Mainline decoder kept for the paper code release. It predicts semantic mask,
  edge mask, and instance masks, and optionally returns the final embedding for
  structure loss.
"""

from pathlib import Path
from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .instanceDecoder import (
    InstanceMaskDecoder,
    InstanceMaskDecoder_v2,
    InstanceMaskDecoder_v3,
    InstanceMaskDecoder_v4,
)
from .transformer import CrossAttentionFusion_v4_1, TwoWayTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASKDECODER_B_CHECKPOINT = PROJECT_ROOT / "sam-hq-param" / "sam_vit_b_maskdecoder.pth"


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


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


class MaskDecoder_Fusion_v3_2(MaskDecoder):
    def __init__(
        self,
        model_type,
        fusion_v=2,
        instacnDecoder=2,
        num_instance_tokens=100,
        use_multiscale_imagefeat=False,
        use_cluster_token=False,
        return_final_embed=True,
        use_fusion=True,
    ):
        super().__init__(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_multimask_outputs=3,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        assert model_type in [
            "vit_b",
            "vit_l",
            "vit_h",
            "vit_h_MultiSpectral",
            "vit_l_MultiSpectral",
            "vit_b_MultiSpectral",
            "channelToken",
        ]

        checkpoint_dict = {
            "vit_b": str(DEFAULT_MASKDECODER_B_CHECKPOINT),
            "vit_l": str(DEFAULT_MASKDECODER_B_CHECKPOINT),
            "vit_h": str(DEFAULT_MASKDECODER_B_CHECKPOINT),
            "vit_h_MultiSpectral": str(DEFAULT_MASKDECODER_B_CHECKPOINT),
            "vit_l_MultiSpectral": str(DEFAULT_MASKDECODER_B_CHECKPOINT),
            "vit_b_MultiSpectral": str(DEFAULT_MASKDECODER_B_CHECKPOINT),
            "channelToken": str(DEFAULT_MASKDECODER_B_CHECKPOINT),
        }
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for _, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim = 256
        vit_dim_dict = {
            "vit_b": 768,
            "vit_l": 1024,
            "vit_h": 1280,
            "vit_h_MultiSpectral": 1280,
            "vit_l_MultiSpectral": 1024,
            "vit_b_MultiSpectral": 768,
            "channelToken": 768,
        }
        vit_dim = vit_dim_dict[model_type]

        self.use_multiscale_imagefeat = use_multiscale_imagefeat
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        if self.use_multiscale_imagefeat:
            self.compress_vit_feat = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            vit_dim, transformer_dim, kernel_size=2, stride=2
                        ),
                        LayerNorm2d(transformer_dim),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            transformer_dim,
                            transformer_dim // 8,
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                    for _ in range(3)
                ]
            )
        else:
            self.compress_vit_feat = nn.Sequential(
                nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim),
                nn.GELU(),
                nn.ConvTranspose2d(
                    transformer_dim, transformer_dim // 8, kernel_size=2, stride=2
                ),
            )

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )

        self.edge_token = nn.Embedding(1, transformer_dim)
        self.edge_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.num_instance_tokens = num_instance_tokens
        self.instance_token = nn.Embedding(self.num_instance_tokens, transformer_dim)
        self.num_mask_tokens = self.num_mask_tokens + 2 + self.num_instance_tokens
        self.fusion = CrossAttentionFusion_v4_1(dim=transformer_dim)
        self.use_fusion = use_fusion

        if instacnDecoder == 1:
            self.instance_mask_decoder = InstanceMaskDecoder(
                token_dim=transformer_dim, feature_dim=transformer_dim // 8
            )
        elif instacnDecoder == 2:
            self.instance_mask_decoder = InstanceMaskDecoder_v2(
                token_dim=transformer_dim, feature_dim=transformer_dim // 8
            )
        elif instacnDecoder == 3:
            self.instance_mask_decoder = InstanceMaskDecoder_v3(
                token_dim=transformer_dim,
                feature_dim=transformer_dim // 8,
                if_tokenCluster=use_cluster_token,
            )
        elif instacnDecoder == 4:
            assert self.use_multiscale_imagefeat is True
            self.instance_mask_decoder = InstanceMaskDecoder_v4(
                token_dim=transformer_dim, feature_dim=transformer_dim // 8
            )
        else:
            raise ValueError(f"Unsupported instacnDecoder={instacnDecoder}")
        self.return_final_embed = return_final_embed

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vit_features = [interm_embeddings[i].permute(0, 3, 1, 2) for i in range(3)]
        if self.use_multiscale_imagefeat:
            compress_vit_features = [
                self.compress_vit_feat[i](vit_features[i]) for i in range(3)
            ]
        else:
            compress_vit_features = [self.compress_vit_feat(vit_features[0])]

        hq_features = self.embedding_encoder(image_embeddings) + compress_vit_features[0]
        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        class_logits = []
        mask_instance_clusters = []
        for i_batch in range(batch_len):
            predict_masks = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=hq_features[i_batch].unsqueeze(0),
                multiscale_imagefeat=compress_vit_features
                if self.use_multiscale_imagefeat
                else None,
            )
            mask, iou_pred, class_logit, mask_instance_cluster, upscaled_embedding_ours = (
                predict_masks
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
            if class_logit is not None:
                class_logits.append(class_logit)
            if mask_instance_cluster is not None:
                mask_instance_clusters.append(mask_instance_cluster)
        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)
        class_logits = torch.cat(class_logits, 0) if len(class_logits) > 0 else None
        mask_instance_clusters = (
            torch.cat(mask_instance_clusters, 0)
            if len(mask_instance_clusters) > 0
            else None
        )

        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]

        masks_hq = masks[
            :,
            slice(
                self.num_mask_tokens - self.num_instance_tokens - 2,
                self.num_mask_tokens - self.num_instance_tokens - 1,
            ),
            :,
            :,
        ]
        masks_edge = masks[
            :,
            slice(
                self.num_mask_tokens - self.num_instance_tokens - 1,
                self.num_mask_tokens - self.num_instance_tokens,
            ),
            :,
            :,
        ]
        masks_instance = masks[
            :,
            slice(self.num_mask_tokens - self.num_instance_tokens, self.num_mask_tokens),
            :,
            :,
        ]

        self._class_logits = class_logits
        self._mask_instance_clusters = mask_instance_clusters
        if self.return_final_embed:
            if hq_token_only:
                return masks_hq, masks_edge, masks_instance, upscaled_embedding_ours
            return masks_sam, masks_hq, masks_edge, masks_instance, upscaled_embedding_ours
        if hq_token_only:
            return masks_hq, masks_edge, masks_instance
        return masks_sam, masks_hq, masks_edge, masks_instance

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
        multiscale_imagefeat=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        instance_tokens = self.instance_token.weight
        output_tokens = torch.cat(
            [
                self.iou_token.weight,
                self.mask_tokens.weight,
                self.hf_token.weight,
                self.edge_token.weight,
                instance_tokens,
            ],
            dim=0,
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature

        edge_feature = upscaled_embedding_ours
        instance_feature = upscaled_embedding_ours

        seg_token = mask_tokens_out[:, 4, :]
        edge_token = mask_tokens_out[:, 5, :]
        instance_token = mask_tokens_out[:, 5 : 5 + self.num_instance_tokens, :]
        if self.use_fusion:
            fusion_seg_token = self.fusion(
                seg_token.unsqueeze(1), instance_token, edge_token.unsqueeze(1)
            ).squeeze(1)
        else:
            fusion_seg_token = seg_token

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            elif i == 4:
                hyper_in_list.append(self.hf_mlp(fusion_seg_token))
            elif i == 5:
                hyper_in_list.append(self.edge_mlp(mask_tokens_out[:, i, :]))
            else:
                break

        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(
            b, -1, h, w
        )
        masks_ours = (hyper_in[:, 4:5] @ upscaled_embedding_ours.view(b, c, h * w)).view(
            b, -1, h, w
        )
        masks_edge = (hyper_in[:, 5:] @ edge_feature.view(b, c, h * w)).view(
            b, -1, h, w
        )

        if self.use_multiscale_imagefeat:
            masks_instance = self.instance_mask_decoder(
                instance_token, instance_feature, multiscale_imagefeat
            )
        else:
            masks_instance = self.instance_mask_decoder(instance_token, instance_feature)
        if len(masks_instance) == 2:
            class_logits, masks_instance = masks_instance
            mask_instance_cluster = None
        elif len(masks_instance) == 3:
            class_logits, masks_instance, mask_instance_cluster = masks_instance
        else:
            class_logits = None
            mask_instance_cluster = None
        masks = torch.cat([masks_sam, masks_ours, masks_edge, masks_instance], dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred, class_logits, mask_instance_cluster, upscaled_embedding_ours
