# modified from https://github.com/Star-xing1/SAUGE.git

import torch
import torch.nn.init as init

import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from .feature_vis import *
from typing import List, Tuple, Type
from einops import rearrange
from .feature_vis import visualize_feature_map,visualize_instance_tokens,visualize_token_mask_feature,visualize_token_attention_heatmap,save_instance_masks,visualize_token_similarity_matrix
from .common import LayerNorm2d
from .transformer import TwoWayTransformer,CrossAttentionFusion_v3
from .instanceDecoder import InstanceMaskDecoder_v3
from .sauge_blocks import STN_Block, FFB, LayerNorm, SegmentationHead

from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASKDECODER_B_CHECKPOINT = PROJECT_ROOT / "sam-hq-param" / "sam_vit_b_maskdecoder.pth"

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
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
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
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class STN(nn.Module):
    def __init__(self, in_channels,LayerNorm_type="WithBias"):
        super(STN, self).__init__()
        self.dblk_img_embd = STN_Block(256)
        self.dblk_img_shallow_1 = STN_Block(in_channels)
        self.dblk_img_shallow_2 = STN_Block(in_channels // 4)
        self.img_embd_shallow_fuse = FFB(in_channels // 16, in_channels // 16)
        self.mask_fuse = FFB(48, in_channels // 16)
        self.proj_side_1 = nn.Sequential(
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=3, padding=1),
            LayerNorm(in_channels // 32, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 32, in_channels // 64, kernel_size=1),
            LayerNorm(in_channels // 64, LayerNorm_type),
            nn.GELU(),
        )
        self.proj_embd = nn.Sequential(
            nn.Conv2d(32, in_channels // 8, kernel_size=3, padding=1),
            LayerNorm(in_channels // 8, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 8, 48, kernel_size=3, padding=1),
            LayerNorm(48, LayerNorm_type),
            nn.GELU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, in_channels // 16, kernel_size=2, stride=2),
            LayerNorm(in_channels // 16, LayerNorm_type),
            nn.GELU()
        )
        self.proj_side_2 = nn.Sequential(
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=3, padding=1),
            LayerNorm(in_channels // 32, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 32, in_channels // 64, kernel_size=1),
            LayerNorm(in_channels // 64, LayerNorm_type),
            nn.GELU(),
        )
        self.proj_side_3 = nn.Sequential(
            nn.Conv2d(in_channels // 16, in_channels // 32, kernel_size=3, padding=1),
            LayerNorm(in_channels // 32, LayerNorm_type),
            nn.GELU(),
            nn.Conv2d(in_channels // 32, in_channels // 64, kernel_size=1),
            LayerNorm(in_channels // 64, LayerNorm_type),
            nn.GELU(),
        )
        self.proj_gate_1 = nn.Sequential(
            nn.Conv2d(in_channels // 64, 1, kernel_size=3, padding=1),
        )
        self.proj_gate_2 = nn.Sequential(
            nn.Conv2d(in_channels // 64, 1, kernel_size=3, padding=1),
        )
    def postprocess_masks(self, masks, input_size, output_size):
        pass

    def forward(self, *x, mask_embd):
        img_feat_shallow = x[0]
        img_embd = x[1]
        mask_embd = self.proj_embd(mask_embd)

        img_embd = self.dblk_img_embd(img_embd)
        img_embd = self.upsample(img_embd)
        side_output_1 = self.proj_side_1(img_embd)
        gate_1 = torch.sigmoid(self.proj_gate_1(side_output_1))
        # side_output_1 = self.postprocess_masks(side_output_1, 1024, img_size) # remove padding and resize 
        img_embd = gate_1 * img_embd

        img_feat_shallow = self.dblk_img_shallow_1(img_feat_shallow)
        img_feat_shallow = self.dblk_img_shallow_2(img_feat_shallow)
        img_feat_fuse = self.img_embd_shallow_fuse(img_embd, img_feat_shallow)

        side_output_2 = self.proj_side_2(img_feat_fuse)
        gate_2 = torch.sigmoid(self.proj_gate_2(side_output_2))
        # side_output_2 = self.postprocess_masks(side_output_2, 1024, img_size)
        img_feat_fuse = gate_2 * img_feat_fuse

        # fuse
        img_mask_fuse = self.mask_fuse(img_feat_fuse, mask_embd)
        # img_mask_fuse = self.postprocess_masks(img_mask_fuse, 1024, img_size)
        side_output_3 = self.proj_side_3(img_mask_fuse)
        return side_output_1, side_output_2, side_output_3
    
class EdgeDecoder_STN(nn.Module):
    def __init__(self, args, classes=1, mode='train'):
        super(EdgeDecoder_STN, self).__init__()
        
        self.STN = STN(in_channels=768)

        self.segmentation_head = SegmentationHead(
            in_channels=12,
            out_channels=classes,
            kernel_size=1
        )

        self.proj_mask = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            LayerNorm(16, LayerNorm_type="WithBias"),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            LayerNorm(3, LayerNorm_type="WithBias"),
            nn.GELU(),
        )

        self.res_fuse_1 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=1)
        )
        self.res_fuse_2 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=1)
        )
        self.res_fuse_3 = nn.Sequential(
            nn.Conv2d(36, 12, kernel_size=1)
        )

        self.args=args
        self.mode = mode

    def forward(self, image_embeddings, feature_list, m_embd):
        img_H, img_W = 1024,1024

        img_feat_list = [item.permute(0,3,1,2) for item in feature_list]
        mask_embd_list = []
        # img_feat_list = []
        img_feat = []

        for i in range(len(img_feat_list[0])):
            tensors_to_merge = [l[i] for l in img_feat_list]
            merged_tensor = torch.cat(tensors_to_merge, dim=0)
            img_feat.append(merged_tensor.unsqueeze(0))

        img_feat.append(image_embeddings)
        # mask_embd = torch.cat(mask_embd_list, dim=0)
        features_in = img_feat
        # output
        side_output_1, side_output_2, side_output_3 = self.STN(*features_in, mask_embd=m_embd)
        
        side_output_2 = self.res_fuse_1(torch.cat([side_output_1, side_output_2], dim=1))

        side_output_3 = self.res_fuse_2(torch.cat([side_output_2, side_output_3], dim=1))

        multi_outputs = self.res_fuse_3(torch.cat([side_output_1, side_output_2, side_output_3], dim=1))

        final_output = self.segmentation_head(multi_outputs)

        return final_output

class MaskDecoder_SAUGE(MaskDecoder): # 
    def __init__(self, model_type,num_instance_tokens=100,use_cluster_token=False):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,
                        )
        assert model_type in ["vit_b","vit_l","vit_h",'vit_h_MultiSpectral','vit_l_MultiSpectral','vit_b_MultiSpectral']
        
        checkpoint_dict = {"vit_b":str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           "vit_l":str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           'vit_h':str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           'vit_h_MultiSpectral':str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           'vit_l_MultiSpectral':str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           'vit_b_MultiSpectral':str(DEFAULT_MASKDECODER_B_CHECKPOINT),} # replace trained 
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False

        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280,"vit_h_MultiSpectral":1280,"vit_l_MultiSpectral":1024,"vit_b_MultiSpectral":768}
        vit_dim = vit_dim_dict[model_type]

        # self._class_logits = None
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        # self.num_mask_tokens = self.num_mask_tokens + 2
        
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

        self.edge_decoder_auge = EdgeDecoder_STN(args=None, classes=1, mode='train')
        # try to learn edge feature
        # edge conv
        # fusion module

        # TODO fixed instance token num  temporarily
        self.num_instance_tokens = num_instance_tokens
        self.instance_token = nn.Embedding(self.num_instance_tokens, transformer_dim)
        # self.pos_embed = nn.Parameter(torch.randn(self.num_instance_tokens, transformer_dim))
        self.num_mask_tokens = self.num_mask_tokens + 1 + self.num_instance_tokens # seg edge instance
        self.fusion = CrossAttentionFusion_v3(dim=transformer_dim)
        # instace head
        self.instance_mask_decoder = InstanceMaskDecoder_v3(token_dim=transformer_dim, feature_dim=transformer_dim // 8,if_tokenCluster=use_cluster_token)

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
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single mask.
        Returns:
          torch.Tensor: batched predicted hq masks
        """
        
        # vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        vit_features = [interm_embeddings[i].permute(0, 3, 1, 2) for i in range(3)]
        # compress_vit_features = [self.compress_vit_feat[i](vit_features[i]) for i in range(3)]
        
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
                image_embeddings_list=interm_embeddings,
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0),
            )
            if len(predict_masks)==2:
                mask, iou_pred = predict_masks
                class_logit = None
            elif len(predict_masks)==3:
                mask, iou_pred, class_logit = predict_masks
            elif len(predict_masks)==4:
                mask, iou_pred, class_logit, mask_instance_cluster = predict_masks

            masks.append(mask)
            iou_preds.append(iou_pred)
            if class_logit is not None:
                class_logits.append(class_logit)
            if mask_instance_cluster is not None:
                mask_instance_clusters.append(mask_instance_cluster)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)
        if len(class_logits)>0:
            class_logits = torch.cat(class_logits,0)
        else:
            class_logits = None
        if len(mask_instance_clusters)>0:
            mask_instance_clusters = torch.cat(mask_instance_clusters,0)
        else:
            mask_instance_clusters = None
        
        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1,self.num_mask_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-self.num_instance_tokens-2, self.num_mask_tokens-self.num_instance_tokens-1), :, :]
        masks_edge = masks[:,slice(self.num_mask_tokens-self.num_instance_tokens-1, self.num_mask_tokens-self.num_instance_tokens), :, :]
        masks_instance = masks[:,slice(self.num_mask_tokens-self.num_instance_tokens, self.num_mask_tokens), :, :]

        if class_logits is not None:
            self._class_logits = class_logits
        else:
            self._class_logits = None
        if mask_instance_clusters is not None:
            self._mask_instance_clusters = mask_instance_clusters
        else:
            self._mask_instance_clusters = None
        
        if hq_token_only:
            return masks_hq , masks_edge , masks_instance
        else:
            return masks_sam, masks_hq , masks_edge ,masks_instance

    def predict_masks( # 在SAM解码器基础上加了一个HQFeature
        self,
        image_embeddings: torch.Tensor,
        image_embeddings_list: List[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        B = sparse_prompt_embeddings.size(0)

        instance_tokens = self.instance_token.weight
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight, self.edge_token.weight,instance_tokens], dim=0)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight,instance_tokens], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src) 
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature

        instance_feature = upscaled_embedding_ours

        # 存储了学习到的Token 后续在这里进行token之间的融合（可以参考coser做特征之间的融合，然后去和instance标签做loss）
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            elif i == 4:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        seg_token,instance_token = mask_tokens_out[:, 4,:] ,mask_tokens_out[:,4 : 4+self.num_instance_tokens ,:]
        fusion_instance_token  = self.fusion(instance_token,seg_token.unsqueeze(1))

        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:5] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        
        masks_edge = self.edge_decoder_auge(image_embeddings,[image_embeddings_list[0]],upscaled_embedding_ours)

        masks_instance = self.instance_mask_decoder(fusion_instance_token,instance_feature)
        if len(masks_instance)==2:
            class_logits, masks_instance = masks_instance
        elif len(masks_instance)==3:
            class_logits, masks_instance, mask_instance_cluster = masks_instance
        else:
            class_logits = None
            mask_instance_cluster=None
        masks = torch.cat([masks_sam,masks_ours,masks_edge,masks_instance],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred, class_logits, mask_instance_cluster
