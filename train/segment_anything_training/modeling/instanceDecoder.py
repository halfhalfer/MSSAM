import torch
from torch import nn
from torch.nn import functional as F
from .feature_vis import visualize_feature_map,visualize_instance_tokens,visualize_token_mask_feature,visualize_token_attention_heatmap,save_instance_masks,visualize_token_similarity_matrix
from typing import List, Tuple, Type

from .common import LayerNorm2d

class DynamicConvMaskHead(nn.Module):
    def __init__(self, token_dim, feature_dim, hidden_dim=64, num_layers=3, kernel_size=3):
        super().__init__()
        self.token_dim = token_dim
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.conv_weight_predictor = nn.Linear(token_dim, num_layers * feature_dim * feature_dim)
        self.norm = nn.GroupNorm(8, feature_dim)

        self.input_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU()
        )
        self.output_proj = nn.Conv2d(feature_dim, 1, kernel_size=1)
    def forward(self, image_feats, instance_tokens):
        """
        image_feats: (B, C, H, W)
        instance_tokens: (B, N, D)
        return: mask_logits (B, N, H, W)
        """
        B, N, D = instance_tokens.shape
        _, C, H, W = image_feats.shape
        image_feats = self.input_proj(image_feats)

        # Predict dynamic conv weights: [B, N, num_layers * C * C] -> [B*N, num_layers, C_out, C_in]
        conv_weights = self.conv_weight_predictor(instance_tokens) \
            .view(B * N, self.num_layers, C, C)

        # Expand image features to (B*N, C, H, W)
        x = image_feats.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)

        # Process per sample
        for i in range(self.num_layers):
            x_list = []
            w = conv_weights[:, i]  # (B*N, C_out, C_in)

            # Step 2: Reshape weights to group conv format
            w = w.contiguous().view(B * N * C, C, 1, 1)  # (B*N*C_out, C_in, 1, 1)

            # Step 3: Reshape x to match group convolution
            x = x.contiguous().view(1, B * N * C, H, W)  # (1, B*N*C_in, H, W)

            # Step 4: Apply group convolution
            x = F.conv2d(x, w, groups=B * N, padding=0)  # (1, B*N*C_out, H, W)

            # Step 5: Reshape back
            x = x.view(B * N, C, H, W)

            x = self.norm(x)
            x = F.relu(x)
        x = self.output_proj(x)
        mask_logits = x.view(B, N, H, W)  # (B, N, H, W)
        return mask_logits
class InstanceMaskDecoder(nn.Module):
    def __init__(self, token_dim, feature_dim, refine=True):
        super().__init__()
        self.query_proj = nn.Linear(token_dim, feature_dim)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        )
        self.refine = refine
        if self.refine:
            self.refine_head = nn.Sequential(
                nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, 1, 1)
            )
    def forward(self, instance_tokens, feature_map):
        """
        instance_tokens: (B, N, D)
        feature_map: (B, C, H, W)
        """
        B, N, D = instance_tokens.shape
        _, C, H, W = feature_map.shape

        # 1. query projection
        queries = self.query_proj(instance_tokens)  # (B, N, C)

        # 2. feature projection
        masks_feat = self.mask_proj(feature_map).view(B, C, -1)  # (B, C, HW)

        # 3. dot-product to get raw masks
        mask_logits = torch.einsum('bnc,bch->bnh', queries, masks_feat)  # (B, N, HW)
        mask_logits = mask_logits.view(B, N, H, W)  # (B, N, H, W)

        if self.refine:
            refined_masks = []
            for i in range(N):
                # Expand mask logits: (B, 1, H, W)
                mask_i = mask_logits[:, i:i+1, :, :]
                # Concatenate with feature map: (B, C+1, H, W)
                concat = torch.cat([feature_map, mask_i], dim=1)
                # Refine the mask
                refined = self.refine_head(concat)  # (B, 1, H, W)
                refined_masks.append(refined)
            # Stack over N instances
            mask_logits = torch.cat(refined_masks, dim=1)  # (B, N, H, W)

        return mask_logits
class InstanceMaskDecoder_v2(nn.Module):
    def __init__(self, token_dim, feature_dim, refine=True):
        super().__init__()
        self.refine = refine
        self.dynamic_head = DynamicConvMaskHead(
            token_dim=token_dim,
            feature_dim=feature_dim,
            num_layers=3
        )

        if self.refine:
            self.refine_head = nn.Sequential(
                nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, 1, 1)
            )

    def forward(self, instance_tokens, feature_map):
        """
        instance_tokens: (B, N, D)
        feature_map: (B, C, H, W)
        """
        mask_logits = self.dynamic_head(feature_map, instance_tokens)  # (B, N, H, W)

        if self.refine:
            refined_masks = []
            for i in range(mask_logits.shape[1]):
                mask_i = mask_logits[:, i:i+1, :, :]
                concat = torch.cat([feature_map, mask_i], dim=1)
                refined = self.refine_head(concat)
                refined_masks.append(refined)
            mask_logits = torch.cat(refined_masks, dim=1)

        return mask_logits

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# MaskFormer的Mask生成方案，使用MLP作为query_proj
class InstanceMaskDecoder_v3(nn.Module):
    def __init__(self, token_dim, feature_dim, refine=True, if_tokenCluster=False):
        super().__init__()
        self.query_proj = MLP(token_dim, token_dim, feature_dim, 3)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        )
        self.refine = refine
        if self.refine:
            self.refine_head = nn.Sequential(
                nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, 1, 1)
            )
        self.if_tokenCluster = if_tokenCluster
        if self.if_tokenCluster:
            self.token_cluster = nn.Linear(100, 1) # token num to 1
        self.num_class = 2 
        self.class_head = nn.Linear(token_dim, self.num_class)
        # self.mask_mlp = MLP(token_dim, token_dim, token_dim, 3)

    def forward(self, instance_tokens, feature_map):
        """
        instance_tokens: (B, N, D)
        feature_map: (B, C, H, W)
        """
        B, N, D = instance_tokens.shape
        _, C, H, W = feature_map.shape
        # visualize_instance_tokens()
        # 1. query projection
        queries = self.query_proj(instance_tokens)  # (B, N, C)

        # 2. feature projection
        masks_feat = self.mask_proj(feature_map).view(B, C, -1)  # (B, C, HW)
        class_logits = self.class_head(instance_tokens)
        # queries = self.mask_mlp(queries)
        # 3. dot-product to get raw masks
        mask_logits = torch.einsum('bnc,bch->bnh', queries, masks_feat)  # (B, N, HW)
        mask_logits = mask_logits.view(B, N, H, W)  # (B, N, H, W)

        if self.refine:
            refined_masks = []
            for i in range(N):
                # Expand mask logits: (B, 1, H, W)
                mask_i = mask_logits[:, i:i+1, :, :]
                # Concatenate with feature map: (B, C+1, H, W)
                concat = torch.cat([feature_map, mask_i], dim=1)
                # Refine the mask
                refined = self.refine_head(concat)  # (B, 1, H, W)
                refined_masks.append(refined)
            # Stack over N instances
            mask_logits = torch.cat(refined_masks, dim=1)  # (B, N, H, W)
        if self.if_tokenCluster:
            # token_cluster_logits = self.token_cluster(instance_tokens)
            queries = queries.transpose(1, 2)
            queries_cluster = self.token_cluster(queries)
            queries_cluster = queries_cluster.transpose(1, 2)
            mask_clusterlogits  = torch.einsum('bnc,bch->bnh', queries_cluster, masks_feat)  # (B, N, HW)
            mask_clusterlogits = mask_clusterlogits.view(B, 1, H, W)
        else:
            mask_clusterlogits = None
        
        return class_logits,mask_logits,mask_clusterlogits

# 多层特征融合
class InstanceMaskDecoder_v4(nn.Module):
    def __init__(self, token_dim, feature_dim, refine=True):
        super().__init__()
        self.query_proj = MLP(token_dim, token_dim, feature_dim, 3)
        self.mask_proj_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        ) for _ in range(4)])

        self.refine = refine
        if self.refine:
            self.refine_head = nn.Sequential(
                nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, 1, 1)
            )
        
        self.num_class = 2 
        self.class_head = nn.Linear(token_dim, self.num_class)
        # self.mask_mlp = MLP(token_dim, token_dim, token_dim, 3)

    def forward(self, instance_tokens, feature_map , multiscale_feature_map):
        """
        instance_tokens: (B, N, D)
        feature_map: (B, C, H, W)
        """
        B, N, D = instance_tokens.shape
        _, C, H, W = feature_map.shape
        
        multiscale_feature_map.append(feature_map)
        multi_scale_logits = []
        for scale_id, feat in enumerate(multiscale_feature_map):
            B, C, H, W = feat.shape
            feat_proj = self.mask_proj_list[scale_id](feat).view(B, C, -1)  # (B, C, HW)
            query_proj = self.query_proj(instance_tokens)  # (B, N, C)
            
            mask_logits = torch.einsum('bnc,bch->bnh', query_proj, feat_proj)  # (B, N, HW)
            mask_logits = mask_logits.view(B, N, H, W)
            mask_logits = F.interpolate(mask_logits, size=(multiscale_feature_map[0].shape[2:]), mode='bilinear', align_corners=False)
            multi_scale_logits.append(mask_logits)
        mask_logits = torch.stack(multi_scale_logits, dim=0).sum(dim=0) 
        class_logits = self.class_head(instance_tokens)
        
        if self.refine:
            refined_masks = []
            for i in range(N):
                # Expand mask logits: (B, 1, H, W)
                mask_i = mask_logits[:, i:i+1, :, :]
                # Concatenate with feature map: (B, C+1, H, W)
                concat = torch.cat([feature_map, mask_i], dim=1)
                # Refine the mask
                refined = self.refine_head(concat)  # (B, 1, H, W)
                refined_masks.append(refined)
            # Stack over N instances
            mask_logits = torch.cat(refined_masks, dim=1)  # (B, N, H, W)
        
        return class_logits,mask_logits

class InstanceMaskDecoder_v5(nn.Module):
    def __init__(self, token_dim, feature_dim, refine=True, if_tokenCluster=False , num_instance_tokens =100):
        super().__init__()
        self.query_proj = MLP(token_dim, token_dim, feature_dim, 3)
        # self.mask_split= InstanceAwareConv(in_ch=num_instance_tokens, expand_ratio=5)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        )
        self.num_class = 2 
        self.class_head = nn.Linear(token_dim, self.num_class)
        self.refine = refine
        if self.refine:
            self.refine_head = nn.Sequential(
                nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, 1, 1)
            )
    def forward(self, instance_tokens, feature_map):
        """
        instance_tokens: (B, N, D)
        feature_map: (B, C, H, W)
        """
        B, N, D = instance_tokens.shape
        _, C, H, W = feature_map.shape
        # 1. query projection
        queries = self.query_proj(instance_tokens)  # (B, N, C)
        queries_split =queries
        N = queries_split.shape[1]
        # 2. feature projection
        masks_feat = self.mask_proj(feature_map).view(B, C, -1)  # (B, C, HW)
        class_logits = self.class_head(instance_tokens)
        
        # 3. dot-product to get raw masks
        mask_logits = torch.einsum('bnc,bch->bnh', queries_split, masks_feat)  # (B, N, HW)
        mask_logits = mask_logits.view(B, N, H, W)  # (B, N, H, W)

        if self.refine:
            refined_masks = []
            for i in range(N):
                # Expand mask logits: (B, 1, H, W)
                mask_i = mask_logits[:, i:i+1, :, :]
                # Concatenate with feature map: (B, C+1, H, W)
                concat = torch.cat([feature_map, mask_i], dim=1)
                # Refine the mask
                refined = self.refine_head(concat)  # (B, 1, H, W)
                refined_masks.append(refined)
            # Stack over N instances
            mask_logits = torch.cat(refined_masks, dim=1) 
        return class_logits,mask_logits

class InstanceAwareConv(nn.Module):
    def __init__(self, in_ch=100, expand_ratio=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*expand_ratio, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch*expand_ratio, in_ch*expand_ratio, 1)
        )
        
        # # 空间注意力竞争
        # self.attention = nn.Sequential(
        #     nn.Conv2d(in_ch*expand_ratio, 1, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # x: [B,100,H,W]
        expanded = self.conv(x)  # [B,400,H,W]
        
        # 空间竞争权重
        # attn = self.attention(expanded)  # [B,1,H,W]
        
        # 通道加权竞争
        return expanded   # [B,400,H,W]
class LightweightAttentionRefiner(nn.Module):
    def __init__(self, feat_dim, attn_dim=32):
        super().__init__()
        self.query_proj = nn.Conv2d(1, attn_dim, 1)  # mask -> query
        self.key_proj   = nn.Conv2d(feat_dim, attn_dim, 1)
        self.value_proj = nn.Conv2d(feat_dim, 1, 1)  # 聚合成 refined mask
        self.softmax = nn.Softmax(dim=1)

    def forward(self, masks, features):
        """
        masks:     (B, N, H, W)
        features:  (B, C, H, W)
        return:    (B, N, H, W)
        """
        B, N, H, W = masks.shape
        _, C, _, _ = features.shape

        masks_ = masks.view(B * N, 1, H, W)         # (B*N, 1, H, W)
        features_ = features.repeat_interleave(N, dim=0)  # (B*N, C, H, W)

        Q = self.query_proj(masks_)                # (B*N, attn_dim, H, W)
        K = self.key_proj(features_)               # (B*N, attn_dim, H, W)

        attn_map = Q * K                           # (B*N, attn_dim, H, W)
        attn_map = attn_map.sum(dim=1, keepdim=True)  # (B*N, 1, H, W)
        attn_map = self.softmax(attn_map)          # soft attention map

        V = self.value_proj(features_)             # (B*N, 1, H, W)
        refined = attn_map * V                     # attention weighted

        return refined.view(B, N, H, W)            # reshape back
class InstanceMaskDecoder_v6(nn.Module):
    def __init__(self, token_dim, feature_dim, refine=False, if_tokenCluster=False , num_instance_tokens =100):
        super().__init__()
        self.query_proj = MLP(token_dim, token_dim, feature_dim, 3)
        self.mask_split= InstanceAwareConv(in_ch=num_instance_tokens, expand_ratio=5)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        )
        
        self.num_class = 2 
        self.class_head = nn.Linear(token_dim, self.num_class)

        self.refine = refine
        if self.refine:
            self.refine_head = nn.Sequential(
                nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, 1, 1)
            )
            self.refine_att = LightweightAttentionRefiner(feature_dim)
            # self.refine_head = nn.Sequential(
            #     nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1, groups=feature_dim + 1),  # depthwise
            #     nn.ReLU(),
            #     nn.Conv2d(feature_dim, 1, 1)  # pointwise
            # )
    def refine_masks_in_batches(self, mask_logits, feature_map, step=100):
        B, N, H, W = mask_logits.shape
        C = feature_map.shape[1]
        # refined_masks = []

        # for i in range(0, N, step):
        #     mask_i = mask_logits[:, i:i+step, :, :]  # (1, step, H, W)
        #     mask_i = mask_i.view(-1, 1, H, W)         # (step, 1, H, W)

        #     feat_i = feature_map.expand(step, -1, -1, -1)  # (step, C, H, W)

        #     concat = torch.cat([feat_i, mask_i], dim=1)    # (step, C+1, H, W)
        #     refined = self.refine_head(concat)             # (step, 1, H, W)
        #     refined_masks.append(refined)
        #     return torch.cat(refined_masks, dim=0).unsqueeze(0)
        # 拼接后 reshape 成 (1, N, 1, H, W)
        mask_logits_ = mask_logits.view(B*N, 1, H, W)
        feature_ = feature_map.repeat_interleave(N, dim=0)  # (B*N, C, H, W)

        concat = torch.cat([feature_, mask_logits_], dim=1)
        refined = self.refine_head(concat)  # (B*N, 1, H, W)
        mask_logits = refined.view(B, N, H, W)
        return 
    def forward(self, instance_tokens, feature_map):
        """
        instance_tokens: (B, N, D)
        feature_map: (B, C, H, W)
        """
        B, N, D = instance_tokens.shape
        _, C, H, W = feature_map.shape
        # 1. query projection
        queries = self.query_proj(instance_tokens)  # (B, N, C)
        queries_split =queries
        N = queries_split.shape[1]
        # 2. feature projection
        masks_feat = self.mask_proj(feature_map).view(B, C, -1)  # (B, C, HW)
        class_logits = self.class_head(instance_tokens)
        
        # 3. dot-product to get raw masks
        mask_logits = torch.einsum('bnc,bch->bnh', queries_split, masks_feat)  # (B, N, HW)
        mask_logits = mask_logits.view(B, N, H, W)  # (B, N, H, W)

        mask_logits = self.mask_split(mask_logits)

        if self.refine:
            refined_masks = self.refine_att(mask_logits, feature_map)
            mask_logits = refined_masks
        return class_logits,mask_logits
class InstanceMaskDecoder_v7(nn.Module):
    def __init__(self, in_channels=256, mid_channels=256, out_size=256):
        super(InstanceMaskDecoder_v7, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.ConvTranspose2d(mid_channels, 1, kernel_size=2, stride=2)  # upsample
        self.final_upsample = nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False)

    def forward(self, roi_feats):  # input: [B, N, C, H, W]
        B, N, C, H, W = roi_feats.size()
        x = roi_feats.view(B * N, C, H, W)
        x = self.conv_blocks(x)
        x = self.deconv(x)  # (B*N, 1, H*2, W*2)
        x = self.final_upsample(x)  # (B*N, 1, 256, 256)
        x = x.view(B, N, 256, 256)
        return x  # MaskLogits

