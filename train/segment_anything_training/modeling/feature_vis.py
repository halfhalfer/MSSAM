import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
#************* 定量计算特征的类内紧凑度 和类间距离*************
def save_excel_imbedding_metrics(early_metrics, late_metrics, save_path="embedding_metrics.xlsx"):
    with pd.ExcelWriter(save_path) as writer:
        df_early = pd.DataFrame(early_metrics)
        df_late = pd.DataFrame(late_metrics)
        df_early.to_excel(writer, index=False, sheet_name='EarlyFeatures')
        df_late.to_excel(writer, index=False, sheet_name='LateFeatures')
    print(f"Saved to {save_path}")
def extract_pixel_embeddings(feat, mask):
    """
    feat: (B, D, H, W)
    mask: (B, H, W)
    返回：
      all_feats: (N, D)
      all_labels: (N,)
    """
    B, D, H, W = feat.shape
    # 展开
    feat = feat.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
    mask = mask.reshape(-1)  # (B*H*W,)

    # 过滤掉 ignore label（如 255）或背景（如 0），按需调整
    valid = mask >= 0
    all_feats = feat[valid]
    all_labels = mask[valid]
    return all_feats, all_labels
from collections import defaultdict

def compute_intra_class_distance(embeddings, labels):
    """
    embeddings: (N, D)
    labels: (N,)
    返回：
        intra_mean: float
        per_class: dict[class_id -> intra_dist]
    """
    class_feats = defaultdict(list)
    for feat, label in zip(embeddings, labels):
        class_feats[int(label)].append(feat.cpu().numpy())

    intra_dists = {}
    for c, feats in class_feats.items():
        feats = np.stack(feats)
        center = np.mean(feats, axis=0, keepdims=True)
        dists = np.linalg.norm(feats - center, axis=1)
        intra_dists[c] = dists.mean()

    intra_mean = np.mean(list(intra_dists.values()))
    return intra_mean, intra_dists

def compute_inter_class_distance(embeddings, labels):
    class_feats = defaultdict(list)
    for feat, label in zip(embeddings, labels):
        class_feats[int(label)].append(feat.cpu().numpy())

    centers = []
    for c in sorted(class_feats.keys()):
        feats = np.stack(class_feats[c])
        centers.append(np.mean(feats, axis=0))
    centers = np.stack(centers)

    C = len(centers)
    inter_dists = []
    for i in range(C):
        for j in range(i+1, C):
            d = np.linalg.norm(centers[i] - centers[j])
            inter_dists.append(d)
    inter_mean = np.mean(inter_dists)
    return inter_mean

def separation_ratio(inter, intra):
    return inter / (intra + 1e-6)  # 防止除零

from sklearn.metrics import silhouette_score

def compute_silhouette(embeddings, labels):
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return silhouette_score(embeddings_np, labels_np)

def compute_feature_structure_metrics(feat: torch.Tensor, mask: torch.Tensor, filename: str):
    """
    feat: (1, D, H, W) - 特征张量
    mask: (1, H, W) - 标签
    filename: 当前图像的文件名
    返回：一个 dict（包含五项）
    """
    # Flatten
    B, D, H, W = feat.shape
    feat_copy = feat.clone()
    
    feat = feat.permute(0, 2, 3, 1).reshape(-1, D)     # (N, D)
    pre_mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
    mask_copy = pre_mask_resized.clone()
    mask = pre_mask_resized.view(-1)                               # (N,)

    # 过滤掉 ignore 或背景，如 255 / 0（你按需调整）
    valid = mask >= 0
    feat = feat[valid]
    label = mask[valid]

    # 类内距离
    class_feats = defaultdict(list)
    for f, l in zip(feat, label):
        class_feats[int(l.item())].append(f.cpu().numpy())

    intra_dists = {}
    centers = {}
    for c, vecs in class_feats.items():
        vecs = np.stack(vecs)
        center = np.mean(vecs, axis=0, keepdims=True)
        centers[c] = center.squeeze()
        dists = np.linalg.norm(vecs - center, axis=1)
        intra_dists[c] = dists.mean()
    intra_mean = np.mean(list(intra_dists.values()))

    # 类间距离
    center_list = list(centers.values())
    inter_dists = []
    for i in range(len(center_list)):
        for j in range(i + 1, len(center_list)):
            inter_dists.append(np.linalg.norm(center_list[i] - center_list[j]))
    inter_mean = np.mean(inter_dists)

    # silhouette score
    try:
        sil_score = silhouette_score(feat.cpu().numpy(), label.cpu().numpy())
    except Exception:
        sil_score = -1  # too few classes or samples
    
    # spectral clustering
    significant_counts, eigvals_all = compute_significant_eigenvalues_with_mask(feat_copy, mask_copy.squeeze(1))
    return {
        "Filename": filename,
        "Intra-class distance": intra_mean,
        "Inter-class distance": inter_mean,
        "Separation ratio": inter_mean / (intra_mean + 1e-6),
        "Silhouette score": sil_score,
        "Significant eigenvalues": significant_counts[0],
    }

def visualize_instance_tokens(instance_tokens, output_dir, name_prefix="default"):
    """
    instance_tokens: (B, N, D)
    可视化每个 batch 中的 token 经过 t-SNE 降维后的分布
    """
    os.makedirs(os.path.join(output_dir, "instance_tokens"), exist_ok=True)
    B, N, D = instance_tokens.shape

    for b in range(B):
        tokens = instance_tokens[b].detach().cpu().numpy()  # (N, D)
        if N < 2:
            continue
        tsne = TSNE(n_components=2, perplexity=min(30, N - 1), random_state=42)
        tokens_2d = tsne.fit_transform(tokens)

        plt.figure(figsize=(6, 6))
        plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], c=np.arange(tokens_2d.shape[0]), cmap='tab20', s=20)
        plt.title(f"t-SNE of Instance Tokens - {name_prefix}_B{b}")
        plt.savefig(os.path.join(output_dir, "instance_tokens", f"{name_prefix}_B{b}.png"))
        plt.close()


def visualize_feature_map(feature_map, output_dir, name_prefix="default", max_channels=6):
    """
    feature_map: (B, C, H, W)
    可视化每个 batch 的 feature map 中选定通道
    """
    os.makedirs(os.path.join(output_dir, "feature_map"), exist_ok=True)
    B, C, H, W = feature_map.shape
    feature_map = feature_map.detach().cpu()

    for b in range(B):
        fmap = feature_map[b]  # (C, H, W)
        channel_indices = torch.linspace(0, C - 1, steps=min(max_channels, C)).long()
        for i in channel_indices:
            ch_map = fmap[i]
            ch_map = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min() + 1e-6)
            plt.imshow(ch_map.numpy(), cmap='viridis')
            plt.title(f"Feature Map B{b} C{i.item()}")
            plt.axis('off')
            fname = f"{name_prefix}_B{b}_C{i.item()}.png"
            plt.savefig(os.path.join(output_dir, "feature_map", fname))
            plt.close()

def visualize_feature_energy(feature_map, output_dir, name_prefix="default"):
    """
    将多通道特征整合为单张响应强度图 (L2 norm)
    feature_map: [B, C, H, W]
    """
    os.makedirs(os.path.join(output_dir, "feature_map_energy"), exist_ok=True)
    B, C, H, W = feature_map.shape
    feature_map = feature_map.detach().cpu()

    for b in range(B):
        fmap = feature_map[b]  # (C, H, W)
        energy_map = torch.norm(fmap, dim=0)  # L2范数 [H, W]
        energy_map = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min() + 1e-6)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(energy_map.numpy(), cmap='magma')
        plt.axis('off')
        plt.title(f"Feature Energy Map - B{b}")
        plt.savefig(os.path.join(output_dir, "feature_map_energy", f"{name_prefix}_B{b}_energy.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

def visualize_token_mask_feature(
    instance_tokens, feature_map, mask_logits,
    output_dir, name_prefix="sample", selected_token_idx=None, max_channels=3
):
    """
    可视化指定 token 的 mask + token 的 t-SNE 投影 + feature map（某几个通道）

    参数：
        instance_tokens: (B, N, D)
        feature_map: (B, C, H, W)
        mask_logits: (B, N, H, W)
        output_dir: 保存目录
        name_prefix: 文件名前缀
        selected_token_idx: 可选，指定 token index；若为 None，默认用第一个 token
        max_channels: 可视化最多几个 feature map 通道（默认3）
    """

    os.makedirs(output_dir, exist_ok=True)
    B, N, D = instance_tokens.shape
    _, C, H, W = feature_map.shape

    for b in range(B):
        token_idx = selected_token_idx if selected_token_idx is not None else 0
        token_vec = instance_tokens[b, token_idx].detach().cpu().numpy()  # (D,)
        mask = mask_logits[b, token_idx].detach().cpu().numpy()  # (H, W)
        fmap = feature_map[b].detach().cpu()  # (C, H, W)

        # 1. mask 可视化
        mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)

        # 2. t-SNE 可视化所有 token
        token_proj = instance_tokens[b].detach().cpu().numpy()  # (N, D)
        tsne = TSNE(n_components=2, perplexity=min(30, N - 1), init='pca', random_state=42)
        tokens_2d = tsne.fit_transform(token_proj)

        # 3. feature map 可视化
        ch_indices = torch.linspace(0, C - 1, steps=min(max_channels, C)).long()
        fmap_imgs = [
            (fmap[c] - fmap[c].min()) / (fmap[c].max() - fmap[c].min() + 1e-6)
            for c in ch_indices
        ]

        # 4. 绘图
        fig, axes = plt.subplots(1, 2 + len(fmap_imgs), figsize=(4 * (2 + len(fmap_imgs)), 4))

        # mask
        axes[0].imshow(mask_norm, cmap='gray')
        axes[0].set_title(f"Mask B{b} N{token_idx}")
        axes[0].axis('off')

        # token t-SNE
        axes[1].scatter(tokens_2d[:, 0], tokens_2d[:, 1], c=np.arange(tokens_2d.shape[0]), cmap='tab20', s=20)
        axes[1].scatter(tokens_2d[token_idx, 0], tokens_2d[token_idx, 1], c='red', s=40)
        axes[1].set_title("t-SNE of Tokens")

        # feature maps
        for i, fmap_img in enumerate(fmap_imgs):
            axes[2 + i].imshow(fmap_img.numpy(), cmap='viridis')
            axes[2 + i].set_title(f"FeatureMap C{ch_indices[i].item()}")
            axes[2 + i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{name_prefix}_B{b}_N{token_idx}.png")
        plt.savefig(save_path)
        plt.close()

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torchvision.transforms.functional as TF
def compute_significant_eigenvalues(feat, threshold=0.001):
    """
    feat: [B, H, W, C] 输入特征
    threshold: 特征值显著性阈值（归一化后大于该值视为显著）
    return: 每个样本中显著 eigenvalue 的数量 [B]
    """
    B, H, W, C = feat.shape
    feat = feat.view(B, H * W, C)
    S = torch.bmm(feat, feat.transpose(1, 2))  # [B, N, N]
    
    eigvals = []
    for b in range(B):
        S_b = S[b]
        eigval_b = torch.linalg.eigvalsh(S_b)  # [N]
        eigvals.append(eigval_b)
    eigvals = torch.stack(eigvals, dim=0)  # [B, N]
    
    eigvals_norm = eigvals / eigvals.sum(dim=1, keepdim=True)
    significant_counts = (eigvals_norm > threshold).sum(dim=1)
    return significant_counts, eigvals

def compute_significant_eigenvalues_with_mask(feat, mask, threshold=0.001):
    """
    feat: [B, C, H, W]   - 特征张量
    mask: [B, H, W]      - 掩膜，只对mask==1区域计算
    threshold: float     - 特征值显著性阈值

    return:
        significant_counts: List[int] 每个样本显著 eigenvalue 的数量
        eigvals_all: List[Tensor] 每个样本的 eigenvalues 向量
    """
    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]

    significant_counts = []
    eigvals_all = []

    for b in range(B):
        # 获取 mask == 1 的位置索引
        mask_b = mask[b]  # [H, W]
        feat_b = feat[b]  # [H, W, C]
        valid_indices = mask_b.bool()

        if valid_indices.sum() < 2:
            # 若有效点不足2个，不进行谱分析，返回0
            significant_counts.append(0)
            eigvals_all.append(torch.zeros(1, device=feat.device))
            continue

        # 取出mask==1的特征： [N, C]
        feat_valid = feat_b[valid_indices]  # [N, C]
        feat_valid = F.normalize(feat_valid, dim=1)  # 可选，归一化更稳定

        # 构建相似度矩阵 S = ZZ^T: [N, N]
        S = feat_valid @ feat_valid.T  # [N, N]

        # 计算特征值
        eigval = torch.linalg.eigvalsh(S)  # [N]
        eigval_norm = eigval / eigval.sum()

        count = (eigval_norm > threshold).sum().item()

        significant_counts.append(count)
        eigvals_all.append(eigval)

    return significant_counts, eigvals_all
def mask_logits_to_instance_id(pred_mask_logits, threshold=0.5,if_logits=True):
    """
    将 (B, N, H, W) 的 mask logits 映射为 (B, H, W) 的 instance ID 图。
    
    参数：
        pred_mask_logits: 预测输出的 mask logits，(B, N, H, W)
        threshold: 过滤掉低置信度 mask 的阈值
    返回：
        pred_instance_ids: 每像素的 instance ID 图，(B, H, W)
    """
    B, N, H, W = pred_mask_logits.shape
    if if_logits:
        pred_probs = torch.sigmoid(pred_mask_logits)  # (B, N, H, W)
    pred_probs = pred_mask_logits
    # 初始化输出 (B, H, W)，全为 0（背景）
    pred_instance_ids = torch.zeros((B, H, W), dtype=torch.long, device=pred_probs.device)

    for b in range(B):
        instance_counter = 1  # 实例 ID 从 1 开始
        for n in range(N):
            mask = pred_probs[b, n] > threshold  # (H, W) 的二值 mask
            if mask.sum() == 0:
                continue  # 跳过空 mask
            # 找到还没被标注的区域
            unassigned = pred_instance_ids[b] == 0
            mask = mask & unassigned
            if mask.sum() == 0:
                continue  # 没有新区域
            pred_instance_ids[b][mask] = instance_counter
            instance_counter += 1
    return pred_instance_ids

def visualize_encoder_features_withInstanceID(features, pre_mask, output_dir, name_list=[]):
    """
    features: (B, C, H, W) - encoder output
    pre_mask: (B, H, W) - 实例掩膜，0为背景，每个非0值表示一个实例
    可视化t-SNE降维图，颜色表示不同实例
    """
    os.makedirs(os.path.join(output_dir, "feature_tsne"), exist_ok=True)
    B, C, H, W = features.shape
    vis_batch = min(10, B)

    # pre_mask = pre_mask.unsqueeze(1).float()
    pre_mask_resized = F.interpolate(pre_mask, size=(H, W), mode='nearest')
    pre_mask_resized = pre_mask_resized.squeeze(1).long()  # (B, H, W)

    for b in range(vis_batch):
        feat = features[b]  # (C, H, W)
        mask = pre_mask_resized[b]  # (H, W)

        feat = feat.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
        mask_flat = mask.flatten().cpu().numpy()     # [H*W]

        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=min(30, feat.shape[0] - 1), random_state=42)
        feat_2d = tsne.fit_transform(feat.detach().cpu().numpy())  # [H*W, 2]

        name_prefix = name_list[b].split('.')[0] if len(name_list) > 0 else f"sample_{b}"

        # 原始 t-SNE 可视化
        plt.figure(figsize=(6, 6))
        plt.scatter(feat_2d[:, 0], feat_2d[:, 1], s=2, alpha=0.6)
        plt.title(f"t-SNE of Features - {name_prefix}")
        plt.savefig(os.path.join(output_dir, "feature_tsne", f"{name_prefix}_tsne.png"))
        plt.close()

        # 实例标签高亮可视化
        plt.figure(figsize=(6, 6))
        unique_ids = np.unique(mask_flat)
        colors = plt.cm.get_cmap('tab20', len(unique_ids))

        for i, inst_id in enumerate(unique_ids):
            idx = mask_flat == inst_id
            if inst_id == 0:
                plt.scatter(feat_2d[idx, 0], feat_2d[idx, 1], s=2, alpha=0.2, color='lightgray', label='Background')
            else:
                plt.scatter(feat_2d[idx, 0], feat_2d[idx, 1], s=6, alpha=0.8, color=colors(i), label=f'Instance {inst_id}')
        
        plt.title(f"t-SNE with Instance Highlight - {name_prefix}")
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_tsne", f"{name_prefix}_instance_highlight.png"))
        plt.close()

# def visualize_encoder_features(features, pre_mask, output_dir, name_list=[]):
#     """
#     features: (B, C, H, W) - encoder output
#     pre_mask: (B, H, W) - 模型预测的耕地掩膜（0/1）
#     生成两张图：原始 t-SNE & 耕地像素高亮 t-SNE
#     """
#     os.makedirs(os.path.join(output_dir, "feature_tsne"), exist_ok=True)
#     B, C, H, W = features.shape
#     vis_batch = min(10, B)
#     # upsampled_feats = F.interpolate(features, size=pre_mask.shape[1:], mode="bilinear", align_corners=False)

#     pre_mask = pre_mask.unsqueeze(1).float()  # 转 float 是因为 interpolate 要求浮点数输入

#     # resize 到 features 的空间尺寸
#     pre_mask_resized = F.interpolate(pre_mask, size=(H, W), mode='nearest')  # mask用nearest防止插值产生非整数标签

#     # squeeze 回 (B, H_feat, W_feat)
#     pre_mask_resized = pre_mask_resized.squeeze(1).long()
    
#     for b in range(vis_batch):
#         feat = features[b]  # (C, H, W)
        
#         feat = feat.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]

#         # t-SNE 降维
#         tsne = TSNE(n_components=2, perplexity=min(30, feat.shape[0] - 1), random_state=42)
#         feat_2d = tsne.fit_transform(feat.detach().cpu().numpy())  # [H*W, 2]

#         name_prefix = name_list[b].split('.')[0] if len(name_list) > 0 else "sample"

#         # 原始 t-SNE 可视化
#         plt.figure(figsize=(6, 6))
#         plt.scatter(feat_2d[:, 0], feat_2d[:, 1], s=2, alpha=0.6)
#         plt.title(f"t-SNE of Features - {name_prefix}_B{b}")
#         plt.savefig(os.path.join(output_dir, "feature_tsne", f"{name_prefix}_B{b}_tsne.png"))
#         plt.close()

#         # 耕地像素高亮
#         is_cropland = (pre_mask_resized[b] > 0).flatten().cpu().numpy()  # bool mask

#         plt.figure(figsize=(6, 6))
#         plt.scatter(feat_2d[:, 0], feat_2d[:, 1], s=2, alpha=0.2, label="Non-cropland")
#         plt.scatter(feat_2d[is_cropland, 0], feat_2d[is_cropland, 1], s=8, color='green', label="Cropland")
#         plt.title(f"t-SNE with Cropland Highlight - {name_prefix}_B{b}")
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, "feature_tsne", f"{name_prefix}_B{b}_cropland_highlight.png"))
#         plt.close()

def visualize_encoder_features(features, pre_mask, output_dir, name_list=[], no_axis_legend=True):
    """
    features: (B, C, H, W) - encoder output
    pre_mask: (B, H, W) - 模型预测的耕地掩膜（0/1）
    no_axis_legend: 是否隐藏坐标轴和图例
    生成两张图：原始 t-SNE & 耕地像素高亮 t-SNE
    """
    os.makedirs(os.path.join(output_dir, "feature_tsne"), exist_ok=True)
    B, C, H, W = features.shape
    vis_batch = min(10, B)
    pre_mask = pre_mask.unsqueeze(1).float()

    # resize 到 features 的空间尺寸
    pre_mask_resized = F.interpolate(pre_mask, size=(H, W), mode='nearest')
    pre_mask_resized = pre_mask_resized.squeeze(1).long()
    
    for b in range(vis_batch):
        feat = features[b]  # (C, H, W)
        feat = feat.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]

        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=min(30, feat.shape[0] - 1), random_state=42)
        feat_2d = tsne.fit_transform(feat.detach().cpu().numpy())  # [H*W, 2]

        name_prefix = name_list[b].split('.')[0] if len(name_list) > 0 else "sample"

        # 原始 t-SNE 可视化
        plt.figure(figsize=(6, 6))
        plt.scatter(feat_2d[:, 0], feat_2d[:, 1], s=2, alpha=0.6)
        plt.title(f"t-SNE of Features - {name_prefix}_B{b}" if not no_axis_legend else "")
        
        if no_axis_legend:
            plt.axis('off')  # 隐藏坐标轴
        plt.savefig(os.path.join(output_dir, "feature_tsne", f"{name_prefix}_B{b}_tsne.png"),
                   bbox_inches='tight', pad_inches=0)
        plt.close()

        # 耕地像素高亮
        is_cropland = (pre_mask_resized[b] > 0).flatten().cpu().numpy()  # bool mask

        plt.figure(figsize=(6, 6))
        plt.scatter(feat_2d[:, 0], feat_2d[:, 1], s=2, alpha=0.2, label="Non-cropland")
        plt.scatter(feat_2d[is_cropland, 0], feat_2d[is_cropland, 1], s=8, color='green', label="Cropland")
        plt.title(f"t-SNE with Cropland Highlight - {name_prefix}_B{b}" if not no_axis_legend else "")
        
        if no_axis_legend:
            plt.axis('off')  # 隐藏坐标轴
        else:
            plt.legend()  # 仅当需要时显示图例
            
        plt.savefig(os.path.join(output_dir, "feature_tsne", f"{name_prefix}_B{b}_cropland_highlight.png"),
                   bbox_inches='tight', pad_inches=0)
        plt.close()


def save_instance_masks(logits: torch.Tensor, output_dir: str , name: str):
    """
    Save each instance channel as a separate mask.
    Args:
        logits: Tensor of shape (b, n, h, w)
        output_dir: Directory to save masks
    """
    os.makedirs(output_dir, exist_ok=True)
    b, n, h, w = logits.shape
    masks = logits.sigmoid()
    for batch_idx in range(b):
        for class_idx in range(n):
            # Get the mask for current batch and class (shape: h, w)
            mask = masks[batch_idx, class_idx].detach().cpu().numpy()
            
            # Normalize to [0, 255] and save as PNG
            mask = ((mask>0.5)* 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_dir, f"name{name}batch{batch_idx}_class{class_idx}.png"),
                mask
            )

def visualize_token_similarity_matrix(instance_tokens, output_dir, name_prefix="default"):
    """
    对每个 batch 的 token 构造 token 相似度矩阵，并保存热力图

    Args:
        instance_tokens: (B, N, D) 的 Tensor，每个样本 N 个 token，维度 D
        output_dir: 输出路径
        name_prefix: 保存图像时的前缀名
    """
    os.makedirs(os.path.join(output_dir, "token_similarity"), exist_ok=True)
    B, N, D = instance_tokens.shape

    for b in range(B):
        tokens = instance_tokens[b].detach().cpu().numpy()  # shape: (N, D)

        # 计算 token 两两余弦相似度
        sim_matrix = cosine_similarity(tokens)  # shape: (N, N)

        # 可视化
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Token Similarity Matrix - {name_prefix}_B{b}")
        plt.xlabel("Token Index")
        plt.ylabel("Token Index")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "token_similarity", f"{name_prefix}_B{b}.png"))
        plt.close()

def visualize_token_attention_heatmap(
    instance_tokens,         # (B, N, D)
    feature_map,             # (B, C, H, W)
    original_images=None,    # (B, 3, H_img, W_img) or list of PIL/Image
    token_idx=0,
    batch_idx=0,
    output_dir="./output",
    name_prefix="token_attn",
    normalize_feature=True,
    overlay=True,
):
    """
    可视化某个 token 在图像上的 attention 响应热图（与 feature map 的点积）。

    参数：
    - instance_tokens: (B, N, D)，N个token
    - feature_map: (B, C, H, W)
    - original_images: 原始图像 (B, 3, H, W) 或 list of PIL.Image
    - token_idx: 要观察的 token 索引
    - batch_idx: 批次索引
    - output_dir: 输出目录
    - name_prefix: 文件名前缀
    - normalize_feature: 是否对 feature 归一化
    - overlay: 是否叠加在原图上
    """
    os.makedirs(os.path.join(output_dir, "attention_heatmaps"), exist_ok=True)
    
    token = instance_tokens[batch_idx, token_idx]  # (D,)
    fmap = feature_map[batch_idx]  # (C, H, W)
    C, H, W = fmap.shape
    
    # Flatten feature map to (H*W, C)
    fmap_flat = fmap.reshape(C, -1).T  # (H*W, C)
    token_vec = token.detach().cpu().numpy()
    fmap_vec = fmap_flat.detach().cpu().numpy()

    # Optional normalize
    if normalize_feature:
        token_vec = token_vec / (np.linalg.norm(token_vec) + 1e-6)
        fmap_vec = fmap_vec / (np.linalg.norm(fmap_vec, axis=1, keepdims=True) + 1e-6)

    # Dot product: (H*W,)
    attn_scores = np.dot(fmap_vec, token_vec)  
    attn_map = attn_scores.reshape(H, W)

    # Normalize to [0, 1]
    attn_map -= attn_map.min()
    attn_map /= (attn_map.max() + 1e-6)

    # Upsample heatmap to match original image size
    # attn_map_resized = cv2.resize(attn_map, (original_images[batch_idx].shape[2], original_images[batch_idx].shape[1]))

    # Save pure heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(attn_map, cmap='jet')
    plt.colorbar()
    plt.title(f"Attention Heatmap - B{batch_idx}_T{token_idx}")
    save_path = os.path.join(output_dir, "attention_heatmaps", f"{name_prefix}_B{batch_idx}_T{token_idx}_heatmap.png")
    plt.savefig(save_path)
    plt.close()

    # Overlay if available
    if original_images is not None and overlay:
        orig = original_images[batch_idx].detach().cpu()
        orig = TF.to_pil_image(orig)
        heatmap_img = (attn_map_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        orig_np = np.array(orig)

        overlayed = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, "attention_heatmaps", f"{name_prefix}_B{batch_idx}_T{token_idx}_overlay.png"), overlayed)


import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2

def visualize_token_behavior(token_idx, instance_tokens, feature_map, mask_logits, original_image, output_dir, name_prefix="default"):
    """
    可视化某个 token 的行为，包括其生成的 mask、注意力热图、t-SNE 分布、原图叠加。
    
    参数：
        token_idx (int): 要分析的 token 索引
        instance_tokens (Tensor): (B, N, D)
        feature_map (Tensor): (B, C, H, W)
        mask_logits (Tensor): (B, N, H, W)
        original_image (Tensor or ndarray): (B, 3, H, W) or (B, H, W, 3)
        output_dir (str): 输出文件夹
        name_prefix (str): 命名前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "token_mask"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "token_attention"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tsne"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)

    B, N, D = instance_tokens.shape
    _, C, H, W = feature_map.shape

    for b in range(B):
        token = instance_tokens[b, token_idx]  # (D,)
        token_np = token.detach().cpu().numpy()

        # ----------------------------
        # 1. mask visualization
        # ----------------------------
        mask = mask_logits[b, token_idx].detach().cpu().numpy()  # (H, W)
        plt.imsave(os.path.join(output_dir, "token_mask", f"{name_prefix}_B{b}_T{token_idx}_mask.png"), mask, cmap='gray')

        # ----------------------------
        # 2. attention heatmap (dot product)
        # ----------------------------
        feat = feature_map[b].detach()  # (C, H, W)
        feat_flat = feat.view(C, -1)  # (C, HW)
        token_proj = token.view(1, -1)  # (1, C)
        attention_map = torch.matmul(token_proj, feat_flat).view(H, W).cpu().numpy()

        attention_norm = (attention_map - attention_map.min()) / (attention_map.ptp() + 1e-6)
        plt.imsave(os.path.join(output_dir, "token_attention", f"{name_prefix}_B{b}_T{token_idx}_attention.png"), attention_norm, cmap='jet')

        # ----------------------------
        # 3. t-SNE of all tokens
        # ----------------------------
        tokens = instance_tokens[b].detach().cpu().numpy()  # (N, D)
        if N >= 2:
            tsne = TSNE(n_components=2, perplexity=min(30, N - 1), random_state=42)
            tokens_2d = tsne.fit_transform(tokens)
            plt.figure(figsize=(6, 6))
            plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1], c=np.arange(N), cmap='tab20', s=20)
            plt.scatter(tokens_2d[token_idx, 0], tokens_2d[token_idx, 1], c='red', s=60, label=f"token {token_idx}")
            plt.legend()
            plt.title(f"t-SNE of Instance Tokens - {name_prefix}_B{b}_T{token_idx}")
            plt.savefig(os.path.join(output_dir, "tsne", f"{name_prefix}_B{b}_T{token_idx}_tsne.png"))
            plt.close()

        # ----------------------------
        # 4. overlay original image with mask
        # ----------------------------
        img = original_image[b]  # (3, H, W) or (H, W, 3)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
            img = (img * 255).astype(np.uint8)
        else:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img

        mask_rgb = (mask > 0).astype(np.uint8) * 255
        mask_rgb = cv2.merge([mask_rgb]*3)

        overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, "overlay", f"{name_prefix}_B{b}_T{token_idx}_overlay.png"), overlay)
