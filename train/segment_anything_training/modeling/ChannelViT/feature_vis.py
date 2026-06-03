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

def visualize_patch_attention_effect(
    feature_map,
    output_dir,
    name_prefix="default",
    patch_idx=0,
    use_sigmoid=False,
    q_proj=None,
    k_proj=None,
    v_proj=None,
    out_proj=None,
    norm_layer=None,
    hidden_dim=None
):
    """
    可视化指定 patch 的注意力效果，包括：
    1. 通道间注意力矩阵
    2. 前后特征图 (shortcut / attn_out)
    3. 特征变化量 (difference map)
    4. 特征能量变化图 (energy map)

    参数：
        feature_map: torch.Tensor
            输入特征，形状为 [B, E, Cin, Hp, Wp]
        output_dir: str
            图像保存路径
        name_prefix: str
            文件名前缀
        patch_idx: int
            可视化的 patch 索引（范围 0 ~ B*Hp*Wp-1）
        use_sigmoid: bool
            是否使用 Sigmoid 注意力权重
        q_proj, k_proj, v_proj, out_proj: torch.nn.Linear
            QKV 与输出映射层
        norm_layer: torch.nn.Module
            残差归一化层
        hidden_dim: int
            注意力缩放维度 (用于除以 sqrt(dim))

    保存结果：
        - {name_prefix}_attn_matrix.png
        - {name_prefix}_shortcut.png
        - {name_prefix}_attn_out.png
        - {name_prefix}_feature_change.png
        - {name_prefix}_energy_diff.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # ============ 准备输入 ============
    B, E, Cin, Hp, Wp = feature_map.shape
    x = feature_map.permute(0, 3, 4, 2, 1).contiguous()  # [B, Hp, Wp, Cin, E]
    x = x.view(B * Hp * Wp, Cin, E)
    shortcut = x

    # ============ 注意力计算 ============
    Q = q_proj(x)
    K = k_proj(x)
    V = v_proj(x)

    attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (hidden_dim ** 0.5)
    attn_weights = torch.sigmoid(attn_scores) if use_sigmoid else F.softmax(attn_scores, dim=-1)
    attn_out = torch.bmm(attn_weights, V)
    attn_out = out_proj(attn_out)
    attn_out = norm_layer(attn_out + shortcut)

    # ============ reshape 回到 patch 结构 ============
    shortcut_map = shortcut.norm(dim=-1).view(B, Hp, Wp, Cin).mean(dim=-1).detach().cpu().numpy()
    attn_map = attn_out.norm(dim=-1).view(B, Hp, Wp, Cin).mean(dim=-1).detach().cpu().numpy()
    diff_map = (attn_out - shortcut).norm(dim=-1).view(B, Hp, Wp, Cin).mean(dim=-1).detach().cpu().numpy()

    # ======================================================
    # 1️⃣ 通道间注意力矩阵
    # ======================================================
    attn_matrix = attn_weights[patch_idx].detach().cpu().numpy()  # [Cin, Cin]
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_matrix, cmap='viridis')
    plt.title(f"Channel Attention Matrix (patch={patch_idx})")
    plt.xlabel("Key channel")
    plt.ylabel("Query channel")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name_prefix}_attn_matrix.png"))
    plt.close()

    # ======================================================
    # 2️⃣ 变化前特征图 (shortcut)
    # ======================================================
    plt.figure(figsize=(5, 4))
    plt.imshow(shortcut_map[0], cmap='viridis')
    plt.title("Before Attention (shortcut)")
    plt.colorbar(label="Feature magnitude")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name_prefix}_shortcut.png"))
    plt.close()

    # ======================================================
    # 3️⃣ 变化后特征图 (attn_out)
    # ======================================================
    plt.figure(figsize=(5, 4))
    plt.imshow(attn_map[0], cmap='viridis')
    plt.title("After Attention (attn_out)")
    plt.colorbar(label="Feature magnitude")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name_prefix}_attn_out.png"))
    plt.close()

    # ======================================================
    # 4️⃣ 前后特征变化量 (difference)
    # ======================================================
    plt.figure(figsize=(5, 4))
    plt.imshow(diff_map[0], cmap='plasma')
    plt.title("Feature Change Map (attn_out - shortcut)")
    plt.colorbar(label="Δ Feature magnitude")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name_prefix}_feature_change.png"))
    plt.close()

    # ======================================================
    # 5️⃣ 特征能量变化 (energy map)
    # ======================================================
    energy_before = shortcut.norm(dim=(1, 2))
    energy_after = attn_out.norm(dim=(1, 2))
    energy_diff = (energy_after - energy_before).view(B, Hp, Wp).detach().cpu().numpy()

    plt.figure(figsize=(5, 4))
    plt.imshow(energy_diff[0], cmap='RdBu')
    plt.title("Energy Difference Map (After Attention)")
    plt.colorbar(label="Δ Feature Energy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name_prefix}_energy_diff.png"))
    plt.close()

    print(f"✅ Visualization saved in: {output_dir}")


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
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

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

from sklearn.metrics import mutual_info_score
# def compute_mutual_info(vectors):
#     """
#     Mutual Information Matrix
#     vectors: shape (Bands, Features)
#     """
#     n = vectors.shape[0]
#     mi_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             mi_matrix[i, j] = mutual_info_score(vectors[i], vectors[j])
#     return mi_matrix
from sklearn.feature_selection import mutual_info_regression

def compute_mutual_info_continuous(vectors, n_neighbors=3):
    """
    使用基于k近邻的互信息估计（适用于连续变量）
    """
    n = vectors.shape[0]
    mi_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # 使用 mutual_info_regression 估计连续变量的互信息
            mi_val = mutual_info_regression(vectors[i].reshape(-1, 1), 
                                          vectors[j], 
                                          n_neighbors=n_neighbors)
            mi_matrix[i, j] = mi_val[0]
    
    return mi_matrix

def visualize_band_correlation_v2(tensor, output_dir, name_prefix="default"):
    """
    计算并可视化波段之间的 Pearson, Cosine, MI 相关性
    tensor: torch.Tensor, shape (B, C, Bands, H, W)
    """
    os.makedirs(os.path.join(output_dir, "band_correlation"), exist_ok=True)

    B, C, Bands, H, W = tensor.shape

    methods = {
        "pearson": lambda v: np.corrcoef(v),
        "cosine": lambda v: cosine_similarity(v),
        "mi": lambda v: compute_mutual_info_continuous(v)
    }

    for b in range(B):
        data = tensor[b]  # shape: (C, Bands, H, W)

        # ✅ 如果空间分辨率大于 1*1，则在 H,W 上求平均
        if H > 1 or W > 1:
            data = data.mean(dim=[2, 3], keepdim=True)  # → (C, Bands, 1, 1)

        # ✅ 转换 shape 为 (Bands, C)
        band_vectors = data.permute(1, 0, 2, 3).reshape(Bands, -1).cpu().numpy()

        for method_name, compute_fn in methods.items():
            matrix = compute_fn(band_vectors)

            plt.figure(figsize=(6, 5))
            sns.heatmap(matrix,
                        cmap="coolwarm",
                        vmin=-1 if method_name == "pearson" else None,
                        vmax=1 if method_name in ["pearson", "cosine"] else None,
                        xticklabels=False, yticklabels=False,
                        cbar=True)

            plt.title("")
            plt.xlabel("Band Index")
            plt.ylabel("Band Index")

            plt.tight_layout()
            save_path = os.path.join(output_dir, "band_correlation",
                                     f"{name_prefix}_B{b}_{method_name}.png")
            plt.savefig(save_path)
            plt.close()

            print(f"✅ Saved: {save_path}")

def visualize_band_correlation_v3(tensor, output_dir, name_prefix="default"):
    """
    计算并可视化波段之间的 Pearson, Cosine, MI 相关性
    """
    os.makedirs(os.path.join(output_dir, "band_correlation"), exist_ok=True)
    B, C, Bands, H, W = tensor.shape

    # 定义更美观的颜色方案
    color_schemes = {
        "pearson": {
            "cmap": "RdYlBu_r",  # 红-黄-蓝，反转后蓝色表示正相关
            "vmin": -1,
            "vmax": 1,
            "center": 0
        },
        "cosine": {
            "cmap": "viridis",    # 紫色-绿色-黄色，现代感强
            "vmin": -1,
            "vmax": 1,
            "center": 0
        },
        "mi": {
            "cmap": "coolwarm",     
            "vmin": 0,
            "vmax": None,
            "center": None
        }
    }
    
    # 备选颜色方案（取消注释即可使用）
    alternative_colors = {
        "pearson": {"cmap": "coolwarm", "vmin": -1, "vmax": 1, "center": 0},
        "cosine": {"cmap": "Spectral", "vmin": -1, "vmax": 1, "center": 0},
        "mi": {"cmap": "YlOrRd", "vmin": 0, "vmax": None, "center": None}
    }

    methods = {
        "pearson": lambda v: np.corrcoef(v),
        "cosine": lambda v: cosine_similarity(v),
        "mi": lambda v: compute_mutual_info_continuous(v)
    }

    for b in range(B):
        data = tensor[b]  # shape: (C, Bands, H, W)

        # ✅ 如果空间分辨率大于 1*1，则在 H,W 上求平均
        if H > 1 or W > 1:
            data = data.mean(dim=[2, 3], keepdim=True)  # → (C, Bands, 1, 1)

        # ✅ 转换 shape 为 (Bands, C)
        band_vectors = data.permute(1, 0, 2, 3).reshape(Bands, -1).cpu().numpy()

        for method_name, compute_fn in methods.items():
            matrix = compute_fn(band_vectors)
            
            # 获取颜色配置
            color_config = color_schemes[method_name]
            
            plt.figure(figsize=(8, 6))
            
            # 创建波段标签
            band_labels = [f'{i+1}' for i in range(Bands)]
            
            # 如果波段数量太多，可以间隔显示标签
            if Bands > 15:
                # 每3个波段显示一个标签
                show_every = max(1, Bands // 10)
                tick_labels = [band_labels[i] if i % show_every == 0 else '' for i in range(Bands)]
            else:
                tick_labels = band_labels
            
            # 创建热图
            ax = sns.heatmap(matrix,
                        cmap=color_config["cmap"],
                        vmin=color_config["vmin"],
                        vmax=color_config["vmax"],
                        center=color_config["center"],
                        xticklabels=tick_labels,
                        yticklabels=tick_labels,
                        cbar=True,
                        square=True,  # 保持正方形
                        annot=False)  # 不显示数值，如果波段少可以设为True
            
            # 美化标签
            plt.xlabel("Band Index", fontsize=12, fontweight='bold')
            plt.ylabel("Band Index", fontsize=12, fontweight='bold')
            plt.title(f"Band Correlation - {method_name.upper()}", fontsize=14, fontweight='bold')
            
            # 旋转x轴标签以避免重叠
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # 调整布局
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, "band_correlation",
                                     f"{name_prefix}_B{b}_{method_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 提高分辨率
            plt.close()

            print(f"✅ Saved: {save_path}")
from matplotlib.colors import LinearSegmentedColormap

def visualize_band_correlation_v4(tensor, output_dir, name_prefix="default", min_value = 0, max_value = 1):
    """
    计算并可视化波段之间的 Pearson, Cosine, MI 相关性。
    对角线为灰色，其他元素根据非对角线最大最小值可视化。
    """
    os.makedirs(os.path.join(output_dir, "band_correlation"), exist_ok=True)
    B, C, Bands, H, W = tensor.shape

    color_schemes = {
        "pearson": {"cmap": "RdYlBu_r"},
        "cosine": {"cmap": "viridis"},
        "mi": {"cmap": "viridis"}
    }

    methods = {
        "pearson": lambda v: np.corrcoef(v),
        "cosine": lambda v: cosine_similarity(v),
        "mi": lambda v: compute_mutual_info_continuous(v)  # 假设你已经定义了
    }

    base_cmap = plt.get_cmap("viridis")
    # sample 256 个颜色
    colors = base_cmap(np.linspace(0.2, 1.0, 256))  # 0->0.4, 1->1
    new_cmap = LinearSegmentedColormap.from_list("viridis_shifted", colors)

    for b in range(B):
        data = tensor[b]  # shape: (C, Bands, H, W)

        if H > 1 or W > 1:
            data = data.mean(dim=[2, 3], keepdim=True)  # → (C, Bands, 1, 1)

        band_vectors = data.permute(1, 0, 2, 3).reshape(Bands, -1).cpu().numpy()

        for method_name, compute_fn in methods.items():
            matrix = compute_fn(band_vectors)

            # 构建 mask，把对角线置为 True
            mask_diag = np.eye(Bands, dtype=bool)

            # 计算非对角线的最大最小值
            non_diag_values = matrix[~mask_diag]
            vmin, vmax = non_diag_values.min(), non_diag_values.max()
            if method_name == 'mi':
                vmin = min_value
                vmax = max_value
            
            matrix_norm = matrix.copy()
            norm_non_diag = (non_diag_values - vmin) / (vmax - vmin)
            matrix_norm[~mask_diag] = norm_non_diag

            # 创建 heatmap 时，使用 mask 将对角线置灰
            plt.figure(figsize=(8, 6))
            cmap = plt.get_cmap(color_schemes[method_name]["cmap"])
            # 先绘制非对角线
            sns.heatmap(matrix_norm,
                        mask=mask_diag,
                        cmap=new_cmap,
                        vmin=0,
                        vmax=1,
                        xticklabels=True,
                        yticklabels=True,
                        square=True,
                        cbar=True,
                        annot=False)
            # 再覆盖对角线为灰色
            for i in range(Bands):
                plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='lightgray', edgecolor='lightgray'))

            band_labels = [f'{i+1}' for i in range(Bands)]
            plt.xticks(np.arange(Bands)+0.5, band_labels, rotation=45, ha='right')
            plt.yticks(np.arange(Bands)+0.5, band_labels, rotation=0)
            plt.xlabel("Band Index", fontsize=12, fontweight='bold')
            plt.ylabel("Band Index", fontsize=12, fontweight='bold')
            # plt.title(f"Band Correlation - {method_name.upper()}", fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = os.path.join(output_dir, "band_correlation",
                                     f"{name_prefix}_B{b}_{method_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {save_path}")

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
