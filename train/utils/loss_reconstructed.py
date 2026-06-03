import torch
import torch.nn.functional as F
import torch.nn as nn

def ssim_loss(pred, target, window_size=11, size_average=True, C1=0.01**2, C2=0.03**2):
    """
    pred, target: [B, C, H, W]
    返回值: scalar 或 [B] 取决于 size_average
    """
    # 计算均值
    mu1 = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu1_mu2

    # SSIM 计算公式
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.view(ssim_map.size(0), -1).mean(dim=1)

def reconstruction_loss(pred, target, alpha=1.0, beta=0.5):
    """
    混合 L2 和 SSIM 作为压缩重建损失
    pred, target: [B, C, H, W]
    """
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    l2 = F.mse_loss(pred, target)
    # ssim = ssim_loss(pred, target)
    return alpha * l2
    # return alpha * l2 + beta * ssim

# 参考 Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models
import torch
import torch.nn.functional as F

def mdms_loss(z, f, margin=0.1):
    """
    Multi-Dimensional Mutual Similarity loss
    
    z: 模型压缩输出的特征 [B, C, H, W]
    f: SAM 输出的特征         [B, C, H, W]
    """
    B, C, H, W = z.shape

    # Step 1: Flatten -> [B, N, D] 其中 N=H*W, D=C
    z = z.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
    f = f.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

    # Step 2: Normalize
    z = F.normalize(z, dim=-1)  # 每个 token 单位化
    f = F.normalize(f, dim=-1)

    # Step 3: Cosine 相似度矩阵
    sim_z = torch.matmul(z, z.transpose(1, 2))  # [B, N, N]
    sim_f = torch.matmul(f, f.transpose(1, 2))

    # Step 4: 相对相似度差异
    diff = torch.abs(sim_z - sim_f)

    # Step 5: ReLU 蒸馏损失
    loss = F.relu(diff - margin)  # 允许 margin 内差异存在

    return loss.mean()
