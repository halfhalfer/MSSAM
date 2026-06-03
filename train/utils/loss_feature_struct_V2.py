# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn.functional as F

def compute_class_avg_features(flatten_features, one_hot_label, 
                               use_batch_class_center=False, 
                               last_class_center=None, 
                               last_class_center_decay=0.9):
    """
    PyTorch实现：CAR中的类中心计算
    
    参数:
        flatten_features: [N, C] 特征向量
        one_hot_label: [N, num_classes] one-hot标签
        use_batch_class_center: 是否跨batch求平均
        last_class_center: 历史类中心向量 [num_classes, C]（可选）
        last_class_center_decay: 滑动平均的衰减率

    返回:
        class_avg_features: [num_classes, C] 当前类中心
        updated_last_center: 若输入了历史类中心，则返回更新后的版本
    """
    N, C = flatten_features.shape
    num_classes = one_hot_label.shape[1]

    # 类别特征求和: [N, C] x [N, K] → [N, K, C]
    class_sum_features = torch.einsum('nc,nk->nkc', flatten_features, one_hot_label)
    class_sum_non_zero_map = one_hot_label  # [N, K]

    if use_batch_class_center:
        # 按类别累加 → [K, C]
        class_sum_features = class_sum_features.sum(dim=0)  # [K, C]
        class_sum_non_zero_map = class_sum_non_zero_map.sum(dim=0)  # [K]

        # 避免除以0
        class_avg_features = torch.where(
            class_sum_non_zero_map.unsqueeze(-1) > 0,
            class_sum_features / (class_sum_non_zero_map.unsqueeze(-1) + 1e-6),
            torch.zeros_like(class_sum_features)
        )

        if last_class_center is not None:
            batch_class_ignore_mask = (class_sum_non_zero_map != 0).int()  # [K]
            class_center_diff = class_avg_features - last_class_center  # [K, C]
            class_center_diff = class_center_diff * (1 - last_class_center_decay) * batch_class_ignore_mask.unsqueeze(-1)
            updated_last_center = last_class_center + class_center_diff
            class_avg_features = updated_last_center.float()
        else:
            updated_last_center = None
    else:
        # 按样本分别平均 → [N, K, C]
        class_avg_features = torch.where(
            class_sum_non_zero_map.unsqueeze(-1) > 0,
            class_sum_features / (class_sum_non_zero_map.unsqueeze(-1) + 1e-6),
            torch.zeros_like(class_sum_features)
        )
        updated_last_center = None

    return class_avg_features, updated_last_center

def get_intra_class_absolute_loss(x, avg_value, remove_max_value=False, not_ignore_spatial_mask=None):
    """
    类内绝对值差损失，用于拉近像素特征与类中心之间的距离
    
    参数：
        x: [B, HW, C] 特征向量
        avg_value: [B, HW, C] 类别中心广播到每个位置（one-hot * class_avg）
        remove_max_value: 是否移除最大差值（用于去除异常值）
        not_ignore_spatial_mask: [B, HW] 是否忽略某些位置（可选）

    返回：
        loss: 标量
    
    """
    # 停止对 avg_value 反向传播
    avg_value = avg_value.detach()

    # 差值绝对值 [B, HW, C]
    value_diff = torch.abs(avg_value - x)

    if not_ignore_spatial_mask is not None:
        value_diff = value_diff * not_ignore_spatial_mask.unsqueeze(-1).float()  # [B, HW, C]

    value_diff = value_diff.permute(0, 2, 1)  # [B, C, HW]

    if remove_max_value:
        # 移除每个通道中最后一个最大值位置（ASCENDING）
        value_diff, _ = torch.sort(value_diff, dim=-1)
        value_diff = value_diff[:, :, :-1]  # [B, C, HW-1]

    # square mean loss: 均值平方损失
    loss = torch.mean(value_diff ** 2)

    # 裁剪小于 1e-5 的 loss（防止梯度爆炸）
    loss = torch.clamp(loss, min=1e-5)

    return loss

def get_intra_class_absolute_loss_V2(x, avg_value, remove_max_value=False, not_ignore_spatial_mask=None):
    """
    类内绝对值差损失，用于拉近像素特征与类中心之间的距离
    
    参数：
        x: [B, HW, C] 特征向量
        avg_value: [B, HW, C] 类别中心广播到每个位置（one-hot * class_avg）
        remove_max_value: 是否移除最大差值（用于去除异常值）
        not_ignore_spatial_mask: [B, HW] 是否忽略某些位置（可选）

    返回：
        loss: 标量
    
    V2: 使用余弦相似度，特征向量方向一致即可，不需要模长也接近
    """
    avg_value = avg_value.detach()

    x_norm = F.normalize(x, dim=-1)  # [B, HW, C]
    avg_norm = F.normalize(avg_value, dim=-1)  # [B, HW, C]

    cos_sim = F.cosine_similarity(x_norm, avg_norm, dim=-1)  # [B, HW]
    loss = 1.0 - cos_sim  # 越接近方向一致，损失越小

    if not_ignore_spatial_mask is not None:
        loss = loss * not_ignore_spatial_mask.float()

    return torch.mean(loss)

def get_inter_class_c2c_loss(class_avg_features, margin=1.0):
    """
    class_avg_features: [num_classes, C]
    计算不同类别之间的距离，鼓励类中心分离
    
    """
    num_classes = class_avg_features.shape[0]
    if num_classes <= 1:
        return torch.tensor(0.0, device=class_avg_features.device)

    loss = 0.0
    count = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            center_i = class_avg_features[i]
            center_j = class_avg_features[j]
            dist = torch.norm(center_i - center_j, p=2)  # L2 距离
            # dist = 1 - F.cosine_similarity(center_i.unsqueeze(0), center_j.unsqueeze(0))
            loss += torch.relu(margin - dist)  # 距离太小则产生惩罚
            count += 1

    return loss / count

# def get_inter_class_c2p_loss(flatten_feature, class_avg_features, one_hot_label, margin=1.0):
#     """
#     Calculates an inter-class center-to-pixel (C2P) loss .

#     Args:
#         flatten_feature (torch.Tensor): Pixel features, shape [N, C].
#         class_avg_features (torch.Tensor): Average features for each class, shape [num_classes, C].
#         one_hot_label (torch.Tensor): One-hot encoded labels for each pixel, shape [N, num_classes].
#         margin (float): The margin for the loss calculation.

#     Returns:
#         torch.Tensor: The calculated inter-class C2P loss.
#     """
#     # Ensure class_avg_features does not receive gradients from this loss calculation
#     class_avg_features = class_avg_features.detach()

#     # (N, C) -> (N, 1, C)
#     expanded_feature = flatten_feature.unsqueeze(1) 
#     # (num_classes, C) -> (1, num_classes, C)
#     expanded_class_avg_features = class_avg_features.unsqueeze(0)

#     dist_sq = torch.sum((expanded_feature - expanded_class_avg_features)**2, dim=-1)

#     negative_class_mask = (1 - one_hot_label).bool()


#     dist_to_negative_classes = dist_sq.masked_fill(~negative_class_mask, float('inf')) 


#     min_dist_negative, _ = torch.min(dist_to_negative_classes, dim=1) # Shape [N]


#     dist_to_positive_class = torch.sum(dist_sq * one_hot_label, dim=1) # Shape [N]

    
#     loss = torch.relu(dist_to_positive_class - min_dist_negative + margin)

#     # Average the loss over the batch
#     return torch.mean(loss)

def get_inter_class_c2p_loss_V2(flatten_feature, class_avg_features, one_hot_label, margin=0.25):
    """
    flatten_feature: [N, C]
    class_avg_features: [num_classes, C]
    one_hot_label: [N, num_classes]
    """
    N, C = flatten_feature.shape
    num_classes = class_avg_features.shape[0]

    class_avg_features = class_avg_features.permute(1, 0)  # [C, num_classes]
    energy = torch.matmul(flatten_feature, class_avg_features)  # [N, num_classes]

    self_energy = class_avg_features * class_avg_features
    self_energy = torch.sum(self_energy, dim=0, keepdim=True)  # [1, num_classes]

    false_label  = (1 - one_hot_label)

    energy *= false_label*energy
    energy += self_energy*one_hot_label

    energy_scale = torch.sqrt(torch.tensor(flatten_feature.shape[-1], dtype=flatten_feature.dtype, device=flatten_feature.device))
    energy = energy / energy_scale
    inter_c2p_relation = F.softmax(energy, dim=-1)

    threshold = margin / (num_classes - 1)

    other_c2p_relation = inter_c2p_relation * false_label  # [N, HW, num_classes]
    other_c2p_relation = torch.where(
        other_c2p_relation > threshold,
        other_c2p_relation - threshold,
        torch.zeros_like(other_c2p_relation)
    )
    # 沿着 class 维度求和，得到 [N, HW]
    other_c2p_relation = other_c2p_relation.sum(dim=-1)
    eps = torch.finfo(other_c2p_relation.dtype).eps
    other_c2p_relation = torch.clamp(other_c2p_relation, min=eps, max=1.0 - eps)
    loss = other_c2p_relation
    loss = torch.mean(loss ** 2)
    return loss
    # other_c2p_relation = inter_c2p_relation * other_label_mask  # [N, HW, class]
    # other_c2p_relation = tf.where(other_c2p_relation > threshold, other_c2p_relation - threshold, 0)
    # other_c2p_relation = tf.reduce_sum(other_c2p_relation, axis=-1)  # [N, HW]

    # other_c2p_relation = tf.clip_by_value(other_c2p_relation, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # loss = other_c2p_relation
    # loss = square_mean_loss(loss)

def get_inter_class_c2p_loss(flatten_feature, class_avg_features, one_hot_label, margin=1.0):
    """
    flatten_feature: [N, C]
    class_avg_features: [num_classes, C]
    one_hot_label: [N, num_classes]
    """
    N, C = flatten_feature.shape
    num_classes = class_avg_features.shape[0]

    # 每个像素的 ground-truth 类中心
    gt_center = torch.matmul(one_hot_label, class_avg_features)  # [N, C]

    total_loss = 0.0
    count = 0
    for class_idx in range(num_classes):
        center = class_avg_features[class_idx]  # [C]
        # 找出不属于该类的像素
        mask = one_hot_label[:, class_idx] == 0  # [N]
        if mask.sum() == 0:
            continue

        dist = torch.norm(flatten_feature[mask] - center, p=2, dim=1)  # [M]
        total_loss += torch.sum(torch.relu(margin - dist))
        count += mask.sum()

    if count == 0:
        return torch.tensor(0.0, device=flatten_feature.device)
    else:
        return total_loss / count
def compute_intra_inter_class_losses(encoder_embedding, semantic_mask, num_classes=2, last_class_center=None, use_batch_class_center=True):
    """
    参数：
        encoder_embedding: [B, C, H, W] - 编码器特征图
        semantic_mask: [B, 1, H0, W0] - 语义分割标签（未下采样前）
        num_classes: 类别数
        use_batch_class_center: 是否使用跨 batch 类中心统计（需配合外部模块管理）

    返回：
        loss_intral: 类内损失
        loss_c2c: 类间中心间距损失
        loss_c2p: 类中心与像素之间的区分损失
        class_avg_features: 平均类特征
        updated_last_center: 更新后的中心（如果内部支持）
    """
    B, C, H, W = encoder_embedding.shape

    # 1. 语义标签下采样到与特征图相同尺寸
    semantic_mask = F.interpolate(semantic_mask, size=(H, W), mode='nearest')

    # 2. flatten feature: [B, C, H, W] -> [B, HW, C]
    flatten_feature = encoder_embedding.permute(0, 2, 3, 1).reshape(B, -1, C)

    # 3. one-hot label flatten: [B, 1, H, W] -> [B, HW, num_classes]
    one_hot_label = F.one_hot(semantic_mask.long().squeeze(1), num_classes=num_classes).float()
    one_hot_label_flatten = one_hot_label.reshape(B, -1, num_classes)

    # 4. compute class center
    class_avg_features, updated_last_center = compute_class_avg_features(
        flatten_feature.squeeze(0), 
        one_hot_label_flatten.squeeze(0), 
        use_batch_class_center=use_batch_class_center,
        last_class_center=last_class_center,
    )  # [num_classes, C]

    # 5. intra-class loss
    same_avg = torch.matmul(one_hot_label_flatten, class_avg_features)  # [B, HW, C]
    loss_intral = get_intra_class_absolute_loss(flatten_feature, same_avg)
    # loss_intral = get_intra_class_absolute_loss_V2(flatten_feature, same_avg)

    # 6. inter-class losses
    # loss_c2c = torch.tensor(0.0, device=encoder_embedding.device)
    # loss_c2p = torch.tensor(0.0, device=encoder_embedding.device)

    loss_c2c = get_inter_class_c2c_loss(class_avg_features)
    loss_c2p = get_inter_class_c2p_loss_V2(
        flatten_feature.squeeze(0), 
        class_avg_features, 
        one_hot_label_flatten.squeeze(0)
    )

    return loss_intral, loss_c2c, loss_c2p, class_avg_features, updated_last_center

if __name__ == '__main__':
    # Test
    encoder_embedding = torch.load('/home/huar/LM_SR/sam-hq/train/utils/6190_encoder_embedding.pth').to('cuda')
    # interm_embeddings = torch.load('/home/huar/LM_SR/sam-hq/train/utils/6190_interm_embeddings_0.pth').to('cuda')
    semantic_mask = torch.load('/home/huar/LM_SR/sam-hq/train/utils/6190_semantic_mask.pth').to('cuda')
    B, C, H, W = encoder_embedding.shape
    semantic_mask = F.interpolate(semantic_mask, size=(64, 64), mode='nearest')
    flatten_feature  = encoder_embedding.permute(0, 2, 3, 1).reshape(B, -1, C)
    one_hot_label = F.one_hot(semantic_mask.long().squeeze(1), num_classes=2).float()
    one_hot_label_flatten = one_hot_label.reshape(B, -1, 2)
    
    class_avg_features, updated_last_center = compute_class_avg_features(flatten_feature.squeeze(0), one_hot_label_flatten.squeeze(0), use_batch_class_center=True)
    
    same_avg = torch.matmul(one_hot_label_flatten, class_avg_features)
    loss_intral = get_intra_class_absolute_loss(flatten_feature,same_avg)
    loss_intral_2 = get_intra_class_absolute_loss_V2(flatten_feature,same_avg)
    loss_c2c = get_inter_class_c2c_loss(class_avg_features)
    loss_c2p = get_inter_class_c2p_loss(flatten_feature.squeeze(0), class_avg_features, one_hot_label_flatten.squeeze(0))
    loss_c2p_v2 = get_inter_class_c2p_loss_V2(flatten_feature.squeeze(0), class_avg_features, one_hot_label_flatten.squeeze(0))
    a = 1