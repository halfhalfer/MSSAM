import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
import utils.misc as misc
from scipy.optimize import linear_sum_assignment

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result

def cross_entropy_loss_RCF_cluster(prediction, labelef, label_prob=None):
    """
    输入:
        prediction: (B, N, H, W) 原始 logits（未 sigmoid）
        labelef:    (B, 1, H, W)，取值为0表示背景，1~4表示第1~4个mask
        label_prob: (B, N, H, W)，可选，对每通道加权
    输出:
        scalar，N个mask的平均加权BCE loss
    """
    B, C, H, W = prediction.shape

    total_loss = 0.0
    for c in range(1, C + 1):  # 注意标签中为 1~4
        # 构造当前通道的 binary mask
        label_c = (labelef == c).float()  # (B, H, W)
        pred_c = prediction[:, c - 1, :, :]  # (B, H, W)

        # 正负样本统计
        num_positive = torch.sum(label_c == 1).float()
        num_negative = torch.sum(label_c == 0).float()
        total = num_positive + num_negative + 1e-6

        # 构建加权掩膜
        weight_mask = label_c.clone()
        weight_mask[label_c == 1] = 1.0 * num_negative / total
        weight_mask[label_c == 0] = 1.1 * num_positive / total

        # 可选加入概率图
        if label_prob is not None:
            prob_c = label_prob[:, c - 1, :, :]
            weight_mask = weight_mask * torch.exp(prob_c)

        # 计算加权 BCE loss
        cost = F.binary_cross_entropy_with_logits(
            pred_c.unsqueeze(1), label_c, weight=weight_mask.detach(), reduction='sum'
        )
        normalizer = torch.sum(weight_mask.detach()) + 1e-6
        loss_c = cost / normalizer

        total_loss += loss_c

    avg_loss = total_loss / C
    return avg_loss
# def cross_entropy_loss_RCF_cluster(prediction, labelef, label_prob=None):
#     """
#     输入:
#         prediction: (B, N, H, W) 原始 logits（未sigmoid）
#         labelef:    (B, N, H, W) ground truth，取值为0或1
#         label_prob: (B, N, H, W) 可选，用于加权掩膜（概率图）
#     输出:
#         scalar：所有通道平均的 weighted BCE loss
#     """
#     B, C, H, W = prediction.shape

#     total_loss = 0.0
#     for c in range(C):
#         pred_c = prediction[:, c, :, :]       # (B, H, W)
#         label_c = labelef[:, c, :, :].long()  # (B, H, W)
#         mask_c = label_c.float()

#         num_positive = torch.sum(mask_c == 1).float()
#         num_negative = torch.sum(mask_c == 0).float()
#         total = num_positive + num_negative + 1e-6

#         weight_mask = mask_c.clone()
#         weight_mask[weight_mask == 1] = 1.0 * num_negative / total
#         weight_mask[weight_mask == 0] = 1.1 * num_positive / total

#         if label_prob is not None:
#             prob_c = label_prob[:, c, :, :]
#             weight_mask = weight_mask * torch.exp(prob_c)

#         cost = F.binary_cross_entropy_with_logits(
#             pred_c, mask_c, weight=weight_mask.detach(), reduction='sum'
#         )
#         normalizer = torch.sum(weight_mask.detach()) + 1e-6
#         loss_c = cost / normalizer

#         total_loss += loss_c

#     avg_loss = total_loss / C
#     return avg_loss
def cross_entropy_loss_RCF(prediction, labelef, label_prob=None): 
    # prediction: (B,1,H,W) raw logits (no sigmoid)
    # labelef: (B,1,H,W) ground truth labels (0 or 1)

    label = labelef.long()
    mask = label.float()

    # 获取正负样本数量
    num_positive = torch.sum(mask == 1).float()
    num_negative = torch.sum(mask == 0).float()

    # 避免除0
    total = num_positive + num_negative + 1e-6

    # 生成加权掩膜
    weight_mask = mask.clone()
    weight_mask[weight_mask == 1] = 1.0 * num_negative / total
    weight_mask[weight_mask == 0] = 1.1 * num_positive / total

    # 可选加入概率图
    if label_prob is not None:
        new_mask = weight_mask * torch.exp(label_prob)
    else:
        new_mask = weight_mask

    # 使用 binary_cross_entropy_with_logits（内部自带 sigmoid）
    cost = F.binary_cross_entropy_with_logits(
        prediction, labelef.float(), weight=new_mask.detach(), reduction='sum'
    )
    normalizer = torch.sum(new_mask.detach()) + 1e-6
    cost = cost / normalizer
    return cost



def cross_entropy_loss_RCF_wDice(prediction, labelef, label_prob=None, lambda_dice=0.5):
    """
    混合损失函数：加权交叉熵损失 + Dice损失
    
    参数:
    - prediction: (B, 1, H, W) 原始logits（未经过sigmoid）
    - labelef: (B, 1, H, W) 真实标签（0或1）
    - label_prob: 可选，概率图
    - lambda_dice: Dice损失的权重系数，默认0.5
    
    返回:
    - 混合损失值
    """
    label = labelef.long()
    mask = label.float()

    # 获取正负样本数量
    num_positive = torch.sum(mask == 1).float()
    num_negative = torch.sum(mask == 0).float()

    # 避免除0
    total = num_positive + num_negative + 1e-6

    # 生成加权掩膜
    weight_mask = mask.clone()
    weight_mask[weight_mask == 1] = 1.0 * num_negative / total
    weight_mask[weight_mask == 0] = 1.1 * num_positive / total

    # 可选加入概率图
    if label_prob is not None:
        new_mask = weight_mask * torch.exp(label_prob)
    else:
        new_mask = weight_mask

    # 1. 计算加权交叉熵损失
    ce_cost = F.binary_cross_entropy_with_logits(
        prediction, labelef.float(), weight=new_mask.detach(), reduction='sum'
    )
    normalizer = torch.sum(new_mask.detach()) + 1e-6
    ce_cost = ce_cost / normalizer

    # 2. 计算Dice损失
    # 将预测值通过sigmoid激活
    prediction_sigmoid = torch.sigmoid(prediction)
    
    # 计算交集和并集
    intersection = torch.sum(prediction_sigmoid * labelef.float())
    union = torch.sum(prediction_sigmoid) + torch.sum(labelef.float())
    
    # Dice系数 = 2 * |A∩B| / (|A| + |B|)
    dice_coeff = (2.0 * intersection + 1e-6) / (union + 1e-6)
    
    # Dice损失 = 1 - Dice系数
    dice_loss = 1.0 - dice_coeff

    # 3. 混合损失：加权交叉熵损失 + λ * Dice损失
    total_loss = ce_cost + lambda_dice * dice_loss
    
    # 可选：返回各项损失用于监控
    # return total_loss, ce_cost, dice_loss
    
    return total_loss





# copy from https://github.com/Bedrettin-Cetinkaya/RankED.git
class RankLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, nms_grad=1, delta=0.1,eps=1e-10,split = 4): 
 

        B,C,W,H = logits.size()
        logits = logits.view(B,-1)
        targets = targets.view(B,-1)
        #loss_weight = torch.exp(1-targets)
        classification_grads=torch.zeros(logits.shape).cuda()
        #Filter fg logits
        fg_labels = (targets > 0)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)
        #fg_targets = targets[fg_labels]

        if fg_num != 0:

            #Do not use bg with scores less than minimum fg logit
            #since changing its score does not have an effect on precision
            threshold_logit = torch.min(fg_logits)-delta
            #threshold_logit = 0.01
            #Get valid bg logits

            relevant_bg_labels=((torch.logical_not(fg_labels))&(logits>=threshold_logit))
            relevant_bg_logits=logits[relevant_bg_labels]
            relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
            ranking_error=torch.zeros(fg_num).cuda()
            fg_grad=torch.zeros(fg_num).cuda()
           
            fg_logits_sorted, sorted_indices =torch.sort(fg_logits)
            #Loops over each positive following the order

            start = 0
            end = fg_num // split
            for ii in range(split):
                fg_relations = fg_logits - fg_logits_sorted[start:end,None]
                fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
                
                bg_relations = relevant_bg_logits - fg_logits_sorted[start:end,None]
                bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)
                
                rank_pos=torch.sum(fg_relations, axis = 1)
                FP_num=torch.sum(bg_relations, axis = 1)
                
                rank=rank_pos+FP_num
                ranking_error[start:end] = FP_num/rank

                FP_num_check = FP_num > eps
                
                # 生成索引时指定设备
                indices = torch.arange(fg_grad.size(0), device=fg_grad.device)
                # 确保 sorted_indices 在相同设备
                sorted_indices = sorted_indices.to(fg_grad.device)
                # 提取目标索引
                selected_indices = indices[sorted_indices][start:end]
                # 确保右侧张量在相同设备
                ranking_error_sub = ranking_error[start:end].to(fg_grad.device)
                FP_num_check_sub = FP_num_check.long().to(fg_grad.device)  # 若需要类型转换
                # 执行原地操作
                fg_grad[selected_indices] -= ranking_error_sub * FP_num_check_sub
                # fg_grad[torch.arange(fg_grad.size(0))[sorted_indices][start:end]] -= ranking_error[start:end] * FP_num_check.long()
               
                relevant_bg_grad +=  torch.sum((bg_relations*(ranking_error[start:end]/(FP_num+eps))[:,None]),axis=0)
                
                start = end
                if ii == split -2:
                  end = fg_num
                else:
                  end *= 2
            #aLRP with grad formulation fg gradient
            classification_grads[fg_labels]= fg_grad #* loss_weight[fg_labels]
            classification_grads[relevant_bg_labels]= relevant_bg_grad 
            
            classification_grads /= fg_num 
            classification_grads = classification_grads.view(B,C,W,H)
            #classification_grads *= nms_grad
            ctx.save_for_backward(classification_grads)

        else:
            ranking_error = torch.zeros((2,1)).sum()
            classification_grads = classification_grads.view(B,C,W,H)
            ctx.save_for_backward(classification_grads)
        return ranking_error.mean()

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None
def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

def dice_loss_instance(pred, target, eps=1e-6):
    """Dice loss for binary masks"""
    pred = pred.sigmoid()
    inter = (pred * target).sum(dim=[1, 2])
    union = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2])
    dice = 1 - (2 * inter + eps) / (union + eps)
    return dice
def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):
    """Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """

    # No need to upsample predictions as we are using normalized coordinates :)

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # get gt labels
        target_masks = target_masks.float()
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

    del src_masks
    del target_masks
    return loss_mask, loss_dice


def celoss(logits, labels):
    """
    Binary Cross Entropy Loss with logits input
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

def diceloss(logits, labels, smooth=1e-6):
    """
    Dice Loss
    """
    # 将logits通过sigmoid变成概率
    probs = torch.sigmoid(logits)
    
    # 拉平成一维
    probs = probs.view(-1)
    labels = labels.view(-1)
    
    intersection = (probs * labels).sum()
    union = probs.sum() + labels.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

class EdgeLossAutoWeight_V2(nn.Module):
    def __init__(self, mode='bce+dice', alpha=0.25, gamma=2.0, 
                 dice_smooth=1e-6, weight_clamp=(0.5, 10.0),  # 新增权重范围参数
                 fp_penalty=0.3):  # 新增FP惩罚系数
        super(EdgeLossAutoWeight_V2, self).__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.dice_smooth = dice_smooth
        self.weight_clamp = weight_clamp  # 更温和的权重范围
        self.fp_penalty = fp_penalty  # FP惩罚系数

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        B = targets.shape[0]
        
        # === 改进1：更安全的权重计算 ===
        with torch.no_grad():
            pos = targets.sum(dim=(1,2,3)) + self.dice_smooth
            neg = targets.numel() / B - pos  # 更稳定的负样本计算
            
            # 关键修改：权重范围从[1,100]改为[0.5,10]
            pos_weight = (neg / pos).clamp(*self.weight_clamp)
            pos_weight = pos_weight.view(B, 1, 1, 1)

        losses = []
        fp_mask = (targets == 0)  # 背景区域掩码

        # === 改进2：增加FP惩罚项 ===
        if self.fp_penalty > 0:
            # 对背景区域的错误预测进行惩罚
            fp_loss = (probs[fp_mask] ** 2).mean()  # 平方惩罚更关注大误差
            losses.append(self.fp_penalty * fp_loss)

        # === 改进3：动态调整损失组合 ===
        if 'bce' in self.mode:
            # 使用动态权重
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, 
                weight=torch.where(targets==1, pos_weight, torch.ones_like(pos_weight))
            )
            losses.append(bce_loss)

        if 'dice' in self.mode:
            # 改进Dice：增加对FP的敏感性
            intersection = (probs * targets).sum(dim=(1,2,3))
            union = (probs + targets).sum(dim=(1,2,3))
            # 分母加入FP惩罚项
            fp_area = (probs * (1 - targets)).sum(dim=(1,2,3))
            dice = (2*intersection + self.dice_smooth) / (union + 0.5*fp_area + self.dice_smooth)
            losses.append(1 - dice.mean())

        if 'focal' in self.mode:
            # 焦点损失抑制易分类背景
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = (self.alpha * (1-pt)**self.gamma * bce_loss)
            # 对背景区域加强惩罚
            focal_loss[fp_mask] *= 1.5  
            losses.append(focal_loss.mean())

        return sum(losses) / len(losses)

class EdgeLossRefine_V2(nn.Module):
    def __init__(self, dice_smooth=1e-6, fp_weight=0.1, fn_weight=0.1, 
                 bce_weight=0.4, dice_weight=0.4, tversky_alpha=0.7):
        super().__init__()
        self.dice_smooth = dice_smooth
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_alpha = tversky_alpha  # 控制FP/FN平衡

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # 1. BCE Loss - 基础分类损失
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # 2. Dice Loss - 处理类别不平衡
        num = (probs * targets).sum(dim=(1,2,3))
        den = (probs + targets).sum(dim=(1,2,3))
        dice_loss = 1 - ((2 * num + self.dice_smooth) / (den + self.dice_smooth)).mean()
        
        # 3. Tversky Loss - 更好平衡FP和FN
        tp = (probs * targets).sum(dim=(1,2,3))
        fp = (probs * (1 - targets)).sum(dim=(1,2,3))
        fn = ((1 - probs) * targets).sum(dim=(1,2,3))
        
        tversky_numerator = tp + self.dice_smooth
        tversky_denominator = tp + self.tversky_alpha * fp + (1 - self.tversky_alpha) * fn + self.dice_smooth
        tversky_loss = 1 - (tversky_numerator / tversky_denominator).mean()
        
        # 4. 显式的FP和FN惩罚
        fp_penalty = (probs * (1 - targets)).mean()  # 假阳性惩罚
        fn_penalty = ((1 - probs) * targets).mean()  # 假阴性惩罚
        
        # 动态权重调整（可选）
        with torch.no_grad():
            # 计算当前batch的FP和FN比例
            fp_rate = (probs > 0.5).float() * (1 - targets)
            fn_rate = (probs <= 0.5).float() * targets
            current_fp_ratio = fp_rate.sum() / (1 - targets).sum().clamp(min=1)
            current_fn_ratio = fn_rate.sum() / targets.sum().clamp(min=1)
            
            # 根据当前情况动态调整权重
            dynamic_fp_weight = self.fp_weight * (1 + current_fp_ratio)
            dynamic_fn_weight = self.fn_weight * (1 + current_fn_ratio)
        
        # 组合损失
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            0.2 * tversky_loss +  # Tversky权重
            dynamic_fp_weight * fp_penalty +
            dynamic_fn_weight * fn_penalty
        )
        
        return total_loss

class EdgeLossRefine(nn.Module):
    def __init__(self, dice_smooth=1e-6, fp_weight=0.1):
        super().__init__()
        self.dice_smooth = dice_smooth
        self.fp_weight = fp_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        num = (probs * targets).sum(dim=(1,2,3))
        den = (probs + targets).sum(dim=(1,2,3))
        dice_loss = 1 - ((2 * num + self.dice_smooth) / (den + self.dice_smooth)).mean()

        fp_penalty = (probs * (1 - targets)).mean()

        total_loss = 0.7 * bce_loss + 0.3 * dice_loss + self.fp_weight * fp_penalty
        return total_loss


class EdgeLossAutoWeight(nn.Module):
    def __init__(self, mode='bce+dice', alpha=0.25, gamma=2.0, dice_smooth=1e-6):
        """
        mode: 损失组合，可选 'bce' / 'dice' / 'focal' / 'bce+dice'
        alpha: focal loss的alpha
        gamma: focal loss的gamma
        dice_smooth: dice防止除0的小常数
        """
        super(EdgeLossAutoWeight, self).__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.dice_smooth = dice_smooth

    def forward(self, logits, targets):
        """
        logits: [B, 1, H, W]，未sigmoid
        targets: [B, 1, H, W]，0/1标签
        """
        probs = torch.sigmoid(logits)

        # 自动计算类别权重
        with torch.no_grad():
            # 正样本（边缘）数量
            pos = (targets == 1).float().sum(dim=(1,2,3))
            # 负样本（背景）数量
            neg = (targets == 0).float().sum(dim=(1,2,3))
            # 避免除0
            pos_weight = (neg + self.dice_smooth) / (pos + self.dice_smooth)
            pos_weight = pos_weight.clamp(min=1.0, max=100.0)  # 限制范围，防爆
            pos_weight = pos_weight.view(-1,1,1,1)  # [B,1,1,1]

        # 开始算Loss
        losses = []

        if 'bce' in self.mode:
            # Weighted BCE
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=pos_weight * targets + (1 - targets)
            ).mean()
            losses.append(bce_loss)

        if 'dice' in self.mode:
            num = (probs * targets).sum(dim=(1,2,3))
            den = (probs + targets).sum(dim=(1,2,3))
            dice = (2 * num + self.dice_smooth) / (den + self.dice_smooth)
            dice_loss = 1 - dice.mean()
            losses.append(dice_loss)

        if 'focal' in self.mode:
            pt = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - pt) ** self.gamma
            focal_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
            focal_loss = (self.alpha * focal_weight * focal_loss).mean()
            losses.append(focal_loss)

        total_loss = sum(losses)/len(losses)

        return total_loss
def affinity_loss(embed_map, gt_map):
    """
    embed_map: (B, C, H, W)
    gt_map: (B, H, W) with instance ids
    """
    B, C, H, W = embed_map.shape

    # normalize embedding
    embed_map = F.normalize(embed_map, p=2, dim=1)  # L2归一化 (B, C, H, W)

    # unfold for 4-neighbors: right (dx=1), down (dy=1)
    shifts = [(0, 1), (1, 0)]  # right, down

    total_loss = 0.0
    count = 0

    for dy, dx in shifts:
        # shift embedding
        shifted_embed = F.pad(embed_map, (dx, 0, dy, 0))[:, :, :H, :W]  # (B, C, H, W)
        shifted_gt = F.pad(gt_map, (dx, 0, dy, 0))[:, :H, :W]  # (B, H, W)

        # mask: both pixels valid
        valid_mask = (gt_map != 0) & (shifted_gt != 0)

        # same instance mask
        same_instance = (gt_map == shifted_gt) & valid_mask  # (B, H, W)

        # embedding dot product: cosine similarity
        sim = (embed_map * shifted_embed).sum(dim=1)  # (B, H, W)

        # loss term
        pos_loss = (1 - sim)[same_instance].mean()
        neg_loss = (sim[same_instance == False] ** 2).mean()  # pull dissimilar

        total_loss += pos_loss + neg_loss
        count += 1

    return total_loss / count
class PixelAffinityLoss(nn.Module):
    def __init__(self, radius=3, pos_margin=0.1, neg_margin=0.4):
        super().__init__()
        self.radius = radius
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, pred, gt):
        """
        pred: (B, 1, H, W), sigmoid后的连续值
        gt:   (B, H, W), 实例ID从1开始，0表示背景
        """
        B, _, H, W = pred.shape
        pred = pred.squeeze(1)  # (B, H, W)
        gt = gt.squeeze(1)  
        pred = torch.sigmoid(pred)

        total_loss = 0.0
        total_count = 0

        device = pred.device

        # 为所有位移生成一个邻域内的偏移坐标
        shifts = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                if dy == 0 and dx == 0:
                    continue
                shifts.append((dy, dx))

        for dy, dx in shifts:
            # 计算 pad 大小（F.pad 的顺序为 [left, right, top, bottom]）
            pad_left = max(dx, 0)
            pad_right = max(-dx, 0)
            pad_top = max(dy, 0)
            pad_bottom = max(-dy, 0)

            padded_pred = F.pad(pred, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
            padded_gt = F.pad(gt, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

            # 对应偏移裁剪
            shifted_pred = padded_pred[:, pad_top - dy : pad_top - dy + H,
                                        pad_left - dx : pad_left - dx + W]

            shifted_gt = padded_gt[:, pad_top - dy : pad_top - dy + H,
                                    pad_left - dx : pad_left - dx + W]

            # 原始像素 vs 邻域像素
            diff = (pred - shifted_pred).abs()  # (B, H, W)
            same_instance = (gt == shifted_gt) & (gt != 0)
            diff_instance = (gt != shifted_gt) & (gt != 0) & (shifted_gt != 0)

            pos_loss = F.relu(diff - self.pos_margin)[same_instance]
            neg_loss = F.relu(self.neg_margin - diff)[diff_instance]

            total_loss += pos_loss.sum() + neg_loss.sum()
            total_count += pos_loss.numel() + neg_loss.numel()

        if total_count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / total_count

def extract_instance_masks(gt_mask):
    """
    gt_mask: Tensor of shape (H, W), each instance has a unique ID, background = 0
    returns: list of binary masks (each of shape H×W)
    """
    instance_ids = gt_mask.unique()
    instance_ids = instance_ids[instance_ids != 0]  # remove background

    masks = [(gt_mask == id_).float() for id_ in instance_ids]
    return masks, instance_ids

def fast_mask_nms_single(masks: torch.Tensor, scores: torch.Tensor = None, iou_thresh=0.5):
    """
    加速版的 mask NMS，适用于 [N, H, W] 的掩码张量。
    使用向量化方式计算 IoU，避免嵌套循环。
    返回一个 bool 向量，指示哪些 mask 被保留。
    """
    N, H, W = masks.shape
    device = masks.device
    masks_flat = masks.reshape(N, -1).float()  # [N, H*W]
    masks_flat = (masks_flat > 0.5).float()

    # 面积
    areas = masks_flat.sum(dim=1)  # [N]

    # 交集：N x N
    inter = torch.matmul(masks_flat, masks_flat.T)

    # 并集
    union = areas.unsqueeze(1) + areas.unsqueeze(0) - inter
    iou_matrix = inter / union.clamp(min=1e-6)

    # 排序
    if scores is not None:
        order = torch.argsort(scores, descending=True)
    else:
        order = torch.arange(N, device=device)

    keep = []
    suppressed = torch.zeros(N, dtype=torch.bool, device=device)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i.item())
        suppressed = suppressed | (iou_matrix[i] > iou_thresh)

    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    keep_mask[keep] = True
    return keep_mask

def fast_mask_nms_batch(masks: torch.Tensor, scores: torch.Tensor = None, iou_thresh=0.5):
    """
    批处理版本的快速 NMS，输入 masks 为 [B, N, H, W]。
    """
    B, N, H, W = masks.shape
    keep_masks = torch.zeros_like(masks)

    for b in range(B):
        cur_masks = masks[b]
        cur_scores = scores[b] if scores is not None else None
        keep_mask = fast_mask_nms_single(cur_masks, cur_scores, iou_thresh=iou_thresh)
        keep_masks[b] = cur_masks * keep_mask[:, None, None]  # broadcasting
    return keep_masks

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
        instance_counter = 1  #实例 ID 从 1 开始
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

def compute_cost_matrix(
    pred_logits: torch.Tensor,  # Shape: [N, H, W]
    gt_bin_masks: torch.Tensor,    # Shape: [M, H, W]
    bce_weight: float,
    dice_weight: float,
    chunk_size: int = None         # 分块计算防止内存溢出
) -> torch.Tensor:
    """
    计算所有 (pred_mask, gt_mask) 对的加权损失矩阵 [N, M]
    """
    # 确保数据在相同设备上
    assert pred_logits.device == gt_bin_masks.device, "Tensors must be on the same device"
    
    # 分块计算模式（适用于大尺寸 N/M）
    if chunk_size is not None:
        return _compute_cost_matrix_chunked(pred_logits, gt_bin_masks, bce_weight, dice_weight, chunk_size)
    pred_probs = torch.sigmoid(pred_logits) 
    # 向量化计算模式（最快）
    N, H, W = pred_logits.shape
    M = gt_bin_masks.shape[0]
    
    # 扩展维度以便广播 [N, M, H, W]
    pred_exp = pred_probs.unsqueeze(1)  # [N, 1, H, W]
    gt_exp = gt_bin_masks.unsqueeze(0)     # [1, M, H, W]
    pred_exp_broadcast = pred_exp.expand(-1, M, -1, -1)  # [N, M, H, W]
    gt_exp_broadcast = gt_exp.expand(N, -1, -1, -1)      # [N, M, H, W]
    # 计算 BCE 矩阵 [N, M]
    bce_matrix = F.binary_cross_entropy(pred_exp_broadcast, gt_exp_broadcast, reduction='none').mean(dim=(2,3))
    
    # 计算 Dice 矩阵 [N, M]
    intersection = (pred_exp * gt_exp).sum(dim=(2,3))
    union = pred_exp.sum(dim=(2,3)) + gt_exp.sum(dim=(2,3))
    dice_matrix = 1 - (2 * intersection) / (union + 1e-8)  # Dice损失
    
    # 合并损失
    cost_matrix = bce_weight * bce_matrix + dice_weight * dice_matrix
    return cost_matrix

def _compute_cost_matrix_chunked(
    pred: torch.Tensor,
    gt: torch.Tensor,
    bce_w: float,
    dice_w: float,
    chunk_size: int
) -> torch.Tensor:
    """
    分块计算模式（内存优化）
    """
    device = pred.device
    N, H, W = pred.shape
    M = gt.shape[0]
    cost = torch.zeros(N, M, device=device)
    pred = torch.sigmoid(pred) 
    # 分块计算
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        pred_chunk = pred[i:i_end].unsqueeze(1)  # [chunk, 1, H, W]
        
        for j in range(0, M, chunk_size):
            j_end = min(j + chunk_size, M)
            gt_chunk = gt[j:j_end].unsqueeze(0)  # [1, chunk, H, W]
            pred_bc = pred_chunk.expand(-1, gt_chunk.size(1), -1, -1)
            gt_bc = gt_chunk.expand(pred_chunk.size(0), -1, -1, -1)

            # gt_chunk = gt_bc
            # pred_chunk = pred_bc
            # 计算当前分块的 BCE
            bce = F.binary_cross_entropy(pred_bc, gt_bc, reduction='none').mean((2,3))
            
            # 计算当前分块的 Dice
            intersection = (pred_bc * gt_bc).sum((2,3))
            union = pred_bc.sum((2,3)) + gt_bc.sum((2,3))
            dice = 1 - (2 * intersection) / (union + 1e-8)
            
            # 更新分块区域
            cost[i:i_end, j:j_end] = bce_w * bce + dice_w * dice
            
    return cost
class InstanceSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_masks, gt_masks):
        """
        pred_masks: (B, N, H, W) logits
        gt_masks: (B, H, W), int, each instance has a unique ID (>0), 0 is background
        """
        B, N, H, W = pred_masks.shape
        total_loss = 0
        total_matches = 0

        for b in range(B):
            pred = pred_masks[b]  # (N, H, W)
            gt = gt_masks[b].squeeze()      # (H, W)

            instance_ids = torch.unique(gt)
            instance_ids = instance_ids[instance_ids > 0]  # exclude background

            if len(instance_ids) == 0:
                continue  # No objects in this image

            gt_bin_masks = torch.stack([(gt == inst_id).float() for inst_id in instance_ids])  # (M, H, W)
            M = gt_bin_masks.shape[0]

            # Resize pred to match GT
            pred_bin_masks = pred  # (N, H, W)

            # Compute pairwise BCE+DICE cost
            with torch.no_grad():
                cost_matrix = compute_cost_matrix(pred_bin_masks, gt_bin_masks, self.bce_weight, self.dice_weight,chunk_size=64)
                # for i in range(N):
                #     for j in range(M):
                #         bce = self.bce_loss(pred_bin_masks[i], gt_bin_masks[j]).mean()
                #         dice = dice_loss_instance(pred_bin_masks[i:i+1], gt_bin_masks[j:j+1]).mean()
                #         cost_matrix[i, j] = bce * self.bce_weight + dice * self.dice_weight

                pred_inds, gt_inds = linear_sum_assignment(cost_matrix.cpu().numpy())

            matched_pred = pred_bin_masks[pred_inds]   # (M, H, W)
            matched_gt = gt_bin_masks[gt_inds]         # (M, H, W)

            # Compute loss on matched pairs
            bce_loss = self.bce_loss(matched_pred, matched_gt).mean()
            dice = dice_loss_instance(matched_pred, matched_gt).mean()

            loss = self.bce_weight * bce_loss + self.dice_weight * dice
            total_loss += loss
            total_matches += 1

        if total_matches == 0:
            return torch.tensor(0.0, device=pred_masks.device)
        return total_loss / total_matches

# 排斥预测其他实例区域 /
# class InstanceSegmentationLoss_v2(nn.Module):
#     def __init__(self, bce_weight=1.0, dice_weight=1.0,
#                  exclu_weight=0.5, bg_weight=0.5, comp_weight=0.1):
#         super().__init__()
#         self.bce_weight = bce_weight
#         self.dice_weight = dice_weight
#         self.exclu_weight = exclu_weight
#         self.bg_weight = bg_weight
#         self.comp_weight = comp_weight

#         self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # reduction done later

#     def forward(self, pred_masks, gt_masks):
#         """
#         pred_masks: (B, N, H, W) - predicted mask logits per instance token
#         gt_masks: (B, H, W) - GT instance masks with integer IDs (>0), 0 for background
#         """
#         B, N, H, W = pred_masks.shape
#         total_loss = 0
#         total_matches = 0

#         for b in range(B):
#             pred = pred_masks[b]         # (N, H, W)
#             gt = gt_masks[b].squeeze()             # (H, W)

#             instance_ids = torch.unique(gt)
#             instance_ids = instance_ids[instance_ids > 0]  # ignore background
#             if len(instance_ids) == 0:
#                 continue

#             # Create GT binary masks (M, H, W)
#             gt_bin_masks = torch.stack([(gt == inst_id).float() for inst_id in instance_ids])
#             M = gt_bin_masks.shape[0]

#             with torch.no_grad():
#                 cost_matrix = compute_cost_matrix(pred, gt_bin_masks, self.bce_weight, self.dice_weight,chunk_size=64)
#                 pred_inds, gt_inds = linear_sum_assignment(cost_matrix.cpu().numpy())

#             matched_pred = pred[pred_inds]         # (M, H, W)
#             matched_gt = gt_bin_masks[gt_inds]     # (M, H, W)

#             # Main Loss: BCE + DICE
#             bce = self.bce_loss(matched_pred, matched_gt).mean()
#             dice = dice_loss_instance(matched_pred, matched_gt).mean()
#             main_loss = self.bce_weight * bce + self.dice_weight * dice

#             matched_pred = matched_pred.sigmoid() 
#             # Exclusivity Loss: mask should NOT respond outside matched_gt
#             exclu_loss = (matched_pred * (1 - matched_gt)).mean()

#             pred = pred.sigmoid()  # (N, H, W)
#             # Background Suppression Loss
#             all_gt_mask = (gt > 0).float()  # union of GT
#             background_mask = 1.0 - all_gt_mask  # (H, W)
#             background_loss = (pred * background_mask.unsqueeze(0)).mean()

#             # Competition Loss: Softmax over masks (N, H, W)
#             prob = torch.softmax(pred, dim=0)
#             comp_loss = -(prob * torch.log(prob + 1e-6)).sum(dim=0).mean()  # entropy

#             loss = (
#                 main_loss
#                 + self.exclu_weight * exclu_loss
#                 + self.bg_weight * background_loss
#                 + self.comp_weight * comp_loss
#             )

#             total_loss += loss
#             total_matches += 1

#         if total_matches == 0:
#             return torch.tensor(0.0, device=pred_masks.device)
#         return total_loss / total_matches

# 排他/背景/竞争/类别分类损失
class InstanceSegmentationLoss_v2(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0,
                 exclu_weight=0.5, bg_weight=0.5, comp_weight=0.1,
                 cls_weight=1.0, semantic_weight=0.5, semantic_supervise=False):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.exclu_weight = exclu_weight
        self.bg_weight = bg_weight
        self.comp_weight = comp_weight
        self.cls_weight = cls_weight
        self.semantic_weight = semantic_weight
        self.semantic_supervise = semantic_supervise

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # reduction done later

    def forward(self, pred_masks, class_logits, gt_masks , instance_cluster=None):
        """
        pred_masks: (B, N, H, W) - predicted mask logits
        class_logits: (B, N, 2) - classification logits (C=2: background/foreground)
        gt_masks: (B, H, W) - GT instance masks with instance IDs (>0), 0 for background
        gt_semantic_masks: (B, H, W) - binary semantic mask (optional, if semantic_supervise=True)
        """
        B, N, H, W = pred_masks.shape
        total_loss = 0
        total_matches = 0

        for b in range(B):
            pred = pred_masks[b]         # (N, H, W)
            gt = gt_masks[b].squeeze()   # (H, W)

            instance_ids = torch.unique(gt)
            instance_ids = instance_ids[instance_ids > 0]  # ignore background
            if len(instance_ids) == 0:
                continue

            # Step 1: 生成 GT 二值 mask
            gt_bin_masks = torch.stack([(gt == inst_id).float() for inst_id in instance_ids])
            M = gt_bin_masks.shape[0]

            # Step 2: 匈牙利匹配
            with torch.no_grad():
                cost_matrix = compute_cost_matrix(pred, gt_bin_masks, self.bce_weight, self.dice_weight, chunk_size=64)
                pred_inds, gt_inds = linear_sum_assignment(cost_matrix.cpu().numpy())

            matched_pred = pred[pred_inds]         # (M, H, W)
            matched_gt = gt_bin_masks[gt_inds]     # (M, H, W)

            # Step 3: 原始 mask 损失部分
            bce = self.bce_loss(matched_pred, matched_gt).mean()
            dice = dice_loss_instance(matched_pred, matched_gt).mean()
            main_loss = self.bce_weight * bce + self.dice_weight * dice

            matched_pred = matched_pred.sigmoid()

            # Step 4: 排他损失
            exclu_loss = (matched_pred * (1 - matched_gt)).mean()

            # Step 5: 背景损失
            pred_all = pred.sigmoid()  # (N, H, W)
            all_gt_mask = (gt > 0).float()
            background_mask = 1.0 - all_gt_mask  # (H, W)
            background_loss = (pred_all * background_mask.unsqueeze(0)).mean()

            # Step 6: 竞争损失
            # prob = torch.softmax(pred_all, dim=0)
            # comp_loss = -(prob * torch.log(prob + 1e-6)).sum(dim=0).mean()

            # Step 7: 分类损失
            if class_logits is not None:
                num_matched = min(len(pred_inds), class_logits.shape[1])
                pred_inds = pred_inds[:num_matched]
                gt_inds = gt_inds[:num_matched]

                class_logit = class_logits[b]  # (N, 2)
                matched_class_logits = class_logit[pred_inds]  # (M, 2)
                M = num_matched
                gt_classes = torch.ones(M, dtype=torch.long, device=gt.device)  # 前景类=1
                cls_loss = F.cross_entropy(matched_class_logits, gt_classes)
            else:
                cls_loss = 0.0

            # Step 8: 语义 mask 聚合监督（可选）
            if self.semantic_supervise and instance_cluster is not None:
                # fused_mask = pred_all.mean(dim=0, keepdim=True)              # (1, H, W)
                fused_mask = instance_cluster[b].sigmoid()  # (1, H, W)
                gt_semantic = (gt > 0).float().unsqueeze(0)                 # (1, H, W)
                semantic_bce = F.binary_cross_entropy(fused_mask, gt_semantic)
                semantic_dice = dice_loss_instance(fused_mask, gt_semantic)
                semantic_loss = semantic_bce + semantic_dice
            else:
                semantic_loss = 0.0

            loss = (
                main_loss
                + self.exclu_weight * exclu_loss
                + self.bg_weight * background_loss
                # + self.comp_weight * comp_loss
                + self.cls_weight * cls_loss
                + self.semantic_weight * semantic_loss
            )

            total_loss += loss
            total_matches += 1

        if total_matches == 0:
            return torch.tensor(0.0, device=pred_masks.device)
        return total_loss / total_matches

def pairwise_distance_v2(proxies, x, squared=False):
    if squared:
        return (torch.cdist(x, proxies, p=2)) ** 2
    else:
        return torch.cdist(x, proxies, p=2)
# def instance_proxy_loss(instance_tokens, instance_proxies, temperature=0.07):
#     """
#     instance_tokens: [N, D] - 当前模型学习到的 instance 表示
#     instance_proxies: [N, D] - 每个实例的“目标 anchor”
#     """

#     # 单位归一化
#     token_norm = F.normalize(instance_tokens, p=2, dim=-1)
#     proxy_norm = F.normalize(instance_proxies, p=2, dim=-1)

#     # 计算 pairwise 相似度（点积）: [N, N]
#     sim_matrix = torch.matmul(token_norm, proxy_norm.T) / temperature
#     token_dis = pairwise_distance_v2(proxies=proxy_norm, x=token_norm, squared=True)
#     # 目标：第 i 个 token 靠近第 i 个 proxy
#     targets = torch.arange(instance_tokens.size(0), device=instance_tokens.device)

#     loss = F.cross_entropy(sim_matrix, targets)
#     return loss
def instance_proxy_loss(instance_tokens: torch.Tensor, instance_proxies: torch.Tensor, temperature: float = 0.07):
    """
    计算 batch 版 instance-proxy 对比损失。

    参数：
    - instance_tokens: [B, N, D]，模型提取的实例 token 表示
    - instance_proxies: [ N, D]，每个实例的目标 anchor 向量
    - temperature: 温度参数

    返回：
    - loss: scalar，对比损失
    """
    B, N, D = instance_tokens.shape

    # 单位归一化
    token_norm = temperature * F.normalize(instance_tokens, p=2, dim=-1)   # [B, N, D]
    proxy_norm = temperature * F.normalize(instance_proxies, p=2, dim=-1)  # [N, D]

    losses = []
    for b in range(B):
        # 当前 batch 内相似度矩阵：[N, N]
        # sim_matrix = torch.matmul(token_norm[b], proxy_norm[b].T) / temperature
        token_dis = pairwise_distance_v2(proxies=proxy_norm, x=token_norm[b], squared=True)
        targets = torch.arange(N, device=instance_tokens.device)
        loss_b = F.cross_entropy(token_dis, targets)
        losses.append(loss_b)

    return torch.stack(losses).mean()
# 类别分类损失
class InstanceSegmentationLoss_v3(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0,
                 exclu_weight=0.5, bg_weight=0.5, comp_weight=0.1,
                 cls_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.exclu_weight = exclu_weight
        self.bg_weight = bg_weight
        self.comp_weight = comp_weight
        self.cls_weight = cls_weight

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # reduction done later

    def forward(self, pred_masks, class_logits, gt_masks):
        """
        pred_masks: (B, N, H, W) - predicted mask logits
        class_logits: (B, N, 2) - classification logits (C=2: background/foreground)
        gt_masks: (B, H, W) - GT instance masks with instance IDs (>0), 0 for background
        """
        B, N, H, W = pred_masks.shape
        total_loss = 0
        total_matches = 0

        for b in range(B):
            pred = pred_masks[b]         # (N, H, W)
            gt = gt_masks[b].squeeze()   # (H, W)
            
            instance_ids = torch.unique(gt)
            instance_ids = instance_ids[instance_ids > 0]  # ignore background
            if len(instance_ids) == 0:
                continue

            # Step 1: 生成 GT 二值 mask
            gt_bin_masks = torch.stack([(gt == inst_id).float() for inst_id in instance_ids])
            M = gt_bin_masks.shape[0]

            # Step 2: 匈牙利匹配
            with torch.no_grad():
                cost_matrix = compute_cost_matrix(pred, gt_bin_masks, self.bce_weight, self.dice_weight, chunk_size=64)
                pred_inds, gt_inds = linear_sum_assignment(cost_matrix.cpu().numpy())

            matched_pred = pred[pred_inds]         # (M, H, W)
            matched_gt = gt_bin_masks[gt_inds]     # (M, H, W)

            # Step 3: 原始 mask 损失部分
            bce = self.bce_loss(matched_pred, matched_gt).mean()
            dice = dice_loss_instance(matched_pred, matched_gt).mean()
            main_loss = self.bce_weight * bce + self.dice_weight * dice

            matched_pred = matched_pred.sigmoid()

            # # Step 4: 排他损失
            # exclu_loss = (matched_pred * (1 - matched_gt)).mean()

            # # Step 5: 背景损失
            # pred_all = pred.sigmoid()  # (N, H, W)
            # all_gt_mask = (gt > 0).float()
            # background_mask = 1.0 - all_gt_mask  # (H, W)
            # background_loss = (pred_all * background_mask.unsqueeze(0)).mean()

            # # Step 6: 竞争损失
            # prob = torch.softmax(pred_all, dim=0)
            # comp_loss = -(prob * torch.log(prob + 1e-6)).sum(dim=0).mean()
            
            if class_logits is not None:
                # 防止超出边界
                num_matched = min(len(pred_inds), class_logits.shape[1])
                pred_inds = pred_inds[:num_matched]
                gt_inds = gt_inds[:num_matched]

                class_logit = class_logits[b]  # (N, 2)
                # Step 7: 类别分类损失
                matched_class_logits = class_logit[pred_inds]  # (M, 2)
                M = num_matched
                gt_classes = torch.ones(M, dtype=torch.long, device=gt.device)  # 前景类=1
                cls_loss = F.cross_entropy(matched_class_logits, gt_classes)
                loss = (
                    main_loss
                    + self.cls_weight * cls_loss
                )
            else:
                loss = main_loss
            # loss = (
            #     main_loss
            #     + self.exclu_weight * exclu_loss
            #     + self.bg_weight * background_loss
            #     + self.comp_weight * comp_loss
            #     + self.cls_weight * cls_loss
            # )
            total_loss += loss
            total_matches += 1

        if total_matches == 0:
            return torch.tensor(0.0, device=pred_masks.device)
        return total_loss / total_matches

# 鼓励不重叠
def orthogonality_loss(mask_logits):
    """
    mask_logits: (B, N, H, W), raw mask logits (before sigmoid)
    """
    B, N, H, W = mask_logits.shape
    # mask_logits: (B, N, H, W)
    mask_probs = torch.sigmoid(mask_logits)
    B, N, H, W = mask_probs.shape

    # Normalize masks
    norm_masks = F.normalize(mask_probs.view(B, N, -1), dim=-1)  # (B, N, HW)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(norm_masks, norm_masks.transpose(1, 2))  # (B, N, N)

    # Ideal情况下是对角阵
    identity = torch.eye(N, device=sim_matrix.device).unsqueeze(0)
    loss_discriminative = ((sim_matrix - identity)**2).mean()
    return loss_discriminative
# 增加连通性
def compactness_loss(mask_logits):
    """
    mask_logits: (B, N, H, W), raw mask logits (before sigmoid)
    """
    B, N, H, W = mask_logits.shape
    device = mask_logits.device
    loss = 0.0

    for b in range(B):
        for n in range(N):
            mask = torch.sigmoid(mask_logits[b, n])  # (H, W)
            # Normalize mask to [0,1], optional
            mask = mask / (mask.max() + 1e-6)
            # Coordinate grid
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            y = y.to(device).float()
            x = x.to(device).float()
            # Centroid
            mass = mask.sum()
            cx = (x * mask).sum() / (mass + 1e-6)
            cy = (y * mask).sum() / (mass + 1e-6)
            # Spatial variance (compactness)
            var = ((x - cx)**2 + (y - cy)**2) * mask
            loss += var.sum() / (mass + 1e-6) /(H**2 + W**2)

    return loss / (B * N)
# class InstanceSegmentationLoss(nn.Module):
#     def __init__(self, use_miou=True, use_dice=True, dice_weight=1.0, miou_weight=1.0):
#         super().__init__()
#         self.use_miou = use_miou
#         self.use_dice = use_dice
#         self.dice_weight = dice_weight
#         self.miou_weight = miou_weight
        
#     def forward(self, pred_instances, gt_instances):
#         """
#         pred_instances: (B, 1, H, W) logits after decoding
#         gt_instances: (B, H, W) instance labels
#         """
#         loss = 0.0
#         B = pred_instances.shape[0]
        
#         for b in range(B):
#             pred = pred_instances[b].squeeze(0)  # (H, W)
#             gt = gt_instances[b]

#             pred = torch.sigmoid(pred)  # 如果输出的是logits，先sigmoid一下

#             if self.use_dice:
#                 dice_loss = self.dice_per_instance_loss(pred, gt)
#                 loss += self.dice_weight * dice_loss

#             if self.use_miou:
#                 miou_loss = self.miou_per_instance_loss(pred, gt)
#                 loss += self.miou_weight * miou_loss

#         loss = loss / B
#         return loss
    
#     def dice_per_instance_loss(self, pred, gt):
#         pred = (pred > 0.5).float()

#         dice_list = []
#         instance_ids = torch.unique(gt)
#         instance_ids = instance_ids[instance_ids != 0]

#         for ins_id in instance_ids:
#             gt_mask = (gt == ins_id).float()
#             intersection = (pred * gt_mask).sum()
#             dice = 2 * intersection / (pred.sum() + gt_mask.sum() + 1e-6)
#             dice_list.append(1 - dice)  # dice loss = 1 - dice coef

#         if len(dice_list) == 0:
#             return torch.tensor(0.0, device=pred.device)
#         return torch.stack(dice_list).mean()

#     def miou_per_instance_loss(self, pred, gt):
#         pred = (pred > 0.5).float()

#         iou_list = []
#         instance_ids = torch.unique(gt)
#         instance_ids = instance_ids[instance_ids != 0]

#         for ins_id in instance_ids:
#             gt_mask = (gt == ins_id).float()
#             intersection = (pred * gt_mask).sum()
#             union = pred.sum() + gt_mask.sum() - intersection
#             if union > 0:
#                 iou = intersection / (union + 1e-6)
#                 iou_list.append(1 - iou)  # iou loss = 1 - iou

#         if len(iou_list) == 0:
#             return torch.tensor(0.0, device=pred.device)
#         return torch.stack(iou_list).mean()

def loss_masks_full(src_masks, target_mask):
    loss_mask = celoss(src_masks, target_mask)
    loss_dice = diceloss(src_masks, target_mask)
    return loss_mask, loss_dice

def diceloss_withmask(input, target, mask=None, smooth=1e-6):
    # input, target: [B, H, W], mask: [B, H, W]
    input = input.contiguous().view(input.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    if mask is not None:
        mask = mask.contiguous().view(mask.size(0), -1)
        input = input * mask
        target = target * mask

    intersection = (input * target).sum(dim=1)
    union = input.sum(dim=1) + target.sum(dim=1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def loss_masks_full_v2(src_masks, target_mask,  diceloss=diceloss_withmask):
    """
    src_masks: [B, 4, H, W]  - 每个聚类中心的预测 mask logits
    target_mask: [B, H, W]   - 聚类结果标签，0 为背景，其它为聚类中心 1~4
    celoss: cross-entropy 函数（必须支持 ignore_index）
    diceloss: 可选的 dice loss 函数
    """
    B, C, H, W = src_masks.shape
    loss_mask_total = 0.0
    loss_dice_total = 0.0

    for i in range(1, C + 1):  # i in [1, 4]
        pred_i = src_masks[:, i - 1]  # [B, H, W]
        # mask 仅包含当前类和背景
        valid_mask = (target_mask == 0) | (target_mask == i)  # [B, H, W]
        # 构建 binary target：当前类 → 1，背景 → 0，其他忽略
        binary_target = torch.where(target_mask == i, 1, 0)  # [B, H, W]
        # 将 invalid 区域标为 ignore
        binary_target = binary_target.masked_fill(~valid_mask, 255)  # 使用 255 作为 ignore_index
        
        valid_mask = (binary_target != 255).float()
        binary_target = binary_target.clone().float()
        binary_target[binary_target == 255] = 0.0
        # 计算 binary CE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_i.unsqueeze(1), binary_target, weight=valid_mask, reduction='mean'
        )

        loss_mask_total += bce_loss

        # Dice Loss（如果有的话）
        if diceloss_withmask is not None:
            # binary_target 中非 255 的位置用于 dice
            pred_sigmoid = torch.sigmoid(pred_i)
            binary_target_masked = binary_target.clone()
            binary_target_masked[binary_target_masked == 255] = 0
            mask_valid = valid_mask.float()
            dice_loss = diceloss_withmask(pred_sigmoid, binary_target_masked.float(), mask=mask_valid)
            loss_dice_total += dice_loss
    mean_loss = (loss_mask_total / C + (loss_dice_total / C if diceloss is not None else None)) / 2
    return mean_loss