import torch
import torch.nn.functional as F
import torch.nn as nn

def encode_deltas(anchors, gt_boxes, eps=1e-6):
    anchor_widths = anchors[:, 2] - anchors[:, 0] + eps
    anchor_heights = anchors[:, 3] - anchors[:, 1] + eps
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + eps
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + eps
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    dw = torch.log(gt_widths / anchor_widths)
    dh = torch.log(gt_heights / anchor_heights)

    return torch.stack((dx, dy, dw, dh), dim=1)

def compute_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

def match_anchors_to_gt(anchors, gt_boxes, pos_iou_threshold=0.7, neg_iou_threshold=0.3):
    ious = compute_iou(anchors, gt_boxes)
    max_iou, matched_gt_idx = ious.max(dim=1)
    labels = torch.full((anchors.shape[0],), -1, dtype=torch.long, device=anchors.device)
    labels[max_iou >= pos_iou_threshold] = 1
    labels[max_iou < neg_iou_threshold] = 0
    matched_gt_boxes = gt_boxes[matched_gt_idx]
    return labels, matched_gt_boxes

def rpn_loss(rpn_logits, rpn_bbox, anchors, gt_boxes):
    """
    Args:
        rpn_logits: (B, N, 2)
        rpn_bbox: (B, N, 4)
        anchors: (N, 4)
        gt_boxes: list of ground-truth boxes per batch, len = B

    Returns:
        cls_loss, reg_loss
    """
    batch_size = rpn_logits.shape[0]
    total_cls_loss, total_reg_loss = 0.0, 0.0

    for b in range(batch_size):
        logits = rpn_logits[b]        # (N, 2)
        bbox_preds = rpn_bbox[b]      # (N, 4)
        gts = gt_boxes[b]             # (M, 4)

        labels, matched_gt = match_anchors_to_gt(anchors, gts)
        pos_mask = labels == 1
        valid_mask = labels >= 0

        # classification loss
        cls_loss = F.cross_entropy(logits[valid_mask], labels[valid_mask])
        total_cls_loss += cls_loss

        # regression loss
        if pos_mask.sum() > 0:
            target_deltas = encode_deltas(anchors[pos_mask], matched_gt[pos_mask])
            reg_loss = F.smooth_l1_loss(bbox_preds[pos_mask], target_deltas)
        else:
            reg_loss = torch.tensor(0.0, device=logits.device)
        total_reg_loss += reg_loss

    return total_cls_loss / batch_size, total_reg_loss / batch_size
class RPNLossComputation:
    def __init__(self, iou_threshold_pos=0.7, iou_threshold_neg=0.5, batch_size_per_image=256, positive_fraction=1):
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def compute_iou(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter

        return inter / union

    def match_anchors(self, anchors, gt_boxes):
        iou = self.compute_iou(anchors, gt_boxes)
        max_iou, matched_idx = iou.max(dim=1)
        labels = torch.full((anchors.shape[0],), -1, dtype=torch.int64, device=anchors.device)
        labels[max_iou < self.iou_threshold_neg] = 0
        labels[max_iou >= self.iou_threshold_pos] = 1
        return labels, matched_idx

    def subsample(self, labels):
        positive = torch.nonzero(labels == 1).squeeze(1)
        negative = torch.nonzero(labels == 0).squeeze(1)

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(negative.numel(), num_neg)

        perm_pos = torch.randperm(positive.numel(), device=labels.device)[:num_pos]
        perm_neg = torch.randperm(negative.numel(), device=labels.device)[:num_neg]

        keep_idx = torch.cat([positive[perm_pos], negative[perm_neg]], dim=0)
        return keep_idx, labels[keep_idx]

    def box2delta(self, anchors, gt_boxes):
        ax, ay, aw, ah = self._xyxy_to_cxcywh(anchors)
        gx, gy, gw, gh = self._xyxy_to_cxcywh(gt_boxes)

        dx = (gx - ax) / aw
        dy = (gy - ay) / ah
        dw = torch.log(gw / aw)
        dh = torch.log(gh / ah)

        return torch.stack([dx, dy, dw, dh], dim=1)
    def coco_to_xyxy(self, boxes):
        # boxes: Tensor of shape (N, 4), format: [x, y, w, h]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)
    def _xyxy_to_cxcywh(self, boxes):
        x1, y1, x2, y2 = boxes.unbind(1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h

    def __call__(self, anchors, pred_cls_logits, pred_bbox_deltas, targets):
        """
        anchors: (B, N, 4)
        pred_cls_logits: (B, N, 2)
        pred_bbox_deltas: (B, N, 4)
        targets: list of dicts, each dict has 'boxes': (M, 4)
        """
        batch_size = pred_cls_logits.shape[0]
        total_cls_loss = 0.0
        total_reg_loss = 0.0

        for b in range(batch_size):
            anchors_b = anchors[b]  # (N, 4)
            cls_logits_b = pred_cls_logits[b]  # (N, 2)
            bbox_deltas_b = pred_bbox_deltas[b]  # (N, 4)
            gt_boxes = targets['boxes'][b].to(anchors_b.device)
            
            # resize to 1/4 of the image size

            if gt_boxes.numel() == 0:
                # No GT boxes: all anchors are negative
                labels = torch.zeros(anchors_b.shape[0], dtype=torch.int64, device=anchors_b.device)
                keep_idx, sampled_labels = self.subsample(labels)
                cls_loss = F.cross_entropy(cls_logits_b[keep_idx], sampled_labels)
                reg_loss = torch.tensor(0.0, device=anchors_b.device)
            else:
                # gt_boxes = self.coco_to_xyxy(gt_boxes)
                gt_boxes = resize_boxes(gt_boxes, 1024/256, 1024/256)
                labels, matched_gt_idx = self.match_anchors(anchors_b, gt_boxes)
                keep_idx, sampled_labels = self.subsample(labels)

                sampled_anchors = anchors_b[keep_idx]
                sampled_logits = cls_logits_b[keep_idx]
                sampled_deltas = bbox_deltas_b[keep_idx]

                matched_gt_boxes = gt_boxes[matched_gt_idx[keep_idx]]
                target_deltas = self.box2delta(sampled_anchors, matched_gt_boxes)

                cls_loss = F.cross_entropy(sampled_logits, sampled_labels)

                pos_mask = sampled_labels == 1
                if pos_mask.sum() > 0:
                    reg_loss = F.smooth_l1_loss(sampled_deltas[pos_mask], target_deltas[pos_mask])
                else:
                    reg_loss = torch.tensor(0.0, device=anchors_b.device)

            total_cls_loss += cls_loss
            total_reg_loss += reg_loss

        return total_cls_loss / batch_size + total_reg_loss / batch_size

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

class __InstanceSegmentationLoss_stage2:
    def __init__(self, iou_threshold_pos=0.5, iou_threshold_neg=0.3):
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg

    def compute_iou(self, boxes1, boxes2):
        """
        boxes1: (N, 4)
        boxes2: (M, 4)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter

        return inter / (union + 1e-6)

    def extract_gt_masks(self, gt_mask, gt_ids):
        """
        gt_mask: (H, W), gt_ids: 1D tensor of unique ids
        return: (M, H, W)
        """
        return torch.stack([(gt_mask == gid).float() for gid in gt_ids], dim=0)

    def resize_masks_to_proposals(self, masks, boxes, size):
        """
        masks: (M, H, W)
        boxes: (N, 4)
        size: (h, w)
        return: (N, h, w)
        """
        # Expand masks to (M, 1, H, W) to comply with roi_align
        M, H, W = masks.shape
        masks = masks.unsqueeze(1)

        # Index for roi_align input, repeat box index M times
        idx = torch.arange(boxes.size(0), device=boxes.device).float()
        rois = torch.cat([idx[:, None], boxes], dim=1)  # (N, 5)

        return roi_align(masks.expand(boxes.size(0), -1, H, W), rois, size).squeeze(1)

    def dice_loss(self, pred, target):
        """
        pred: (N, H, W), raw logits
        target: (N, H, W), binary
        """
        pred = pred.sigmoid()
        smooth = 1.0
        num = 2 * (pred * target).sum(dim=(1, 2)) + smooth
        den = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth
        loss = 1 - num / den
        return loss.mean()

    def __call__(self, all_proposals, instance_mask, class_logits, gt_boxes, gt_mask):
        """
        all_proposals: (B, N, 4)
        instance_mask: (B, N, H, W) - predicted mask logits
        gt_boxes: list of (Mi, 4)
        gt_mask: (B, H_img, W_img) - each instance labeled with 1, 2, ...
        """
        B, N, H, W = instance_mask.shape
        total_loss = 0.0

        for b in range(B):
            proposals = all_proposals[b]  # (N, 4)
            pred_masks = instance_mask[b]  # (N, H, W)
            gt_boxes_b = gt_boxes[b]  # (M, 4)
            gt_mask_b = gt_mask[b]  # (H_img, W_img)

            if gt_boxes_b.numel() == 0:
                continue

            # Get valid GT ids and masks
            gt_ids = torch.unique(gt_mask_b)
            gt_ids = gt_ids[gt_ids > 0]
            masks = self.extract_gt_masks(gt_mask_b, gt_ids.to(gt_mask_b.device))  # (M, H, W)

            ious = self.compute_iou(proposals, gt_boxes_b)  # (N, M)
            max_ious, matched_gt_idx = ious.max(dim=1)

            labels = torch.full((N,), -1, dtype=torch.int64, device=proposals.device)
            labels[max_ious < self.iou_threshold_neg] = 0
            labels[max_ious >= self.iou_threshold_pos] = 1

            pos_inds = torch.nonzero(labels == 1).squeeze(1)
            if pos_inds.numel() == 0:
                continue

            matched_boxes = proposals[pos_inds]  # (n_pos, 4)
            pred_pos_masks = pred_masks[pos_inds]  # (n_pos, H, W)
            matched_gt_masks = masks[matched_gt_idx[pos_inds]]  # (n_pos, H_gt, W_gt)

            # Resize GT mask to (H, W)
            resized_gt_masks = self.resize_masks_to_proposals(matched_gt_masks, matched_boxes, (H, W))  # (n_pos, H, W)

            loss = self.dice_loss(pred_pos_masks, resized_gt_masks)
            total_loss += loss

        return total_loss / B

class InstanceSegmentationLoss_stage2:
    def __init__(self, iou_threshold_pos=0.5, iou_threshold_neg=0.3):
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg

    def compute_iou(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter

        return inter / (union + 1e-6)

    def extract_gt_masks(self, gt_mask, gt_ids):
        return torch.stack([(gt_mask == gid).float() for gid in gt_ids], dim=0)

    def resize_masks_to_proposals(self, masks, boxes, size):
        M, H, W = masks.shape
        masks = masks.unsqueeze(1)
        idx = torch.arange(boxes.size(0), device=boxes.device).float()
        rois = torch.cat([idx[:, None], boxes], dim=1)  # (N, 5)
        return roi_align(masks.expand(boxes.size(0), -1, H, W), rois, size).squeeze(1)

    def dice_loss(self, pred, target):
        pred = pred.sigmoid()
        smooth = 1.0
        num = 2 * (pred * target).sum(dim=(1, 2)) + smooth
        den = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth
        loss = 1 - num / den
        return loss.mean()
    def coco_to_xyxy(self,boxes):
        # boxes: Tensor of shape (N, 4), format: [x, y, w, h]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)

    def __call__(self, all_proposals, instance_mask, class_logits, gt_boxes, gt_mask):
        """
        all_proposals: (B, N, 4)
        instance_mask: (B, N, H, W)
        class_logits: (B, N, 2)
        gt_boxes: list of (Mi, 4)
        gt_mask: (B, H_img, W_img)
        """
        B, N, H, W = instance_mask.shape
        total_dice_loss = torch.tensor(0.0, device=instance_mask.device)
        total_cls_loss = torch.tensor(0.0, device=instance_mask.device)
        gt_boxes=gt_boxes.to(gt_mask.device)
        
        for b in range(B):
            proposals = all_proposals[b]
            pred_masks = instance_mask[b]
            logits = class_logits[b]  # (N, 2)
            gt_boxes_b = gt_boxes[b]
            gt_mask_b = gt_mask[b].squeeze()
            
            if gt_boxes_b.numel() == 0:
                # 所有样本为负样本，标签为 0
                labels = torch.zeros(N, dtype=torch.long, device=proposals.device)
                cls_loss = F.cross_entropy(logits, labels)
                total_cls_loss += cls_loss
                continue
            # gt_boxes_b = self.coco_to_xyxy(gt_boxes_b)
            gt_boxes_b = resize_boxes(gt_boxes_b, 1024/256, 1024/256)

            gt_ids = torch.unique(gt_mask_b)
            gt_ids = gt_ids[gt_ids > 0]
            masks = self.extract_gt_masks(gt_mask_b, gt_ids.to(gt_mask_b.device))

            ious = self.compute_iou(proposals, gt_boxes_b)
            max_ious, matched_gt_idx = ious.max(dim=1)

            labels = torch.full((N,), -1, dtype=torch.long, device=proposals.device)
            labels[max_ious < self.iou_threshold_neg] = 0  # 背景
            labels[max_ious >= self.iou_threshold_pos] = 1  # 前景

            # 分类损失（忽略 -1 标签）
            valid_cls_inds = labels >= 0
            if valid_cls_inds.sum() > 0:
                cls_loss = F.cross_entropy(logits[valid_cls_inds], labels[valid_cls_inds])
                total_cls_loss += cls_loss
            else:
                total_cls_loss += 0.0

            # 掩膜 DiceLoss（只对正样本）
            pos_inds = torch.nonzero(labels == 1).squeeze(1)
            if pos_inds.numel() == 0:
                continue
            safe_inds = (matched_gt_idx[pos_inds] >= 0) & (matched_gt_idx[pos_inds] < masks.shape[0])
            pos_inds = pos_inds[safe_inds]
            
            matched_boxes = proposals[pos_inds]
            pred_pos_masks = pred_masks[pos_inds]
            
            matched_gt_masks = masks[matched_gt_idx[pos_inds]]

            resized_gt_masks = self.resize_masks_to_proposals(matched_gt_masks, matched_boxes, (H, W))

            dice_loss = self.dice_loss(pred_pos_masks, resized_gt_masks)
            total_dice_loss += dice_loss

        total_dice_loss = torch.tensor(total_dice_loss, device=gt_mask.device)

        # 取 batch 均值
        return {
            'dice_loss': total_dice_loss / B,
            'cls_loss': total_cls_loss / B,
            'total_loss': (total_dice_loss + total_cls_loss) / B
        }
    

def resize_boxes(boxes, scale_x, scale_y):
    """
    boxes: Tensor [N, 4] in xyxy format
    """
    boxes[:, 0] *= scale_x  # x1
    boxes[:, 1] *= scale_y  # y1
    boxes[:, 2] *= scale_x  # x2
    boxes[:, 3] *= scale_y  # y2
    return boxes

def decode_boxes(anchors, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    widths  = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x   = anchors[:, 0] + 0.5 * widths
    ctr_y   = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0] * weights[0]
    dy = deltas[:, 1] * weights[1]
    dw = deltas[:, 2] * weights[2]
    dh = deltas[:, 3] * weights[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = widths * torch.exp(dw)
    pred_h = heights * torch.exp(dh)

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
