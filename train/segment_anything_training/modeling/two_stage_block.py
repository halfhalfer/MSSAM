
import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms,RoIAlign
from torch.autograd import Variable,Function
# Copy from https://github.com/multimodallearning/pytorch-mask-rcnn.git
############################################################
#  CropAndResize Function
############################################################
class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)

        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)
        else:
            _backend.crop_and_resize_forward(
                image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width, crops)

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)

        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        else:
            _backend.crop_and_resize_backward(
                grad_outputs, boxes, box_ind, grad_image
            )
        return grad_image, None, None

class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)

############################################################
#  Proposal Layer
############################################################
def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2

def pth_nms(dets, thresh):
  """
  dets has to be a tensor
  """
  if not dets.is_cuda:
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]
  else:
    x1 = dets[:, 1]
    y1 = dets[:, 0]
    x2 = dets[:, 3]
    y2 = dets[:, 2]
    scores = dets[:, 4]

    dets_temp = torch.FloatTensor(dets.size()).cuda()
    dets_temp[:, 0] = dets[:, 1]
    dets_temp[:, 1] = dets[:, 0]
    dets_temp[:, 2] = dets[:, 3]
    dets_temp[:, 3] = dets[:, 2]
    dets_temp[:, 4] = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.cpu().numpy().argsort()[::-1])).long().cuda()

    dets = dets[order].contiguous()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    # keep = torch.cuda.LongTensor(dets.size(0))
    # num_out = torch.cuda.LongTensor(1)
    nms.gpu_nms(keep, num_out, dets_temp, thresh)

    return order[keep[:num_out[0]].cuda()].contiguous()
def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :] # TODO: Support batch size > 1 ff.
    anchors = anchors[order.data, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2,5)


    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix  = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]),requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled

############################################################
#  Region Proposal Network
############################################################
class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
    
class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]

class RegionProposalNet(nn.Module):
    def __init__(self, in_channels=256, anchors_per_location=9, anchor_stride=1,
                 base_size=16, ratios=[0.5, 1, 2], scales=[0.5, 1, 2], feat_stride=4):
        super(RegionProposalNet, self).__init__()
        self.anchor_stride = anchor_stride
        self.anchors_per_location = anchors_per_location
        self.base_size = base_size
        self.ratios = ratios
        self.scales = scales
        self.feat_stride = feat_stride  # 对应特征图相对于原图的缩放比例

        self.shared = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1)
        self.bbox_pred = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat, image_size):
        B, C, H, W = feat.shape

        # --- Stage 1: 得到 logits 和回归偏移 ---
        x = F.relu(self.shared(feat))
        logits = self.cls_logits(x)  # (B, 2*A, H, W)
        bbox_deltas = self.bbox_pred(x)  # (B, 4*A, H, W)

        logits = logits.permute(0, 2, 3, 1).reshape(B, -1, 2)  # (B, H*W*A, 2)
        probs = self.softmax(logits)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(B, -1, 4)  # (B, H*W*A, 4)

        # --- Stage 2: 生成 anchors ---
        anchors = generate_anchors(
            base_size=self.base_size,
            scales=self.scales,
            ratios=self.ratios,
            feat_h=H,
            feat_w=W,
            stride=self.feat_stride
        ).to(feat.device)  # shape: (H*W*A, 4)

        # --- Stage 3: 对每个 batch 生成 proposals ---
        proposals_all = []
        for i in range(B):
            proposals, _ = generate_proposals(
                probs[i], bbox_deltas[i], anchors, top_n=1000, nms_thresh=0.7
            )
            # 限制 proposals 在图像范围内
            proposals[:, [0, 2]] = proposals[:, [0, 2]].clamp(0, image_size[1])
            proposals[:, [1, 3]] = proposals[:, [1, 3]].clamp(0, image_size[0])
            proposals_all.append(proposals)

        return proposals_all  # List[Tensor] of shape (N, 4) for each image in batch

def generate_anchors(base_size, scales, ratios, feat_h, feat_w, stride):
    """
    Returns anchors of shape (A, 4): (x1, y1, x2, y2)
    """
    anchors = []
    for y in range(feat_h):
        for x in range(feat_w):
            center_x = x * stride + stride // 2
            center_y = y * stride + stride // 2
            for scale in scales:
                for ratio in ratios:
                    area = base_size * base_size * scale
                    w = math.sqrt(area / ratio)
                    h = w * ratio
                    anchors.append([
                        center_x - w / 2,
                        center_y - h / 2,
                        center_x + w / 2,
                        center_y + h / 2,
                    ])
    return torch.tensor(anchors)

# --- Step 3: Apply deltas to anchors ---
def apply_deltas_to_anchors(anchors, deltas):
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas.unbind(dim=1)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.stack([
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ], dim=1)

    return pred_boxes

# --- Step 4: Proposal Layer ---
def generate_proposals(rpn_probs, rpn_bbox, anchors, top_n=2000, nms_thresh=0.7):
    scores = rpn_probs[:, 1]  # foreground prob
    proposals = apply_deltas_to_anchors(anchors, rpn_bbox)
    keep = nms(proposals, scores, nms_thresh)
    keep = keep[:top_n]
    return proposals[keep], scores[keep]

# --- Step 5: Full pipeline example ---
def rpn_pipeline(vit_feats):
    """
    vit_feats: tensor of shape (B, 256, 64, 64)
    """
    B, C, H, W = vit_feats.shape
    stride = 4  # 64x64特征图假设来源于256x256图像

    # 创建 anchor（假设在0~256图像上）
    anchors = generate_anchors(base_size=16, scales=[1.0, 2.0, 4.0], ratios=[0.5, 1.0, 2.0], feat_h=H, feat_w=W, stride=stride).to(vit_feats.device)

    rpn = RPN(in_channels=256, anchors_per_location=9).to(vit_feats.device)
    logits, probs, bbox_deltas = rpn(vit_feats)

    all_proposals = []
    for b in range(B):
        proposals, _ = generate_proposals(probs[b], bbox_deltas[b], anchors)
        all_proposals.append(proposals)

    return all_proposals  # 每个图像的 proposal (N, 4)

# --- Step 6: RoIAlign 提取 proposal 区域特征 ---
def extract_roi_features(feature_map, proposals, output_size=(7, 7)):
    """
    feature_map: (B, C, H, W)
    proposals: List of tensors, each [N, 4] in image coords
    """
    roi_align = RoIAlign(output_size, spatial_scale=1/4, sampling_ratio=2, aligned=True)

    roi_feats = []
    for i, prop in enumerate(proposals):
        # 拼接 batch 索引
        rois = torch.cat([torch.full((prop.shape[0], 1), i, device=prop.device), prop], dim=1)  # (N, 5)
        feats = roi_align(feature_map, rois)
        roi_feats.append(feats)
    return roi_feats  # List of (N, C, 7, 7)

class RoIFeatureExtractor(nn.Module):
    def __init__(self, feat_channels=256, pool_size=7):
        super().__init__()
        self.rpn = RegionProposalNet(in_channels=feat_channels)
        self.roi_align = RoIAlign(output_size=pool_size, spatial_scale=1/4, sampling_ratio=2)
        self.roi_mlp = nn.Sequential(
            nn.Linear(feat_channels * pool_size * pool_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )

    def forward(self, feat, image_size, batch_size):
        proposals = self.rpn(feat, image_size, batch_size)
        feat_ = feat.flatten(0, 1)  # Merge batch and token dimensions
        pooled = self.roi_align(feat_, proposals)
        roi_feat = self.roi_mlp(pooled.flatten(1))
        return roi_feat, proposals
    
class BoxClassifier(nn.Module):
    def __init__(self, in_channels=256, pool_size=(32, 32), num_classes=2):
        super().__init__()
        self.pool_size = pool_size
        self.fc1 = nn.Linear(in_channels * pool_size[0] * pool_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)  # 这里是 2 分类

    def forward(self, roi_feats):  # roi_feats: [B, N, C, H, W]
        B, N, C, H, W = roi_feats.shape
        x = roi_feats.view(B * N, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_logits = self.cls_score(x)  # [B*N, 2]
        cls_logits = cls_logits.view(B, N, 2)
        return cls_logits