# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import cv2
import random
# import segment_anything_training.modeling.loralib as loralib
import loralib
from typing import Dict, List, Tuple

from segment_anything_training.modeling.feature_vis import visualize_encoder_features,visualize_feature_map,visualize_instance_tokens,visualize_encoder_features_withInstanceID,compute_feature_structure_metrics,save_excel_imbedding_metrics
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import MaskDecoder_SAUGE,MaskDecoder_Fusion_MultiSegToken_V2
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder ,MaskDecoder_HQ_Edge, MaskDecoder_Fusion,MaskDecoder_Fusion_v2,MaskDecoder_Fusion_v3_2,MaskDecoder_Fusion_v3,MaskDecoder_Fusion_v4,MaskDecoder_Fusion_v5,MaskDecoder_Fusion_v6,MaskDecoder_Fusion_v7,MaskDecoder_Fusion_v8,MaskDecoder_Fusion_v9

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter,create_dataloaders_crop
from utils.loss_mask import loss_masks,mask_logits_to_instance_id,fast_mask_nms_batch
import utils.misc as misc
import json
from ana_utils.Seg_Edge_2_Instance import process_batch_watershed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "sam-hq-param" / "sam_hq_vit_b.pth"
DEFAULT_MASKDECODER_B_CHECKPOINT = PROJECT_ROOT / "sam-hq-param" / "sam_vit_b_maskdecoder.pth"


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

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
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
                        iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h",'vit_h_MultiSpectral']
        
        checkpoint_dict = {"vit_b":str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           "vit_l":str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           'vit_h':str(DEFAULT_MASKDECODER_B_CHECKPOINT),
                           'vit_h_MultiSpectral':str(DEFAULT_MASKDECODER_B_CHECKPOINT)} # replace trained 
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False

        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280,"vit_h_MultiSpectral":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

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
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted hq masks
        """
        
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

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

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
        
        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks( # 在SAM解码器基础上加了一个HQFeature
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
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

        upscaled_embedding_sam = self.output_upscaling(src) #TODO： 为什么要再编码一次
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM Test', add_help=False)

    # ======================
    # pipeline 强控参数
    # ======================
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)

    parser.add_argument("--restore_model", type=str, default=None)
    parser.add_argument("--restore_model_multispectral", type=str, default=None)
    parser.add_argument("--pretrained_multispectral_encoder", type=str, default=None)

    # ======================
    # SAM / backbone
    # ======================
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_SAM_CHECKPOINT)
    )

    parser.add_argument("--model-type", type=str, default="vit_b_MultiSpectral")
    parser.add_argument("--decoder-type", type=str, default="hq_instance")
    parser.add_argument("--use_lora", type=str2bool, default=True, help="whether to use lora finetune , if set to False,  all vit parameters will train")
    # ======================
    # 数据 & 输入
    # ======================
    parser.add_argument("--data_type", type=str, default="MultiSpectral")
    parser.add_argument("--data_type_input", type=str, default="MultiSpectral")
    parser.add_argument("--seasons",type=str,default="full_year") # spring summer autumn winter
    parser.add_argument("--input_size", type=list, default=[256, 256])
    parser.add_argument("--batch_size_test", type=int, default=1)


    # ======================
    # 模型结构（test 需保持一致）
    # ======================
    parser.add_argument("--multispectral-encoder", type=str, default="CBAM")

    parser.add_argument("--maskdecoder_v", type=int, default=3)
    parser.add_argument("--maskdecoder_v_v", type=int, default=2)
    parser.add_argument("--fusion_v", type=int, default=3)

    parser.add_argument("--instance_decoder", type=int, default=3)
    parser.add_argument("--num_instance_token", type=int, default=100)

    parser.add_argument("--use_multiscale_feature", type=str2bool, default=False)
    parser.add_argument("--use_semantic_supervise_for_instance", type=str2bool, default=False)
    parser.add_argument("--use_instance_diversity_loss", type=str2bool, default=False)
    parser.add_argument("--use_fusion", type=str2bool, default=True)

    # ======================
    # 推理 / 可视化
    # ======================
    parser.add_argument("--withPostProcess", type=str2bool, default=True)
    parser.add_argument("--mulitspectral_visualize", type=str2bool, default=True)
    parser.add_argument("--tnse_visualize", type=str2bool, default=False)
    parser.add_argument("--cal_feature_structure", type=str2bool, default=False)

    # ======================
    # 分布式
    # ======================
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--find_unused_params", action="store_true")

    parser.add_argument("--eval", action="store_true")

    return parser.parse_args()

def save_args_json(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=2)

def main(net, test_datasets, args):
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    save_args_json(args, args.output + '/args.json')

    if not args.eval:
        print("--- create testing dataloader ---")
        test_dataloaders, test_datasets = create_dataloaders_crop(test_datasets,
                                                        batch_size = args.batch_size_test,
                                                        training = False,
                                                        data_type=args.data_type
                                                        )
        print(len(test_dataloaders), " test dataloaders created")
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net = net.to(device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    else:
        net = net.to(device)
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net_without_ddp = net.module
    else:
        net_without_ddp = net


    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint,multispectral_encoder_type=args.multispectral_encoder,checkpoint_multispectral=args.restore_model_multispectral,in_channels=args.in_channels)
    _ = sam.to(device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    else:
        sam = sam.to(device=device)
    if hasattr(net_without_ddp, 'prompt_encoder'):
            net_without_ddp.prompt_encoder = sam.prompt_encoder
    print("restore model from:", args.restore_model)
    if torch.cuda.is_available():
        net_without_ddp.load_state_dict(torch.load(args.restore_model))
    else:
        net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))

    evaluate(args, net, sam, test_dataloaders)
    if args.withPostProcess:
        seg = args.output + '/test_hq_mask'
        edge = args.output + '/test_hq_edge'
        out_bin = args.output + '/parcels/bin'
        out_color = args.output + '/parcels/color_result'
        process_batch_watershed(seg_dir=seg,edge_dir=edge,save_bin_dir=out_bin,save_rgb_dir=out_color)






def compute_iou(preds, target,threshold=128):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i],threshold=threshold)
    return iou / len(preds)

def compute_boundary_iou(preds, target,threshold=128):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i],threshold=threshold)
    return iou / len(preds)

def evaluate(args, net, sam, valid_dataloaders, visualize=True):
    net.eval()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("testing...")
    test_stats = {}
    early_metrics = []
    late_metrics = []
    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('test_dataloader len:', len(valid_dataloader))

        progress_bar = tqdm(enumerate(metric_logger.log_every(valid_dataloader, 1000)),
                    desc="testing", unit="batch", total=len(valid_dataloader))
        
        for step,data_val in progress_bar:
            inputs, mask_semantic, mask_edge, mask_distance ,image_name = data_val['image']*255, data_val['mask_semantic'].float(), data_val['mask_edge'].float(), data_val['mask_distance'].float() , data_val['image_name']

            if torch.cuda.is_available():
                inputs_val = inputs.to(device)
                mask_semantic_val = mask_semantic.to(device)
                mask_edge_val = mask_edge.to(device)
                mask_distance_val = mask_distance.to(device)
                # labels_val = mask_semantic.cuda()
                # labels_val = labels_val.cuda()
                # labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(mask_semantic_val[:,0,:,:],threshold=0.5)
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)

                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )
            if args.decoder_type == 'hq_edge':
                masks_hq ,masks_edge = masks_hq
            elif args.decoder_type == 'hq_instance':
                if len(masks_hq) == 3:
                    masks_hq ,masks_edge, masks_instance = masks_hq
                    instance_token_afer_fusion = None
                elif len(masks_hq) == 4:
                    masks_hq ,masks_edge, masks_instance, instance_token_afer_fusion = masks_hq
            if args.cal_feature_structure:
                late_res = compute_feature_structure_metrics(encoder_embedding,masks_hq,data_val['image_name'][-1])
                early_res = compute_feature_structure_metrics(interm_embeddings[0].permute(0,3,1,2),masks_hq,data_val['image_name'][-1])
                early_metrics.append(early_res)
                late_metrics.append(late_res)
            if args.tnse_visualize:
                # if data_val['image_name'][-1] in ['ES_496_S2_10m_256.tif','FR_19117_S2_10m_256.tif','FR_21172_S2_10m_256.tif','FR_57611_S2_10m_256.tif']:
                    # continue
                # if data_val['image_name'][-1] not in ['AT_3950_S2_10m_256.tif']:
                #     continue
                out_path = '/mnt/disk3/har/Param/Cropland/A_ViT_FeatureVis/AI4Boundary/BaseModel_clp'
                visualize_feature_map(encoder_embedding,out_path,data_val['image_name'][-1])
                # tnse 
                masks_hq_sigmoid = torch.sigmoid(masks_hq)
                masks_bin = (masks_hq_sigmoid > 0.5).float()
                masks_bin = masks_bin.squeeze(1)

                masks_instance_gt = data_val['mask_instance']
                # visualize_encoder_features(encoder_embedding,masks_bin,out_path,data_val['image_name'])
                visualize_encoder_features_withInstanceID(encoder_embedding,masks_instance_gt,out_path,data_val['image_name'])
                for i in range(len(interm_embeddings)):
                    interm_name = ['interm_'+str(i)+'_'+item for item in data_val['image_name']]
                    visualize_encoder_features_withInstanceID(interm_embeddings[i].permute(0,3,1,2),masks_instance_gt,out_path,interm_name)

            iou = compute_iou(masks_hq,mask_semantic_val,threshold=0.5)
            boundary_iou = compute_boundary_iou(masks_hq,mask_semantic_val,threshold=0.5)

            loss_mask, loss_dice = loss_masks(masks_hq, mask_semantic_val, len(masks_hq))
            loss = loss_mask + loss_dice
            # if args.mulitspectral_visualize:
            #     for b_i in range(batch_len):
            #         image_3d = batched_output[b_i]['multispectral_in_3d']
            #         image_3d = image_3d.squeeze().permute(1, 2, 0).cpu().numpy()
            #         # 归一化到0-255
            #         image_3d = (image_3d - image_3d.min()) / (image_3d.max() - image_3d.min()) * 255
            #         # resize to 256
            #         image_3d = cv2.resize(image_3d, (256, 256), interpolation=cv2.INTER_CUBIC)
            #         os.makedirs(os.path.join(args.output, 'a_multispectral_vis'), exist_ok=True)
            #         cv2.imwrite(os.path.join(args.output, 'a_multispectral_vis', image_name[b_i].split('.')[0]+'.png'), image_3d)
                #continue #TODO Delte
                # pass # a_multispectral_vis
            
            # if step == 100:
            #     save_excel_imbedding_metrics(early_metrics, late_metrics,os.path.join(args.output,'feature_structure_metrics.xlsx'))
            #     break
            save_logits(masks_hq,args.output,image_name,'test_hq_mask')
            if args.decoder_type == 'hq_edge':
                save_logits(masks_edge,args.output,image_name,'test_hq_edge')
            if args.decoder_type == 'hq_instance':
                save_logits(masks_edge,args.output,image_name,'test_hq_edge')
                masks_instance_sigmoid = torch.sigmoid(masks_instance)

                masks_filtered = fast_mask_nms_batch(masks_instance_sigmoid, scores=None)
                red_instance_ids = mask_logits_to_instance_id(masks_filtered,if_logits=False)
                save_instance(red_instance_ids,args.output,image_name,'test_hq_instance')
            
            
            loss_dict = {"loss_value": loss, "loss_mask": loss_mask, "loss_dice":loss_dice,"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}

            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
            progress_bar.set_postfix({
                "Loss": f"{loss:.4f}",
                "val_iou": f"{loss_dict_reduced['val_iou_'+str(k)].item():.4f}",
                "val_boundary_iou": f"{loss_dict_reduced['val_boundary_iou_'+str(k)].item():.4f}"
            })
            # 可选：每 100 个 batch 打印一次详细日志，不打乱进度条
            if step % 2 == 0:
                tqdm.write(f"[Iter {step}]  val_iou: {loss_dict_reduced['val_iou_'+str(k)].item():.4f}, val_boundary_iou: {loss_dict_reduced['val_boundary_iou_'+str(k)].item():.4f}")
            # if step==5:
            #     break
        if args.cal_feature_structure:
            save_excel_imbedding_metrics(early_metrics, late_metrics,os.path.join(args.output,'feature_structure_metrics.xlsx'))
            
        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
    return test_stats
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
def save_logits(logits, save_path, name, data_type):
# loss_dict = {}
    for b_i in range(len(logits)):
        b_image_name = name[b_i].split('.')[0]
        b_masks_logits = logits[b_i].squeeze(0)
        b_masks_sigmoid = torch.sigmoid(b_masks_logits)
        b_masks_hq_mask = (b_masks_sigmoid>0.5).int().squeeze(0)
        # b_masks_hq_mask = (masks_hq>0)[b_i].int().squeeze(0)
        os.makedirs(os.path.join(save_path, data_type), exist_ok=True)
        os.makedirs(os.path.join(save_path, data_type+'_logits'), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, data_type , b_image_name+'.png'), b_masks_hq_mask.cpu().numpy()*255)
        cv2.imwrite(os.path.join(save_path, data_type +'_logits', b_image_name+'.png'), b_masks_logits.detach().cpu().numpy())
def save_instance(instance, save_path, name, data_type):
    """
    保存实例分割结果
    - instance : 张量 [b, h, w]，每个像素值为实例 ID
    - save_path: 保存根路径
    - name     : 批次中每个图像的名称列表
    - data_type: 数据类型标记（如 train/val）
    """
    # 创建保存目录
    os.makedirs(os.path.join(save_path, data_type), exist_ok=True)
    os.makedirs(os.path.join(save_path, f"{data_type}_rgb"), exist_ok=True)
    
    # 生成固定颜色的调色板（支持最多 65535 个实例）
    np.random.seed(42)  # 固定随机种子保证颜色一致
    color_palette = np.random.randint(0, 256, (65536, 3), dtype=np.uint8)
    color_palette[0] = [0, 0, 0]  # 背景设为黑色

    for b_i in range(len(instance)):
        # 提取当前图像的实例 ID 矩阵
        b_image_name = name[b_i].split('.')[0]
        instance_ids = instance[b_i].cpu().detach().numpy().astype(np.uint16)
        
        # --- 保存实例 ID 矩阵（16 位 PNG）---
        id_save_path = os.path.join(save_path, data_type, b_image_name+'.png')
        cv2.imwrite(id_save_path, instance_ids)  # 直接保存 uint16 矩阵

        # --- 生成 RGB 可视化 ---
        # 将实例 ID 映射到颜色表
        rgb_image = color_palette[instance_ids]
        # OpenCV 需要 BGR 格式
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 保存 RGB 图像
        rgb_save_path = os.path.join(save_path, f"{data_type}_rgb", b_image_name+'.png')
        cv2.imwrite(rgb_save_path, bgr_image)




# def compute_iou_2(mask1, mask2):
#     """ 计算两个二值 mask 的 IoU """
#     intersection = (mask1 & mask2).sum().float()
#     union = (mask1 | mask2).sum().float()
#     return intersection / (union + 1e-6)


if __name__ == "__main__":

    #              "gt_ext": ".png"}
    # train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    # valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 

    
    # valid_datasets = [dataset_ai4boundaries_val]
    args = get_args_parser()
    if args.dataset == 'AI4Boundaries':
        dataset_ai4boundaries = {"name": "AI4Boundaries",
                    "im_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/test/MultiSpectral",
                    "gt_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/mask/test",
                    "im_ext": ".tif",
                    "gt_ext": ".png"}
    elif args.dataset == 'S4A':
        # parser.add_argument("--seasons",type=str,default="full_year") # spring summer autumn winter
        if args.seasons == "full_year":
            dataset_ai4boundaries = {"name": "S4A",
                    "im_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/test/MultiSpectral",
                    "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/test",
                    "im_ext": ".tif",}
        elif args.seasons == 'spring':
            dataset_ai4boundaries = {"name": "S4A",
                        "im_dir": "/home/huar/DataSet/S4A/Process_Data/Season_TimeSeries/test_output/MultiSpectral/spring",
                        "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/test",
                        "im_ext": ".tif",
                        "gt_ext": ".png"}
        elif args.seasons == 'summer':
                dataset_ai4boundaries = {"name": "S4A",
                            "im_dir": "/home/huar/DataSet/S4A/Process_Data/Season_TimeSeries/test_output/MultiSpectral/summer",
                            "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/test",
                            "im_ext": ".tif",
                            "gt_ext": ".png"}
        elif args.seasons == 'autumn':
                dataset_ai4boundaries = {"name": "S4A",
                            "im_dir": "/home/huar/DataSet/S4A/Process_Data/Season_TimeSeries/test_output/MultiSpectral/autumn",
                            "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/test",
                            "im_ext": ".tif",
                            "gt_ext": ".png"}
        elif args.seasons == 'winter':
                dataset_ai4boundaries = {"name": "S4A",
                            "im_dir": "/home/huar/DataSet/S4A/Process_Data/Season_TimeSeries/test_output/MultiSpectral/winter",
                            "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/test",
                            "im_ext": ".tif",
                            "gt_ext": ".png"}
        
        
    test_datasets = [dataset_ai4boundaries]
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, 'test_hq_mask')):
        os.makedirs(os.path.join(args.output, 'test_hq_mask'))
    if not os.path.exists(os.path.join(args.output, 'test_hq_mask_logits')):
        os.makedirs(os.path.join(args.output, 'test_hq_mask_logits'))
    if args.decoder_type == 'hq':
        net = MaskDecoderHQ(args.model_type) 
    elif args.decoder_type == 'hq_edge' :
        net = MaskDecoder_HQ_Edge(args.model_type)
    elif args.decoder_type == 'hq_instance' :
        if args.maskdecoder_v == 1:
            net = MaskDecoder_Fusion(args.model_type,args.fusion_v)
        elif args.maskdecoder_v == 2:
            net = MaskDecoder_Fusion_v3_2(args.model_type,args.fusion_v,args.instance_decoder) # fuison 2; dyConv 
        elif args.maskdecoder_v == 3:
            net = MaskDecoder_Fusion_v3_2(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature,args.use_semantic_supervise_for_instance)
        elif args.maskdecoder_v == 4:
            net = MaskDecoder_Fusion_v4(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature) 
        elif args.maskdecoder_v == 5:
            net = MaskDecoder_Fusion_v5(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature) 
        elif args.maskdecoder_v == 6:
            net = MaskDecoder_Fusion_v6(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature) 
        elif args.maskdecoder_v == 7:
            net = MaskDecoder_Fusion_v7(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature) 
        elif args.maskdecoder_v == 8:
            net = MaskDecoder_Fusion_v8(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature) 
        elif args.maskdecoder_v == 9:
            net = MaskDecoder_Fusion_v9(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature,args.use_instance_diversity_loss) 
        elif args.maskdecoder_v == 10:
            net = MaskDecoder_SAUGE(args.model_type)
        elif args.maskdecoder_v == 11:
            net = MaskDecoder_Fusion_MultiSegToken_V2(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature,args.use_instance_diversity_loss)
    main(net, test_datasets, args)
