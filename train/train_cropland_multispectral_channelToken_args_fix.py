# Modified from HQ-SAM

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

import pandas as pd
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import cv2
import random
import json
import shutil
import loralib
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry_channelToken
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import MaskDecoder_Fusion_v3_2
from segment_anything_training.modeling.feature_vis import visualize_encoder_features,visualize_feature_map,visualize_instance_tokens,visualize_encoder_features_withInstanceID,compute_feature_structure_metrics,save_excel_imbedding_metrics

from utils.dataloader import create_dataloaders_crop
from utils.loss_mask import loss_masks,cross_entropy_loss_RCF,cross_entropy_loss_RCF_wDice,loss_masks_full,EdgeLossAutoWeight,EdgeLossAutoWeight_V2,EdgeLossRefine,EdgeLossRefine_V2,InstanceSegmentationLoss,mask_logits_to_instance_id,InstanceSegmentationLoss_v2,InstanceSegmentationLoss_v3,fast_mask_nms_batch,instance_proxy_loss #v1 不带class_logits / v2 带上class_logits和其他限制性损失/ v3 带上class_logits
from utils.loss_reconstructed import mdms_loss
# from utils.loss_feature_struct import compute_intra_inter_class_losses
from utils.loss_feature_struct_V2 import compute_intra_inter_class_losses
import utils.misc as misc
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "sam-hq-param" / "sam_hq_vit_b.pth"

def str2bool(v):
    return v.lower() in ("true", "1", "yes")
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


def validate_mainline_args(args):
    if args.decoder_type != "hq_instance":
        raise ValueError("This streamlined script only supports decoder_type=hq_instance")
    if args.maskdecoder_v != 3:
        raise ValueError("This streamlined script only supports maskdecoder_v=3")
    if args.maskdecoder_v_v != 2:
        raise ValueError("This streamlined script only supports maskdecoder_v_v=2")
    if args.fusion_v != 3:
        raise ValueError("This streamlined script only supports fusion_v=3")
    if not args.use_fusion:
        raise ValueError("This streamlined script only supports use_fusion=True")


def build_edge_loss_fn(args):
    if args.edgeloss_v == 1:
        return EdgeLossAutoWeight(mode=args.edgeloss_type)
    if args.edgeloss_v == 2:
        return EdgeLossAutoWeight_V2(mode=args.edgeloss_type)
    if args.edgeloss_v == 3:
        return EdgeLossRefine()
    if args.edgeloss_v == 4:
        return EdgeLossRefine_V2()
    if args.edgeloss_v == 5:
        return cross_entropy_loss_RCF_wDice
    if args.edgeloss_v == 6:
        return cross_entropy_loss_RCF
    raise ValueError(f"Unsupported edgeloss_v={args.edgeloss_v}")


def build_instance_loss_fn(args):
    if args.instance_loss_v == 1:
        return InstanceSegmentationLoss()
    if args.instance_loss_v == 2:
        return InstanceSegmentationLoss_v2(
            semantic_supervise=args.use_semantic_supervise_for_instance
        )
    if args.instance_loss_v == 3:
        return InstanceSegmentationLoss_v3()
    raise ValueError(f"Unsupported instance_loss_v={args.instance_loss_v}")


def build_main_decoder(args):
    decoder_kwargs = dict(
        model_type=args.model_type,
        fusion_v=args.fusion_v,
        instacnDecoder=args.instance_decoder,
        num_instance_tokens=args.num_instance_token,
        use_multiscale_imagefeat=args.use_multiscale_feature,
        use_cluster_token=args.use_semantic_supervise_for_instance,
        use_fusion=args.use_fusion,
    )
    if args.struct_loss_in_finalEmbedding:
        if not args.use_feature_struct_loss:
            raise ValueError(
                "use_feature_struct_loss should be True when struct_loss_in_finalEmbedding is True"
            )
        return MaskDecoder_Fusion_v3_2(**decoder_kwargs)
    return MaskDecoder_Fusion_v3_2(**decoder_kwargs, return_final_embed=False)


def unpack_main_decoder_outputs(outputs, expect_final_embedding):
    if expect_final_embedding:
        if len(outputs) != 4:
            raise ValueError("Expected 4 decoder outputs when final embedding is enabled")
        masks_hq, masks_edge, masks_instance, upscaled_embedding_ours = outputs
        seg_tokens = None
    else:
        if len(outputs) == 3:
            masks_hq, masks_edge, masks_instance = outputs
            seg_tokens = None
        elif len(outputs) == 4:
            masks_hq, masks_edge, masks_instance, seg_tokens = outputs
        else:
            raise ValueError("Unexpected decoder output structure")
        upscaled_embedding_ours = None
    return masks_hq, masks_edge, masks_instance, upscaled_embedding_ours, seg_tokens

def get_args_parser():

    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    # ======================
    # pipeline 强控参数（必须显式给）
    # ======================
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--pathEmbed_v", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--use_channelToken", type=str2bool, required=True)
    parser.add_argument("--use_orth_loss", type=str2bool, required=True)

    parser.add_argument("--max_epoch_num", type=int, required=True)
    parser.add_argument("--debug_trainstep", type=int, required=True)
    parser.add_argument("--debug_valstep", type=int, required=True)

    parser.add_argument("--gpu", type=int, required=True)
    # ======================
    # 数据类型
    # ======================
    parser.add_argument("--data_type_input", type=str, default="MultiSpectral", help="the data type of input image, RGB or MultiSpectral;Set RGB,will load MultiSpectral data first and transform to RGB")
    parser.add_argument("--data_type", type=str, default="MultiSpectral")
    parser.add_argument(
        "--continue_sample_list",
        type=str,
        default=None,
        help="optional sample list for partial training / ablation"
    )
    # ======================
    # 损失
    # ======================
    parser.add_argument("--edgeloss_v", type=int, default=1, help="the version of edgeloss")
    parser.add_argument("--edgeloss-type", type=str, default="bce+dice",help="Edge loss type" )
    parser.add_argument("--segloss-type", type=str, default="Point",help="whether to use Full loss or Point loss")
    # ======================
    # 训练过程配置
    # ======================
    parser.add_argument('--model_save_fre', default=5, type=int)
    parser.add_argument('--visualize', type=str2bool, default=False, help='whether to visualize the evaluation results')

    # ======================
    # SAM / checkpoint（允许 default）
    # ======================
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_SAM_CHECKPOINT)
    )
    parser.add_argument("--restore_model", type=str, default=None)
    parser.add_argument("--checkpoint_vit", type=str, default=None)

    # ======================
    # 模型结构参数（默认即可）
    # ======================
    parser.add_argument("--use_lora", type=str2bool, default=True, help="whether to use lora finetune , if set to False,  all vit parameters will train")
    parser.add_argument("--decoder-type", type=str, default="hq_instance")
    parser.add_argument("--model-type", type=str, default="channelToken")

    parser.add_argument("--maskdecoder_v", type=int, default=3)
    parser.add_argument("--fusion_v", type=int, default=3)
    parser.add_argument("--maskdecoder_v_v", type=int, default=2)

    parser.add_argument("--instance_decoder", type=int, default=3)
    parser.add_argument("--instance_loss_v", type=int, default=2)
    parser.add_argument("--num_instance_token", type=int, default=100)

    parser.add_argument("--use_multiscale_feature", type=str2bool, default=False)
    parser.add_argument("--use_semantic_supervise_for_instance", type=str2bool, default=False)
    parser.add_argument("--use_instance_diversity_loss", type=str2bool, default=False)

    parser.add_argument("--use_fusion", type=str2bool, default=True)
    parser.add_argument("--use_instance_task", type=str2bool, default=True)

    parser.add_argument("--use_feature_struct_loss", type=str2bool, default=True)
    parser.add_argument("--struct_loss_in_finalEmbedding", type=str2bool, default=True)
    parser.add_argument("--use_sam_h_alig", type=str2bool,  default=False, help="whether to use ViT-H aligment loss")
    # ======================
    # 训练控制参数
    # ======================
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--pe_lr_scale", type=str2bool, default=False)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--lr_drop_epoch", type=int, default=20)

    parser.add_argument("--batch_size_train", type=int, default=1)
    parser.add_argument("--batch_size_valid", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=16)

    parser.add_argument("--debug", type=str2bool, default=True)

    # ======================
    # 分布式 & 其他
    # ======================
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--find_unused_params", action="store_true")

    parser.add_argument("--eval", action="store_true")

    return parser.parse_args()

def main(net,train_datasets, valid_datasets, test_datasets, args):
    validate_mainline_args(args)
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        # train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders_crop(train_datasets,
                                                        # my_transforms = [
                                                        #             RandomHFlip(),
                                                        #             LargeScaleJitter()
                                                        #             ],
                                                        batch_size = args.batch_size_train,
                                                        training = True,
                                                        data_type=args.data_type,
                                                        select_sample_list=args.continue_sample_list
                                                        )
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    # valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders_crop(valid_datasets,
                                                        #   my_transforms = [
                                                        #                 Resize(args.input_size)
                                                        #             ],
                                                        batch_size=args.batch_size_valid,
                                                        training=False,
                                                        data_type=args.data_type)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    print("--- create test dataloader ---")
    # valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    test_dataloaders, test_datasets = create_dataloaders_crop(test_datasets,
                                                        #   my_transforms = [
                                                        #                 Resize(args.input_size)
                                                        #             ],
                                                        batch_size=args.batch_size_valid,
                                                        training=False,
                                                        data_type=args.data_type)
    print(len(test_dataloaders), " test dataloaders created")

    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    else:
        net = net.to(device)
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net_without_ddp = net.module
    else:
        net_without_ddp = net

    loss_edge_fn = build_edge_loss_fn(args)
    loss_instance_fn = build_instance_loss_fn(args)
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        if args.model_type != 'channelToken':
            # sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint, multispectral_encoder_type='CBAM',checkpoint_multispectral=args.checkpoint_vit)
            sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint,use_lora=args.use_lora,checkpoints_lora=args.checkpoint_vit).to('cuda')
        else:
            sam = sam_model_registry_channelToken['vit_b_MultiSpectral'](checkpoint=args.checkpoint,checkpoint_vit=args.checkpoint_vit,use_lora=args.use_lora,use_orth_loss=args.use_orth_loss,in_channels=args.in_channels,args=args)
            
        sam = sam.to(device)
        if hasattr(net_without_ddp, 'prompt_encoder'):
            net_without_ddp.prompt_encoder = sam.prompt_encoder
        # if hasattr(net_without_ddp, 'transformer_2'):
        #     net_without_ddp.init_two_way_transformer(args.restore_model)
        # 先进行DDP包装
        if args.distributed:
            sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        # sam = sam_model_registry_channelToken[args.model_type](checkpoint=args.checkpoint,multispectral_encoder_type=args.multispectral_encoder)
        def get_sam_param(model):
            # 1. PatchEmbedding parameters
            patch_embedding = []
            lora = []
            for n,p in model.named_parameters():
                if 'patch_embed' in n:
                    patch_embedding.append(p)
                if "lora_A" in n or "lora_B" in n :
                    lora.append(p)
            #2. Lora parameters
            return patch_embedding,lora
        patch_embed, lora_parameters = get_sam_param(sam)

        if args.pe_lr_scale:
            param_groups = [
                {"params": patch_embed, "lr": 5e-4},
                {"params": lora_parameters, "lr": 2e-3},  # 或 2e-3
                {"params": net.parameters(),"lr":1e-3}
            ]
        else:
            param_groups = [
                {"params": patch_embed, "lr": 1e-3},
                {"params": lora_parameters, "lr": 2e-3},  # 或 2e-3
                {"params": net.parameters(),"lr":1e-3}
            ]

        if args.continue_sample_list is not None: # finetune lr
            optimizer = optim.AdamW(list(net.parameters())+list(sam.parameters()),lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            if args.use_lora:
                optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            else:
                optimizer = optim.AdamW(list(net.parameters())+list(sam.parameters()),lr=1e-3,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        
        lr_scheduler.last_epoch = args.start_epoch

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model),strict=False)
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
        if args.use_sam_h_alig:
            sam_h = sam_model_registry['vit_h'](checkpoint = '/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_h_4b8939.pth').to(device=device)
            # sam_h = sam_model_registry['vit_b'](checkpoint = '/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_b_01ec64.pth').to(device=device)
        else:
            sam_h = None
        train_accumgrad(
            args,
            net,
            optimizer,
            train_dataloaders,
            valid_dataloaders,
            lr_scheduler,
            loss_edge_fn,
            loss_instance_fn,
            sam,
            sam_h,
        )

        evaluate(args, net, sam, test_dataloaders, args.visualize,loss_edge_fn,loss_instance_fn,if_test=True)

    else:
        if args.model_type != 'channelToken':
            sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint,checkpoint_vit=args.checkpoint_vit,use_lora=args.use_lora,use_orth_loss=args.use_orth_loss,args=args)
        else:
            sam = sam_model_registry_channelToken['vit_b_MultiSpectral'](checkpoint=args.checkpoint,checkpoint_vit=args.checkpoint_vit,use_lora=args.use_lora,use_orth_loss=args.use_orth_loss,in_channels=args.in_channels,args=args)
        _ = sam.to(device=device)

        if hasattr(net_without_ddp, 'prompt_encoder'):
            net_without_ddp.prompt_encoder = sam.prompt_encoder

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        else:
            sam = sam.to(device=device)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
    
        evaluate(args, net, sam, valid_dataloaders, args.visualize,loss_edge_fn,loss_instance_fn)

def train_accumgrad(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler,loss_edge_fn=None,loss_instance_fn=None,sam=None,sam_h=None):

    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # copy train code
    shutil.copy(os.path.abspath(__file__), args.output)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    net.train()
    _ = net.to(device=device)
    
    if sam is None:
        sam = sam_model_registry_channelToken['vit_b_MultiSpectral'](checkpoint=args.checkpoint,checkpoint_vit=args.checkpoint_vit,use_lora=args.use_lora,use_orth_loss=args.use_orth_loss,in_channels=args.in_channels,args=args)
        _ = sam.to(device=device)
    
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        if misc.is_main_process():
            os.makedirs(args.output, exist_ok=True)
            # 创建损失日志文件并写入表头
            log_file = os.path.join(args.output, "training_log.csv")
            if not os.path.exists(log_file) or epoch == 0:
                with open(log_file, "w") as f:
                    f.write("epoch,lr,train_loss,loss_mask,loss_dice,val_loss,...\n")  # 根据实际验证指标补充列名

        sampler = train_dataloaders.batch_sampler.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        # train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        progress_bar = tqdm(enumerate(metric_logger.log_every(train_dataloaders, 1000)),
                    desc="Training", unit="batch", total=len(train_dataloaders))
        # for data in metric_logger.log_every(train_dataloaders,1000):
        best_val_loss = 1e5

        accum_steps = args.accumulation_steps  # 定义累积步数
        accum_count = 0  # 初始化计数器 

        feature_center = None
        for step,data in progress_bar:
            if args.dataset == 'AI4Boundaries_orth':
                inputs, mask_semantic, mask_edge, mask_distance,mask_instance = data['image'], data['mask_semantic'].float(), data['mask_edge'].float(), data['mask_distance'].float(), data['mask_instance']
            else:
                inputs, mask_semantic, mask_edge, mask_distance,mask_instance = data['image']*255, data['mask_semantic'].float(), data['mask_edge'].float(), data['mask_distance'].float(), data['mask_instance']
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                # labels = labels.cuda()
                mask_semantic = mask_semantic.to(device)
                mask_edge = mask_edge.to(device)
                mask_distance = mask_distance.to(device)
                mask_instance = mask_instance.to(device)
            if args.data_type_input == "RGB":
                if args.dataset == "S4A":
                    inputs = inputs[:,1:4,:,:]
                elif args.dataset == "AI4Boundaries":
                    inputs = inputs[:,:3,:,:]
            imgs = inputs.permute(0, 2, 3, 1)
            labels_256 = F.interpolate(mask_semantic, size=(256, 256), mode='bilinear')
            edge_256 = F.interpolate(mask_edge, size=(256, 256), mode='bilinear')
            instance_256 = F.interpolate(mask_instance, size=(256, 256), mode='nearest')

            # set prompt to None
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                # input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                input_image = torch.as_tensor(imgs[b_i], device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)
            if args.use_sam_h_alig:
                with torch.no_grad():
                    batched_input_h = []
                    dict_input_h = dict()
                    dict_input_h['image'] = input_image[:3,:,:]
                    dict_input_h['original_size'] = imgs[b_i].shape[:2]
                    batched_input_h.append(dict_input_h)
                    batched_output_h,interm_embeddings_h = sam_h(batched_input_h, multimask_output=False) if sam_h is not None else None
                    batch_len = len(batched_output_h)
                    encoder_embedding_h = torch.cat([batched_output_h[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            batched_output, interm_embeddings = sam(batched_input, multimask_output=False,)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            ortho_loss_channelTokens = [batched_output[i_l]['ortho_loss'] for i_l in range(batch_len)]
            ortho_loss_channelTokens = torch.stack(ortho_loss_channelTokens).mean()
            
            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )

            (
                masks_hq,
                masks_edge,
                masks_instance,
                upscaled_embedding_ours,
                seg_tokens,
            ) = unpack_main_decoder_outputs(
                masks_hq, args.use_feature_struct_loss and args.struct_loss_in_finalEmbedding
            )
            if hasattr(net,'_class_logits'):
                class_logits = net._class_logits
            else:
                class_logits = None
            if hasattr(net,'_mask_instance_clusters'):
                mask_instance_clusters = net._mask_instance_clusters
            else:
                mask_instance_clusters = None

            if seg_tokens is None and hasattr(net,'seg_tokens'):
                seg_tokens = net.seg_tokens
                
            if args.segloss_type == 'Full':
                loss_mask, loss_dice = loss_masks_full(masks_hq, labels_256)
            else:
                loss_mask, loss_dice = loss_masks(masks_hq, mask_semantic, len(masks_hq)) 
            
            if seg_tokens is not None:
                loss_seg_token_diversity = instance_proxy_loss(seg_tokens,sam.image_encoder.patch_embed.channel_embed.weight)
            else:
                loss_seg_token_diversity = torch.tensor(0.0).to(device=loss_mask.device)

            loss_edge = loss_edge_fn(masks_edge, edge_256)
            loss_instance = loss_instance_fn(
                masks_instance,
                class_logits,
                instance_256,
                mask_instance_clusters,
            )

            instance_token_afer_fusion = None
            if args.use_instance_diversity_loss and instance_token_afer_fusion is not None:
                loss_instance_diversity = instance_proxy_loss(
                    instance_token_afer_fusion, net.instance_token_proxy
                )
            else:
                loss_instance_diversity = torch.tensor(0.0).to(device=loss_instance.device)

            if args.use_feature_struct_loss:
                if args.struct_loss_in_finalEmbedding:
                    assert upscaled_embedding_ours is not None, "upscaled_embedding_ours is None"
                    feature_struct_losses = compute_intra_inter_class_losses(
                        encoder_embedding=upscaled_embedding_ours,
                        semantic_mask=labels_256,
                        num_classes=2,
                        last_class_center=feature_center,
                    )
                else:
                    feature_struct_losses = compute_intra_inter_class_losses(
                        encoder_embedding=encoder_embedding,
                        semantic_mask=labels_256,
                        num_classes=2,
                        last_class_center=feature_center,
                    )
                loss_intral, loss_c2c, loss_c2p, class_avg_features, updated_last_center = feature_struct_losses
                if updated_last_center is not None:
                    feature_center = updated_last_center.detach()
                else:
                    feature_center = class_avg_features.detach()
                struct_weight_list = [0.5, 0.5, 0.5]
                loss_struct = (
                    loss_intral * struct_weight_list[0]
                    + loss_c2c * struct_weight_list[1]
                    + loss_c2p * struct_weight_list[2]
                )
            else:
                loss_struct = torch.tensor(0.0).to(device=loss_instance.device)

            weight_list = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.5]
            if not args.use_instance_task:
                weight_list[3] = 0.0

            loss = (
                loss_mask * weight_list[0]
                + loss_dice * weight_list[1]
                + loss_edge * weight_list[2]
                + loss_instance * weight_list[3]
                + loss_instance_diversity * weight_list[4]
                + loss_seg_token_diversity * weight_list[5]
                + loss_struct
                + ortho_loss_channelTokens * weight_list[5]
            )

            if args.use_sam_h_alig:
                loss_h_alig = mdms_loss(encoder_embedding_h, encoder_embedding)
            else:
                loss_h_alig = torch.tensor(0.0).to(device=loss.device)
            loss += loss_h_alig * weight_list[6]
            loss_dict = {
                "loss": loss,
                "loss_mask": loss_mask,
                "loss_dice": loss_dice,
                "loss_edge": loss_edge,
                "loss_instance": loss_instance,
                "loss_instance_diversity": loss_instance_diversity,
                "loss_seg_token_diversity": loss_seg_token_diversity,
                "loss_h_alig": loss_h_alig,
                "loss_struct": loss_struct,
                "ortho_loss_channelTokens": ortho_loss_channelTokens,
            }

            # del batched_output, interm_embeddings, encoder_embedding, image_pe, sparse_embeddings, dense_embeddings
            # torch.cuda.empty_cache()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            # losses_reduced_scaled = sum(loss_dict_reduced.values())

            loss_value = loss_dict_reduced['loss'].item()
            
            loss = loss / accum_steps
            # 反向传播（梯度累积）
            loss.backward()
            # del batched_output, interm_embeddings, encoder_embedding, masks_hq
            # torch.cuda.empty_cache()
            accum_count += 1
            # for name, param in sam.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is None:
            #             print(f"❌ No gradient: {name}")
            #         else:
            #             print(
            #                 f"✅ Gradient OK: {name}, "
            #                 f"grad norm: {param.grad.norm().item():.6f}, "
            #                 f"param mean: {param.mean().item():.6f}"
            #             )
            # 累积到指定步数后更新参数
            if accum_count % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0  # 重置计数器

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            # 在 tqdm 上更新显示内容（只展示一个主指标）
            progress_bar.set_postfix({
                "Loss": f"{loss_value:.4f}",
                "Mask": f"{loss_dict_reduced['loss_mask'].item():.4f}",
                "Dice": f"{loss_dict_reduced['loss_dice'].item():.4f}"
            })
            # 可选：每 100 个 batch 打印一次详细日志，不打乱进度条
            if step % 2 == 0:
                if args.use_instance_diversity_loss:
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}, Edge: {loss_dict_reduced['loss_edge'].item():.4f}, instance: {loss_dict_reduced['loss_instance'].item():.4f},loss_ortho: {loss_dict_reduced['channelToken_diversity'].item():.4f}, loss_instance_diversity: {loss_dict_reduced['loss_instance_diversity'].item():.4f}, loss_feature_struct: {loss_dict_reduced['loss_struct'].item():.4f},ortho_proxy_loss: {loss_dict_reduced['ortho_loss_channelTokens'].item():.4f}")
                else:
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}, Edge: {loss_dict_reduced['loss_edge'].item():.4f}, instance: {loss_dict_reduced['loss_instance'].item():.4f}, loss_seg_token_diversity: {loss_dict_reduced['loss_seg_token_diversity'].item():.4f},loss_h_alig: {loss_dict_reduced['loss_h_alig'].item():.4f}, loss_feature_struct: {loss_dict_reduced['loss_struct'].item():.4f},ortho_proxy_loss: {loss_dict_reduced['ortho_loss_channelTokens'].item():.4f}")
            if step==args.debug_trainstep and args.debug:
                break
        
        if accum_count > 0:
            optimizer.step()
            optimizer.zero_grad()
        
        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, net, sam, valid_dataloaders,visualize=args.visualize,loss_edge_fn=loss_edge_fn,loss_instance_fn=loss_instance_fn,epoch=epoch)
        train_stats.update(test_stats)
        
        net.train()  
        sam.train()
        # if misc.is_main_process():
        log_file = os.path.join(args.output, "training_log.csv")
        epoch_log = {
            "Epoch": epoch,
            "LR": optimizer.param_groups[0]['lr'],
            "TrainLoss": train_stats.get("loss", -1),
            "TrainMaskLoss": train_stats.get("loss_mask", -1),
            "TrainDiceLoss": train_stats.get("loss_dice", -1),
            "TrainEdgeLoss": train_stats.get("loss_edge", -1),
            "TrainOrthoLoss": train_stats.get("ortho_loss_channelTokens", -1),
            "loss_feature_struct": train_stats.get("loss_struct", -1),
            "ValLoss": test_stats.get("loss", -1),
            "ValMaskLoss": test_stats.get("loss_mask", -1),
            "ValDiceLoss": test_stats.get("loss_dice", -1),
            "ValEdgeLoss": test_stats.get("loss_edge", -1),
            "ValOrthoLoss": test_stats.get("ortho_loss_channelTokens", -1),
            "loss_feature_struct": test_stats.get("loss_struct", -1),
        }
        df_epoch = pd.DataFrame([epoch_log])
        # 判断是否第一次写入（文件不存在）
        if not os.path.exists(log_file):
            df_epoch.to_csv(log_file, index=False)
        else:
            df_old_csv = pd.read_csv(log_file)
            df_combined_csv = pd.concat([df_old_csv, df_epoch], ignore_index=True)
            df_combined_csv.to_csv(log_file, index=False)

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                misc.save_on_master(net.module.state_dict(), args.output + model_name)
            else:
                misc.save_on_master(net.state_dict(), args.output + model_name)
            # save lora and multiSpectral parameters
            model_lora_multiSpectral = "/lora_multiSpectral_epoch_"+str(epoch)+".pth"
            if args.use_lora:
                save_sam_params(sam,args.output + model_lora_multiSpectral)
            else:
                save_sam_params(sam,args.output + model_lora_multiSpectral,save_backbone=True)
            
        val_loss = test_stats.get("loss", -1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = "/best_model_decoder.pth"
            # model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                misc.save_on_master(net.module.state_dict(), args.output + model_name)
            else:
                misc.save_on_master(net.state_dict(), args.output + model_name)
            # save lora and multiSpectral parameters
            model_lora_multiSpectral = "/best_model_lora_multiSpectral.pth"
            if args.use_lora:
                save_sam_params(sam,args.output + model_lora_multiSpectral)
            else:
                save_sam_params(sam,args.output + model_lora_multiSpectral,save_backbone=True)
            
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        model_name = f"/lora_multiSpectral_epoch_{epoch}.pth"
        if args.use_lora:
            save_sam_params(sam,args.output + model_lora_multiSpectral)
        else:
            save_sam_params(sam,args.output + model_lora_multiSpectral,save_backbone=True)

        #3. HQ Token Mask Decoder
        model_name = "/Final_epoch_"+str(epoch)+".pth"
        print('come here save at', args.output + model_name)
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
        else:
            misc.save_on_master(net.state_dict(), args.output + model_name)
def print_trainable_gradients(model, tag=""):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"❌ [{tag}] No gradient: {name}")
            else:
                grad_norm = param.grad.norm().item()
                param_mean = param.data.mean().item()
                print(f"✅ [{tag}] Gradient OK: {name} | grad norm: {grad_norm:.6f} | param mean: {param_mean:.6f}")
def save_sam_params(net,output_dir,save_backbone=False):
    tmp_ckpt = {}
    if save_backbone:
        for name, param in net.image_encoder.state_dict().items():
            tmp_ckpt[f"image_encoder.{name}"] = param.data.cpu()
        torch.save(tmp_ckpt, output_dir)
    else:
        #1. SAM ViT Backbone
        for name, param in net.image_encoder.state_dict().items():
            if 'patch_embed' in name:
                tmp_ckpt[f"image_encoder.{name}"] = param.data.cpu()
            if 'pos_embed' in name:
                tmp_ckpt[f"image_encoder.{name}"] = param.data.cpu()
        #2. LoRA in ViT
        for block_idx, block in enumerate(net.image_encoder.blocks):
            # 假设 qkv 层已被替换为 LoRALinear 类型
            if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, loralib.layers.Linear):
                qkv = block.attn.qkv
                # 保存 lora_A 和 lora_B
                tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_A"] = qkv.lora_A.data.cpu()
                tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_B"] = qkv.lora_B.data.cpu()
            if hasattr(net,'multispectral_encoder'):
                for name, param in net.multispectral_encoder.state_dict().items():
                        tmp_ckpt[f"multispectral_encoder.{name}"] = param.data.cpu()
        torch.save(tmp_ckpt, output_dir)

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
def mask_vis(image_dict,masks_hq,mask_semantic,sam_out,save_path,filename,index=0):
    image = image_dict[index]['image'].squeeze().permute(1,2,0).cpu().numpy()
    masks_hq_save = (masks_hq>0)[index].int().squeeze(0).cpu().numpy()*255
    mask_semantic_save = mask_semantic[index].int().squeeze(0).cpu().numpy()*255
    sam_out_save = sam_out[index]['masks'].float().squeeze().cpu().numpy()*255
    cv2.imwrite(save_path+'/'+filename+'_test_image.png',image)
    cv2.imwrite(save_path+'/'+filename+'_test_hq.png',masks_hq_save)
    cv2.imwrite(save_path+'/'+filename+'_test_semantic.png',mask_semantic_save)
    cv2.imwrite(save_path+'/'+filename+'_test_sam_out.png',sam_out_save)
def evaluate(args, net, sam, valid_dataloaders, visualize=False,loss_edge_fn=None,loss_instance_fn=None,epoch=0,if_test=False):
    net.eval()
    print("Validating...")
    test_stats = {}
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        progress_bar = tqdm(enumerate(metric_logger.log_every(valid_dataloader, 1000)),
                    desc="Validating", unit="batch", total=len(valid_dataloader))
        feature_center = None
        for step,data_val in progress_bar:
            inputs, mask_semantic, mask_edge, mask_distance ,mask_instance ,image_name= data_val['image']*255, data_val['mask_semantic'].float(), data_val['mask_edge'].float(), data_val['mask_distance'].float(),data_val['mask_instance'].float(),data_val['image_name']
            if args.data_type_input == "RGB":
                if args.dataset == "S4A":
                    inputs = inputs[:,1:4,:,:]
                elif args.dataset == "AI4Boundaries":
                    inputs = inputs[:,:3,:,:]
            if torch.cuda.is_available():
                inputs_val = inputs.to(device)
                mask_semantic_val = mask_semantic.to(device)
                mask_edge_val = mask_edge.to(device)
                mask_distance_val = mask_distance.to(device)
                mask_instance_val = mask_instance.to(device)
                # labels_val = mask_semantic.cuda()
                # labels_val = labels_val.cuda()
                # labels_ori = labels_ori.cuda()
            labels_256 = F.interpolate(mask_semantic_val, size=(256, 256), mode='bilinear')
            edge_256 = F.interpolate(mask_edge_val, size=(256, 256), mode='bilinear')
            # instance_256 = F.interpolate(mask_distance_val, size=(256, 256), mode='bilinear')
            instance_256 = F.interpolate(mask_instance_val, size=(256, 256), mode='bilinear')
            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(mask_semantic_val[:,0,:,:],threshold=0.5)
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                # if input_type == 'box':
                #     dict_input['boxes'] = labels_box[b_i:b_i+1]
                # elif input_type == 'point':
                #     point_coords = labels_points[b_i:b_i+1]
                #     dict_input['point_coords'] = point_coords
                #     dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                # elif input_type == 'noise_mask':
                #     dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                # else:
                #     raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                # if args.use_orth_loss:
                #     batched_output, interm_embeddings = sam(batched_input, multimask_output=False,train_model=False)
                # else:
                #     batched_output, interm_embeddings = sam(batched_input, multimask_output=False)

                batched_output, interm_embeddings = sam(batched_input, multimask_output=False,train_model=False)

                batch_len = len(batched_output)
                encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
                image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
                sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
                dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
                ortho_loss_channelTokens = [batched_output[i_l]['ortho_loss'] for i_l in range(batch_len)]
                ortho_loss_channelTokens = torch.stack(ortho_loss_channelTokens).mean()
                masks_hq = net(
                    image_embeddings=encoder_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    hq_token_only=True,
                    interm_embeddings=interm_embeddings,
                )

                if hasattr(net,'_class_logits'):
                    class_logits = net._class_logits
                else:
                    class_logits = None
                if hasattr(net,'seg_tokens'):
                    seg_tokens = net.seg_tokens
                else:
                    seg_tokens = None
                
                
                (
                    masks_hq,
                    masks_edge,
                    masks_instance,
                    upscaled_embedding_ours,
                    seg_tokens,
                ) = unpack_main_decoder_outputs(
                    masks_hq, args.use_feature_struct_loss and args.struct_loss_in_finalEmbedding
                )

                iou = compute_iou(masks_hq,mask_semantic_val,threshold=0.5)
                boundary_iou = compute_boundary_iou(masks_hq,mask_semantic_val,threshold=0.5)
                if args.segloss_type == 'Full':
                    loss_mask, loss_dice =loss_masks_full(masks_hq, labels_256)
                else:
                    loss_mask, loss_dice = loss_masks(masks_hq, mask_semantic_val, len(masks_hq)) 
                
                if seg_tokens is not None:
                    loss_seg_token_diversity = instance_proxy_loss(seg_tokens,sam.image_encoder.patch_embed.channel_embed.weight)
                else:
                    loss_seg_token_diversity = torch.tensor(0.0).to(device=loss_mask.device)
                # loss_seg_token_diversity = instance_proxy_loss(seg_tokens,sam.image_encoder.patch_embed.channel_embed.weight)
                loss_edge = loss_edge_fn(masks_edge, edge_256)
                loss_instance = loss_instance_fn(masks_instance, class_logits, instance_256)

                instance_token_afer_fusion = None
                if args.use_instance_diversity_loss and instance_token_afer_fusion is not None:
                    loss_instance_diversity = instance_proxy_loss(instance_token_afer_fusion,net.instance_token_proxy)
                else:
                    loss_instance_diversity = torch.tensor(0.0).to(device=loss_instance.device)

                if args.use_feature_struct_loss:
                    if args.struct_loss_in_finalEmbedding:
                        assert upscaled_embedding_ours is not None ,"upscaled_embedding_ours is None"
                        feature_struct_losses = compute_intra_inter_class_losses(encoder_embedding=upscaled_embedding_ours,semantic_mask=labels_256,num_classes=2,last_class_center=feature_center)
                    else:
                        feature_struct_losses = compute_intra_inter_class_losses(encoder_embedding=encoder_embedding,semantic_mask=labels_256,num_classes=2,last_class_center=feature_center)
                    loss_intral, loss_c2c, loss_c2p, class_avg_features, updated_last_center = feature_struct_losses
                    if updated_last_center is not None:
                        feature_center = updated_last_center.detach()
                    else:
                        feature_center = class_avg_features.detach()
                    struct_weight_list = [0.1,0.1,0.1]
                    loss_struct = loss_intral*struct_weight_list[0] + loss_c2c*struct_weight_list[1] + loss_c2p*struct_weight_list[2]
                else:
                    loss_struct = torch.tensor(0.0).to(device=loss_instance.device)

                weight_list = [1.0,1.0,1.0,1.0,0.1,0.1]
                loss = loss_mask*weight_list[0] + loss_dice*weight_list[1] + loss_edge*weight_list[2] + loss_instance*weight_list[3] + loss_instance_diversity*weight_list[4]+loss_instance_diversity*weight_list[5] + loss_struct + ortho_loss_channelTokens*weight_list[5]
                loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge}


            if visualize:
                print("visualize")
                if if_test:
                    eval_vis_path = os.path.join(args.output, 'vis_test')
                else:
                    eval_vis_path = os.path.join(args.output, 'eval_vis_'+str(epoch))
                save_logits(masks_hq,eval_vis_path,image_name,'test_hq_mask')
                save_logits(masks_edge,eval_vis_path,image_name,'test_hq_edge')
                pred_instance_ids = mask_logits_to_instance_id(masks_instance)
                save_instance(pred_instance_ids,eval_vis_path,image_name,'test_hq_instance')
                    
            loss_dict = {"loss": loss,"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge,'loss_instance':loss_instance,'loss_instance_diversity':loss_instance_diversity,"loss_seg_token_diversity":loss_seg_token_diversity,"iou":iou,"boundary_iou":boundary_iou,'loss_struct':loss_struct,"ortho_loss_channelTokens":ortho_loss_channelTokens}

            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                # "val_iou": f"{loss_dict_reduced['val_iou_'+str(k)].item():.4f}",
                # "val_boundary_iou": f"{loss_dict_reduced['val_boundary_iou_'+str(k)].item():.4f}"
            })
            # 可选：每 100 个 batch 打印一次详细日志，不打乱进度条
            if step % 2 == 0:
                if args.use_instance_diversity_loss:
                    tqdm.write(f"[Iter {step}]  val_iou: {loss_dict_reduced['iou'].item():.4f}, val_boundary_iou: {loss_dict_reduced['boundary_iou'].item():.4f}, loss_instance: {loss_dict_reduced['loss_instance'].item():.4f}, loss_ortho: {loss_dict_reduced['channelToken_diversity'].item():.4f}, loss_seg_token_diversity: {loss_dict_reduced['loss_seg_token_diversity'].item():.4f}, loss_feature_struct: {loss_dict_reduced['loss_struct'].item():.4f}, ortho_proxy_loss: {loss_dict_reduced['ortho_loss_channelTokens'].item():.4f}")
                else:
                    tqdm.write(f"[Iter {step}]  val_iou: {loss_dict_reduced['iou'].item():.4f}, val_boundary_iou: {loss_dict_reduced['boundary_iou'].item():.4f}, loss_instance: {loss_dict_reduced['loss_instance'].item():.4f}, loss_seg_token_diversity: {loss_dict_reduced['loss_seg_token_diversity'].item():.4f}, loss_feature_struct: {loss_dict_reduced['loss_struct'].item():.4f}, ortho_proxy_loss: {loss_dict_reduced['ortho_loss_channelTokens'].item():.4f}")
            if step==args.debug_valstep and args.debug:
                break

        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
    return test_stats

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
        instance_ids = instance[b_i].cpu().numpy().astype(np.uint16)
        
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
if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------

    dataset_ai4boundaries = {"name": "AI4Boundaries",
                 "im_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/train/MultiSpectral",
                 "gt_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/mask/train",
                 "im_ext": ".tif",
                 "gt_ext": ".png"}
    # valid set
    
    dataset_ai4boundaries_val = {"name": "AI4Boundaries",
                 "im_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/val/MultiSpectral",
                 "gt_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/mask/val",
                 "im_ext": ".tif",}
    
    dataset_ai4boundaries_test = {"name": "AI4Boundaries",
                 "im_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/test/MultiSpectral",
                 "gt_dir": "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/mask/test",
                 "im_ext": ".tif",}
    
    
    #*****************************************************
    # dataset S4A
    dataset_s4a = {"name": "S4A",
                 "im_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/train/MultiSpectral",
                 "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/train",
                 "im_ext": ".tif",
                 "gt_ext": ".png"}
    dataset_s4a_val = {"name": "S4A",
                 "im_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/val/MultiSpectral",
                 "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/val",
                 "im_ext": ".tif",}
    dataset_s4a_test = {"name": "S4A",
                 "im_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/test/MultiSpectral",
                 "gt_dir": "/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/mask/test",
                 "im_ext": ".tif",}


    args = get_args_parser()
    if args.dataset == 'S4A':
        train_datasets = [dataset_s4a]
        valid_datasets = [dataset_s4a_val]
        test_datasets  = [dataset_s4a_test]
    elif args.dataset == 'AI4Boundaries':
        train_datasets = [dataset_ai4boundaries]
        valid_datasets = [dataset_ai4boundaries_val]
        test_datasets  = [dataset_ai4boundaries_test]

    
    net = build_main_decoder(args)
    main(net,train_datasets, valid_datasets,test_datasets, args)
