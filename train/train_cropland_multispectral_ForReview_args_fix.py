# Modified from HQ-SAM

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 向最新版的代码对齐

import os
from pathlib import Path
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import argparse
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

import loralib
import json
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import MaskDecoder_SAUGE
from segment_anything_training.modeling import MaskDecoderHQ,MaskDecoder_HQ_Edge,MaskDecoder_Fusion,MaskDecoder_Fusion_v2,MaskDecoder_Fusion_v3,MaskDecoder_Fusion_v3_2,MaskDecoder_Fusion_v4,MaskDecoder_Fusion_v5,MaskDecoder_Fusion_v6,MaskDecoder_Fusion_v7,MaskDecoder_Fusion_v8,MaskDecoder_Fusion_v9

from utils.dataloader import create_dataloaders_crop
from utils.loss_mask import loss_masks,RankLoss,cross_entropy_loss_RCF,loss_masks_full,EdgeLossAutoWeight,InstanceSegmentationLoss,mask_logits_to_instance_id,InstanceSegmentationLoss_v2,InstanceSegmentationLoss_v3,fast_mask_nms_batch,instance_proxy_loss#v1 不带class_logits / v2 带上class_logits和其他限制性损失/ v3 带上class_logits
from utils.loss_feature_struct import compute_intra_inter_class_losses
import utils.misc as misc
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "sam-hq-param" / "sam_hq_vit_b.pth"

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

def save_args_json(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=2)

def str2bool(v):
    return v.lower() in ("true", "1", "yes")

def get_args_parser():

    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    # ======================
    # pipeline 强控参数（必须显式给）
    # ======================
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True) # S4A AI4Boundaries AI4Boundaries_orth
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--multispectral-encoder", type=str, required=True, help="multispectral encoder type: Conv , CBAM , ViT")
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
    parser.add_argument("--restore_model_multispectral", type=str, default=None)
    parser.add_argument("--pretrained_multispectral_encoder", type=str, default=None)


    # ======================
    # 模型结构参数（默认即可）
    # ======================
    parser.add_argument("--use_lora", type=str2bool, default=True, help="whether to use lora finetune , if set to False,  all vit parameters will train")
    parser.add_argument("--decoder-type", type=str, default="hq_instance")
    parser.add_argument("--model-type", type=str, default="vit_b_MultiSpectral")

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
    parser.add_argument("--learning_rate", type=float, default=1e-3)
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


def main(net,train_datasets, valid_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    save_args_json(args, args.output + '/args.json')
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

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
                                                        data_type=args.data_type
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

    if args.edgeloss_type == 'bce+dice':
        loss_edge_fn = EdgeLossAutoWeight(mode='bce+dice')
        
    else:
        loss_edge_fn = None
    if args.decoder_type == 'hq_instance':
        if args.instance_loss_v==1:
            loss_instance_fn = InstanceSegmentationLoss()
        elif args.instance_loss_v==2:
            loss_instance_fn = InstanceSegmentationLoss_v2(semantic_supervise=args.use_semantic_supervise_for_instance)
        elif args.instance_loss_v==3:
            loss_instance_fn = InstanceSegmentationLoss_v3()
    else:
        loss_instance_fn = None
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint, multispectral_encoder_type=args.multispectral_encoder,checkpoint_multispectral=args.restore_model_multispectral,checkpoint_pretrained_multispectral_encoder=args.pretrained_multispectral_encoder, in_channels=args.in_channels)
        sam = sam.to(device)
        if hasattr(net_without_ddp, 'prompt_encoder'):
            net_without_ddp.prompt_encoder = sam.prompt_encoder
        # if hasattr(net_without_ddp, 'transformer_2'):
        #     net_without_ddp.init_two_way_transformer(args.restore_model)
        # 先进行DDP包装
        if args.distributed:
            sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        # sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint,multispectral_encoder_type=args.multispectral_encoder)
        optimizer = optim.Adam(list(net.parameters())+list(sam.parameters()), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model),strict=False)
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
        
        if args.accumulation_steps > 1:
            train_accumgrad(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler,loss_edge_fn,loss_instance_fn,sam)
        else:
            train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler,loss_edge_fn,loss_instance_fn,sam)
    else:
        sam = sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint, multispectral_encoder_type=args.multispectral_encoder,checkpoint_multispectral=args.restore_model_multispectral,checkpoint_pretrained_multispectral_encoder=args.pretrained_multispectral_encoder, in_channels=args.in_channels)
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


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler,loss_edge_fn=None,loss_instance_fn=None,sam=None):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)
    
    if sam is None:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint,multispectral_encoder_type=args.multispectral_encoder)
        _ = sam.to(device=args.device)
    
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #     sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    # else:
    #     sam = sam.to(device=args.device)
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        if misc.is_main_process():
            os.makedirs(args.output, exist_ok=True)
            # 创建损失日志文件并写入表头
            log_file = os.path.join(args.output, "training_log.csv")
            if not os.path.exists(log_file) or epoch == 0:
                # if os.path.exists(log_file):
                #     os.remove(log_file)
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
        for step,data in progress_bar:
            inputs, mask_semantic, mask_edge, mask_distance,mask_instance = data['image']*255, data['mask_semantic'].float(), data['mask_edge'].float(), data['mask_distance'].float(), data['mask_instance'].float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                # labels = labels.cuda()
                mask_semantic = mask_semantic.cuda()
                mask_edge = mask_edge.cuda()
                mask_distance = mask_distance.cuda()
                mask_instance = mask_instance.cuda()
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            input_keys = ['box','noise_mask']
            labels_box = misc.masks_to_boxes(mask_semantic[:,0,:,:],threshold=0.5)
            try:
                labels_points = misc.masks_sample_points(mask_semantic[:,0,:,:],threshold=0.5)
            except:
                # less than 10 points
                input_keys = ['noise_mask']
            labels_256 = F.interpolate(mask_semantic, size=(256, 256), mode='bilinear')
            edge_256 = F.interpolate(mask_edge, size=(256, 256), mode='bilinear')
            instance_256 = F.interpolate(mask_instance, size=(256, 256), mode='bilinear')
            # labels_256 = mask_semantic
            labels_noisemask = misc.masks_noise(labels_256,mask_max=1)

            # set prompt to None
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                # input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                input_image = torch.as_tensor(imgs[b_i], device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                # input_type = random.choice(input_keys)
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
                masks_hq ,masks_edge, masks_instance = masks_hq
            
            if args.segloss_type == 'Full':
                loss_mask, loss_dice = loss_masks_full(masks_hq, labels_256)
            else:
                loss_mask, loss_dice = loss_masks(masks_hq, mask_semantic, len(masks_hq)) 
            
            if args.decoder_type == 'hq_edge' or args.decoder_type == 'hq_instance':
                if loss_edge_fn is not None:
                    loss_edge = loss_edge_fn(masks_edge, edge_256)
                else:
                    loss_edge = RankLoss.apply(masks_edge, edge_256) #OOM !!
                if args.decoder_type == 'hq_instance':
                    if loss_instance_fn is not None:
                        loss_instance = loss_instance_fn(masks_instance, instance_256)
                        # 暂时设置一个权值
                        weight_list = [0.5,0.5, 0.5, 1.0]
                        loss = loss_mask*weight_list[0] + loss_dice*weight_list[1] + loss_edge*weight_list[2] + loss_instance*weight_list[2]
                        loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge,'loss_instance':loss_instance}
                else:
                    loss = loss_mask + loss_dice + loss_edge
                    loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge}
            else:
                loss = loss_mask + loss_dice
                loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}


            sam_param_backup = {
                name: param.detach().clone()
                for name, param in sam.named_parameters()
                if param.requires_grad
            }
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for name, param in sam.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        print(f"❌ No gradient: {name}")
                    else:
                        print(f"✅ Gradient OK: {name}, grad norm: {param.grad.norm().item():.6f}")
            # Debug
            # 3. 对比参数是否更新
            updated_params = []
            for name, param in sam.named_parameters():
                if param.requires_grad:
                    before = sam_param_backup[name]
                    after = param.detach()
                    if not torch.allclose(before, after, atol=1e-6):
                        updated_params.append(name)

            if updated_params:
                print("✅ SAM parameters updated:")
                for name in updated_params:
                    print(f"  - {name}")
            else:
                print("❌ SAM parameters NOT updated!")

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            # 在 tqdm 上更新显示内容（只展示一个主指标）
            progress_bar.set_postfix({
                "Loss": f"{loss_value:.4f}",
                "Mask": f"{loss_dict_reduced['loss_mask'].item():.4f}",
                "Dice": f"{loss_dict_reduced['loss_dice'].item():.4f}"
            })
            # 可选：每 100 个 batch 打印一次详细日志，不打乱进度条
            if step % 2 == 0:
                if args.decoder_type == 'hq_edge':
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}, Edge: {loss_dict_reduced['loss_edge'].item():.4f}")
                elif args.decoder_type == 'hq_instance':
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}, Edge: {loss_dict_reduced['loss_edge'].item():.4f}, instance: {loss_dict_reduced['loss_instance'].item():.4f}")
                else:
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}")
            if step==args.debug_trainstep and args.debug:
                break

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
        with open(log_file, "a") as f:
            # 获取训练损失（使用同步后的平均值）
            train_loss = train_stats.get("training_loss", -1)
            loss_mask = train_stats.get("loss_mask", -1)
            loss_dice = train_stats.get("loss_dice", -1)
            
            # 获取验证损失（根据evaluate函数返回结构调整）
            val_loss = test_stats.get("val_loss", -1)  
            # 添加其他需要的验证指标...
            
            # 写入CSV行
            f.write(
                f"{epoch},{optimizer.param_groups[0]['lr']:.6f},"
                f"{train_loss:.4f},{loss_mask:.4f},{loss_dice:.4f},"
                f"{val_loss:.4f},...\n"  # 补充验证指标
            )

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                misc.save_on_master(net.module.state_dict(), args.output + model_name)
            else:
                misc.save_on_master(net.state_dict(), args.output + model_name)
            # save lora and multiSpectral parameters
            model_lora_multiSpectral = "/lora_multiSpectral_epoch_"+str(epoch)+".pth"
            tmp_ckpt = {}
            #1. MultiSpectral Parameter
            for name, param in sam.multispectral_encoder.state_dict().items():
                tmp_ckpt[f"multispectral_encoder.{name}"] = param.data.cpu()
            
            #2. LoRA in ViT
            for block_idx, block in enumerate(sam.image_encoder.blocks):
                # 假设 qkv 层已被替换为 LoRALinear 类型
                if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, loralib.layers.Linear):
                    qkv = block.attn.qkv
                    # 保存 lora_A 和 lora_B
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_A"] = qkv.lora_A.data.cpu()
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_B"] = qkv.lora_B.data.cpu()
            
            torch.save(tmp_ckpt, args.output + model_lora_multiSpectral)
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
            tmp_ckpt = {}
            #1. MultiSpectral Parameter
            for name, param in sam.multispectral_encoder.state_dict().items():
                tmp_ckpt[f"multispectral_encoder.{name}"] = param.data.cpu()
            
            #2. LoRA in ViT
            for block_idx, block in enumerate(sam.image_encoder.blocks):
                # 假设 qkv 层已被替换为 LoRALinear 类型
                if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, loralib.layers.Linear):
                    qkv = block.attn.qkv
                    # 保存 lora_A 和 lora_B
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_A"] = qkv.lora_A.data.cpu()
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_B"] = qkv.lora_B.data.cpu()
            
            torch.save(tmp_ckpt, args.output + model_lora_multiSpectral)

    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = {}
        #1. MultiSpectral Parameter
        for name, param in sam.multispectral_encoder.state_dict().items():
            sam_ckpt[f"multispectral_encoder.{name}"] = param.data.cpu()
        
        #2. LoRA in ViT
        for block_idx, block in enumerate(sam.image_encoder.blocks):
            # 假设 qkv 层已被替换为 LoRALinear 类型
            if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, loralib.layers.Linear):
                qkv = block.attn.qkv
                # 保存 lora_A 和 lora_B
                sam_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_A"] = qkv.lora_A.data.cpu()
                sam_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_B"] = qkv.lora_B.data.cpu()
        
        #3. HQ Token Mask Decoder
        model_name = "/Final_epoch_"+str(epoch)+".pth"
        print('come here save at', args.output + model_name)
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
        else:
            misc.save_on_master(net.state_dict(), args.output + model_name)
        # hq_decoder = torch.load(args.output + model_name)
        # for key in hq_decoder.keys():
        #     sam_key = f"mask_decoder.{key}"
        #     sam_ckpt[sam_key] = hq_decoder[key]
        
        model_name = f"/lora_multiSpectral_epoch_{epoch}.pth"
        torch.save(sam_ckpt, args.output + model_name)

def train_accumgrad(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler,loss_edge_fn=None,loss_instance_fn=None,sam=None):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

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
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint, multispectral_encoder_type=args.multispectral_encoder,checkpoint_multispectral=args.restore_model_multispectral,checkpoint_pretrained_multispectral_encoder=args.pretrained_multispectral_encoder, in_channels=args.in_channels)
        _ = sam.to(device=device)
    
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #     sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    # else:
    #     sam = sam.to(device=args.device)
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        if misc.is_main_process():
            os.makedirs(args.output, exist_ok=True)
            # 创建损失日志文件并写入表头
            log_file = os.path.join(args.output, "training_log.csv")
            if not os.path.exists(log_file) or epoch == 0:
                # if os.path.exists(log_file):
                #     os.remove(log_file)
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

        # struct loss
        feature_center = None
        for step,data in progress_bar:
            inputs, mask_semantic, mask_edge, mask_distance,mask_instance = data['image']*255, data['mask_semantic'].float(), data['mask_edge'].float(), data['mask_distance'].float(), data['mask_instance']
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                # labels = labels.cuda()
                mask_semantic = mask_semantic.to(device)
                mask_edge = mask_edge.to(device)
                mask_distance = mask_distance.to(device)
                mask_instance = mask_instance.to(device)
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            input_keys = ['box','noise_mask']
            labels_box = misc.masks_to_boxes(mask_semantic[:,0,:,:],threshold=0.5)
            try:
                labels_points = misc.masks_sample_points(mask_semantic[:,0,:,:],threshold=0.5)
            except:
                # less than 10 points
                input_keys = ['noise_mask']
            labels_256 = F.interpolate(mask_semantic, size=(256, 256), mode='bilinear')
            edge_256 = F.interpolate(mask_edge, size=(256, 256), mode='bilinear')
            instance_256 = F.interpolate(mask_instance, size=(256, 256), mode='nearest')
            # labels_256 = mask_semantic
            labels_noisemask = misc.masks_noise(labels_256,mask_max=1)

            # set prompt to None
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                # input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                input_image = torch.as_tensor(imgs[b_i], device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

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
            
            if hasattr(net,'_class_logits'):
                class_logits = net._class_logits
            else:
                class_logits = None
            if hasattr(net,'_mask_instance_clusters'):
                mask_instance_clusters = net._mask_instance_clusters
            else:
                mask_instance_clusters = None

            if args.segloss_type == 'Full':
                loss_mask, loss_dice = loss_masks_full(masks_hq, labels_256)
            else:
                loss_mask, loss_dice = loss_masks(masks_hq, mask_semantic, len(masks_hq)) 
            
            if args.decoder_type == 'hq_edge' or args.decoder_type == 'hq_instance':
                if loss_edge_fn is not None:
                    loss_edge = cross_entropy_loss_RCF(masks_edge, edge_256)
                else:
                    loss_edge = RankLoss.apply(masks_edge, edge_256) #OOM !!

                if args.decoder_type == 'hq_instance':
                    if loss_instance_fn is not None:
                        loss_instance = loss_instance_fn(masks_instance ,class_logits ,instance_256 ,mask_instance_clusters)
                        
                        if args.use_instance_diversity_loss and instance_token_afer_fusion is not None:
                            loss_instance_diversity = instance_proxy_loss(instance_token_afer_fusion,net.instance_token_proxy)
                        else:
                            loss_instance_diversity = torch.tensor(0.0).to(device=loss_instance.device)

                        if args.use_feature_struct_loss:
                            feature_struct_losses = compute_intra_inter_class_losses(encoder_embedding=encoder_embedding,semantic_mask=labels_256,num_classes=2,last_class_center=feature_center)
                            loss_intral, loss_c2c, loss_c2p, class_avg_features, updated_last_center = feature_struct_losses
                            if updated_last_center is not None:
                                feature_center = updated_last_center.detach()
                            else:
                                feature_center = class_avg_features.detach()
                            # feature_center = updated_last_center
                            struct_weight_list = [0.5,0.5,0.5]
                            loss_struct = loss_intral*struct_weight_list[0] + loss_c2c*struct_weight_list[1] + loss_c2p*struct_weight_list[2]
                        else:
                            loss_struct = torch.tensor(0.0).to(device=loss_instance.device)

                        weight_list = [1.0,1.0,1.0,1.0,0.1]
                        loss = loss_mask*weight_list[0] + loss_dice*weight_list[1] + loss_edge*weight_list[2] + loss_instance*weight_list[3] + loss_instance_diversity*weight_list[4] + loss_struct
                        loss_dict = {"loss": loss,"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge,'loss_instance':loss_instance,'loss_instance_diversity':loss_instance_diversity,'loss_struct':loss_struct}
                else:
                    loss = loss_mask + loss_dice + loss_edge
                    loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge}
            else:
                loss = loss_mask + loss_dice
                loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            
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
            #             print(f"✅ Gradient OK: {name}, grad norm: {param.grad.norm().item():.6f}")
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
                if args.decoder_type == 'hq_edge':
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}, Edge: {loss_dict_reduced['loss_edge'].item():.4f}")
                elif args.decoder_type == 'hq_instance':
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}, Edge: {loss_dict_reduced['loss_edge'].item():.4f}, instance: {loss_dict_reduced['loss_instance'].item():.4f}, loss_instance_diversity: {loss_dict_reduced['loss_instance_diversity'].item():.4f}, loss_feature_struct: {loss_dict_reduced['loss_struct'].item():.4f}")
                else:
                    tqdm.write(f"[Iter {step}] Loss: {loss_value:.4f}, Mask: {loss_dict_reduced['loss_mask'].item():.4f}, Dice: {loss_dict_reduced['loss_dice'].item():.4f}")
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
        with open(log_file, "a") as f:
            # 获取训练损失（使用同步后的平均值）
            train_loss = train_stats.get("training_loss", -1)
            loss_mask = train_stats.get("loss_mask", -1)
            loss_dice = train_stats.get("loss_dice", -1)
            
            # 获取验证损失（根据evaluate函数返回结构调整）
            val_loss = test_stats.get("loss_value", -1)  
            # 添加其他需要的验证指标...
            
            # 写入CSV行
            f.write(
                f"{epoch},{optimizer.param_groups[0]['lr']:.6f},"
                f"{train_loss:.4f},{loss_mask:.4f},{loss_dice:.4f},"
                f"{val_loss:.4f},...\n"  # 补充验证指标
            )

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                misc.save_on_master(net.module.state_dict(), args.output + model_name)
            else:
                misc.save_on_master(net.state_dict(), args.output + model_name)
            # save lora and multiSpectral parameters
            model_lora_multiSpectral = "/lora_multiSpectral_epoch_"+str(epoch)+".pth"
            tmp_ckpt = {}
            #1. MultiSpectral Parameter
            for name, param in sam.multispectral_encoder.state_dict().items():
                tmp_ckpt[f"multispectral_encoder.{name}"] = param.data.cpu()
            
            #2. LoRA in ViT
            for block_idx, block in enumerate(sam.image_encoder.blocks):
                # 假设 qkv 层已被替换为 LoRALinear 类型
                if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, loralib.layers.Linear):
                    qkv = block.attn.qkv
                    # 保存 lora_A 和 lora_B
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_A"] = qkv.lora_A.data.cpu()
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_B"] = qkv.lora_B.data.cpu()
            
            torch.save(tmp_ckpt, args.output + model_lora_multiSpectral)
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
            tmp_ckpt = {}
            #1. MultiSpectral Parameter
            for name, param in sam.multispectral_encoder.state_dict().items():
                tmp_ckpt[f"multispectral_encoder.{name}"] = param.data.cpu()
            
            #2. LoRA in ViT
            for block_idx, block in enumerate(sam.image_encoder.blocks):
                # 假设 qkv 层已被替换为 LoRALinear 类型
                if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, loralib.layers.Linear):
                    qkv = block.attn.qkv
                    # 保存 lora_A 和 lora_B
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_A"] = qkv.lora_A.data.cpu()
                    tmp_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_B"] = qkv.lora_B.data.cpu()
            
            torch.save(tmp_ckpt, args.output + model_lora_multiSpectral)

    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = {}
        #1. MultiSpectral Parameter
        for name, param in sam.multispectral_encoder.state_dict().items():
            sam_ckpt[f"multispectral_encoder.{name}"] = param.data.cpu()
        
        #2. LoRA in ViT
        for block_idx, block in enumerate(sam.image_encoder.blocks):
            # 假设 qkv 层已被替换为 LoRALinear 类型
            if hasattr(block.attn, 'qkv') and isinstance(block.attn.qkv, loralib.layers.Linear):
                qkv = block.attn.qkv
                # 保存 lora_A 和 lora_B
                sam_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_A"] = qkv.lora_A.data.cpu()
                sam_ckpt[f"image_encoder.blocks.{block_idx}.attn.qkv.lora_B"] = qkv.lora_B.data.cpu()
        
        #3. HQ Token Mask Decoder
        model_name = "/Final_epoch_"+str(epoch)+".pth"
        print('come here save at', args.output + model_name)
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
        else:
            misc.save_on_master(net.state_dict(), args.output + model_name)
        # hq_decoder = torch.load(args.output + model_name)
        # for key in hq_decoder.keys():
        #     sam_key = f"mask_decoder.{key}"
        #     sam_ckpt[sam_key] = hq_decoder[key]
        
        model_name = f"/lora_multiSpectral_epoch_{epoch}.pth"
        torch.save(sam_ckpt, args.output + model_name)




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
def evaluate(args, net, sam, valid_dataloaders, visualize=False,loss_edge_fn=None,loss_instance_fn=None,epoch=0):
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

                if hasattr(net,'_class_logits'):
                    class_logits = net._class_logits
                else:
                    class_logits = None

                if args.decoder_type == 'hq_edge':
                    masks_hq ,masks_edge = masks_hq
                elif args.decoder_type == 'hq_instance':
                    if len(masks_hq) == 3:
                        masks_hq ,masks_edge, masks_instance = masks_hq
                        instance_token_afer_fusion = None
                    elif len(masks_hq) == 4:
                        masks_hq ,masks_edge, masks_instance, instance_token_afer_fusion = masks_hq

                iou = compute_iou(masks_hq,mask_semantic_val,threshold=0.5)
                boundary_iou = compute_boundary_iou(masks_hq,mask_semantic_val,threshold=0.5)
                if args.segloss_type == 'Full':
                    loss_mask, loss_dice =loss_masks_full(masks_hq, labels_256)
                else:
                    loss_mask, loss_dice = loss_masks(masks_hq, mask_semantic_val, len(masks_hq)) 
                if args.decoder_type == 'hq_edge' or args.decoder_type == 'hq_instance':
                    if loss_edge_fn is not None:
                        loss_edge = loss_edge_fn(masks_edge, edge_256)
                    else:
                        loss_edge = RankLoss.apply(masks_edge, edge_256)  #OOM !!
                    if args.decoder_type == 'hq_instance':
                        if loss_instance_fn is not None:
                            loss_instance = loss_instance_fn(masks_instance ,class_logits ,instance_256)
                            
                            if args.use_instance_diversity_loss and instance_token_afer_fusion is not None:
                                loss_instance_diversity = instance_proxy_loss(instance_token_afer_fusion,net.instance_token_proxy)
                            else:
                                loss_instance_diversity = torch.tensor(0.0).to(device=loss_instance.device)
                            # loss weight for mask , dice_mask , edge ,instance
                            # if epoch > 5:
                            #     weight_list = [0.1,0.1,0.1,1.0,0.1]
                            # else:
                            if args.use_feature_struct_loss:
                                feature_struct_losses = compute_intra_inter_class_losses(encoder_embedding=encoder_embedding,semantic_mask=labels_256,num_classes=2,last_class_center=feature_center)
                                loss_intral, loss_c2c, loss_c2p, class_avg_features, updated_last_center = feature_struct_losses
                                if updated_last_center is not None:
                                    feature_center = updated_last_center.detach()
                                else:
                                    feature_center = class_avg_features.detach()
                                # feature_center = updated_last_center
                                struct_weight_list = [0.1,0.1,0.1]
                                loss_struct = loss_intral*struct_weight_list[0] + loss_c2c*struct_weight_list[1] + loss_c2p*struct_weight_list[2]
                            else:
                                loss_struct = torch.tensor(0.0).to(device=loss_instance.device)
                            weight_list = [0.5,0.5,0.5,1.0,0.1]
                            
                            loss = loss_mask*weight_list[0] + loss_dice*weight_list[1] + loss_edge*weight_list[2] + loss_instance*weight_list[3] + loss_instance_diversity*weight_list[4] + loss_struct
                            # loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge,'loss_instance':loss_instance}
                        # if loss_instance_fn is not None:
                        #     # if isinstance(loss_instance_fn, List):
                        #     #     loss_instance_fn , loss_compactness_fn = loss_instance_fn
                        #     loss_instance = loss_instance_fn(masks_instance,class_logits, instance_256)
                        #     # loss_compactness = loss_compactness_fn(masks_instance)
                        #     loss = loss_mask + loss_dice + loss_edge + loss_instance
                    else:
                        loss = loss_mask + loss_dice + loss_edge
                    loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, "loss_edge":loss_edge}
                else:
                    loss = loss_mask + loss_dice
                    loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}
            
            # loss_dict = {}

            if visualize:
                print("visualize")
                eval_vis_path = os.path.join(args.output, 'eval_vis_'+str(epoch))
                save_logits(masks_hq,eval_vis_path,image_name,'test_hq_mask')
                if args.decoder_type == 'hq_edge':
                    save_logits(masks_edge,eval_vis_path,image_name,'test_hq_edge')
                if args.decoder_type == 'hq_instance':
                    save_logits(masks_edge,eval_vis_path,image_name,'test_hq_edge')
                    pred_instance_ids = mask_logits_to_instance_id(masks_instance)
                    save_instance(pred_instance_ids,eval_vis_path,image_name,'test_hq_instance')
                    # save_logits(masks_edge,args.output,image_name,'test_hq_edge')
                    
                    # masks_instance_sigmoid = torch.sigmoid(masks_instance)
                    # masks_filtered = fast_mask_nms_batch(masks_instance_sigmoid, scores=None)
                    # red_instance_ids = mask_logits_to_instance_id(masks_filtered,if_logits=False)
                    # save_instance(red_instance_ids,args.output,image_name,'test_hq_instance')
            loss_dict = {"loss_value": loss, "loss_mask": loss_mask, "loss_dice":loss_dice,"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou,'loss_instance':loss_instance,'loss_instance_diversity':loss_instance_diversity,'loss_struct':loss_struct}

            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "val_iou": f"{loss_dict_reduced['val_iou_'+str(k)].item():.4f}",
                "val_boundary_iou": f"{loss_dict_reduced['val_boundary_iou_'+str(k)].item():.4f}"
            })
            # 可选：每 100 个 batch 打印一次详细日志，不打乱进度条
            if step % 2 == 0:
                tqdm.write(f"[Iter {step}]  val_iou: {loss_dict_reduced['val_iou_'+str(k)].item():.4f}, val_boundary_iou: {loss_dict_reduced['val_boundary_iou_'+str(k)].item():.4f}, loss_instance: {loss_dict_reduced['loss_instance'].item():.4f}, loss_instance_diversity: {loss_dict_reduced['loss_instance_diversity'].item():.4f}, loss_feature_struct: {loss_dict_reduced['loss_struct'].item():.4f}")
            if step==50 : # 减少测试次数，加快测试速度
                break
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
    dataset_ai4boundaries_orth = {"name": "AI4Boundaries",
                 "im_dir": "/home/huar/DataSet/AI4Boundary_orthos/images/train",
                 "gt_dir": "/home/huar/DataSet/AI4Boundary_orthos/masks/train",
                 "im_ext": ".tif",
                 "gt_ext": ".png"}
    # valid set
    
    dataset_ai4boundaries_val_orth = {"name": "AI4Boundaries",
                 "im_dir": "/home/huar/DataSet/AI4Boundary_orthos/images/val",
                 "gt_dir": "/home/huar/DataSet/AI4Boundary_orthos/masks/val",
                 "im_ext": ".tif",}
    
    dataset_ai4boundaries_test_orth = {"name": "AI4Boundaries",
                 "im_dir": "/home/huar/DataSet/AI4Boundary_orthos/images/test",
                 "gt_dir": "/home/huar/DataSet/AI4Boundary_orthos/masks/test",
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
    elif args.dataset == 'AI4Boundaries_orth':
        train_datasets = [dataset_ai4boundaries_orth]
        valid_datasets = [dataset_ai4boundaries_val_orth]
        test_datasets  = [dataset_ai4boundaries_test_orth]
    #              "gt_ext": ".png"}
    # train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    # valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 
    # print(args.dataset)
    # print(train_datasets)
    if args.decoder_type == 'hq':
        net = MaskDecoderHQ(args.model_type) 
    elif args.decoder_type == 'hq_edge' :
        net = MaskDecoder_HQ_Edge(args.model_type)
    elif args.decoder_type == 'hq_instance' :
        if args.maskdecoder_v == 1:
            net = MaskDecoder_Fusion(args.model_type,args.fusion_v)
        elif args.maskdecoder_v == 2:
            net = MaskDecoder_Fusion_v2(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token) # fuison 2; dyConv 
        elif args.maskdecoder_v == 3:
            net = MaskDecoder_Fusion_v3_2(args.model_type,args.fusion_v,args.instance_decoder,args.num_instance_token,args.use_multiscale_feature,args.use_semantic_supervise_for_instance, return_final_embed=False) # 和/home/huar/LM_SR/sam-hq/train/train_cropland_multispectral_channelToken_low_continue_A4B.py 统一
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
            net = MaskDecoder_SAUGE(model_type=args.model_type)

    main(net,train_datasets, valid_datasets, args)
