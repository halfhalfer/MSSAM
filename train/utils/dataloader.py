# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import imageio.v3 as iio
import cv2
import os
import json
from glob import glob
# import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler,SequentialSampler
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torchvision.transforms import Resize
import torchvision.transforms as T
from torch.utils.data.distributed import DistributedSampler

#### --------------------- dataloader online ---------------------####
def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()



def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})

    return name_im_gt_list

def create_dataloaders_crop_coco(name_im_gt_list, my_transforms=[], batch_size=1, training=False, foreground_class=1, data_type='RGB'):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = CroplandDatasetCOCO(name_im_gt_list[i]['im_dir'], name_im_gt_list[i]['gt_dir'],name_im_gt_list[i]['coco_json_dir'],foreground_class=foreground_class,data_type=data_type,if_test=True)
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        if is_dist_avail_and_initialized():
            sampler = DistributedSampler(gos_dataset)
        else:
            sampler = RandomSampler(gos_dataset)  

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = CroplandDatasetCOCO(name_im_gt_list[i]['im_dir'], name_im_gt_list[i]['gt_dir'],name_im_gt_list[i]['coco_json_dir'],foreground_class=foreground_class,data_type=data_type,if_test=True)
            if is_dist_avail_and_initialized():
                sampler = DistributedSampler(gos_dataset)
            else:
                sampler = RandomSampler(gos_dataset)  
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets
def create_dataloaders_crop(name_im_gt_list, my_transforms=[], batch_size=1, training=False, foreground_class=1, data_type='RGB',if_test=False,select_sample_list=None):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = CroplandDataset(name_im_gt_list[i]['im_dir'], name_im_gt_list[i]['gt_dir'],foreground_class=foreground_class,data_type=data_type,if_test=True,image_list_txt=select_sample_list)
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        if is_dist_avail_and_initialized():
            sampler = DistributedSampler(gos_dataset)
        else:
            sampler = RandomSampler(gos_dataset)  

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = CroplandDataset(name_im_gt_list[i]['im_dir'], name_im_gt_list[i]['gt_dir'],foreground_class=foreground_class,data_type=data_type,if_test=True,image_list_txt=select_sample_list)
            if is_dist_avail_and_initialized():
                sampler = DistributedSampler(gos_dataset,shuffle=False)
            else:
                if if_test:
                    sampler = SequentialSampler(gos_dataset)
                else:
                    sampler = RandomSampler(gos_dataset)  
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False, foreground_class=1):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        if is_dist_avail_and_initialized():
            sampler = DistributedSampler(gos_dataset)
        else:
            sampler = RandomSampler(gos_dataset)  

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            if is_dist_avail_and_initialized():
                sampler = DistributedSampler(gos_dataset)
            else:
                sampler = RandomSampler(gos_dataset)  
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class Resize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}

class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}




class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)

        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "label": gt,
        "shape": torch.tensor(im.shape[-2:]),
        }
        
        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample



import torch
import numpy as np
import os
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random

class CroplandDataset(Dataset):
    def __init__(
        self,
        # data_dir,
        images_dir,
        masks_dir,
        input_image_reshape=(1024, 1024),
        foreground_class=1,
        augmentation=True,
        if_test=False,
        data_type='RGB',
        image_list_txt = None,
        # '/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/a4b_continue_train_list.txt'
        # '/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/cloud_image_list.txt'
        # '/mnt/disk1/huar/DataSet/S4A/data/Process_DataSet/s4a_continue_train.txt'
    ):
        valid_exts = ('.tif', '.jpg', '.png')
        all_ids = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
        ext_type = os.path.splitext(all_ids[0])[1]
        if image_list_txt:
            with open(image_list_txt, 'r') as f:
                exclude_names = set(line.strip() for line in f if line.strip())
                exclude_names = [item.split('.')[0] for item in exclude_names]
            self.ids = [f for f in all_ids if os.path.splitext(f)[0] in exclude_names]
        else:
            self.ids = all_ids
        # images_dir = 
        
        # 路径配置保持不变
        self.images_filepaths = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_semantic_dir = os.path.join(masks_dir, 'Semantic')
        self.masks_edge_dir = os.path.join(masks_dir, 'Edge')
        self.masks_distance_dir = os.path.join(masks_dir, 'Distance')
        self.masks_instance_dir = os.path.join(masks_dir, 'channel_3') # Instance
        self.masks_semantic_auxiliary_dir = os.path.join(masks_dir, 'Semantic_Cluster')
        self.masks_edge_auxiliary_dir = os.path.join(masks_dir, 'Edge_Cluster')
        # 定义PyTorch增强管道
        # self.augmentation =  ResizeWithMasks((1024, 1024)) if augmentation else None

        self.if_test = if_test
        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.ext_type = ext_type
        self.input_image_type = data_type

    def __getitem__(self, i):
        # 加载原始数据
        image_name = self.ids[i].split('.')[0]
        
        # 读取图像
        if self.ext_type == '.tif':
            image = iio.imread(self.images_filepaths[i])
        else:
            image = cv2.cvtColor(cv2.imread(self.images_filepaths[i]), cv2.COLOR_BGR2RGB)
        image = image[:, :, :3] if self.input_image_type == 'RGB' else image
        # ---- Resize 图像 ----
        h, w = self.input_image_reshape
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # if os.path.exists(os.path.join(self.masks_semantic_auxiliary_dir, f"{image_name}.png")):
            # 读取并处理掩膜
        masks = {
            'semantic': self._load_mask(self.masks_semantic_dir, image_name),
            'edge': self._load_mask(self.masks_edge_dir, image_name),
            'distance': self._load_distance(self.masks_distance_dir, image_name),
            'semantic_cluster': self._load_instance(self.masks_semantic_auxiliary_dir, image_name),
            'edge_cluster': self._load_instance(self.masks_edge_auxiliary_dir, image_name),
            # 'instance': self._load_mask(self.masks_instance_dir, image_name)
        }
        #     semantic_cluster_flage = True
        # else:
        #     masks = {
        #         'semantic': self._load_mask(self.masks_semantic_dir, image_name),
        #         'edge': self._load_mask(self.masks_edge_dir, image_name),
        #         'distance': self._load_distance(self.masks_distance_dir, image_name),
        #         # 'instance': self._load_mask(self.masks_instance_dir, image_name)
        #     }
        #     semantic_cluster_flage = False
        masks['instance'] = self._load_instance(self.masks_instance_dir, image_name)
        # 转换为Tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # mask 添加一个维度
        masks = {k: np.expand_dims(v, axis=0) for k, v in masks.items()}
        masks = {k: torch.from_numpy(v).float() for k, v in masks.items()}

        # masks resize 
        for k, v in masks.items():
            if k == 'distance':
                continue
            if v.shape[-1] !=  256:
                masks[k] = TF.resize(v, 256, interpolation=TF.InterpolationMode.NEAREST)

        # # 后处理 
        # masks = {k: v.squeeze().long() if k != 'distance' else v.squeeze() 
        #         for k, v in mask_tensors.items()}
        # 
        # return {'image': image_tensor,'mask_semantic': masks['semantic'],'mask_edge': masks['edge'],'mask_distance': masks['distance'],'mask_instance': masks['instance'],'mask_semantic_cluster': masks['semantic_cluster'], 'image_name': os.path.basename(self.images_filepaths[i])}
        return {'image': image_tensor,'mask_semantic': masks['semantic'],'mask_edge': masks['edge'],'mask_distance': masks['distance'],'mask_instance': masks['instance'],'mask_semantic_cluster': masks['semantic_cluster'], 'image_name': os.path.basename(self.images_filepaths[i]), 'edge_cluster': masks['edge_cluster']}
        # if self.if_test:
        #     if semantic_cluster_flage:
        #         return {'image': image_tensor,'mask_semantic': masks['semantic'],'mask_edge': masks['edge'],'mask_distance': masks['distance'],'mask_instance': masks['instance'],'mask_semantic_cluster': masks['semantic_cluster'], 'image_name': os.path.basename(self.images_filepaths[i])}
        #     else:
        #         return {'image': image_tensor,'mask_semantic': masks['semantic'],'mask_edge': masks['edge'],'mask_distance': masks['distance'],'mask_instance': masks['instance'], 'image_name': os.path.basename(self.images_filepaths[i])}
        # # image_tensor, masks['semantic'], masks['edge'], masks['distance'], os.path.basename(self.images_filepaths[i])
        # else:
        #     if semantic_cluster_flage:
        #         return {'image': image_tensor,'mask_semantic': masks['semantic'],'mask_edge': masks['edge'],'mask_distance': masks['distance'],'mask_instance': masks['instance'],'mask_semantic_cluster': masks['semantic_cluster']}
        #     else:
        #         return {'image': image_tensor,'mask_semantic': masks['semantic'],'mask_edge': masks['edge'],'mask_distance': masks['distance'],'mask_instance': masks['instance']}
        # image_tensor, masks['semantic'], masks['edge'], masks['distance']

    def _load_mask(self, mask_dir, image_name):
        mask_path = os.path.join(mask_dir, f"{image_name}.png")
        if not os.path.exists(mask_path):
            return np.zeros(self.input_image_reshape, dtype=np.uint8)
        mask = cv2.imread(mask_path, 0)
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)
        return cv2.resize(mask_remap, self.input_image_reshape)
    # def _load_semantic_cluster_mask(self, mask_dir, image_name):
    #     mask_path = os.path.join(mask_dir, f"{image_name}.png")
    #     if not os.path.exists(mask_path):
    #         return np.zeros(self.input_image_reshape, dtype=np.uint8)
    #     mask = cv2.imread(mask_path, 0)
    #     # mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)
    #     return cv2.resize(mask_remap, self.input_image_reshape)
    def _load_distance(self, mask_dir, image_name):
        mask_path = os.path.join(mask_dir, f"{image_name}.tif")
        if not os.path.exists(mask_path):
            return np.zeros(self.input_image_reshape, dtype=np.float16)
        mask = iio.imread(mask_path)
        # mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.float16)
        return cv2.resize(mask, self.input_image_reshape)
    def _load_instance(self, instance_dir, image_name):
        mask_path = os.path.join(instance_dir, f"{image_name}.png")
        if not os.path.exists(mask_path):
            return np.zeros(self.input_image_reshape, dtype=np.uint16)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)
        return cv2.resize(mask, self.input_image_reshape)
    def __len__(self):
        return len(self.ids)

class CroplandDatasetCOCO(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        coco_json_path,
        input_image_reshape=(1024, 1024),
        foreground_class=1,
        augmentation=True,
        if_test=False,
        data_type='RGB',
    ):
        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.if_test = if_test
        self.input_image_type = data_type

        valid_exts = ('.tif', '.jpg', '.png')
        self.ids = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
        self.ext_type = os.path.splitext(self.ids[0])[1]
        self.images_filepaths = [os.path.join(images_dir, img_id) for img_id in self.ids]

        self.masks_semantic_dir = os.path.join(masks_dir, 'Semantic')
        self.masks_edge_dir = os.path.join(masks_dir, 'Edge')
        self.masks_distance_dir = os.path.join(masks_dir, 'Distance')
        self.masks_instance_dir = os.path.join(masks_dir, 'channel_3')

        # --- 加载 COCO 格式注释 ---
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        self.image_id_to_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        self.filename_to_img_id = {img["file_name"].split('.')[0]: img["id"] for img in self.coco_data["images"]}

    def _load_mask(self, mask_dir, image_name):
        mask_path = os.path.join(mask_dir, image_name + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(mask, self.input_image_reshape[::-1], interpolation=cv2.INTER_NEAREST)

    def _load_instance(self, mask_dir, image_name):
        mask_path = os.path.join(mask_dir, image_name + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(mask, self.input_image_reshape[::-1], interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, i):
        image_name = self.ids[i].split('.')[0]
        image_path = self.images_filepaths[i]

        # 图像读取与处理
        if self.ext_type == '.tif':
            image = iio.imread(image_path)
        else:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = image[:, :, :3] if self.input_image_type == 'RGB' else image
        image = cv2.resize(image, self.input_image_reshape[::-1], interpolation=cv2.INTER_LINEAR)

        # 读取 mask
        masks = {
            'semantic': self._load_mask(self.masks_semantic_dir, image_name),
            'edge': self._load_mask(self.masks_edge_dir, image_name),
            'distance': self._load_mask(self.masks_distance_dir, image_name),
            'instance': self._load_instance(self.masks_instance_dir, image_name)
        }

        # 转换为 Tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        masks = {k: np.expand_dims(v, axis=0) for k, v in masks.items()}
        masks = {k: torch.from_numpy(v).float() for k, v in masks.items()}

        # 获取 COCO 注释
        image_id = self.filename_to_img_id[image_name]
        annotations = self.image_id_to_annotations.get(image_id, [])

        bboxes = []
        labels = []
        areas = []
        for ann in annotations:
            bbox = ann["bbox"]  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            bboxes.append([x1, y1, x2, y2])
            labels.append(ann.get("category_id", 1))  # 默认前景类
            areas.append(ann.get("area", w * h))

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "image_id": torch.tensor([image_id]),
            "iscrowd": torch.zeros(len(bboxes), dtype=torch.uint8)
        }

        # 返回结构
        result = {
            'image': image_tensor,
            'mask_semantic': masks['semantic'],
            'mask_edge': masks['edge'],
            'mask_distance': masks['distance'],
            'mask_instance': masks['instance'],
            'target': target
        }

        if self.if_test:
            result["image_name"] = os.path.basename(image_path)

        return result

    def __len__(self):
        return len(self.ids)
        # image_tensor, masks['semantic'], masks['edge'], masks['distance']

    def _load_mask(self, mask_dir, image_name):
        mask_path = os.path.join(mask_dir, f"{image_name}.png")
        mask = cv2.imread(mask_path, 0)
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)
        return cv2.resize(mask_remap, self.input_image_reshape)
    def _load_instance(self, instance_dir, image_name):
        mask_path = os.path.join(instance_dir, f"{image_name}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)
        return cv2.resize(mask, self.input_image_reshape)
    def __len__(self):
        return len(self.ids)


# ========== 自定义增强类 ==========
class ResizeWithMasks:
    def __init__(self, size):
        self.size = size  # (H, W)
        self.resize_image = Resize(size)

    def __call__(self, inputs):
        image, masks = inputs  # image: Tensor, masks: Dict[str, Tensor]

        # Resize image
        image = self.resize_image(image)

        # Resize masks
        resized_masks = {}
        for k, v in masks.items():
            # v: shape (1, H, W) or (H, W) → convert to 1CH for interpolation
            if v.dim() == 2:
                v = v.unsqueeze(0)
            elif v.dim() == 3 and v.shape[0] != 1:
                raise ValueError(f"Unexpected mask shape: {v.shape}")
            
            # Use nearest to avoid soft edges in segmentation masks
            resized_v = F.interpolate(v.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
            resized_masks[k] = resized_v

        return image, resized_masks
# class RandomSpatialTransform:
#     def __init__(self, rotation_range=(-15, 15), flip_prob=0.5, scale_range=(0.95, 1.05)):
#         self.rotation_range = rotation_range
#         self.flip_prob = flip_prob
#         self.scale_range = scale_range

#     def __call__(self, data):
#         img, masks = data
#         params = self._get_random_params()

#         # 应用几何变换
#         img = self._apply_transform(img, **params, interpolation=TF.InterpolationMode.BILINEAR)
        
#         # 为不同掩膜应用不同插值
#         new_masks = {}
#         for mask_type, mask in masks.items():
#             interp = TF.InterpolationMode.NEAREST if mask_type != 'distance' \
#                     else TF.InterpolationMode.BILINEAR
#             new_masks[mask_type] = self._apply_transform(mask, **params, interpolation=interp)
        
#         return img, new_masks

#     def _get_random_params(self):
#         return {
#             'angle': random.uniform(*self.rotation_range),
#             'scale': random.uniform(*self.scale_range),
#             'hflip': random.random() < self.flip_prob,
#             'vflip': random.random() < self.flip_prob
#         }

#     def _apply_transform(self, tensor, angle, scale, hflip, vflip, interpolation):
#         # 缩放
#         if scale != 1.0:
#             h, w = tensor.shape[-2:]
#             new_h, new_w = int(h * scale), int(w * scale)
#             tensor = TF.resize(tensor, (new_h, new_w), interpolation=interpolation)
#             tensor = TF.center_crop(tensor, (h, w))

#         # 旋转
#         tensor = TF.rotate(tensor, angle, interpolation=interpolation, fill=0)

#         # 翻转
#         if hflip:
#             tensor = TF.hflip(tensor)
#         if vflip:
#             tensor = TF.vflip(tensor)

#         return tensor

# class ColorJitter:
#     def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
#         self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

#     def __call__(self, data):
#         img, masks = data  # img 是 Tensor 或 PIL.Image，视你数据而定

#         # 如果是 Tensor 且通道数大于 3
#         if isinstance(img, torch.Tensor):
#             if img.shape[0] > 3:
#                 rgb = img[:3, :, :]
#                 rest = img[3:, :, :]
#                 rgb_jittered = self.jitter(rgb)
#                 img = torch.cat([rgb_jittered, rest], dim=0)
#             else:
#                 img = self.jitter(img)

        # # 如果是 PIL 图像，需要转为 RGB 再转回原格式（仅适用于通道为 4 时）
        # elif isinstance(img, Image.Image):
        #     if img.mode == 'RGBA':
        #         img = img.convert('RGB')
        #         img = self.jitter(img)
        #     elif img.mode == 'RGB':
        #         img = self.jitter(img)
        #     else:
        #         # 其他通道格式如 'L', 'I', 多通道 Tiff 等，视任务决定是否处理
        #         pass

        return img, masks
def save_images(dataset, output_dir, num_samples=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_samples):
        image, mask_semantic, mask_edge, mask_distance, image_name = dataset[i]['image'], dataset[i]['mask_semantic'], dataset[i]['mask_edge'], dataset[i]['mask_distance'], dataset[i]['image_name']
        image_name = os.path.splitext(image_name)[0]
        # Convert tensor to numpy for saving as an image
        image = image.permute(1, 2, 0).numpy()*255
        mask_semantic = mask_semantic.numpy()
        mask_edge = mask_edge.numpy()
        mask_distance = mask_distance.numpy()

        # Save image and masks to files
        image_filename = os.path.join(output_dir, f'{image_name}_image.png')
        mask_semantic_filename = os.path.join(output_dir, f'{image_name}_semantic_mask.png')
        mask_edge_filename = os.path.join(output_dir, f'{image_name}_edge_mask.png')
        mask_distance_filename = os.path.join(output_dir, f'{image_name}_distance_mask.png')

        # Save images
        cv2.imwrite(image_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_semantic_filename, mask_semantic)
        cv2.imwrite(mask_edge_filename, mask_edge)
        cv2.imwrite(mask_distance_filename, mask_distance)
        
        print(f"Saved {image_filename}, {mask_semantic_filename}, {mask_edge_filename}, {mask_distance_filename}")


if __name__ == '__main__':
    data_path = '/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/mask/test'
    # Test the dataset and saving functionality
    images_dir = "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/test/MultiSpectral"
    masks_dir = data_path
    output_dir = "/mnt/disk3/har/DataSet/Cropland/Ai4Boundaries/Processed_data/Edge_Sample/dataloader_test"

    dataset = CroplandDataset(images_dir=images_dir, masks_dir=masks_dir, input_image_reshape=(1024, 1024), if_test=True , data_type='MultiSpectral')
    a = dataset.__getitem__(0)
    save_images(dataset, output_dir, num_samples=5)