# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT,ChannelImageEncoderViT , replace_qkv_with_lora
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .multispectral_encoder import MultispectralEncoder_Conv, MultispectralEncoder_ViT


class SamMultiSpectralChannelToken(nn.Module): # TODO 1.输入编码暂时使用简单的Covolution，后续需要改成ViT（再分是否经过MAE训练，）2. 中间层使用LoRA微调 3. 解码层暂时保持HQ的那个解码
    mask_threshold: float = 0.0
    image_format: str = "MultiSpectral"
    def __init__(
        self,
        image_encoder: ChannelImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean:List[float] = [0.18806978, 0.20257292,0.16138335,0.47405199], #List[float] = [123.675, 116.28, 103.53], RGB
        pixel_std: List[float] = [0.09411742,0.08356307,0.07693374,0.13019605], #List[float] = [58.395, 57.12, 57.375],RGB
        use_lora: bool = True,
    ) -> None:
        """
        SAM predicts object masks from an multi-spectral image and input prompts.

        Arguments:
          multispectral_encoder (nn.Module): The pre-encoder used to encode the multi-spectral image into a 3-dimensional space.
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        # scale mean and std to 0-255
        pixel_mean_255 = [m * 255 for m in pixel_mean]
        pixel_std_255 = [s * 255 for s in pixel_std]

        self.mask_decoder = mask_decoder
        self.use_lora = use_lora
        #Set LoRA
        if self.use_lora:
          replace_qkv_with_lora(self.image_encoder,r=8,lora_alpha=16)
          # for param in self.image_encoder.parameters():
          #     param.requires_grad = False
          for name, param in self.image_encoder.named_parameters():
              if not name.startswith("patch_embed"):
                  param.requires_grad = False
          for name, param in self.image_encoder.named_parameters():
              if "lora_A" in name or "lora_B" in name:
                  param.requires_grad = True
          
          for param in self.prompt_encoder.parameters():
              param.requires_grad = False
          for name, param in self.mask_decoder.named_parameters():
              param.requires_grad = False
            
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean_255).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std_255).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward_with_interm_feats(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, interm_embeddings,ortho_loss = self.image_encoder(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # with torch.no_grad():
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            boxes = image_record.get("boxes", None)
            if boxes is not None:
                boxes = boxes
            mask_inputs = image_record.get("mask_inputs", None)
            if mask_inputs is not None:
                mask_inputs = mask_inputs
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_inputs,
            )
            sparse_embeddings = sparse_embeddings#.detach()
            dense_embeddings = dense_embeddings#.detach()
            image_pe = self.prompt_encoder.get_dense_pe()#.detach()
            # with torch.autocast(device_type='cuda', dtype=torch.float16), torch.no_grad():
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),#.detach(),  # 关键：分离编码器输出
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )

            # with torch.no_grad():
            masks = self.postprocess_masks(
                low_res_masks.float(),  # 转换回FP32用于阈值处理
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                    "ortho_loss":ortho_loss,
                }
            )

        return outputs, interm_embeddings
    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        train_model:bool=True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        image_embeddings, interm_embeddings,ortho_loss = self.image_encoder(input_images,train_model)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                    'ortho_loss':ortho_loss,
                }
            )
        return outputs, interm_embeddings

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class SamMultiSpectral(nn.Module): # TODO 1.输入编码暂时使用简单的Covolution，后续需要改成ViT（再分是否经过MAE训练，）2. 中间层使用LoRA微调 3. 解码层暂时保持HQ的那个解码
    mask_threshold: float = 0.0
    image_format: str = "MultiSpectral"

    def __init__(
        self,
        multispectral_encoder: nn.Module,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean:List[float] = [0.18806978, 0.20257292,0.16138335,0.47405199], #List[float] = [123.675, 116.28, 103.53], RGB
        pixel_std: List[float] = [0.09411742,0.08356307,0.07693374,0.13019605], #List[float] = [58.395, 57.12, 57.375],RGB
    ) -> None:
        """
        SAM predicts object masks from an multi-spectral image and input prompts.

        Arguments:
          multispectral_encoder (nn.Module): The pre-encoder used to encode the multi-spectral image into a 3-dimensional space.
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.multispectral_encoder = multispectral_encoder
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        # scale mean and std to 0-255
        pixel_mean_255 = [m * 255 for m in pixel_mean]
        pixel_std_255 = [s * 255 for s in pixel_std]

        self.mask_decoder = mask_decoder

        #Set LoRA
        replace_qkv_with_lora(self.image_encoder,r=8,lora_alpha=16)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for name, param in self.image_encoder.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
        
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for name, param in self.mask_decoder.named_parameters():
            param.requires_grad = False
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean_255).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std_255).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward_with_interm_feats(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_3d = self.multispectral_encoder(input_images) # encoder multi-spectral image
        if isinstance(image_3d, tuple):# ViT MultiSpectral Encoder 
            _, _, image_3d = image_3d
      
        image_embeddings, interm_embeddings = self.image_encoder(image_3d)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # with torch.no_grad():
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            boxes = image_record.get("boxes", None)
            if boxes is not None:
                boxes = boxes
            mask_inputs = image_record.get("mask_inputs", None)
            if mask_inputs is not None:
                mask_inputs = mask_inputs
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_inputs,
            )
            sparse_embeddings = sparse_embeddings#.detach()
            dense_embeddings = dense_embeddings#.detach()
            image_pe = self.prompt_encoder.get_dense_pe()#.detach()
            # with torch.autocast(device_type='cuda', dtype=torch.float16), torch.no_grad():
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),#.detach(),  # 关键：分离编码器输出
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )

            # with torch.no_grad():
            masks = self.postprocess_masks(
                low_res_masks.float(),  # 转换回FP32用于阈值处理
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold


            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                    "multispectral_in_3d":image_3d,
                }
            )

        return outputs, interm_embeddings
    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        image_3d = self.multispectral_encoder(input_images) # encoder multi-spectral image
        if isinstance(image_3d, tuple):# ViT MultiSpectral Encoder 
            _, _, image_3d = image_3d
        image_embeddings, interm_embeddings = self.image_encoder(image_3d)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                    "multispectral_in_3d":image_3d,
                }
            )

        return outputs, interm_embeddings

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        use_lora: bool = False,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.use_lora = use_lora
        #Set LoRA
        if self.use_lora:
          replace_qkv_with_lora(self.image_encoder,r=8,lora_alpha=16)
          # for param in self.image_encoder.parameters():
          #     param.requires_grad = False
          for name, param in self.image_encoder.named_parameters():
              if not name.startswith("patch_embed"):
                  param.requires_grad = False
          for name, param in self.image_encoder.named_parameters():
              if "lora_A" in name or "lora_B" in name:
                  param.requires_grad = True
          
          for param in self.prompt_encoder.parameters():
              param.requires_grad = False
          for name, param in self.mask_decoder.named_parameters():
              param.requires_grad = False

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        image_embeddings, interm_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                }
            )

        return outputs, interm_embeddings

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
