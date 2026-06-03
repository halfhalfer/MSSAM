# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, SamMultiSpectral, MultispectralEncoder_Conv, MultispectralEncoder_ViT, CBAMUNet , ChannelCompressViT #,manually_load_qkv_lora_weights
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )

def build_sam_vit_b(checkpoint=None,use_lora=False,checkpoints_lora=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        use_lora=use_lora,
        checkpoints_lora=checkpoints_lora
    )

def build_sam_vit_h_MultiSpectral(checkpoint=None,multispectral_encoder_type='Conv',checkpoint_multispectral=None,checkpoint_pretrained_multispectral_encoder=None,in_channels=4 ):
    return _build_sam_MultiSpectral(
        multispectral_encoder_type=multispectral_encoder_type,
        # Ori SAM Param
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint_multispectral = checkpoint_multispectral,#multispectralEncoder and lora
        checkpoint=checkpoint,
        checkpoints_pretrained_multispectral_encoder = checkpoint_pretrained_multispectral_encoder,
        in_channels = in_channels,
    )

def build_sam_vit_l_MultiSpectral(checkpoint=None,multispectral_encoder_type='Conv',checkpoint_multispectral=None,checkpoint_pretrained_multispectral_encoder=None,in_channels=4 ):
    return _build_sam_MultiSpectral(
        multispectral_encoder_type=multispectral_encoder_type,
        # Ori SAM Param
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        
        checkpoint_multispectral = checkpoint_multispectral,#multispectralEncoder and lora
        checkpoint=checkpoint,
        checkpoints_pretrained_multispectral_encoder = checkpoint_pretrained_multispectral_encoder,
        in_channels = in_channels,
    )

def build_sam_vit_b_MultiSpectral(checkpoint=None,multispectral_encoder_type='Conv',checkpoint_multispectral=None,checkpoint_pretrained_multispectral_encoder=None,in_channels=4 ):
    return _build_sam_MultiSpectral(
        multispectral_encoder_type=multispectral_encoder_type,
        # Ori SAM Param
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint_multispectral = checkpoint_multispectral,#multispectralEncoder and lora
        checkpoint=checkpoint,
        checkpoints_pretrained_multispectral_encoder = checkpoint_pretrained_multispectral_encoder,
        in_channels = in_channels,
    )
# def build_sam_vit_b(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=768,
#         encoder_depth=12,
#         encoder_num_heads=12,
#         encoder_global_attn_indexes=[2, 5, 8, 11],
#         checkpoint=checkpoint,
#     )

def build_multispectral_encoder(encoder_type, **kwargs):
    if encoder_type == 'Conv':
        return MultispectralEncoder_Conv(
            in_channels=kwargs['params']['in_channels'],
            out_channels=3,
            kernel_size=3,
        )
    elif encoder_type == 'CBAM':
        return CBAMUNet(
            in_channels=kwargs['params']['in_channels'],
            out_channels=3,
        )
    elif encoder_type == 'ViT': #TODO 此处ViT 使用l版本的权重
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        # image_embedding_size = image_size // vit_patch_size
        encoder_embed_dim=768
        encoder_depth=12
        encoder_num_heads=12
        encoder_global_attn_indexes=[2, 5, 8, 11]
        
    # Channel Embed Bug?
        return ChannelCompressViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            in_chans=4,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            use_channel_embedding=False,
        )
        # return ImageEncoderViT(
        #     depth=encoder_depth,
        #     embed_dim=encoder_embed_dim,
        #     img_size=image_size,
        #     mlp_ratio=4,
        #     norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        #     num_heads=encoder_num_heads,
        #     patch_size=vit_patch_size,
        #     qkv_bias=True,
        #     use_rel_pos=True,
        #     global_attn_indexes=encoder_global_attn_indexes,
        #     window_size=14,
        #     out_chans=prompt_embed_dim,
        # )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    # encoder_embed_dim=768,
    #     encoder_depth=12,
    #     encoder_num_heads=12,
    #     encoder_global_attn_indexes=[2, 5, 8, 11],
    #     checkpoint=checkpoint,
#     prompt_embed_dim = 256
#     image_size = 1024
#     vit_patch_size = 16
#     image_embedding_size = image_size // vit_patch_size
#     sam = Sam(
#         image_encoder=ImageEncoderViT(
#             depth=encoder_depth,
#             embed_dim=encoder_embed_dim,
#             img_size=image_size,
#             mlp_ratio=4,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             num_heads=encoder_num_heads,
#             patch_size=vit_patch_size,
#             qkv_bias=True,
#             use_rel_pos=True,
#             global_attn_indexes=encoder_global_attn_indexes,
#             window_size=14,
#             out_chans=prompt_embed_dim,
#         ),
sam_model_registry = {
    "default": build_sam,
    "vit_h_MultiSpectral": build_sam_vit_h_MultiSpectral,
    "vit_l_MultiSpectral": build_sam_vit_l_MultiSpectral,
    "vit_b_MultiSpectral": build_sam_vit_b_MultiSpectral,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    use_lora=False,
    checkpoints_lora=None
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        use_lora=use_lora,
    )
    # sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        # 字段筛选
        image_encoder_state_dict = {
            k.replace("image_encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("image_encoder.")
        }

        prompt_encoder_state_dict = {
            k.replace("prompt_encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("prompt_encoder.")
        }

        mask_decoder_state_dict = {
            k.replace("mask_decoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("mask_decoder.")
        }
        missing = sam.image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        missing = sam.prompt_encoder.load_state_dict(prompt_encoder_state_dict, strict=False)
        missing = sam.mask_decoder.load_state_dict(mask_decoder_state_dict, strict=False)
    if checkpoints_lora is not None:
        with open(checkpoints_lora, "rb") as f:
            state_dict = torch.load(f)
        missing_multi = sam.load_state_dict(state_dict, strict=False)
        # print("Load Encoder:")
        # print("Missing keys:", missing_multi.missing_keys)
        # print("Unexpected keys:", missing_multi.unexpected_keys)
    return sam


def _build_sam_MultiSpectral(
    multispectral_encoder_type,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint_multispectral,
    checkpoint=None,
    checkpoints_pretrained_multispectral_encoder=None,
    in_channels=4,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    
    if multispectral_encoder_type == 'Conv':
        mulitspectral_encoder_config ={
            "type": "Conv",
            "params": {
                "in_channels": in_channels,
            }
        }
    elif multispectral_encoder_type == 'CBAM':
        mulitspectral_encoder_config ={
            "type": "CBAM",
            "params": {
                "in_channels": in_channels,
            }
        }
    elif multispectral_encoder_type == 'ViT': 
        mulitspectral_encoder_config= {
            "type": "ViT",
            "params": { 
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "global_attn_indexes": [2, 5, 8, 11] 
            }
        }
    multispectral_encoder = build_multispectral_encoder(
        encoder_type=multispectral_encoder_type,
        **mulitspectral_encoder_config,
    )
    
    if in_channels == 13:
        mean_list = [0.24978395,0.13515302, 0.17456125, 0.16935185, 0.28596567, 0.34511887,0.33905805, 0.32499178, 0.41162524, 0.44552976, 0.3670932,  0.28490318,0.35702788]
        std_list = [0.11355606, 0.07113282, 0.08557547, 0.09546637, 0.11946635, 0.1455717,0.14316145, 0.14242597, 0.1516711,  0.13336884, 0.14251775, 0.12708959,0.14493368]
    else:
        mean_list = [0.18806978, 0.20257292,0.16138335,0.47405199] #List[float] = [123.675, 116.28, 103.53], RGB
        std_list = [0.09411742,0.08356307,0.07693374,0.13019605] #List[float] = [58.395, 57.12, 57.375],RGB
    # print("mean_list:", mean_list)
    sam = SamMultiSpectral(
        multispectral_encoder=multispectral_encoder,
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=mean_list,
        pixel_std=std_list,
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        # 字段筛选
        image_encoder_state_dict = {
            k.replace("image_encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("image_encoder.")
        }

        prompt_encoder_state_dict = {
            k.replace("prompt_encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("prompt_encoder.")
        }

        mask_decoder_state_dict = {
            k.replace("mask_decoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("mask_decoder.")
        }
        missing = sam.image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        missing = sam.prompt_encoder.load_state_dict(prompt_encoder_state_dict, strict=False)
        missing = sam.mask_decoder.load_state_dict(mask_decoder_state_dict, strict=False)
        # print("SAM Missing keys (not found in the state_dict):", missing.missing_keys)
        # print("SAM Unexpected keys (not used by the model):", missing.unexpected_keys)
    
    # 如果 预训练的 multispectral_encoder 存在，则先加载；
    if checkpoints_pretrained_multispectral_encoder is not None:
        with open(checkpoints_pretrained_multispectral_encoder, "rb") as f:
            state_dict = torch.load(f)
        
        missing_multi = sam.multispectral_encoder.load_state_dict(state_dict, strict=False)
        print("Load Multispectral Encoder:")
        print("Missing keys:", missing_multi.missing_keys)
        print("Unexpected keys:", missing_multi.unexpected_keys)

    if checkpoint_multispectral is not None :
        with open(checkpoint_multispectral, "rb") as f:
            state_dict = torch.load(f)
        # 不严格读取
        multispectral_dict = {k.replace("multispectral_encoder.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("multispectral_encoder.")
        }

        # 先加载 multispectral_encoder
        missing_multi = sam.multispectral_encoder.load_state_dict(multispectral_dict, strict=False)
        print("Load Multispectral Encoder:")
        print("Missing keys:", missing_multi.missing_keys)
        print("Unexpected keys:", missing_multi.unexpected_keys)

        # 分离 ViT 中的 LoRA 参数 (适配替换后的结构)
        lora_dict = {
            k: v for k, v in state_dict.items()
            if "lora_A" in k or "lora_B" in k  # 根据实际参数名调整
        }
        missing_lora = sam.load_state_dict(lora_dict, strict=False)
        print("\nLoad Lora Encoder in ViT:")
    else:
        if multispectral_encoder_type == 'ViT':
            # param load
            checkpoint_path = "/mnt/disk3/har/DataSet/HQSeg/sam-hq-training/pretrained_checkpoint/sam_vit_b_01ec64.pth" # SAM ViT-B
            state_dict = torch.load(checkpoint_path, map_location='cuda')
            vit_prefix = 'image_encoder.'  
            vit_state_dict = {
                k[len(vit_prefix):]: v for k, v in state_dict.items() if k.startswith(vit_prefix)
            }
            vit_state_dict = {
                k: v for k, v in vit_state_dict.items()
                if not (k.startswith("patch_embed") or k.startswith("pos_embed") or k.startswith("group_fusion") or k.startswith("decoder")) #or k.startswith("pos_embed")
            }

            # 加载参数，strict=False 以防有部分不匹配（比如额外添加了 channel_embed）
            missing_keys, unexpected_keys = sam.multispectral_encoder.load_state_dict(vit_state_dict, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        # print("Missing keys:", missing_lora.missing_keys)
        # print("Unexpected keys:", missing_lora.unexpected_keys)
    return sam
