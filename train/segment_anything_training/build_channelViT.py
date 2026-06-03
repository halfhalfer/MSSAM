# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, SamMultiSpectral, MultispectralEncoder_Conv, MultispectralEncoder_ViT, ChannelTokenImageEncoderViT,CBAMUNet , ChannelCompressViT,SamMultiSpectralChannelToken #,manually_load_qkv_lora_weights
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


DATASET_PIXEL_STATS = {
    "S4A": {
        "in_channels": 13,
        "pixel_mean": [
            0.24978395, 0.13515302, 0.17456125, 0.16935185, 0.28596567,
            0.34511887, 0.33905805, 0.32499178, 0.41162524, 0.44552976,
            0.3670932, 0.28490318, 0.35702788,
        ],
        "pixel_std": [
            0.11355606, 0.07113282, 0.08557547, 0.09546637, 0.11946635,
            0.1455717, 0.14316145, 0.14242597, 0.1516711, 0.13336884,
            0.14251775, 0.12708959, 0.14493368,
        ],
    },
    "AI4Boundaries": {
        "in_channels": 4,
        "pixel_mean": [0.18806978, 0.20257292, 0.16138335, 0.47405199],
        "pixel_std": [0.09411742, 0.08356307, 0.07693374, 0.13019605],
    },
    "AI4Boundaries_orth": {
        "in_channels": 3,
        "pixel_mean": [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
        "pixel_std": [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
    },
}


def get_dataset_normalization_stats(dataset_name, in_channels):
    if dataset_name not in DATASET_PIXEL_STATS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}' for multispectral normalization. "
            f"Supported datasets: {sorted(DATASET_PIXEL_STATS.keys())}"
        )

    stats = DATASET_PIXEL_STATS[dataset_name]
    expected_in_channels = stats["in_channels"]
    if in_channels != expected_in_channels:
        raise ValueError(
            f"Dataset '{dataset_name}' expects in_channels={expected_in_channels}, "
            f"but got in_channels={in_channels}."
        )

    return stats["pixel_mean"], stats["pixel_std"]

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


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam_vit_h_MultiSpectral(checkpoint=None,checkpoint_vit=None,in_channels=4,use_lora=True,use_orth_loss=True,args=None):
    return _build_sam_MultiSpectral(
        # Ori SAM Param
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint_vit = checkpoint_vit,#multispectralEncoder and lora
        checkpoint=checkpoint,
        in_channels = in_channels,
        use_lora=use_lora,
        use_orth_loss=use_orth_loss,
        args=args
    )

def build_sam_vit_l_MultiSpectral(checkpoint=None,checkpoint_vit=None,in_channels=4,use_lora=True,use_orth_loss=True,args=None):
    return _build_sam_MultiSpectral(
        # Ori SAM Param
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint_vit = checkpoint_vit,#multispectralEncoder and lora
        checkpoint=checkpoint,
        in_channels = in_channels,
        use_lora=use_lora,
        use_orth_loss=use_orth_loss,
        args=args
    )

def build_sam_vit_b_MultiSpectral(checkpoint=None,checkpoint_vit=None,in_channels=4,use_lora=True,use_orth_loss=True,args=None):
    return _build_sam_MultiSpectral(
        # Ori SAM Param
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint_vit = checkpoint_vit,#multispectralEncoder and lora
        checkpoint=checkpoint,
        in_channels = in_channels,
        use_lora=use_lora,
        use_orth_loss=use_orth_loss,
        args=args
    )


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

    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

sam_model_registry_channelToken = {
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
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

def _build_sam_MultiSpectral(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    checkpoint_vit=None,
    in_channels=4,
    use_lora=True,
    use_orth_loss=True,
    args=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    dataset_name = getattr(args, "dataset", None)
    if dataset_name is None:
        raise ValueError("args.dataset is required to build the multispectral model")
    mean_list, std_list = get_dataset_normalization_stats(dataset_name, in_channels)
    
    sam = SamMultiSpectralChannelToken(
        image_encoder=ChannelTokenImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            in_chans=in_channels,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            use_orth_loss = use_orth_loss,
            use_channelToken = args.use_channelToken,
            pathEmbed_v = getattr(args, 'pathEmbed_v', '2'),
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
        use_lora=use_lora,
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
        # 去除patch embeding
        image_encoder_state_dict = {
            k: v for k, v in image_encoder_state_dict.items()
            if not (k.startswith("patch_embed")) }#or k.startswith("pos_embed")
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

    if checkpoint_vit is not None :
        with open(checkpoint_vit, "rb") as f:
            state_dict = torch.load(f)
        if use_lora:
            patch_embed_dict = {
                k.replace("image_encoder.",""): v for k, v in state_dict.items()
                if "patch_embed" in k or "pos_embed" in k  
            }
            missing_patch_embed = sam.image_encoder.load_state_dict(patch_embed_dict, strict=False)
            # print("\n missing patch_embed: ", missing_patch_embed)
            # 分离 ViT 中的 LoRA 参数 (适配替换后的结构)
            lora_dict = {
                k: v for k, v in state_dict.items()
                if "lora_A" in k or "lora_B" in k  
            }
            missing_lora = sam.load_state_dict(lora_dict, strict=False)
            print("\nLoad Lora Encoder in ViT:")
        else:
            param_dict = {
                k.replace("image_encoder.",""): v for k, v in state_dict.items()
            }
            missing = sam.image_encoder.load_state_dict(param_dict, strict=False)
    return sam
