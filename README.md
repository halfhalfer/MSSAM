# Spectral Representation-Enhanced Multitask SAM for Cropland Parcel Delineation

Official codebase for **Spectral Representation-Enhanced Multitask SAM for Cropland Parcel Delineation**.


## Overview

This project is built around a multitask training and inference pipeline for cropland parcel delineation. The model predicts:

- semantic parcel masks
- parcel boundaries
- auxiliary instance cues for structure-aware training

The final parcel visualization is generated from the semantic, boundary, and parcel post-processing outputs together.

## Resources

The main resources for this project will be released on Hugging Face, including:

- the processed dataset used in this work
- the trained model checkpoints produced in this project, corresponding to the local folder `Trained_Param/`

Planned links:

- Dataset: `TBD` (processed from the publicly available [Sen4AgriNet / S4A](https://github.com/Orion-AI-Lab/S4A) and [AI4Boundaries](http://data.europa.eu/89h/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9) datasets; we gratefully acknowledge the original dataset providers)
- Trained model checkpoints: https://huggingface.co/halfhalfer/mssam-cropland-checkpoints

After downloading the dataset, please organize the files according to the paths expected by the training scripts in this repository. 


### Expected Dataset Folder Tree

The exact released package may include additional metadata files, but the expected structure should follow a layout similar to:

```text
AI4Boundaries/
├── train/
│   ├── images/
│   └── mask/
│       ├── Semantic/
│       ├── Edge/
│       └── channel_3/
├── val/
│   ├── images/
│   └── mask/
│       ├── Semantic/
│       ├── Edge/
│       └── channel_3/
└── test/
    ├── images/
    └── mask/
        ├── Semantic/
        ├── Edge/
        └── channel_3/
```

Expected label meanings:

- `Semantic`: binary semantic parcel mask
- `Edge`: parcel boundary mask
- `channel_3`: parcel-level instance / object annotation

### Important Note for Custom Datasets

If you use your own dataset, you should also update the dataset-specific normalization statistics in:

- [train/segment_anything_training/build_channelViT.py](train/segment_anything_training/build_channelViT.py)

In particular, check and modify the entries in `DATASET_PIXEL_STATS`, including:

- `in_channels`
- `pixel_mean`
- `pixel_std`

Incorrect dataset statistics can have a large impact on model behavior and final performance, especially for multispectral inputs.

## Checkpoint Preparation

### SAM Backbone Checkpoint

The backbone initialization comes from the official SAM ViT-B checkpoint released by Meta:

- SAM project page: https://segment-anything.com
- Official repository: https://github.com/facebookresearch/segment-anything
- Common filename: `sam_vit_b_01ec64.pth`

In the current codebase, training and testing scripts accept the backbone path through `--checkpoint`. If you keep the default local layout used in this repo, place the backbone checkpoint under:

- `sam-hq-param/`


### Pretrained Mask Decoder

The pretrained mask decoder should come from [SysCV/SAM-HQ](https://github.com/SysCV/SAM-HQ):

- `sam_vit_b_maskdecoder.pth`
- Place this file under the project folder `sam-hq-param/`

This decoder checkpoint is separate from the SAM backbone checkpoint above. Please place or configure it according to your local training setup before launching training.

## Training and Testing

The recommended entry point for the main paper pipeline is:

- [train/pipelines/run_train_test.py](train/pipelines/run_train_test.py)

This script runs the complete training and testing pipeline.

### Example Command

You can adapt the following example for your own run:

```bash
python train/pipelines/run_train_test.py \
  --dataset AI4Boundaries \
  --in_channels 4 \
  --use_channelToken True \
  --use_orth_loss True \
  --pathEmbed_v 2_5_2 \
  --seed 1234 \
  --max_epoch_num 51 \
  --debug False \
  --visualize False \
  --edgeloss_v 6 \
  --pe_lr_scale True \
  --train_output /path/to/train_output \
  --test_output /path/to/test_output \
  --gpu 0
```

## Output Structure

A typical training + inference run produces both model checkpoints and image outputs. An example test output folder is:

- [smoke_runs/a4b_existing_ckpt_test](smoke_runs/a4b_existing_ckpt_test)

### Model Outputs

The main training pipeline saves two core model checkpoints:

- `best_model_decoder.pth`: the trained task decoder checkpoint
- `best_model_lora_multiSpectral.pth`: the checkpoint containing the multispectral encoder together with the LoRA parameters used to adapt the SAM backbone

In the inference command:

- `--restore_model` should point to `best_model_decoder.pth`
- `--checkpoint_vit` should point to `best_model_lora_multiSpectral.pth`

### Image Outputs

Typical test / inference image outputs include:

- `test_hq_mask`: predicted semantic masks
- `test_hq_mask_logits`: raw or intermediate semantic prediction exports
- `test_hq_edge`: predicted boundary maps
- `test_hq_edge_logits`: raw or intermediate boundary prediction exports
- `test_hq_instance`: auxiliary instance prediction results used mainly for training / debugging visualization
- `test_hq_instance_rgb`: colorized auxiliary instance visualization
- `parcels/bin`: final parcel instance maps used for parcel-level evaluation
- `parcels/color_result`: final parcel visualization results
- `smokerun`: example evaluation outputs and smoke-run artifacts

Important note:

- `test_hq_instance` is an auxiliary visualization result rather than the final parcel output.
- `parcels` contains the final parcel-level delineation outputs.
- If you want the final parcel delineation result, focus on `parcels/bin` and `parcels/color_result`.

## Evaluation

The migrated AI4Boundaries evaluation script is:

- [scripts/a4b_metrics_eval.py](scripts/a4b_metrics_eval.py)

It currently supports:

- semantic mask evaluation
- boundary evaluation
- instance / parcel-level evaluation

An example smoke-run output is available at:

- [smoke_runs/a4b_existing_ckpt_test/smokerun/metrics_a4b_eval](smoke_runs/a4b_existing_ckpt_test/smokerun/metrics_a4b_eval)

## Acknowledgements

This codebase is built on top of and inspired by the following excellent projects:

- SAM-HQ: https://github.com/SysCV/SAM-HQ
- SAM: https://segment-anything.com
- Diverse Channel ViT: https://github.com/chaudatascience/diverse_channel_vit

We thank the authors and contributors of these projects for their valuable work.
