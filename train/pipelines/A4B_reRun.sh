#!/bin/bash

# 1. 定义你想要运行的种子列表
SEEDS=(1234)  # (1234 12345) 

# 2. 基础路径设置（方便统一修改）
BASE_OUT_DIR="/home/huar/Param/MSSAM_review_Exp/A4B/ForReview_Exp/re_run_other_model/MainRe"
SUFFIX="LR1e3_RCF_StructLossFix_PELRScale"

for SEED in "${SEEDS[@]}"
do
    echo "========================================"
    echo "Starting training with SEED: $SEED"
    echo "========================================"

    # 动态生成输出路径
    CUR_TRAIN_OUT="${BASE_OUT_DIR}/Seed_${SEED}_${SUFFIX}/train"
    CUR_TEST_OUT="${BASE_OUT_DIR}/Seed_${SEED}_${SUFFIX}/test"

    # 执行 Python 命令
    python train/pipelines/run_train_test.py \
        --dataset AI4Boundaries \
        --in_channels 4 \
        --use_channelToken True \
        --use_orth_loss True \
        --pathEmbed_v 2_5_2 \
        --seed $SEED \
        --max_epoch_num 51 \
        --debug True \
        --debug_trainstep 4000 \
        --debug_valstep 700 \
        --visualize False \
        --edgeloss_v 6 \
        --pe_lr_scale True \
        --train_output "$CUR_TRAIN_OUT" \
        --test_output "$CUR_TEST_OUT" \
        --gpu 1

    echo "Finished SEED: $SEED"
done
