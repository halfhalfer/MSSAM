#!/bin/bash

DATASET="S4A"
GPU=4
BASE_OUT="/home/huar/Param/MSSAM_review_Exp/S4A/ForReview_Exp/New_Adapter_Conv/Model_Param"

for SEED in 100
do
    echo "Running experiment with SEED $SEED"
    
    python train/pipelines/run_train_test_conv_seasons.py \
        --dataset $DATASET \
        --gpu $GPU \
        --in_channels 13 \
        --max_epoch_num 30 \
        --seed $SEED \
        --train_output "$BASE_OUT/Seed_$SEED/train" \
        --test_output_root "$BASE_OUT/Seed_$SEED/test_seasons" \
        --debug True
done