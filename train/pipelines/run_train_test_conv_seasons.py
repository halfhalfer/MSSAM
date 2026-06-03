import os
import subprocess
import argparse

def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser("Train-Test Pipeline with Seasons")

    # ========= 通用参数 =========
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--in_channels", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multispectral_encoder", type=str, default="Conv")
    
    # ========= 训练参数 =========
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--max_epoch_num", type=int, default=50)
    parser.add_argument("--debug", type=str, default="True")

    # ========= 测试参数 =========
    parser.add_argument("--test_output_root", type=str, required=True, help="测试结果的总根目录")

    # ========= 设备 =========
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()

    # ========== 1. 执行训练 (保持不变) ==========
    train_cmd = [
        "python", "train/train_cropland_multispectral_ForReview_args_fix.py",
        "--gpu", str(args.gpu),
        "--output", args.train_output,
        "--dataset", args.dataset,
        "--in_channels", str(args.in_channels),
        "--multispectral-encoder", args.multispectral_encoder,
        "--seed", str(args.seed),
        "--debug", args.debug,
        "--max_epoch_num", str(args.max_epoch_num),
    ]
    # 如果你只是想对现有模型跑测试，可以注释掉下面这一行
    # run(train_cmd)

    # ========== 2. 自动定位 Checkpoints ==========
    decoder_ckpt = os.path.join(args.train_output, "best_model_decoder.pth")
    # 如果你的模型有 LoRA 分支
    vit_ckpt = os.path.join(args.train_output, "best_model_lora_multiSpectral.pth")

    if not os.path.exists(decoder_ckpt):
        raise FileNotFoundError(f"Missing {decoder_ckpt}")

    # ========== 3. 自动循环执行 春、夏、秋、冬 测试 ==========
    seasons = ["spring", "summer", "autumn", "winter"]
    
    for season in seasons:
        print(f"\n>>> Starting Test for Season: {season} <<<")
        
        # 为每个季节创建独立的子文件夹
        season_output = os.path.join(args.test_output_root, season)
        os.makedirs(season_output, exist_ok=True)

        test_cmd = [
            "python", "train/test_multispectral_args_fix.py",
            "--gpu", str(args.gpu),
            "--output", season_output,          # 季节性输出路径
            "--dataset", args.dataset,
            "--in_channels", str(args.in_channels),
            "--multispectral-encoder", args.multispectral_encoder,
            "--seed", str(args.seed),
            "--season", season,                 # 传入季节参数
            "--restore_model", decoder_ckpt,
            "--restore_model_multispectral", vit_ckpt
        ]
        
        try:
            run(test_cmd)
        except subprocess.CalledProcessError:
            print(f"Error occurred during {season} testing, skipping to next season.")

    print("\n[DONE] All seasons testing finished.")

if __name__ == "__main__":
    main()