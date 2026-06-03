import os
import subprocess
import argparse

def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    parser = argparse.ArgumentParser("Train-Test Pipeline")

    # ========= 通用实验参数 =========
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--in_channels", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multispectral_encoder", type=str, default="Conv")
    # ========= 训练参数 =========
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--max_epoch_num", type=int, default=50)
    parser.add_argument("--debug_trainstep", type=int, default=2000)
    parser.add_argument("--debug_valstep", type=int, default=200)
    parser.add_argument("--debug", type=str, default=True)

    # ========= 测试参数 =========
    parser.add_argument("--test_output", type=str, required=True)

    # ========= 设备 =========
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()

    # ========== 1. 训练 ==========
    train_cmd = [
        "python", "train/train_cropland_multispectral_ForReview_args_fix.py",
        "--gpu", str(args.gpu),
        "--output", args.train_output,

        "--dataset", args.dataset,
        "--in_channels", str(args.in_channels),
        "--multispectral-encoder", args.multispectral_encoder,
        "--seed", str(args.seed),
        
        "--debug", str(args.debug),
        "--max_epoch_num", str(args.max_epoch_num),
        "--debug_trainstep", str(args.debug_trainstep),
        "--debug_valstep", str(args.debug_valstep),
    ]
    run(train_cmd)

    # debug 用
    # ========== 2. 自动定位 checkpoint ==========
    decoder_ckpt = os.path.join(args.train_output, "best_model_decoder.pth")
    vit_ckpt = os.path.join(args.train_output, "best_model_lora_multiSpectral.pth")


    if not os.path.exists(decoder_ckpt):
        raise FileNotFoundError(f"Missing {decoder_ckpt}")

    # ========== 3. 测试 ==========
    test_cmd = [
        "python", "train/test_multispectral_args_fix.py",
        "--gpu", str(args.gpu),
        "--output", args.test_output,

        "--dataset", args.dataset,
        "--in_channels", str(args.in_channels),
        "--multispectral-encoder", args.multispectral_encoder,
        "--seed", str(args.seed),

        "--restore_model", decoder_ckpt,
        "--restore_model_multispectral", vit_ckpt
    ]
    run(test_cmd)


    

if __name__ == "__main__":
    main()
