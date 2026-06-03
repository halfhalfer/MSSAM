import os
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM_HQ_PARAM_DIR = PROJECT_ROOT / "sam-hq-param"
SAM_BACKBONE_CKPT = SAM_HQ_PARAM_DIR / "sam_hq_vit_b.pth"

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
    parser.add_argument("--use_channelToken", type=str2bool, default=True)
    parser.add_argument("--use_orth_loss", type=str2bool, default=False)
    parser.add_argument("--pathEmbed_v", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # ========= 训练参数 =========
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--max_epoch_num", type=int, default=10)
    parser.add_argument("--debug_trainstep", type=int, default=20)
    parser.add_argument("--debug_valstep", type=int, default=20)
    parser.add_argument("--debug", type=str, default=True)
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--pe_lr_scale", type=str2bool, default=False)
    parser.add_argument("--edgeloss_v", type=int, default=5)
    # ========= 测试参数 =========
    parser.add_argument("--test_output", type=str, required=True)


    # ========= 设备 =========
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # ========== 1. 训练 ==========
    train_cmd = [
        "python", "train/train_cropland_multispectral_channelToken_args_fix.py",
        "--gpu", str(args.gpu),
        "--output", args.train_output,

        "--dataset", args.dataset,
        "--in_channels", str(args.in_channels),
        "--use_channelToken", str(args.use_channelToken),
        "--use_orth_loss", str(args.use_orth_loss),
        "--pathEmbed_v", args.pathEmbed_v,
        "--seed", str(args.seed),
        
        "--learning_rate", "1e-3",
        "--pe_lr_scale", str(args.pe_lr_scale),
        "--visualize", str(args.visualize),
        "--debug", str(args.debug),
        "--max_epoch_num", str(args.max_epoch_num),
        "--debug_trainstep", str(args.debug_trainstep),
        "--debug_valstep", str(args.debug_valstep),
        "--edgeloss_v", str(args.edgeloss_v),
        "--checkpoint", str(SAM_BACKBONE_CKPT),
    ]
    run(train_cmd)

    # debug 用
    # args.train_output = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/ChannelToken/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss'
    # ========== 2. 自动定位 checkpoint ==========
    decoder_ckpt = os.path.join(args.train_output, "best_model_decoder.pth")
    vit_ckpt = os.path.join(args.train_output, "best_model_lora_multiSpectral.pth")

    if not os.path.exists(decoder_ckpt):
        raise FileNotFoundError(f"Missing {decoder_ckpt}")

    # ========== 3. 测试 ==========
    test_cmd = [
        "python", "train/test_multispectral_channelToken_args_fix.py",
        "--gpu", str(args.gpu),
        "--output", args.test_output,

        "--dataset", args.dataset,
        "--in_channels", str(args.in_channels),
        "--use_channelToken", str(args.use_channelToken),
        "--use_orth_loss", str(args.use_orth_loss),
        "--pathEmbed_v", args.pathEmbed_v,
        "--seed", str(args.seed),

        "--restore_model", decoder_ckpt,
        "--checkpoint_vit", vit_ckpt,
        "--checkpoint", str(SAM_BACKBONE_CKPT),
    ]
    run(test_cmd)




if __name__ == "__main__":
    main()
