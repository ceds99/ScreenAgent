"""
合并 LoRA 权重脚本

训练完成后，LoRA 权重和基础模型是分开保存的。
这个脚本把 LoRA 权重合并到基础模型中，生成一个完整的模型。

使用方法:
    python scripts/merge_weights.py --exp_dir logs/stage1/2026-01-30_xx-xx-xx
"""

import argparse
import os
import re
import sys
import glob
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoModelForVision2Seq

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import find_lora_target_modules


def parse_args():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重")
    parser.add_argument(
        '--exp_dir',
        type=str,
        required=True,
        help="实验目录，比如 logs/stage1/2026-01-30_xx-xx-xx"
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default=None,
        help="基础模型路径，默认从 args.json 读取"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="输出目录，默认是 <exp_dir>/ckpt_model/merged_model"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="加载模型用的设备，显存不够可以用 cpu"
    )
    return parser.parse_args()


def find_weight_file(ckpt_dir):
    """
    在检查点目录里找训练好的权重文件

    DeepSpeed 存出来的结构是 ckpt_model/global_step<N>/mp_rank_00_model_states.pt，
    训练跑了几个 epoch 就会留下几个 global_step 目录，要取步数最大的那个。

    Args:
        ckpt_dir: ckpt_model 目录

    Returns:
        权重文件路径，找不到返回 None
    """
    # 1. 普通的 pytorch_model.bin
    direct = os.path.join(ckpt_dir, 'pytorch_model.bin')
    if os.path.isfile(direct):
        return direct

    # 2. DeepSpeed 的 model_states 文件
    candidates = glob.glob(
        os.path.join(ckpt_dir, '**', '*model_states*.pt'), recursive=True
    )
    if not candidates:
        return None

    def sort_key(path):
        match = re.search(r'global_step(\d+)', path)
        step = int(match.group(1)) if match else -1
        return (step, os.path.getmtime(path))

    candidates.sort(key=sort_key)
    return candidates[-1]


def main():
    args = parse_args()

    # 读取训练时的参数
    args_json_path = os.path.join(args.exp_dir, 'args.json')
    if not os.path.exists(args_json_path):
        print(f"[错误] 找不到 {args_json_path}")
        print("  --exp_dir 要指向实验目录（里面有 args.json 和 ckpt_model/），")
        print("  比如 logs/stage1/2026-01-30_20-46-53")
        sys.exit(1)

    with open(args_json_path, 'r') as f:
        train_args = json.load(f)

    print("=" * 60)
    print("合并 LoRA 权重")
    print("=" * 60)
    print(f"实验目录: {args.exp_dir}")

    # 确定基础模型路径
    if args.base_model:
        base_model_path = args.base_model
    elif train_args.get('local_weight'):
        base_model_path = train_args.get('local_weight_dir', '')
        # 查找包含 config.json 的目录
        if os.path.isdir(base_model_path) and not os.path.isfile(
                os.path.join(base_model_path, 'config.json')):
            for root, dirs, files in os.walk(base_model_path):
                if 'config.json' in files:
                    base_model_path = root
                    break
    else:
        base_model_path = train_args.get('model_id', 'Qwen/Qwen2.5-VL-3B-Instruct')

    print(f"基础模型: {base_model_path}")

    # LoRA 权重路径
    ckpt_dir = os.path.join(args.exp_dir, 'ckpt_model')
    weight_path = find_weight_file(ckpt_dir)

    if weight_path is None:
        print(f"[错误] 在 {ckpt_dir} 下找不到权重文件")
        print("  期望是 pytorch_model.bin，或 DeepSpeed 的 global_step*/*model_states*.pt")
        if os.path.isdir(ckpt_dir):
            print(f"  目录下实际内容: {sorted(os.listdir(ckpt_dir))}")
        sys.exit(1)

    print(f"权重文件: {weight_path}")

    # 输出路径
    save_path = args.output or os.path.join(ckpt_dir, 'merged_model')
    print(f"输出路径: {save_path}")
    print("")

    # 确定数据类型
    precision = train_args.get('precision', 'bf16')
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # 加载 processor
    print("[1/5] 加载 processor...")
    min_pixels = train_args.get('min_visual_tokens', 256) * 28 * 28
    max_pixels = train_args.get('max_visual_tokens', 1280) * 28 * 28
    model_max_length = train_args.get('model_max_length', 4096)

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    processor.tokenizer.model_max_length = model_max_length

    # 加载基础模型
    print("[2/5] 加载基础模型...")
    attn_impl = train_args.get('attn_imple', 'sdpa')
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        device_map=args.device,
    )
    model.config.use_cache = False

    # 配置 LoRA
    print("[3/5] 配置 LoRA...")
    lora_r = train_args.get('lora_r', 8)
    lora_alpha = train_args.get('lora_alpha', 16)
    lora_dropout = train_args.get('lora_dropout', 0.05)
    # 排除规则要跟训练时一样，不然 target_modules 对不上，strict=False 又不报错，
    # 最后合出来一个没学到东西的模型
    tune_visual_encoder = train_args.get('tune_visual_encoder', False)

    if lora_r > 0:
        exclude_modules = [] if tune_visual_encoder else ["visual"]
        target_modules = find_lora_target_modules(
            model, exclude_keywords=exclude_modules, verbose=False
        )
        print(f"  target_modules: {len(target_modules)} 个 "
              f"(tune_visual_encoder={tune_visual_encoder})")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("  lora_r=0，跳过 LoRA 配置")

    # 加载训练好的权重
    print("[4/5] 加载训练好的权重...")
    state_dict = torch.load(weight_path, map_location="cpu")

    # DeepSpeed 的 state_dict 可能需要处理
    if isinstance(state_dict, dict) and 'module' in state_dict:
        state_dict = state_dict['module']

    # 移除可能的前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'module.' 前缀
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    # 先核对一下 LoRA 权重能不能对上号。load_state_dict 用的是 strict=False，
    # key 对不上它不报错，直接跳过，合出来的模型跟基座一模一样，这种很难查
    model_keys = set(model.state_dict().keys())
    ckpt_lora_keys = [k for k in new_state_dict if 'lora_' in k]
    matched_lora_keys = [k for k in ckpt_lora_keys if k in model_keys]

    print(f"  检查点里的 LoRA 权重: {len(ckpt_lora_keys)} 个")
    print(f"  能和当前模型对上的: {len(matched_lora_keys)} 个")

    if lora_r > 0 and len(matched_lora_keys) == 0:
        print("")
        print("[错误] 没有任何 LoRA 权重能对上，合并会得到和基座一样的模型。")
        print("  可能原因：base_model 和训练时用的不是同一个；")
        print("            lora_r / tune_visual_encoder 和训练时不一致。")
        if ckpt_lora_keys:
            print(f"  检查点里的 key 示例: {ckpt_lora_keys[:3]}")
        model_lora_keys = [k for k in model_keys if 'lora_' in k]
        if model_lora_keys:
            print(f"  当前模型的 key 示例: {model_lora_keys[:3]}")
        sys.exit(1)

    load_result = model.load_state_dict(new_state_dict, strict=False)
    unexpected = getattr(load_result, 'unexpected_keys', [])
    if unexpected:
        print(f"  [提示] 有 {len(unexpected)} 个权重不属于当前模型，已忽略")

    # 合并 LoRA 权重
    print("[5/5] 合并并保存模型...")
    if lora_r > 0:
        model = model.merge_and_unload()

    # 保存合并后的模型
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=False)
    processor.save_pretrained(save_path)

    print("")
    print("=" * 60)
    print("合并完成！")
    print(f"模型保存在: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
