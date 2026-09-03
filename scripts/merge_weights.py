"""
合并 LoRA 权重脚本

训练完成后，LoRA 权重和基础模型是分开保存的。
这个脚本把 LoRA 权重合并到基础模型中，生成一个完整的模型。

使用方法:
    python scripts/merge_weights.py --exp_dir logs/stage1/2025-01-30_xx-xx-xx
"""

import argparse
import os
import sys
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
        help="实验目录，比如 logs/stage1/2025-01-30_xx-xx-xx"
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default=None,
        help="基础模型路径，默认从 args.json 读取"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 读取训练时的参数
    args_json_path = os.path.join(args.exp_dir, 'args.json')
    if not os.path.exists(args_json_path):
        print(f"[错误] 找不到 {args_json_path}")
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
        if os.path.isdir(base_model_path):
            for root, dirs, files in os.walk(base_model_path):
                if 'config.json' in files:
                    base_model_path = root
                    break
    else:
        base_model_path = train_args.get('model_id', 'Qwen/Qwen2.5-VL-3B-Instruct')

    print(f"基础模型: {base_model_path}")

    # LoRA 权重路径
    ckpt_dir = os.path.join(args.exp_dir, 'ckpt_model')
    weight_path = os.path.join(ckpt_dir, 'pytorch_model.bin')

    if not os.path.exists(weight_path):
        # DeepSpeed 保存的检查点格式可能不同，尝试其他路径
        # 查找 mp_rank_00_model_states.pt
        for root, dirs, files in os.walk(ckpt_dir):
            for f in files:
                if 'model_states' in f:
                    weight_path = os.path.join(root, f)
                    break

    if not os.path.exists(weight_path):
        print(f"[错误] 找不到权重文件")
        print(f"  尝试的路径: {os.path.join(ckpt_dir, 'pytorch_model.bin')}")
        sys.exit(1)

    print(f"权重文件: {weight_path}")

    # 输出路径
    save_path = os.path.join(ckpt_dir, 'merged_model')
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
        device_map="cuda:0",
    )
    model.config.use_cache = False

    # 配置 LoRA
    print("[3/5] 配置 LoRA...")
    lora_r = train_args.get('lora_r', 8)
    lora_alpha = train_args.get('lora_alpha', 16)
    lora_dropout = train_args.get('lora_dropout', 0.05)

    if lora_r > 0:
        # 找到 LoRA 目标模块
        exclude_modules = ["visual"]
        target_modules = find_lora_target_modules(model, exclude_keywords=exclude_modules, verbose=False)

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
    if 'module' in state_dict:
        state_dict = state_dict['module']

    # 移除可能的前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'module.' 前缀
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

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
