"""
模型工具函数

主要包含：
- LoRA 目标模块查找
- 模型参数统计
- 其他辅助函数
"""

import torch


def find_lora_target_modules(model, exclude_keywords=None, num_modules=-1, verbose=True):
    """
    查找模型中可以应用 LoRA 的 Linear 层

    LoRA（Low-Rank Adaptation）是一种参数高效微调方法，只训练少量参数。
    这个函数用于找出模型中所有的 Linear 层，排除掉不需要训练的部分。

    Args:
        model: PyTorch 模型
        exclude_keywords: 要排除的模块名关键词列表
            默认排除视觉编码器和 lm_head，因为：
            - 视觉编码器通常已经训练好了，不需要再调
            - lm_head 是输出层，一般也不需要 LoRA
        num_modules: 只返回最后 N 个模块（-1 表示全部）
        verbose: 是否打印找到的模块列表

    Returns:
        模块名列表，可以直接传给 LoraConfig 的 target_modules 参数
    """
    # 默认排除的模块
    if exclude_keywords is None:
        exclude_keywords = [
            "visual",           # 视觉编码器
            "vision_model",     # 视觉模型
            "img_projection",   # 图像投影层
            "lm_head",          # 语言模型输出层
        ]

    linear_cls = torch.nn.Linear
    target_modules = []

    for name, module in model.named_modules():
        # 跳过排除列表中的模块
        if any(keyword in name for keyword in exclude_keywords):
            continue

        # 只要 Linear 层
        if isinstance(module, linear_cls):
            target_modules.append(name)

    # 只取最后 N 个（如果指定了的话）
    if num_modules > 0:
        target_modules = target_modules[-num_modules:]

    if verbose:
        print(f"[LoRA] 找到 {len(target_modules)} 个目标模块")
        # 打印前几个和后几个
        if len(target_modules) > 6:
            print(f"  前3个: {target_modules[:3]}")
            print(f"  后3个: {target_modules[-3:]}")
        else:
            print(f"  {target_modules}")

    return target_modules


def count_parameters(model, trainable_only=False):
    """
    统计模型参数数量

    Args:
        model: PyTorch 模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model):
    """
    打印模型的可训练参数统计

    会显示：
    - 总参数数量
    - 可训练参数数量
    - 可训练参数占比
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"[模型参数]")
    print(f"  总参数: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"  可训练: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"  占比: {100 * trainable_params / total_params:.2f}%")


def freeze_module(module):
    """
    冻结模块的所有参数（不参与训练）

    Args:
        module: 要冻结的模块
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """
    解冻模块的所有参数（参与训练）

    Args:
        module: 要解冻的模块
    """
    for param in module.parameters():
        param.requires_grad = True
