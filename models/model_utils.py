"""
模型工具函数

主要包含：
- LoRA 目标模块查找
"""

import torch


def find_lora_target_modules(model, exclude_keywords=None, num_modules=-1, verbose=True):
    """
    查找模型中可以应用 LoRA 的 Linear 层

    LoRA（Low-Rank Adaptation）是一种参数高效微调方法，只训练少量参数。
    这个函数用于找出模型中所有的 Linear 层，排除掉不需要训练的部分。

    Args:
        model: PyTorch 模型
        exclude_keywords: 额外要排除的模块名关键词列表（会与下面的默认排除词合并，
            而不是互相替代——lm_head 等默认排除项始终生效，不会因为调用方传入了
            自己的排除列表就被意外撤销）
            默认排除视觉编码器和 lm_head，因为：
            - 视觉编码器通常已经训练好了，不需要再调
            - lm_head 是输出层，一般也不需要 LoRA
        num_modules: 只返回最后 N 个模块（-1 表示全部）
        verbose: 是否打印找到的模块列表

    Returns:
        模块名列表，可以直接传给 LoraConfig 的 target_modules 参数
    """
    # 默认排除的模块，始终生效；调用方传入的 exclude_keywords 是在此基础上的追加项
    default_exclude_keywords = [
        "visual",           # 视觉编码器
        "vision_model",     # 视觉模型
        "img_projection",   # 图像投影层
        "lm_head",          # 语言模型输出层
    ]
    if exclude_keywords is None:
        exclude_keywords = default_exclude_keywords
    else:
        exclude_keywords = list(set(default_exclude_keywords) | set(exclude_keywords))

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
