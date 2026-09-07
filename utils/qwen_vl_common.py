"""
Qwen2.5-VL 相关的共享逻辑

训练（train_dataset.py）、评估（evaluator.py）、推理（inference.py）三处原来
各自独立实现了一份 smart_resize / 坐标转换 / chat template，容易在只改其中一处时
造成三处行为不一致（参见代码审查问题清单 M3/M4/M5）。这里统一成唯一实现。

设计约束：本模块只依赖标准库（math、re），不引入 torch/deepspeed/wandb 等重型依赖，
这样 inference.py 仍可以保持"轻量部署、不依赖完整训练框架"的特性。
"""

import math
import re


# Qwen2.5-VL 的 chat template（训练和推理必须完全一致，否则输入格式会有细微差异）
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def smart_resize(height, width, factor=28, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
    """
    Qwen2.5-VL 的图像智能 resize 逻辑

    保持长宽都是 factor 的倍数，同时控制总像素数在 [min_pixels, max_pixels] 之间。

    Raises:
        ValueError: 图片过小（小于一个 factor）或长宽比过于极端（>200:1）
    """
    if height < factor or width < factor:
        raise ValueError(f"图片太小: {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"长宽比太极端: {height}x{width}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def point_to_qwen_coords(point, orig_height, orig_width, factor=28,
                         min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
    """
    把归一化坐标（0~1，[x, y]）转换成 Qwen2.5-VL resize 后图像上的绝对像素坐标

    训练时用来构造标签：模型的标准答案是"resize 后图像上的绝对坐标"。
    """
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)

    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    x, y = point
    x_new = round(x * orig_width * scale_w)
    y_new = round(y * orig_height * scale_h)

    x_new = max(0, min(x_new, new_width - 1))
    y_new = max(0, min(y_new, new_height - 1))

    return [x_new, y_new]


def qwen_coords_to_point(point, orig_height, orig_width, factor=28,
                         min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
    """
    把 Qwen2.5-VL 输出的坐标（resize 后图像上的绝对像素坐标）转换回原图的绝对像素坐标

    评估、推理时用来把模型输出还原成原图坐标。
    """
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)

    scale_w = orig_width / new_width
    scale_h = orig_height / new_height

    x_resized, y_resized = point
    x_orig = round(x_resized * scale_w)
    y_orig = round(y_resized * scale_h)

    x_orig = max(0, min(x_orig, orig_width - 1))
    y_orig = max(0, min(y_orig, orig_height - 1))

    return [x_orig, y_orig]


def extract_coordinates(text):
    """
    从模型生成的文本中提取形如 [x, y] 的坐标子串

    模型输出有时会夹杂自然语言修饰（比如"点击位置是[512, 300]"），
    这里先尝试正则提取，找不到再交给调用方回退处理（比如整段 ast.literal_eval）。

    Returns:
        提取到的坐标字符串（如 "[512, 300]"），找不到则返回 None
    """
    match = re.search(r'\[\s*[\d\.]+\s*,\s*[\d\.]+\s*\]', text)
    if match:
        return match.group(0)
    return None
