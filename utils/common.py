"""
通用工具函数

包含一些常用的辅助函数，比如：
- 参数保存/加载
- JSON 文件操作
- 日志目录创建
"""

import os
import json


def save_args_to_json(args, filepath):
    """
    把命令行参数保存到 JSON 文件

    方便后续查看训练时用了什么参数

    Args:
        args: argparse 解析的参数对象
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def load_args_from_json(filepath):
    """
    从 JSON 文件加载参数

    Args:
        filepath: JSON 文件路径

    Returns:
        参数字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    """
    保存数据到 JSON 文件

    Args:
        data: 要保存的数据（字典或列表）
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath):
    """
    加载 JSON 文件

    Args:
        filepath: JSON 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_log_dir(base_dir):
    """
    创建日志目录

    如果目录已存在，会自动添加后缀（_1, _2, ...）

    Args:
        base_dir: 基础目录路径

    Returns:
        实际创建的目录路径
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    # 目录已存在，添加后缀
    idx = 1
    new_dir = f"{base_dir}_{idx}"
    while os.path.exists(new_dir):
        idx += 1
        new_dir = f"{base_dir}_{idx}"

    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def ensure_dir(path):
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_timestamp():
    """
    获取当前时间戳字符串

    Returns:
        格式化的时间字符串，如 "2025-01-29_15-30-00"
    """
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
