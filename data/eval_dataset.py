"""
评估数据集

支持的评估集：
- ScreenSpot: 标准 GUI Grounding 评估集
- ScreenSpot-v2: 改进版评估集（修复了标注问题）
"""

import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset

from .template import build_eval_prompt


# 数据集目录名映射
EVAL_DATASET_DIR_MAP = {
    "screenspot": "ScreenSpot",
    "screenspot2": "ScreenSpot-v2",
}


class ScreenSpotDataset(Dataset):
    """
    ScreenSpot 评估数据集

    评估任务：给定元素描述，预测点击坐标，判断是否落在目标元素的 bbox 内
    """

    def __init__(self, dataset_dir, dataset_name, json_file, processor, args_dict=None):
        """
        初始化评估数据集

        Args:
            dataset_dir: 数据集根目录（比如 datasets/eval/）
            dataset_name: 数据集名称（screenspot 或 screenspot2）
            json_file: 元数据 JSON 文件名
            processor: Qwen-VL processor
            args_dict: 额外参数
        """
        self.processor = processor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels

        # 数据集路径
        actual_dir_name = EVAL_DATASET_DIR_MAP.get(dataset_name, dataset_name)
        self.base_dir = os.path.join(dataset_dir, actual_dir_name)
        self.image_dir = os.path.join(self.base_dir, "images")
        meta_dir = os.path.join(self.base_dir, "metadata")

        # 加载元数据
        json_path = os.path.join(meta_dir, f"{json_file}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 解析参数
        args_dict = args_dict or {}
        self.xy_int = args_dict.get('xy_int', False)

        print(f"[评估数据集] {dataset_name}: {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._get_sample(idx)

    def _get_sample(self, idx):
        """获取一条评估样本"""
        item = self.data[idx]

        # 加载图片
        if 'img_url' in item:
            image_path = os.path.join(self.image_dir, item['img_url'])
            image = Image.open(image_path).convert('RGB')
            image_list = [image]
            item['img_url_abs'] = image_path
        else:
            image_list = None
            item['img_url_abs'] = ""

        # 获取任务描述
        task = item['task']

        # 构建 prompt
        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        messages = build_eval_prompt(task, img_dict, self.xy_int)

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(text=prompt, images=image_list, return_tensors="pt")

        # 评估时不需要 labels，但为了兼容性还是加上
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids']

        data_dict = {
            'input_ids': batch['input_ids'][0],
            'labels': batch['labels'][0],
            'pixel_values': batch['pixel_values'],
            'image_sizes': batch['image_grid_thw'],
        }

        return data_dict, item


class TrainingEvalDataset(Dataset):
    """
    训练数据评估集

    用于在训练过程中评估模型在训练数据上的表现
    """

    def __init__(self, dataset_dir, dataset_name, json_file, processor, args_dict=None):
        """
        初始化

        和训练数据集类似，但用于评估（不做数据增强）
        """
        self.processor = processor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels

        # 使用训练数据的目录映射
        from .train_dataset import DATASET_DIR_MAP
        actual_dir_name = DATASET_DIR_MAP.get(dataset_name, dataset_name)

        # 特殊处理 training-eval
        if dataset_name == "training-eval":
            actual_dir_name = "Training-data"

        # 训练数据评估集需要从 train 目录读取，而不是 eval 目录
        # 把 eval 替换成 train
        if '/eval' in dataset_dir:
            dataset_dir = dataset_dir.replace('/eval', '/train')

        self.base_dir = os.path.join(dataset_dir, actual_dir_name)
        self.image_dir = os.path.join(self.base_dir, "images")
        meta_dir = os.path.join(self.base_dir, "metadata")

        # 加载元数据
        json_path = os.path.join(meta_dir, f"{json_file}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 解析参数
        args_dict = args_dict or {}
        self.xy_int = args_dict.get('xy_int', False)

        print(f"[训练评估数据集] {dataset_name}: {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._get_sample(idx)

    def _get_sample(self, idx):
        """获取一条样本"""
        import random

        item = self.data[idx]

        # 加载图片
        if 'img_url' in item:
            image_path = os.path.join(self.image_dir, item['img_url'])
            image = Image.open(image_path).convert('RGB')
            image_list = [image]
            item['img_url_abs'] = image_path
        else:
            image_list = None
            item['img_url_abs'] = ""

        # 固定随机种子选择元素
        random.seed(idx)
        element_idx = random.randint(0, len(item['element']) - 1)
        task = item['element'][element_idx]['instruction']

        # 构建 prompt
        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        messages = build_eval_prompt(task, img_dict, self.xy_int)

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(text=prompt, images=image_list, return_tensors="pt")

        if 'labels' not in batch:
            batch['labels'] = batch['input_ids']

        data_dict = {
            'input_ids': batch['input_ids'][0],
            'labels': batch['labels'][0],
            'pixel_values': batch['pixel_values'],
            'image_sizes': batch['image_grid_thw'],
        }

        # 只保留选中的元素
        item['element'] = [item['element'][element_idx]]

        return data_dict, item
