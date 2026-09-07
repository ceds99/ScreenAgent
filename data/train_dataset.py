"""
训练数据集

支持的数据集：
- ShowUI-desktop: 桌面界面截图
- ShowUI-web-8k: Web 界面截图
- AMEX-8k: 移动应用截图
- UGround-V1-8k: 多分辨率 Web 截图
"""

import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from .template import build_grounding_prompt, add_answer_to_batch, add_multiturn_answer
from utils.qwen_vl_common import smart_resize, point_to_qwen_coords


# 数据集目录名映射（数据集参数名 -> 实际目录名）
DATASET_DIR_MAP = {
    "showui-desktop": "ShowUI-desktop",
    "showui-web": "ShowUI-web-8k",
    "amex": "AMEX-8k",
    "uground": "UGround-V1-8k",
}


class GroundingTrainDataset(Dataset):
    """
    GUI Grounding 训练数据集

    每个样本包含：
    - 一张屏幕截图
    - 一个或多个 UI 元素的描述和对应坐标

    训练任务：给定元素描述，预测点击坐标
    """

    def __init__(self, dataset_dir, dataset_name, json_file, processor, args_dict=None):
        """
        初始化数据集

        Args:
            dataset_dir: 数据集根目录（比如 datasets/train/）
            dataset_name: 数据集名称（showui-desktop, showui-web, amex, uground）
            json_file: 元数据 JSON 文件名（不含 .json 后缀）
            processor: Qwen-VL processor
            args_dict: 额外参数字典
        """
        self.processor = processor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels

        # 数据集路径
        actual_dir_name = DATASET_DIR_MAP.get(dataset_name, dataset_name)
        self.base_dir = os.path.join(dataset_dir, actual_dir_name)
        self.image_dir = os.path.join(self.base_dir, "images")
        meta_dir = os.path.join(self.base_dir, "metadata")

        # 加载元数据
        json_path = os.path.join(meta_dir, f"{json_file}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 解析参数
        args_dict = args_dict or {}
        self.num_turn = args_dict.get('num_turn', 1)           # 多轮对话轮数
        self.shuffle_prompt = args_dict.get('shuffle_image_token', False)
        self.uniform_prompt = args_dict.get('uniform_prompt', False)
        self.xy_int = args_dict.get('xy_int', False)           # 坐标是否用整数
        self.crop_min = args_dict.get('crop_min', 1.0)         # 随机裁剪最小比例
        self.crop_max = args_dict.get('crop_max', 1.0)         # 随机裁剪最大比例
        self.model_id = args_dict.get('model_id', 'Qwen/Qwen2.5-VL-3B-Instruct')

        # 任务类型采样概率（默认只做 text2point）
        self.sample_prob = np.array([
            args_dict.get('text2point', 1),
            args_dict.get('text2bbox', 0),
            args_dict.get('point2text', 0),
            args_dict.get('bbox2text', 0)
        ])
        self.sample_prob = self.sample_prob / self.sample_prob.sum()

        self.dataset_name = dataset_name
        print(f"[训练数据集] {dataset_name}: {len(self.data)} 条样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取一条训练样本

        返回的时候会自动处理异常，如果某个样本加载失败就随机换一个
        """
        max_retry = 10
        for _ in range(max_retry):
            try:
                return self._get_sample(idx)
            except Exception as e:
                print(f"[警告] 加载样本 {idx} 失败: {e}")
                idx = random.randint(0, len(self.data) - 1)

        raise RuntimeError(f"连续 {max_retry} 次加载失败")

    def _get_sample(self, idx):
        """实际的样本加载逻辑"""
        idx = idx % len(self.data)
        item = self.data[idx]

        # 加载图片
        if 'img_url' in item:
            image_path = os.path.join(self.image_dir, item['img_url'])
            image = Image.open(image_path).convert('RGB')
            image_list = [image]

            # 随机裁剪（数据增强）
            if self.crop_min != 1.0 or self.crop_max != 1.0:
                image, item = self._random_crop(image, item, (self.crop_min, self.crop_max))
                image_list = [image]
        else:
            image_list = None

        # 随机选择任务类型（text2point, text2bbox, point2text, bbox2text）
        # 注意：这里刻意不绑定 idx 相关的随机种子——只有单轮模式下的元素选择
        # （下面 random.seed(idx) 那一支）需要强可复现性，方便调试时对照同一个
        # idx 反复检查同一条样本；任务类型选择和多轮模式的元素选择允许在
        # 同一个 idx 被重复访问时产出不同内容，增加数据多样性，是有意为之，不是遗漏。
        task_type = np.random.choice(len(self.sample_prob), p=self.sample_prob)

        # 选择要定位的元素（支持多轮）
        elements = item['element']
        num_elements = min(self.num_turn, len(elements))
        assert num_elements > 0, "样本中没有元素"

        if self.num_turn == 1:
            # 单轮：固定随机种子保证可复现
            random.seed(idx)
            element_idx = random.randint(0, len(elements) - 1)
            selected_elements = [elements[element_idx]]
        else:
            # 多轮：随机选择多个元素（有意不绑定种子，见上面的说明）
            selected_elements = random.choices(elements, k=num_elements)

        # 构建训练数据
        if len(selected_elements) == 1:
            data_dict = self._build_single_turn(
                item, selected_elements[0], image_list, task_type
            )
        else:
            data_dict = self._build_multi_turn(
                item, selected_elements, image_list, task_type
            )

        return data_dict, item

    def _build_single_turn(self, item, element, image_list, task_type):
        """构建单轮训练数据"""
        element_name = element['instruction']

        # 获取答案坐标
        # point 任务的坐标系统必须和 evaluator.py / inference.py 保持一致：
        # 对 Qwen2.5-VL，统一使用 resize 后的绝对像素坐标（而不是归一化坐标），
        # 与 _build_multi_turn 的处理方式相同，避免同一次训练里单轮/多轮样本
        # 各自产出不同坐标系统的标签。
        if task_type in [0, 2]:  # point 任务
            if 'Qwen2.5-VL' in self.model_id:
                answer_xy = self._convert_point_for_qwen(
                    element['point'], item['img_size'][1], item['img_size'][0]
                )
            elif self.xy_int:
                answer_xy = [int(x * 1000) for x in element['point']]
            else:
                answer_xy = [round(x, 2) for x in element['point']]
        else:  # bbox 任务
            answer_xy = element['bbox']
            if self.xy_int:
                answer_xy = [int(x * 1000) for x in answer_xy]
            else:
                answer_xy = [round(x, 2) for x in answer_xy]

        # point2text / bbox2text 任务需要交换输入输出
        if task_type in [2, 3]:
            element_name, answer_xy = answer_xy, element_name

        # 构建 prompt
        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        messages = build_grounding_prompt(
            element_name, img_dict, task_type,
            self.shuffle_prompt, self.xy_int, self.uniform_prompt
        )

        # 处理输入
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(text=prompt, images=image_list, return_tensors="pt")

        # 拼接答案
        batch, _ = add_answer_to_batch(batch, answer_xy, self.processor)

        # 检查序列长度
        max_len = self.processor.tokenizer.model_max_length
        assert batch['input_ids'][0].shape[0] <= max_len, \
            f"序列长度超限: {batch['input_ids'][0].shape[0]} > {max_len}"

        return {
            'input_ids': batch['input_ids'][0],
            'labels': batch['labels'][0],
            'pixel_values': batch['pixel_values'],
            'image_sizes': batch['image_grid_thw'],
        }

    def _build_multi_turn(self, item, elements, image_list, task_type):
        """构建多轮训练数据"""
        element_names = [e['instruction'] for e in elements]

        # 获取所有答案
        if task_type in [0, 2]:  # point 任务
            if 'Qwen2.5-VL' in self.model_id:
                # Qwen2.5-VL 使用 resize 后的绝对坐标
                answers = [
                    self._convert_point_for_qwen(
                        e['point'], item['img_size'][1], item['img_size'][0]
                    )
                    for e in elements
                ]
            elif self.xy_int:
                answers = [[int(x * 1000) for x in e['point']] for e in elements]
            else:
                answers = [[round(x, 2) for x in e['point']] for e in elements]
        else:  # bbox 任务
            if self.xy_int:
                answers = [[int(x * 1000) for x in e['bbox']] for e in elements]
            else:
                answers = [[round(x, 2) for x in e['bbox']] for e in elements]

        # point2text / bbox2text 需要交换
        if task_type in [2, 3]:
            element_names, answers = answers, element_names

        # 构建第一轮 prompt
        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        messages = build_grounding_prompt(
            element_names[0], img_dict, task_type,
            self.shuffle_prompt, self.xy_int, self.uniform_prompt
        )

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(text=prompt, images=image_list, return_tensors="pt")

        # 拼接多轮答案
        batch, _ = add_multiturn_answer(
            batch, answers[0], self.processor,
            append_elements=element_names[1:],
            append_answers=answers[1:]
        )

        # 检查序列长度
        max_len = self.processor.tokenizer.model_max_length
        assert batch['input_ids'][0].shape[0] <= max_len, \
            f"序列长度超限: {batch['input_ids'][0].shape[0]} > {max_len}"

        return {
            'input_ids': batch['input_ids'][0],
            'labels': batch['labels'][0],
            'pixel_values': batch['pixel_values'],
            'image_sizes': batch['image_grid_thw'],
        }

    def _convert_point_for_qwen(self, point, orig_height, orig_width):
        """
        把归一化坐标转换成 Qwen2.5-VL 的绝对坐标格式（委托给共享实现，
        和 evaluator.py / inference.py 保持完全一致，见 utils/qwen_vl_common.py）
        """
        return point_to_qwen_coords(
            point, orig_height, orig_width,
            factor=28, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def _smart_resize(self, height, width, factor=28, min_pixels=256*28*28, max_pixels=1280*28*28):
        """智能 resize（委托给共享实现），保持长宽都是 factor 的倍数，同时控制总像素数"""
        return smart_resize(height, width, factor, min_pixels, max_pixels)

    def _random_crop(self, image, metadata, scale_range=(0.5, 1.0)):
        """
        随机裁剪图片（数据增强）

        裁剪后会更新元素的坐标，只保留完全在裁剪区域内的元素
        """
        orig_width, orig_height = metadata['img_size']
        img_copy = image.copy()

        scale_w = random.uniform(*scale_range)
        scale_h = random.uniform(*scale_range)

        crop_width = int(orig_width * scale_w)
        crop_height = int(orig_height * scale_h)

        # 如果裁剪尺寸大于原图，需要 padding
        pad_x = pad_y = 0
        if crop_width > orig_width or crop_height > orig_height:
            pad_x = max(0, (crop_width - orig_width) // 2)
            pad_y = max(0, (crop_height - orig_height) // 2)

            padded_img = Image.new('RGB', (crop_width, crop_height), (255, 255, 255))
            padded_img.paste(img_copy, (pad_x, pad_y))

            image = padded_img
            img_width, img_height = crop_width, crop_height
        else:
            img_width, img_height = orig_width, orig_height

        # 随机选择裁剪位置
        crop_x_min = random.randint(0, img_width - crop_width)
        crop_y_min = random.randint(0, img_height - crop_height)
        crop_x_max = crop_x_min + crop_width
        crop_y_max = crop_y_min + crop_height

        cropped_img = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

        # 更新元素坐标
        new_elements = []
        for element in metadata['element']:
            bbox = element['bbox']
            point = element['point']

            # 转成绝对坐标
            bbox_abs = [
                int(bbox[0] * orig_width) + pad_x,
                int(bbox[1] * orig_height) + pad_y,
                int(bbox[2] * orig_width) + pad_x,
                int(bbox[3] * orig_height) + pad_y
            ]
            point_abs = [
                int(point[0] * orig_width) + pad_x,
                int(point[1] * orig_height) + pad_y
            ]

            # 检查元素是否完全在裁剪区域内
            if (bbox_abs[0] >= crop_x_min and bbox_abs[2] <= crop_x_max and
                bbox_abs[1] >= crop_y_min and bbox_abs[3] <= crop_y_max):

                # 转回相对坐标
                new_bbox = [
                    (bbox_abs[0] - crop_x_min) / crop_width,
                    (bbox_abs[1] - crop_y_min) / crop_height,
                    (bbox_abs[2] - crop_x_min) / crop_width,
                    (bbox_abs[3] - crop_y_min) / crop_height
                ]
                new_point = [
                    (point_abs[0] - crop_x_min) / crop_width,
                    (point_abs[1] - crop_y_min) / crop_height
                ]

                new_element = element.copy()
                new_element['bbox'] = new_bbox
                new_element['point'] = new_point
                new_elements.append(new_element)

        # 如果没有元素在裁剪区域内，返回原图
        if len(new_elements) == 0:
            return img_copy, metadata

        metadata['element'] = new_elements
        metadata['element_size'] = len(new_elements)
        metadata['img_size'] = cropped_img.size

        return cropped_img, metadata
