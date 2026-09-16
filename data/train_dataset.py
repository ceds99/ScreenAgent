"""
训练数据集

支持的数据集：
- ShowUI-desktop: 桌面界面截图
- ShowUI-web-8k: Web 界面截图
- AMEX-8k: 移动应用截图
- UGround-V1-8k: 多分辨率 Web 截图

训练答案用什么坐标空间由 --coord_format 决定，转换逻辑在 utils/coordinate.py。
单轮和多轮共用 _format_answer，保证两条路径出来的格式一致。
"""

import os
import copy
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from .template import build_grounding_prompt, add_answer_to_batch, add_multiturn_answer
from .data_utils import load_metadata, prepare_annotations
from utils.coordinate import (
    DEFAULT_COORD_FORMAT,
    resolve_coord_format,
    encode_point,
    encode_bbox,
)


# 多轮序列超长时最多裁几次。每次按超出比例估算能装下几轮，
# 正常一两次就收敛了，给 4 次是留余量
MAX_TRIM_ATTEMPTS = 4


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

        # 加载元数据（文件名对不上时会自动兜底并给出可用文件列表）
        self.data, self.meta_path = load_metadata(meta_dir, json_file, dataset_name)

        # 各数据集的标注格式不一样（有的绝对像素有的归一化，bbox 的四个数含义也不同），
        # 这里判一次并统一转成归一化的 [x1,y1,x2,y2]，下游就不用再管格式
        self.ann_format = prepare_annotations(
            self.data, dataset_name,
            scale=(args_dict or {}).get('ann_scale'),
            bbox_layout=(args_dict or {}).get('ann_bbox_layout'),
        )

        # 解析参数
        args_dict = args_dict or {}
        self.num_turn = args_dict.get('num_turn', 1)           # 多轮对话轮数
        self.shuffle_prompt = args_dict.get('shuffle_image_token', False)
        self.uniform_prompt = args_dict.get('uniform_prompt', False)
        self.crop_min = args_dict.get('crop_min', 1.0)         # 随机裁剪最小比例
        self.crop_max = args_dict.get('crop_max', 1.0)         # 随机裁剪最大比例

        # 坐标格式，评估和推理要用同一个值
        self.coord_format = resolve_coord_format(args_dict)

        # 任务类型采样概率（默认只做 text2point）
        self.sample_prob = np.array([
            args_dict.get('text2point', 1),
            args_dict.get('text2bbox', 0),
            args_dict.get('point2text', 0),
            args_dict.get('bbox2text', 0)
        ])
        self.sample_prob = self.sample_prob / self.sample_prob.sum()

        self.dataset_name = dataset_name
        self.trim_count = 0        # 因为序列超长而减少轮数的次数

        print(f"[训练数据集] {dataset_name}: {len(self.data)} 条样本 "
              f"(标注={self.ann_format['scale']}/{self.ann_format['bbox_layout']}, "
              f"归一化了 {self.ann_format['changed']} 个坐标, "
              f"coord_format={self.coord_format}, num_turn={self.num_turn})")

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
            # _random_crop 会改写 element 里的坐标，先深拷贝一份，别动到 self.data
            if self.crop_min != 1.0 or self.crop_max != 1.0:
                item = copy.deepcopy(item)
                image, item = self._random_crop(image, item, (self.crop_min, self.crop_max))
                image_list = [image]
        else:
            image_list = None

        # 原图尺寸 (宽, 高)，坐标转换要用
        img_size = self._resolve_img_size(item, image_list)

        # 随机选择任务类型（text2point, text2bbox, point2text, bbox2text）
        task_type = np.random.choice(len(self.sample_prob), p=self.sample_prob)

        # 选择要定位的元素（支持多轮）
        elements = item['element']
        num_elements = min(self.num_turn, len(elements))
        assert num_elements > 0, "样本中没有元素"

        if self.num_turn == 1:
            # 单轮：用独立的随机数发生器，同一个 idx 选到同一个元素，又不动全局随机状态
            rng = random.Random(idx)
            element_idx = rng.randint(0, len(elements) - 1)
            selected_elements = [elements[element_idx]]
        else:
            # 多轮：随机选择多个元素
            selected_elements = random.choices(elements, k=num_elements)

        # 构建训练数据
        if len(selected_elements) == 1:
            data_dict = self._build_single_turn(
                selected_elements[0], image_list, img_size, task_type
            )
        else:
            data_dict = self._build_multi_turn(
                selected_elements, image_list, img_size, task_type
            )

        return data_dict, item

    def _resolve_img_size(self, item, image_list):
        """
        拿到坐标转换要用的原图尺寸 (宽, 高)

        优先用实际送进 processor 的那张图，裁剪之后也是准的；没有图片才退回
        metadata 里的 img_size。
        """
        if image_list:
            return image_list[0].size  # PIL 的 size 就是 (宽, 高)
        if 'img_size' in item:
            return (item['img_size'][0], item['img_size'][1])
        raise ValueError("样本既没有图片也没有 img_size，无法做坐标转换")

    def _format_answer(self, element, img_size, task_type):
        """
        把元素的归一化标注转成训练答案

        单轮和多轮都走这里，格式只在这一个地方定，免得两边写岔了。
        入参的 point / bbox 已经由 prepare_annotations 统一成归一化的了。

        Args:
            element: 元素标注，含 instruction / point / bbox
            img_size: 原图 (宽, 高)
            task_type: 0 text2point / 1 text2bbox / 2 point2text / 3 bbox2text

        Returns:
            格式化后的坐标列表
        """
        if task_type in [0, 2]:
            point = element.get('point')
            if point is None:
                raise ValueError(f"[{self.dataset_name}] 这条标注没有 point 字段")
            return encode_point(
                point, img_size, self.coord_format,
                min_pixels=self.min_pixels, max_pixels=self.max_pixels
            )

        bbox = element.get('bbox')
        if bbox is None:
            # UGround 就是这样，只有 point 没有 bbox
            raise ValueError(
                f"[{self.dataset_name}] 这个数据集的标注里没有 bbox，"
                f"跑不了 text2bbox / bbox2text。把 --text2bbox 设回 0，或者换数据集"
            )
        return encode_bbox(
            bbox, img_size, self.coord_format,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def _build_single_turn(self, element, image_list, img_size, task_type):
        """构建单轮训练数据"""
        element_name = element['instruction']
        answer_xy = self._format_answer(element, img_size, task_type)

        # point2text / bbox2text 任务需要交换输入输出
        if task_type in [2, 3]:
            element_name, answer_xy = answer_xy, element_name

        # 构建 prompt
        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        messages = build_grounding_prompt(
            element_name, img_dict, task_type,
            self.shuffle_prompt, self.coord_format, self.uniform_prompt
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

    def _build_multi_turn(self, elements, image_list, img_size, task_type):
        """
        构建多轮训练数据

        元素多、描述又长的图拼出来可能超过 model_max_length。这种情况下砍掉
        尾部几轮重拼，而不是把整张图丢掉换一张 —— 一张图能出多少轮监督信号，
        正是这套训练方式的核心，扔掉元素最丰富的那些图等于把最有价值的数据扔了。

        顺带也是显存的闸：序列越长激活值越大，24GB 的卡上长尾序列是会 OOM 的。
        """
        element_names = [e['instruction'] for e in elements]
        answers = [self._format_answer(e, img_size, task_type) for e in elements]

        # point2text / bbox2text 需要交换
        if task_type in [2, 3]:
            element_names, answers = answers, element_names

        img_dict = {'type': 'image', 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}
        max_len = self.processor.tokenizer.model_max_length
        wanted_turns = len(element_names)

        for _ in range(MAX_TRIM_ATTEMPTS):
            messages = build_grounding_prompt(
                element_names[0], img_dict, task_type,
                self.shuffle_prompt, self.coord_format, self.uniform_prompt
            )
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch = self.processor(text=prompt, images=image_list, return_tensors="pt")

            batch, _ = add_multiturn_answer(
                batch, answers[0], self.processor,
                append_elements=element_names[1:],
                append_answers=answers[1:]
            )

            seq_len = batch['input_ids'][0].shape[0]
            if seq_len <= max_len:
                break

            if len(element_names) <= 1:
                raise ValueError(
                    f"只剩一轮还是超长（{seq_len} > {max_len}）。"
                    f"图片的视觉 token 或者元素描述太长了，"
                    f"调小 --max_visual_tokens 或者调大 --model_max_length"
                )

            # 按超出比例估算能装下几轮，乘 0.9 留点余量少试几次；
            # 至少砍掉一轮，保证循环一定收敛
            keep = int(len(element_names) * max_len / seq_len * 0.9)
            keep = max(1, min(keep, len(element_names) - 1))
            element_names = element_names[:keep]
            answers = answers[:keep]
        else:
            raise ValueError(
                f"连续 {MAX_TRIM_ATTEMPTS} 次裁剪都没能压到 {max_len} 以内"
            )

        # 真的裁过就记一笔，但别每条都打，刷屏
        if len(element_names) < wanted_turns:
            self.trim_count += 1
            if self.trim_count <= 3 or self.trim_count % 500 == 0:
                print(f"[{self.dataset_name}] 第 {self.trim_count} 次因序列超长减少轮数: "
                      f"{wanted_turns} -> {len(element_names)}")

        return {
            'input_ids': batch['input_ids'][0],
            'labels': batch['labels'][0],
            'pixel_values': batch['pixel_values'],
            'image_sizes': batch['image_grid_thw'],
        }

    def _random_crop(self, image, metadata, scale_range=(0.5, 1.0)):
        """
        随机裁剪图片（数据增强）

        裁剪后会更新元素的坐标，只保留完全在裁剪区域内的元素。
        这里会直接改写传进来的 metadata，调用方记得先深拷贝。
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
