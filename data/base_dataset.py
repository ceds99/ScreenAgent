"""
混合数据集

用于训练时同时加载多个数据集，支持：
- 按比例采样不同数据集
- 平衡采样（避免某个数据集过大导致其他数据集训练不足）
- 轮询采样（Round-Robin）
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .train_dataset import GroundingTrainDataset
from .eval_dataset import ScreenSpotDataset, TrainingEvalDataset
from .data_utils import IGNORE_INDEX


def collate_fn(batch, processor=None):
    """
    DataLoader 的 collate 函数

    把多个样本组合成一个 batch，主要处理：
    - padding：把不同长度的序列 pad 到相同长度
    - 拼接 pixel_values 和 image_sizes

    Args:
        batch: 样本列表，每个元素是 (data_dict, metadata)
        processor: tokenizer/processor

    Returns:
        batch 字典
    """
    # 分离数据和元数据
    data_list = [x[0] for x in batch]
    meta_list = [x[1] for x in batch]

    # 提取 input_ids 和 labels
    input_ids = [item['input_ids'] for item in data_list]
    labels = [item['labels'] for item in data_list]

    # Padding
    pad_token_id = processor.tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )

    # 截断到最大长度
    max_len = processor.tokenizer.model_max_length
    input_ids = input_ids[:, :max_len]
    labels = labels[:, :max_len]

    # 处理图像数据
    # Qwen2.5-VL 的 pixel_values 是 2D tensor [num_patches, hidden_dim]
    if data_list[0]['pixel_values'] is not None:
        pixel_values = [item['pixel_values'] for item in data_list]
        # 检查是不是 2D tensor，如果是就用 stack
        if len(pixel_values[0].shape) == 2:
            pixel_values = torch.stack(pixel_values, dim=0)
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = [item['image_sizes'] for item in data_list]
        image_sizes = torch.cat(image_sizes, dim=0)
    else:
        pixel_values = None
        image_sizes = None

    result = {
        'input_ids': input_ids,
        'labels': labels,
        'pixel_values': pixel_values,
        'image_sizes': image_sizes,
        'meta_data': meta_list,
    }

    return result


class HybridDataset(Dataset):
    """
    混合数据集

    支持同时加载多个训练数据集，并按照指定比例采样。

    采样策略：
    - random_sample=True, record_sample=False: 按概率随机采样
    - random_sample=True, record_sample=True: 轮询采样，保证每个数据集都被均匀采样
    - random_sample=False: 顺序遍历所有数据集
    """

    def __init__(self, processor, inference, args):
        """
        初始化混合数据集

        Args:
            processor: Qwen-VL processor
            inference: 是否是推理/评估模式
            args: 命令行参数
        """
        self.inference = inference
        self.processor = processor

        # 根据模式选择数据集配置
        # 先把路径规范化，去掉末尾的 /train 或 /eval
        base_dir = args.dataset_dir.rstrip('/')
        if base_dir.endswith('/train'):
            base_dir = base_dir[:-6]
        elif base_dir.endswith('/eval'):
            base_dir = base_dir[:-5]

        if inference:
            # 评估模式
            dataset_list = args.val_dataset
            json_list = args.val_json
            sample_rates = args.val_ratio
            dataset_dir = os.path.join(base_dir, 'eval')
        else:
            # 训练模式
            dataset_list = args.train_dataset
            json_list = args.train_json
            sample_rates = args.train_ratio
            dataset_dir = os.path.join(base_dir, 'train')

        # 解析数据集列表
        self.dataset_names = [x.strip() for x in dataset_list.split(',') if x.strip()]
        self.json_files = [x.strip() for x in json_list.split(',') if x.strip()]

        # 解析采样比例
        rates = np.array([float(x) for x in sample_rates.split(',')])
        self.sample_rates = rates / rates.sum()

        # 采样设置
        self.samples_per_epoch = getattr(args, 'samples_per_epoch', 1000)
        self.random_sample = getattr(args, 'random_sample', False)
        self.record_sample = getattr(args, 'record_sample', False)

        # 参数字典（传给子数据集）
        args_dict = vars(args)

        # 加载所有数据集
        self.datasets = []
        for ds_name, json_file in zip(self.dataset_names, self.json_files):
            if inference:
                # 评估数据集
                if ds_name in ['screenspot', 'screenspot2']:
                    dataset = ScreenSpotDataset(
                        dataset_dir, ds_name, json_file, processor, args_dict
                    )
                else:
                    dataset = TrainingEvalDataset(
                        dataset_dir, ds_name, json_file, processor, args_dict
                    )
            else:
                # 训练数据集
                dataset = GroundingTrainDataset(
                    dataset_dir, ds_name, json_file, processor, args_dict
                )
            self.datasets.append(dataset)

        # 验证配置
        if inference:
            assert len(self.datasets) == 1, "评估模式只支持单个数据集"
        else:
            assert len(self.datasets) == len(self.sample_rates), \
                f"数据集数量 ({len(self.datasets)}) 和采样比例数量 ({len(self.sample_rates)}) 不匹配"

        # 采样记录（用于轮询采样）
        self.sample_records = [set() for _ in self.datasets]
        self.current_dataset_idx = 0

        # 打印信息
        mode = "评估" if inference else "训练"
        print(f"[{mode}] 加载了 {len(self.datasets)} 个数据集")

    def __len__(self):
        """返回数据集长度"""
        total = sum(len(ds) for ds in self.datasets)

        if self.inference:
            return total

        if self.random_sample:
            return self.samples_per_epoch
        else:
            return total

    def __getitem__(self, idx):
        """获取一条样本"""
        if self.inference:
            # 评估模式：直接从第一个数据集取
            return self.datasets[0][idx]

        # 训练模式：根据采样策略选择数据集和样本
        if self.random_sample and not self.record_sample:
            # 按概率随机采样
            ds_idx = np.random.choice(len(self.datasets), p=self.sample_rates)
            sample_idx = np.random.randint(0, len(self.datasets[ds_idx]))
            return self.datasets[ds_idx][sample_idx]

        elif self.random_sample and self.record_sample:
            # 轮询采样：保证每个数据集都被均匀采样
            ds_idx = self.current_dataset_idx
            dataset = self.datasets[ds_idx]

            # 找一个还没采样过的样本
            unseen = set(range(len(dataset))) - self.sample_records[ds_idx]
            if not unseen:
                # 这个数据集的样本都采样过了，重置记录
                print(f"[采样] 数据集 {self.dataset_names[ds_idx]} 已完整遍历，重置")
                self.sample_records[ds_idx] = set()
                unseen = set(range(len(dataset)))

            sample_idx = random.choice(list(unseen))
            self.sample_records[ds_idx].add(sample_idx)

            # 轮询到下一个数据集
            self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.datasets)

            return dataset[sample_idx]

        else:
            # 顺序遍历
            for dataset in self.datasets:
                if idx < len(dataset):
                    return dataset[idx]
                idx -= len(dataset)

            raise IndexError("索引超出范围")
