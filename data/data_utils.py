"""
数据处理工具函数

包含训练过程中常用的工具类和函数，比如：
- 训练指标统计（AverageMeter）
- 进度显示（ProgressMeter）
- 数据转换等
"""

import numpy as np
import torch
import torch.distributed as dist
from enum import Enum


# 训练时忽略的标签值，用于计算 loss 时跳过 padding 部分
IGNORE_INDEX = -100


class Summary(Enum):
    """指标汇总方式"""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """
    用于统计训练过程中的指标，比如 loss、accuracy 等

    用法：
        losses = AverageMeter('Loss', ':.4f')
        losses.update(loss_value, batch_size)
        print(losses)  # 显示当前值和平均值
    """

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """重置所有统计值"""
        self.val = 0      # 当前值
        self.avg = 0      # 平均值
        self.sum = 0      # 累计和
        self.count = 0    # 累计次数

    def update(self, val, n=1):
        """
        更新统计值

        Args:
            val: 当前值
            n: 样本数量（默认为1）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """
        分布式训练时，跨 GPU 同步统计值
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist() + [self.count],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count],
                dtype=torch.float32,
                device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)

        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()

        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """返回汇总字符串"""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError(f"不支持的汇总类型: {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    训练进度显示器

    用法：
        progress = ProgressMeter(num_batches, [losses, acc], prefix="Epoch [1]")
        progress.display(batch_idx)
    """

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """显示当前 batch 的进度"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        """显示汇总信息"""
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict, device="cuda"):
    """
    把字典中的 tensor 都转移到 GPU 上

    Args:
        input_dict: 包含 tensor 的字典
        device: 目标设备

    Returns:
        转移后的字典
    """
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(device, non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(device, non_blocking=True) for ele in v]

    return input_dict
