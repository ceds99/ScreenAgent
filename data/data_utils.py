"""
数据处理工具函数

包含训练过程中常用的工具类和函数，比如：
- 训练指标统计（AverageMeter）
- 进度显示（ProgressMeter）
- 元数据加载（load_metadata）
- 数据转换等
"""

import os
import json
import glob

import numpy as np
import torch
import torch.distributed as dist
from enum import Enum

from utils.coordinate import normalize_bbox, normalize_point


# 训练时忽略的标签值，用于计算 loss 时跳过 padding 部分
IGNORE_INDEX = -100


# 判定坐标格式用的几个阈值
#
# NORM_TOLERANCE 取 1.1 而不是 1.0，是给标注噪声留余量 —— 有些数据集的框会
# 略微超出图像边界，归一化之后变成 1.02 这种。之所以放宽到 1.1 也不会误判，
# 是因为「归一化带噪声」和「绝对像素」这两种情况差了好几个数量级：
# 真的是绝对像素的话，一个数据集扫下来最大值必然在几百到几千。
NORM_TOLERANCE = 1.1
BOUNDS_TOLERANCE = 1.02   # 判 xywh 时允许框略微超出图像边界


# 常见的元数据文件名，找不到指定文件时按这个顺序兜底
COMMON_META_NAMES = [
    "hf_train",
    "hf_train_ori_coord",
    "hf_test_full",
    "hf_test",
    "train",
    "test",
]


def load_metadata(meta_dir, json_file, dataset_name=""):
    """
    加载数据集的元数据 JSON

    不同数据集仓库里元数据的文件名不统一，比如 ShowUI-desktop 有的版本叫
    hf_train_ori_coord、有的叫 hf_train，写死一个名字很容易踩空，所以这里加了兜底：

    1. 指定的文件在就直接用
    2. 不在就按 COMMON_META_NAMES 的顺序找同目录下其它的 json
    3. 还找不到，目录里只有一个 json 就用它
    4. 都不行就报错，顺便把目录里实际有哪些文件打出来

    Args:
        meta_dir: metadata 目录路径
        json_file: 期望的文件名（不含 .json 后缀）
        dataset_name: 数据集名，打日志用

    Returns:
        (data, 实际用到的文件路径)
    """
    wanted = os.path.join(meta_dir, f"{json_file}.json")

    if os.path.isfile(wanted):
        with open(wanted, 'r', encoding='utf-8') as f:
            return json.load(f), wanted

    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(
            f"[{dataset_name}] 找不到 metadata 目录: {meta_dir}\n"
            f"  请确认数据集已下载完整（每个数据集下应有 images/ 和 metadata/ 两个子目录）"
        )

    available = sorted(glob.glob(os.path.join(meta_dir, "*.json")))
    available_names = [os.path.splitext(os.path.basename(p))[0] for p in available]

    # 按常见命名找一个
    fallback = None
    for name in COMMON_META_NAMES:
        if name in available_names:
            fallback = os.path.join(meta_dir, f"{name}.json")
            break

    # 只有一个 json 就用它
    if fallback is None and len(available) == 1:
        fallback = available[0]

    if fallback is None:
        raise FileNotFoundError(
            f"[{dataset_name}] 找不到元数据文件: {wanted}\n"
            f"  metadata 目录: {meta_dir}\n"
            f"  目录下实际可用的文件: {available_names if available_names else '(空)'}\n"
            f"  请用 --train_json / --val_json 指定正确的文件名（不含 .json 后缀）"
        )

    print(f"[警告] {dataset_name}: 找不到 {json_file}.json，"
          f"自动改用 {os.path.basename(fallback)}")

    with open(fallback, 'r', encoding='utf-8') as f:
        return json.load(f), fallback


# ============================================================
# 标注坐标格式的判别与归一化
# ============================================================

def _iter_annotations(data):
    """
    遍历所有标注，统一产出 (img_size, bbox, point)

    训练集是 element 列表，评估集是顶层一个 bbox，这里抹平这个差异。
    """
    for item in data:
        img_size = item.get('img_size')
        if not img_size or len(img_size) != 2:
            continue
        if item.get('element'):
            for element in item['element']:
                yield img_size, element.get('bbox'), element.get('point')
        elif 'bbox' in item:
            yield img_size, item.get('bbox'), item.get('point')


def _valid_coords(values, count):
    """坐标是不是一个长度对、且没有 None 的数值列表"""
    if not values or len(values) != count:
        return None
    if any(v is None for v in values):
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def detect_annotation_format(data, dataset_name="", max_records=2000,
                             scale=None, bbox_layout=None):
    """
    扫标注，判断坐标是什么格式

    要判两件事：
      1. 数值是绝对像素还是归一化的 0-1
      2. bbox 的四个数是 [x1,y1,x2,y2] 还是 [x,y,宽,高]

    各个数据集的格式不一样（实测：ShowUI-desktop 是归一化的，ShowUI-web 和
    AMEX 是绝对像素，UGround 干脆没有 bbox，ScreenSpot 是 [x,y,宽,高]），
    所以不能写死，每次加载都判一遍。

    判不出来就抛异常，不猜。这类格式错误最麻烦的地方是不报错 ——
    训练 loss 照降（模型会去学一个被夹到边界的常数）、评测分数悄悄归零，
    等发现的时候几十个小时已经烧掉了。

    Args:
        data: metadata 列表
        dataset_name: 打日志和报错用
        max_records: 最多扫多少条，None 表示全扫
        scale: 显式指定 'norm' / 'absolute'，跳过判别
        bbox_layout: 显式指定 'xyxy' / 'xywh'，跳过判别

    Returns:
        {"scale", "bbox_layout", "has_bbox", "has_point", "checked"}
        没有 bbox 的数据集（比如 UGround）bbox_layout 是 None

    Raises:
        ValueError: 扫不到有效标注，或者判不出 bbox 的格式
    """
    records = data if max_records is None else data[:max_records]

    max_value = 0.0
    n_bbox = n_point = checked = 0
    xyxy_bad = xywh_bad = 0                  # 按该布局解释时的违例数
    center_xyxy_win = center_xywh_win = 0    # 有 point 时，哪种布局的中心更接近

    for img_size, bbox, point in _iter_annotations(records):
        checked += 1
        width, height = float(img_size[0]), float(img_size[1])

        bbox = _valid_coords(bbox, 4)
        point = _valid_coords(point, 2)

        values = (bbox or []) + (point or [])
        if values:
            max_value = max(max_value, max(abs(v) for v in values))
        if point:
            n_point += 1
        if not bbox:
            continue
        n_bbox += 1

        b0, b1, b2, b3 = bbox

        # 当成 [x1,y1,x2,y2]：必须 x1 < x2、y1 < y2，否则是个反的框
        if not (b0 < b2 and b1 < b3):
            xyxy_bad += 1

        # 当成 [x,y,宽,高]：宽高得是正的，而且不能整体超出图像
        if not (b2 > 0 and b3 > 0
                and b0 + b2 <= width * BOUNDS_TOLERANCE
                and b1 + b3 <= height * BOUNDS_TOLERANCE):
            xywh_bad += 1

        # 有 point 的话拿它和两种布局的中心比，谁近算谁的。
        # 这个判据最硬，不需要设容差，几千条样本投票下来不会错
        if point:
            px, py = point
            err_xyxy = max(abs((b0 + b2) / 2 - px), abs((b1 + b3) / 2 - py))
            err_xywh = max(abs(b0 + b2 / 2 - px), abs(b1 + b3 / 2 - py))
            if err_xyxy < err_xywh:
                center_xyxy_win += 1
            elif err_xywh < err_xyxy:
                center_xywh_win += 1

    if checked == 0:
        raise ValueError(
            f"[{dataset_name}] 扫不到有效标注。"
            f"每条记录需要有 img_size，以及 element 列表或顶层 bbox"
        )

    detected_scale = scale or ("norm" if max_value <= NORM_TOLERANCE else "absolute")

    if bbox_layout:
        detected_layout = bbox_layout
    elif n_bbox == 0:
        detected_layout = None
    elif center_xyxy_win or center_xywh_win:
        detected_layout = "xyxy" if center_xyxy_win >= center_xywh_win else "xywh"
    elif xyxy_bad > 0 and xywh_bad == 0:
        detected_layout = "xywh"
    elif xywh_bad > 0 and xyxy_bad == 0:
        detected_layout = "xyxy"
    else:
        raise ValueError(
            f"[{dataset_name}] 判不出 bbox 的格式。扫了 {checked} 条标注："
            f"当成 xyxy 有 {xyxy_bad} 条违例，当成 xywh 有 {xywh_bad} 条违例，"
            f"又没有 point 可以对照。人工确认后调用时传 bbox_layout 指定"
        )

    return {
        "scale": detected_scale,
        "bbox_layout": detected_layout,
        "has_bbox": n_bbox > 0,
        "has_point": n_point > 0,
        "checked": checked,
    }


def normalize_annotations(data, ann_format):
    """
    把标注原地统一成「归一化的 [x1,y1,x2,y2] + 归一化的 point」

    在加载时转一次，下游就都可以假定坐标是归一化的 —— 答案格式化
    （encode_point / encode_bbox）、随机裁剪、评测判分都是按这个前提写的，
    不用各自再判一遍格式。

    Args:
        data: metadata 列表，会被原地修改
        ann_format: detect_annotation_format 的返回值

    Returns:
        改写了多少个坐标字段
    """
    scale = ann_format["scale"]
    layout = ann_format["bbox_layout"]

    # 本来就是归一化的 xyxy，什么都不用做
    if scale == "norm" and layout in (None, "xyxy"):
        return 0

    changed = 0
    for item in data:
        img_size = item.get('img_size')
        if not img_size or len(img_size) != 2:
            continue
        size = (float(img_size[0]), float(img_size[1]))

        # 训练集改 element 里的，评估集改顶层的
        targets = item['element'] if item.get('element') else [item]

        for target in targets:
            bbox = _valid_coords(target.get('bbox'), 4)
            if bbox:
                target['bbox'] = normalize_bbox(bbox, size, scale, layout)
                changed += 1
            point = _valid_coords(target.get('point'), 2)
            if point:
                target['point'] = normalize_point(point, size, scale)
                changed += 1

    return changed


def validate_normalized_annotations(data, dataset_name=""):
    """
    归一化之后检查一遍，坐标应该都落在 0-1 附近

    normalize_annotations 转完就在这里拦一道。格式判错的话数值会明显越界，
    与其让它一路跑到评测再表现为「分数恒定为 0」，不如现在就报错。

    Args:
        data: 已经归一化过的 metadata 列表
        dataset_name: 报错用

    Returns:
        {"max_value": 最大的坐标值, "slightly_out": 轻微越界的坐标个数}

    Raises:
        ValueError: 有坐标明显超出 0-1
    """
    worst = 0.0
    worst_sample = None
    slightly_out = 0

    for img_size, bbox, point in _iter_annotations(data):
        for values in (_valid_coords(bbox, 4), _valid_coords(point, 2)):
            if not values:
                continue
            for value in values:
                value = abs(value)
                if value > 1.0:
                    slightly_out += 1
                if value > worst:
                    worst = value
                    worst_sample = {"img_size": list(img_size),
                                    "bbox": bbox, "point": point}

    if worst > NORM_TOLERANCE:
        raise ValueError(
            f"[{dataset_name}] 归一化之后还有坐标明显越界，最大 {worst:.3f}，"
            f"出现在 {worst_sample}。说明坐标格式判定错了，"
            f"先用 python scripts/inspect_datasets.py 看一眼实际格式"
        )

    return {"max_value": worst, "slightly_out": slightly_out}


def prepare_annotations(data, dataset_name="", scale=None, bbox_layout=None):
    """
    判别格式 + 归一化 + 校验，数据集加载时调一次就够

    Args:
        data: metadata 列表，会被原地修改
        dataset_name: 打日志用
        scale / bbox_layout: 显式指定，跳过自动判别

    Returns:
        detect_annotation_format 的结果，额外带上 changed 和 max_value
    """
    ann_format = detect_annotation_format(
        data, dataset_name, scale=scale, bbox_layout=bbox_layout
    )
    changed = normalize_annotations(data, ann_format)
    checked = validate_normalized_annotations(data, dataset_name)

    ann_format["changed"] = changed
    ann_format["max_value"] = checked["max_value"]
    return ann_format


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
