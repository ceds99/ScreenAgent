"""
评估器

实现 ScreenSpot 等评估集上的评估逻辑

坐标转换统一走 utils/coordinate.py，解码方式要和训练时的编码方式对应上。
"""

import os
import json
import random
import logging

import torch
import wandb
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch.distributed as dist
from accelerate.utils import gather_object

from data import dict_to_cuda
from utils.coordinate import (
    smart_resize,
    decode_point,
    extract_coordinates,
    parse_predicted_point,
    resolve_coord_format,
)

logging.basicConfig(level=logging.INFO)

# 上面的 smart_resize 和 extract_coordinates 这个文件里没直接调，
# 但外面有 from trainer.evaluator import 的写法，留着别删


def convert_point_from_qwen_format(point, orig_height, orig_width,
                                   min_pixels=256*28*28, max_pixels=1280*28*28):
    """
    把 Qwen2.5-VL 输出的坐标转换回原图坐标

    只处理 qwen_abs 这一种格式，新代码直接用 utils.coordinate.decode_point，
    那边三种格式都支持。注意入参是 (height, width) 顺序。
    """
    return decode_point(
        point, (orig_width, orig_height), coord_format="qwen_abs",
        min_pixels=min_pixels, max_pixels=max_pixels
    )


def point_in_bbox(point, bbox):
    """
    判断点是否在边界框内

    Args:
        point: [x, y]
        bbox: [x1, y1, x2, y2]

    Returns:
        True/False
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def get_bbox_from_annotation(bbox_normalized, img_size):
    """
    把归一化的 [x1, y1, x2, y2] 标注转换为绝对像素坐标

    入参必须是归一化的。数据集加载时 prepare_annotations 已经统一过一遍，
    这里再挡一道：ScreenSpot 的原始标注是 [x,y,宽,高] 的绝对像素，直接传进来
    会算出几十万的框，point_in_bbox 恒为 false，而且一声不响，
    只有准确率悄悄归零。

    Args:
        bbox_normalized: 归一化的 [x1, y1, x2, y2]，范围 0-1
        img_size: (width, height)

    Returns:
        [x1, y1, x2, y2] 绝对像素坐标

    Raises:
        ValueError: 入参看起来不是归一化的
    """
    x1, y1, x2, y2 = bbox_normalized

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.05:
        raise ValueError(
            f"bbox 不像是归一化的: {list(bbox_normalized)}。"
            f"数据集加载时应该由 data.prepare_annotations 统一成归一化的 "
            f"[x1,y1,x2,y2]，先跑 python scripts/inspect_datasets.py 看实际格式"
        )

    width, height = img_size
    # 用 round 不用 int：int 是截断，78/540*540 会算成 77.9999 然后变 77，
    # 每条边都往内缩近 1 像素，小图标上是百分之几的面积损失，而且只往一个方向偏
    return [
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height)
    ]


def draw_result(image_path, pred_point=None, gt_bbox=None, radius=5, line_width=3):
    """
    在图片上画出预测点和真实框

    用于可视化评估结果
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    if pred_point is not None:
        x, y = pred_point
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill='blue', outline='blue'
        )

    if gt_bbox is not None:
        x1, y1, x2, y2 = gt_bbox
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line_width)

    return image


def broadcast_value(value, src=0, local_rank=0):
    """分布式训练时广播数值"""
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()


def calculate_metrics(results):
    """
    计算各类别的准确率

    Args:
        results: 按类别组织的结果字典

    Returns:
        metrics: 各类别的准确率
    """
    metrics = {}
    for category, samples in results.items():
        correct = sum(s["acc"] for s in samples)
        total = len(samples)
        metrics[f"{category} Accuracy"] = correct / total if total > 0 else 0

    for key, value in metrics.items():
        logging.info(f"[{key}]: {value:.4f}")

    return metrics


def _cast_pixel_values(input_dict, precision):
    """把 pixel_values 转成训练/推理用的精度"""
    if input_dict.get("pixel_values") is None:
        return input_dict

    if precision == "fp16":
        input_dict["pixel_values"] = input_dict["pixel_values"].half()
    elif precision == "bf16":
        input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
    else:
        input_dict["pixel_values"] = input_dict["pixel_values"].float()

    return input_dict


@torch.no_grad()
def evaluate_screenspot(val_loader, model, processor, epoch, global_step, writer, args, save_images=True):
    """
    在 ScreenSpot 数据集上评估模型

    Args:
        val_loader: 验证数据加载器（必须是 batch_size=1，见下面的说明）
        model: 模型
        processor: tokenizer/processor
        epoch: 当前 epoch
        global_step: 当前全局步数
        writer: TensorBoard writer
        args: 参数
        save_images: 是否保存可视化结果

    Returns:
        平均准确率
    """
    model.eval()

    # 收集结果，预测文本和还原后的坐标都塞在 all_outputs 的每条记录里
    all_answers = []
    all_outputs = []

    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # 坐标格式要和训练时一样，不然还原出来的坐标整体偏
    coord_format = resolve_coord_format(vars(args))
    min_pixels = getattr(args, 'min_visual_tokens', 256) * 28 * 28
    max_pixels = getattr(args, 'max_visual_tokens', 1280) * 28 * 28
    max_new_tokens = getattr(args, 'max_new_tokens', 64)

    logging.info(f"开始评估 (coord_format={coord_format}, max_new_tokens={max_new_tokens})")

    # 遍历验证集
    for input_dict in tqdm(val_loader, desc="评估中"):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')

        # meta 和几个默认值放在 try 外面，generate 挂了下面收集结果还要用
        meta = input_dict['meta_data'][0]
        raw_output = ""
        generated_text = "[]"
        pred_point = None

        # 处理图像数据类型
        input_dict = _cast_pixel_values(input_dict, args.precision)

        # 构建生成参数
        generate_dict = {
            "pixel_values": input_dict["pixel_values"],
            "input_ids": input_dict["input_ids"],
            "image_grid_thw": input_dict["image_sizes"],
        }
        # 评估固定 batch_size=1，mask 其实全是 1，传上去是为了和训练走同一条分支
        if input_dict.get("attention_mask") is not None:
            generate_dict["attention_mask"] = input_dict["attention_mask"]

        try:
            # 生成预测。答案是 [699, 113] 这种，算上 eos 十来个 token 就够，
            # 上限给松了的话模型没收敛时会每条都顶到上限，评估能拖到几个小时
            generate_ids = model.generate(
                **generate_dict,
                max_new_tokens=max_new_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            # 只取生成的部分
            generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
            raw_output = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            # 解析坐标，正则提取 + literal_eval，拿到 4 个值的框会自动取中心点
            pred_point = parse_predicted_point(raw_output)

            if pred_point is not None:
                img_size = meta.get('img_size')
                if not img_size:
                    raise ValueError("meta 里没有 img_size，没法还原坐标")

                # 还原回原图像素
                pred_point = decode_point(
                    pred_point, (img_size[0], img_size[1]),
                    coord_format=coord_format,
                    min_pixels=min_pixels, max_pixels=max_pixels
                )
                generated_text = str(pred_point)
            else:
                logging.warning(f"没能从模型输出里解析出坐标: {raw_output[:100]!r}")

        except Exception as e:
            logging.warning(f"解析预测结果失败: {e}")
            generated_text = "[]"
            pred_point = None

        # 收集结果
        output_info = {
            "split": meta.get('split', 'unknown'),
            "data_type": meta.get('data_type', 'unknown'),
            "anno_id": meta.get('id', 'unknown'),
            "img_path": meta.get('img_url_abs', ''),
            "instruction": meta.get('task', ''),
            "prediction": generated_text,
            "raw_output": raw_output,
            "pred_point": pred_point,
            "bbox": meta.get('bbox', []),
            "meta": meta,
        }

        all_answers.append(meta.get('bbox', []))
        all_outputs.append(output_info)

    # 分布式训练时收集所有进程的结果
    all_answers = gather_object(all_answers)
    all_outputs = gather_object(all_outputs)

    # 只在主进程计算指标
    if global_rank == 0:
        results = {}

        for ans, output in zip(all_answers, all_outputs):
            split = output['split']
            data_type = output['data_type']

            if split not in results:
                results[split] = {}
            if data_type not in results[split]:
                results[split][data_type] = []

            step_result = output.copy()

            # 计算准确率：预测点落在真实框内算对
            img_size = output['meta'].get('img_size') or [100, 100]
            pred_point = output.get('pred_point')

            if ans and len(ans) == 4:
                gt_bbox = get_bbox_from_annotation(ans, img_size)
            else:
                gt_bbox = None
            step_result['gt_bbox'] = gt_bbox

            if (pred_point is not None and len(pred_point) == 2
                    and gt_bbox is not None and point_in_bbox(pred_point, gt_bbox)):
                step_result["acc"] = 1
            else:
                step_result["acc"] = 0

            results[split][data_type].append(step_result)

        # 计算各类别指标
        eval_dict = {}
        for split in results.keys():
            logging.info("=" * 30)
            logging.info(f"Split: {split}")
            logging.info("=" * 30)
            eval_dict[split] = calculate_metrics(results[split])

        # 记录到 TensorBoard 和 wandb
        if not args.debug and writer is not None:
            for split, metrics in eval_dict.items():
                for key, value in metrics.items():
                    writer.add_scalar(f"eval/{split}/{key}", value, epoch)
                    if getattr(args, 'use_wandb', False):
                        wandb.log({f"eval/{split}/{key}": value}, step=global_step)

        # 计算总体平均准确率
        all_scores = [v for split_metrics in eval_dict.values() for v in split_metrics.values()]
        avg_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0

        eval_dict['avg_accuracy'] = avg_accuracy
        if not args.debug and writer is not None:
            writer.add_scalar("eval/avg_accuracy", avg_accuracy, epoch)
            if getattr(args, 'use_wandb', False):
                wandb.log({"eval/avg_accuracy": avg_accuracy}, step=global_step)

        logging.info(f"平均准确率: {avg_accuracy:.4f}")

        # 保存可视化结果
        if save_images and hasattr(args, 'tmp_dir'):
            save_dir = os.path.join(args.tmp_dir, "eval_images")
            os.makedirs(save_dir, exist_ok=True)

            rng = random.Random(42)
            for split in results:
                for data_type in results[split]:
                    # 随机抽取几个样本可视化
                    samples = rng.sample(
                        results[split][data_type],
                        min(5, len(results[split][data_type]))
                    )
                    for sample in samples:
                        try:
                            img = draw_result(
                                sample['img_path'],
                                sample.get('pred_point'),
                                sample.get('gt_bbox')
                            )
                            save_path = os.path.join(
                                save_dir,
                                f"epoch{epoch}_{split}_{data_type}_{sample['anno_id']}.png"
                            )
                            img.save(save_path)
                        except Exception as e:
                            logging.warning(f"保存可视化结果失败: {e}")

        # 保存详细结果
        if hasattr(args, 'tmp_dir'):
            os.makedirs(args.tmp_dir, exist_ok=True)

            result_path = os.path.join(args.tmp_dir, f"eval_results_epoch{epoch}.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(eval_dict, f, indent=2, ensure_ascii=False)

            # 逐条明细另存一份，出问题的时候能看到模型到底输出了什么、还原到哪去了
            detail_path = os.path.join(args.tmp_dir, f"eval_details_epoch{epoch}.json")
            details = []
            for split in results:
                for data_type in results[split]:
                    for sample in results[split][data_type]:
                        details.append({
                            "split": sample.get("split"),
                            "data_type": sample.get("data_type"),
                            "anno_id": sample.get("anno_id"),
                            "instruction": sample.get("instruction"),
                            "raw_output": sample.get("raw_output"),
                            "pred_point": sample.get("pred_point"),
                            "gt_bbox": sample.get("gt_bbox"),
                            "acc": sample.get("acc"),
                        })
            with open(detail_path, 'w', encoding='utf-8') as f:
                json.dump(details, f, indent=2, ensure_ascii=False)

        metric = avg_accuracy
    else:
        metric = 0

    # 广播结果到所有进程
    if dist.is_available() and dist.is_initialized():
        metric = broadcast_value(metric, src=0, local_rank=local_rank)

    return metric


@torch.no_grad()
def evaluate_training_data(val_loader, model, processor, epoch, global_step, writer, args):
    """
    在训练数据上评估（用于监控训练效果）

    这个评估比较简单，只计算 loss
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    for input_dict in tqdm(val_loader, desc="训练数据评估"):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')
        input_dict = _cast_pixel_values(input_dict, args.precision)

        forward_dict = {
            "pixel_values": input_dict["pixel_values"],
            "input_ids": input_dict["input_ids"],
            "labels": input_dict["labels"],
            "image_grid_thw": input_dict["image_sizes"],
        }
        # 跟训练的前向保持一致，attention_mask 别漏
        if input_dict.get("attention_mask") is not None:
            forward_dict["attention_mask"] = input_dict["attention_mask"]

        output = model(**forward_dict)

        total_loss += output['loss'].item()
        total_samples += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    if args.global_rank == 0:
        logging.info(f"训练数据评估 - Loss: {avg_loss:.4f}")
        if not args.debug and writer is not None:
            writer.add_scalar("eval_train/loss", avg_loss, epoch)
            if getattr(args, 'use_wandb', False):
                wandb.log({"eval_train/loss": avg_loss}, step=global_step)

    # 返回 1/loss 作为 "score"（loss 越小越好）
    return 1.0 / (avg_loss + 1e-6)
