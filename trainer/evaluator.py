"""
评估器

实现 ScreenSpot 等评估集上的评估逻辑
"""

import os
import re
import ast
import math
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

logging.basicConfig(level=logging.INFO)


def smart_resize(height, width, factor=28, min_pixels=256*28*28, max_pixels=1280*28*28):
    """
    Qwen2.5-VL 的图像 resize 逻辑

    保持长宽都是 factor 的倍数，同时控制总像素数
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


def convert_point_from_qwen_format(point, orig_height, orig_width,
                                   min_pixels=256*28*28, max_pixels=1280*28*28):
    """
    把 Qwen2.5-VL 输出的坐标转换回原图坐标

    模型输出的是 resize 后的绝对坐标，需要转回原图的绝对坐标
    """
    new_height, new_width = smart_resize(orig_height, orig_width, 28, min_pixels, max_pixels)

    scale_w = orig_width / new_width
    scale_h = orig_height / new_height

    x_resized, y_resized = point
    x_orig = round(x_resized * scale_w)
    y_orig = round(y_resized * scale_h)

    # 确保在范围内
    x_orig = max(0, min(x_orig, orig_width - 1))
    y_orig = max(0, min(y_orig, orig_height - 1))

    return [x_orig, y_orig]


def extract_coordinates(text):
    """
    从模型输出中提取坐标

    支持多种格式：[x, y]、[x,y]、(x, y) 等
    """
    # 尝试匹配 [x, y] 格式
    match = re.search(r'\[\s*[\d\.]+\s*,\s*[\d\.]+\s*\]', text)
    if match:
        return match.group(0)
    return None


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
    把归一化的 [x1, y1, x2, y2] 格式的标注转换为绝对像素坐标

    Args:
        bbox_normalized: 归一化的 [x1, y1, x2, y2]，范围 0-1
        img_size: (width, height)

    Returns:
        [x1, y1, x2, y2] 绝对像素坐标
    """
    x1, y1, x2, y2 = bbox_normalized
    width, height = img_size
    return [
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height)
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


@torch.no_grad()
def evaluate_screenspot(val_loader, model, processor, epoch, global_step, writer, args, save_images=True):
    """
    在 ScreenSpot 数据集上评估模型

    Args:
        val_loader: 验证数据加载器
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

    # 收集结果
    all_predictions = []
    all_answers = []
    all_outputs = []

    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 遍历验证集
    for input_dict in tqdm(val_loader, desc="评估中"):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')

        # 处理图像数据类型
        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        # 构建生成参数
        generate_dict = {
            "pixel_values": input_dict["pixel_values"],
            "input_ids": input_dict["input_ids"],
            "image_grid_thw": input_dict["image_sizes"],
        }

        try:
            # 生成预测
            generate_ids = model.generate(
                **generate_dict,
                max_new_tokens=2048,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

            # 只取生成的部分
            generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
            generated_text = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            meta = input_dict['meta_data'][0]

            # 解析坐标（先尝试提取坐标格式，再解析）
            coord_str = extract_coordinates(generated_text)
            if coord_str:
                pred_point = ast.literal_eval(coord_str)
            else:
                pred_point = ast.literal_eval(generated_text)

            # 如果是边界框格式，取中心点
            if len(pred_point) == 4:
                pred_point = [
                    (pred_point[0] + pred_point[2]) / 2,
                    (pred_point[1] + pred_point[3]) / 2
                ]

            # 转换坐标（Qwen2.5-VL 输出的是 resize 后的坐标）
            if 'Qwen2.5-VL' in args.model_id:
                pred_point = convert_point_from_qwen_format(
                    pred_point,
                    meta['img_size'][1],  # height
                    meta['img_size'][0],  # width
                    min_pixels=args.min_visual_tokens * 28 * 28,
                    max_pixels=args.max_visual_tokens * 28 * 28
                )
                generated_text = str(pred_point)

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
            "bbox": meta.get('bbox', []),
            "meta": meta,
        }

        all_predictions.append(generated_text)
        all_answers.append(meta.get('bbox', []))
        all_outputs.append(output_info)

    # 分布式训练时收集所有进程的结果
    all_predictions = gather_object(all_predictions)
    all_answers = gather_object(all_answers)
    all_outputs = gather_object(all_outputs)

    # 只在主进程计算指标
    if global_rank == 0:
        results = {}

        for pred, ans, output in zip(all_predictions, all_answers, all_outputs):
            split = output['split']
            data_type = output['data_type']

            if split not in results:
                results[split] = {}
            if data_type not in results[split]:
                results[split][data_type] = []

            step_result = output.copy()

            # 计算准确率
            img_size = output['meta'].get('img_size', [100, 100])
            gt_bbox = get_bbox_from_annotation(ans, img_size)
            step_result['gt_bbox'] = gt_bbox

            try:
                pred_point = ast.literal_eval(pred)
                step_result['pred_point'] = pred_point

                if point_in_bbox(pred_point, gt_bbox):
                    step_result["acc"] = 1
                else:
                    step_result["acc"] = 0
            except:
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
        if not args.debug:
            for split, metrics in eval_dict.items():
                for key, value in metrics.items():
                    writer.add_scalar(f"eval/{split}/{key}", value, epoch)
                    if getattr(args, 'use_wandb', False):
                        wandb.log({f"eval/{split}/{key}": value}, step=global_step)

        # 计算总体平均准确率
        all_scores = [v for split_metrics in eval_dict.values() for v in split_metrics.values()]
        avg_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0

        eval_dict['avg_accuracy'] = avg_accuracy
        if not args.debug:
            writer.add_scalar("eval/avg_accuracy", avg_accuracy, epoch)
            if getattr(args, 'use_wandb', False):
                wandb.log({"eval/avg_accuracy": avg_accuracy}, step=global_step)

        logging.info(f"平均准确率: {avg_accuracy:.4f}")

        # 保存可视化结果
        if save_images and hasattr(args, 'tmp_dir'):
            save_dir = os.path.join(args.tmp_dir, "eval_images")
            os.makedirs(save_dir, exist_ok=True)

            random.seed(42)
            for split in results:
                for data_type in results[split]:
                    # 随机抽取几个样本可视化
                    samples = random.sample(
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
            result_path = os.path.join(args.tmp_dir, f"eval_results_epoch{epoch}.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(eval_dict, f, indent=2, ensure_ascii=False)

        metric = avg_accuracy
    else:
        metric = 0

    # 广播结果到所有进程
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

        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()

        output = model(
            pixel_values=input_dict["pixel_values"],
            input_ids=input_dict["input_ids"],
            labels=input_dict["labels"],
            image_grid_thw=input_dict["image_sizes"],
        )

        total_loss += output['loss'].item()
        total_samples += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    if args.global_rank == 0:
        logging.info(f"训练数据评估 - Loss: {avg_loss:.4f}")
        if not args.debug:
            writer.add_scalar("eval_train/loss", avg_loss, epoch)
            if getattr(args, 'use_wandb', False):
                wandb.log({"eval_train/loss": avg_loss}, step=global_step)

    # 返回 1/loss 作为 "score"（loss 越小越好）
    return 1.0 / (avg_loss + 1e-6)
