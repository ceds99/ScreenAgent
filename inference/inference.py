"""
ScreenAgent 推理模块

提供简单的 API 来使用训练好的模型进行 GUI 元素定位。

使用方法:
    from inference import ScreenAgentInference

    model = ScreenAgentInference("path/to/model")
    x, y = model.predict("path/to/screenshot.png", "点击登录按钮")

coord_format / min_visual_tokens / max_visual_tokens 要和训练时一致，不然坐标还原会整体偏。
默认值跟训练脚本里的一样，没改过训练参数就不用管。
"""

import os
import sys

from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# 添加项目根目录到 path，直接 python inference/inference.py 跑的时候也能 import 到 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.coordinate import (
    DEFAULT_COORD_FORMAT,
    smart_resize,
    decode_point,
    parse_predicted_point,
)


class ScreenAgentInference:
    """
    ScreenAgent 推理类

    加载训练好的模型，输入截图和指令，输出点击坐标。
    """

    def __init__(self, model_path, device="cuda:0", precision="bf16",
                 coord_format=DEFAULT_COORD_FORMAT,
                 min_visual_tokens=256, max_visual_tokens=1280):
        """
        初始化推理模型

        Args:
            model_path: 模型路径（合并后的完整模型）
            device: 运行设备
            precision: 精度 (bf16/fp16/fp32)
            coord_format: 模型输出的坐标格式，必须和训练时的 --coord_format 一致
            min_visual_tokens: 最小视觉 token 数，必须和训练时一致
            max_visual_tokens: 最大视觉 token 数，必须和训练时一致
        """
        self.device = device
        self.coord_format = coord_format

        # 确定数据类型
        if precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif precision == "fp16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # 视觉 token 范围（和训练时保持一致）
        self.min_pixels = min_visual_tokens * 28 * 28
        self.max_pixels = max_visual_tokens * 28 * 28

        print(f"加载模型: {model_path}")
        print(f"  坐标格式: {self.coord_format}")

        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # 设置 chat template
        CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        self.processor.chat_template = CHAT_TEMPLATE
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.chat_template = CHAT_TEMPLATE

        # 加载模型
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self.model.eval()

        print("模型加载完成")

    def predict(self, image_path, instruction, return_normalized=False,
                max_new_tokens=64):
        """
        预测点击坐标（从图片路径读图）

        Args:
            image_path: 截图路径
            instruction: 自然语言指令，比如 "click the login button"
            return_normalized: 是否返回归一化坐标 (0-1)，默认返回像素坐标
            max_new_tokens: 生成长度上限，坐标答案约 12 个 token，64 已经很宽裕

        Returns:
            (x, y): 点击坐标，解析失败返回 (None, None)
        """
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image, instruction, return_normalized, max_new_tokens)

    def predict_image(self, image, instruction, return_normalized=False,
                      max_new_tokens=64):
        """
        预测点击坐标（直接收 PIL.Image）

        agent 循环里拿到的是内存里的截图，没必要先落盘再读一遍。

        Args:
            image: PIL.Image
            instruction: 自然语言指令
            return_normalized: 是否返回归一化坐标 (0-1)
            max_new_tokens: 生成长度上限

        Returns:
            (x, y): 点击坐标，解析失败返回 (None, None)
        """
        point, _ = self.locate(image, instruction, max_new_tokens)
        if point is None:
            return None, None

        if return_normalized:
            width, height = image.size
            return point[0] / width, point[1] / height

        return point[0], point[1]

    def locate(self, image, instruction, max_new_tokens=64):
        """
        定位的核心实现

        比 predict 多返回一个模型原始输出，出问题的时候能看出来是解析的问题
        还是模型本身没学会。定位服务（serve/grounder_server.py）用的是这个。

        Args:
            image: PIL.Image
            instruction: 元素描述
            max_new_tokens: 生成长度上限

        Returns:
            (point, raw_output)
            point 是原图像素坐标 [x, y]，解析或转换失败时是 None
            raw_output 是模型输出的原始文本
        """
        image = image.convert("RGB")
        orig_width, orig_height = image.size

        # 构建 prompt（和训练/评估时的 build_eval_prompt 保持一致）
        system_prompt = "Based on the screenshot of the page, I give a text description and you give its corresponding location."
        if self.coord_format == "int1000":
            system_prompt += (" The coordinate represents a clickable location [x, y] for an element,"
                              " which is a relative coordinate on the screenshot, scaled from 1 to 1000.")
        else:
            system_prompt += " The coordinate represents a clickable location [x, y] for an element."

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image", "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                {"type": "text", "text": instruction},
            ]
        }]

        # 处理输入
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(text=prompt, images=[image], return_tensors="pt")

        # 移到 GPU
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if batch.get("pixel_values") is not None:
            batch["pixel_values"] = batch["pixel_values"].to(self.torch_dtype)

        generate_dict = {
            "input_ids": batch["input_ids"],
            "pixel_values": batch["pixel_values"],
            "image_grid_thw": batch["image_grid_thw"],
        }
        if batch.get("attention_mask") is not None:
            generate_dict["attention_mask"] = batch["attention_mask"]

        # 生成
        with torch.no_grad():
            generate_ids = self.model.generate(
                **generate_dict,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # 解码
        generate_ids = generate_ids[:, batch["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        # 解析坐标（和评估用同一套解析逻辑：正则提取 -> literal_eval -> 框取中心）
        pred_point = parse_predicted_point(output_text)
        if pred_point is None:
            print(f"解析坐标失败，模型输出: {output_text!r}")
            return None, output_text

        try:
            # 把模型输出的坐标转换回原图坐标
            pred_point = self._convert_point_to_original(
                pred_point, orig_height, orig_width
            )
        except Exception as e:
            print(f"坐标转换失败: {e}")
            return None, output_text

        return pred_point, output_text

    def _convert_point_to_original(self, point, orig_height, orig_width):
        """
        把模型输出的坐标转换回原图坐标

        实际转换在 utils/coordinate.decode_point，注意入参是 (height, width) 顺序。
        """
        return decode_point(
            point, (orig_width, orig_height),
            coord_format=self.coord_format,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def _smart_resize(self, height, width, factor=28):
        """
        算 Qwen2.5-VL resize 后的尺寸，实现在 utils/coordinate
        """
        return smart_resize(
            height, width, factor=factor,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="ScreenAgent 推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--image", type=str, required=True, help="截图路径")
    parser.add_argument("--instruction", type=str, required=True, help="指令")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--coord_format", type=str, default=DEFAULT_COORD_FORMAT,
                        choices=["qwen_abs", "norm", "int1000"],
                        help="模型输出的坐标格式，要和训练时一致")
    parser.add_argument("--min_visual_tokens", type=int, default=256)
    parser.add_argument("--max_visual_tokens", type=int, default=1280)
    args = parser.parse_args()

    # 加载模型
    model = ScreenAgentInference(
        args.model, device=args.device,
        coord_format=args.coord_format,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
    )

    # 推理
    x, y = model.predict(args.image, args.instruction)

    if x is not None:
        print(f"预测坐标: ({x}, {y})")
    else:
        print("预测失败")


if __name__ == "__main__":
    main()
