"""
ScreenAgent 推理模块

提供简单的 API 来使用训练好的模型进行 GUI 元素定位。

使用方法:
    from inference import ScreenAgentInference
    
    model = ScreenAgentInference("path/to/model")
    x, y = model.predict("path/to/screenshot.png", "点击登录按钮")
"""

import os
import sys
import ast
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# 添加项目根目录到 path，确保直接用 `python inference/inference.py` 运行时
# 也能 import 到项目根目录下的 utils 包（而不是只有 `inference/` 目录本身）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.qwen_vl_common import CHAT_TEMPLATE, smart_resize, qwen_coords_to_point, extract_coordinates


class ScreenAgentInference:
    """
    ScreenAgent 推理类
    
    加载训练好的模型，输入截图和指令，输出点击坐标。
    """
    
    def __init__(self, model_path, device="cuda:0", precision="bf16"):
        """
        初始化推理模型
        
        Args:
            model_path: 模型路径（合并后的完整模型）
            device: 运行设备
            precision: 精度 (bf16/fp16/fp32)
        """
        self.device = device
        
        # 确定数据类型
        if precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif precision == "fp16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        # 视觉 token 范围（和训练时保持一致）
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        
        print(f"加载模型: {model_path}")
        
        # 加载 processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        
        # 设置 chat template（和 train.py 共用同一份定义）
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
    
    def predict(self, image_path, instruction, return_normalized=False):
        """
        预测点击坐标
        
        Args:
            image_path: 截图路径
            instruction: 自然语言指令，比如 "点击登录按钮"
            return_normalized: 是否返回归一化坐标 (0-1)，默认返回像素坐标
        
        Returns:
            (x, y): 点击坐标
        """
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        
        # 构建 prompt（和训练时一致）
        system_prompt = "Based on the screenshot of the page, I give a text description and you give its corresponding location."
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
        
        # 生成
        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                max_new_tokens=2048,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # 解码
        generate_ids = generate_ids[:, batch["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        # 解析坐标（先尝试用正则从文本中提取 [x, y] 子串，兼容模型输出夹杂自然语言
        # 修饰的情况，比如"点击位置是[512, 300]"，容错能力和 evaluator.py 保持一致）
        try:
            coord_str = extract_coordinates(output_text)
            pred_point = ast.literal_eval(coord_str if coord_str else output_text.strip())

            # 如果是 bbox 格式，取中心点
            if len(pred_point) == 4:
                pred_point = [
                    (pred_point[0] + pred_point[2]) / 2,
                    (pred_point[1] + pred_point[3]) / 2
                ]

            # 把模型输出的坐标转换回原图坐标
            pred_point = self._convert_point_to_original(
                pred_point, orig_height, orig_width
            )

            if return_normalized:
                return pred_point[0] / orig_width, pred_point[1] / orig_height
            else:
                return pred_point[0], pred_point[1]

        except Exception as e:
            print(f"解析坐标失败: {e}")
            print(f"模型输出: {output_text}")
            return None, None

    def _convert_point_to_original(self, point, orig_height, orig_width):
        """
        把模型输出的坐标转换回原图坐标（委托给共享实现，
        和 evaluator.py 保持完全一致，见 utils/qwen_vl_common.py）

        模型输出的是基于 resize 后图像的绝对坐标，需要转回原图
        """
        return qwen_coords_to_point(
            point, orig_height, orig_width,
            factor=28, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def _smart_resize(self, height, width, factor=28):
        """计算 Qwen2.5-VL 的 resize 后尺寸（委托给共享实现，带尺寸/长宽比合法性校验）"""
        return smart_resize(height, width, factor, self.min_pixels, self.max_pixels)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ScreenAgent 推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--image", type=str, required=True, help="截图路径")
    parser.add_argument("--instruction", type=str, required=True, help="指令")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    args = parser.parse_args()
    
    # 加载模型
    model = ScreenAgentInference(args.model, device=args.device)
    
    # 推理
    x, y = model.predict(args.image, args.instruction)
    
    if x is not None:
        print(f"预测坐标: ({x}, {y})")
    else:
        print("预测失败")


if __name__ == "__main__":
    main()
