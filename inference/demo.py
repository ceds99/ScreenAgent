"""
ScreenAgent 演示脚本

展示如何使用训练好的模型进行 GUI 元素定位，并可视化结果。

使用方法:
    python inference/demo.py --model path/to/model --image path/to/screenshot.png --instruction "点击登录按钮"
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import ScreenAgentInference


def draw_point_on_image(image_path, point, output_path=None, radius=10):
    """
    在图片上画出预测的点击位置
    
    Args:
        image_path: 原图路径
        point: (x, y) 坐标
        output_path: 输出路径，默认在原图旁边生成 xxx_result.png
        radius: 点的半径
    
    Returns:
        输出图片路径
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    x, y = point
    
    # 画一个红色的圆点
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill='red',
        outline='red'
    )
    
    # 画十字线，更容易看清位置
    line_length = radius * 2
    draw.line([(x - line_length, y), (x + line_length, y)], fill='red', width=2)
    draw.line([(x, y - line_length), (x, y + line_length)], fill='red', width=2)
    
    # 保存结果
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_result{ext}"
    
    image.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="ScreenAgent 演示")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--image", type=str, required=True, help="截图路径")
    parser.add_argument("--instruction", type=str, required=True, help="指令，比如'点击登录按钮'")
    parser.add_argument("--output", type=str, default=None, help="结果图片保存路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--coord_format", type=str, default="qwen_abs",
                        choices=["qwen_abs", "norm", "int1000"],
                        help="模型输出的坐标格式，要和训练时一致")
    parser.add_argument("--min_visual_tokens", type=int, default=256)
    parser.add_argument("--max_visual_tokens", type=int, default=1280)
    args = parser.parse_args()
    
    print("=" * 60)
    print("ScreenAgent 演示")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"图片: {args.image}")
    print(f"指令: {args.instruction}")
    print("")
    
    # 检查文件
    if not os.path.exists(args.image):
        print(f"[错误] 图片不存在: {args.image}")
        return
    
    # 加载模型
    print("加载模型...")
    model = ScreenAgentInference(
        args.model,
        device=args.device,
        coord_format=args.coord_format,
        min_visual_tokens=args.min_visual_tokens,
        max_visual_tokens=args.max_visual_tokens,
    )
    
    # 推理
    print("执行推理...")
    x, y = model.predict(args.image, args.instruction)
    
    if x is not None:
        print(f"\n预测结果: ({x}, {y})")
        
        # 可视化
        output_path = draw_point_on_image(args.image, (x, y), args.output)
        print(f"结果已保存: {output_path}")
    else:
        print("\n预测失败")
    
    print("")
    print("=" * 60)


if __name__ == "__main__":
    main()
