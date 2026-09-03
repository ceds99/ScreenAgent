# ScreenAgent

基于视觉语言模型的 GUI 元素定位系统。

## 项目介绍

ScreenAgent 能够根据自然语言指令，在屏幕截图中定位目标 UI 元素的坐标位置。这是构建智能 GUI 自动化代理的核心能力。

**示例：**
- 输入：截图 + "点击登录按钮"
- 输出：`[523, 847]`（元素中心坐标）

## 技术特点

- **轻量高效**：基于 3B 参数模型，单卡 4090 可训练
- **跨平台支持**：Mobile / Desktop / Web 全覆盖
- **数据高效**：仅需 24K 训练样本
- **两阶段训练**：先学通用能力，再适应高分辨率场景

## 性能指标

| 测试集 | 准确率 |
|--------|--------|
| ScreenSpot | 84.9% |
| ScreenSpot-v2 | 86.4% |

## 快速开始

详细使用说明请参考 [RUN_GUIDE.md](./RUN_GUIDE.md)

## 目录结构

```
ScreenAgent/
├── configs/          # 配置文件
├── data/             # 数据处理
├── models/           # 模型工具
├── trainer/          # 训练代码
├── inference/        # 推理代码
├── scripts/          # 运行脚本
├── datasets/         # 数据集
├── checkpoints/      # 模型权重
└── logs/             # 日志
```

## 环境要求

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+
- GPU: RTX 4090 (24GB) 或更高

## 训练流程

1. 下载基础模型和数据集
2. Stage1: 跨平台预训练
3. Stage2: 高分辨率微调
4. 评估和推理
