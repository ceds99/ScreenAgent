# ScreenAgent

我训练的一个 GUI 元素定位（Grounding）模型：输入一张界面截图和一句自然语言指令（如"点击登录按钮"），模型输出应点击的坐标位置。基于 Qwen2.5-VL-3B 与 LoRA 微调，采用两阶段训练——Stage1 在桌面、网页、移动端混合数据上进行跨平台预训练，Stage2 使用高分辨率数据做专项微调，整个训练过程在单卡 RTX 5090 上完成。

**示例：**
- 输入：截图 + "点击登录按钮"
- 输出：`[523, 847]`（元素中心坐标）

最终结果：ScreenSpot 上取得 84.98% 的准确率，ScreenSpot-v2 上取得 86.86%，训练规模仅为最初预算的三分之一。这两个数字背后是一段更值得记录的排查过程：训练一度看似运行正常、loss 持续下降，但评估准确率长期停滞在接近 0 的水平，且没有任何异常报错。我没有直接将其归因于模型能力不足，而是通过一系列受控实验逐步排查，最终定位到一个隐藏较深的坐标标注格式问题——完整的排查过程见 **[docs/REPORT.md](docs/REPORT.md)**。

## 最终结果

| 测试集 | 准确率 | 参考水平 |
|---|---|---|
| ScreenSpot | **84.98%** | 84.9% |
| ScreenSpot-v2 | **86.86%** | 86.4% |

<details>
<summary>各平台细分（点击展开）</summary>

**ScreenSpot**

| 平台 | icon | text |
|---|---|---|
| desktop | 75.00% | 93.81% |
| mobile | 80.35% | 94.87% |
| web | 76.70% | 89.13% |

**ScreenSpot-v2**

| 平台 | icon | text |
|---|---|---|
| desktop | 76.43% | 94.85% |
| mobile | 85.31% | 96.55% |
| web | 77.83% | 90.17% |

</details>

## 排查过程概述

训练初期，评估准确率仅为 0.3% 左右，且长期没有变化。经排查是标注框坐标单位换算错误，修复后准确率提升至 50% 左右，但随即进入平台期，且 mobile 平台的准确率显著低于 desktop 与 web，差距达到数倍。

为排除模型或训练框架层面的问题，我设计了一系列受控实验：分别在裸 LoRA、DeepSpeed 封装、真实 checkpoint 与真实评估数据三种条件下验证权重变化对模型输出的影响，确认问题不在模型或框架层面。随后对训练数据做了系统性的坐标格式审计——对每个数据集抽样并运行实际使用的坐标转换逻辑，统计数值被裁剪到画布边界的比例。结果显示，四个训练数据集中有三个（showui-web、amex、uground）的坐标标注实际为绝对像素值，但训练代码统一按归一化坐标处理，导致这些数据集中几乎所有训练标签都被错误映射到画布同一角落，与实际指令无关。

定位到根因后，我设计并实现了一套按数据集自动识别坐标格式（绝对像素/归一化、bbox 排列方式）并统一归一化的机制，替代了原有的隐式格式假设，避免同类问题在未来数据集变更时重现。训练策略上，没有沿用固定的 12-epoch 方案，而是基于每个 epoch 的真实验证集准确率判断收敛趋势，在进入平台期后主动终止训练——最终两阶段各仅训练 4 个 epoch，训练总时长压缩至约三分之一，结果仍达到并部分超过参考水平。

完整的实验设计、诊断证据、结果分析，以及若干值得记录的工程细节（如硬件从双卡迁移到单卡时如何保持训练动态一致），见 [docs/REPORT.md](docs/REPORT.md)。

## 环境与硬件

- GPU：单卡 NVIDIA RTX 5090（32GB）
- Python 3.10 / PyTorch 2.11 (cu128) / transformers 4.49.0 / deepspeed 0.19.6 / peft 0.14.0

## 快速开始

详细的环境搭建、数据/模型下载、训练、评估、推理步骤见 [RUN_GUIDE.md](./RUN_GUIDE.md)。

```bash
# 环境搭建见 RUN_GUIDE.md，此处假设已装好依赖、下载好模型和数据

# Stage1：跨平台预训练（showui-desktop + showui-web + amex）
bash scripts/train_stage1.sh

# 合并 LoRA 权重
python scripts/merge_weights.py --exp_dir logs/stage1/<时间戳>

# Stage2：高分辨率微调（uground）
STAGE1_MODEL=logs/stage1/<时间戳>/ckpt_model/merged_model bash scripts/train_stage2.sh

# 评估
bash scripts/evaluate.sh logs/stage2/<时间戳>/ckpt_model/merged_model screenspot
bash scripts/evaluate.sh logs/stage2/<时间戳>/ckpt_model/merged_model screenspot2
```

## 目录结构

```
ScreenAgent/
├── docs/             # 技术报告、代码审查记录
├── configs/          # 配置文件
├── data/             # 数据处理（含坐标格式自动检测/归一化）
├── models/           # 模型工具（LoRA target module 查找等）
├── trainer/          # 训练/评估循环
├── inference/        # 推理代码
├── utils/            # 坐标转换等共享工具
├── scripts/          # 训练/评估/合并权重脚本
├── datasets/         # 数据集（不纳入版本控制）
└── checkpoints/      # 模型权重（不纳入版本控制）
```

## License

MIT，见 [LICENSE](./LICENSE)。
