# ScreenAgent

基于 Qwen2.5-VL-3B + LoRA 的 GUI 元素定位（Grounding）系统：给一张截图和一句自然语言指令，输出应该点击的坐标。

**示例：**
- 输入：截图 + "点击登录按钮"
- 输出：`[523, 847]`（元素中心坐标）

这是一个基于开源 ScreenAgent 训练方案的复现项目，但过程中偏离了"照着跑一遍"——训练刚跑起来评估分数就长期卡在接近 0，順藤摸瓜查出一个隐藏很深的数据标注格式 bug，绕了一大圈才真正跑出可用的结果。完整的排查过程、每一步的证据和最终技术报告见 **[docs/REPORT.md](docs/REPORT.md)**。

## 最终结果

单卡 RTX 5090，只用了原始训练配方 1/3 的 epoch 数（4+4 而非 12+12），结果：

| 测试集 | 本项目 | 原始基线 |
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

## 这个项目做了什么（不只是跑通训练脚本）

1. **代码审查**：系统排查了训练/评估/推理代码中的 13 个正确性与鲁棒性问题（坐标系统不一致、LoRA 排除逻辑失效、`attention_mask` 丢失等），详见 [docs/code-review-findings.md](docs/code-review-findings.md) 和 [docs/code-review-fixes.md](docs/code-review-fixes.md)。
2. **深层 bug 排查**：代码修好之后，评估分数依然长期卡死在 0.3% 附近纹丝不动。没有直接怀疑"模型学不会"，而是用一套三阶段的受控扰动实验，逐步排除 LoRA / DeepSpeed 包装层的嫌疑，最终定位到真正的根因——**四个训练数据集里有三个的坐标标注其实是绝对像素，但训练代码统一当归一化坐标处理，导致这三个数据集的训练标签被系统性地错误夹到画布右下角**。详见报告的"故障排查"章节。
3. **架构修复**：不是给单个数据集打补丁，而是引入了按数据集自动判别坐标格式（绝对/归一化、xyxy/xywh）并统一归一化的机制，避免未来数据集版本变化时重蹈覆辙。
4. **效率优先的训练策略**：不迷信论文原始的固定 12-epoch 方案，而是实时监控每个 epoch 的真实验证集准确率，一旦进入平台期或已经追平/超过基线就主动停止训练——最终两阶段各只用 4 个 epoch，训练总时长压缩到约三分之一，结果反而与/略超论文基线。

## 环境与硬件

- GPU：单卡 NVIDIA RTX 5090（32GB），原始方案假设 2×RTX 4090——用梯度累积把有效 batch size 保持在 96，训练动态与原方案等价
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
├── docs/             # 技术报告、代码审查记录（本项目的核心叙事都在这里）
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

## 致谢

- 基础训练方案（Qwen2.5-VL-3B + LoRA 两阶段训练配方、数据集选型）来自开源 ScreenAgent 项目
- 坐标格式自动检测的思路参考了一份同学/同事独立重写的实现，在此基础上做了整体移植，并补回了几处在移植过程中被回退、但已验证有效的修复

## License

MIT，见 [LICENSE](./LICENSE)。
