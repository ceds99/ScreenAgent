# ScreenAgent

我训练的 GUI 元素定位（Grounding）系统：给一张截图和一句自然语言指令，模型输出应该点击的坐标。基于 Qwen2.5-VL-3B + LoRA 两阶段训练，覆盖 Mobile / Desktop / Web 三端场景。

**示例：**
- 输入：截图 + "点击登录按钮"
- 输出：`[523, 847]`（元素中心坐标）

训练一开始用的是一套现成的模型/数据集/超参数配方，但真正决定结果的是后面发生的事：评估分数长期卡在接近 0 且完全不报错。从怎么判断问题出在哪、设计什么诊断实验去验证、定位根因、设计修复架构，到最后用什么策略决定训练到什么程度就停——这一整套排查和实验设计都是我独立完成的。完整过程、每一步的证据和最终技术报告见 **[docs/REPORT.md](docs/REPORT.md)**。

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

## 核心创新点

下面这几件事是我自己独立设计和实现的，不是套用现成脚本就能得到的，完整推导过程和量化证据见 [docs/REPORT.md](docs/REPORT.md)（开头的"核心创新点"表格可快速浏览）：

1. **三阶段受控消融诊断法**：评估分数长期卡死在 0.3% 附近时，没有直接假设"模型学不会"，而是设计实验分别在裸 LoRA、DeepSpeed 封装、真实 checkpoint+真实数据三个层面隔离验证，逐步排除框架层面的嫌疑，把排查方向锁定在数据本身。
2. **跨数据集坐标格式实证审计**：不依赖文档或假设，直接抽样跑一遍真实的坐标转换代码，量化统计出"四个训练数据集里有三个的坐标标注其实是绝对像素，但训练代码统一当归一化坐标处理，导致这三个数据集的训练标签被系统性地错误夹到画布右下角"——这是整个项目的根因，且原代码全程不报错。
3. **架构修复**：不是给单个数据集打补丁，而是设计了按数据集自动判别坐标格式（绝对/归一化、xyxy/xywh）并统一归一化的机制，避免未来数据集版本变化时重蹈覆辙。
4. **效率优先的训练策略**：不迷信原方案固定的 12-epoch 方案，而是实时监控每个 epoch 的真实验证集准确率，一旦进入平台期或已经追平/超过基线就主动停止训练——最终两阶段各只用 4 个 epoch，训练总时长压缩到约三分之一，结果反而与/略超原方案基线。

第 1 步的另一半是先做了一轮系统性代码审查，修复了 13 个正确性与鲁棒性问题（坐标系统不一致、LoRA 排除逻辑失效、`attention_mask` 丢失等），详见 [docs/code-review-findings.md](docs/code-review-findings.md) 和 [docs/code-review-fixes.md](docs/code-review-fixes.md)——但这一轮审查完成后训练依然跑不出结果，真正的根因是上面第 2 点。

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
- 坐标格式自动检测的架构思路参考了一份同事独立重写的实现；根因定位（找到问题出在哪三个数据集、为什么）是移植之前独立完成的，移植到本项目时重新核实了判别逻辑，并补回了几处在对方实现里被意外回退、但本项目此前已验证有效的修复（详见 [docs/REPORT.md](docs/REPORT.md) 第 3.4 节）

## License

MIT，见 [LICENSE](./LICENSE)。
