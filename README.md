# ScreenAgent

我训练的一个 GUI 元素定位模型：给它一张界面截图，再告诉它"点击登录按钮"，它会告诉我该点哪个坐标。底层是 Qwen2.5-VL-3B 加 LoRA，分两个阶段训练——先在桌面、网页、移动端混合的数据上打基础，再用一批高分辨率数据做专项微调，跑在单卡 RTX 5090 上。

**示例：**
- 输入：截图 + "点击登录按钮"
- 输出：`[523, 847]`（元素中心坐标）

最后的结果是 ScreenSpot 上 84.98%、ScreenSpot-v2 上 86.86%，训练规模压到了最初预算的三分之一。但比这两个数字更值得说的是拿到它们之前发生的事——训练一度跑起来了，loss 也在正常往下降，可评估分数长期卡在接近 0，而且不报任何错。我没有直接怀疑"模型学不会"，而是一步步设计实验去排除嫌疑，最后揪出一个藏得很深的坐标标注格式问题——完整的排查过程写在 **[docs/REPORT.md](docs/REPORT.md)** 里。

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

## 排查过程简述

训练刚跑起来时评估分数只有 0.3% 左右，完全不动。我先查了一遍评估代码，发现标注框的坐标单位弄错了，改完之后分数跳到 50% 左右——但接着就卡住了，怎么调都上不去，而且 mobile 平台的准确率断崖式地比 desktop、web 低了一大截。

这时候我没有急着怀疑模型能力不够，而是设计了几组对照实验：先在没有 DeepSpeed 的情况下扰动 LoRA 权重看输出变不变，再在 DeepSpeed 包装下重复一遍，最后直接拿真实训练出来的权重在真实图片上跑一遍推理——三轮下来确认模型和训练框架都没问题。于是我把矛头转向数据本身，抽样跑了一遍每个训练集的坐标转换逻辑，统计有多少坐标被数值裁剪函数夹到了画布角落。结果很清楚：四个训练集里有三个，坐标标注其实是绝对像素，但代码统一按 0-1 归一化处理，导致这三个数据集里几乎每一条标签都被错误地钉在了同一个角落——不管指令说的是什么。

找到根因之后，我没有为这三个数据集单独打补丁，而是写了一套自动判别坐标格式（绝对像素还是归一化、bbox 是哪种排列）的检测逻辑，加载数据时统一跑一遍、统一归一化，这样以后换数据集也不会再踩同一个坑。训练策略上也没有照搬固定的 epoch 数——两个阶段我都是盯着每个 epoch 的真实验证集准确率，一看到进入平台期或者已经追上目标水平就直接停，最终两阶段各只训了 4 个 epoch，总时长压缩到约三分之一，结果反而没有掉。

详细的数据、每一步的证据、还有几个我认为值得记录的工程细节（比如硬件从双卡换成单卡时怎么保证训练动态不变），都写在 [docs/REPORT.md](docs/REPORT.md) 里。

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
