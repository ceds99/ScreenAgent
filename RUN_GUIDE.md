# ScreenAgent 项目运行指南

## 项目结构

```
ScreenAgent/
├── configs/                    # 配置文件
│   ├── deepspeed_zero2.json   # DeepSpeed ZeRO-2 优化配置
│   └── train_config.py        # 训练参数说明
│
├── data/                       # 数据处理模块
│   ├── __init__.py
│   ├── data_utils.py          # 数据工具函数（AverageMeter等）
│   ├── template.py            # Prompt模板和格式转换
│   ├── train_dataset.py       # 训练数据集
│   ├── eval_dataset.py        # 评估数据集
│   └── base_dataset.py        # 混合数据集和collate_fn
│
├── models/                     # 模型相关
│   ├── __init__.py
│   └── model_utils.py         # 模型工具函数
│
├── trainer/                    # 训练相关
│   ├── __init__.py
│   ├── train.py               # 主训练入口（deepspeed启动）
│   ├── trainer.py             # 训练循环逻辑
│   └── evaluator.py           # ScreenSpot评估逻辑
│
├── scripts/                    # 运行脚本
│   ├── download_model.sh      # 下载模型
│   ├── download_data.sh       # 下载数据
│   ├── train_debug.sh         # 调试训练（验证代码）
│   ├── train_stage1.sh        # Stage1训练
│   ├── train_stage2.sh        # Stage2训练
│   ├── merge_weights.py       # 合并权重
│   └── evaluate.sh            # 评估脚本
│
├── inference/                  # 推理模块
│   ├── __init__.py
│   ├── inference.py           # 推理脚本
│   └── demo.ipynb             # 演示Notebook
│
├── utils/                      # 工具函数
│   ├── __init__.py
│   └── common.py              # JSON保存、日志目录等通用工具
│
├── datasets/                   # 数据集存放目录
│   ├── train/                 # 训练数据
│   └── eval/                  # 评估数据
│
├── checkpoints/               # 模型检查点
│   ├── base_model/           # 基础模型 Qwen2.5-VL-3B
│   ├── stage1/               # Stage1 训练结果
│   └── stage2/               # Stage2 训练结果（最终模型）
│
├── logs/                      # 训练日志
├── requirements.txt           # Python依赖
├── RUN_GUIDE.md              # 运行指南
└── README.md                 # 项目说明
```

---

## 一、环境搭建

> **重要提示**：安装顺序很重要！必须先装 PyTorch，再装其他依赖。

### 1.1 创建虚拟环境

```bash
conda create -n screenagent python=3.10 -y
conda activate screenagent
```

### 1.2 安装 PyTorch（必须先安装）

```bash
# CUDA 11.8 版本（根据你的 CUDA 版本选择）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

验证安装：
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

### 1.3 安装其他依赖

```bash
cd /path/to/ScreenAgent
pip install -r requirements.txt
```

### 1.4 确认 PyTorch 没被覆盖

有些依赖可能会覆盖 PyTorch 版本，安装完后再检查一次：
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

如果版本变了（不是 2.1.2+cu118），需要重新安装：
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### 1.5 验证关键依赖

```bash
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import deepspeed; print(f'deepspeed: {deepspeed.__version__}')"
python -c "import peft; print(f'peft: {peft.__version__}')"
python -c "import accelerate; print(f'accelerate: {accelerate.__version__}')"
```

### 1.6 完整验证脚本

一次性验证所有关键依赖：
```bash
python -c "
import torch
import transformers
import deepspeed
import peft
import accelerate
import bitsandbytes

print('=' * 50)
print('环境验证')
print('=' * 50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
print(f'transformers: {transformers.__version__}')
print(f'deepspeed: {deepspeed.__version__}')
print(f'peft: {peft.__version__}')
print(f'accelerate: {accelerate.__version__}')
print('=' * 50)
print('环境配置成功！')
"
```

### 1.7 常见问题

**Q1: pip install 时 torch 被覆盖了怎么办？**

A: 重新安装指定版本的 PyTorch：
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

**Q2: bitsandbytes 安装失败？**

A: 尝试以下命令：
```bash
pip install bitsandbytes==0.43.1 --no-cache-dir
```

**Q3: deepspeed 编译报错？**

A: 确保系统安装了编译工具：
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# 或者使用 conda
conda install -c conda-forge compilers
```

**Q4: CUDA 版本不匹配？**

A: 检查系统 CUDA 版本：
```bash
nvcc --version
nvidia-smi
```
如果是 CUDA 12.x，可以尝试：
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## 二、数据/模型下载

### 2.1 下载基础模型（Qwen2.5-VL-3B）

基础模型约 6GB，下载需要一些时间。

**方式一：从 ModelScope 下载（国内推荐）**
```bash
cd /path/to/ScreenAgent
bash scripts/download_model.sh modelscope
```

**方式二：从 HuggingFace 下载**
```bash
cd /path/to/ScreenAgent
bash scripts/download_model.sh huggingface
```

下载完成后，模型保存在 `checkpoints/base_model/` 目录。

**验证模型下载：**
```bash
ls -lh checkpoints/base_model/
```

### 2.2 下载数据集

数据集从 HuggingFace 下载，包括训练集和评估集。

**下载全部数据集（推荐）：**
```bash
cd /path/to/ScreenAgent
bash scripts/download_data.sh all
```

**只下载训练数据集：**
```bash
bash scripts/download_data.sh train
```

**只下载评估数据集：**
```bash
bash scripts/download_data.sh eval
```

### 2.3 数据集说明

**训练数据集**（共约 24K 样本）：

| 数据集 | 说明 | 样本数 |
|--------|------|--------|
| UGround-V1-8k | 多分辨率 Web 截图 | ~8K |
| AMEX-8k | Mobile 应用截图 | ~8K |
| ShowUI-web-8k | Web 界面截图 | ~8K |
| ShowUI-desktop | Desktop 界面截图 | 数千 |

**评估数据集**：

| 数据集 | 说明 |
|--------|------|
| ScreenSpot | GUI Grounding 标准测试集 |
| ScreenSpot-v2 | 改进版测试集（修复了标注问题） |

### 2.4 验证数据集

```bash
# 查看训练数据
ls -lh datasets/train/

# 查看评估数据
ls -lh datasets/eval/
```

---

## 三、模型训练

训练分为两个阶段：
- **Stage1**：跨平台预训练，使用 3 个数据集混合训练（showui-desktop, showui-web, amex）
- **Stage2**：专项训练，只使用 uground 数据集

### 3.1 快速验证（调试模式）

在正式训练前，先跑一下调试脚本，确保代码没问题：

```bash
cd /path/to/ScreenAgent
bash scripts/train_debug.sh
```

这个脚本只跑 10 个 step，不保存模型，用来验证训练流程。

### 3.2 Stage1 训练

Stage1 使用 3 个数据集混合训练，让模型学习跨平台的 UI 元素定位能力。

```bash
cd /path/to/ScreenAgent
bash scripts/train_stage1.sh
```

**默认配置：**

| 参数 | 值 | 说明 |
|------|------|------|
| GPU | 2 卡 | |
| Epochs | 12 | |
| Steps/Epoch | 122 | |
| Batch Size | 1 x 48 | batch_size=1，梯度累积=48 |
| 有效 Batch | 96 | 2卡 x 48 |
| 学习率 | 1e-4 | |
| num_turn | 30 | 多轮对话（关键！） |
| LoRA r | 8 | |
| 数据集 | 3个 | showui-desktop, showui-web, amex |

**显存占用：** 约 20GB/卡（RTX 4090）

训练完成后，检查点保存在 `logs/stage1/时间戳/ckpt_model/` 目录。

### 3.3 合并 Stage1 权重

Stage1 训练完成后，需要把 LoRA 权重合并到基础模型：

```bash
cd /path/to/ScreenAgent
python scripts/merge_weights.py --exp_dir logs/stage1/时间戳
```

合并后的模型保存在 `logs/stage1/时间戳/ckpt_model/merged_model/` 目录。

### 3.4 Stage2 训练

Stage2 在 Stage1 合并模型的基础上，只使用 uground 数据集继续训练：

```bash
cd /path/to/ScreenAgent
# 需要先编辑脚本，设置 STAGE1_MERGED_MODEL 变量
bash scripts/train_stage2.sh
```

**Stage2 配置：**

| 参数 | 值 | 说明 |
|------|------|------|
| 基础模型 | Stage1 合并模型 | |
| 学习率 | 5e-5 | 比 Stage1 低 |
| 数据集 | uground | 只用一个 |

### 3.5 完整训练流程

```bash
# 1. 调试验证
bash scripts/train_debug.sh

# 2. Stage1 训练
bash scripts/train_stage1.sh 2>&1 | tee train_stage1.log

# 3. 合并 Stage1 权重
python scripts/merge_weights.py --exp_dir logs/stage1/时间戳

# 4. Stage2 训练（需要先设置 Stage1 模型路径）
bash scripts/train_stage2.sh 2>&1 | tee train_stage2.log

# 5. 合并 Stage2 权重（最终模型）
python scripts/merge_weights.py --exp_dir logs/stage2/时间戳
```

### 3.6 常见问题

**Q1: CUDA out of memory**

A: 减小 `grad_accumulation_steps`（比如从 48 改成 24）

**Q2: 评估准确率很低**

A: 检查 `num_turn` 是否设置为 30，这是关键参数！

**Q3: 训练很慢**

A: `grad_accumulation_steps=48` 是正常设置，每 48 个 batch 才更新一次参数。

### 3.7 自定义训练参数

如果需要调整参数，可以直接修改脚本中的配置区域，或者手动运行：

```bash
deepspeed --num_gpus=2 trainer/train.py \
    --exp_id "my_exp" \
    --local_weight \
    --local_weight_dir "./checkpoints/base_model/path/to/model" \
    --dataset_dir "./datasets" \
    --train_dataset "showui-web,amex" \
    --train_json "hf_train,hf_train" \
    --train_ratio "1,1" \
    --epochs 5 \
    --batch_size 1 \
    --lr 1e-4 \
    --lora_r 8 \
    --gradient_checkpointing \
    --random_sample
```

### 3.8 常见问题

**Q1: CUDA out of memory**

A: 尝试以下方法：
- 减小 `max_visual_tokens`（比如从 1280 改成 768）
- 减小 `batch_size`
- 开启 `gradient_checkpointing`（脚本默认已开启）

**Q2: 找不到数据集**

A: 检查 `datasets/train/` 目录下是否有数据，每个数据集目录下应该有 `images/` 和 `metadata/` 子目录。

**Q3: 训练 loss 不下降**

A: 检查学习率是否合适，LoRA 训练一般用 1e-4 到 1e-5。

---

## 四、模型评估

训练完成后，可以在 ScreenSpot 数据集上评估模型效果。

### 4.1 评估命令

```bash
cd /path/to/ScreenAgent

# 在 ScreenSpot 上评估
bash scripts/evaluate.sh logs/stage2/时间戳/ckpt_model/merged_model screenspot

# 在 ScreenSpot-v2 上评估
bash scripts/evaluate.sh logs/stage2/时间戳/ckpt_model/merged_model screenspot2
```

### 4.2 评估指标

评估会输出各类别的准确率：

| 指标 | 说明 |
|------|------|
| desktop/icon Accuracy | 桌面端图标定位准确率 |
| desktop/text Accuracy | 桌面端文字定位准确率 |
| mobile/icon Accuracy | 移动端图标定位准确率 |
| mobile/text Accuracy | 移动端文字定位准确率 |
| web/icon Accuracy | 网页端图标定位准确率 |
| web/text Accuracy | 网页端文字定位准确率 |
| Avg Accuracy | 平均准确率 |

---

## 五、模型推理

训练好的模型可以用来对任意截图进行 GUI 元素定位。

### 5.1 Python API

```python
from inference import ScreenAgentInference

# 加载模型
model = ScreenAgentInference("logs/stage2/时间戳/ckpt_model/merged_model")

# 预测点击坐标
x, y = model.predict("screenshot.png", "点击登录按钮")
print(f"预测坐标: ({x}, {y})")
```

### 5.2 命令行推理

```bash
cd /path/to/ScreenAgent

python inference/inference.py \
    --model logs/stage2/时间戳/ckpt_model/merged_model \
    --image screenshot.png \
    --instruction "点击登录按钮"
```

### 5.3 可视化演示

```bash
# 会在图片上画出预测的点击位置，并保存为 xxx_result.png
python inference/demo.py \
    --model logs/stage2/时间戳/ckpt_model/merged_model \
    --image screenshot.png \
    --instruction "点击登录按钮"
```

---

