#!/bin/bash
# ============================================================
# Stage1 训练脚本
#
# Stage1 是跨平台预训练阶段，使用多个数据集混合训练，
# 让模型学习不同平台（Desktop、Web、Mobile）的 UI 元素定位能力。
#
# 使用方法:
#   bash scripts/train_stage1.sh
#
# 2卡 4090 实测显存占用约 20GB/卡
# ============================================================

set -e

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_DIR}"

echo "============================================================"
echo "ScreenAgent Stage1 训练"
echo "项目目录: ${PROJECT_DIR}"
echo "============================================================"

# ==================== 配置区域 ====================

# 实验名称
EXP_ID="stage1"

# GPU 设置
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# 模型路径
MODEL_DIR="${PROJECT_DIR}/checkpoints/base_model"

# 数据集目录
DATASET_DIR="${PROJECT_DIR}/datasets"

# 训练数据集配置（和原始项目一致：3个数据集，不含 uground）
TRAIN_DATASETS="showui-desktop,showui-web,amex"
TRAIN_JSONS="hf_train,hf_train,hf_train"
TRAIN_RATIOS="1,1,1"

# 评估数据集
VAL_DATASET="screenspot"
VAL_JSON="hf_test_full"

# 训练超参数（和原始项目一致）
EPOCHS=12
STEPS_PER_EPOCH=122
BATCH_SIZE=1
GRAD_ACCUM=48
LR=0.0001
WARMUP_STEPS=122

# LoRA 配置
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# 多轮对话设置（关键参数！原始项目用 30）
NUM_TURN=30

# 视觉 token 数量
MIN_VISUAL_TOKENS=256
MAX_VISUAL_TOKENS=1280

# 其他设置
PRECISION="bf16"
WORKERS=0
PRINT_FREQ=1

# ==================== 检查环境 ====================

echo ""
echo "[检查] 模型路径..."
MODEL_PATH=""
if [ -f "${MODEL_DIR}/config.json" ]; then
    MODEL_PATH="${MODEL_DIR}"
else
    CONFIG_FILE=$(find "${MODEL_DIR}" -name "config.json" -type f 2>/dev/null | head -1)
    if [ -n "${CONFIG_FILE}" ]; then
        MODEL_PATH="$(dirname "${CONFIG_FILE}")"
    fi
fi

if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[错误] 找不到模型文件，请先运行 bash scripts/download_model.sh"
    exit 1
fi
echo "  模型路径: ${MODEL_PATH}"

echo ""
echo "[检查] 数据集..."
if [ ! -d "${DATASET_DIR}/train" ]; then
    echo "[错误] 找不到训练数据集，请先运行 bash scripts/download_data.sh"
    exit 1
fi
echo "  训练数据: ${DATASET_DIR}/train"
echo "  评估数据: ${DATASET_DIR}/eval"

echo ""
echo "[检查] GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ==================== 开始训练 ====================

echo "============================================================"
echo "开始 Stage1 训练"
echo "============================================================"
echo "  实验ID: ${EXP_ID}"
echo "  GPU数量: ${NUM_GPUS}"
echo "  Epochs: ${EPOCHS}"
echo "  Steps/Epoch: ${STEPS_PER_EPOCH}"
echo "  Batch Size: ${BATCH_SIZE} x ${GRAD_ACCUM} (梯度累积)"
echo "  学习率: ${LR}"
echo "  多轮对话: ${NUM_TURN} 轮"
echo "  LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "============================================================"
echo ""

# 启动训练
deepspeed --num_gpus=${NUM_GPUS} \
    trainer/train.py \
    --exp_id "${EXP_ID}" \
    --local_weight \
    --local_weight_dir "${MODEL_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --train_dataset "${TRAIN_DATASETS}" \
    --train_json "${TRAIN_JSONS}" \
    --train_ratio "${TRAIN_RATIOS}" \
    --val_dataset "${VAL_DATASET}" \
    --val_json "${VAL_JSON}" \
    --epochs ${EPOCHS} \
    --steps_per_epoch ${STEPS_PER_EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --grad_accumulation_steps ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_turn ${NUM_TURN} \
    --min_visual_tokens ${MIN_VISUAL_TOKENS} \
    --max_visual_tokens ${MAX_VISUAL_TOKENS} \
    --precision "${PRECISION}" \
    --workers ${WORKERS} \
    --print_freq ${PRINT_FREQ} \
    --gradient_checkpointing \
    --random_sample \
    --record_sample \
    --uniform_prompt

echo ""
echo "============================================================"
echo "Stage1 训练完成！"
echo "检查点保存在: logs/${EXP_ID}/"
echo "============================================================"
