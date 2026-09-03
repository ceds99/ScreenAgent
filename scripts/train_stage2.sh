#!/bin/bash
# ============================================================
# Stage2 训练脚本
#
# Stage2 在 Stage1 的基础上继续训练，只使用 uground 数据集。
# 需要先完成 Stage1 训练，然后合并权重再进行 Stage2。
#
# 使用方法:
#   bash scripts/train_stage2.sh
#
# 注意：需要先完成 Stage1 训练并合并权重
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_DIR}"

echo "============================================================"
echo "ScreenAgent Stage2 训练"
echo "项目目录: ${PROJECT_DIR}"
echo "============================================================"

# ==================== 配置区域 ====================

# 实验名称
EXP_ID="stage2"

# GPU 设置
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# Stage1 合并后的模型路径
# 可以通过环境变量指定：STAGE1_MODEL="xxx" bash scripts/train_stage2.sh
# 或者直接在这里填写路径
STAGE1_MERGED_MODEL="${STAGE1_MODEL:-}"

# 如果没有指定，尝试自动查找
if [ -z "${STAGE1_MERGED_MODEL}" ]; then
    STAGE1_DIR="${PROJECT_DIR}/logs/stage1"
    if [ -d "${STAGE1_DIR}" ]; then
        LATEST_DIR=$(ls -td "${STAGE1_DIR}"/*/ 2>/dev/null | head -1)
        if [ -n "${LATEST_DIR}" ] && [ -d "${LATEST_DIR}/ckpt_model/merged_model" ]; then
            STAGE1_MERGED_MODEL="${LATEST_DIR}/ckpt_model/merged_model"
        fi
    fi
fi

# 数据集目录
DATASET_DIR="${PROJECT_DIR}/datasets"

# 训练数据集配置（Stage2 只用 uground）
TRAIN_DATASETS="uground"
TRAIN_JSONS="hf_train"
TRAIN_RATIOS="1"

# 评估数据集
VAL_DATASET="screenspot"
VAL_JSON="hf_test_full"

# 训练超参数（和原始项目一致，学习率更低）
EPOCHS=12
STEPS_PER_EPOCH=122
BATCH_SIZE=1
GRAD_ACCUM=48
LR=0.00005
WARMUP_STEPS=122

# LoRA 配置
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# 多轮对话设置
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
echo "[检查] Stage1 合并模型..."
if [ -z "${STAGE1_MERGED_MODEL}" ] || [ ! -d "${STAGE1_MERGED_MODEL}" ]; then
    echo "[错误] 找不到 Stage1 合并后的模型"
    echo ""
    echo "请先完成以下步骤："
    echo "1. 运行 Stage1 训练: bash scripts/train_stage1.sh"
    echo "2. 合并 LoRA 权重: python scripts/merge_weights.py --ckpt_dir logs/stage1/时间戳/ckpt_model"
    echo "3. 设置 STAGE1_MERGED_MODEL 变量指向合并后的模型路径"
    echo ""
    exit 1
fi
echo "  Stage1 模型: ${STAGE1_MERGED_MODEL}"

echo ""
echo "[检查] 数据集..."
if [ ! -d "${DATASET_DIR}/train/UGround-V1-8k" ]; then
    echo "[错误] 找不到 UGround 数据集"
    exit 1
fi
echo "  训练数据: ${DATASET_DIR}/train/UGround-V1-8k"

echo ""
echo "[检查] GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ==================== 开始训练 ====================

echo "============================================================"
echo "开始 Stage2 训练"
echo "============================================================"
echo "  实验ID: ${EXP_ID}"
echo "  GPU数量: ${NUM_GPUS}"
echo "  基础模型: Stage1 合并模型"
echo "  数据集: uground"
echo "  Epochs: ${EPOCHS}"
echo "  学习率: ${LR}"
echo "============================================================"
echo ""

# 启动训练
deepspeed --num_gpus=${NUM_GPUS} \
    trainer/train.py \
    --exp_id "${EXP_ID}" \
    --local_weight \
    --local_weight_dir "${STAGE1_MERGED_MODEL}" \
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
echo "Stage2 训练完成！"
echo "检查点保存在: logs/${EXP_ID}/"
echo "============================================================"
