#!/bin/bash
# ============================================================
# 评估脚本
#
# 在 ScreenSpot 或 ScreenSpot-v2 数据集上评估模型
#
# 使用方法:
#   bash scripts/evaluate.sh [模型路径] [数据集]
#
# 示例:
#   bash scripts/evaluate.sh logs/stage2/xxx/ckpt_model/merged_model screenspot
#   bash scripts/evaluate.sh logs/stage2/xxx/ckpt_model/merged_model screenspot2
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_DIR}"

# 参数处理
MODEL_PATH="${1:-}"
VAL_DATASET="${2:-screenspot}"

echo "============================================================"
echo "ScreenAgent 模型评估"
echo "============================================================"

# 检查模型路径
if [ -z "${MODEL_PATH}" ]; then
    echo "用法: bash scripts/evaluate.sh [模型路径] [数据集]"
    echo ""
    echo "参数说明:"
    echo "  模型路径: 合并后的模型目录，比如 logs/stage2/xxx/ckpt_model/merged_model"
    echo "  数据集: screenspot 或 screenspot2（默认 screenspot）"
    echo ""
    echo "示例:"
    echo "  bash scripts/evaluate.sh logs/stage2/2025-01-30_xx/ckpt_model/merged_model screenspot"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[错误] 找不到模型: ${MODEL_PATH}"
    echo "请确保路径指向合并后的模型目录（包含 config.json）"
    exit 1
fi

echo "模型路径: ${MODEL_PATH}"
echo "评估数据集: ${VAL_DATASET}"
echo "坐标格式: ${COORD_FORMAT}"

# 数据集目录
DATASET_DIR="${PROJECT_DIR}/datasets"
if [ ! -d "${DATASET_DIR}/eval" ]; then
    echo "[错误] 找不到评估数据集"
    exit 1
fi

# GPU 设置
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

# 评估参数
VAL_JSON="hf_test_full"
MIN_VISUAL_TOKENS=256
MAX_VISUAL_TOKENS=1280
PRECISION="bf16"

# 坐标格式，要和训练时的 --coord_format 一样，不一样准确率会掉到接近随机
# 不确定训练时用的什么，查 logs/<exp_id>/<时间戳>/args.json
COORD_FORMAT="${COORD_FORMAT:-qwen_abs}"

echo ""
echo "[检查] GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "============================================================"
echo "开始评估..."
echo "============================================================"

# 运行评估（使用 eval_only 模式）
deepspeed --num_gpus=${NUM_GPUS} \
    trainer/train.py \
    --exp_id "eval_${VAL_DATASET}" \
    --local_weight \
    --local_weight_dir "${MODEL_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --val_dataset "${VAL_DATASET}" \
    --val_json "${VAL_JSON}" \
    --coord_format "${COORD_FORMAT}" \
    --min_visual_tokens ${MIN_VISUAL_TOKENS} \
    --max_visual_tokens ${MAX_VISUAL_TOKENS} \
    --precision "${PRECISION}" \
    --lora_r 0 \
    --eval_only \
    --uniform_prompt

echo ""
echo "============================================================"
echo "评估完成！"
echo "============================================================"
