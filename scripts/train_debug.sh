#!/bin/bash
# ============================================================
# 调试训练脚本
#
# 用最小配置快速验证训练流程是否正常。
# 只跑几个 step，不保存模型，不记录日志。
#
# 使用方法:
#   bash scripts/train_debug.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_DIR}"

echo "============================================================"
echo "ScreenAgent 调试训练"
echo "============================================================"

# GPU 设置（调试时用单卡就行）
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

# 和正式训练保持一致
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 模型路径
MODEL_DIR="${PROJECT_DIR}/checkpoints/base_model"
MODEL_PATH=""
if [ -f "${MODEL_DIR}/config.json" ]; then
    MODEL_PATH="${MODEL_DIR}"
else
    CONFIG_FILE=$(find "${MODEL_DIR}" -name "config.json" -type f 2>/dev/null | head -1)
    if [ -n "${CONFIG_FILE}" ]; then
        MODEL_PATH="$(dirname "${CONFIG_FILE}")"
    fi
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "[错误] 找不到模型，请先下载"
    exit 1
fi
echo "模型路径: ${MODEL_PATH}"

# 数据集
DATASET_DIR="${PROJECT_DIR}/datasets"
if [ ! -d "${DATASET_DIR}/train" ]; then
    echo "[错误] 找不到数据集"
    exit 1
fi

# 多轮对话轮数，默认 1 最省显存。正式训练用的是 30，
# 想顺便验证多轮那条路径就跑 NUM_TURN=30 bash scripts/train_debug.sh
NUM_TURN="${NUM_TURN:-1}"

echo ""
echo "开始调试训练（只跑 10 个 step，num_turn=${NUM_TURN}）..."
echo "建议再用 NUM_TURN=30 跑一次，正式训练走的是多轮，显存占用不一样"
echo ""

# 用最小配置启动
deepspeed --num_gpus=${NUM_GPUS} \
    trainer/train.py \
    --exp_id "debug" \
    --local_weight \
    --local_weight_dir "${MODEL_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --train_dataset "showui-web" \
    --train_json "hf_train" \
    --train_ratio "1" \
    --val_dataset "screenspot" \
    --val_json "hf_test_full" \
    --epochs 1 \
    --steps_per_epoch 10 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --lr 1e-4 \
    --warmup_steps 2 \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_turn ${NUM_TURN} \
    --coord_format "qwen_abs" \
    --min_visual_tokens 256 \
    --max_visual_tokens 512 \
    --precision "bf16" \
    --workers 2 \
    --print_freq 1 \
    --gradient_checkpointing \
    --no_eval \
    --debug

echo ""
echo "============================================================"
echo "调试完成！如果没有报错，说明训练流程正常"
echo "============================================================"
