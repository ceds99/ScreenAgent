#!/bin/bash
# ============================================================
# 下载训练和评估数据集
#
# 使用方法:
#   bash scripts/download_data.sh all       # 下载全部数据集
#   bash scripts/download_data.sh train     # 只下载训练数据集
#   bash scripts/download_data.sh eval      # 只下载评估数据集
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_DIR="${PROJECT_DIR}/datasets/train"
EVAL_DIR="${PROJECT_DIR}/datasets/eval"

# 默认下载全部
TARGET=${1:-all}

echo "============================================================"
echo "下载数据集"
echo "目标: ${TARGET}"
echo "训练数据路径: ${TRAIN_DIR}"
echo "评估数据路径: ${EVAL_DIR}"
echo "============================================================"

# 创建目录
mkdir -p "${TRAIN_DIR}"
mkdir -p "${EVAL_DIR}"

# 下载训练数据集的函数
download_train_data() {
    echo ""
    echo "========== 下载训练数据集 =========="

    # 训练数据集列表（HuggingFace 仓库名）
    # 格式: "仓库名|本地目录名"
    TRAIN_DATASETS=(
        "zonghanHZH/UGround-V1-8k|UGround-V1-8k"
        "zonghanHZH/AMEX-8k|AMEX-8k"
        "zonghanHZH/ShowUI-web-8k|ShowUI-web-8k"
        "showlab/ShowUI-desktop|ShowUI-desktop"
    )

    for item in "${TRAIN_DATASETS[@]}"; do
        REPO_NAME="${item%%|*}"
        LOCAL_NAME="${item##*|}"
        LOCAL_PATH="${TRAIN_DIR}/${LOCAL_NAME}"

        echo ""
        echo "[下载] ${REPO_NAME} -> ${LOCAL_PATH}"

        if [ -d "${LOCAL_PATH}" ] && [ "$(ls -A ${LOCAL_PATH} 2>/dev/null)" ]; then
            echo "  已存在，跳过"
            continue
        fi

        python -c "
from huggingface_hub import snapshot_download
import os

save_path = '${LOCAL_PATH}'
os.makedirs(save_path, exist_ok=True)

print(f'  正在下载 ${REPO_NAME}...')
snapshot_download(
    repo_id='${REPO_NAME}',
    repo_type='dataset',
    local_dir=save_path,
    local_dir_use_symlinks=False
)
print(f'  完成!')
"
    done

    echo ""
    echo "训练数据集下载完成！"
}

# 下载评估数据集的函数
download_eval_data() {
    echo ""
    echo "========== 下载评估数据集 =========="

    # 评估数据集列表
    EVAL_DATASETS=(
        "KevinQHLin/ScreenSpot|ScreenSpot"
        "zonghanHZH/ScreenSpot-v2|ScreenSpot-v2"
    )

    for item in "${EVAL_DATASETS[@]}"; do
        REPO_NAME="${item%%|*}"
        LOCAL_NAME="${item##*|}"
        LOCAL_PATH="${EVAL_DIR}/${LOCAL_NAME}"

        echo ""
        echo "[下载] ${REPO_NAME} -> ${LOCAL_PATH}"

        if [ -d "${LOCAL_PATH}" ] && [ "$(ls -A ${LOCAL_PATH} 2>/dev/null)" ]; then
            echo "  已存在，跳过"
            continue
        fi

        python -c "
from huggingface_hub import snapshot_download
import os

save_path = '${LOCAL_PATH}'
os.makedirs(save_path, exist_ok=True)

print(f'  正在下载 ${REPO_NAME}...')
snapshot_download(
    repo_id='${REPO_NAME}',
    repo_type='dataset',
    local_dir=save_path,
    local_dir_use_symlinks=False
)
print(f'  完成!')
"
    done

    echo ""
    echo "评估数据集下载完成！"
}

# 根据参数执行下载
case $TARGET in
    "all")
        download_train_data
        download_eval_data
        ;;
    "train")
        download_train_data
        ;;
    "eval")
        download_eval_data
        ;;
    *)
        echo "错误: 未知的目标 '${TARGET}'"
        echo "使用方法: bash scripts/download_data.sh [all|train|eval]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "数据集下载完成！"
echo "============================================================"

echo ""
echo "训练数据集:"
ls -lh "${TRAIN_DIR}" 2>/dev/null || echo "  (空)"

echo ""
echo "评估数据集:"
ls -lh "${EVAL_DIR}" 2>/dev/null || echo "  (空)"
