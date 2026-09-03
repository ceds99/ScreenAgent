#!/bin/bash
# ============================================================
# 下载 Qwen2.5-VL-3B-Instruct 基础模型
#
# 使用方法:
#   bash scripts/download_model.sh modelscope   # 从 ModelScope 下载（国内推荐）
#   bash scripts/download_model.sh huggingface  # 从 HuggingFace 下载
# ============================================================

set -e

# 获取脚本所在目录，然后定位到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/checkpoints/base_model"

# 模型名称
MODEL_NAME="Qwen2.5-VL-3B-Instruct"

# 默认使用 ModelScope
SOURCE=${1:-modelscope}

echo "============================================================"
echo "下载基础模型: ${MODEL_NAME}"
echo "下载源: ${SOURCE}"
echo "保存路径: ${MODEL_DIR}"
echo "============================================================"

# 创建目录
mkdir -p "${MODEL_DIR}"

if [ "$SOURCE" == "modelscope" ]; then
    echo ""
    echo "[1/2] 安装 modelscope（如果没装的话）..."
    pip install modelscope -q

    echo ""
    echo "[2/2] 从 ModelScope 下载模型..."
    echo "模型地址: https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct"
    echo ""

    # 使用 Python 调用 modelscope 下载
    python -c "
from modelscope import snapshot_download

print('开始下载，模型约 6GB，请耐心等待...')
model_dir = snapshot_download(
    'qwen/Qwen2.5-VL-3B-Instruct',
    cache_dir='${MODEL_DIR}',
    revision='master'
)
print(f'下载完成！模型路径: {model_dir}')
"

elif [ "$SOURCE" == "huggingface" ]; then
    echo ""
    echo "[1/2] 检查 huggingface-cli..."

    echo ""
    echo "[2/2] 从 HuggingFace 下载模型..."
    echo "模型地址: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"
    echo ""

    # 使用 huggingface-cli 下载
    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
        --local-dir "${MODEL_DIR}/Qwen2.5-VL-3B-Instruct" \
        --local-dir-use-symlinks False

else
    echo "错误: 未知的下载源 '${SOURCE}'"
    echo "请使用: bash scripts/download_model.sh [modelscope|huggingface]"
    exit 1
fi

echo ""
echo "============================================================"
echo "模型下载完成！"
echo "路径: ${MODEL_DIR}"
echo "============================================================"

# 列出下载的文件
echo ""
echo "下载的文件:"
ls -lh "${MODEL_DIR}"
