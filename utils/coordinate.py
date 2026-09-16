"""
坐标转换工具

GUI 定位任务里，训练答案用的是什么坐标空间，评估和推理就要用对应的方式还原回原图，
两边对不上定位就整体偏了。所以把 smart_resize 和正反转换都集中放在这个文件，
训练（data/train_dataset.py）、评估（trainer/evaluator.py）、推理
（inference/inference.py）统一从这里取，不要各写一份。

支持三种坐标格式，由命令行 --coord_format 指定：

    qwen_abs   答案形如 [699, 113]，Qwen2.5-VL 缩放后图像上的绝对像素，默认用这个
    norm       答案形如 [0.52, 0.15]，归一化坐标，保留两位小数
    int1000    答案形如 [520, 150]，归一化坐标乘 1000 取整

另外注意本模块对外函数的 img_size 参数都是 (宽, 高)，和数据集 metadata 里
img_size 字段的顺序一致，调用的时候不用自己调换。

encode_* 这一组要求入参是归一化的 0-1。但各个数据集的原始标注格式并不统一
（实测 ShowUI-desktop 是归一化的，ShowUI-web 和 AMEX 是绝对像素，
ScreenSpot 是 [x,y,宽,高] 的绝对像素），所以另外提供了
normalize_point / normalize_bbox，由数据集在加载时统一转一次。
"""

import re
import math


# 支持的坐标格式
COORD_FORMATS = ("qwen_abs", "norm", "int1000")
DEFAULT_COORD_FORMAT = "qwen_abs"

# Qwen2.5-VL 视觉编码的默认参数
DEFAULT_FACTOR = 28                      # ViT patch 大小，缩放后边长必须是它的倍数
DEFAULT_MIN_PIXELS = 256 * 28 * 28       # 最小像素数
DEFAULT_MAX_PIXELS = 1280 * 28 * 28      # 最大像素数


def resolve_coord_format(args_dict=None, coord_format=None, xy_int=None):
    """
    解析最终使用的坐标格式

    --xy_int 是早期的开关，保留下来兼容老命令，这里统一转成 coord_format，
    免得两个参数各说各话。

    优先级：显式传入 > args_dict['coord_format'] > xy_int > 默认 qwen_abs

    Args:
        args_dict: 参数字典（vars(args) 或数据集的 args_dict）
        coord_format: 显式指定的坐标格式
        xy_int: 显式指定的老开关

    Returns:
        'qwen_abs' / 'norm' / 'int1000' 之一
    """
    args_dict = args_dict or {}

    fmt = coord_format if coord_format is not None else args_dict.get('coord_format')
    if fmt:
        if fmt not in COORD_FORMATS:
            raise ValueError(f"不支持的 coord_format: {fmt}，可选 {COORD_FORMATS}")
        return fmt

    use_int = xy_int if xy_int is not None else args_dict.get('xy_int', False)
    if use_int:
        return "int1000"

    return DEFAULT_COORD_FORMAT


def smart_resize(height, width, factor=DEFAULT_FACTOR,
                 min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS):
    """
    Qwen2.5-VL 的图像 resize 逻辑

    算出图片会被缩放到什么尺寸。输出边长是 factor(28) 的倍数，因为 ViT 按 28x28 切 patch；
    总像素数控制在 [min_pixels, max_pixels] 之间，也就是控制视觉 token 数量。

    Args:
        height: 原图高
        width: 原图宽
        factor: patch 大小
        min_pixels: 最小像素数
        max_pixels: 最大像素数

    Returns:
        (h_bar, w_bar) 缩放后的高和宽

    入参顺序是 (height, width)，和 PIL 的 (width, height) 反着来，这里跟 Qwen 官方实现保持一致。
    """
    if height < factor or width < factor:
        raise ValueError(f"图片太小: {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"长宽比太极端: {height}x{width}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # 超过上限等比例缩小，向下取整保证不超
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor

    # 低于下限等比例放大，向上取整保证不低于
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def _resized_size(img_size, min_pixels, max_pixels, factor=DEFAULT_FACTOR):
    """给定原图 (宽, 高)，返回缩放后的 (宽, 高)"""
    orig_width, orig_height = img_size
    new_height, new_width = smart_resize(
        orig_height, orig_width, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels
    )
    return new_width, new_height


def _clamp(value, upper):
    """把坐标夹在 [0, upper-1]，防止四舍五入之后越界"""
    return max(0, min(int(value), int(upper) - 1))


# ============================================================
# 标注入口：各数据集的原始格式 -> 归一化
# ============================================================

def normalize_point(point, img_size, scale):
    """
    把标注里的点坐标转成归一化的 0-1

    各个数据集的标注格式不统一，有的是绝对像素有的已经归一化了，
    在加载数据时统一转一次，下游 encode_point 就只需要处理归一化输入。

    Args:
        point: [x, y]
        img_size: (原图宽, 原图高)
        scale: 'absolute' 表示入参是绝对像素，'norm' 表示已经是 0-1

    Returns:
        [x, y]，归一化坐标
    """
    x, y = float(point[0]), float(point[1])
    if scale == "norm":
        return [x, y]

    width, height = float(img_size[0]), float(img_size[1])
    return [x / width, y / height]


def normalize_bbox(bbox, img_size, scale, layout="xyxy"):
    """
    把标注里的边界框统一成归一化的 [x1, y1, x2, y2]

    两件事一起做：布局统一（有的数据集是 [x,y,宽,高]）和数值归一化。
    顺序不能反 —— 先在原来的尺度上把 [x,y,宽,高] 拼成 [x1,y1,x2,y2]，再归一化。

    Args:
        bbox: 四个数
        img_size: (原图宽, 原图高)
        scale: 'absolute' / 'norm'
        layout: 'xyxy' 表示入参是 [x1,y1,x2,y2]，'xywh' 表示 [x,y,宽,高]

    Returns:
        [x1, y1, x2, y2]，归一化坐标
    """
    b0, b1, b2, b3 = (float(v) for v in bbox)

    if layout == "xywh":
        b2, b3 = b0 + b2, b1 + b3

    if scale == "norm":
        return [b0, b1, b2, b3]

    width, height = float(img_size[0]), float(img_size[1])
    return [b0 / width, b1 / height, b2 / width, b3 / height]


# ============================================================
# 正向：归一化标注 -> 训练答案
# ============================================================

def encode_point(point, img_size, coord_format=DEFAULT_COORD_FORMAT,
                 min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
                 factor=DEFAULT_FACTOR):
    """
    把归一化的点坐标转成训练答案

    Args:
        point: [x, y]，归一化坐标，范围 0-1
        img_size: (原图宽, 原图高)
        coord_format: 目标坐标格式
        min_pixels/max_pixels/factor: 只有 qwen_abs 用得上

    Returns:
        [x, y]，格式由 coord_format 决定
    """
    x, y = point[0], point[1]

    if coord_format == "qwen_abs":
        new_width, new_height = _resized_size(img_size, min_pixels, max_pixels, factor)
        return [_clamp(round(x * new_width), new_width),
                _clamp(round(y * new_height), new_height)]

    if coord_format == "int1000":
        return [int(x * 1000), int(y * 1000)]

    return [round(float(x), 2), round(float(y), 2)]


def encode_bbox(bbox, img_size, coord_format=DEFAULT_COORD_FORMAT,
                min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
                factor=DEFAULT_FACTOR):
    """
    把归一化的边界框转成训练答案

    Args:
        bbox: [x1, y1, x2, y2]，归一化坐标，范围 0-1
        img_size: (原图宽, 原图高)
        coord_format: 目标坐标格式

    Returns:
        [x1, y1, x2, y2]，格式由 coord_format 决定
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    if coord_format == "qwen_abs":
        new_width, new_height = _resized_size(img_size, min_pixels, max_pixels, factor)
        return [_clamp(round(x1 * new_width), new_width),
                _clamp(round(y1 * new_height), new_height),
                _clamp(round(x2 * new_width), new_width),
                _clamp(round(y2 * new_height), new_height)]

    if coord_format == "int1000":
        return [int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)]

    return [round(float(v), 2) for v in (x1, y1, x2, y2)]


# ============================================================
# 反向：模型输出 -> 原图像素
# ============================================================

def decode_point(point, img_size, coord_format=DEFAULT_COORD_FORMAT,
                 min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
                 factor=DEFAULT_FACTOR):
    """
    把模型输出的坐标还原成原图的绝对像素坐标

    encode_point 的逆运算，两个必须成对用，用错定位就整体偏了。

    Args:
        point: 模型输出的 [x, y]
        img_size: (原图宽, 原图高)
        coord_format: 模型输出所在的坐标空间

    Returns:
        [x, y]，原图绝对像素坐标
    """
    orig_width, orig_height = img_size
    x, y = point[0], point[1]

    if coord_format == "qwen_abs":
        new_width, new_height = _resized_size(img_size, min_pixels, max_pixels, factor)
        x_orig = round(x * orig_width / new_width)
        y_orig = round(y * orig_height / new_height)
    elif coord_format == "int1000":
        x_orig = round(x / 1000.0 * orig_width)
        y_orig = round(y / 1000.0 * orig_height)
    else:
        x_orig = round(x * orig_width)
        y_orig = round(y * orig_height)

    return [_clamp(x_orig, orig_width), _clamp(y_orig, orig_height)]


def decode_bbox(bbox, img_size, coord_format=DEFAULT_COORD_FORMAT,
                min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
                factor=DEFAULT_FACTOR):
    """
    把模型输出的边界框还原成原图的绝对像素坐标

    Args:
        bbox: 模型输出的 [x1, y1, x2, y2]
        img_size: (原图宽, 原图高)
        coord_format: 模型输出所在的坐标空间

    Returns:
        [x1, y1, x2, y2]，原图绝对像素坐标
    """
    x1, y1 = decode_point([bbox[0], bbox[1]], img_size, coord_format,
                          min_pixels, max_pixels, factor)
    x2, y2 = decode_point([bbox[2], bbox[3]], img_size, coord_format,
                          min_pixels, max_pixels, factor)
    return [x1, y1, x2, y2]


# ============================================================
# 文本解析
# ============================================================

# 匹配 [x, y] 或 [x1, y1, x2, y2]，允许负号和小数
_COORD_PATTERN = re.compile(
    r'\[\s*-?[\d\.]+\s*,\s*-?[\d\.]+\s*(?:,\s*-?[\d\.]+\s*,\s*-?[\d\.]+\s*)?\]'
)


def extract_coordinates(text):
    """
    从模型输出的文本里提取坐标片段

    模型有时候会在坐标前后带上解释文字，直接 ast.literal_eval 会失败，
    所以先用正则把 [x, y] 或 [x1, y1, x2, y2] 抠出来。

    Args:
        text: 模型输出的原始文本

    Returns:
        坐标字符串（如 "[523, 147]"），没匹配到返回 None
    """
    if not text:
        return None
    match = _COORD_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def parse_predicted_point(text):
    """
    把模型输出的文本解析成一个点

    评估和推理都用这个，保证两边解析行为一致。正则提取不到就退回整段文本试一次，
    拿到 4 个值的边界框就取中心点。

    Args:
        text: 模型输出的原始文本

    Returns:
        [x, y]，解析失败返回 None
    """
    import ast

    if text is None:
        return None

    coord_str = extract_coordinates(text)
    if coord_str is None:
        coord_str = text.strip()

    try:
        value = ast.literal_eval(coord_str)
    except Exception:
        return None

    if not isinstance(value, (list, tuple)):
        return None

    if len(value) == 4:
        return [(value[0] + value[2]) / 2, (value[1] + value[3]) / 2]

    if len(value) == 2:
        return [value[0], value[1]]

    return None
