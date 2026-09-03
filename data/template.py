"""
数据模板和格式转换

这个文件定义了训练和评估时的 prompt 模板，以及数据格式转换函数。
主要包括：
- 训练时的多样化 prompt 模板
- 评估时的统一 prompt 模板
- 答案拼接函数
"""

import random
import torch

from .data_utils import IGNORE_INDEX


# ============================================================
# 系统提示词模板（训练时随机选择，增加多样性）
# ============================================================

# text2point 任务的系统提示词：给定文字描述，预测点击位置
SYSTEM_PROMPTS_TEXT2POINT = [
    "In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions.",
    "Based on the screenshot of the page, I give a text description and you give its corresponding location.",
    "In the image above, I will give a series of descriptions of the elements to be clicked. Please predict where you want to click.",
    "I will give textual descriptions of certain elements in the screenshot. Please predict the location of the corresponding element.",
    "Please identify the coordinates of the webpage elements I describe based on the provided screenshot.",
    "Given a screenshot, I will describe specific elements; your task is to predict their locations.",
    "Using the image of this webpage, can you determine the coordinates of the elements I describe?",
    "In this webpage capture, I will describe certain elements. Please locate them for me.",
    "I'll provide textual descriptions of elements in this webpage screenshot. Can you find their coordinates?",
    "From the given webpage screenshot, I need you to identify the locations of described elements.",
    "Based on this screenshot, I'll describe some elements. Please pinpoint their exact locations.",
    "For the elements I describe in this page capture, can you predict their positions?",
    "I will describe elements from a webpage screenshot; your role is to locate them.",
    "Using the attached screenshot of a webpage, please find the coordinates of described elements.",
    "From the image of this webpage, I will describe elements for you to locate.",
]

# point2text 任务的系统提示词：给定坐标位置，预测元素内容
SYSTEM_PROMPTS_POINT2TEXT = [
    "Based on the screenshot of the web page, I give you the location to click on and you predict the text content of the corresponding element.",
    "In the image above, I give a series of coordinates and ask you to describe the corresponding elements.",
    "On this page, I will give you a series of coordinates and ask you to predict the text of the clickable element that corresponds to these coordinates.",
    "Given a webpage screenshot, I provide coordinates; predict the text content of the elements at these locations.",
    "In this screenshot, I'll give coordinates and ask you to describe the text of the elements there.",
]

# 坐标格式说明
COORD_POINT_DESC = "The coordinate represents a clickable location [x, y] for an element."
COORD_BBOX_DESC = "The coordinates represent a bounding box [x1, y1, x2, y2] for an element."
COORD_POINT_INT_DESC = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 1 to 1000."

# 评估用的统一提示词（和训练保持一致）
SCREENSPOT_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location."


# ============================================================
# 格式转换函数
# ============================================================

def build_grounding_prompt(element_name, image_dict, sample_type=0,
                           shuffle_prompt=True, xy_int=False, uniform_prompt=False):
    """
    构建 grounding 任务的 prompt

    这个函数用于训练时构建输入数据。根据不同的任务类型生成对应的 prompt。

    Args:
        element_name: 元素名称或坐标（取决于任务类型）
        image_dict: 图像配置字典，包含 type, min_pixels, max_pixels
        sample_type: 任务类型
            0 - text2point: 给文字，预测点击坐标
            1 - text2bbox: 给文字，预测边界框
            2 - point2text: 给坐标，预测文字
            3 - bbox2text: 给边界框，预测文字
        shuffle_prompt: 是否随机打乱图片和文字的顺序
        xy_int: 坐标是否使用整数（0-1000）
        uniform_prompt: 是否使用统一的 prompt（评估时用）

    Returns:
        构建好的消息列表，可以直接传给 tokenizer
    """
    user_content = []

    # 选择系统提示词
    if sample_type in [0, 1]:
        system_prompt = random.choice(SYSTEM_PROMPTS_TEXT2POINT)
    else:
        system_prompt = random.choice(SYSTEM_PROMPTS_POINT2TEXT)

    # 评估时使用统一的提示词
    if uniform_prompt:
        system_prompt = SYSTEM_PROMPTS_TEXT2POINT[1]  # "Based on the screenshot..."

    # 添加坐标格式说明
    if sample_type in [0, 2]:
        # 点坐标
        coord_desc = COORD_POINT_INT_DESC if xy_int else COORD_POINT_DESC
    else:
        # 边界框坐标
        coord_desc = COORD_BBOX_DESC

    system_prompt = system_prompt + ' ' + coord_desc

    # 构建用户消息（随机调整图片和文字的顺序，增加鲁棒性）
    if shuffle_prompt:
        order = random.choice(['img_first', 'text_first', 'text_last'])
    else:
        order = 'text_first'

    if order == 'img_first':
        user_content.append(image_dict)
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append({"type": "text", "text": element_name})
    elif order == 'text_first':
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append(image_dict)
        user_content.append({"type": "text", "text": element_name})
    else:
        user_content.append({"type": "text", "text": system_prompt})
        user_content.append({"type": "text", "text": element_name})
        user_content.append(image_dict)

    return [{"role": "user", "content": user_content}]


def build_eval_prompt(element_name, image_dict, xy_int=False):
    """
    构建评估时的 prompt

    评估时使用统一的格式，方便比较不同模型的性能。

    Args:
        element_name: 要定位的元素描述
        image_dict: 图像配置字典
        xy_int: 坐标是否使用整数

    Returns:
        构建好的消息列表
    """
    user_content = []

    # 使用固定的系统提示词
    if xy_int:
        system_prompt = SCREENSPOT_SYSTEM + ' ' + COORD_POINT_INT_DESC
    else:
        system_prompt = SCREENSPOT_SYSTEM + ' ' + COORD_POINT_DESC

    # 固定顺序：系统提示 -> 图片 -> 元素描述
    user_content.append({"type": "text", "text": system_prompt})
    user_content.append(image_dict)
    user_content.append({"type": "text", "text": element_name})

    return [{"role": "user", "content": user_content}]


def add_answer_to_batch(batch, answer, processor, add_eos=True):
    """
    把答案拼接到输入后面，并生成对应的 labels

    训练时，我们只计算答案部分的 loss，所以需要把输入部分的 labels 设为 -100。

    Args:
        batch: processor 处理后的 batch 数据
        answer: 答案字符串或坐标列表
        processor: tokenizer/processor
        add_eos: 是否在答案末尾添加 eos token

    Returns:
        处理后的 batch 和答案字符串
    """
    prompt_input_ids = batch['input_ids']

    # 把答案转成字符串
    if not isinstance(answer, str):
        answer = str(answer)

    # 添加结束符
    if add_eos:
        answer = answer + processor.tokenizer.eos_token

    # 对答案进行 tokenize
    answer_input_ids = processor.tokenizer(
        answer, add_special_tokens=False, return_tensors='pt'
    )['input_ids']

    # 拼接 input_ids
    input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)

    # 生成 labels：输入部分用 -100 标记（不计算 loss），答案部分保留原值
    labels = torch.cat([
        torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])).unsqueeze(0),
        answer_input_ids,
    ], dim=1)

    # 更新 batch
    batch['input_ids'] = input_ids
    batch['labels'] = labels

    # 删除 attention_mask（后面会重新生成）
    if 'attention_mask' in batch:
        del batch['attention_mask']

    return batch, answer


def add_multiturn_answer(batch, answer, processor,
                         append_elements=None, append_answers=None, add_eos=True):
    """
    多轮对话的答案拼接

    支持多轮 grounding 任务，每轮都有一个元素描述和对应的答案。

    Args:
        batch: processor 处理后的 batch 数据
        answer: 第一轮的答案
        processor: tokenizer/processor
        append_elements: 后续轮次的元素描述列表
        append_answers: 后续轮次的答案列表
        add_eos: 是否添加 eos token

    Returns:
        处理后的 batch 和完整的答案字符串
    """
    # 先处理第一轮
    prompt_input_ids = batch['input_ids']

    if not isinstance(answer, str):
        answer = str(answer)
    if add_eos:
        answer = answer + processor.tokenizer.eos_token

    answer_input_ids = processor.tokenizer(
        answer, add_special_tokens=False, return_tensors='pt'
    )['input_ids']

    input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
    labels = torch.cat([
        torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])).unsqueeze(0),
        answer_input_ids,
    ], dim=1)

    # 处理后续轮次
    full_answer = answer
    if append_elements and append_answers:
        for element_i, answer_i in zip(append_elements, append_answers):
            if not isinstance(answer_i, str):
                answer_i = str(answer_i)
            if not isinstance(element_i, str):
                element_i = str(element_i)
            if add_eos:
                answer_i = answer_i + processor.tokenizer.eos_token

            # 构建用户消息
            source_i = [{"role": "user", "content": element_i}]
            element_prompt = '\n' + processor.tokenizer.apply_chat_template(
                source_i,
                chat_template=processor.chat_template,
                tokenize=False,
                add_generation_prompt=True
            )

            full_answer += element_prompt + answer_i

            # tokenize 并拼接
            element_input_ids = processor.tokenizer(
                element_prompt, add_special_tokens=False, return_tensors='pt'
            )['input_ids']
            answer_input_ids = processor.tokenizer(
                answer_i, add_special_tokens=False, return_tensors='pt'
            )['input_ids']

            input_ids = torch.cat([input_ids, element_input_ids, answer_input_ids], dim=1)
            labels = torch.cat([
                labels,
                torch.tensor([IGNORE_INDEX] * len(element_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ], dim=1)

    # 更新 batch
    batch['input_ids'] = input_ids
    batch['labels'] = labels
    if 'attention_mask' in batch:
        del batch['attention_mask']

    return batch, full_answer
