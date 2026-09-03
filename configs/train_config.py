"""
训练配置参数说明

这个文件不会被直接执行，只是用来说明各个训练参数的含义。
实际训练时通过命令行参数传入，见 scripts/train_stage1.sh 和 scripts/train_stage2.sh
"""

# ============================================================
# Stage 1 训练配置（跨平台预训练）
# ============================================================
STAGE1_CONFIG = {
    # 模型相关
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",  # 基础模型
    "local_weight": False,                       # 是否从本地加载
    "local_weight_dir": "",                      # 本地模型路径

    # 数据相关
    "dataset_dir": "./datasets/train",           # 训练数据目录
    "train_dataset": "showui-desktop,showui-web,amex",  # 训练数据集
    "train_json": "hf_train_ori_coord,hf_train,hf_train",  # 元数据文件
    "train_ratio": "1.0,1.0,1.0",               # 采样比例（平衡采样）
    "val_dataset": "screenspot",                 # 验证数据集
    "val_json": "hf_test_full",                  # 验证元数据

    # 训练参数
    "epochs": 12,                                # 训练轮数
    "steps_per_epoch": 122,                      # 每轮步数
    "batch_size": 1,                             # 单卡 batch size
    "grad_accumulation_steps": 48,               # 梯度累积步数
    "lr": 0.0001,                                # 学习率
    "warmup_steps": 122,                         # warmup 步数

    # LoRA 配置
    "lora_r": 8,                                 # LoRA 秩
    "lora_alpha": 16,                            # LoRA alpha

    # 视觉编码配置
    "min_visual_tokens": 256,                    # 最小视觉 token 数
    "max_visual_tokens": 1280,                   # 最大视觉 token 数
    "model_max_length": 4096,                    # 最大序列长度

    # 其他
    "precision": "bf16",                         # 训练精度
    "attn_imple": "sdpa",                        # 注意力实现方式
    "ds_zero": "zero2",                          # DeepSpeed ZeRO 阶段
    "gradient_checkpointing": True,              # 梯度检查点（节省显存）
    "num_turn": 30,                              # 多轮对话轮数
    "uniform_prompt": True,                      # 使用统一 prompt
    "random_sample": True,                       # 随机采样
    "record_sample": True,                       # 记录采样（轮询）
}

# ============================================================
# Stage 2 训练配置（高分辨率专门化）
# ============================================================
STAGE2_CONFIG = {
    # 继承 Stage 1 的大部分配置，主要区别：
    "local_weight": True,                        # 从 Stage 1 加载
    "local_weight_dir": "./checkpoints/stage1/merged_model",

    # 只用 UGround 数据（多分辨率）
    "train_dataset": "uground",
    "train_json": "hf_train",
    "train_ratio": "1.0",

    # 学习率降低（微调阶段）
    "lr": 0.00005,

    # 其他参数和 Stage 1 相同
}

# ============================================================
# 评估配置
# ============================================================
EVAL_CONFIG = {
    "eval_only": True,                           # 只评估，不训练
    "lora_r": 0,                                 # 评估时不用 LoRA
    "val_dataset": "screenspot",                 # 评估数据集
    "val_json": "hf_test_full",
}
