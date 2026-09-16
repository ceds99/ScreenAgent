"""
ScreenAgent 主训练脚本

使用方法:
    deepspeed trainer/train.py --参数...

详细参数见 --help 或 configs/train_config.py
"""

import argparse
import glob
import os
import shutil
import sys
import json
import time
from functools import partial
from datetime import datetime

import torch
import torch.distributed as dist
import deepspeed
import wandb
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import HybridDataset, collate_fn
from models import find_lora_target_modules
from utils import save_args_to_json, ensure_dir, get_timestamp, resolve_coord_format
from trainer.trainer import train_one_epoch
from trainer.evaluator import evaluate_screenspot, evaluate_training_data


def init_distributed():
    """初始化分布式训练环境"""
    if 'WORLD_SIZE' in os.environ:
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12875')
        print(f"分布式训练: WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ.get('RANK', 0)}")


def broadcast_timestamp(src=0, local_rank=0):
    """广播时间戳到所有进程（确保日志目录一致）"""
    if dist.get_rank() == src:
        timestamp = torch.tensor([datetime.now().timestamp()], dtype=torch.float64).to(f'cuda:{local_rank}')
    else:
        timestamp = torch.zeros(1, dtype=torch.float64).to(f'cuda:{local_rank}')

    dist.broadcast(timestamp, src=src)
    return datetime.fromtimestamp(timestamp.item()).strftime('%Y-%m-%d_%H-%M-%S')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ScreenAgent 训练脚本")

    # ========== 环境相关 ==========
    parser.add_argument("--wandb_key", type=str, default=None, help="wandb API key")
    parser.add_argument("--local_rank", type=int, default=0, help="本地 GPU 编号")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--ds_zero", type=str, default="zero2", choices=["zero1", "zero2", "zero3"])
    parser.add_argument("--attn_imple", type=str, default="sdpa", choices=["eager", "flash_attention_2", "sdpa"])

    # ========== 模型相关 ==========
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--local_weight", action="store_true", help="从本地加载模型")
    parser.add_argument("--local_weight_dir", type=str, default="./checkpoints/base_model")
    # 2560 是按实测定的：num_turn=30 时序列长度在 1300-1930 之间，2560 有 33% 余量。
    # 这个值同时是显存的闸 —— 激活值随序列长度增长，24GB 的卡上放到 4096 会被
    # 长尾序列撑爆。超过这个长度的样本不会被丢掉，多轮构建时会自动少问几轮
    parser.add_argument("--model_max_length", type=int, default=2560)
    parser.add_argument("--min_visual_tokens", type=int, default=256)
    parser.add_argument("--max_visual_tokens", type=int, default=1280)

    # ========== LoRA 相关 ==========
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--tune_visual_encoder", action="store_true", help="是否微调视觉编码器")

    # ========== 数据相关 ==========
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--train_dataset", type=str, default="showui-web")
    parser.add_argument("--train_json", type=str, default="hf_train")
    parser.add_argument("--train_ratio", type=str, default="1")
    parser.add_argument("--val_dataset", type=str, default="screenspot")
    parser.add_argument("--val_json", type=str, default="hf_test_full")
    parser.add_argument("--val_ratio", type=str, default="1")
    parser.add_argument("--num_turn", type=int, default=1, help="多轮对话轮数")
    parser.add_argument("--uniform_prompt", action="store_true")
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--record_sample", action="store_true")

    # ========== 训练相关 ==========
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--warmup_type", type=str, default="linear")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)

    # ========== 日志和保存 ==========
    parser.add_argument("--log_base_dir", type=str, default="./logs")
    parser.add_argument("--exp_id", type=str, default="debug")
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    # 评估时的生成长度上限。答案就是 [x, y] 这种十几个 token 的东西，
    # 给太松的话模型没收敛时会一路生成到上限，评估时间能差几十倍
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="评估生成的 token 上限，坐标答案约 12 个 token")
    parser.add_argument("--debug", action="store_true", help="调试模式，不保存模型和日志")

    # ========== 坐标格式 ==========
    # 训练答案和评估解码用同一个值，具体转换见 utils/coordinate.py
    #   qwen_abs  Qwen2.5-VL 缩放后图像上的绝对像素，默认
    #   norm      归一化 0-1 浮点
    #   int1000   归一化乘 1000 的整数
    parser.add_argument("--coord_format", type=str, default=None,
                        choices=["qwen_abs", "norm", "int1000"],
                        help="坐标格式，不指定则用 qwen_abs")
    parser.add_argument("--xy_int", action="store_true",
                        help="[兼容旧参数] 等价于 --coord_format int1000")

    # 标注坐标的格式默认是自动判别的（见 data/data_utils.py 的
    # detect_annotation_format）。数据集太小或者样本都是歧义的时候判不出来，
    # 那种情况下用这两个参数手动指定
    parser.add_argument("--ann_scale", type=str, default=None,
                        choices=["absolute", "norm"],
                        help="标注是绝对像素还是归一化，默认自动判别")
    parser.add_argument("--ann_bbox_layout", type=str, default=None,
                        choices=["xyxy", "xywh"],
                        help="标注的 bbox 是 [x1,y1,x2,y2] 还是 [x,y,宽,高]，默认自动判别")
    parser.add_argument("--text2point", type=float, default=1)
    parser.add_argument("--text2bbox", type=float, default=0)
    parser.add_argument("--crop_min", type=float, default=1)
    parser.add_argument("--crop_max", type=float, default=1)

    return parser.parse_args()


def main():
    """主函数"""
    print("=" * 60)
    print("ScreenAgent - GUI Grounding 训练")
    print("=" * 60)

    # 初始化分布式环境
    init_distributed()
    args = parse_args()

    # 设置环境变量
    args.global_rank = int(os.environ.get('RANK', 0))
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1

    # 解析出最终的 coord_format（兼容 --xy_int），数据集和评估器都读这一个值，
    # 顺便会写进 args.json，后面合并权重和评估的时候可以查
    args.coord_format = resolve_coord_format(vars(args))
    print(f"[坐标格式] {args.coord_format}")

    # 设置 attention 实现
    # 只在 eager 模式下关闭加速后端；sdpa 模式选它就是为了让它自动挑选加速路径，
    # 把这两个开关也关掉会让 sdpa 退化成最慢的实现，自相矛盾
    if args.attn_imple == "eager":
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    # 获取时间戳（分布式训练时同步）
    timestamp = get_timestamp()
    if args.distributed:
        deepspeed.init_distributed(dist_backend="nccl", rank=args.global_rank, world_size=args.world_size)
        timestamp = broadcast_timestamp(0, args.local_rank)

    # 设置日志目录
    args.log_dir = os.path.join(args.log_base_dir, args.exp_id, timestamp)
    args.tmp_dir = os.path.join(args.log_dir, "tmp")

    # 初始化 wandb（可选，需要配置 wandb_key）
    args.use_wandb = args.wandb_key is not None
    if args.use_wandb:
        wandb.login(key=args.wandb_key)

    # 创建目录和 writer（只在主进程）
    writer = None
    if args.global_rank == 0:
        ensure_dir(args.log_dir)
        ensure_dir(args.tmp_dir)
        save_args_to_json(args, os.path.join(args.log_dir, "args.json"))

        if not args.debug:
            writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
            if args.use_wandb:
                wandb.init(
                    project="ScreenAgent",
                    group=args.exp_id,
                    name=f'{args.exp_id}_{timestamp}',
                    dir=args.log_dir,
                    config=vars(args)
                )

    print(f"[实验] {args.exp_id}")
    print(f"[日志目录] {args.log_dir}")

    # ========== 加载模型和处理器 ==========
    print("\n[1/4] 加载模型...")

    # 确定模型路径
    if args.local_weight:
        model_path = args.local_weight_dir
        # 处理 ModelScope 下载的嵌套路径
        # 比如 checkpoints/base_model/qwen/Qwen2___5-VL-3B-Instruct
        if os.path.isdir(model_path):
            # 查找包含 config.json 的子目录
            for root, dirs, files in os.walk(model_path):
                if 'config.json' in files:
                    model_path = root
                    break
    else:
        model_path = args.model_id

    print(f"  模型路径: {model_path}")

    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
    )
    processor.tokenizer.model_max_length = args.model_max_length

    # 设置 chat template（Qwen2.5-VL 格式）
    CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    processor.chat_template = CHAT_TEMPLATE
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.chat_template = CHAT_TEMPLATE

    # 确定数据类型
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.half if args.precision == "fp16" else torch.float32)

    # 加载模型
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_imple,
        device_map=f"cuda:{args.local_rank}",
    )
    model.config.use_cache = False

    # ========== 配置 LoRA ==========
    if args.eval_only:
        print("评估模式，跳过 LoRA 配置")
        args.lora_r = 0
    elif args.lora_r > 0:
        print(f"\n[2/4] 配置 LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

        # 找到要训练的模块（lm_head 始终不参与 LoRA，见 find_lora_target_modules 的默认排除逻辑）
        exclude_modules = ["visual"] if not args.tune_visual_encoder else []
        target_modules = find_lora_target_modules(model, exclude_keywords=exclude_modules)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        if args.global_rank == 0:
            model.print_trainable_parameters()

    # 冻结视觉编码器
    if not args.tune_visual_encoder:
        print("冻结视觉编码器")
        if args.lora_r > 0:
            for p in model.base_model.model.visual.parameters():
                p.requires_grad = False
        else:
            for p in model.visual.parameters():
                p.requires_grad = False

    # 梯度检查点
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # ========== 创建数据集 ==========
    print("\n[3/4] 创建数据集...")

    # 计算每 epoch 的样本数
    args.samples_per_epoch = (
        args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * args.world_size
    )

    # 纯评估不用建训练集，省得只跑个评估还得先把训练数据下全
    train_dataset = None
    if not args.eval_only:
        train_dataset = HybridDataset(processor, inference=False, args=args)

    val_dataset = HybridDataset(processor, inference=True, args=args)

    # ========== 初始化 DeepSpeed ==========
    print("\n[4/4] 初始化 DeepSpeed...")

    # 构建 DeepSpeed 配置
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps,
                "warmup_type": args.warmup_type,
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
    }

    # 加载 ZeRO 配置
    zero_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "configs",
        f"deepspeed_{args.ds_zero}.json"
    )
    if os.path.exists(zero_config_path):
        with open(zero_config_path) as f:
            ds_config.update(json.load(f))

    # 获取可训练参数
    params_to_train = [p for p in model.parameters() if p.requires_grad]

    if args.eval_only:
        # 评估模式
        for p in model.parameters():
            p.requires_grad = False
        model_engine = model
    else:
        # 训练模式
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=params_to_train,
            training_data=train_dataset,
            collate_fn=partial(collate_fn, processor=processor),
            config=ds_config,
        )

    # 创建验证数据加载器
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False, drop_last=False
    ) if args.distributed else None

    # 评估这里固定 batch_size=1。collate_fn 是右 padding，批量 generate 要的是左 padding，
    # 开大了生成结果会错位
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=val_sampler,
        collate_fn=partial(collate_fn, processor=processor),
    )

    # ========== 仅评估模式 ==========
    if args.eval_only:
        print("\n开始评估...")
        model_engine = model_engine.to(f'cuda:{args.local_rank}')
        evaluate_screenspot(val_loader, model_engine, processor, 0, 0, writer, args)
        return

    # ========== 训练循环 ==========
    print("\n开始训练...")
    train_iter = iter(train_loader)
    best_score = 0

    for epoch in range(args.epochs):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 40}")

        # 训练一个 epoch
        train_iter, global_step = train_one_epoch(
            train_loader, model_engine, epoch, scheduler, writer, train_iter, args
        )

        # 评估
        if not args.no_eval:
            torch.cuda.empty_cache()
            score = evaluate_screenspot(
                val_loader, model_engine, processor, epoch, global_step, writer, args
            )

            # 用 >= 而不是 >：当前评估指标很粗（1272 个样本、离散坐标匹配），
            # 后续 epoch 打平前一个最佳分数是常见情况；用严格大于会导致打平的
            # epoch 全部不落盘，只保留最早、训练最少的那一份权重。
            is_best = score >= best_score
            best_score = max(score, best_score)
            torch.cuda.empty_cache()
        else:
            is_best = True

        # 保存检查点
        # debug 模式不落盘（train_debug.sh 的设计意图就是"只跑几个 step，不保存模型"）
        if (args.no_eval or is_best) and not args.debug:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.global_rank == 0:
                ensure_dir(save_dir)
                # 每个 exp 只保留最新一份 checkpoint，避免每次 is_best 都新增一个
                # ~10GB 的 global_stepN 目录、把磁盘撑爆
                for old_ckpt in glob.glob(os.path.join(save_dir, "global_step*")):
                    shutil.rmtree(old_ckpt, ignore_errors=True)
                torch.save(
                    {"epoch": epoch, "best_score": best_score},
                    os.path.join(save_dir, f"meta_epoch{epoch}_score{best_score:.4f}.pth"),
                )

            if args.distributed:
                dist.barrier()

            try:
                model_engine.save_checkpoint(save_dir)
                print(f"检查点已保存: {save_dir}")
            except Exception as e:
                print(f"保存检查点失败: {e}")

    # 清理
    if args.global_rank == 0 and not args.debug:
        if args.use_wandb:
            wandb.finish()
        if writer:
            writer.close()

    print("\n训练完成！")


if __name__ == "__main__":
    main()
