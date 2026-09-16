"""
训练器

实现主要的训练循环逻辑
"""

import time
import wandb

from data import AverageMeter, ProgressMeter, dict_to_cuda


def train_one_epoch(train_loader, model, epoch, scheduler, writer, train_iter, args):
    """
    训练一个 epoch

    Args:
        train_loader: 训练数据加载器
        model: DeepSpeed 封装的模型
        epoch: 当前 epoch 数
        scheduler: 学习率调度器
        writer: TensorBoard writer
        train_iter: 数据迭代器（用于断点续训）
        args: 训练参数

    Returns:
        train_iter: 更新后的数据迭代器
        global_step: 当前全局步数
    """
    # 初始化统计器
    batch_time = AverageMeter("Batch(s)", ":6.3f")
    iter_time = AverageMeter("Iter(s)", ":6.3f")
    epoch_time = AverageMeter("Epoch(h)", ":6.3f")
    remain_time = AverageMeter("剩余(h)", ":6.3f")
    data_time = AverageMeter("Data(s)", ":6.3f")
    seq_len = AverageMeter("SeqLen", ":6.0f")
    losses = AverageMeter("Loss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, iter_time, epoch_time, remain_time, losses, seq_len],
        prefix=f"Epoch [{epoch}]",
    )

    # 切换到训练模式
    model.train()
    end = time.time()

    # 训练循环
    for local_step in range(args.steps_per_epoch):
        global_step = local_step + epoch * args.steps_per_epoch

        # 梯度累积
        for _ in range(args.grad_accumulation_steps):
            # 获取下一个 batch
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)

            # 数据转到 GPU
            input_dict = dict_to_cuda(input_dict)

            # 处理图像数据类型
            if input_dict["pixel_values"] is not None:
                if args.precision == "fp16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].half()
                elif args.precision == "bf16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
                else:
                    input_dict["pixel_values"] = input_dict["pixel_values"].float()

            # 构建前向传播参数
            # 别开 output_hidden_states，每层的隐藏状态都留在显存里，用不上还占地方
            forward_dict = {
                "pixel_values": input_dict["pixel_values"],
                "input_ids": input_dict["input_ids"],
                "labels": input_dict["labels"],
            }

            # attention_mask 要传，Qwen2.5-VL 算 RoPE 位置索引也用它，batch > 1 时不传就错了
            if input_dict.get("attention_mask") is not None:
                forward_dict["attention_mask"] = input_dict["attention_mask"]

            # 添加图像尺寸信息
            if input_dict.get("image_sizes") is not None:
                forward_dict["image_grid_thw"] = input_dict["image_sizes"]

            # 前向传播
            output_dict = model.forward(**forward_dict)
            loss = output_dict["loss"]

            # 更新统计
            losses.update(loss.item(), input_dict["input_ids"].size(0))

            # 反向传播（DeepSpeed 会自动处理梯度累积）
            model.backward(loss)
            model.step()

        # 计算耗时
        batch_sec = time.time() - end
        iter_sec = batch_sec / args.grad_accumulation_steps
        batch_time.update(batch_sec)
        iter_time.update(iter_sec)
        epoch_time.update(iter_sec * args.steps_per_epoch / 3600)
        remain_time.update((iter_sec * (args.steps_per_epoch - local_step - 1)) / 3600)
        end = time.time()

        # 更新序列长度统计
        seq_len.update(input_dict["input_ids"].size(1))

        # 定期打印和记录
        if global_step % args.print_freq == 0:
            # 分布式训练时同步统计值
            if args.distributed:
                batch_time.all_reduce()
                iter_time.all_reduce()
                epoch_time.all_reduce()
                remain_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                seq_len.all_reduce()

            # 只在主进程打印和记录
            if args.global_rank == 0:
                progress.display(global_step + 1)

                if not args.debug:
                    # TensorBoard
                    writer.add_scalar("train/loss", losses.avg, global_step)
                    writer.add_scalar("metrics/batch_time", batch_time.avg, global_step)
                    writer.add_scalar("metrics/iter_time", iter_time.avg, global_step)

                    # wandb（如果配置了的话）
                    if getattr(args, 'use_wandb', False):
                        wandb.log({
                            "epoch": epoch,
                            "train_loss": losses.avg,
                            "batch_time": batch_time.avg,
                            "iter_time": iter_time.avg,
                            "seq_len": seq_len.avg,
                        }, step=global_step)

            # 重置统计器
            batch_time.reset()
            iter_time.reset()
            epoch_time.reset()
            remain_time.reset()
            data_time.reset()
            losses.reset()
            seq_len.reset()

        # 记录学习率
        if global_step != 0 and args.global_rank == 0 and not args.debug:
            curr_lr = scheduler.get_last_lr()
            writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter, global_step
