# coding=gb2312
import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

#分布式训练中避免多进程重复打印
def Logger(content):
    if not ddp or dist.get_rank() == 0: #获取当前进程ID
        print(content)

#余弦退火（Cosine Annealing）结合固定偏移
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb, start_step = 0):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if epoch == start_epoch and step < start_step:
            continue
        #数据加载
        X = X.to(args.device)
        Y = Y.to(args.device)
        #损失掩码
        loss_mask = loss_mask.to(args.device)

        global_step = epoch * iter_per_epoch + step  # 全局步数计算包含历史进度
        lr = get_lr(global_step, args.epochs * iter_per_epoch, args.learning_rate)
        #更新所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:   #混合精度上下文
            res = model(X)
            # 计算交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            #moe辅助损失
            loss += res.aux_loss
            #梯度积累，损失除以累积步数
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()   #损失缩放并反向传播

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  #取消梯度缩放
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            #更新参数并调整缩放器
            scaler.step(optimizer)  
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()

            #保存
            save_checkpoint(epoch, step, model, optimizer, scaler, args, lm_config)
            model.train()


def init_model(lm_config, checkpoint = None):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model = MiniMindLM(lm_config).to(args.device)
    moe_path = '_moe' if lm_config.use_moe else ''
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("load_success")
    else:
        #ckp = f'./out/pretrain_{lm_config.dim}{moe_path}_test.pth'
        ckp = f'./out/full_sft_{lm_config.dim}{moe_path}_mixed.pth'
        # checkpoint = torch.load(ckp, map_location=args.device)
        # model.load_state_dict(checkpoint['model'], strict=False)
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False) 

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer

#修改保存检查点
def save_checkpoint(epoch, step, model, optimizer, scaler, args, lm_config):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'lm_config': lm_config.__dict__,
        'args': vars(args)
    }

    moe_path = '_moe' if lm_config.use_moe else ''
    ckt_path = f'{args.out_dir}/full_sft_{lm_config.dim}{moe_path}_test_{args.extra}.pth'
    torch.save(checkpoint, ckt_path)
    Logger(f"Checkpoint saved: {ckt_path}")

#加载检查点
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
    lm_config = LMConfig(**checkpoint['lm_config'])
    args_dict = checkpoint['args']
        # 保持命令行参数优先级
    for k, v in vars(args).items():
        args_dict[k] = v 
    return checkpoint, lm_config, argparse.Namespace(**args_dict)


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")
    parser.add_argument("--resume", type=str, help = 'Path to checkpoint to resume training')
    parser.add_argument("--extra", type=str, default='', help = 'extra information')
    args = parser.parse_args()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    checkpoint = None
    if args.resume:
        checkpoint, lm_config, new_args = load_checkpoint(args.resume)
        args = new_args
        Logger(f"Resuming training from checkpoint: {args.resume}")
    else:
        lm_config = LMConfig(dim=args.dim, 
                            n_layers=args.n_layers,
                            max_seq_len=args.max_seq_len,
                            use_moe=args.use_moe)
        
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"

    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config, checkpoint=checkpoint)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )


    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    if checkpoint and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    start_epoch = checkpoint['epoch'] if checkpoint else 0
    start_step = checkpoint['step'] if checkpoint else 0

    for epoch in range(start_epoch, args.epochs):
        # 计算当前epoch的起始step
        current_start_step = start_step if epoch == start_epoch else 0

        train_epoch(epoch, wandb, start_step=current_start_step)  
        # 每个epoch结束后重置start_step
        start_step = 0
