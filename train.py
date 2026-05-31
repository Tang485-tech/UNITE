from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.build import build_criterion, build_dataloader, build_model
from src.engine.trainer import train
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.experiment import resolve_output_dir
from src.utils.seed import set_seed


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNITE on FF++ C23.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    # torchrun-injected env vars
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=60))
        torch.cuda.set_device(local_rank)

    args = parse_args()
    config = load_config(args.config)
    output_dir = resolve_output_dir(config, args.run_name)
    if rank == 0:
        print(f"Output directory: {output_dir}")

    # user-specified device takes priority; DDP overrides with local_rank
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    set_seed(int(config.seed) + rank)

    distributed = world_size > 1
    train_loader = build_dataloader(config, "train", distributed=distributed)
    val_loader = build_dataloader(config, "val", distributed=False)

    model = build_model(config).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = build_criterion(config).to(device)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.scheduler_step_size,
        gamma=config.train.scheduler_gamma,
    )

    # scale gradient accumulation so global effective batch size stays unchanged
    accumulation = max(1, int(config.train.gradient_accumulation_steps) // world_size)
    config.train.gradient_accumulation_steps = accumulation

    start_epoch = 0
    global_step = 0
    writer_step = 0
    best_metric = float("-inf")
    resume_stage = "epoch_complete"
    if args.resume:
        checkpoint_model = model.module if distributed else model
        checkpoint = load_checkpoint(args.resume, checkpoint_model, optimizer, scheduler, criterion, map_location=device)
        start_epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", 0)))
        global_step = int(checkpoint.get("global_step", 0))
        writer_step = int(checkpoint.get("writer_step", global_step))
        best_metric = float(checkpoint.get("best_metric", best_metric))
        checkpoint_stage = checkpoint.get("checkpoint_stage", "legacy")
        resume_stage = checkpoint_stage
        completed_epoch = checkpoint.get("completed_epoch", "unknown")
        if _is_distributed():
            print(
                f"Rank {rank}: resumed from {args.resume} "
                f"stage={checkpoint_stage}, completed_epoch={completed_epoch}, "
                f"start_epoch={start_epoch}, step={global_step}, writer_step={writer_step}"
            )
        else:
            print(
                f"Resumed from {args.resume} stage={checkpoint_stage}, "
                f"completed_epoch={completed_epoch}, start_epoch={start_epoch}, "
                f"step={global_step}, writer_step={writer_step}"
            )

    train(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        start_epoch=start_epoch,
        global_step=global_step,
        writer_step=writer_step,
        best_metric=best_metric,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        resume_stage=resume_stage,
    )

    if _is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
