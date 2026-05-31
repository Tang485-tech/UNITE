from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.engine.evaluator import evaluate
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import CSVLogger


def _is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config,
    device: torch.device,
    start_epoch: int = 0,
    global_step: int = 0,
    writer_step: int = 0,
    best_metric: float = -math.inf,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
    resume_stage: str = "epoch_complete",
) -> None:
    main = _is_main_process()
    output_dir = Path(config.paths.output_dir)
    if main:
        output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard") if main else None  # type: ignore[assignment]
    csv_logger = CSVLogger(output_dir / "metrics.csv") if main else None

    scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device.type == "cuda")
    accumulation_steps = max(1, int(config.train.gradient_accumulation_steps))
    grad_clip = float(config.train.get("grad_clip_max_norm", 1.0))
    warmup_steps = int(config.train.get("warmup_steps", 0))
    log_grad_norm = bool(config.train.get("log_grad_norm", False))
    log_timing = bool(config.train.get("log_timing", False))
    timing_log_every = max(1, int(config.train.get("timing_log_every", 20)))
    base_lr = float(config.train.lr)
    best_metric_name = config.metrics.best_metric

    if config.train.get("debug_anomaly", False) and main:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled.")

    model.to(device)
    criterion.to(device)

    try:
        for epoch in range(start_epoch, int(config.train.epochs)):
            skip_train = epoch == start_epoch and resume_stage == "train_complete"
            model.train()
            criterion.train()
            optimizer.zero_grad(set_to_none=True)
            running: dict[str, float] = {}

            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            if skip_train:
                if main:
                    print(f"Resume stage train_complete: skipping train for epoch {epoch + 1}, running validation only.")
            if not skip_train:
                total_dataset_iters = len(train_loader)
                if max_train_steps is not None:
                    effective_dataset_iters = min(max_train_steps, total_dataset_iters)
                else:
                    effective_dataset_iters = total_dataset_iters
                optim_steps_visible = max(1, effective_dataset_iters // accumulation_steps)
                progress = (
                    tqdm(total=optim_steps_visible, desc=f"epoch {epoch + 1}/{config.train.epochs}")
                    if main
                    else None
                )

                last_iter_end = time.perf_counter()
                for step, batch in enumerate(train_loader, start=1):
                    iter_start = time.perf_counter()
                    data_time = iter_start - last_iter_end

                    h2d_start = time.perf_counter()
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    valid_mask = batch["valid_mask"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)
                    pixel_values_bg: torch.Tensor | None = batch.get("pixel_values_bg")
                    if pixel_values_bg is not None:
                        pixel_values_bg = pixel_values_bg.to(device, non_blocking=True)
                    if log_timing:
                        _sync_if_cuda(device)
                    h2d_time = time.perf_counter() - h2d_start

                    forward_start = time.perf_counter()
                    with torch.amp.autocast("cuda", enabled=config.train.amp and device.type == "cuda"):
                        outputs = model(pixel_values, valid_mask=valid_mask, return_attn=True)
                        outputs_bg: dict | None = None
                        if pixel_values_bg is not None:
                            outputs_bg = model(pixel_values_bg, valid_mask=valid_mask, return_attn=False)
                        loss, stats = criterion(outputs, labels, outputs_bg=outputs_bg, valid_mask=valid_mask)
                        loss_for_backward = loss / accumulation_steps
                    if log_timing:
                        _sync_if_cuda(device)
                    forward_time = time.perf_counter() - forward_start

                    if torch.isnan(loss) or torch.isinf(loss):
                        if progress is not None:
                            progress.set_postfix(loss="NaN", lr=f"{optimizer.param_groups[0]['lr']:.2e}", iter=f"{step}/{effective_dataset_iters}")
                        print(f"Loss is NaN/inf at step {step}, epoch {epoch + 1}. Stopping epoch early.")
                        break

                    backward_start = time.perf_counter()
                    scaler.scale(loss_for_backward).backward()
                    reached_step_limit = max_train_steps is not None and step >= max_train_steps
                    should_step = step % accumulation_steps == 0 or step == effective_dataset_iters or reached_step_limit
                    if should_step:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        if warmup_steps > 0 and global_step < warmup_steps:
                            scale = (global_step + 1) / max(1, warmup_steps)
                            for pg in optimizer.param_groups:
                                pg["lr"] = base_lr * scale
                        optimizer.step()
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        global_step += 1
                        if main and log_grad_norm:
                            print(f"Step {global_step}: grad_norm={float(grad_norm.detach().cpu()):.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
                        if writer is not None:
                            writer.add_scalar("train/grad_norm", float(grad_norm.detach().cpu()), global_step)
                        if progress is not None:
                            progress.update(1)
                    if log_timing:
                        _sync_if_cuda(device)
                    backward_step_time = time.perf_counter() - backward_start
                    total_step_time = time.perf_counter() - iter_start
                    last_iter_end = time.perf_counter()
                    if log_timing and step % timing_log_every == 0:
                        stats.update({
                            "time_data": data_time,
                            "time_h2d": h2d_time,
                            "time_forward": forward_time,
                            "time_backward_step": backward_step_time,
                            "time_step": total_step_time,
                        })

                    if writer is not None:
                        writer_step += 1
                        for name, value in stats.items():
                            running[name] = value
                            writer.add_scalar(f"train/{name}", value, writer_step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                    if progress is not None:
                        progress.set_postfix(
                            loss=f"{stats['loss_total']:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                            iter=f"{step}/{effective_dataset_iters}",
                        )

                    if main and csv_logger is not None and global_step > 0 and global_step % int(config.train.log_every) == 0:
                        csv_logger.log({
                            "phase": "train",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "lr": optimizer.param_groups[0]["lr"],
                            **stats,
                        })

                    if max_train_steps is not None and step >= max_train_steps:
                        break

                if progress is not None:
                    progress.close()

                if config.train.save_every_epoch and main:
                    save_checkpoint(
                        output_dir / "latest_train.ckpt",
                        _unwrap_model(model),
                        optimizer,
                        scheduler,
                        _unwrap_model(criterion) if hasattr(criterion, "module") else criterion,
                        epoch,
                        global_step,
                        best_metric,
                        dict(config),
                        completed_epoch=epoch,
                        next_epoch=epoch,
                        checkpoint_stage="train_complete",
                        writer_step=writer_step,
                    )

            if config.train.validate_every_epoch and main:
                metrics = evaluate(_unwrap_model(model), val_loader, device=device, max_steps=max_val_steps)
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(f"val/{name}", value, epoch + 1)
                if csv_logger is not None:
                    csv_logger.log({"phase": "val", "epoch": epoch + 1, "step": global_step, **metrics})
                metric_value = metrics.get(best_metric_name, float("nan"))
                print(f"Validation epoch {epoch + 1}: {metrics}")
                if not math.isnan(metric_value) and metric_value > best_metric:
                    best_metric = metric_value
                    save_checkpoint(
                        output_dir / "best_auc.ckpt",
                        _unwrap_model(model),
                        optimizer,
                        scheduler,
                        _unwrap_model(criterion) if hasattr(criterion, "module") else criterion,
                        epoch + 1,
                        global_step,
                        best_metric,
                        dict(config),
                        completed_epoch=epoch + 1,
                        next_epoch=epoch + 1,
                        checkpoint_stage="epoch_complete",
                        writer_step=writer_step,
                    )

            if config.train.save_every_epoch and main:
                save_checkpoint(
                    output_dir / "last.ckpt",
                    _unwrap_model(model),
                    optimizer,
                    scheduler,
                    _unwrap_model(criterion) if hasattr(criterion, "module") else criterion,
                    epoch + 1,
                    global_step,
                    best_metric,
                    dict(config),
                    completed_epoch=epoch + 1,
                    next_epoch=epoch + 1,
                    checkpoint_stage="epoch_complete",
                )

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            resume_stage = "epoch_complete"
    finally:
        if writer is not None:
            writer.close()
