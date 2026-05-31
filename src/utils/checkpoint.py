from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(item) for item in value)
    return value


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: torch.nn.Module | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: dict[str, Any],
    completed_epoch: int | None = None,
    next_epoch: int | None = None,
    checkpoint_stage: str = "epoch_complete",
    writer_step: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "criterion": criterion.state_dict() if criterion is not None else None,
        "epoch": epoch,
        "completed_epoch": epoch if completed_epoch is None else completed_epoch,
        "next_epoch": epoch if next_epoch is None else next_epoch,
        "checkpoint_stage": checkpoint_stage,
        "global_step": global_step,
        "writer_step": global_step if writer_step is None else writer_step,
        "best_metric": best_metric,
        "config": _to_plain(config),
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    criterion: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if criterion is not None and checkpoint.get("criterion") is not None:
        criterion.load_state_dict(checkpoint["criterion"])
    return checkpoint
