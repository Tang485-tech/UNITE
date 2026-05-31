from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_binary_metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []

    for step, batch in enumerate(tqdm(dataloader, desc="eval", leave=False), start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        outputs = model(pixel_values, valid_mask=valid_mask, return_attn=False)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)[:, 1]
        labels.extend(label.cpu().tolist())
        scores.extend(probs.cpu().tolist())
        if max_steps is not None and step >= max_steps:
            break

    unique_labels = set(labels)
    if len(unique_labels) < 2:
        class_counts = {c: labels.count(c) for c in unique_labels}
        print(
            f"Warning: validation set contains only {len(unique_labels)} class(es): {class_counts}. "
            f"AUC/PR AUC will be NaN. Increase --max_val_steps or remove the limit."
        )

    return compute_binary_metrics(labels, scores)
