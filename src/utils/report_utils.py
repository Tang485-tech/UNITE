"""Helpers for the post-training reporting notebook.

The notebook (`notebooks/report_visualization.ipynb`) imports these functions
to keep cells short. Everything here is read-only with respect to project
state — nothing trains, fine-tunes, or writes checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.data.video_reader import normalize_clip, read_video_clip
from src.engine.build import build_dataloader, build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def load_model_from_ckpt(
    config_path: str | Path,
    ckpt_path: str | Path,
    device: str | torch.device = "cuda",
) -> tuple[torch.nn.Module, Any]:
    """Build a UNITEModel matching the config and load weights from `ckpt_path`."""
    config = load_config(str(config_path))
    device = torch.device(device)
    model = build_model(config).to(device)
    load_checkpoint(str(ckpt_path), model, map_location=device)
    model.eval()
    return model, config


def read_segment(
    video_path: str | Path,
    num_frames: int = 64,
    temporal_stride: int = 2,
    image_size: int = 384,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read a video and return (raw [T,3,H,W] in [0,1], normalized [T,3,H,W], valid_mask [T])."""
    raw, valid_mask = read_video_clip(
        video_path,
        num_frames=num_frames,
        temporal_stride=temporal_stride,
        image_size=image_size,
        random_start=False,
    )
    normalized = normalize_clip(raw, image_mean, image_std)
    return raw, normalized, valid_mask


def denormalize(
    clip: torch.Tensor,
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Invert `normalize_clip` so a normalized [T,3,H,W] tensor returns to [0,1] for plotting."""
    mean = torch.tensor(image_mean, dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    std = torch.tensor(image_std, dtype=clip.dtype, device=clip.device).view(1, 3, 1, 1)
    return (clip * std + mean).clamp(0.0, 1.0)


def attention_to_heatmap(
    attn_one_head: torch.Tensor,
    image_size: int = 384,
    grid: int | None = None,
) -> np.ndarray:
    """Resize a [t_s] spatial attention vector into an [image_size, image_size] heatmap.

    Args:
        attn_one_head: shape [t_s], e.g. 729 for SigLIP-So400m at 384.
        image_size: output resolution.
        grid: side length of the patch grid. If None, defaults to int(sqrt(t_s)).

    Returns:
        H × W float numpy array in [0, 1] (min-max scaled per heatmap).
    """
    if attn_one_head.ndim != 1:
        raise ValueError(f"Expected 1-D attention vector, got {tuple(attn_one_head.shape)}")
    tokens = attn_one_head.shape[0]
    if grid is None:
        grid = int(round(tokens ** 0.5))
        if grid * grid != tokens:
            raise ValueError(f"t_s={tokens} is not a perfect square; pass grid explicitly")
    grid_attn = attn_one_head.view(1, 1, grid, grid).float()
    upsampled = F.interpolate(grid_attn, size=(image_size, image_size), mode="bilinear", align_corners=False)
    heatmap = upsampled.squeeze().detach().cpu().numpy()
    lo, hi = float(heatmap.min()), float(heatmap.max())
    if hi - lo > 1e-12:
        heatmap = (heatmap - lo) / (hi - lo)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


@torch.no_grad()
def run_model_on_segment(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """Forward a single (un-batched) clip through the model with attention enabled."""
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    if valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)
    pixel_values = pixel_values.to(device)
    valid_mask = valid_mask.to(device)
    return model(pixel_values, valid_mask=valid_mask, return_attn=True)


def iter_dataloader_samples(
    config_path: str | Path,
    split: str = "val",
    max_samples: int | None = 16,
) -> Iterable[dict[str, Any]]:
    """Yield raw dataloader batches without modifying the project's pipeline."""
    config = load_config(str(config_path))
    loader = build_dataloader(config, split, distributed=False)
    for index, batch in enumerate(loader):
        yield batch
        if max_samples is not None and index + 1 >= max_samples:
            break


def overlay_heatmap_on_frame(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB frame [H,W,3] in [0,1] using a jet-like colormap."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap("jet")
    colored = cmap(heatmap)[..., :3]
    return (1.0 - alpha) * frame + alpha * colored


def parse_method_from_rel_path(rel_path: str) -> str:
    """FF++ rel_path looks like '<Method>/<file>.mp4'. Returns the leading directory."""
    parts = Path(rel_path).parts
    return parts[0] if parts else "unknown"
