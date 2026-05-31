from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.video_segment_dataset import VideoSegmentDataset
from src.losses.unite_loss import UNITELoss
from src.models.unite import UNITEModel


def build_dataset(config, split: str) -> VideoSegmentDataset:
    split_file = Path(config.paths.split_dir) / config.splits[split]
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Run data/build_ffpp_splits.py before training."
        )
    innovation = config.get("innovation", {})
    innovation_enabled = bool(innovation.get("enabled", False))
    bg_aug = innovation.get("background_aug", {}) if innovation_enabled else None
    return_counterfactual = bool(innovation.get("counterfactual", {}).get("return_bg_view", False)) if innovation_enabled else False
    return VideoSegmentDataset(
        split_csv=split_file,
        mode="train" if split == "train" else split,
        num_frames=config.data.num_frames,
        temporal_stride=config.data.temporal_stride,
        image_size=config.data.image_size,
        image_mean=config.data.image_mean,
        image_std=config.data.image_std,
        train_random_start=config.data.train_random_start,
        require_exists=config.data.require_exists,
        innovation_enabled=innovation_enabled and split == "train",
        bg_aug=bg_aug if split == "train" else None,
        return_counterfactual=return_counterfactual and split == "train",
    )


def build_dataloader(config, split: str, distributed: bool = False) -> DataLoader:
    dataset = build_dataset(config, split)
    sampler: DistributedSampler | None = None
    if distributed and split == "train":
        sampler = DistributedSampler(dataset, shuffle=True)

    num_workers = int(config.data.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": config.train.batch_size,
        "shuffle": split == "train" and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": config.data.pin_memory and torch.cuda.is_available(),
        "drop_last": split == "train" and len(dataset) >= config.train.batch_size,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(config.data.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config.data.get("prefetch_factor", 2))
    return DataLoader(dataset, **loader_kwargs)


def build_model(config) -> UNITEModel:
    return UNITEModel(
        siglip_model_name=config.model.siglip_model_name,
        local_files_only=config.model.local_files_only,
        freeze_siglip=config.model.freeze_siglip,
        encoder_batch_size=config.model.encoder_batch_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
        num_frames=config.model.num_frames,
        transformer_depth=config.model.transformer_depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    )


def build_criterion(config) -> UNITELoss:
    class_balance = config.loss.get("class_balance", {})
    class_weight = None
    if class_balance.get("enabled", False):
        mode = class_balance.get("mode", "inverse_freq")
        if mode == "inverse_freq":
            fake_ratio = class_balance.get("fake_ratio", 0.857)
            real_ratio = 1.0 - fake_ratio
            w_real = min(1.0 / max(real_ratio, 0.01), 5.0)
            w_fake = min(1.0 / max(fake_ratio, 0.01), 5.0)
            class_weight = (w_real, w_fake)
    return UNITELoss(
        num_classes=config.model.num_classes,
        num_heads=config.model.num_heads,
        num_frames=config.model.num_frames,
        ce_weight=config.loss.ce_weight,
        ad_weight=config.loss.ad_weight,
        center_eta=config.loss.center_eta,
        delta_between=config.loss.delta_between,
        delta_within=config.loss.delta_within,
        attn_entropy_weight=float(config.loss.get("attn_entropy_weight", 0.0)),
        cf_consistency_weight=float(config.loss.get("cf_consistency_weight", 0.0)),
        class_weight=class_weight,
    )
