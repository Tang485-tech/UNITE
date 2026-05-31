from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .face_mask_augment import FaceMaskAugmenter
from .video_reader import normalize_clip, read_video_clip


class VideoSegmentDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        mode: str,
        num_frames: int = 64,
        temporal_stride: int = 2,
        image_size: int = 384,
        image_mean: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
        train_random_start: bool = True,
        require_exists: bool = True,
        innovation_enabled: bool = False,
        bg_aug: dict[str, Any] | None = None,
        return_counterfactual: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.mode = mode
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.train_random_start = train_random_start
        self.require_exists = require_exists
        self.innovation_enabled = innovation_enabled
        self.return_counterfactual = return_counterfactual

        self.augmenter: FaceMaskAugmenter | None = None
        if innovation_enabled and bg_aug:
            self.augmenter = FaceMaskAugmenter(
                method=bg_aug.get("method", "mean"),
                haar_scale_factor=float(bg_aug.get("haar_scale_factor", 1.1)),
                haar_min_neighbors=int(bg_aug.get("haar_min_neighbors", 4)),
                haar_min_size_ratio=float(bg_aug.get("haar_min_size_ratio", 0.08)),
                fallback_ellipse_ratio=tuple(bg_aug.get("fallback_ellipse_ratio", (0.22, 0.30))),
                expand_ratio=float(bg_aug.get("expand_ratio", 0.15)),
            )

        self.df = pd.read_csv(self.split_csv)
        required = {"abs_path", "rel_path", "label"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Split CSV {self.split_csv} is missing columns: {sorted(missing)}")
        if self.require_exists:
            exists = self.df["abs_path"].map(lambda value: Path(value).exists())
            missing_count = int((~exists).sum())
            if missing_count:
                print(f"Skipping {missing_count} missing videos from {self.split_csv}")
            self.df = self.df.loc[exists].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Split CSV is empty after filtering: {self.split_csv}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        pixel_values, valid_mask = read_video_clip(
            row["abs_path"],
            num_frames=self.num_frames,
            temporal_stride=self.temporal_stride,
            image_size=self.image_size,
            random_start=self.mode == "train" and self.train_random_start,
        )
        result: dict[str, torch.Tensor | str] = {
            "pixel_values": normalize_clip(pixel_values, self.image_mean, self.image_std),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "valid_mask": valid_mask,
            "rel_path": str(row["rel_path"]),
        }
        if self.innovation_enabled and self.mode == "train" and self.augmenter is not None:
            pixel_values_bg = self.augmenter.apply_mask_to_clip(pixel_values)
            result["pixel_values_bg"] = normalize_clip(pixel_values_bg, self.image_mean, self.image_std)
        return result
