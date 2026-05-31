from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch


class VideoReadError(RuntimeError):
    pass


def _choose_start(frame_count: int, needed_span: int, random_start: bool) -> int:
    if frame_count <= 0 or frame_count <= needed_span:
        return 0
    max_start = frame_count - needed_span
    return random.randint(0, max_start) if random_start else max_start // 2


def read_video_clip(
    video_path: str | Path,
    num_frames: int = 64,
    temporal_stride: int = 2,
    image_size: int = 384,
    random_start: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoReadError(f"Failed to open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        needed_span = max(1, (num_frames - 1) * temporal_stride + 1)
        start = _choose_start(frame_count, needed_span, random_start)
        frames: list[np.ndarray] = []
        valid_mask: list[float] = []
        last_frame: np.ndarray | None = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for offset in range(needed_span):
            ok, frame = cap.read()
            if offset % temporal_stride != 0:
                if not ok or frame is None:
                    break
                continue

            if not ok or frame is None:
                if last_frame is None:
                    break
                frames.append(last_frame.copy())
                valid_mask.append(0.0)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            last_frame = frame
            frames.append(frame)
            valid_mask.append(1.0)
            if len(frames) >= num_frames:
                break

        if not frames:
            raise VideoReadError(f"No frames decoded from video: {video_path}")

        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
            valid_mask.append(0.0)

        array = np.stack(frames[:num_frames]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
        mask = torch.tensor(valid_mask[:num_frames], dtype=torch.float32)
        return tensor, mask
    finally:
        cap.release()


def normalize_clip(
    pixel_values: torch.Tensor,
    mean: list[float] | tuple[float, float, float],
    std: list[float] | tuple[float, float, float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    return (pixel_values - mean_tensor) / std_tensor
