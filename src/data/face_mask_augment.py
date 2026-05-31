from __future__ import annotations

import cv2
import numpy as np
import torch


class FaceMaskAugmenter:
    def __init__(
        self,
        method: str = "mean",
        haar_scale_factor: float = 1.1,
        haar_min_neighbors: int = 4,
        haar_min_size_ratio: float = 0.08,
        fallback_ellipse_ratio: tuple[float, float] = (0.22, 0.30),
        expand_ratio: float = 0.15,
    ) -> None:
        self.method = method
        self.expand_ratio = expand_ratio
        self.fallback_ellipse_ratio = fallback_ellipse_ratio
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.haar_scale_factor = haar_scale_factor
        self.haar_min_neighbors = haar_min_neighbors
        self.haar_min_size_ratio = haar_min_size_ratio

    def _face_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        min_size = max(20, int(min(h, w) * self.haar_min_size_ratio))
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.haar_scale_factor,
            minNeighbors=self.haar_min_neighbors,
            minSize=(min_size, min_size),
        )
        mask = np.zeros((h, w), dtype=np.float32)
        if len(faces) > 0:
            for x, y, fw, fh in faces:
                dx = int(fw * self.expand_ratio)
                dy = int(fh * self.expand_ratio)
                x1 = max(0, x - dx)
                y1 = max(0, y - dy)
                x2 = min(w, x + fw + dx)
                y2 = min(h, y + fh + dy)
                mask[y1:y2, x1:x2] = 1.0
        else:
            cx, cy = w // 2, h // 2
            rx = int(w * self.fallback_ellipse_ratio[0])
            ry = int(h * self.fallback_ellipse_ratio[1])
            yy, xx = np.ogrid[:h, :w]
            ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
            mask[ellipse <= 1.0] = 1.0
        return mask

    def apply_face_mask_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        mask = self._face_mask(frame_rgb)
        mask_3c = np.stack([mask] * 3, axis=-1)
        if self.method == "zero":
            return (frame_rgb * (1.0 - mask_3c)).astype(frame_rgb.dtype)
        if self.method == "blur":
            blurred = cv2.GaussianBlur(frame_rgb, (21, 21), 0)
            return (frame_rgb * (1.0 - mask_3c) + blurred * mask_3c).astype(frame_rgb.dtype)
        mean_color = frame_rgb.mean(axis=(0, 1), keepdims=True)
        return (frame_rgb * (1.0 - mask_3c) + mean_color * mask_3c).astype(frame_rgb.dtype)

    def apply_mask_to_clip(self, clip: torch.Tensor) -> torch.Tensor:
        clip_np = (clip.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        frames_out = []
        for i in range(clip_np.shape[0]):
            frame_rgb = clip_np[i]
            masked = self.apply_face_mask_frame(frame_rgb)
            frames_out.append(masked.astype(np.float32) / 255.0)
        out = np.stack(frames_out, axis=0)
        return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().to(clip.device, dtype=clip.dtype)
