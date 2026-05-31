from __future__ import annotations

import torch
from torch import nn
from transformers import SiglipVisionModel


class FrozenSigLIPEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        local_files_only: bool = False,
        freeze: bool = True,
        encoder_batch_size: int = 4,
    ) -> None:
        super().__init__()
        self.vision_model = SiglipVisionModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.freeze = freeze
        self.encoder_batch_size = max(1, int(encoder_batch_size))
        if freeze:
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.vision_model.config.hidden_size)

    def train(self, mode: bool = True) -> "FrozenSigLIPEncoder":
        super().train(mode)
        if self.freeze:
            self.vision_model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected [B,T,3,H,W] pixel_values, got {tuple(pixel_values.shape)}")
        batch_size, num_frames, channels, height, width = pixel_values.shape
        flat_pixels = pixel_values.view(batch_size * num_frames, channels, height, width)

        token_chunks = []
        grad_context = torch.no_grad() if self.freeze else torch.enable_grad()
        with grad_context:
            for chunk in flat_pixels.split(self.encoder_batch_size, dim=0):
                outputs = self.vision_model(pixel_values=chunk)
                token_chunks.append(outputs.last_hidden_state)

        tokens = torch.cat(token_chunks, dim=0)
        return tokens.view(batch_size, num_frames, tokens.shape[1], tokens.shape[2])
