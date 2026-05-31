from __future__ import annotations

import torch
from torch import nn

from .siglip_encoder import FrozenSigLIPEncoder
from .unite_transformer import UNITETransformer


class UNITEModel(nn.Module):
    def __init__(
        self,
        siglip_model_name: str,
        local_files_only: bool,
        freeze_siglip: bool,
        encoder_batch_size: int,
        hidden_size: int,
        num_classes: int,
        num_frames: int,
        transformer_depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = FrozenSigLIPEncoder(
            model_name=siglip_model_name,
            local_files_only=local_files_only,
            freeze=freeze_siglip,
            encoder_batch_size=encoder_batch_size,
        )
        actual_hidden_size = self.encoder.hidden_size
        if hidden_size != actual_hidden_size:
            print(f"Config hidden_size={hidden_size}, SigLIP hidden_size={actual_hidden_size}; using SigLIP value.")
            hidden_size = actual_hidden_size
        self.classifier = UNITETransformer(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_frames=num_frames,
            depth=transformer_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_attn: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        features = self.encoder(pixel_values)
        return self.classifier(features, valid_mask=valid_mask, return_attn=return_attn)
