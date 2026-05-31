from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_position_encoding(max_len: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class SpatialAttentionPooler(nn.Module):
    """Block-0 of UNITE: spatial multi-head attention pooler.

    Each head owns a learnable query vector that attends across the t_s spatial
    SigLIP tokens of every frame, producing 𝒜 ∈ ℝ^(B, T, H, t_s) — the spatial
    attention map referenced by the AD-loss in Sec. 3.4 of the paper. The pooled
    per-frame feature ([B, T, D]) is what subsequent temporal encoder blocks
    consume.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(f"Expected features [B,T,t_s,D], got {tuple(features.shape)}")
        batch, frames, tokens, dim = features.shape
        if dim != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {dim}")

        x = self.norm(features)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(batch, frames, tokens, self.num_heads, self.head_dim)
        v = v.view(batch, frames, tokens, self.num_heads, self.head_dim)

        scores = torch.einsum("hd, btjhd -> bthj", self.queries, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn_drop = self.attn_dropout(attn)

        pooled = torch.einsum("bthj, btjhd -> bthd", attn_drop, v)
        pooled = pooled.reshape(batch, frames, dim)
        pooled = self.proj_dropout(self.out_proj(pooled))
        return pooled, attn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights if return_attention else None


class UNITETransformer(nn.Module):
    """UNITE transformer (Sec. 3.3 of the paper).

    Architecture (depth=4 by default):
      - SpatialAttentionPooler front-end — produces 𝒜 ∈ [B, T, H, t_s] used to
        build the AD-loss feature 𝒫 (Eq. 2).
      - `depth` temporal MHSA encoder blocks operating on [B, T, D].
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_classes: int = 2,
        num_frames: int = 64,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.temporal_depth = depth
        self.spatial_pooler = SpatialAttentionPooler(hidden_size, num_heads, dropout)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.register_buffer(
            "pos_encoding",
            sinusoidal_position_encoding(num_frames, hidden_size),
            persistent=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _masked_mean(self, x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            return x.mean(dim=1)
        weights = valid_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def _spatial_attn_to_ad_features(
        self,
        features: torch.Tensor,
        spatial_attn: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute 𝒫_{b,h,f} per Eq. (2).

        Paper: 𝒫_{h,f} = Σ_j Σ_k 𝒜_{h,j,k} · ξ_{f,j,k}. Treating 𝒜 as the
        spatial attention 𝒜 ∈ [H, t_s] (the `d_s` axis in the formula is a
        symbol artifact — physically there is one weight per spatial token),
        we use channel-mean of ξ over k for numerical stability rather than
        a raw sum, which would scale with d_s and dwarf δ_within.
        """
        batch, frames = features.shape[0], features.shape[1]
        xi_chan_mean = features.mean(dim=-1)
        ad_features = torch.einsum("bthj, btj -> bht", spatial_attn, xi_chan_mean)
        if valid_mask is not None:
            mask = valid_mask.to(dtype=ad_features.dtype, device=ad_features.device).unsqueeze(1)
            ad_features = ad_features * mask
        if frames != self.num_frames:
            padded = ad_features.new_zeros(batch, self.num_heads, self.num_frames)
            length = min(self.num_frames, frames)
            padded[..., :length] = ad_features[..., :length]
            ad_features = padded
        return ad_features

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_attn: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        if features.ndim != 4:
            raise ValueError(f"Expected features [B,T,t_s,D], got {tuple(features.shape)}")
        if features.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {features.shape[-1]}")
        if features.shape[1] > self.num_frames:
            features = features[:, : self.num_frames]
            if valid_mask is not None:
                valid_mask = valid_mask[:, : self.num_frames]

        pooled, spatial_attn = self.spatial_pooler(features)
        ad_features = self._spatial_attn_to_ad_features(features, spatial_attn, valid_mask)

        batch_size, frames, _ = pooled.shape
        x = self.input_norm(pooled)
        x = x + self.pos_encoding[:, :frames].to(dtype=x.dtype, device=x.device)
        x = self.dropout(x)

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.to(device=x.device, dtype=torch.bool)

        first_temporal_attn = None
        for block_index, block in enumerate(self.blocks):
            x, attn = block(
                x,
                key_padding_mask=key_padding_mask,
                return_attention=return_attn and block_index == 0,
            )
            if block_index == 0:
                first_temporal_attn = attn

        x = self.final_norm(x)
        logits = self.classifier(self._masked_mean(x, valid_mask))

        return {
            "logits": logits,
            "spatial_attn": spatial_attn if return_attn else None,
            "first_attn": first_temporal_attn,
            "ad_features": ad_features,
        }
