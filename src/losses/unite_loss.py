from __future__ import annotations

import torch
from torch import nn

from .anti_collapse import counterfactual_consistency, temporal_attention_entropy
from .attention_diversity import AttentionDiversityLoss


class UNITELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_heads: int,
        num_frames: int,
        ce_weight: float = 0.5,
        ad_weight: float = 0.5,
        center_eta: float = 0.05,
        delta_between: float = 0.5,
        delta_within: list[float] | tuple[float, ...] = (0.01, -2.0),
        attn_entropy_weight: float = 0.0,
        cf_consistency_weight: float = 0.0,
        class_weight: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.ad_weight = ad_weight
        self.attn_entropy_weight = attn_entropy_weight
        self.cf_consistency_weight = cf_consistency_weight
        ce_kwargs: dict = {}
        if class_weight is not None and len(class_weight) == num_classes:
            ce_kwargs["weight"] = torch.tensor(class_weight, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.ad = AttentionDiversityLoss(
            num_classes=num_classes,
            num_heads=num_heads,
            num_frames=num_frames,
            center_eta=center_eta,
            delta_between=delta_between,
            delta_within=delta_within,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        labels: torch.Tensor,
        outputs_bg: dict[str, torch.Tensor | None] | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        ad_features = outputs["ad_features"]
        if logits is None or ad_features is None:
            raise ValueError("UNITELoss requires logits and ad_features in model outputs")

        ce_loss = self.ce(logits, labels)
        ad_loss, ad_stats = self.ad(ad_features, labels, update_centers=True)
        total = self.ce_weight * ce_loss + self.ad_weight * ad_loss
        stats = {
            "loss_total": float(total.detach().cpu()),
            "loss_ce": float(ce_loss.detach().cpu()),
            "loss_ad": float(ad_loss.detach().cpu()),
        }
        stats.update({name: float(value.detach().cpu()) for name, value in ad_stats.items()})

        # Anti-collapse: temporal attention entropy
        if self.attn_entropy_weight > 0:
            first_attn = outputs.get("first_attn")
            ent_loss = temporal_attention_entropy(first_attn, valid_mask)
            total = total + self.attn_entropy_weight * ent_loss
            stats["loss_attn_entropy"] = float(ent_loss.detach().cpu())

        # Counterfactual consistency
        if self.cf_consistency_weight > 0 and outputs_bg is not None:
            logits_bg = outputs_bg.get("logits")
            if logits_bg is not None:
                cons_loss = counterfactual_consistency(logits, logits_bg)
                total = total + self.cf_consistency_weight * cons_loss
                stats["loss_cf_consistency"] = float(cons_loss.detach().cpu())

        return total, stats
