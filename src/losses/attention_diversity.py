from __future__ import annotations

import itertools

import torch
from torch import nn
import torch.nn.functional as F


class AttentionDiversityLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_heads: int = 12,
        num_frames: int = 64,
        center_eta: float = 0.05,
        delta_between: float = 0.5,
        delta_within: list[float] | tuple[float, ...] = (0.01, -2.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.center_eta = center_eta
        self.delta_between = delta_between
        if len(delta_within) != num_classes:
            raise ValueError("delta_within length must match num_classes")
        self.register_buffer("centers", torch.zeros(num_classes, num_heads, num_frames))
        self.register_buffer("delta_within", torch.tensor(delta_within, dtype=torch.float32))

    def _update_centers(self, ad_features: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            detached = ad_features.detach()
            for class_id in range(self.num_classes):
                class_mask = labels == class_id
                if not torch.any(class_mask):
                    continue
                class_mean = detached[class_mask].mean(dim=0)
                self.centers[class_id].mul_(1.0 - self.center_eta).add_(class_mean * self.center_eta)

    def forward(
        self,
        ad_features: torch.Tensor,
        labels: torch.Tensor,
        update_centers: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if ad_features.ndim != 3:
            raise ValueError(f"Expected ad_features [B,H,T], got {tuple(ad_features.shape)}")
        if ad_features.shape[1:] != (self.num_heads, self.num_frames):
            raise ValueError(
                f"Expected ad_features [B,{self.num_heads},{self.num_frames}], got {tuple(ad_features.shape)}"
            )

        labels = labels.long()
        centers_for_labels = self.centers.index_select(0, labels)
        within_distance = torch.linalg.vector_norm(ad_features - centers_for_labels, dim=(1, 2))
        class_delta = self.delta_within.index_select(0, labels).to(ad_features.dtype)
        within_loss = F.relu(within_distance - class_delta).mean()

        between_terms = []
        # Paper Eq. (5) literally writes the sum over (k,l) ∈ (n_h, n_h), but the
        # surrounding text describes "feature centers of different classes" and
        # Fig. 3 frames the goal as separating real vs. fake. We take the
        # class-pair reading: distances are measured between class centers,
        # not between heads. With num_classes=2 this collapses to a single term.
        for left, right in itertools.combinations(range(self.num_classes), 2):
            distance = torch.linalg.vector_norm(self.centers[left] - self.centers[right])
            between_terms.append(F.relu(torch.as_tensor(self.delta_between, device=distance.device) - distance))
        if between_terms:
            between_loss = torch.stack(between_terms).mean().to(ad_features.dtype)
        else:
            between_loss = ad_features.new_zeros(())

        loss = within_loss + between_loss
        if update_centers and self.training:
            self._update_centers(ad_features, labels)

        stats = {
            "loss_ad_within": within_loss.detach(),
            "loss_ad_between": between_loss.detach(),
            "center_distance_0_1": torch.linalg.vector_norm(self.centers[0] - self.centers[1]).detach()
            if self.num_classes >= 2
            else ad_features.new_zeros(()),
        }
        return loss, stats
