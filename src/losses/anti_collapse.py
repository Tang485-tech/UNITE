from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_attention_entropy(
    first_attn: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shannon entropy of the first temporal block's attention along the key axis.

    After the spatial-pooler refactor, `first_attn` is the temporal MHSA output
    of `blocks[0]` (shape [B, H, T_q, T_k]). This regularizer encourages
    diverse attention *across frames*, not across spatial regions — the
    spatial AD-loss (Eq. 2 / Sec. 3.4) handles spatial diversity.
    """
    if first_attn is None:
        return torch.zeros(())
    batch, heads, f_query, f_key = first_attn.shape
    # nn.MultiheadAttention rows are already softmaxed; mean-over-queries
    # preserves sum=1 along the key axis, so we renormalize after masking
    # rather than applying softmax again.
    attn_key = first_attn.mean(dim=2)
    if valid_mask is not None:
        if valid_mask.shape[1] != f_key:
            valid_mask = valid_mask[:, :f_key] if valid_mask.shape[1] > f_key else torch.nn.functional.pad(
                valid_mask, (0, f_key - valid_mask.shape[1]), value=0.0
            )
        mask = valid_mask.to(dtype=attn_key.dtype, device=attn_key.device).unsqueeze(1)
        attn_key = attn_key * mask
    denom = attn_key.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    prob = attn_key / denom
    entropy = -(prob * torch.log(prob + 1e-9)).sum(dim=-1)
    return entropy.mean()


def counterfactual_consistency(
    logits_orig: torch.Tensor,
    logits_bg: torch.Tensor,
) -> torch.Tensor:
    p_orig = F.softmax(logits_orig, dim=-1).clamp_min(1e-9)
    p_bg = F.softmax(logits_bg, dim=-1).clamp_min(1e-9)
    kl_orig_to_bg = (p_orig * (torch.log(p_orig) - torch.log(p_bg))).sum(dim=-1).mean()
    kl_bg_to_orig = (p_bg * (torch.log(p_bg) - torch.log(p_orig))).sum(dim=-1).mean()
    return 0.5 * (kl_orig_to_bg + kl_bg_to_orig)
