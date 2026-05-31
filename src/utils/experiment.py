from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _safe_tag(value: Any) -> str:
    text = str(value)
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    return text.strip("-")


def _model_short_name(model_name: str) -> str:
    return _safe_tag(model_name.split("/")[-1])


def _float_tag(value: Any) -> str:
    return _safe_tag(f"{float(value):g}")


def resolve_output_dir(config, run_name: str | None = None) -> Path:
    existing = str(config.paths.get("output_dir", "auto"))
    if existing and existing != "auto":
        return Path(existing)

    dataset = _safe_tag(config.get("experiment", {}).get("dataset", "experiment"))
    variant = _safe_tag(config.get("experiment", {}).get("variant", "baseline"))
    root = Path(config.paths.get("output_root", "./outputs"))
    selected_run_name = run_name or config.get("experiment", {}).get("run_name") or f"seed{config.seed}"

    model_tag = (
        f"{_model_short_name(config.model.siglip_model_name)}"
        f"_d{config.model.transformer_depth}"
        f"_h{config.model.num_heads}"
    )
    data_tag = (
        f"img{config.data.image_size}"
        f"_nf{config.data.num_frames}"
        f"_stride{config.data.temporal_stride}"
    )
    loss_tag = f"ce{_float_tag(config.loss.ce_weight)}_ad{_float_tag(config.loss.ad_weight)}"
    train_tag = (
        f"lr{_float_tag(config.train.lr)}"
        f"_effbs{config.train.effective_batch_size}"
        f"_gc{_float_tag(config.train.grad_clip_max_norm)}"
        f"_amp{int(bool(config.train.amp))}"
    )
    output_dir = root / dataset / variant / model_tag / data_tag / loss_tag / train_tag / _safe_tag(selected_run_name)
    config.paths.output_dir = str(output_dir)
    if "experiment" in config:
        config.experiment.run_name = selected_run_name
    return output_dir
