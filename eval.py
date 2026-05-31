from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.build import build_dataloader, build_model
from src.engine.evaluator import evaluate
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a UNITE checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_val_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.seed))
    device = torch.device(args.device)
    dataloader = build_dataloader(config, args.split)
    model = build_model(config).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    metrics = evaluate(model, dataloader, device=device, max_steps=args.max_val_steps)
    print(metrics)


if __name__ == "__main__":
    main()
