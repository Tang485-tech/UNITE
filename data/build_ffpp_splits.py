from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ffpp_metadata import load_ffpp_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified FF++ C23 train/val/test splits.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--require_exists", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must leave a positive test ratio")

    df = load_ffpp_metadata(args.input_csv, args.data_root, require_exists=args.require_exists)
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=df["label"],
    )
    relative_val = args.val_ratio / (args.val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = out_dir / f"{name}.csv"
        split_df.reset_index(drop=True).to_csv(path, index=False)
        counts = split_df["label_name"].value_counts().to_dict()
        print(f"Wrote {path} ({len(split_df)} rows, labels={counts})")


if __name__ == "__main__":
    main()
