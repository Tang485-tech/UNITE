from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_MAP = {"REAL": 0, "FAKE": 1}


def load_ffpp_metadata(
    csv_path: str | Path,
    data_root: str | Path,
    require_exists: bool = True,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    data_root = Path(data_root)
    df = pd.read_csv(csv_path)

    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed") or str(col) == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required = {"File Path", "Label", "Frame Count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["rel_path"] = df["File Path"].astype(str).str.replace("\\\\", "/", regex=False)
    out["abs_path"] = out["rel_path"].map(lambda value: str(data_root / value))
    out["label_name"] = df["Label"].astype(str).str.upper()
    unknown_labels = sorted(set(out["label_name"]) - set(LABEL_MAP))
    if unknown_labels:
        raise ValueError(f"Unknown labels in metadata CSV: {unknown_labels}")
    out["label"] = out["label_name"].map(LABEL_MAP).astype("int64")
    out["frame_count"] = pd.to_numeric(df["Frame Count"], errors="coerce").fillna(0).astype("int64")

    for optional in ("Width", "Height", "Codec", "File Size(MB)"):
        if optional in df.columns:
            normalized = optional.lower().replace(" ", "_").replace("(", "").replace(")", "")
            out[normalized] = df[optional]

    exists = out["abs_path"].map(lambda value: Path(value).exists())
    missing_count = int((~exists).sum())
    if missing_count:
        print(f"Missing FF++ videos: {missing_count}/{len(out)}")
    if require_exists:
        out = out.loc[exists].reset_index(drop=True)

    if out.empty:
        raise ValueError(
            "No FF++ videos are available after filtering. Check data_root or run without --require_exists while downloading."
        )
    return out
