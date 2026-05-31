from __future__ import annotations

from pathlib import Path


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, object]) -> None:
        keys = list(row.keys())
        with self.path.open("a", encoding="utf-8") as handle:
            if not self.header_written:
                handle.write(",".join(keys) + "\n")
                self.header_written = True
            handle.write(",".join(str(row[key]) for key in keys) + "\n")
