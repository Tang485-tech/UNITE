import shutil
from pathlib import Path

import kagglehub

DATASET = "xdxd003/ff-c23"
DATASET_DIR_NAME = "FaceForensics++_C23"
CACHE_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / "xdxd003"
TARGET_DIR = Path.cwd() / "data" / DATASET_DIR_NAME

# Download
download_path = Path(kagglehub.dataset_download(DATASET))
source_dir = download_path / DATASET_DIR_NAME
print(f"Downloaded to: {download_path}")

# Move FaceForensics++_C23 to ./data/
if not source_dir.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {source_dir}")
if TARGET_DIR.exists():
    raise FileExistsError(f"Target directory already exists: {TARGET_DIR}")

TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(source_dir), str(TARGET_DIR))
print(f"Moved to: {TARGET_DIR}")

# Clean up cache
if CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)
    print(f"Cache cleaned: {CACHE_DIR}")
