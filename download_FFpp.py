# !/usr/bin/env python
# coding: utf-8
# Author: rentG
# Contact: 2512734334@qq.com
# Date: 2026/05/25 15:58

import shutil
from pathlib import Path

import kagglehub

DATASET = "xdxd003/ff-c23"
CACHE_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / DATASET
TARGET_DIR = Path.cwd() / "data"

# Download (最新版本自动使用缓存)
download_path = kagglehub.dataset_download(DATASET)
print(f"Downloaded to: {download_path}")

# Move to ./data/
TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(download_path), str(TARGET_DIR))
print(f"Moved to: {TARGET_DIR}")

# Clean up cache
if CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)
    print(f"Cache cleaned: {CACHE_DIR}")
