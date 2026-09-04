"""项目内的固定路径。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSET_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

#: 所有中间产物与结果的落盘位置。跑多台机器人时用 ``UMR_OUTPUT_DIR`` 分开存放。
OUTPUT_DIR = Path(os.environ.get("UMR_OUTPUT_DIR") or PROJECT_ROOT / "outputs")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
