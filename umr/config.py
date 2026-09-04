"""YAML 配置加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from umr.paths import CONFIG_DIR, PROJECT_ROOT

DEFAULT_CONFIG = CONFIG_DIR / "g1_29dof_rev_1_0.yaml"


class Config(dict):
    """支持点号访问的嵌套配置字典。"""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val

    def resolve(self, key: str) -> Path:
        """把配置里的相对路径解析成项目内的绝对路径。"""
        return (PROJECT_ROOT / str(self[key])).resolve()


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
