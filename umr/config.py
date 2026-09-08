"""YAML 配置加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from umr.paths import CONFIG_DIR, PROJECT_ROOT

#: ``--robot`` 的取值 -> 配置文件。加一台机器人就在这里补一行；不登记也行，
#: 直接把 yaml 路径传给 ``--robot``。
ROBOT_CONFIGS: dict[str, Path] = {
    "unitree_g1": CONFIG_DIR / "g1_29dof_rev_1_0.yaml",
}
#: 型号简写。
ROBOT_ALIASES: dict[str, str] = {
    "g1": "unitree_g1",
}
DEFAULT_ROBOT = "unitree_g1"
DEFAULT_CONFIG = ROBOT_CONFIGS[DEFAULT_ROBOT]


def robot_key(spec: str | Path | None = None) -> str:
    """归一化 ``--robot`` 的取值，用作输出目录名。"""
    if spec is None:
        return DEFAULT_ROBOT
    key = str(spec).lower()
    key = ROBOT_ALIASES.get(key, key)
    if key in ROBOT_CONFIGS:
        return key
    return Path(spec).stem


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


def resolve_config(spec: str | Path | None = None) -> Path:
    """把 ``unitree_g1`` / ``g1`` 这类型号解析成配置路径；也接受直接给出的 yaml 路径。"""
    if spec is None:
        return DEFAULT_CONFIG
    key = str(spec).lower()
    key = ROBOT_ALIASES.get(key, key)
    if key in ROBOT_CONFIGS:
        return ROBOT_CONFIGS[key]
    path = Path(spec)
    if path.suffix.lower() in (".yaml", ".yml"):
        return path if path.is_absolute() else (PROJECT_ROOT / path)
    known = " / ".join(sorted(ROBOT_CONFIGS))
    raise ValueError(f"未知的机器人型号 {spec!r}，可用: {known}，或直接给一个 yaml 路径")


def load_config(spec: str | Path | None = None) -> Config:
    """按型号简写或配置路径加载配置。"""
    path = resolve_config(spec)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
