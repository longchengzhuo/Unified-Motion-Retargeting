"""两个入口脚本共用的命令行处理：入参定义、路径解析、动作文件发现。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from umr.bodies.skeletons import DEFAULT_HUMAN, SKELETONS, Skeleton, get_skeleton
from umr.config import DEFAULT_ROBOT, ROBOT_CONFIGS, Config, load_config, robot_key
from umr.paths import OUTPUT_DIR, PROJECT_ROOT, ClipLayout, clip_layout

DEFAULT_SAVE_PATH = OUTPUT_DIR.name


def add_target_args(ap: argparse.ArgumentParser) -> None:
    """「哪段动作 + 哪套源骨架 + 哪台机器人 + 存到哪」这一组共用入参。"""
    ap.add_argument(
        "--motion_file", required=True,
        help="BVH 文件，或装着一堆 BVH 的目录（递归查找，按文件名排序处理）",
    )
    ap.add_argument(
        "--human", default=DEFAULT_HUMAN, type=str.lower, choices=sorted(SKELETONS),
        help=f"源动捕骨架的命名约定，默认 {DEFAULT_HUMAN}",
    )
    ap.add_argument(
        "--robot", default=DEFAULT_ROBOT, type=str.lower,
        help=f"目标机器人：已登记的型号（{' / '.join(sorted(ROBOT_CONFIGS))}）或一个 "
             f"yaml 配置路径，默认 {DEFAULT_ROBOT}",
    )
    ap.add_argument(
        "--save_path", default=DEFAULT_SAVE_PATH,
        help=f"输出根目录，默认 {DEFAULT_SAVE_PATH}/",
    )


def resolve_path(spec: str | Path, what: str = "路径") -> Path:
    """定位一个输入路径：相对路径先按当前目录找，找不到再按项目根目录找。

    ``data/xxx.bvh`` 这种写法在 README 和配置里到处都是，而脚本可能从任意目录
    （IDE 的运行配置、家目录……）启动，所以两处都试一下。
    """
    spec = Path(spec)
    candidates = [spec] if spec.is_absolute() else [Path.cwd() / spec, PROJECT_ROOT / spec]
    for path in candidates:
        if path.exists():
            return path.resolve()
    tried = "\n".join(f"  {p}" for p in candidates)
    raise SystemExit(f"找不到{what}: {spec}\n已尝试:\n{tried}")


def resolve_save_path(spec: str | Path) -> Path:
    """输出根目录：相对路径按项目根目录解析，避免从别处启动时散落一地。"""
    path = Path(spec)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Clip:
    """一段待处理的动作，以及它在输出目录里的位置。"""

    path: Path
    rel_dir: Path
    layout: ClipLayout

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def label(self) -> str:
        return str(self.rel_dir / self.path.name) if str(self.rel_dir) != "." else self.path.name


def discover_clips(args: argparse.Namespace, skeleton: Skeleton) -> list[Clip]:
    """把 ``--motion_file`` 展开成待处理的片段列表。

    给目录时递归收集该骨架支持的后缀，并保留相对目录结构，让输出与输入一一对应。
    """
    root = resolve_path(args.motion_file, "动作文件")
    save_path = resolve_save_path(args.save_path)
    robot = robot_key(args.robot)

    if root.is_dir():
        files = sorted(
            p for ext in skeleton.extensions for p in root.rglob(f"*{ext}") if p.is_file()
        )
        if not files:
            exts = " / ".join(skeleton.extensions)
            raise SystemExit(f"目录里没有 {exts} 文件: {root}")
        rel_of = {p: p.relative_to(root).parent for p in files}
    else:
        files = [root]
        rel_of = {root: Path(".")}

    return [
        Clip(
            path=p,
            rel_dir=rel_of[p],
            layout=clip_layout(save_path, skeleton.name, robot, rel_of[p], p.stem),
        )
        for p in files
    ]


def resolve_target(args: argparse.Namespace) -> tuple[Config, Skeleton, list[Clip]]:
    """解析出机器人配置、源骨架，以及待处理的片段列表。"""
    cfg = load_config(args.robot)
    skeleton = get_skeleton(args.human)
    return cfg, skeleton, discover_clips(args, skeleton)


__all__ = [
    "Clip",
    "DEFAULT_SAVE_PATH",
    "add_target_args",
    "discover_clips",
    "resolve_path",
    "resolve_save_path",
    "resolve_target",
]
