"""项目内的固定路径与产物布局。

产物分成两层，因为两者的复用粒度完全不同：

**Setup（Stage 0 / I）** —— 人机 MJCF、T-pose 表面点云、学到的点云对应。它只取决于
「机器人 + 源骨架 + 演员骨架尺寸」，跟具体是哪一段动作无关。同一个演员的几十段
BVH 共用一套，所以放在 ``<save_path>/.setup/<human>_to_<robot>/<签名>/`` 下按骨架
签名寻址。Stage I 在 CPU 上要跑十来分钟，批量处理一个目录时这一层复用是决定性的。

**Clip（Stage II）** —— 逐帧关节角与评估产物，每段动作一份：
``<save_path>/<human>_to_<robot>/<相对目录>/<片段名>.pkl``（视频是同名 ``.mp4``），
中间件收在同名子目录里，免得把输出根目录弄乱。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSET_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

ROBOT_DIR = ASSET_DIR / "robots"


@dataclass(frozen=True)
class SetupLayout:
    """Stage 0 / I 的产物，可被同一演员的多段动作共用。"""

    root: Path

    @property
    def human_xml(self) -> Path:
        return self.root / "human.xml"

    @property
    def bodies(self) -> Path:
        return self.root / "bodies.npz"

    @property
    def correspondence(self) -> Path:
        return self.root / "correspondence.npz"

    @property
    def corr_image(self) -> Path:
        return self.root / "correspondence.png"

    def ensure(self) -> "SetupLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class ClipLayout:
    """单段动作的 Stage II 产物。

    ``stem`` 是不带后缀的主产物路径，``.pkl`` / ``.mp4`` 与它同级同名，其余中间件
    在同名目录 ``work`` 里。
    """

    stem: Path

    def _sibling(self, ext: str) -> Path:
        """与 ``stem`` 同级的产物文件。

        这里刻意不用 ``Path.with_suffix``：``take_006.bvh_actor1`` 这类文件名会被它
        当成后缀是 ``.bvh_actor1``，于是同一 take 的两个演员被写进同一个
        ``take_006.pkl``，后跑的悄悄覆盖先跑的。有些动捕导出器就是这么命名的。
        """
        return self.stem.parent / f"{self.stem.name}{ext}"

    @property
    def motion_pkl(self) -> Path:
        return self._sibling(".pkl")

    @property
    def video(self) -> Path:
        return self._sibling(".mp4")

    @property
    def work(self) -> Path:
        return self.stem

    @property
    def motion(self) -> Path:
        return self.work / "motion.npz"

    @property
    def report(self) -> Path:
        return self.work / "report.md"

    @property
    def metrics(self) -> Path:
        return self.work / "metrics.npz"

    def ensure(self) -> "ClipLayout":
        self.work.mkdir(parents=True, exist_ok=True)
        return self


def pair_name(human: str, robot: str) -> str:
    return f"{human}_to_{robot}"


def setup_layout(save_path: str | Path, human: str, robot: str, signature: str) -> SetupLayout:
    """按骨架签名寻址的 Stage 0 / I 产物目录。"""
    return SetupLayout(Path(save_path) / ".setup" / pair_name(human, robot) / signature)


def clip_layout(
    save_path: str | Path, human: str, robot: str, rel_dir: str | Path, stem: str
) -> ClipLayout:
    """``<save_path>/<human>_to_<robot>/<相对目录>/<片段名>`` 的产物布局。"""
    return ClipLayout(Path(save_path) / pair_name(human, robot) / Path(rel_dir) / stem)
