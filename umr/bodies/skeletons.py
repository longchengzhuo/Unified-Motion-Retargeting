"""源动捕骨架定义：把不同厂商的 BVH 关节命名统一映射到 UMR 的分段标签。

目前支持两种源：

``xsens``
    Xsens MVN 段骨架，23 个关节，脊柱是 ``Chest/Chest2/Chest3/Chest4``，
    手臂是 ``LeftCollar/LeftShoulder/LeftElbow/LeftWrist``，腿是
    ``LeftHip/LeftKnee/LeftAnkle/LeftToe``。

``fzmotion``
    FZMotion 导出的 Mixamo 风格骨架，57 个关节（含 30 个手指），脊柱是
    ``Spine1/Spine2/Chest``，手臂是 ``LeftShoulder/LeftArm/LeftForeArm/LeftHand``，
    腿是 ``LeftUpLeg/LeftLeg/LeftFoot/LeftToe``，且末端用真实关节
    （``HeadEnd`` / ``LToeEnd`` / ``LeftHandPalm``）而不是 End Site。

两套骨架产出**同一组分段标签**（:data:`umr.bodies.human_mjcf.ALL_SEGMENTS`），
所以下游的分段权重、测地图邻接、对应学习与残差全部不用改——这正是论文"统一接口"
主张在源端的体现。
"""

from __future__ import annotations

from dataclasses import dataclass, field


#: 表面几何 spec 的字段：
#:   seg   : 分段标签，用于残差权重与对应关系着色（机器人点通过学到的对应继承它）
#:   kind  : capsule | ellipsoid | box
#:   axis  : 用于定义骨骼轴向的子关节名；"END" 表示使用 End Site
#:   radius: capsule 半径（参考身高 1.8 m 下，米）
#:   size  : ellipsoid 的三个半轴 / box 的 (沿轴向额外留量, 半宽, 半高)
#:   along : 几何中心沿骨骼轴向的比例位置


@dataclass(frozen=True)
class Skeleton:
    """一种源动捕骨架的命名约定与表面几何定义。

    Attributes:
        name: 源标识，即命令行 ``--human`` 的取值。
        segments: 关节名 -> 表面几何 spec，字段含义见本模块下方的注释。
        foot_chains: ``(踝, 趾)`` 关节名对，用于从 T-pose 估计角色朝向。
        hip_pair: 踝趾向量退化时改用的髋部连线。
        strip_prefix: 是否按 ``name.split("_")[-1]`` 去掉导出器加的前缀
            （如 ``character1_Hips`` -> ``Hips``）。
        extensions: 目录批处理时要收集的文件后缀。
    """

    name: str
    segments: dict[str, dict]
    foot_chains: tuple[tuple[str, str], ...]
    hip_pair: tuple[str, str]
    strip_prefix: bool = False
    extensions: tuple[str, ...] = field(default=(".bvh",))


def _limb_specs(
    *,
    clavicle: str, upperarm: str, forearm: str, hand: str, hand_axis: str,
    thigh: str, shin: str, foot: str, toe: str, toe_axis: str,
    side: str,
) -> dict[str, dict]:
    """一侧肢体的几何 spec。两套骨架只是关节名不同，尺寸取值共用。"""
    s = side[0].lower()  # "l" / "r"
    return {
        clavicle: dict(seg=f"{s}_clavicle", kind="capsule", axis=upperarm, radius=0.056),
        upperarm: dict(seg=f"{s}_upperarm", kind="capsule", axis=forearm, radius=0.052),
        forearm: dict(seg=f"{s}_forearm", kind="capsule", axis=hand, radius=0.044),
        hand: dict(seg=f"{s}_hand", kind="capsule", axis=hand_axis, radius=0.038),
        thigh: dict(seg=f"{s}_thigh", kind="capsule", axis=shin, radius=0.086),
        shin: dict(seg=f"{s}_shin", kind="capsule", axis=foot, radius=0.060),
        foot: dict(seg=f"{s}_foot", kind="box", axis=toe, size=(0.010, 0.045, 0.032)),
        toe: dict(seg=f"{s}_toe", kind="box", axis=toe_axis, size=(0.005, 0.042, 0.022)),
    }


# ----------------------------------------------------------------------
# Xsens MVN
# ----------------------------------------------------------------------
_XSENS_SEGMENTS: dict[str, dict] = {
    "Hips": dict(seg="pelvis", kind="ellipsoid", axis="Chest", size=(0.115, 0.150, 0.120), along=0.35),
    "Chest": dict(seg="torso", kind="capsule", axis="Chest2", radius=0.128),
    "Chest2": dict(seg="torso", kind="capsule", axis="Chest3", radius=0.136),
    "Chest3": dict(seg="torso", kind="capsule", axis="Chest4", radius=0.142),
    "Chest4": dict(seg="chest", kind="ellipsoid", axis="Neck", size=(0.115, 0.185, 0.135), along=0.30),
    "Neck": dict(seg="neck", kind="capsule", axis="Head", radius=0.055),
    "Head": dict(seg="head", kind="ellipsoid", axis="END", size=(0.098, 0.088, 0.115), along=0.55),
}
for _side in ("Left", "Right"):
    _XSENS_SEGMENTS.update(_limb_specs(
        clavicle=f"{_side}Collar", upperarm=f"{_side}Shoulder",
        forearm=f"{_side}Elbow", hand=f"{_side}Wrist", hand_axis="END",
        thigh=f"{_side}Hip", shin=f"{_side}Knee",
        foot=f"{_side}Ankle", toe=f"{_side}Toe", toe_axis="END",
        side=_side,
    ))

XSENS = Skeleton(
    name="xsens",
    segments=_XSENS_SEGMENTS,
    foot_chains=(("LeftAnkle", "LeftToe"), ("RightAnkle", "RightToe")),
    hip_pair=("LeftHip", "RightHip"),
)


# ----------------------------------------------------------------------
# FZMotion
# ----------------------------------------------------------------------
# 脊柱只有两节（xsens 有三节），单节因此更长；半径取 xsens 三节的中间值，
# 保证躯干粗细与 xsens 一致。
_FZMOTION_SEGMENTS: dict[str, dict] = {
    "Hips": dict(seg="pelvis", kind="ellipsoid", axis="Spine1", size=(0.115, 0.150, 0.120), along=0.35),
    "Spine1": dict(seg="torso", kind="capsule", axis="Spine2", radius=0.130),
    "Spine2": dict(seg="torso", kind="capsule", axis="Chest", radius=0.140),
    "Chest": dict(seg="chest", kind="ellipsoid", axis="Neck", size=(0.115, 0.185, 0.135), along=0.30),
    "Neck": dict(seg="neck", kind="capsule", axis="Head", radius=0.055),
    "Head": dict(seg="head", kind="ellipsoid", axis="HeadEnd", size=(0.098, 0.088, 0.115), along=0.55),
}
for _side, _abbr in (("Left", "L"), ("Right", "R")):
    _FZMOTION_SEGMENTS.update(_limb_specs(
        clavicle=f"{_side}Shoulder", upperarm=f"{_side}Arm",
        forearm=f"{_side}ForeArm", hand=f"{_side}Hand", hand_axis=f"{_side}HandPalm",
        thigh=f"{_side}UpLeg", shin=f"{_side}Leg",
        foot=f"{_side}Foot", toe=f"{_side}Toe", toe_axis=f"{_abbr}ToeEnd",
        side=_side,
    ))

FZMOTION = Skeleton(
    name="fzmotion",
    segments=_FZMOTION_SEGMENTS,
    foot_chains=(("LeftFoot", "LeftToe"), ("RightFoot", "RightToe")),
    hip_pair=("LeftUpLeg", "RightUpLeg"),
    # FZMotion 本身不带前缀，但同族导出器会把关节名写成 "character1_Hips"。
    strip_prefix=True,
)


SKELETONS: dict[str, Skeleton] = {s.name: s for s in (XSENS, FZMOTION)}
DEFAULT_HUMAN = XSENS.name


def get_skeleton(human: str) -> Skeleton:
    try:
        return SKELETONS[str(human).lower()]
    except KeyError:
        known = " / ".join(sorted(SKELETONS))
        raise ValueError(f"未知的源骨架 {human!r}，可用: {known}") from None
