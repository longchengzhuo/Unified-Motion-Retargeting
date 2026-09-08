"""由 BVH 层级程序化生成人体 MJCF，并用 mink.Configuration 驱动。

论文 III-A 把 "rigged humanoid characters" 列为合法的源表示。BVH 只有骨架，
所以这里按骨架 offset 的真实尺寸给每根骨骼套一个胶囊/椭球/盒体，得到一个
刚性蒙皮的人体表面。每个 geom 刚性附着在一个 body 上，因此逐帧的表面点由
前向运动学直接搬运 —— 等价于论文对可形变网格所用的 barycentric transport。

人体与机器人由此共享完全相同的运动学机制（MuJoCo FK + mink.Configuration），
正是论文所主张的"统一接口"。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from umr.bodies.bvh import BvhData
from umr.bodies.skeletons import DEFAULT_HUMAN, Skeleton, get_skeleton

# 各分段几何参数以身高 1.8 m 为参考，实际按演员身高线性缩放。
HEIGHT_REF = 1.8

# 每个 BVH 关节的表面几何定义按源骨架分别给出，见 umr.bodies.skeletons。

#: 分段之间的相邻关系（用于分段感知测地图，避免左右腿贴近时误连边）。
SEGMENT_ADJACENCY: list[tuple[str, str]] = [
    ("pelvis", "torso"), ("torso", "chest"), ("chest", "neck"), ("neck", "head"),
    ("chest", "l_clavicle"), ("l_clavicle", "l_upperarm"), ("l_upperarm", "l_forearm"), ("l_forearm", "l_hand"),
    ("chest", "r_clavicle"), ("r_clavicle", "r_upperarm"), ("r_upperarm", "r_forearm"), ("r_forearm", "r_hand"),
    ("pelvis", "l_thigh"), ("l_thigh", "l_shin"), ("l_shin", "l_foot"), ("l_foot", "l_toe"),
    ("pelvis", "r_thigh"), ("r_thigh", "r_shin"), ("r_shin", "r_foot"), ("r_foot", "r_toe"),
]

#: 论文式 (7) 的分段权重 w^p_i / w^n_i。躯干/骨盆决定整体姿态，末端决定细节。
SEGMENT_POSITION_WEIGHT: dict[str, float] = {
    "pelvis": 3.0, "torso": 1.5, "chest": 2.0, "neck": 0.6, "head": 0.8,
    "l_clavicle": 0.6, "l_upperarm": 1.2, "l_forearm": 1.2, "l_hand": 1.5,
    "r_clavicle": 0.6, "r_upperarm": 1.2, "r_forearm": 1.2, "r_hand": 1.5,
    "l_thigh": 1.5, "l_shin": 1.5, "l_foot": 3.0, "l_toe": 2.0,
    "r_thigh": 1.5, "r_shin": 1.5, "r_foot": 3.0, "r_toe": 2.0,
}

SEGMENT_NORMAL_WEIGHT: dict[str, float] = {
    "pelvis": 0.5, "torso": 0.3, "chest": 0.4, "neck": 0.1, "head": 0.2,
    "l_clavicle": 0.1, "l_upperarm": 0.25, "l_forearm": 0.25, "l_hand": 0.3,
    "r_clavicle": 0.1, "r_upperarm": 0.25, "r_forearm": 0.25, "r_hand": 0.3,
    "l_thigh": 0.3, "l_shin": 0.3, "l_foot": 0.6, "l_toe": 0.4,
    "r_thigh": 0.3, "r_shin": 0.3, "r_foot": 0.6, "r_toe": 0.4,
}

ALL_SEGMENTS: list[str] = list(SEGMENT_POSITION_WEIGHT.keys())
SEGMENT_TO_ID: dict[str, int] = {s: i for i, s in enumerate(ALL_SEGMENTS)}


def body_name(joint: str) -> str:
    return f"h_{joint}"


def _quat_align(local_axis: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """返回把 ``local_axis`` 旋到 ``direction`` 的四元数 (wxyz)。"""
    a = local_axis / np.linalg.norm(local_axis)
    b = np.asarray(direction, dtype=np.float64)
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    b = b / nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        # 反向：绕任意垂直轴转 180 度
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return np.concatenate([[0.0], axis])
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1, 1))).as_quat(
        scalar_first=True
    )


@dataclass
class HumanModelInfo:
    """生成人体 MJCF 时产生的元数据。"""

    xml_path: str
    scale: float
    actor_height: float
    joint_names: list[str]
    body_names: list[str]
    #: geom 名 -> 分段标签
    geom_segment: dict[str, str] = field(default_factory=dict)


def _axis_vector(bvh: BvhData, joint_idx: int, spec: dict, scale: float) -> np.ndarray:
    """骨骼轴向：从本关节指向 ``spec['axis']`` 指定的子关节 / End Site。"""
    axis = spec["axis"]
    if axis == "END":
        off = bvh.end_sites.get(joint_idx)
        if off is None:
            return np.array([0.0, 0.0, 0.05])
        return np.asarray(off) * scale
    child = bvh.index(axis)
    return bvh.offsets[child] * scale


def build_human_mjcf(
    bvh: BvhData,
    scale: float = 1.0,
    model_name: str = "umr_human",
    human: str | Skeleton = DEFAULT_HUMAN,
) -> tuple[str, HumanModelInfo]:
    """由 BVH 层级生成人体 MJCF 字符串。

    Args:
        bvh: 已转换到 MuJoCo 坐标系的 BVH 数据。
        scale: 整体缩放（用于把演员归一化到机器人身高）。
        model_name: MJCF 模型名。
        human: 源骨架标识或 :class:`~umr.bodies.skeletons.Skeleton`，决定关节名
            到表面几何的映射。

    Returns:
        ``(xml_string, info)``。
    """
    from umr.bodies.bvh import actor_height as _actor_height

    skeleton = human if isinstance(human, Skeleton) else get_skeleton(human)
    segment_specs = skeleton.segments
    missing = [j for j in segment_specs if j not in bvh.names]
    if missing:
        raise ValueError(
            f"BVH 不含 {skeleton.name} 骨架的关节 {missing[:5]}（共 {len(missing)} 个）。"
            f"检查 --human 是否选对，文件实际关节: {bvh.names[:8]}…"
        )

    h_actor = _actor_height(bvh)
    geom_scale = (h_actor * scale) / HEIGHT_REF

    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")

    default = ET.SubElement(root, "default")
    # 人体模型只用于运动学，不参与碰撞。
    ET.SubElement(
        default, "geom",
        contype="0", conaffinity="0", group="2", density="985",
        rgba="0.85 0.66 0.55 1",
    )

    worldbody = ET.SubElement(root, "worldbody")

    geom_segment: dict[str, str] = {}
    body_elems: dict[int, ET.Element] = {}
    body_names: list[str] = []
    geomless: list[ET.Element] = []

    for j, jname in enumerate(bvh.names):
        parent = int(bvh.parents[j])
        if parent < 0:
            elem = ET.SubElement(worldbody, "body", name=body_name(jname), pos="0 0 0")
            ET.SubElement(elem, "freejoint", name="h_root")
        else:
            pos = bvh.offsets[j] * scale
            elem = ET.SubElement(
                body_elems[parent], "body",
                name=body_name(jname),
                pos=" ".join(f"{v:.6f}" for v in pos),
            )
            ET.SubElement(elem, "joint", name=f"h_{jname}", type="ball", damping="0.1")
        body_elems[j] = elem
        body_names.append(body_name(jname))

        spec = segment_specs.get(jname)
        if spec is None:
            geomless.append(elem)
            continue
        seg = spec["seg"]
        d = _axis_vector(bvh, j, spec, scale)
        L = float(np.linalg.norm(d))
        # 多个关节可能共享同一分段（如 Chest/Chest2/Chest3 都属于 torso），
        # 因此 geom 名按关节唯一化，分段标签另存映射。
        gname = f"hg_{jname}"
        geom_segment[gname] = seg

        if spec["kind"] == "capsule":
            r = spec["radius"] * geom_scale
            if L < 1e-4:
                ET.SubElement(
                    elem, "geom", name=gname, type="sphere",
                    size=f"{r:.6f}", pos="0 0 0",
                )
            else:
                ET.SubElement(
                    elem, "geom", name=gname, type="capsule",
                    size=f"{r:.6f}",
                    fromto=" ".join(f"{v:.6f}" for v in np.concatenate([np.zeros(3), d])),
                )
        elif spec["kind"] == "ellipsoid":
            size = np.asarray(spec["size"]) * geom_scale
            center = d * spec.get("along", 0.5)
            # 椭球的局部 Z 轴对齐骨骼轴向
            quat = _quat_align(np.array([0.0, 0.0, 1.0]), d)
            ET.SubElement(
                elem, "geom", name=gname, type="ellipsoid",
                size=" ".join(f"{v:.6f}" for v in size),
                pos=" ".join(f"{v:.6f}" for v in center),
                quat=" ".join(f"{v:.6f}" for v in quat),
            )
        elif spec["kind"] == "box":
            pad, hw, hh = np.asarray(spec["size"]) * geom_scale
            half_len = max(L * 0.5 + pad, 1e-3)
            center = d * 0.5
            quat = _quat_align(np.array([1.0, 0.0, 0.0]), d)
            ET.SubElement(
                elem, "geom", name=gname, type="box",
                size=f"{half_len:.6f} {hw:.6f} {hh:.6f}",
                pos=" ".join(f"{v:.6f}" for v in center),
                quat=" ".join(f"{v:.6f}" for v in quat),
            )
        else:
            raise ValueError(f"未知几何类型: {spec['kind']}")

    # 没有表面几何的关节（FZMotion 的手指与 *End 末端节点）质量为零，而 MuJoCo
    # 要求带关节的 body 质量大于 mjMINVAL。人体模型只跑 FK、不参与动力学，给个
    # 名义惯量即可。
    for elem in geomless:
        elem.insert(0, ET.Element(
            "inertial", pos="0 0 0", mass="1e-6", diaginertia="1e-9 1e-9 1e-9"
        ))

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")

    info = HumanModelInfo(
        xml_path="",
        scale=scale,
        actor_height=h_actor,
        joint_names=list(bvh.names),
        body_names=body_names,
        geom_segment=geom_segment,
    )
    return xml, info


def write_human_mjcf(
    bvh: BvhData,
    out_path: str | Path,
    scale: float = 1.0,
    human: str | Skeleton = DEFAULT_HUMAN,
) -> HumanModelInfo:
    """生成并写出人体 MJCF 文件。"""
    xml, info = build_human_mjcf(bvh, scale=scale, human=human)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")
    info.xml_path = str(out_path)
    return info


class HumanBody:
    """人体 MJCF 的 mink 封装：把 BVH 逐帧姿态写进 qpos 并做 FK。"""

    def __init__(self, xml_path: str | Path, bvh: BvhData, scale: float = 1.0):
        import mink

        self.bvh = bvh
        self.scale = float(scale)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.configuration = mink.Configuration(self.model)

        # 每个 BVH 关节对应的 qpos 地址
        self._ball_adr: list[int] = []
        for j, name in enumerate(bvh.names):
            if j == 0:
                self._ball_adr.append(-1)
                continue
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"h_{name}")
            if jid < 0:
                raise ValueError(f"人体模型缺少关节 h_{name}")
            self._ball_adr.append(int(self.model.jnt_qposadr[jid]))

        self.body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name(n))
                for n in bvh.names
            ]
        )
        self.root_offset = np.zeros(3)

    @property
    def data(self):
        return self.configuration.data

    def qpos_for_frame(self, t: int) -> np.ndarray:
        """构造第 ``t`` 帧的 qpos。"""
        q = np.zeros(self.model.nq)
        q[0:3] = self.bvh.root_pos[t] * self.scale + self.root_offset
        q[3:7] = self.bvh.local_quat[t, 0]
        for j in range(1, self.bvh.num_joints):
            adr = self._ball_adr[j]
            q[adr : adr + 4] = self.bvh.local_quat[t, j]
        return q

    def set_frame(self, t: int) -> None:
        """把第 ``t`` 帧写入配置并运行 FK。"""
        self.configuration.update(q=self.qpos_for_frame(t))

    def set_tpose(self) -> None:
        """置为 canonical T-pose。

        BVH 第 0 帧所有通道为零，配合 offset 本身就是标准 T-pose，所以这里把全部
        局部旋转设为单位四元数，并把根放在原点上方的静止高度。
        """
        self.configuration.update(q=self.tpose_qpos())

    def tpose_qpos(self) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[3] = 1.0
        for j in range(1, self.bvh.num_joints):
            q[self._ball_adr[j]] = 1.0
        q[2] = float(self.bvh.root_pos[0][2] * self.scale + self.root_offset[2])
        return q


FOOT_SEGMENTS = ("l_foot", "r_foot", "l_toe", "r_toe")


def foot_geom_ids(model: mujoco.MjModel, geom_segment: dict[str, str]) -> list[int]:
    """人体模型中属于脚/脚趾分段的 geom。"""
    out = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if geom_segment.get(name) in FOOT_SEGMENTS:
            out.append(g)
    return out


def compute_ground_offset(
    human: HumanBody,
    frames: np.ndarray,
    geom_segment: dict[str, str],
    percentile: float = 1.0,
) -> float:
    """估计把动作贴到 z=0 地面所需的竖直平移量。

    取各帧脚部 geom 最低点的低分位数，避免个别穿透帧把整段动作抬起来。
    """
    from umr.bodies.robot import geom_lowest_z

    geoms = foot_geom_ids(human.model, geom_segment)
    if not geoms:
        raise ValueError("人体模型中找不到脚部 geom")
    lows = []
    for t in frames:
        human.set_frame(int(t))
        lows.append(min(geom_lowest_z(human.model, human.data, g) for g in geoms))
    return -float(np.percentile(lows, percentile))
