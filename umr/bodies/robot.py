"""机器人的 MuJoCo/mink 封装，以及批量 Jacobian 工具。

Stage II 的性能关键：**不逐点调用 Jacobian**。mink 的
``Configuration.get_frame_jacobian(name, "body")`` 返回 body 局部系的 6xnv
Jacobian；把它转回世界系后，同一 body 上任意点的 Jacobian 都能由刚体运动学
向量化推出::

    J_point  = jacp_w - skew(Δ) @ jacr_w        # Δ = R_wb @ p_local
    J_normal = -skew(n_w) @ jacr_w              # n_w = R_wb @ n_local

于是每次 Gauss-Newton 迭代只需 O(nbody) 次 Jacobian 调用，而不是 O(npoints) 次。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RobotSpec:
    """一台机器人的形态约定，对应配置里的 ``robot`` 段。

    默认值取通行的人形命名，配置里不写这些键也能跑。换机型时在 yaml 里覆盖需要改
    的那几项即可，不必动代码。
    """

    #: canonical T-pose 的关节角，其余关节取 0。注意"零位即伸直"不是通例：G1 的肘
    #: 角为 0 时前臂垂直于上臂，只设肩 roll 会得到一个前倾 46 度、肩到腕仅 0.28 m
    #: 的假 T-pose；肘角取 +90 度才真正伸直（前倾 0.9 度，0.37 m）。
    tpose_joints: dict[str, float] = field(
        default_factory=lambda: {
            "left_shoulder_roll_joint": 1.5708,
            "right_shoulder_roll_joint": -1.5708,
        }
    )
    #: 这些 body 上的 geom 只是标记点，不属于机器人表面。有的 MJCF 会把接触点、
    #: 抓取点单独挂成 body，采表面时必须排除掉，否则会采到悬空的小球上。
    marker_body_prefixes: tuple[str, ...] = ()
    marker_body_suffixes: tuple[str, ...] = ()
    #: 求 robot_foot_height 用的踝 link。
    foot_bodies: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
    #: 找足底几何时匹配的 body 名关键字（见 :func:`sole_sample_points`）。
    foot_name_keys: tuple[str, ...] = ("ankle", "foot")

    @classmethod
    def from_config(cls, robot_cfg: Mapping[str, Any]) -> RobotSpec:
        """从配置的 ``robot`` 段构造，没写的键沿用默认值。"""
        kwargs: dict[str, Any] = {}
        if "tpose_joints" in robot_cfg:
            kwargs["tpose_joints"] = {
                str(k): float(v) for k, v in robot_cfg["tpose_joints"].items()
            }
        for key in (
            "marker_body_prefixes",
            "marker_body_suffixes",
            "foot_bodies",
            "foot_name_keys",
        ):
            if key in robot_cfg:
                kwargs[key] = tuple(str(v) for v in robot_cfg[key])
        return cls(**kwargs)


#: 不给 spec 时的兜底。
DEFAULT_SPEC = RobotSpec()


def skew(v: np.ndarray) -> np.ndarray:
    """反对称矩阵，支持批量输入 (..., 3) -> (..., 3, 3)。"""
    v = np.asarray(v)
    z = np.zeros(v.shape[:-1] + (3, 3))
    z[..., 0, 1] = -v[..., 2]
    z[..., 0, 2] = v[..., 1]
    z[..., 1, 0] = v[..., 2]
    z[..., 1, 2] = -v[..., 0]
    z[..., 2, 0] = -v[..., 1]
    z[..., 2, 1] = v[..., 0]
    return z


def is_marker_body(name: str, spec: RobotSpec = DEFAULT_SPEC) -> bool:
    return name.startswith(spec.marker_body_prefixes) or name.endswith(
        spec.marker_body_suffixes
    )


def tpose_qpos(
    model: mujoco.MjModel, ground: bool = True, spec: RobotSpec = DEFAULT_SPEC
) -> np.ndarray:
    """构造机器人的 canonical T-pose qpos，并把脚底贴到 z=0。"""
    q = np.zeros(model.nq)
    q[3] = 1.0
    for name, val in spec.tpose_joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"机器人模型缺少关节 {name}")
        lo, hi = model.jnt_range[jid]
        q[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))
    if ground:
        data = mujoco.MjData(model)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        q[2] = -lowest_surface_z(model, data, spec)
    return q


def lowest_surface_z(
    model: mujoco.MjModel, data: mujoco.MjData, spec: RobotSpec = DEFAULT_SPEC
) -> float:
    """当前姿态下机器人表面（不含世界几何与标记点）的最低 z。"""
    lows = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if is_marker_body(bname, spec):
            continue
        lows.append(geom_lowest_z(model, data, g))
    return float(min(lows))


def geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> float:
    """单个 geom 的世界最低 z（按其包围体保守估计）。"""
    R = data.geom_xmat[g].reshape(3, 3)
    c = data.geom_xpos[g]
    gtype = int(model.geom_type[g])
    size = model.geom_size[g]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(c[2] - size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(c[2] - abs(R[2, 2]) * size[1] - size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(c[2] - np.abs(R[2, :3] * size[:3]).sum())
    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(c[2] - np.linalg.norm(R[2, :3] * size[:3]))
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        v0 = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[v0 : v0 + n].reshape(-1, 3)
        world = verts @ R.T + c
        return float(world[:, 2].min())
    return float(c[2])


def sole_sample_points(
    model: mujoco.MjModel, name_keys: tuple[str, ...] = ("ankle", "foot")
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """足底的极值几何点，用于施加式 (14) 的地面净空约束。

    论文式 (14) 里的 :math:`F_t` 是"靠近地面的机器人**表面点**"。只用学到的对应点
    并不可靠：足底最低处未必恰好被采样到，结果就是脚陷进地面。这里改用真正决定
    足底高度的几何——box geom 的 8 个角点与 sphere geom 的球心（配一个半径偏移）。

    Returns:
        ``(body_ids, local_pos, radii)``，其中约束按 ``z(q) - radius >= z_f`` 施加。
    """
    body_ids: list[int] = []
    local_pos: list[np.ndarray] = []
    radii: list[float] = []

    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not any(k in bname for k in name_keys):
            continue
        gtype = int(model.geom_type[g])
        size = model.geom_size[g]
        R = Rotation.from_quat(model.geom_quat[g], scalar_first=True).as_matrix()
        p = model.geom_pos[g]

        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            pts = p + (signs * size[:3]) @ R.T
            r = 0.0
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts = p[None, :]
            r = float(size[0])
        else:
            continue
        for q in pts:
            body_ids.append(b)
            local_pos.append(q)
            radii.append(r)

    if not body_ids:
        raise ValueError("机器人模型中找不到足底几何")
    return (
        np.asarray(body_ids, dtype=np.int64),
        np.asarray(local_pos, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
    )


def prepare_robot_xml(
    src_xml: str | Path,
    dst_xml: str | Path | None = None,
    offwidth: int = 1920,
    offheight: int = 1080,
    spec: RobotSpec = DEFAULT_SPEC,
) -> Path:
    """把机器人 MJCF 准备成 UMR 可用的形式。

    1. 追加 ``T_pose`` keyframe，之后可用 ``configuration.update_from_keyframe("T_pose")``；
    2. 放大离屏帧缓冲，使高分辨率渲染可用。

    Returns:
        写出的 XML 路径（``dst_xml`` 为 None 时原地修改 ``src_xml``）。
    """
    src_xml = Path(src_xml)
    dst_xml = Path(dst_xml) if dst_xml is not None else src_xml

    model = mujoco.MjModel.from_xml_path(str(src_xml))
    q = tpose_qpos(model, spec=spec)

    tree = ET.parse(src_xml)
    root = tree.getroot()

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(offwidth))
    glob.set("offheight", str(offheight))

    for kf in root.findall("keyframe"):
        root.remove(kf)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="T_pose", qpos=" ".join(f"{v:.8f}" for v in q))

    ET.indent(root, space="  ")
    tree.write(dst_xml, encoding="UTF-8", xml_declaration=True)
    return dst_xml


@dataclass
class BodyJacobians:
    """一次迭代中缓存的按 body 的世界系 Jacobian。"""

    jacp: np.ndarray  # (nbody, 3, nv)
    jacr: np.ndarray  # (nbody, 3, nv)
    xpos: np.ndarray  # (nbody, 3)
    xmat: np.ndarray  # (nbody, 3, 3)


def body_jacobians(
    configuration: "mink.Configuration", body_names: list[str], body_ids: np.ndarray
) -> BodyJacobians:
    """对给定 body 计算世界系的平移/旋转 Jacobian。

    每个 body 只调用一次 mink 的 ``get_frame_jacobian``。mink 返回的是 body 局部系
    的 6xnv Jacobian（``blkdiag(Rᵀ, Rᵀ) @ [jacp; jacr]``），左乘 R 即还原成世界系。
    """
    data = configuration.data
    nv = configuration.model.nv
    n = len(body_ids)
    jacp = np.zeros((n, 3, nv))
    jacr = np.zeros((n, 3, nv))
    xpos = np.zeros((n, 3))
    xmat = np.zeros((n, 3, 3))
    for k, b in enumerate(body_ids):
        b = int(b)
        R = data.xmat[b].reshape(3, 3)
        J = configuration.get_frame_jacobian(body_names[b], "body")
        jacp[k] = R @ J[:3]
        jacr[k] = R @ J[3:]
        xpos[k] = data.xpos[b]
        xmat[k] = R
    return BodyJacobians(jacp=jacp, jacr=jacr, xpos=xpos, xmat=xmat)


def point_kinematics(
    body_index: np.ndarray,
    local_pos: np.ndarray,
    local_normal: np.ndarray | None,
    jac: BodyJacobians,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """由缓存的 body Jacobian 向量化求出点的位置/法线及其 Jacobian。

    .. math::

        J_{point}  &= jac_p - [\\Delta]_\\times jac_r, \\quad \\Delta = R_{wb} p_{local} \\\\
        J_{normal} &= -[n_w]_\\times jac_r, \\quad n_w = R_{wb} n_{local}

    Args:
        body_index: (P,) 每个点在 ``jac`` 中对应的行号。
        local_pos: (P, 3) body 局部坐标。
        local_normal: (P, 3) body 局部法线，None 表示不需要法线。
        jac: :func:`body_jacobians` 的结果。

    Returns:
        ``(pos_w, nrm_w, Jp, Jn)``，形状 (P,3)、(P,3)、(P,3,nv)、(P,3,nv)。
    """
    R = jac.xmat[body_index]
    delta = np.einsum("pij,pj->pi", R, local_pos)
    pos_w = jac.xpos[body_index] + delta

    jacp = jac.jacp[body_index]
    jacr = jac.jacr[body_index]
    Jp = jacp - np.einsum("pij,pjk->pik", skew(delta), jacr)

    nrm_w = None
    Jn = None
    if local_normal is not None:
        nrm_w = np.einsum("pij,pj->pi", R, local_normal)
        Jn = -np.einsum("pij,pjk->pik", skew(nrm_w), jacr)
    return pos_w, nrm_w, Jp, Jn


class RobotBody:
    """机器人的 mink 封装。"""

    def __init__(self, xml_path: str | Path, spec: RobotSpec = DEFAULT_SPEC):
        self.xml_path = str(xml_path)
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.configuration = mink.Configuration(self.model)
        self.nv = int(self.model.nv)
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            for b in range(self.model.nbody)
        ]
        self.set_tpose()

    @property
    def data(self):
        return self.configuration.data

    @property
    def q(self) -> np.ndarray:
        return self.configuration.q

    def set_tpose(self) -> None:
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "T_pose")
        if key >= 0:
            self.configuration.update_from_keyframe("T_pose")
        else:
            self.configuration.update(q=tpose_qpos(self.model, spec=self.spec))

    def set_qpos(self, q: np.ndarray) -> None:
        self.configuration.update(q=np.asarray(q, dtype=np.float64))

    # ------------------------------------------------------------------
    # 批量 Jacobian（薄封装，实现见模块级函数）
    # ------------------------------------------------------------------
    def body_jacobians(self, body_ids: np.ndarray) -> BodyJacobians:
        return body_jacobians(self.configuration, self.body_names, body_ids)

    def point_kinematics(
        self,
        body_index: np.ndarray,
        local_pos: np.ndarray,
        local_normal: np.ndarray | None,
        jac: BodyJacobians,
    ):
        return point_kinematics(body_index, local_pos, local_normal, jac)

    # ------------------------------------------------------------------
    def surface_geoms(self) -> list[int]:
        """参与表面采样的 geom。

        取 visual 组（group==1，代表真实外形）加上所有基元 geom（脚底的 box/球在
        collision 组里，但它们才是真正的足底表面），并排除标记点 body。
        """
        m = self.model
        out = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b == 0 or is_marker_body(self.body_names[b], self.spec):
                continue
            is_visual = int(m.geom_group[g]) == 1
            is_primitive = int(m.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH
            if is_visual or is_primitive:
                out.append(g)
        return out

    def height(self) -> float:
        """T-pose 下机器人从脚底到头顶的高度。"""
        q_save = self.q
        self.set_tpose()
        data = self.data
        low = lowest_surface_z(self.model, data, self.spec)
        highs = []
        for g in range(self.model.ngeom):
            b = int(self.model.geom_bodyid[g])
            if b == 0 or is_marker_body(self.body_names[b], self.spec):
                continue
            R = data.geom_xmat[g].reshape(3, 3)
            c = data.geom_xpos[g]
            gtype = int(self.model.geom_type[g])
            size = self.model.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(self.model.geom_dataid[g])
                v0 = int(self.model.mesh_vertadr[mid])
                n = int(self.model.mesh_vertnum[mid])
                verts = self.model.mesh_vert[v0 : v0 + n].reshape(-1, 3)
                highs.append(float((verts @ R.T + c)[:, 2].max()))
            else:
                highs.append(float(c[2] + np.abs(R[2, :3] * size[:3]).sum()))
        self.set_qpos(q_save)
        return float(max(highs) - low)

    def foot_height(self) -> float:
        """脚踝 roll link 原点到脚底的距离（论文的 robot_foot_height）。"""
        q_save = self.q
        self.set_tpose()
        data = self.data
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.spec.foot_bodies[0])
        ankle_z = float(data.xpos[bid][2])
        low = lowest_surface_z(self.model, data, self.spec)
        self.set_qpos(q_save)
        return ankle_z - low

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """返回按 qpos 索引的关节上下限（free joint 部分为 +-inf）。"""
        lo = np.full(self.model.nq, -np.inf)
        hi = np.full(self.model.nq, np.inf)
        for j in range(self.model.njnt):
            if not self.model.jnt_limited[j]:
                continue
            adr = int(self.model.jnt_qposadr[j])
            lo[adr], hi[adr] = self.model.jnt_range[j]
        return lo, hi
