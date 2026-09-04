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

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RobotSpec:
    """一台机器人在 MJCF 之外还需要的少量语义信息。

    全部来自配置文件的 ``robot`` 段，因此换机器人不需要改代码。

    Attributes:
        name: 展示用名称。
        root_body: 浮动基所在的 body；留空则取世界下的第一个 body。
        tpose_joints: canonical T-pose 中需要偏置的关节角（其余为 0）。
        marker_body_prefixes: 纯标记 body 的名字前缀，其 geom 不算外表面。
        marker_body_suffixes: 同上，按后缀匹配。
        foot_name_keys: 判定"属于足部"的 body 名子串。
        foot_bodies: 左/右脚的末端 body；留空则按 ``foot_name_keys`` 自动推断。
    """

    name: str = "robot"
    root_body: str = ""
    tpose_joints: dict[str, float] = field(default_factory=dict)
    marker_body_prefixes: tuple[str, ...] = ()
    marker_body_suffixes: tuple[str, ...] = ()
    foot_name_keys: tuple[str, ...] = ("ankle", "foot")
    foot_bodies: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, robot_cfg: Mapping) -> "RobotSpec":
        return cls(
            name=str(robot_cfg.get("name", "robot")),
            root_body=str(robot_cfg.get("root_body", "") or ""),
            tpose_joints={str(k): float(v) for k, v in (robot_cfg.get("tpose_joints") or {}).items()},
            marker_body_prefixes=tuple(robot_cfg.get("marker_body_prefixes") or ()),
            marker_body_suffixes=tuple(robot_cfg.get("marker_body_suffixes") or ()),
            foot_name_keys=tuple(robot_cfg.get("foot_name_keys") or ("ankle", "foot")),
            foot_bodies=tuple(robot_cfg.get("foot_bodies") or ()),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "RobotSpec":
        return cls.from_config(json.loads(text))

    # ------------------------------------------------------------------
    def is_marker(self, body_name: str) -> bool:
        return body_name.startswith(self.marker_body_prefixes) or body_name.endswith(
            self.marker_body_suffixes
        )

    def resolve(self, model: mujoco.MjModel) -> "RobotSpec":
        """填上可以从模型本身推断出来的字段。"""
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            for b in range(model.nbody)
        ]
        root = self.root_body or (names[1] if model.nbody > 1 else "")
        if root not in names:
            raise ValueError(f"模型里没有 root_body={root!r}")

        feet = tuple(self.foot_bodies)
        if not feet:
            # 每侧取名字命中 foot_name_keys 且在运动链上最深的那个 body
            def deepest(side_keys: tuple[str, ...]) -> str | None:
                hits = [
                    n for n in names[1:]
                    if any(k in n for k in self.foot_name_keys)
                    and any(k in n for k in side_keys)
                    and not self.is_marker(n)
                ]
                return hits[-1] if hits else None

            feet = tuple(f for f in (deepest(("left", "_l_")), deepest(("right", "_r_"))) if f)
        missing = [f for f in feet if f not in names]
        if missing:
            raise ValueError(f"模型里没有足部 body {missing}")
        if not feet:
            raise ValueError(
                f"无法从 foot_name_keys={self.foot_name_keys} 推断足部 body，请在配置里写明 robot.foot_bodies"
            )
        return RobotSpec(
            name=self.name,
            root_body=root,
            tpose_joints=dict(self.tpose_joints),
            marker_body_prefixes=self.marker_body_prefixes,
            marker_body_suffixes=self.marker_body_suffixes,
            foot_name_keys=self.foot_name_keys,
            foot_bodies=feet,
        )


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


def tpose_qpos(model: mujoco.MjModel, spec: RobotSpec, ground: bool = True) -> np.ndarray:
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


def lowest_surface_z(model: mujoco.MjModel, data: mujoco.MjData, spec: RobotSpec) -> float:
    """当前姿态下机器人表面（不含世界几何与标记点）的最低 z。"""
    lows = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if spec.is_marker(bname):
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
    model: mujoco.MjModel, spec: RobotSpec
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
        if not any(k in bname for k in spec.foot_name_keys):
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
    spec: RobotSpec,
    dst_xml: str | Path | None = None,
    offwidth: int = 1920,
    offheight: int = 1080,
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
    q = tpose_qpos(model, spec.resolve(model))

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

    def __init__(self, xml_path: str | Path, spec: RobotSpec):
        self.xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.spec = spec.resolve(self.model)
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
            self.configuration.update(q=tpose_qpos(self.model, self.spec))

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
    def height(self) -> float:
        """T-pose 下机器人从脚底到头顶的高度。"""
        q_save = self.q
        self.set_tpose()
        data = self.data
        low = lowest_surface_z(self.model, data, self.spec)
        highs = []
        for g in range(self.model.ngeom):
            b = int(self.model.geom_bodyid[g])
            if b == 0 or self.spec.is_marker(self.body_names[b]):
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

    @staticmethod
    def from_bodies(bodies) -> "RobotBody":
        """从 ``scripts/01_build_bodies.py`` 写出的 ``bodies.npz`` 还原。"""
        return RobotBody(str(bodies["robot_xml"]), RobotSpec.from_json(str(bodies["robot_spec"])))

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
