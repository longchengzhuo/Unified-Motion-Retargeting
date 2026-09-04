"""统一的 MuJoCo 表面点云采样器（人体与机器人共用）。

论文把"外表面点云"作为人机之间的统一接口，因此采样器不区分来源：任何 MuJoCo
模型都被当成一组 geom，逐个转成 body 局部系的三角网格后统一采样。几何直接取自
``mjModel``（``mesh_vert``/``mesh_face``、以及 box/sphere/capsule/ellipsoid 的
``geom_size``），保证与 MJCF 中的位姿和缩放完全一致。

内点剔除用每个 geom 的**凸包**做包含性判断：MuJoCo 本身也把 mesh 当凸包处理，
而人体模型全部由凸基元构成，所以这个判据既快又与物理模型自洽。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


@dataclass
class SurfacePointCloud:
    """采样得到的外表面点云及其绑定信息。

    Attributes:
        points: (N, 3) 采样时姿态下的世界坐标。
        normals: (N, 3) 世界法线。
        body_ids: (N,) 所属 body。
        geom_ids: (N,) 所属 geom。
        local_pos: (N, 3) body 局部坐标（论文的 link-local binding）。
        local_normal: (N, 3) body 局部法线。
        segment: (N,) 分段 id，-1 表示未标注。
        segment_names: 分段 id -> 名称。
    """

    points: np.ndarray
    normals: np.ndarray
    body_ids: np.ndarray
    geom_ids: np.ndarray
    local_pos: np.ndarray
    local_normal: np.ndarray
    segment: np.ndarray
    segment_names: list[str]

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def subset(self, idx: np.ndarray) -> "SurfacePointCloud":
        return SurfacePointCloud(
            points=self.points[idx],
            normals=self.normals[idx],
            body_ids=self.body_ids[idx],
            geom_ids=self.geom_ids[idx],
            local_pos=self.local_pos[idx],
            local_normal=self.local_normal[idx],
            segment=self.segment[idx],
            segment_names=list(self.segment_names),
        )

    def to_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}points": self.points,
            f"{prefix}normals": self.normals,
            f"{prefix}body_ids": self.body_ids,
            f"{prefix}geom_ids": self.geom_ids,
            f"{prefix}local_pos": self.local_pos,
            f"{prefix}local_normal": self.local_normal,
            f"{prefix}segment": self.segment,
            f"{prefix}segment_names": np.array(self.segment_names, dtype=object),
        }

    @staticmethod
    def from_dict(d, prefix: str = "") -> "SurfacePointCloud":
        return SurfacePointCloud(
            points=d[f"{prefix}points"],
            normals=d[f"{prefix}normals"],
            body_ids=d[f"{prefix}body_ids"],
            geom_ids=d[f"{prefix}geom_ids"],
            local_pos=d[f"{prefix}local_pos"],
            local_normal=d[f"{prefix}local_normal"],
            segment=d[f"{prefix}segment"],
            segment_names=[str(s) for s in d[f"{prefix}segment_names"]],
        )


def geom_mesh_body_local(model: mujoco.MjModel, g: int) -> trimesh.Trimesh | None:
    """把一个 geom 转成 **body 局部系** 的三角网格。"""
    gtype = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=np.float64)

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        v0, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        f0, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        verts = np.asarray(model.mesh_vert[v0 : v0 + nv]).reshape(-1, 3).astype(np.float64)
        faces = np.asarray(model.mesh_face[f0 : f0 + nf]).reshape(-1, 3).astype(np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
    elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(size[:3])
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size[:3])
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # MuJoCo 胶囊 size=[半径, 半长]，沿局部 z 轴以原点为中心；
        # trimesh 的胶囊同样以原点为中心，height 指圆柱段长度。
        r, hl = float(size[0]), float(size[1])
        mesh = trimesh.creation.capsule(height=2.0 * hl, radius=r, count=[16, 16])
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
    else:
        return None

    # geom -> body 局部变换
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(model.geom_quat[g], scalar_first=True).as_matrix()
    T[:3, 3] = model.geom_pos[g]
    mesh.apply_transform(T)
    return mesh


def _hull_planes(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray] | None:
    """凸包的半空间表示 ``n·x <= d``。"""
    try:
        hull = mesh.convex_hull
    except Exception:
        return None
    if len(hull.faces) == 0:
        return None
    n = np.asarray(hull.face_normals, dtype=np.float64)
    d = np.einsum("ij,ij->i", n, np.asarray(hull.triangles[:, 0], dtype=np.float64))
    return n, d


def farthest_point_sampling(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """最远点采样，返回被选中的索引。"""
    m = points.shape[0]
    if n >= m:
        return np.arange(m)
    rng = np.random.default_rng(seed)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(m)
    dist = np.linalg.norm(points - points[idx[0]], axis=1)
    for i in range(1, n):
        idx[i] = int(np.argmax(dist))
        np.minimum(dist, np.linalg.norm(points - points[idx[i]], axis=1), out=dist)
    return idx


def sample_model_surface(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_points: int = 4096,
    geom_ids: list[int] | None = None,
    geom_segment: dict[str, str] | None = None,
    segment_names: list[str] | None = None,
    oversample: float = 8.0,
    cull_margin: float = 0.006,
    seed: int = 0,
) -> SurfacePointCloud:
    """在当前姿态下采样模型的外表面点云。

    Args:
        model, data: 已经跑过 ``mj_kinematics`` 的模型与数据。
        n_points: 输出点数 N。
        geom_ids: 参与采样的 geom；None 表示除世界几何外全部。
        geom_segment: geom 名 -> 分段标签。
        segment_names: 固定的分段名顺序；None 时按出现顺序生成。
        oversample: 先超采样多少倍，再做内点剔除与 FPS。
        cull_margin: 判定"落在别的 geom 内部"的深度阈值（米）。
        seed: 随机种子。
    """
    if geom_ids is None:
        geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0]

    meshes: dict[int, trimesh.Trimesh] = {}
    hulls: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    areas: dict[int, float] = {}
    for g in geom_ids:
        mesh = geom_mesh_body_local(model, g)
        if mesh is None or len(mesh.faces) == 0:
            continue
        meshes[g] = mesh
        hp = _hull_planes(mesh)
        if hp is not None:
            hulls[g] = hp
        areas[g] = float(mesh.area)

    if not meshes:
        raise ValueError("没有可采样的 geom")

    total_area = sum(areas.values())
    budget = int(n_points * oversample)
    rng = np.random.default_rng(seed)

    cand_local, cand_normal, cand_geom = [], [], []
    for g, mesh in meshes.items():
        k = max(16, int(round(budget * areas[g] / total_area)))
        pts, face_idx = trimesh.sample.sample_surface(mesh, k, seed=int(rng.integers(1 << 30)))
        cand_local.append(np.asarray(pts))
        cand_normal.append(np.asarray(mesh.face_normals[face_idx]))
        cand_geom.append(np.full(len(pts), g, dtype=np.int64))

    cand_local = np.concatenate(cand_local)
    cand_normal = np.concatenate(cand_normal)
    cand_geom = np.concatenate(cand_geom)
    cand_body = model.geom_bodyid[cand_geom].astype(np.int64)

    # body 局部 -> 世界
    R = data.xmat[cand_body].reshape(-1, 3, 3)
    world = np.einsum("pij,pj->pi", R, cand_local) + data.xpos[cand_body]
    world_n = np.einsum("pij,pj->pi", R, cand_normal)

    # --- 内点剔除：落在其它 geom 凸包内部超过 margin 的点丢掉 ---
    keep = np.ones(len(world), dtype=bool)
    for g, (hn, hd) in hulls.items():
        b = int(model.geom_bodyid[g])
        Rb = data.xmat[b].reshape(3, 3)
        pb = data.xpos[b]
        # 只需检查不属于该 geom 的候选点
        mask = cand_geom != g
        if not mask.any():
            continue
        local = (world[mask] - pb) @ Rb
        inside = np.all(local @ hn.T <= hd[None, :] - cull_margin, axis=1)
        idx = np.where(mask)[0]
        keep[idx[inside]] = False

    if keep.sum() < n_points:
        # 剔除过狠时放宽（例如模型本身互相深度重叠）
        keep = np.ones(len(world), dtype=bool)

    world, world_n = world[keep], world_n[keep]
    cand_local, cand_normal = cand_local[keep], cand_normal[keep]
    cand_geom, cand_body = cand_geom[keep], cand_body[keep]

    sel = farthest_point_sampling(world, n_points, seed=seed)
    world, world_n = world[sel], world_n[sel]
    cand_local, cand_normal = cand_local[sel], cand_normal[sel]
    cand_geom, cand_body = cand_geom[sel], cand_body[sel]

    norm = np.linalg.norm(world_n, axis=1, keepdims=True)
    world_n = world_n / np.maximum(norm, 1e-12)
    cand_normal = cand_normal / np.maximum(
        np.linalg.norm(cand_normal, axis=1, keepdims=True), 1e-12
    )

    # --- 分段标签 ---
    if segment_names is None:
        segment_names = sorted(set((geom_segment or {}).values()))
    seg_to_id = {s: i for i, s in enumerate(segment_names)}
    segment = np.full(len(world), -1, dtype=np.int64)
    if geom_segment:
        for g in np.unique(cand_geom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
            sid = seg_to_id.get(geom_segment.get(gname or "", ""), -1)
            segment[cand_geom == g] = sid

    return SurfacePointCloud(
        points=world,
        normals=world_n,
        body_ids=cand_body,
        geom_ids=cand_geom,
        local_pos=cand_local,
        local_normal=cand_normal,
        segment=segment,
        segment_names=list(segment_names),
    )


def transport_points(
    data: mujoco.MjData, body_ids: np.ndarray, local_pos: np.ndarray, local_normal: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """按当前 FK 结果把 body 局部的点/法线搬运到世界系（刚性蒙皮）。"""
    R = data.xmat[body_ids].reshape(-1, 3, 3)
    pos = np.einsum("pij,pj->pi", R, local_pos) + data.xpos[body_ids]
    nrm = None
    if local_normal is not None:
        nrm = np.einsum("pij,pj->pi", R, local_normal)
    return pos, nrm
