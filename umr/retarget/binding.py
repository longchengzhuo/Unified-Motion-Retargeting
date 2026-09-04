"""把学到的对应点吸附并绑定到机器人 link（论文 III-C 的 "binds the paired
points to their respective meshes"）。

网络输出的 :math:`\\hat{X}^r` 只是逼近机器人表面，并不精确落在上面。这里把每个
点投影到最近的机器人表面三角面，得到它所属的 body、body 局部坐标和局部法线；
之后运动过程中这些点就随 forward kinematics 刚性搬运。

人体一侧不需要这一步：Stage 0 采样时每个点已经天然绑定在某个 body 上（刚性
蒙皮），等价于论文对可形变网格所用的 barycentric transport。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import trimesh

from umr.bodies.surface import geom_mesh_body_local


@dataclass
class LinkBinding:
    """一组绑定在机器人 link 上的表面点。

    Attributes:
        body_ids: (N,) 所属 body id。
        local_pos: (N, 3) body 局部坐标。
        local_normal: (N, 3) body 局部单位法线。
        snap_distance: (N,) 吸附前后的距离（米），可用来诊断对应质量。
        world_pos: (N, 3) T-pose 下吸附后的世界坐标。
    """

    body_ids: np.ndarray
    local_pos: np.ndarray
    local_normal: np.ndarray
    snap_distance: np.ndarray
    world_pos: np.ndarray

    def to_dict(self, prefix: str = "bind_") -> dict:
        return {
            f"{prefix}body_ids": self.body_ids,
            f"{prefix}local_pos": self.local_pos,
            f"{prefix}local_normal": self.local_normal,
            f"{prefix}snap_distance": self.snap_distance,
            f"{prefix}world_pos": self.world_pos,
        }

    @staticmethod
    def from_dict(d, prefix: str = "bind_") -> "LinkBinding":
        return LinkBinding(
            body_ids=d[f"{prefix}body_ids"],
            local_pos=d[f"{prefix}local_pos"],
            local_normal=d[f"{prefix}local_normal"],
            snap_distance=d[f"{prefix}snap_distance"],
            world_pos=d[f"{prefix}world_pos"],
        )


def build_world_surface(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """把若干 geom 合并成一个世界系三角网格。

    Returns:
        ``(mesh, face_body)``，``face_body[f]`` 是第 f 个三角面所属的 body id。
    """
    verts_all, faces_all, face_body = [], [], []
    voff = 0
    for g in geom_ids:
        mesh = geom_mesh_body_local(model, g)
        if mesh is None or len(mesh.faces) == 0:
            continue
        b = int(model.geom_bodyid[g])
        R = data.xmat[b].reshape(3, 3)
        v = np.asarray(mesh.vertices) @ R.T + data.xpos[b]
        verts_all.append(v)
        faces_all.append(np.asarray(mesh.faces) + voff)
        face_body.append(np.full(len(mesh.faces), b, dtype=np.int64))
        voff += len(v)

    mesh = trimesh.Trimesh(
        vertices=np.concatenate(verts_all),
        faces=np.concatenate(faces_all),
        process=False,
    )
    return mesh, np.concatenate(face_body)


def bind_points_to_links(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    points: np.ndarray,
    geom_ids: list[int],
    flip_normals_outward: bool = True,
) -> LinkBinding:
    """把世界系的点吸附到最近的机器人表面并绑定到对应 link。

    Args:
        model, data: 处于 canonical T-pose 且已跑过 FK 的机器人。
        points: (N, 3) 待绑定的世界坐标点（网络重建的 ``\\hat{X}^r``）。
        geom_ids: 参与构成表面的 geom。
        flip_normals_outward: 让法线朝向背离该 body 质心的一侧。
    """
    mesh, face_body = build_world_surface(model, data, geom_ids)
    closest, dist, face_id = trimesh.proximity.closest_point(mesh, points)

    body_ids = face_body[face_id]
    normals = np.asarray(mesh.face_normals[face_id], dtype=np.float64)

    if flip_normals_outward:
        # STL 的绕序不一定一致，用"背离本 body 表面几何中心"作为朝外判据。
        centroid = np.zeros((model.nbody, 3))
        counts = np.zeros(model.nbody)
        tri_centers = mesh.triangles.mean(axis=1)
        np.add.at(centroid, face_body, tri_centers)
        np.add.at(counts, face_body, 1.0)
        centroid[counts > 0] /= counts[counts > 0, None]
        outward = closest - centroid[body_ids]
        flip = np.einsum("pi,pi->p", normals, outward) < 0
        normals[flip] *= -1.0

    R = data.xmat[body_ids].reshape(-1, 3, 3)
    local_pos = np.einsum("pji,pj->pi", R, closest - data.xpos[body_ids])
    local_normal = np.einsum("pji,pj->pi", R, normals)
    local_normal /= np.maximum(np.linalg.norm(local_normal, axis=1, keepdims=True), 1e-12)

    return LinkBinding(
        body_ids=body_ids.astype(np.int64),
        local_pos=local_pos,
        local_normal=local_normal,
        snap_distance=np.asarray(dist, dtype=np.float64),
        world_pos=np.asarray(closest, dtype=np.float64),
    )
