"""绑定点的运动学缓存。

多个 Task 共享同一组绑定点，若各自独立求 Jacobian 会重复大量计算。这里在一次
Gauss-Newton 迭代内缓存：每个涉及的 body 只调一次 mink 的
``get_frame_jacobian``（39 个 body），再向量化推出所有点的位置、法线与 Jacobian。
"""

from __future__ import annotations

import mink
import numpy as np

from umr.bodies.robot import body_jacobians, point_kinematics


class RobotSurfaceCache:
    """缓存一组 link-bound 表面点在当前配置下的运动学量。"""

    def __init__(
        self,
        body_names: list[str],
        body_ids: np.ndarray,
        local_pos: np.ndarray,
        local_normal: np.ndarray | None = None,
    ):
        self.body_names = body_names
        self.unique_bodies, self.body_index = np.unique(body_ids, return_inverse=True)
        self.body_index = self.body_index.reshape(-1)
        self.local_pos = np.asarray(local_pos, dtype=np.float64)
        self.local_normal = (
            np.asarray(local_normal, dtype=np.float64) if local_normal is not None else None
        )
        self._q: np.ndarray | None = None
        self.pos: np.ndarray | None = None
        self.nrm: np.ndarray | None = None
        self.Jp: np.ndarray | None = None
        self.Jn: np.ndarray | None = None
        self.n_jacobian_calls = 0

    def refresh(self, configuration: mink.Configuration, force: bool = False) -> None:
        """若配置变化则重算（qpos 比较很便宜，避免同一迭代内重复计算）。"""
        q = configuration.data.qpos
        if not force and self._q is not None and np.array_equal(self._q, q):
            return
        self._q = q.copy()
        jac = body_jacobians(configuration, self.body_names, self.unique_bodies)
        self.n_jacobian_calls += len(self.unique_bodies)
        self.pos, self.nrm, self.Jp, self.Jn = point_kinematics(
            self.body_index, self.local_pos, self.local_normal, jac
        )

    def positions(self, configuration: mink.Configuration) -> np.ndarray:
        self.refresh(configuration)
        return self.pos

    def normals(self, configuration: mink.Configuration) -> np.ndarray:
        self.refresh(configuration)
        return self.nrm
