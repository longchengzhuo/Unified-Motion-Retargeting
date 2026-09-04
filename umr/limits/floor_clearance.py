"""论文式 (14) 的地面净空约束，实现为一个 ``mink.Limit``。

对靠近地面的机器人表面点 :math:`i \\in F_t`，要求 :math:`z^r_i(q_t + \\Delta q) \\ge z_f`。
一阶展开后得到

.. math::

    -J^z_i(q_t)\\,\\Delta q \\le z^r_i(q_t) - z_f

把这些行堆起来就是式 (13) 中的 :math:`A_t \\Delta q \\le b_t`。mink 的
``Limit.compute_qp_inequalities`` 返回的正是 :math:`G\\Delta q \\le h`，所以直接对应。
"""

from __future__ import annotations

import mink
import numpy as np
from mink.limits import Constraint, Limit

from umr.tasks.cache import RobotSurfaceCache


class FloorClearanceLimit(Limit):
    """禁止机器人表面点穿透地面。"""

    def __init__(
        self,
        cache: RobotSurfaceCache,
        floor_height: float = 0.0,
        band: float = 0.06,
        margin: float = 0.002,
        gain: float = 1.0,
        max_rows: int = 400,
        radii: np.ndarray | None = None,
    ):
        """
        Args:
            cache: 参与约束的点的运动学缓存。传入的应当是**足底几何点**加上靠近
                地面的对应点，而不只是优化选中集——否则足底最低处可能没被约束到，
                脚会陷进地面。
            floor_height: 式 (14) 的 :math:`z_f`。
            band: 只对 :math:`z < z_f + band` 的点建立约束，控制问题规模。
            margin: 期望保留的最小离地间隙（米）。
            gain: 约束松弛系数，(0, 1]，越小越保守。
            max_rows: 单帧最多保留的约束行数（按高度从低到高取）。
            radii: (N,) 每个点的球半径偏移，约束按 ``z(q) - r >= z_f`` 施加；
                足底的球形接触点用得上，其余为 0。
        """
        self.cache = cache
        self.floor_height = float(floor_height)
        self.band = float(band)
        self.margin = float(margin)
        self.gain = float(gain)
        self.max_rows = int(max_rows)
        self.radii = (
            np.zeros(len(cache.local_pos)) if radii is None else np.asarray(radii, dtype=np.float64)
        )
        self.n_active = 0

    def compute_qp_inequalities(
        self, configuration: mink.Configuration, dt: float
    ) -> Constraint:
        del dt  # 约束直接作用于 Δq，与步长无关
        self.cache.refresh(configuration)
        z = self.cache.pos[:, 2] - self.radii
        idx = np.where(z < self.floor_height + self.band)[0]
        if idx.size == 0:
            self.n_active = 0
            return Constraint()
        if idx.size > self.max_rows:
            idx = idx[np.argsort(z[idx])[: self.max_rows]]
        self.n_active = int(idx.size)

        # 式 (14):  -J^z Δq <= z(q) - z_f
        G = -self.cache.Jp[idx, 2, :]
        h = self.gain * (z[idx] - self.floor_height - self.margin)
        return Constraint(G=G, h=h)
