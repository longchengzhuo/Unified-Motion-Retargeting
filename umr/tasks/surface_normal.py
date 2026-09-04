"""论文式 (7) 的法线残差，实现为一个 ``mink.Task``。

.. math::

    r_{n,i}(q_t) = \\sqrt{w^n_i}\\,\\big(\\bar{n}^r_i(q_t) - \\bar{n}^h_{t,i}\\big)

论文说法线偏移是"相对各自 T-pose 绑定"度量的。由于人机两侧的法线都以 body 局部
法线的形式存下来，运动中 :math:`\\bar n(q) = R_{wb}(q)\\,n_{local}`，两者在 T-pose
下相等，因此它们的差天然就是相对绑定的朝向变化量。

Jacobian 为 :math:`-[n_w]_\\times jac_r`（见 :func:`umr.bodies.robot.point_kinematics`）。
"""

from __future__ import annotations

import mink
import numpy as np

from umr.tasks.cache import RobotSurfaceCache


class SurfaceNormalTask(mink.Task):
    """把机器人表面对应点的法线对齐到人体表面对应点的法线。"""

    def __init__(
        self,
        cache: RobotSurfaceCache,
        indices: np.ndarray,
        weights: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.target = np.zeros((len(self.indices), 3))
        cost = np.repeat(np.sqrt(self.weights), 3)
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)

    def set_target(self, target: np.ndarray) -> None:
        """设置本帧的人体法线目标 ``n^h_{t,i}``，形状 (P, 3)，应为单位向量。"""
        self.target = np.asarray(target, dtype=np.float64)

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        self.cache.refresh(configuration)
        return (self.cache.nrm[self.indices] - self.target).reshape(-1)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        self.cache.refresh(configuration)
        return self.cache.Jn[self.indices].reshape(-1, configuration.model.nv)
