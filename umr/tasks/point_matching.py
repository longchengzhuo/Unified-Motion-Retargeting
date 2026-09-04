"""论文式 (7) 的位置残差，实现为一个 ``mink.Task``。

.. math::

    r_{p,i}(q_t) = \\sqrt{w^p_i}\\,\\big(x^r_i(q_t) - x^h_{t,i}\\big), \\quad i \\in I

mink 的 ``Task._assemble_qp`` 会构造 :math:`H=(WJ)^T(WJ)`、:math:`c=-(We)^T(WJ)`，
配合 ``build_ik`` 加上的 :math:`\\mu I` 阻尼项，正好就是论文式 (13) 的阻尼
Gauss-Newton 子问题。因此 ``cost`` 直接取 :math:`\\sqrt{w^p_i}`。
"""

from __future__ import annotations

import mink
import numpy as np

from umr.tasks.cache import RobotSurfaceCache


class PointMatchingTask(mink.Task):
    """把机器人表面对应点匹配到人体表面目标点。"""

    def __init__(
        self,
        cache: RobotSurfaceCache,
        indices: np.ndarray,
        weights: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """
        Args:
            cache: 共享的绑定点运动学缓存。
            indices: (P,) 参与优化的点在缓存中的下标，即论文的选中集 ``I``。
            weights: (P,) 每点的位置权重 ``w^p_i``（按人体分段查表）。
            gain: mink 的任务增益 α。
            lm_damping: 目标不可行时的额外 LM 阻尼。
        """
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.target = np.zeros((len(self.indices), 3))
        cost = np.repeat(np.sqrt(self.weights), 3)
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)

    def set_target(self, target: np.ndarray) -> None:
        """设置本帧的人体目标点 ``x^h_{t,i}``，形状 (P, 3)。"""
        self.target = np.asarray(target, dtype=np.float64)

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        self.cache.refresh(configuration)
        return (self.cache.pos[self.indices] - self.target).reshape(-1)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        self.cache.refresh(configuration)
        return self.cache.Jp[self.indices].reshape(-1, configuration.model.nv)
