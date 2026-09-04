"""论文式 (8)-(11) 的接触残差，实现为一个 ``mink.Task``。

论文把接触图表示成"从每个关键点指向环境表面最近点的方向向量"（Fig. 4）：

.. math::

    \\pi_t(i) &= \\arg\\min_j \\|x^h_{t,i} - y_{t,j}\\|_2 \\\\
    c^h_{t,i} &= x^h_{t,i} - y_{t,\\pi_t(i)} \\\\
    c^r_i(q_t) &= x^r_i(q_t) - y_{t,\\pi_t(i)} \\\\
    C_t &= \\{ i \\in I : \\|c^h_{t,i}\\|_2 \\le \\tau_c \\}

关键在于 :math:`\\pi_t(i)` 由**人体**点决定、人机**共用**同一个环境点，这正是
"无需额外身体映射即可直接迁移接触"的来源。注意由此残差 :math:`c^r_i - c^h_{t,i}`
在代数上会化简为 :math:`x^r_i(q_t) - x^h_{t,i}`：也就是说接触项的作用是在激活的
接触点上**追加权重**，而不是引入新的几何量。这里仍按论文的结构显式计算 c 向量，
以便将来换成物体/场景点云时可以直接复用。
"""

from __future__ import annotations

import mink
import numpy as np
from scipy.spatial import cKDTree

from umr.tasks.cache import RobotSurfaceCache


class GroundPlane:
    """把地面 z=z_f 当成环境点云：最近点即竖直投影。"""

    def __init__(self, height: float = 0.0):
        self.height = float(height)

    def nearest(self, points: np.ndarray) -> np.ndarray:
        y = np.asarray(points, dtype=np.float64).copy()
        y[:, 2] = self.height
        return y


class PointCloudEnvironment:
    """一般的环境点云（物体或场景表面），用 KD-tree 查最近点。"""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=np.float64)
        self.tree = cKDTree(self.points)

    def nearest(self, points: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(points, k=1)
        return self.points[idx]


class ContactMapTask(mink.Task):
    """接触图匹配：在激活接触点上追加权重。

    每帧调用 :meth:`update_contacts` 重新确定激活集 ``C_t``；未激活的点权重为 0，
    因而不影响 QP。
    """

    def __init__(
        self,
        cache: RobotSurfaceCache,
        indices: np.ndarray,
        environment,
        threshold: float = 0.05,
        weight: float = 4.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """
        Args:
            cache: 共享的绑定点运动学缓存。
            indices: (P,) 选中集 ``I`` 在缓存中的下标。
            environment: 提供 ``nearest(points)`` 的环境对象。
            threshold: 式 (11) 的 τ_c。
            weight: 式 (8) 的 w^c。
            gain, lm_damping: mink 的任务参数。
        """
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.environment = environment
        self.threshold = float(threshold)
        self.weight = float(weight)

        p = len(self.indices)
        self.env_points = np.zeros((p, 3))
        self.contact_human = np.zeros((p, 3))
        self.active = np.zeros(p, dtype=bool)
        super().__init__(cost=np.zeros(3 * p), gain=gain, lm_damping=lm_damping)

    def update_contacts(self, human_points: np.ndarray) -> int:
        """由本帧的人体点确定激活接触集，返回激活点数。"""
        human_points = np.asarray(human_points, dtype=np.float64)
        self.env_points = self.environment.nearest(human_points)      # y_{t,π(i)}
        self.contact_human = human_points - self.env_points           # 式 (10) 的 c^h
        self.active = np.linalg.norm(self.contact_human, axis=1) <= self.threshold  # 式 (11)
        w = np.where(self.active, self.weight, 0.0)
        self.cost = np.repeat(np.sqrt(w), 3)
        return int(self.active.sum())

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        self.cache.refresh(configuration)
        contact_robot = self.cache.pos[self.indices] - self.env_points  # 式 (10) 的 c^r
        return (contact_robot - self.contact_human).reshape(-1)        # 式 (8)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        self.cache.refresh(configuration)
        # d c^r_i / dq = d x^r_i / dq（环境点与 q 无关）
        return self.cache.Jp[self.indices].reshape(-1, configuration.model.nv)
