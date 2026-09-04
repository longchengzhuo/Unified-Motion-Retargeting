"""论文式 (13) 的信赖域 :math:`\\|\\Delta q\\|_2 \\le \\eta`。

mink 的 ``build_ik`` 组装的是一个只含线性不等式的 QP，无法直接表达二阶锥。这里给
两种做法：

``box``（默认）
    用无穷范数盒式外近似 :math:`\\|\\Delta q\\|_\\infty \\le \\eta/\\sqrt{n_v}`，
    它是 L2 球的内接盒，因此**满足** L2 约束，可配任意 QP 后端。实现为一个普通的
    ``mink.Limit``。

``l2``（严格）
    用 :func:`solve_ik_socp`：仍然由 ``mink.build_ik`` 组装 :math:`P, q, G, h`，
    再追加一行二阶锥交给 Clarabel 求解 SOCP，与论文完全一致。
"""

from __future__ import annotations

import mink
import numpy as np
from mink.limits import Constraint, Limit


class TrustRegionLimit(Limit):
    """L2 信赖域的盒式内近似。"""

    def __init__(self, nv: int, radius: float = 0.35, inscribed: bool = True):
        """
        Args:
            nv: 切空间维度。
            radius: 式 (13) 的 η。
            inscribed: True 时用内接盒（每个分量上界 ``η/sqrt(nv)``，严格满足
                ``‖Δq‖₂ ≤ η``）；False 时每个分量直接以 η 为界（外接盒，更宽松）。
        """
        self.nv = int(nv)
        self.radius = float(radius)
        self.inscribed = bool(inscribed)
        self._eye = np.eye(self.nv)

    @property
    def component_bound(self) -> float:
        return self.radius / np.sqrt(self.nv) if self.inscribed else self.radius

    def compute_qp_inequalities(
        self, configuration: mink.Configuration, dt: float
    ) -> Constraint:
        del configuration, dt
        b = self.component_bound
        G = np.vstack([self._eye, -self._eye])
        h = np.full(2 * self.nv, b)
        return Constraint(G=G, h=h)


def solve_ik_socp(
    configuration: mink.Configuration,
    tasks,
    dt: float,
    radius: float,
    damping: float = 1e-3,
    limits=None,
    safety_break: bool = False,
) -> np.ndarray:
    """严格 L2 信赖域版本的求解：mink 组装 + Clarabel 解 SOCP。

    直接复用 ``mink.build_ik`` 得到式 (13) 的目标与线性约束，再补上二阶锥

    .. math::  \\|\\Delta q\\|_2 \\le \\eta

    Clarabel 原生支持二阶锥，这也是论文选用它的原因。

    Returns:
        切空间速度 :math:`v = \\Delta q / dt`，与 ``mink.solve_ik`` 的返回一致。
    """
    import clarabel
    from scipy import sparse

    configuration.check_limits(safety_break=safety_break)
    problem = mink.build_ik(configuration, tasks, dt, damping, limits)
    P = np.asarray(problem.P, dtype=np.float64)
    q = np.asarray(problem.q, dtype=np.float64)
    n = P.shape[0]

    A_blocks, b_blocks, cones = [], [], []
    if problem.G is not None and problem.h is not None:
        A_blocks.append(np.asarray(problem.G, dtype=np.float64))
        b_blocks.append(np.asarray(problem.h, dtype=np.float64))
        cones.append(clarabel.NonnegativeConeT(problem.h.shape[0]))
    if problem.lb is not None:
        finite = np.isfinite(problem.lb)
        if finite.any():
            A_blocks.append(-np.eye(n)[finite])
            b_blocks.append(-problem.lb[finite])
            cones.append(clarabel.NonnegativeConeT(int(finite.sum())))
    if problem.ub is not None:
        finite = np.isfinite(problem.ub)
        if finite.any():
            A_blocks.append(np.eye(n)[finite])
            b_blocks.append(problem.ub[finite])
            cones.append(clarabel.NonnegativeConeT(int(finite.sum())))

    # 二阶锥 ‖Δq‖₂ ≤ η，写成 (η, -Δq) ∈ SOC(nv+1)
    nv = configuration.model.nv
    soc_A = np.zeros((nv + 1, n))
    soc_A[1:, :nv] = -np.eye(nv)
    soc_b = np.zeros(nv + 1)
    soc_b[0] = radius
    A_blocks.append(soc_A)
    b_blocks.append(soc_b)
    cones.append(clarabel.SecondOrderConeT(nv + 1))

    A = sparse.csc_matrix(np.vstack(A_blocks))
    b = np.concatenate(b_blocks)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(
        sparse.csc_matrix(np.triu(P)), q, A, b, cones, settings
    )
    solution = solver.solve()
    x = np.asarray(solution.x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise mink.NoSolutionFound("clarabel-socp")
    return x[:nv] / dt
