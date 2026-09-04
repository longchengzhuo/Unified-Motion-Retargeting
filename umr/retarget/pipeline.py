"""Stage II 的逐帧重定向流水线（论文 III-C）。

每帧的求解就是论文式 (12)-(13) 的阻尼 Gauss-Newton 迭代，而这恰好是 mink 的
``solve_ik`` + ``Configuration.integrate_inplace`` 所做的事：

* ``build_ik`` 把各 Task 的 :math:`(H, c)` 累加，并加上 :math:`\\mu I` 阻尼，
  得到式 (13) 的目标函数；
* ``ConfigurationLimit`` 给出关节限位 :math:`q^- \\le q + \\Delta q \\le q^+`；
* :class:`~umr.limits.floor_clearance.FloorClearanceLimit` 给出式 (14) 的
  :math:`A_t \\Delta q \\le b_t`；
* :class:`~umr.limits.trust_region.TrustRegionLimit` 给出 :math:`\\|\\Delta q\\| \\le \\eta`；
* ``qpsolvers`` 用 Clarabel 求解，与论文一致。

上一帧的解作为下一帧的初值（warm start），提供论文所要求的时间一致性。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mink
import numpy as np
from scipy.spatial.transform import Rotation

from umr.bodies.human_mjcf import (
    SEGMENT_NORMAL_WEIGHT,
    SEGMENT_POSITION_WEIGHT,
    HumanBody,
)
from umr.bodies.robot import RobotBody, sole_sample_points
from umr.bodies.surface import transport_points
from umr.limits.floor_clearance import FloorClearanceLimit
from umr.limits.trust_region import TrustRegionLimit, solve_ik_socp
from umr.tasks.cache import RobotSurfaceCache
from umr.tasks.contact_map import ContactMapTask, GroundPlane
from umr.tasks.point_matching import PointMatchingTask
from umr.tasks.surface_normal import SurfaceNormalTask


def select_correspondence_points(
    segment: np.ndarray,
    segment_names: list[str],
    n_selected: int,
    seed: int = 0,
    min_per_segment: int = 8,
) -> np.ndarray:
    """论文式 (7) 中的选中集 :math:`I`：按分段分层采样。

    分层可以保证手、脚这类点数少但语义重要的分段不会被躯干淹没。
    """
    rng = np.random.default_rng(seed)
    labelled = np.where(segment >= 0)[0]
    segs = np.unique(segment[labelled])
    total = len(labelled)

    picks: list[np.ndarray] = []
    for s in segs:
        pool = labelled[segment[labelled] == s]
        quota = max(min_per_segment, int(round(n_selected * len(pool) / total)))
        quota = min(quota, len(pool))
        picks.append(rng.choice(pool, size=quota, replace=False))
    idx = np.unique(np.concatenate(picks))
    if len(idx) > n_selected:
        idx = np.sort(rng.choice(idx, size=n_selected, replace=False))
    return idx.astype(np.int64)


def segment_weights(
    segment: np.ndarray, segment_names: list[str], table: dict[str, float], default: float = 1.0
) -> np.ndarray:
    """把分段权重表展开成逐点权重。"""
    return np.array(
        [table.get(segment_names[int(s)], default) if s >= 0 else default for s in segment],
        dtype=np.float64,
    )


@dataclass
class RetargetResult:
    """一段动作的重定向结果。"""

    qpos: np.ndarray                 # (T, nq)
    frame_indices: np.ndarray        # (T,) 对应的 BVH 帧号
    fps: float
    point_error: np.ndarray          # (T,) 选中点的平均位置误差（米）
    normal_error: np.ndarray         # (T,) 选中点的平均法线夹角（弧度）
    contact_count: np.ndarray        # (T,) 激活接触点数
    floor_rows: np.ndarray           # (T,) 地面净空约束行数
    solve_failures: int = 0
    timings: dict[str, float] = field(default_factory=dict)


class UMRRetargeter:
    """把源动作重定向到机器人。"""

    def __init__(
        self,
        robot: RobotBody,
        human: HumanBody,
        *,
        human_body_ids: np.ndarray,
        human_local_pos: np.ndarray,
        human_local_normal: np.ndarray,
        robot_body_ids: np.ndarray,
        robot_local_pos: np.ndarray,
        robot_local_normal: np.ndarray,
        segment: np.ndarray,
        segment_names: list[str],
        n_selected: int = 512,
        iterations: int = 6,
        dt: float = 0.02,
        damping: float = 1e-3,
        solver: str = "clarabel",
        trust_region: str = "box",
        trust_region_radius: float = 0.35,
        floor_height: float = 0.0,
        floor_band: float = 0.06,
        floor_margin: float = 0.002,
        contact_threshold: float = 0.05,
        contact_weight: float = 4.0,
        posture_cost: float = 0.02,
        self_collision: bool = False,
        seed: int = 0,
    ):
        self.robot = robot
        self.human = human
        self.iterations = int(iterations)
        self.dt = float(dt)
        self.damping = float(damping)
        self.solver = solver
        self.trust_region = trust_region
        self.trust_region_radius = float(trust_region_radius)

        self.human_body_ids = human_body_ids
        self.human_local_pos = human_local_pos
        self.human_local_normal = human_local_normal

        # --- 选中集 I 与逐点权重 ---
        self.selected = select_correspondence_points(
            segment, segment_names, n_selected, seed=seed
        )
        wp = segment_weights(segment, segment_names, SEGMENT_POSITION_WEIGHT)
        wn = segment_weights(segment, segment_names, SEGMENT_NORMAL_WEIGHT, default=0.2)

        # --- 任务缓存：只覆盖选中集 I ---
        robot.set_tpose()
        self.active = self.selected
        self.task_idx = np.arange(len(self.selected))
        self.cache = RobotSurfaceCache(
            robot.body_names,
            robot_body_ids[self.selected],
            robot_local_pos[self.selected],
            robot_local_normal[self.selected],
        )

        # --- 地面约束缓存：足底极值几何 + 靠近地面的对应点 ---
        sole_bodies, sole_local, sole_radii = sole_sample_points(robot.model, robot.spec)
        tpose_pos, _ = transport_points(robot.data, robot_body_ids, robot_local_pos)
        low = np.where(tpose_pos[:, 2] < 0.35)[0]
        self.floor_cache = RobotSurfaceCache(
            robot.body_names,
            np.concatenate([sole_bodies, robot_body_ids[low]]),
            np.concatenate([sole_local, robot_local_pos[low]]),
        )
        floor_radii = np.concatenate([sole_radii, np.zeros(len(low))])

        # --- Tasks（论文式 7、8）---
        self.pos_task = PointMatchingTask(self.cache, self.task_idx, wp[self.selected])
        self.nrm_task = SurfaceNormalTask(self.cache, self.task_idx, wn[self.selected])
        self.contact_task = ContactMapTask(
            self.cache, self.task_idx, GroundPlane(floor_height),
            threshold=contact_threshold, weight=contact_weight,
        )
        self.tasks: list[mink.Task] = [self.pos_task, self.nrm_task, self.contact_task]
        if posture_cost > 0:
            posture = mink.PostureTask(robot.model, cost=posture_cost)
            posture.set_target(robot.model.key_qpos[0] if robot.model.nkey else robot.q)
            self.tasks.append(posture)

        # --- Limits（论文式 13、14）---
        self.floor_limit = FloorClearanceLimit(
            self.floor_cache,
            floor_height=floor_height,
            band=floor_band,
            margin=floor_margin,
            radii=floor_radii,
        )
        self.trust_limit = TrustRegionLimit(robot.nv, radius=trust_region_radius)
        self.limits: list[mink.Limit] = [
            mink.ConfigurationLimit(robot.model),
            self.floor_limit,
        ]
        if trust_region == "box":
            self.limits.append(self.trust_limit)
        if self_collision:
            from mink import CollisionAvoidanceLimit
            from mink.utils import get_subtree_geom_ids

            geoms = get_subtree_geom_ids(robot.model, 1)
            self.limits.append(
                CollisionAvoidanceLimit(robot.model, [(geoms, geoms)], minimum_distance_from_collisions=0.02)
            )

    # ------------------------------------------------------------------
    def human_targets(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """本帧的人体表面点与法线（世界系），刚性蒙皮搬运。"""
        self.human.set_frame(frame)
        return transport_points(
            self.human.data, self.human_body_ids, self.human_local_pos, self.human_local_normal
        )

    def initialize_root(self, frame: int) -> None:
        """用人体骨盆的位姿初始化机器人浮动基，给第一帧一个好的起点。"""
        self.human.set_frame(frame)
        hips = self.human.data.xpos[self.human.body_ids[0]]
        R = self.human.data.xmat[self.human.body_ids[0]].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))

        q = self.robot.model.key_qpos[0].copy() if self.robot.model.nkey else self.robot.q.copy()
        q[0:2] = hips[:2]
        q[3:7] = Rotation.from_euler("z", yaw).as_quat(scalar_first=True)
        self.robot.set_qpos(q)

    def solve_frame(self, frame: int, iterations: int | None = None) -> dict:
        """求解单帧，返回诊断信息。"""
        human_pos, human_nrm = self.human_targets(frame)
        target_pos = human_pos[self.selected]
        target_nrm = human_nrm[self.selected]
        self.pos_task.set_target(target_pos)
        self.nrm_task.set_target(target_nrm)
        n_contact = self.contact_task.update_contacts(target_pos)

        cfg = self.robot.configuration
        failures = 0
        for _ in range(iterations or self.iterations):
            try:
                if self.trust_region == "l2":
                    v = solve_ik_socp(
                        cfg, self.tasks, self.dt, self.trust_region_radius,
                        damping=self.damping, limits=self.limits,
                    )
                else:
                    v = mink.solve_ik(
                        cfg, self.tasks, self.dt, self.solver,
                        damping=self.damping, limits=self.limits,
                    )
            except mink.NoSolutionFound:
                # 约束偶尔会冲突（例如脚已经略微穿透地面）；保留上一次迭代的解。
                failures += 1
                break
            cfg.integrate_inplace(v, self.dt)

        self.cache.refresh(cfg, force=True)
        self.floor_cache.refresh(cfg, force=True)
        err = np.linalg.norm(self.cache.pos[self.task_idx] - target_pos, axis=1)
        cos = np.clip(
            np.einsum("ij,ij->i", self.cache.nrm[self.task_idx], target_nrm), -1.0, 1.0
        )
        return {
            "qpos": cfg.q,
            "point_error": float(err.mean()),
            "normal_error": float(np.arccos(cos).mean()),
            "contact_count": n_contact,
            "floor_rows": self.floor_limit.n_active,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    def run(
        self,
        frame_indices: np.ndarray,
        fps: float,
        warmup_iterations: int = 60,
        progress: bool = True,
    ) -> RetargetResult:
        """重定向整段动作。"""
        n = len(frame_indices)
        qpos = np.zeros((n, self.robot.model.nq))
        point_error = np.zeros(n)
        normal_error = np.zeros(n)
        contact_count = np.zeros(n, dtype=np.int64)
        floor_rows = np.zeros(n, dtype=np.int64)
        failures = 0

        self.initialize_root(int(frame_indices[0]))
        t0 = time.perf_counter()
        self.solve_frame(int(frame_indices[0]), iterations=warmup_iterations)
        t_warm = time.perf_counter() - t0

        t0 = time.perf_counter()
        for k, f in enumerate(frame_indices):
            out = self.solve_frame(int(f))  # 上一帧的 qpos 天然成为本帧初值
            qpos[k] = out["qpos"]
            point_error[k] = out["point_error"]
            normal_error[k] = out["normal_error"]
            contact_count[k] = out["contact_count"]
            floor_rows[k] = out["floor_rows"]
            failures += out["failures"]
            if progress and (k % 200 == 0 or k == n - 1):
                el = time.perf_counter() - t0
                print(
                    f"  frame {k+1:5d}/{n}  err={out['point_error']*1000:6.1f}mm  "
                    f"n_ang={np.degrees(out['normal_error']):5.1f}deg  "
                    f"contacts={out['contact_count']:3d}  floor={out['floor_rows']:3d}  "
                    f"{(k+1)/max(el,1e-9):6.1f} FPS"
                )
        elapsed = time.perf_counter() - t0

        return RetargetResult(
            qpos=qpos,
            frame_indices=np.asarray(frame_indices),
            fps=fps,
            point_error=point_error,
            normal_error=normal_error,
            contact_count=contact_count,
            floor_rows=floor_rows,
            solve_failures=failures,
            timings={
                "warmup": t_warm,
                "retarget": elapsed,
                "fps": n / elapsed if elapsed > 0 else 0.0,
                "jacobian_calls": float(self.cache.n_jacobian_calls),
            },
        )
