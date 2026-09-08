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
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from umr.bodies.human_mjcf import (
    SEGMENT_NORMAL_WEIGHT,
    SEGMENT_POSITION_WEIGHT,
    HumanBody,
)
from umr.bodies.robot import RobotBody, sole_sample_points
from umr.bodies.surface import farthest_point_sampling, transport_points
from umr.limits.floor_clearance import FloorClearanceLimit
from umr.limits.trust_region import TrustRegionLimit, solve_ik_socp
from umr.tasks.cache import RobotSurfaceCache
from umr.tasks.contact_map import ContactMapTask, GroundPlane
from umr.tasks.point_matching import PointMatchingTask
from umr.tasks.surface_normal import SurfaceNormalTask

POINT_SELECTIONS = ("fps", "random")


def stratified_quota(
    counts: np.ndarray, n_selected: int, min_per_segment: int
) -> np.ndarray:
    """按点数比例给各分段分配配额，总和恰为 ``n_selected``。

    保底会把总量顶高（21 个分段各保底 8 个就已占掉 168），超出的部分只从「高于保底」
    的那一截里按比例回收，于是大分段被削掉一点而小分段的保底不受影响。原来的做法是
    先各段独立取整、再随机裁掉多余的点，那一步随机裁剪本身就会重新打散均匀性。

    Args:
        counts: (S,) 各分段的候选点数。
        n_selected: 目标总点数。
        min_per_segment: 每段保底点数。

    Returns:
        (S,) 整数配额，``sum == min(n_selected, counts.sum())``。
    """
    counts = np.asarray(counts, dtype=np.int64)
    target = min(int(n_selected), int(counts.sum()))
    q = np.clip(target * counts / counts.sum(), min_per_segment, counts).astype(np.float64)
    # 保底与上限会互相顶，几轮回收就能收敛到总和为 target。
    for _ in range(8):
        gap = q.sum() - target
        if abs(gap) < 1e-9:
            break
        free = np.maximum(q - min_per_segment, 0.0) if gap > 0 else np.maximum(counts - q, 0.0)
        if free.sum() <= 0:
            break
        q = np.clip(q - gap * free / free.sum(), min_per_segment, counts)

    # 最大余数法取整：floor 只会让总和变小，缺口按小数部分从大到小补回去。
    out = np.clip(np.floor(q), 0, counts).astype(np.int64)
    short = target - int(out.sum())
    for i in np.argsort(-(q - out)):
        if short <= 0:
            break
        if out[i] < counts[i]:
            out[i] += 1
            short -= 1
    return out


def select_correspondence_points(
    segment: np.ndarray,
    segment_names: list[str],
    n_selected: int,
    seed: int = 0,
    min_per_segment: int = 8,
    points: np.ndarray | None = None,
    method: str = "fps",
) -> np.ndarray:
    """论文式 (7) 中的选中集 :math:`I`：按分段分层采样。

    分层可以保证手、脚这类点数少但语义重要的分段不会被躯干淹没。段内默认走最远点
    采样：Stage 0 用 FPS 攒下的均匀性会被随机抽样重新打散，实测 512 点的最近邻间距
    变异系数从 0.12 恶化到 0.40，身上还会留下 7.5 cm 的空洞——那些区域在 QP 里一个
    约束点都没有。``method="random"`` 保留原来的随机抽样，仅供对照。

    Args:
        segment: (N,) 逐点分段 id，负数表示无标签。
        segment_names: 分段名，仅为保持调用方签名稳定。
        n_selected: 目标点数 :math:`|I|`。
        seed: 随机种子（FPS 只用它挑起始点）。
        min_per_segment: 每段保底点数。
        points: (N, 3) 人体 T-pose 世界坐标，``method="fps"`` 时必须给。
        method: ``fps`` | ``random``。
    """
    del segment_names
    if method not in POINT_SELECTIONS:
        raise ValueError(f"未知的 point_selection: {method!r}（可用 {' / '.join(POINT_SELECTIONS)}）")
    if method == "fps" and points is None:
        raise ValueError("point_selection=fps 需要提供人体 T-pose 点坐标")

    labelled = np.where(segment >= 0)[0]
    segs, counts = np.unique(segment[labelled], return_counts=True)
    quota = stratified_quota(counts, n_selected, min_per_segment)

    rng = np.random.default_rng(seed)
    picks: list[np.ndarray] = []
    for s, q in zip(segs, quota):
        if q <= 0:
            continue
        pool = labelled[segment[labelled] == s]
        if method == "fps":
            picks.append(pool[farthest_point_sampling(points[pool], int(q), seed=seed)])
        else:
            picks.append(rng.choice(pool, size=int(q), replace=False))
    # 各段的池互不相交，配额之和已经等于目标，所以直接拼接排序即可。
    return np.sort(np.concatenate(picks)).astype(np.int64)


def segment_weights(
    segment: np.ndarray, segment_names: list[str], table: dict[str, float], default: float = 1.0
) -> np.ndarray:
    """把分段权重表展开成逐点权重。"""
    return np.array(
        [table.get(segment_names[int(s)], default) if s >= 0 else default for s in segment],
        dtype=np.float64,
    )


def compensate_tpose_offset(
    human_tpose_rot: np.ndarray,
    human_local_pos: np.ndarray,
    human_local_normal: np.ndarray,
    offset: np.ndarray,
    robot_tpose_normal: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """把 T-pose 下的人机形状差折进人体绑定，消掉式 (7) 残差里的常量偏置。

    式 (7) 字面上要求机器人的表面点落到人体的表面点上，可两具身体的表面本就差着几
    厘米（实测 T-pose 下平均 44.7 mm、p95 达 82 mm），这个残差再怎么解也消不掉。于是
    优化器只能拿姿态去换：弯膝、低头、把脚踝翻到限位，每一下都能蹭掉几毫米。改成让
    机器人点跟随「自己在 T-pose 的位置被人体肢体旋转搬运后的落点」，T-pose 处残差恰好
    归零，剩下的就是纯粹的相对运动匹配。

    这里不需要在逐帧路径上加任何东西。目标点

    .. math::  \\text{target}_i = x^h_{t,i} + R^h_{b(i)}(t)\\,\\delta^{local}_i

    与 :math:`x^h_{t,i} = R^h_{b(i)}(t)\\,p^{local}_i + \\text{xpos}_{b(i)}(t)` 合并后是

    .. math::  \\text{target}_i = R^h_{b(i)}(t)\\,(p^{local}_i + \\delta^{local}_i) + \\text{xpos}_{b(i)}(t)

    也就是说把偏置一次性加进 ``human_local_pos`` 即可，``transport_points`` 照原样跑。

    Args:
        human_tpose_rot: (N, 3, 3) T-pose 下各点所属人体 body 的世界旋转。
        human_local_pos: (N, 3) 人体 body 局部坐标。
        human_local_normal: (N, 3) 人体 body 局部法线。
        offset: (N, 3) 世界系偏置 :math:`x^r_i(T) - x^h_i(T)`。
        robot_tpose_normal: (N, 3) T-pose 下机器人对应点的世界法线。
        alpha: 补偿系数，0 退回论文式 (7) 原式，1 为全补偿。

    Returns:
        ``(local_pos, local_normal)``，可直接交给 :func:`~umr.bodies.surface.transport_points`。
    """
    if alpha == 0.0:
        return human_local_pos, human_local_normal
    # 偏置里含 (+12.0, 0.0, -32.4) mm 的整体平移（人体 T-pose 本就没严丝合缝地贴地）。
    # 不减掉的话机器人会被整体压低三厘米，而当前的全局对齐（质心差 1~2 mm）本来是好的，
    # 所以只补偿去均值后的局部形状差。
    centred = offset - offset.mean(axis=0)
    local_pos = human_local_pos + np.einsum("pji,pj->pi", human_tpose_rot, alpha * centred)

    # 法线同样改成跟随机器人自己的 T-pose 朝向。附带把吸附投影选错三角面朝向的那
    # 6% 反向法线一并中和：不再拿人机法线对比，翻转就不再是误差来源。
    world = (1.0 - alpha) * np.einsum(
        "pij,pj->pi", human_tpose_rot, human_local_normal
    ) + alpha * robot_tpose_normal
    world /= np.maximum(np.linalg.norm(world, axis=1, keepdims=True), 1e-12)
    return local_pos, np.einsum("pji,pj->pi", human_tpose_rot, world)


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
        point_selection: str = "fps",
        tpose_offset: float = 1.0,
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

        # --- T-pose 偏置补偿 ---
        robot.set_tpose()
        human.set_tpose()
        tpose_pos, r_nrm = transport_points(
            robot.data, robot_body_ids, robot_local_pos, robot_local_normal
        )
        h_pos, h_nrm = transport_points(
            human.data, human_body_ids, human_local_pos, human_local_normal
        )
        self.tpose_offset = float(tpose_offset)
        self.human_local_pos, self.human_local_normal = compensate_tpose_offset(
            human.data.xmat[human_body_ids].reshape(-1, 3, 3),
            human_local_pos, human_local_normal,
            tpose_pos - h_pos, r_nrm, self.tpose_offset,
        )
        # 论文式 (7) 原始残差在 T-pose 下的下界：人机表面本就不重合，这个量补偿前
        # 无法被任何关节角消掉，日志里报出来才好判断逐帧误差是大是小。
        self.tpose_baseline_mm = float(np.linalg.norm(tpose_pos - h_pos, axis=1).mean() * 1e3)
        self.tpose_baseline_deg = float(
            np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", r_nrm, h_nrm), -1.0, 1.0))).mean()
        )

        # --- 选中集 I 与逐点权重 ---
        self.selected = select_correspondence_points(
            segment, segment_names, n_selected, seed=seed,
            points=h_pos, method=point_selection,
        )
        wp = segment_weights(segment, segment_names, SEGMENT_POSITION_WEIGHT)
        wn = segment_weights(segment, segment_names, SEGMENT_NORMAL_WEIGHT, default=0.2)

        # --- 任务缓存：只覆盖选中集 I ---
        self.active = self.selected
        self.task_idx = np.arange(len(self.selected))
        self.cache = RobotSurfaceCache(
            robot.body_names,
            robot_body_ids[self.selected],
            robot_local_pos[self.selected],
            robot_local_normal[self.selected],
        )

        # --- 地面约束缓存：足底极值几何 + 靠近地面的对应点 ---
        sole_bodies, sole_local, sole_radii = sole_sample_points(
            robot.model, robot.spec.foot_name_keys
        )
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
    def ankle_roll_joints(self) -> list[tuple[str, int]]:
        """机器人上的踝 roll 关节及其 qpos 地址。按关节名里的 ``ankle_roll`` 匹配。"""
        model = self.robot.model
        out = []
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "ankle_roll" in name:
                out.append((name, int(model.jnt_qposadr[j])))
        return out

    def lock_ankle_roll(self, result: RetargetResult, n_static: int) -> dict[str, float]:
        """把踝 roll 冻结在片头静止段的取值上，并重算受影响的诊断量。

        动捕解算的踝 roll 是整条链里最不可靠的通道：脚在腾空或落地的瞬间标记点最
        容易被遮挡，解算器一旦跳到错误分支往往就再也回不来（实测某条 take 的左踝
        在两帧内跳了 47°，之后 2.7 秒一直卡在 −50°，把机器人的踝关节顶到限位）。
        站定那一段是唯一能确信「脚平放在地上」的时刻，把它的取值锁住即可。

        踝 roll 轴离足底很近，事后覆盖对足底高度的影响实测只有毫米级（mean 1.4 mm、
        max 4.0 mm，未引入任何穿透帧），所以不必回到 QP 里加约束。

        Args:
            result: :meth:`run` 的结果，``qpos`` 会被就地改写。
            n_static: 片头静止段的帧数，取该段内各关节的中位数作为锁定值。

        Returns:
            关节名 -> 锁定角度（弧度）。
        """
        joints = self.ankle_roll_joints()
        if not joints or n_static < 1:
            return {}
        locked = {}
        for name, adr in joints:
            value = float(np.median(result.qpos[:n_static, adr]))
            result.qpos[:, adr] = value
            locked[name] = value

        # qpos 变了，run() 里逐帧算出的误差就过期了，按新姿态重算一遍免得报告失真。
        cfg = self.robot.configuration
        for k, f in enumerate(result.frame_indices):
            human_pos, human_nrm = self.human_targets(int(f))
            target_pos = human_pos[self.selected]
            cfg.update(q=result.qpos[k])
            self.cache.refresh(cfg, force=True)
            cos = np.clip(
                np.einsum("ij,ij->i", self.cache.nrm[self.task_idx], human_nrm[self.selected]),
                -1.0, 1.0,
            )
            result.point_error[k] = float(
                np.linalg.norm(self.cache.pos[self.task_idx] - target_pos, axis=1).mean()
            )
            result.normal_error[k] = float(np.arccos(cos).mean())
        return locked

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
