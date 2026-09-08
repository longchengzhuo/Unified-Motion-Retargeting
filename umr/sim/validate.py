"""用 MuJoCo 动力学校验重定向结果。

运动学重定向只保证表面姿态匹配，不保证物理上合理。这里用 MuJoCo 的动力学量做
几项检查：

* ``mj_inverse`` 反求关节力矩，看幅值是否落在合理范围；
* ``mj_makeM`` / ``Configuration.get_inertia_matrix`` 给出关节空间惯量矩阵；
* 质心轨迹与 ZMP 是否落在支撑多边形内；
* 可选：用 PD 控制在 ``mj_step`` 里回放。

两点必须说明：

1. 参考轨迹是 30 FPS 的**运动学**结果，直接做二阶差分会放大噪声，因此速度与
   加速度都经过 Savitzky-Golay 平滑后才送进反动力学。
2. PD 回放是一次**开环欠驱动仿真**，没有任何平衡控制器。运动学参考在这种条件
   下跌倒是预期行为——论文的下游结果同样需要先训练 RL 跟踪策略。它的价值在于
   提供一个可复现的可行性探针，而不是稳定性判据。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.signal import savgol_filter

from umr.bodies.robot import geom_lowest_z, is_marker_body, lowest_surface_z

GRAVITY = 9.81


@dataclass
class ValidationResult:
    torque: np.ndarray            # (T, nv-6) 反动力学关节力矩
    com: np.ndarray               # (T, 3) 质心
    zmp: np.ndarray               # (T, 2) 质心动力学导出的 ZMP
    foot_height: np.ndarray       # (T,) 机器人最低表面点高度
    support_ok: np.ndarray        # (T,) ZMP 是否落在支撑多边形内
    support_size: np.ndarray      # (T,) 支撑多边形的接触点数
    zmp_margin: np.ndarray        # (T,) ZMP 到支撑多边形的有符号距离（负=在内部）
    base_residual: np.ndarray     # (T,) 自由基座 6 维残余力/力矩范数


def _smooth(x: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    """沿时间轴做 Savitzky-Golay 平滑，序列过短时原样返回。"""
    if x.shape[0] < window or window <= poly:
        return x
    if window % 2 == 0:
        window += 1
    return savgol_filter(x, window, poly, axis=0)


def differentiate_configuration(
    model: mujoco.MjModel, qpos: np.ndarray, dt: float, smooth_window: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """由 qpos 序列求出平滑的 qvel 与 qacc。

    差分在切空间中用 ``mj_differentiatePos`` 完成，因此四元数被正确处理。
    """
    n, nv = qpos.shape[0], model.nv
    qvel = np.zeros((n, nv))
    for k in range(1, n):
        v = np.zeros(nv)
        mujoco.mj_differentiatePos(model, v, dt, qpos[k - 1], qpos[k])
        qvel[k] = v
    if n > 1:
        qvel[0] = qvel[1]
    qvel = _smooth(qvel, smooth_window)

    qacc = np.zeros((n, nv))
    if n > 2:
        qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)
        qacc[0], qacc[-1] = qacc[1], qacc[-2]
    qacc = _smooth(qacc, smooth_window)
    return qvel, qacc


def foot_contact_points(
    model: mujoco.MjModel, data: mujoco.MjData, height_tol: float = 0.02
) -> np.ndarray:
    """当前姿态下贴近地面的足部几何顶点，投影到水平面。

    用 box 的 8 个角点与球心（而不是 geom 中心），这样单脚支撑时支撑多边形仍然
    有宽度，不会退化成一条线。
    """
    pts = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if is_marker_body(name) or ("ankle" not in name and "foot" not in name):
            continue
        c = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        gtype = int(model.geom_type[g])
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
            corners = c + (signs * size[:3]) @ R.T
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            corners = np.array([c - [0.0, 0.0, size[0]]])
        else:
            continue
        pts.extend(corners[corners[:, 2] < height_tol][:, :2])
    return np.array(pts) if pts else np.zeros((0, 2))


def foot_geom_sides(model: mujoco.MjModel, kind: str) -> list[list[int]]:
    """左右脚各自的足部 geom 下标，``kind`` 取 ``"human"`` 或 ``"robot"``。

    人体 MJCF 的足部是 ``hg_<Side>Ankle`` / ``hg_<Side>Toe`` 这样的 geom 名，机器人
    一侧则按 geom 所属 body 名里的 ``left_ankle`` / ``right_ankle`` 来找。按帧扫模型
    很贵，所以先一次性解析出下标，之后逐帧只做取最小值。
    """
    if kind == "human":
        keys = ("Left", "Right")
    elif kind == "robot":
        keys = ("left_ankle", "right_ankle")
    else:
        raise ValueError(f"kind 只能是 'human' 或 'robot'，收到 {kind!r}")

    sides = []
    for key in keys:
        ids = []
        for g in range(model.ngeom):
            if kind == "human":
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                hit = name.startswith(f"hg_{key}") and ("Ankle" in name or "Toe" in name)
            else:
                b = int(model.geom_bodyid[g])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                hit = key in name
            if hit:
                ids.append(g)
        if not ids:
            raise ValueError(f"{kind} 模型里找不到 {key} 一侧的足部 geom")
        sides.append(ids)
    return sides


def side_foot_heights(
    model: mujoco.MjModel, data: mujoco.MjData, sides: list[list[int]]
) -> list[float]:
    """当前姿态下左右脚各自最低点的高度。"""
    return [min(geom_lowest_z(model, data, g) for g in ids) for ids in sides]


def hull_signed_distance(p: np.ndarray, pts: np.ndarray) -> float:
    """点到一组二维点凸包的有符号距离：负值表示在内部。

    比"是否在支撑多边形内"的布尔量信息更多。行走本质上是受控的失衡过程，单支撑
    相里 ZMP 短暂越过脚缘是正常现象，用带符号的裕度才能看清程度。
    """
    if len(pts) < 3:
        return float("nan")
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(pts)
    except QhullError:
        return float("nan")  # 退化情形（接触点共线）
    # equations 为 [n, d]，满足 n·x + d <= 0 时点在内部
    return float(np.max(hull.equations[:, :2] @ p + hull.equations[:, 2]))


def com_zmp(com: np.ndarray, dt: float, floor: float = 0.0, smooth_window: int = 9) -> np.ndarray:
    """由质心轨迹求 ZMP（忽略角动量变化率的标准近似）。

    .. math::

        x_{zmp} = x_{com} - \\frac{(z_{com}-z_f)\\,\\ddot{x}_{com}}{\\ddot{z}_{com} + g}
    """
    c = _smooth(np.asarray(com, dtype=np.float64), smooth_window)
    acc = np.zeros_like(c)
    if len(c) > 2:
        acc[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) / (dt**2)
        acc[0], acc[-1] = acc[1], acc[-2]
    acc = _smooth(acc, smooth_window)
    denom = acc[:, 2] + GRAVITY
    denom = np.where(np.abs(denom) < 1e-3, 1e-3, denom)
    h = c[:, 2] - floor
    return np.stack([c[:, 0] - h * acc[:, 0] / denom, c[:, 1] - h * acc[:, 1] / denom], axis=1)


def validate_motion(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
    compute_support: bool = True,
    smooth_window: int = 9,
) -> ValidationResult:
    """对整段重定向结果做动力学校验。"""
    dt = 1.0 / fps
    data = mujoco.MjData(model)
    n = qpos.shape[0]
    qvel, qacc = differentiate_configuration(model, qpos, dt, smooth_window)

    torque = np.zeros((n, model.nv - 6))
    com = np.zeros((n, 3))
    foot_h = np.zeros(n)
    base_res = np.zeros(n)

    # 反动力学期间关掉接触/约束求解。参考轨迹的足部会有毫米级穿透，若保留接触，
    # mj_inverse 会把巨大的穿透恢复力算进 qfrc_inverse，力矩量级完全失真。关掉之后
    # qfrc_inverse[6:] 就是纯粹的所需关节力矩，qfrc_inverse[:6] 则是必须由地面
    # 反作用力提供的基座外力旋量。
    saved_flags = model.opt.disableflags
    model.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    )
    try:
        for k in range(n):
            data.qpos[:] = qpos[k]
            data.qvel[:] = qvel[k]
            data.qacc[:] = qacc[k]
            mujoco.mj_inverse(model, data)
            f = data.qfrc_inverse
            base_res[k] = float(np.linalg.norm(f[:6]))
            torque[k] = f[6:]
            com[k] = data.subtree_com[0]
            foot_h[k] = lowest_surface_z(model, data)
    finally:
        model.opt.disableflags = saved_flags

    zmp = com_zmp(com, dt, smooth_window=smooth_window)

    support_ok = np.zeros(n, dtype=bool)
    support_size = np.zeros(n, dtype=np.int64)
    margin = np.full(n, np.nan)
    if compute_support:
        for k in range(n):
            data.qpos[:] = qpos[k]
            mujoco.mj_kinematics(model, data)
            pts = foot_contact_points(model, data)
            support_size[k] = len(pts)
            margin[k] = hull_signed_distance(zmp[k], pts)
            support_ok[k] = bool(margin[k] < 0)

    return ValidationResult(
        torque=torque, com=com, zmp=zmp, foot_height=foot_h,
        support_ok=support_ok, support_size=support_size,
        zmp_margin=margin, base_residual=base_res,
    )


def replay_with_pd(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
    kp: float = 400.0,
    kd: float = 20.0,
    max_seconds: float | None = None,
    fall_height: float = 0.35,
) -> tuple[dict[str, float], np.ndarray]:
    """用 PD 跟踪在 ``mj_step`` 里开环回放参考轨迹。

    机器人 MJCF 没有 actuator，这里把 PD 力矩直接写进 ``data.qfrc_applied`` 的关节
    部分，自由基座不施加任何外力，所以这是一次真实的欠驱动仿真。没有平衡控制器，
    因此**跌倒是预期结果**；这里报告的是"在跌倒前能跟住多久"这一可行性指标。

    Returns:
        ``(指标, 逐帧仿真 qpos)``。后者形如 ``(帧数, nq)``，只到跌倒那一帧为止，
        可直接喂给 ``umr.sim.views.launch_replay_viewer`` 与参考轨迹并排回看。
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)

    sim_dt = model.opt.timestep
    # qfrc_applied 里的阻尼项是显式积分的，稳定条件为 kd*dt/M < 2（M 为关节空间惯量
    # 对角元）。小惯量关节会让一个固定的 kd 直接发散——armature 只有 4e-4 的头部关节
    # 在 kd=20 时 kd*dt/M≈16，第 8 个仿真步就 NaN。按自由度截断到稳定域内，惯量足够
    # 大的关节不受影响。
    #
    # 注意这一步救不了完全没写 armature 的模型：腕部这种自由度的 M0 只有 3.7e-4 时，
    # 光重力就有 127 rad/s^2，两项增益怎么截断都会发散。那属于模型缺转子惯量，只能在
    # MJCF 里补——G1 的官方文件一个 armature 都没写，本仓库的副本已就地补上。
    kd_vec = np.minimum(kd, 1.8 * model.dof_M0[6:] / sim_dt)
    steps_per_frame = max(1, int(round((1.0 / fps) / sim_dt)))
    n = qpos.shape[0] if max_seconds is None else min(qpos.shape[0], int(max_seconds * fps))

    tracking = []
    rollout = np.empty((n, model.nq))
    fell_at = -1
    for k in range(n):
        ref = qpos[k, 7:]
        for _ in range(steps_per_frame):
            data.qfrc_applied[6:] = kp * (ref - data.qpos[7:]) - kd_vec * data.qvel[6:]
            mujoco.mj_step(model, data)
        rollout[k] = data.qpos
        tracking.append(float(np.abs(data.qpos[7:] - ref).mean()))
        if data.qpos[2] < fall_height:
            fell_at = k
            break

    stats = {
        "frames_simulated": len(tracking),
        "frames_total": n,
        "fell_at_frame": fell_at,
        "survived_seconds": len(tracking) / fps,
        "joint_tracking_error_rad": float(np.mean(tracking)) if tracking else float("nan"),
        "final_base_height": float(data.qpos[2]),
    }
    return stats, rollout[: len(tracking)]
