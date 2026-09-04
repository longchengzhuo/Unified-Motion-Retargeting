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

from umr.bodies.robot import RobotSpec, lowest_surface_z

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


@dataclass
class FootGeometry:
    """足部接触候选点，按 geom 局部坐标缓存一次，之后每帧只做一次刚体变换。"""

    geom_ids: np.ndarray   # (P,)
    local: np.ndarray      # (P, 3) geom 局部坐标
    radius: np.ndarray     # (P,)   球体的半径偏移，其余为 0


def foot_geometry(model: mujoco.MjModel, spec: RobotSpec) -> FootGeometry:
    """收集足部 geom 上能真正落地的极值点。

    box 取 8 个角点、sphere 取球心配半径偏移、mesh 取凸包顶点。三者都要覆盖：有的机型
    直接用 box 建足底，有的（如 Unitree G1）足底只有视觉网格、碰撞球嵌在网格里面 15 mm 处，
    只看基元就会得到"整段动作都腾空"的假象。
    """
    from scipy.spatial import ConvexHull, QhullError

    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    geom_ids: list[int] = []
    local: list[np.ndarray] = []
    radius: list[float] = []

    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if spec.is_marker(name) or not any(k in name for k in spec.foot_name_keys):
            continue

        size = model.geom_size[g]
        gtype = int(model.geom_type[g])
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            pts, r = signs * size[:3], 0.0
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts, r = np.zeros((1, 3)), float(size[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            v0, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            verts = model.mesh_vert[v0 : v0 + n].reshape(-1, 3)
            try:
                pts = verts[ConvexHull(verts).vertices]
            except (QhullError, ValueError):
                pts = verts
            r = 0.0
        else:
            continue

        geom_ids.extend([g] * len(pts))
        local.extend(pts)
        radius.extend([r] * len(pts))

    if not geom_ids:
        raise ValueError(f"找不到足部几何：foot_name_keys={spec.foot_name_keys}")
    return FootGeometry(
        geom_ids=np.asarray(geom_ids, dtype=np.int64),
        local=np.asarray(local, dtype=np.float64),
        radius=np.asarray(radius, dtype=np.float64),
    )


def foot_contact_points(
    data: mujoco.MjData, foot: FootGeometry, height_tol: float = 0.02
) -> np.ndarray:
    """当前姿态下贴近地面的足部极值点，投影到水平面。

    用极值点而不是 geom 中心，这样单脚支撑时支撑多边形仍然有宽度，不会退化成一条线。
    """
    R = data.geom_xmat[foot.geom_ids].reshape(-1, 3, 3)
    world = np.einsum("pij,pj->pi", R, foot.local) + data.geom_xpos[foot.geom_ids]
    return world[world[:, 2] - foot.radius < height_tol][:, :2]


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
    spec: RobotSpec,
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
            foot_h[k] = lowest_surface_z(model, data, spec)
    finally:
        model.opt.disableflags = saved_flags

    zmp = com_zmp(com, dt, smooth_window=smooth_window)

    support_ok = np.zeros(n, dtype=bool)
    support_size = np.zeros(n, dtype=np.int64)
    margin = np.full(n, np.nan)
    if compute_support:
        foot = foot_geometry(model, spec)
        for k in range(n):
            data.qpos[:] = qpos[k]
            mujoco.mj_kinematics(model, data)
            pts = foot_contact_points(data, foot)
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
    stiffness: float = 1600.0,
    damping_ratio: float = 1.0,
    sim_dt: float = 5e-4,
    max_seconds: float | None = None,
    fall_height: float = 0.35,
) -> dict[str, float]:
    """用 PD 跟踪在 ``mj_step`` 里开环回放参考轨迹。

    机器人 MJCF 没有 actuator，这里把 PD 力矩直接写进 ``data.qfrc_applied`` 的关节
    部分，自由基座不施加任何外力，所以这是一次真实的欠驱动仿真。没有平衡控制器，
    因此**跌倒是预期结果**；这里报告的是"在跌倒前能跟住多久"这一可行性指标。

    增益按各关节的**有效惯量** :math:`m^{eff}_j = 1/(M^{-1})_{jj}` 归一化：
    ``kp = stiffness * m_eff``、``kd = 2ζ·sqrt(stiffness)·m_eff``，于是每个关节的闭环
    固有频率都是 ``sqrt(stiffness)``，与连杆质量无关。用绝对增益或 ``M_jj`` 对角元都不
    行——浮动基运动链的耦合会让 ``M_jj`` 高估惯量最多一个量级，轻手腕上会直接发散。

    只保留机器人与地面的接触。重定向默认不做自碰撞规避（``retarget.self_collision``），
    参考轨迹里手腕贴着髋部这类厘米级自穿透是允许的；若把它们喂给接触求解器，第一步就会
    产生巨大的恢复力，测到的是初值伪影而不是动力学可行性。

    积分步长取 ``min(sim_dt, MJCF 自带的 timestep)``。脚踝一类连杆的惯量只有 1e-4 量级，
    若模型没写 ``armature``（Unitree G1 就没有），2 ms 的显式积分一碰地就会发散。
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]

    is_world = model.geom_bodyid == 0
    saved = model.geom_contype.copy(), model.geom_conaffinity.copy(), model.opt.timestep
    model.geom_contype[:] = 1
    model.geom_conaffinity[:] = np.where(is_world, 1, 0)
    model.opt.timestep = min(sim_dt, model.opt.timestep)

    try:
        mujoco.mj_forward(model, data)

        inertia = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, inertia, data.qM)
        m_eff = 1.0 / np.diag(np.linalg.inv(inertia))[6:]
        kp = stiffness * m_eff
        kd = 2.0 * damping_ratio * np.sqrt(stiffness) * m_eff

        steps_per_frame = max(1, int(round((1.0 / fps) / model.opt.timestep)))
        n = qpos.shape[0] if max_seconds is None else min(qpos.shape[0], int(max_seconds * fps))

        tracking = []
        fell_at = -1
        for k in range(n):
            ref = qpos[k, 7:]
            for _ in range(steps_per_frame):
                data.qfrc_applied[6:] = kp * (ref - data.qpos[7:]) - kd * data.qvel[6:]
                mujoco.mj_step(model, data)
            tracking.append(float(np.abs(data.qpos[7:] - ref).mean()))
            if data.qpos[2] < fall_height:
                fell_at = k
                break
    finally:
        model.geom_contype[:], model.geom_conaffinity[:], model.opt.timestep = saved

    return {
        "frames_simulated": len(tracking),
        "frames_total": n,
        "fell_at_frame": fell_at,
        "survived_seconds": len(tracking) / fps,
        "joint_tracking_error_rad": float(np.mean(tracking)) if tracking else float("nan"),
        "final_base_height": float(data.qpos[2]),
    }
