"""重定向质量指标。

包含论文 Table I 形式的耗时/吞吐统计，以及针对本次复现的表面匹配、接触与
可行性指标。
"""

from __future__ import annotations

import numpy as np


def summarize(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64) * scale
    return {
        "min": float(v.min()),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def foot_ground_metrics(
    heights: np.ndarray, contact_threshold: float = 0.02
) -> dict[str, float]:
    """足部离地统计。

    Args:
        heights: (T,) 每帧机器人足部最低点的高度（米）。
        contact_threshold: 认定"着地"的高度阈值。
    """
    h = np.asarray(heights, dtype=np.float64)
    penetration = np.clip(-h, 0.0, None)
    return {
        "penetration_mean_mm": float(penetration.mean() * 1000),
        "penetration_max_mm": float(penetration.max() * 1000),
        "float_mean_mm": float(np.clip(h, 0.0, None).mean() * 1000),
        "contact_ratio": float((h < contact_threshold).mean()),
    }


def joint_limit_violation(
    qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-6
) -> dict[str, float]:
    """关节限位违反统计（自由基座部分为 +-inf，自动不参与）。"""
    q = np.asarray(qpos, dtype=np.float64)
    lim = np.isfinite(lower) & np.isfinite(upper)
    if not lim.any():
        return {"violation_ratio": 0.0, "max_violation_rad": 0.0}
    below = np.clip(lower[lim] - q[:, lim], 0.0, None)
    above = np.clip(q[:, lim] - upper[lim], 0.0, None)
    viol = np.maximum(below, above)
    return {
        "violation_ratio": float((viol > tol).mean()),
        "max_violation_rad": float(viol.max()),
    }


def smoothness(qpos: np.ndarray, fps: float) -> dict[str, float]:
    """关节速度/加速度的幅值，用于评估时间一致性。"""
    q = np.asarray(qpos, dtype=np.float64)[:, 7:]  # 跳过自由基座
    if q.shape[0] < 3:
        return {"joint_vel_mean": 0.0, "joint_acc_mean": 0.0, "joint_vel_max": 0.0}
    v = np.diff(q, axis=0) * fps
    a = np.diff(v, axis=0) * fps
    return {
        "joint_vel_mean": float(np.abs(v).mean()),
        "joint_vel_max": float(np.abs(v).max()),
        "joint_acc_mean": float(np.abs(a).mean()),
    }


def foot_tracking(human_height: np.ndarray, robot_height: np.ndarray) -> dict[str, float]:
    """逐脚的离地高度跟踪质量。

    这是判断步态是否被真实复现的关键指标。注意**不要**用"两脚中较低者"的高度做
    相关性分析：支撑脚高度几乎恒为 0，那样测到的只是噪声。这里对左右脚分别计算。

    Args:
        human_height: (T, 2) 源动作左右脚底最低点高度（米）。
        robot_height: (T, 2) 重定向结果的对应量。
    """
    h = np.asarray(human_height, dtype=np.float64)
    r = np.asarray(robot_height, dtype=np.float64)
    out: dict[str, float] = {}
    for i, side in enumerate(("left", "right")):
        out[f"{side}_corr"] = float(np.corrcoef(h[:, i], r[:, i])[0, 1])
        out[f"{side}_rmse_mm"] = float(np.sqrt(((h[:, i] - r[:, i]) ** 2).mean()) * 1000)
    out["swing_range_mm"] = float((h.max(axis=0) - h.min(axis=0)).mean() * 1000)
    return out


def format_table(rows: list[tuple[str, str, str]], title: str = "") -> str:
    """渲染论文 Table I 风格的三列表格。"""
    w0 = max(len(r[0]) for r in rows) + 2
    w1 = max(len(r[1]) for r in rows) + 2
    w2 = max(len(r[2]) for r in rows) + 2
    line = "-" * (w0 + w1 + w2)
    out = []
    if title:
        out += [title, line]
    out.append(f"{'Stage':<{w0}}{'System Component':<{w1}}{'Computational Cost':<{w2}}")
    out.append(line)
    for a, b, c in rows:
        out.append(f"{a:<{w0}}{b:<{w1}}{c:<{w2}}")
    out.append(line)
    return "\n".join(out)
