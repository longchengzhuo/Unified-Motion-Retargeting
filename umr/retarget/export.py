"""重定向结果的导出。

pkl 采用与 `GMR <https://github.com/YanjieZe/GMR>`_ 相同的字段命名
（``root_trans`` / ``root_rot`` / ``dof`` /
``dof_full`` / ``fps`` / ``dof_names`` / ``body_names``），因此可以直接喂给已有的
下游工具；另外附带 UMR 自己的 ``qpos`` 与质量指标。

注意 ``root_rot`` 按下游约定存成 **xyzw**，而 MuJoCo 的 qpos 里是 **wxyz**。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np


def joint_names(model: mujoco.MjModel) -> list[str]:
    """按 qpos 顺序列出各铰接关节名（跳过自由基座）。"""
    names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}")
    return names


def body_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        for b in range(1, model.nbody)
    ]


def motion_to_dict(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
    extra: dict | None = None,
) -> dict:
    """把 qpos 序列整理成与 GMR 兼容的字典。"""
    qpos = np.asarray(qpos, dtype=np.float64)
    out = {
        "root_trans": qpos[:, 0:3].copy(),
        "root_rot": qpos[:, [4, 5, 6, 3]].copy(),  # wxyz -> xyzw
        "dof": qpos[:, 7:].copy(),
        "dof_full": qpos[:, 7:].copy(),
        "fps": float(fps),
        "dof_names": joint_names(model),
        "body_names": body_names(model),
        "qpos": qpos.copy(),  # MuJoCo 原始布局（root_rot 为 wxyz）
    }
    if extra:
        out.update(extra)
    return out


def save_motion_pkl(
    path: str | Path,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(motion_to_dict(model, qpos, fps, extra), f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_motion_pkl(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
