"""把机器人与源人体合并进同一个 MuJoCo 场景，便于实时并排对照。

用 ``mujoco.MjSpec`` 把人体模型以 ``H_`` 前缀挂到机器人的 worldbody 上，得到一个
包含两副骨架的模型：qpos 前段是机器人（nq=34），后段是人体（nq=95）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HUMAN_PREFIX = "H_"


@dataclass
class CombinedScene:
    """人机合并后的模型及两侧的 qpos 切片。"""

    model: mujoco.MjModel
    robot_slice: slice
    human_slice: slice
    human_offset: np.ndarray
    human_prefix: str = HUMAN_PREFIX

    def set_qpos(self, data: mujoco.MjData, robot_qpos: np.ndarray, human_qpos: np.ndarray) -> None:
        """写入两侧 qpos。

        人体的根是 free joint，其 qpos 直接决定世界位姿，会覆盖 attach 时的静态
        frame 偏移，所以摆放偏移必须加在根平移上。
        """
        data.qpos[self.robot_slice] = robot_qpos
        human_qpos = np.asarray(human_qpos, dtype=np.float64).copy()
        human_qpos[0:3] += self.human_offset
        data.qpos[self.human_slice] = human_qpos

    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)


def build_combined_scene(
    robot_xml: str | Path,
    human_xml: str | Path,
    human_offset: tuple[float, float, float] = (0.0, 1.2, 0.0),
    human_prefix: str = HUMAN_PREFIX,
) -> CombinedScene:
    """合并两个 MJCF。

    Args:
        robot_xml, human_xml: 两侧模型路径。
        human_offset: 人体的摆放偏移。取 ``(0, 1.2, 0)`` 是并排对照，取 ``(0, 0, 0)``
            则两者重叠，可以直接看贴合程度。
        human_prefix: 人体一侧所有名字的前缀。
    """
    robot_spec = mujoco.MjSpec.from_file(str(robot_xml))
    human_spec = mujoco.MjSpec.from_file(str(human_xml))

    n_robot = mujoco.MjModel.from_xml_path(str(robot_xml)).nq
    n_human = mujoco.MjModel.from_xml_path(str(human_xml)).nq

    robot_spec.attach(human_spec, prefix=human_prefix, frame=robot_spec.worldbody.add_frame())
    model = robot_spec.compile()

    if model.nq != n_robot + n_human:
        raise RuntimeError(
            f"合并后 nq={model.nq}，期望 {n_robot}+{n_human}={n_robot + n_human}"
        )
    return CombinedScene(
        model=model,
        robot_slice=slice(0, n_robot),
        human_slice=slice(n_robot, n_robot + n_human),
        human_offset=np.asarray(human_offset, dtype=np.float64),
        human_prefix=human_prefix,
    )
