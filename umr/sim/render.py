"""MuJoCo 离线渲染：人机并排对比视频与静态对比图。"""

from __future__ import annotations

import numpy as np

import mujoco


def add_point_markers(
    scene: "mujoco.MjvScene",
    points: np.ndarray,
    rgba: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 1.0),
    size: float = 0.012,
) -> None:
    """把一组点作为小球叠加到已有场景上。"""
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([size, 0.0, 0.0]),
            pos=np.asarray(p, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


class SceneRenderer:
    """单个模型的跟随式渲染器。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 640,
        height: int = 480,
        distance: float = 3.0,
        elevation: float = -12.0,
        azimuth: float = 120.0,
        max_markers: int = 2048,
    ):
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width)
        self.renderer.scene.maxgeom = max(self.renderer.scene.maxgeom, max_markers)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = distance
        self.camera.elevation = elevation
        self.camera.azimuth = azimuth

    def render(
        self,
        data: mujoco.MjData,
        lookat: np.ndarray | None = None,
        markers: list[tuple[np.ndarray, tuple, float]] | None = None,
    ) -> np.ndarray:
        if lookat is not None:
            self.camera.lookat[:] = lookat
        self.renderer.update_scene(data, self.camera)
        for pts, rgba, size in markers or []:
            add_point_markers(self.renderer.scene, pts, rgba, size)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def label_strip(width: int, height: int, text: str) -> np.ndarray:
    """用 matplotlib 生成一条文字条（避免额外引入字体依赖）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("black")
    fig.text(0.5, 0.5, text, color="white", ha="center", va="center", fontsize=height * 0.42)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
