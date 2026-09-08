"""三种看结果的方式：交互式播放器、人机并排视频、T-pose 对应关系图。

播放器开的是真实的 GLFW 窗口；另外两个走 MuJoCo 的离屏渲染，只写文件。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from umr.bodies import bvh as bvh_mod
from umr.bodies.human_mjcf import HumanBody
from umr.bodies.robot import RobotBody
from umr.bodies.surface import SurfacePointCloud, transport_points
from umr.paths import ClipLayout, SetupLayout
from umr.retarget.binding import LinkBinding
from umr.sim.combined import build_combined_scene
from umr.sim.interactive import FrameScrubber
from umr.sim.render import SceneRenderer, label_strip

HUMAN_RGBA = (0.95, 0.25, 0.25, 1.0)
ROBOT_RGBA = (0.20, 0.90, 0.35, 1.0)


@dataclass
class LoadedRun:
    """一次运行的全部产物，外加已经建好的人机模型。"""

    bodies: np.lib.npyio.NpzFile
    corr: np.lib.npyio.NpzFile
    motion: np.lib.npyio.NpzFile
    robot: RobotBody
    human: HumanBody
    human_pc: SurfacePointCloud
    binding: LinkBinding

    @property
    def robot_name(self) -> str:
        return str(self.bodies["robot_name"])

    @property
    def human_skeleton(self) -> str:
        return str(self.motion["human_skeleton"])

    @property
    def bvh_name(self) -> str:
        return Path(str(self.motion["bvh_path"])).name

    @property
    def root_body(self) -> str:
        """机器人根 body 名，用作相机跟随目标（body 1 是 worldbody 的唯一子体）。

        写死 "base_link" 的话，根叫 pelvis 的 G1 会静悄悄地跟随不上。
        """
        return self.robot.body_names[1]

    def camera_frame(self) -> tuple[float, float]:
        """按机器人身高定出相机的注视高度与距离。

        人体已被归一化到机器人身高，两侧用同一组取值即可；写死成绝对米数的话，
        换一台只有一半高的机器人就只能占到画面的一角。
        """
        h = float(self.bodies["robot_height"])
        return 0.534 * h, 1.884 * h

    def human_points(self, frame: int) -> np.ndarray:
        self.human.set_frame(int(frame))
        return transport_points(self.human.data, self.human_pc.body_ids, self.human_pc.local_pos)[0]

    def robot_points(self, qpos: np.ndarray) -> np.ndarray:
        self.robot.set_qpos(qpos)
        return transport_points(self.robot.data, self.binding.body_ids, self.binding.local_pos)[0]


def clip_from_meta(meta) -> "bvh_mod.BvhData":
    """按 ``motion.npz`` 里记录的参数复原 Stage II 实际用的那一段动作。

    帧率、裁剪区间与插值方式都存在 npz 里，所以这里重建出的片段与当初求解时逐帧
    一一对应，``frame_indices`` 可以直接用。
    """
    data = bvh_mod.load_bvh(
        str(meta["bvh_path"]),
        scale=float(meta["length_scale"]),
        auto_face_x=bool(meta["auto_face_x"]),
        human=str(meta["human_skeleton"]),
    )
    duration = float(meta["duration"])
    return bvh_mod.prepare_clip(
        data,
        tgt_fps=float(meta["fps"]),
        start=float(meta["start"]),
        duration=None if duration < 0 else duration,
        interpolation=str(meta["interpolation"]),
    )


def load_run(clip: ClipLayout, setup: SetupLayout | None = None) -> LoadedRun:
    """加载一段重定向结果，以及它所引用的那份共享 setup。"""
    if not clip.motion.exists():
        raise SystemExit(f"找不到重定向结果 {clip.motion}，请先跑 scripts/retarget.py")
    motion = np.load(clip.motion, allow_pickle=True)

    setup = setup or SetupLayout(Path(str(motion["setup_dir"])))
    for path in (setup.bodies, setup.correspondence):
        if not path.exists():
            raise SystemExit(f"找不到 {path}，请先跑 scripts/retarget.py")
    bodies = np.load(setup.bodies, allow_pickle=True)
    corr = np.load(setup.correspondence, allow_pickle=True)

    robot = RobotBody(str(motion["robot_xml"]))
    human = HumanBody(
        str(motion["human_xml"]), clip_from_meta(motion), scale=float(motion["scale"])
    )
    human.root_offset = np.array([0.0, 0.0, float(motion["ground_offset"])])
    return LoadedRun(
        bodies=bodies, corr=corr, motion=motion, robot=robot, human=human,
        human_pc=SurfacePointCloud.from_dict(bodies, "human_"),
        binding=LinkBinding.from_dict(corr, "bind_"),
    )


# ----------------------------------------------------------------------
# 交互式播放器
# ----------------------------------------------------------------------
def launch_viewer(
    run: LoadedRun,
    *,
    points: bool = True,
    robot_only: bool = False,
    human_offset: float = 1.2,
    width: int = 1280,
    height: int = 800,
    max_seconds: float | None = None,
) -> None:
    """开窗实时播放：起始停在第 0 帧，按住方向键播放/回退，松手暂停。"""
    qpos = run.motion["qpos"]
    frames = run.motion["frame_indices"]
    fps = float(run.motion["fps"])
    selected = run.motion["selected"]

    if robot_only:
        model = run.robot.model

        def apply_frame(data, k):
            data.qpos[:] = qpos[k]

        shift = np.zeros(3)
    else:
        scene = build_combined_scene(
            str(run.bodies["robot_xml"]), str(run.bodies["human_xml"]),
            human_offset=(0.0, human_offset, 0.0),
        )
        model = scene.model

        def apply_frame(data, k):
            scene.set_qpos(data, qpos[k], run.human.qpos_for_frame(int(frames[k])))

        shift = np.array([0.0, human_offset, 0.0])

    def markers(k):
        # 人体目标点（红）与机器人对应点（绿），两者共享下标
        return [
            (run.human_points(frames[k])[selected] + shift, HUMAN_RGBA, 0.011),
            (run.robot_points(qpos[k])[selected], ROBOT_RGBA, 0.011),
        ]

    print(
        f"[viewer] {len(qpos)} 帧 @ {fps:.1f} FPS，起始为第 0 帧\n"
        "         按住 → 正向播放，按住 ← 反向回退，松手暂停\n"
        "         空格=自动播放  . / ,=单步  Home/End=首尾  [ / ]=调速\n"
        "         T=相机跟随  P=对应点  Esc=退出"
    )
    FrameScrubber(
        model, len(qpos), apply_frame, fps=fps,
        markers=markers if points else None,
        track_body=run.root_body,
        width=width, height=height,
        title=f"UMR  {run.bvh_name} -> {run.robot_name}",
    ).run(max_seconds=max_seconds)


def launch_replay_viewer(
    run: LoadedRun,
    replay_qpos: np.ndarray,
    *,
    offset: float = 1.2,
    width: int = 1280,
    height: int = 800,
    max_seconds: float | None = None,
) -> None:
    """并排回看 PD 开环回放：原地的是运动学参考，沿 +Y 偏开的是仿真里真跟出来的姿态。

    两侧是同一个机器人模型的两份实例，所以看到的差异就是 PD 跟踪误差加上欠驱动
    基座的漂移。``offset=0`` 会让两者重叠，便于直接看贴合程度。仿真在跌倒帧就停了，
    因此参考轨迹按 ``replay_qpos`` 的长度截断，两侧始终对齐同一帧。
    """
    ref = run.motion["qpos"][: len(replay_qpos)]
    fps = float(run.motion["fps"])
    scene = build_combined_scene(
        run.robot.xml_path, run.robot.xml_path,
        human_offset=(0.0, offset, 0.0), human_prefix="SIM_",
    )

    def apply_frame(data, k):
        scene.set_qpos(data, ref[k], replay_qpos[k])

    print(
        f"[viewer] PD 回放 {len(ref)} 帧 @ {fps:.1f} FPS：近处=运动学参考，"
        f"偏移 {offset:g} m 处=PD 仿真\n"
        "         按住 → 正向播放，按住 ← 反向回退，松手暂停\n"
        "         空格=自动播放  . / ,=单步  Home/End=首尾  [ / ]=调速  T=相机跟随  Esc=退出"
    )
    FrameScrubber(
        scene.model, len(ref), apply_frame, fps=fps,
        track_body=run.root_body,
        width=width, height=height,
        title=f"UMR PD replay  {run.bvh_name} -> {run.robot_name}",
    ).run(max_seconds=max_seconds)


# ----------------------------------------------------------------------
# 离屏渲染
# ----------------------------------------------------------------------
def render_video(
    run: LoadedRun,
    out_path: str | Path,
    *,
    width: int = 560,
    height: int = 560,
    distance: float | None = None,
    stride: int = 2,
    max_frames: int | None = 900,
    points: bool = True,
) -> Path:
    """人体（源）与机器人（重定向结果）并排的对比视频。"""
    import imageio

    qpos = run.motion["qpos"]
    frames = run.motion["frame_indices"]
    selected = run.motion["selected"]
    fps = float(run.motion["fps"])

    n = len(frames) if max_frames is None else min(len(frames), max_frames)
    step = max(1, stride)

    look_z, auto_distance = run.camera_frame()
    distance = auto_distance if distance is None else distance
    hr = SceneRenderer(run.human.model, width, height, distance=distance)
    rr = SceneRenderer(run.robot.model, width, height, distance=distance)
    lbl_h = label_strip(
        width, 34, f"Source: {run.human_skeleton} BVH (rigged human surface)"
    )
    lbl_r = label_strip(width, 34, f"UMR retargeted: {run.robot_name}")

    writer = imageio.get_writer(out_path, fps=fps / step, macro_block_size=1)
    t0 = time.perf_counter()
    for k in range(0, n, step):
        hpos = run.human_points(frames[k])
        rpos = run.robot_points(qpos[k])

        center = run.human.data.qpos[:3].copy()
        center[2] = look_z
        rcenter = run.robot.data.qpos[:3].copy()
        rcenter[2] = look_z

        hm = [(hpos[selected], HUMAN_RGBA, 0.011)] if points else None
        rm = [(rpos[selected], ROBOT_RGBA, 0.011)] if points else None
        img_h = hr.render(run.human.data, center, hm)
        img_r = rr.render(run.robot.data, rcenter, rm)
        writer.append_data(np.hstack([np.vstack([lbl_h, img_h]), np.vstack([lbl_r, img_r])]))
    writer.close()
    print(f"[viz] {out_path}  ({n // step} 帧, {time.perf_counter() - t0:.1f}s)")
    return Path(out_path)


def render_correspondence(
    run: LoadedRun,
    out_path: str | Path,
    *,
    width: int = 560,
    height: int = 560,
    distance: float | None = None,
    max_points: int = 1500,
) -> Path:
    """论文 Fig.2 风格的 T-pose 对应关系图，人机两侧按人体分段同色着色。"""
    import imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segment = run.corr["inherited_segment"]
    names = [str(s) for s in run.corr["segment_names"]]
    cmap = plt.get_cmap("turbo")
    palette = np.array([cmap(i / max(len(names) - 1, 1)) for i in range(len(names))])

    run.human.set_tpose()
    run.robot.set_tpose()
    hpos = transport_points(run.human.data, run.human_pc.body_ids, run.human_pc.local_pos)[0]
    rpos = transport_points(run.robot.data, run.binding.body_ids, run.binding.local_pos)[0]

    sub = np.arange(0, len(segment), max(1, len(segment) // max_points))

    look_z, auto_distance = run.camera_frame()
    distance = auto_distance * 0.97 if distance is None else distance
    hr = SceneRenderer(run.human.model, width, height, distance=distance, azimuth=170)
    rr = SceneRenderer(run.robot.model, width, height, distance=distance, azimuth=170)

    def markers(pts):
        out = []
        for s in np.unique(segment[sub]):
            if s < 0:
                continue
            m = sub[segment[sub] == s]
            out.append((pts[m], tuple(palette[int(s)]), 0.014))
        return out

    img_h = hr.render(run.human.data, np.array([0, 0, look_z]), markers(hpos))
    img_r = rr.render(run.robot.data, np.array([0, 0, look_z * 0.94]), markers(rpos))
    lbl_h = label_strip(width, 34, "Human template  X^h  (ordered, segment-coloured)")
    lbl_r = label_strip(width, 34, "Learned correspondence  X^r  (same ordering)")
    imageio.imwrite(
        out_path, np.hstack([np.vstack([lbl_h, img_h]), np.vstack([lbl_r, img_r])])
    )
    print(f"[viz] {out_path}  分段着色的人机对应点（共享下标）")
    return Path(out_path)
