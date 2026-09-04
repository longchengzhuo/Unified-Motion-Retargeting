#!/usr/bin/env python
"""可视化：人机并排对比视频、T-pose 对应关系着色图、交互式 viewer。

- ``--mode video``   人体（源）与机器人（重定向结果）并排，可叠加对应点
- ``--mode corr``    Fig.2 风格的 T-pose 对应关系图，按人体分段着色
- ``--mode viewer``  MuJoCo 交互式播放
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from umr.bodies import bvh as bvh_mod
from umr.bodies.human_mjcf import HumanBody
from umr.bodies.robot import RobotBody
from umr.bodies.surface import SurfacePointCloud, transport_points
from umr.paths import OUTPUT_DIR
from umr.retarget.binding import LinkBinding
from umr.sim.combined import build_combined_scene
from umr.sim.interactive import FrameScrubber
from umr.sim.render import SceneRenderer, label_strip


def segment_palette(n: int) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("turbo")
    return np.array([cmap(i / max(n - 1, 1)) for i in range(n)])


def load_all(args):
    bodies = np.load(args.bodies, allow_pickle=True)
    corr = np.load(args.correspondence, allow_pickle=True)
    motion_npz = np.load(args.motion, allow_pickle=True) if Path(args.motion).exists() else None

    robot = RobotBody.from_bodies(bodies)
    bvh = bvh_mod.load_bvh(str(bodies["bvh_path"]))
    human = HumanBody(str(bodies["human_xml"]), bvh, scale=float(bodies["scale"]))
    human.root_offset = np.array([0.0, 0.0, float(bodies["ground_offset"])])
    return bodies, corr, motion_npz, robot, human


def mode_video(args) -> None:
    import imageio

    bodies, corr, motion, robot, human = load_all(args)
    if motion is None:
        raise SystemExit(f"找不到重定向结果: {args.motion}，请先跑 03_retarget.py")

    human_pc = SurfacePointCloud.from_dict(bodies, "human_")
    binding = LinkBinding.from_dict(corr, "bind_")
    selected = motion["selected"]
    qpos = motion["qpos"]
    frames = motion["frame_indices"]
    fps = float(motion["fps"])

    n = len(frames) if args.max_frames is None else min(len(frames), args.max_frames)
    step = max(1, args.stride)

    hr = SceneRenderer(human.model, args.width, args.height, distance=args.distance)
    rr = SceneRenderer(robot.model, args.width, args.height, distance=args.distance)
    lbl_h = label_strip(args.width, 34, "Source: Xsens BVH (rigged human surface)")
    lbl_r = label_strip(args.width, 34, f"UMR retargeted: {robot.spec.name}")

    writer = imageio.get_writer(args.out, fps=fps / step, macro_block_size=1)
    t0 = time.perf_counter()
    for k in range(0, n, step):
        human.set_frame(int(frames[k]))
        robot.set_qpos(qpos[k])

        hpos, _ = transport_points(human.data, human_pc.body_ids, human_pc.local_pos)
        rpos, _ = transport_points(robot.data, binding.body_ids, binding.local_pos)

        center = human.data.qpos[:3].copy()
        center[2] = 0.85
        rcenter = robot.data.qpos[:3].copy()
        rcenter[2] = 0.85

        hm = [(hpos[selected], (0.95, 0.25, 0.25, 1.0), 0.011)] if args.points else None
        rm = [(rpos[selected], (0.2, 0.9, 0.35, 1.0), 0.011)] if args.points else None
        img_h = hr.render(human.data, center, hm)
        img_r = rr.render(robot.data, rcenter, rm)
        writer.append_data(
            np.hstack([np.vstack([lbl_h, img_h]), np.vstack([lbl_r, img_r])])
        )
    writer.close()
    print(f"[viz] {args.out}  ({n//step} 帧, {time.perf_counter()-t0:.1f}s)")


def mode_corr(args) -> None:
    import imageio

    bodies, corr, _, robot, human = load_all(args)
    human_pc = SurfacePointCloud.from_dict(bodies, "human_")
    binding = LinkBinding.from_dict(corr, "bind_")
    segment = corr["inherited_segment"]
    names = [str(s) for s in corr["segment_names"]]

    human.set_tpose()
    robot.set_tpose()
    hpos, _ = transport_points(human.data, human_pc.body_ids, human_pc.local_pos)
    rpos, _ = transport_points(robot.data, binding.body_ids, binding.local_pos)

    palette = segment_palette(len(names))
    sub = np.arange(0, len(segment), max(1, len(segment) // 1500))

    hr = SceneRenderer(human.model, args.width, args.height, distance=2.9, azimuth=170)
    rr = SceneRenderer(robot.model, args.width, args.height, distance=2.9, azimuth=170)

    def markers(pts):
        out = []
        for s in np.unique(segment[sub]):
            if s < 0:
                continue
            m = sub[segment[sub] == s]
            out.append((pts[m], tuple(palette[int(s)]), 0.014))
        return out

    img_h = hr.render(human.data, np.array([0, 0, 0.85]), markers(hpos))
    img_r = rr.render(robot.data, np.array([0, 0, 0.80]), markers(rpos))
    lbl_h = label_strip(args.width, 34, "Human template  X^h  (ordered, segment-coloured)")
    lbl_r = label_strip(args.width, 34, "Learned correspondence  X^r  (same ordering)")
    out = np.hstack([np.vstack([lbl_h, img_h]), np.vstack([lbl_r, img_r])])
    imageio.imwrite(args.out, out)
    print(f"[viz] {args.out}  分段着色的人机对应点（共享下标）")


def mode_viewer(args) -> None:
    """实时逐帧播放器：起始停在第 0 帧，按住方向键播放/回退，松手暂停。"""
    bodies, corr, motion, robot, human = load_all(args)
    if motion is None:
        raise SystemExit(f"找不到重定向结果: {args.motion}，请先跑 03_retarget.py")

    qpos = motion["qpos"]
    frames = motion["frame_indices"]
    fps = float(motion["fps"])
    human_pc = SurfacePointCloud.from_dict(bodies, "human_")
    binding = LinkBinding.from_dict(corr, "bind_")
    selected = motion["selected"]

    track = robot.spec.root_body
    if args.robot_only:
        model = robot.model

        def apply_frame(data, k):
            data.qpos[:] = qpos[k]
    else:
        scene = build_combined_scene(
            str(bodies["robot_xml"]), str(bodies["human_xml"]),
            human_offset=(0.0, args.human_offset, 0.0),
        )
        model = scene.model

        def apply_frame(data, k):
            scene.set_qpos(data, qpos[k], human.qpos_for_frame(int(frames[k])))

    # 对应点叠加：人体目标点（红）与机器人对应点（绿）
    def markers(k):
        human.set_frame(int(frames[k]))
        hpos, _ = transport_points(human.data, human_pc.body_ids, human_pc.local_pos)
        robot.set_qpos(qpos[k])
        rpos, _ = transport_points(robot.data, binding.body_ids, binding.local_pos)
        shift = np.array([0.0, args.human_offset, 0.0]) if not args.robot_only else np.zeros(3)
        return [
            (hpos[selected] + shift, (0.95, 0.25, 0.25, 1.0), 0.011),
            (rpos[selected], (0.2, 0.9, 0.35, 1.0), 0.011),
        ]

    print(
        f"[viz] 实时播放器：{len(qpos)} 帧 @ {fps:.1f} FPS，起始为第 0 帧\n"
        "      按住 → 正向播放，按住 ← 反向回退，松手暂停\n"
        "      空格=自动播放  . / ,=单步  Home/End=首尾  [ / ]=调速\n"
        "      T=相机跟随  P=对应点  Esc=退出"
    )
    FrameScrubber(
        model, len(qpos), apply_frame, fps=fps,
        markers=markers if args.points else None,
        track_body=track,
        width=args.window_width, height=args.window_height,
        title=f"UMR  {Path(str(bodies['bvh_path'])).name} -> {robot.spec.name}",
    ).run()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["video", "corr", "viewer"], default="video")
    ap.add_argument("--bodies", default=str(OUTPUT_DIR / "bodies.npz"))
    ap.add_argument("--correspondence", default=str(OUTPUT_DIR / "correspondence.npz"))
    ap.add_argument("--motion", default=str(OUTPUT_DIR / "motion.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--distance", type=float, default=3.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--points", action="store_true", help="叠加对应点")
    # viewer 模式
    ap.add_argument("--robot_only", action="store_true", help="viewer 只显示机器人")
    ap.add_argument("--human_offset", type=float, default=1.2,
                    help="viewer 中人体沿 +Y 的摆放偏移（米）")
    ap.add_argument("--window_width", type=int, default=1280)
    ap.add_argument("--window_height", type=int, default=800)
    args = ap.parse_args()

    if args.out is None:
        args.out = str(
            OUTPUT_DIR / {"video": "retarget.mp4", "corr": "correspondence.png", "viewer": ""}[args.mode]
        )

    {"video": mode_video, "corr": mode_corr, "viewer": mode_viewer}[args.mode](args)


if __name__ == "__main__":
    main()
