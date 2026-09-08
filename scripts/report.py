#!/usr/bin/env python
"""校验重定向结果并生成指标报告（默认不开窗口）。

用法::

    python scripts/report.py --motion_file data/walk_slow.bvh
    python scripts/report.py --motion_file data/my_session --replay
    python scripts/report.py --motion_file data/x.bvh --video --corr_image
    python scripts/report.py --motion_file data/x.bvh --replay_viewer

入参与 ``scripts/retarget.py`` 一致，据此定位它产出的结果，写出 ``report.md`` 与
``metrics.npz``（在片段的同名子目录里）。加 ``--replay_viewer`` 会在报告写完后开一
个窗口，把运动学参考和 PD 开环仿真的结果并排播出来。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umr.bootstrap import prepare_runtime

# 默认走 EGL 离屏，无显示器的机器上也能出报告；只有要开窗口时才换成 glfw。
# 后端在 import mujoco 之前就得定下来，所以这里只能先扫一眼 argv。
prepare_runtime("glfw" if "--replay_viewer" in sys.argv else "egl")

import argparse

from umr.cli import add_target_args, resolve_target
from umr.report import evaluate_run


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_target_args(ap)

    g = ap.add_argument_group("校验")
    g.add_argument("--replay", action="store_true", help="额外做 PD 开环回放仿真")
    g.add_argument("--replay_viewer", action="store_true",
                   help="开窗并排回看 PD 回放（运动学参考 + 仿真实际），隐含 --replay")
    g.add_argument("--replay_offset", type=float, default=1.2,
                   help="并排间距（米），0 表示两者重叠")
    g.add_argument("--replay_seconds", type=float, default=5.0)
    g.add_argument("--max_frames", type=int, default=None, help="限制校验帧数以加快速度")

    g = ap.add_argument_group("附带渲染（离屏，不开窗口）")
    g.add_argument("--corr_image", action="store_true",
                   help="输出 T-pose 对应关系图 correspondence.png")
    g.add_argument("--record_video", action="store_true",
                   help="输出人机并排对比视频（与片段同名的 .mp4）")
    g.add_argument("--video_stride", type=int, default=2, help="视频抽帧步长")
    g.add_argument("--video_max_frames", type=int, default=900)
    g.add_argument("--width", type=int, default=560)
    g.add_argument("--height", type=int, default=560)
    g.add_argument("--no_points", action="store_true", help="视频里不叠加对应点")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg, skeleton, clips = resolve_target(args)
    print(f"[run] {cfg.robot['name']}  <-  {skeleton.name}  x  {len(clips)} 段")

    missing = [c for c in clips if not c.layout.motion.exists()]
    todo = [c for c in clips if c.layout.motion.exists()]
    for c in missing:
        print(f"[skip] {c.label} 还没有重定向结果，先跑 scripts/retarget.py")
    if not todo:
        raise SystemExit("没有可校验的结果")

    for clip in todo:
        print(f"\n{'=' * 72}\n[clip] {clip.label}\n{'=' * 72}")
        ev = evaluate_run(
            clip.layout,
            replay_seconds=args.replay_seconds if (args.replay or args.replay_viewer) else None,
            max_frames=args.max_frames,
        )

        if args.corr_image or args.record_video:
            from umr.paths import SetupLayout
            from umr.sim.views import load_run, render_correspondence, render_video

            run = load_run(clip.layout)
            if args.corr_image:
                setup = SetupLayout(Path(str(run.motion["setup_dir"])))
                ev.figures.append(
                    render_correspondence(
                        run, setup.corr_image, width=args.width, height=args.height
                    )
                )
            if args.record_video:
                ev.figures.append(
                    render_video(
                        run, clip.layout.video, width=args.width, height=args.height,
                        stride=args.video_stride, max_frames=args.video_max_frames,
                        points=not args.no_points,
                    )
                )

        ev.save()
        print(f"\n[save] {clip.layout.report}\n[save] {clip.layout.metrics}")

        # 放在保存之后：播放器会一直阻塞到用户关窗，报告不该等它。
        if args.replay_viewer and ev.replay_qpos is not None and len(ev.replay_qpos):
            from umr.sim.views import launch_replay_viewer, load_run

            launch_replay_viewer(
                load_run(clip.layout), ev.replay_qpos, offset=args.replay_offset
            )


if __name__ == "__main__":
    main()
