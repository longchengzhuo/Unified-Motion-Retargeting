#!/usr/bin/env python
"""把 BVH 动作重定向到人形机器人，然后打开交互式可视化界面。

用法::

    # 单个文件 + 交互式播放
    python scripts/retarget.py --motion_file data/walk_slow.bvh

    # 整个目录批处理，输出 50 FPS，顺便导出视频，多进程无窗口
    python scripts/retarget.py --motion_file data/my_session --human xsens \\
        --tgt_fps 50 --robot g1 --save_path output \\
        --record_video --multi_process --override

    # FZMotion 骨架
    python scripts/retarget.py --motion_file data/my_fzmotion_session --human fzmotion

依次跑 Stage 0（建人机 MuJoCo 身体 + T-pose 表面采样）、Stage I（点云对应学习，
论文式 1-5）、Stage II（对应引导的逐帧重定向，论文式 6-14）。

源帧率从 BVH 头部的 ``Frame Time`` 自动读出，无需入参；``--tgt_fps`` 与之不同时
按 slerp（旋转）+ 线性（位置）插值重采样。

Stage 0 / I 的产物按「机器人 + 源骨架 + 演员骨架尺寸」寻址共享，所以批量处理同一
个演员的几十段导出时，十来分钟的对应学习只跑一次。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umr.bootstrap import prepare_runtime

prepare_runtime()

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from umr.bodies.skeletons import get_skeleton
from umr.cli import Clip, add_target_args, resolve_save_path, resolve_target
from umr.config import load_config, robot_key
from umr.paths import SetupLayout, setup_layout
from umr.stages import (
    DEFAULT_INTERPOLATION,
    SHARE_MODES,
    build_bodies,
    learn_correspondence,
    retarget_motion,
    setup_representative,
    setup_signature,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_target_args(ap)

    g = ap.add_argument_group("帧率与片段")
    g.add_argument("--tgt_fps", type=float, default=None,
                   help="输出帧率，默认与源一致（源帧率从 BVH 的 Frame Time 自动读取）")
    g.add_argument("--interpolation_method", default=DEFAULT_INTERPOLATION,
                   choices=["slerp", "linear"],
                   help="重采样时旋转的插值方式，默认 slerp")
    g.add_argument("--start", type=float, default=0.0, help="起始时间（秒）")
    g.add_argument("--duration", type=float, default=None, help="截取时长（秒）")

    g = ap.add_argument_group("批处理")
    g.add_argument("--override", action="store_true",
                   help="输出已存在时也重新处理（默认跳过）")
    g.add_argument("--multi_process", action="store_true",
                   help="多进程跑 Stage II（无窗口；Stage 0/I 仍在主进程里只做一次）")
    g.add_argument("--num_workers", type=int, default=8,
                   help="--multi_process 的并行进程数，0 表示用全部 CPU 核心")
    g.add_argument("--share_setup", default="dir", choices=SHARE_MODES,
                   help="Stage 0/I 的共享粒度：dir=同目录共用一套对应关系（默认），"
                        "file=每个文件各算一套（目录里混了不同体型的演员时用）")

    g = ap.add_argument_group("Stage I 对应学习")
    g.add_argument("--epochs", type=int, default=None, help="训练轮数，默认取配置")
    g.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"],
                   help="训练设备，默认取配置里的 correspondence.device")

    g = ap.add_argument_group("Stage II 重定向")
    g.add_argument("--n_selected", type=int, default=None, help="式 (7) 的选中点集 |I|")
    g.add_argument("--point_selection", choices=["fps", "random"], default=None,
                   help="选中集 I 的段内采样方式")
    g.add_argument("--tpose_offset", type=float, default=None,
                   help="T-pose 偏置补偿系数，0 = 论文式 (7) 原式")
    g.add_argument("--lock_ankle_roll", action="store_true",
                   help="把踝 roll 锁到片头静止段的取值（动捕的踝 roll 解算常在腾空/"
                        "落地时跳到错误分支再也回不来）")
    g.add_argument("--iterations", type=int, default=None, help="每帧 Gauss-Newton 迭代次数")
    g.add_argument("--trust_region", choices=["box", "l2"], default=None)
    g.add_argument("--solver", default=None, help="QP 后端，如 clarabel / proxqp")
    g.add_argument("--no_pkl", action="store_true", help="不导出 GMR 兼容的 pkl")

    g = ap.add_argument_group("可视化")
    g.add_argument("--record_video", action="store_true",
                   help="离屏导出人机并排对比视频（与片段同名的 .mp4）")
    g.add_argument("--video_width", type=int, default=560)
    g.add_argument("--video_height", type=int, default=560)
    g.add_argument("--video_stride", type=int, default=2, help="视频抽帧步长")
    g.add_argument("--video_max_frames", type=int, default=900)
    g.add_argument("--no_viewer", action="store_true", help="只算不看，跑完直接退出")
    g.add_argument("--no_points", action="store_true", help="不叠加对应点")
    g.add_argument("--robot_only", action="store_true", help="只显示机器人，不并排放人体")
    g.add_argument("--human_offset", type=float, default=1.2,
                   help="人体沿 +Y 的摆放偏移（米）")
    g.add_argument("--window_width", type=int, default=1280)
    g.add_argument("--window_height", type=int, default=800)

    ap.add_argument("--force", action="store_true",
                    help="忽略阶段缓存，Stage 0/I/II 全部重算")
    return ap.parse_args()


def stage2_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        tgt_fps=args.tgt_fps,
        interpolation=args.interpolation_method,
        start=args.start,
        duration=args.duration,
        n_selected=args.n_selected,
        point_selection=args.point_selection,
        tpose_offset=args.tpose_offset,
        lock_ankle_roll=args.lock_ankle_roll,
        iterations=args.iterations,
        trust_region=args.trust_region,
        solver=args.solver,
        export_pkl=not args.no_pkl,
        force=args.force,
    )


def is_done(clip: Clip, args: argparse.Namespace) -> bool:
    """该片段的产物是否齐全：Stage II 的 npz，以及（除非 --no_pkl）导出的 pkl。"""
    return clip.layout.motion.exists() and (args.no_pkl or clip.layout.motion_pkl.exists())


def render_clip_video(clip: Clip, args: argparse.Namespace, log=print) -> None:
    from umr.sim.views import load_run, render_video

    render_video(
        load_run(clip.layout),
        clip.layout.video,
        width=args.video_width,
        height=args.video_height,
        stride=args.video_stride,
        max_frames=args.video_max_frames,
        points=not args.no_points,
    )


def process_clip(clip: Clip, args: argparse.Namespace, setup: SetupLayout, prefix: str = "") -> bool:
    """跑一段动作的 Stage II（外加可选的视频导出）。返回是否真的处理了。"""
    if is_done(clip, args) and not (args.override or args.force):
        print(f"{prefix}[skip] 已存在 {clip.layout.motion}（要重跑加 --override）")
        return False

    cfg = load_config(args.robot)
    skeleton = get_skeleton(args.human)

    def log(msg: str) -> None:
        print(f"{prefix}{msg}")

    log(f"[clip] {clip.label}")
    retarget_motion(
        cfg, clip.path, setup, clip.layout, skeleton=skeleton, log=log, **stage2_kwargs(args)
    )
    if args.record_video:
        render_clip_video(clip, args, log=log)
    return True


def show_clip(clip: Clip, args: argparse.Namespace) -> None:
    from umr.sim.views import launch_viewer, load_run

    launch_viewer(
        load_run(clip.layout),
        points=not args.no_points,
        robot_only=args.robot_only,
        human_offset=args.human_offset,
        width=args.window_width,
        height=args.window_height,
    )


def _worker(payload: tuple) -> tuple[str, bool, str]:
    """子进程入口：只做 Stage II，共享的 Stage 0/I 已由主进程算好。"""
    clip, args, setup_root = payload
    try:
        processed = process_clip(clip, args, SetupLayout(setup_root), prefix=f"[{clip.name}] ")
        return clip.label, processed, ""
    except Exception as exc:  # 单个片段失败不该带走整批
        return clip.label, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    cfg, skeleton, clips = resolve_target(args)

    print(f"[run] {cfg.robot['name']}  <-  {skeleton.name}  x  {len(clips)} 段")
    pending = [
        c for c in clips
        if args.override or args.force or not is_done(c, args)
    ]
    if len(pending) < len(clips):
        print(f"[run] 跳过 {len(clips) - len(pending)} 段已有结果（要重跑加 --override）")
    if not pending:
        # 结果都在了。指的是单段动作时直接开窗看，不然「想再看一眼」还得先 --override
        # 把十几分钟的求解重跑一遍。
        if len(clips) == 1 and not (args.no_viewer or args.multi_process):
            print("[run] 结果已存在，直接打开播放器（要重算加 --override）")
            show_clip(clips[0], args)
        else:
            print("[run] 没有需要处理的片段（要重跑加 --override）")
        return

    # Stage 0 / I 按 --share_setup 分组共用。分组用的是**全部**片段而不只是待处理的，
    # 这样代表帧不会因为某几段已经跑过就换人，setup 的内容也就不会跟着抖。
    groups: dict[str, list[Clip]] = {}
    for clip in clips:
        sig = setup_signature(cfg, skeleton, clip.path, args.share_setup)
        groups.setdefault(sig, []).append(clip)
    pending_set = set(pending)
    todo = {sig: [c for c in m if c in pending_set] for sig, m in groups.items()}
    active = {sig: m for sig, m in groups.items() if todo[sig]}
    print(f"[run] {len(pending)} 段待处理，{len(active)} 套骨架 setup"
          f"（--share_setup {args.share_setup}）")

    save_path = resolve_save_path(args.save_path)
    robot = robot_key(args.robot)

    t0 = time.perf_counter()
    done = 0
    for sig, members in active.items():
        setup = setup_layout(save_path, skeleton.name, robot, sig)
        rep = setup_representative([c.path for c in members])
        work = todo[sig]
        print(f"\n[setup] {sig}  ({len(work)}/{len(members)} 段待处理)  -> {setup.root}")
        build_bodies(cfg, rep, setup, skeleton=skeleton, force=args.force)
        learn_correspondence(
            cfg, setup, epochs=args.epochs, device=args.device, force=args.force
        )

        if args.multi_process and len(work) > 1:
            workers = args.num_workers or (os.cpu_count() or 1)
            workers = max(1, min(workers, len(work)))
            print(f"[run] 多进程 Stage II：{workers} 个进程 x {len(work)} 段")
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_worker, (clip, args, setup.root)) for clip in work]
                for fut in as_completed(futures):
                    label, processed, err = fut.result()
                    if err:
                        print(f"[fail] {label}: {err}")
                    done += int(processed)
        else:
            for clip in work:
                done += int(process_clip(clip, args, setup))

    elapsed = time.perf_counter() - t0
    print(f"\n[run] 完成 {done} 段，耗时 {elapsed:.1f}s"
          + (f"（平均 {elapsed / done:.1f}s/段）" if done else ""))

    if args.no_viewer or args.multi_process:
        return
    if len(pending) > 1:
        print("[run] 多段结果不自动开窗，用 --motion_file 指到单个文件查看")
        return

    show_clip(pending[0], args)


if __name__ == "__main__":
    main()
