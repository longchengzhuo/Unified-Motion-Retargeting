#!/usr/bin/env python
"""Stage II：对应引导的重定向（论文 III-C，式 6-14）。

复用 Stage I 学到的对应点，在 mink 的约束 Gauss-Newton QP 中逐帧匹配表面位置、
法线与接触图，求出机器人的广义坐标序列。

输入: $UMR_OUTPUT_DIR/{bodies.npz, correspondence.npz}
输出: $UMR_OUTPUT_DIR/motion.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umr.bodies import bvh as bvh_mod
from umr.bodies.human_mjcf import HumanBody
from umr.bodies.robot import RobotBody
from umr.bodies.surface import SurfacePointCloud
from umr.config import load_config
from umr.paths import OUTPUT_DIR
from umr.retarget.binding import LinkBinding
from umr.retarget.export import save_motion_pkl
from umr.retarget.pipeline import UMRRetargeter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--bodies", default=str(OUTPUT_DIR / "bodies.npz"))
    ap.add_argument("--correspondence", default=str(OUTPUT_DIR / "correspondence.npz"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "motion.npz"))
    ap.add_argument(
        "--pkl", default=None,
        help="额外导出的 pkl 路径（与 agmr/GMR 字段兼容）；默认与 --out 同名改后缀",
    )
    ap.add_argument("--no_pkl", action="store_true", help="不导出 pkl")
    ap.add_argument("--fps", type=float, default=None, help="目标帧率")
    ap.add_argument("--start", type=float, default=0.0, help="起始时间（秒）")
    ap.add_argument("--duration", type=float, default=None, help="截取时长（秒）")
    ap.add_argument("--n_selected", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--trust_region", choices=["box", "l2"], default=None)
    ap.add_argument("--solver", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rcfg = cfg.retarget

    bodies = np.load(args.bodies, allow_pickle=True)
    corr = np.load(args.correspondence, allow_pickle=True)

    human_pc = SurfacePointCloud.from_dict(bodies, "human_")
    binding = LinkBinding.from_dict(corr, "bind_")
    segment = corr["inherited_segment"]
    segment_names = [str(s) for s in corr["segment_names"]]

    # ------------------------------------------------------------------
    robot = RobotBody.from_bodies(bodies)
    motion = bvh_mod.load_bvh(
        str(bodies["bvh_path"]),
        scale=float(cfg.source["length_scale"]),
        auto_face_x=bool(cfg.source["auto_face_x"]),
    )
    human = HumanBody(str(bodies["human_xml"]), motion, scale=float(bodies["scale"]))
    human.root_offset = np.array([0.0, 0.0, float(bodies["ground_offset"])])

    # ------------------------------------------------------------------
    src_fps = float(bodies["fps"])
    tgt_fps = args.fps or float(cfg.source["target_fps"])
    total = motion.num_frames
    rest_frames = bvh_mod.first_motion_frame(motion)
    if rest_frames:
        print(f"[stage2] 跳过 BVH 开头 {rest_frames} 个合成静止帧（真实动作从此开始）")
    f0 = max(rest_frames, int(round(args.start * src_fps)))
    f1 = total if args.duration is None else min(total, f0 + int(round(args.duration * src_fps)))
    n_out = max(2, int(round((f1 - f0) * tgt_fps / src_fps)))
    frame_indices = np.round(np.linspace(f0, f1 - 1, n_out)).astype(int)
    print(
        f"[stage2] 源 {src_fps:.1f} FPS x {total} 帧 -> 目标 {tgt_fps:.1f} FPS x {n_out} 帧 "
        f"({(f1-f0)/src_fps:.1f}s)"
    )

    # ------------------------------------------------------------------
    retargeter = UMRRetargeter(
        robot, human,
        human_body_ids=human_pc.body_ids,
        human_local_pos=human_pc.local_pos,
        human_local_normal=human_pc.local_normal,
        robot_body_ids=binding.body_ids,
        robot_local_pos=binding.local_pos,
        robot_local_normal=binding.local_normal,
        segment=segment,
        segment_names=segment_names,
        n_selected=args.n_selected or int(rcfg["n_selected"]),
        iterations=args.iterations or int(rcfg["iterations"]),
        dt=float(rcfg["dt"]),
        damping=float(rcfg["damping"]),
        solver=args.solver or str(rcfg["solver"]),
        trust_region=args.trust_region or str(rcfg["trust_region"]),
        trust_region_radius=float(rcfg["trust_region_radius"]),
        floor_height=float(rcfg["floor_height"]),
        floor_band=float(rcfg["floor_band"]),
        floor_margin=float(rcfg["floor_margin"]),
        contact_threshold=float(rcfg["contact_threshold"]),
        contact_weight=float(rcfg["contact_weight"]),
        posture_cost=float(rcfg["posture_cost"]),
        self_collision=bool(rcfg["self_collision"]),
    )
    print(
        f"[stage2] |I|={len(retargeter.selected)}  地面约束候选点="
        f"{len(retargeter.floor_cache.local_pos)}  solver={retargeter.solver}  "
        f"trust_region={retargeter.trust_region}  iters/frame={retargeter.iterations}"
    )

    result = retargeter.run(frame_indices, tgt_fps)

    print(
        f"[stage2] 点匹配误差 mean={result.point_error.mean()*1000:.1f}mm  "
        f"median={np.median(result.point_error)*1000:.1f}mm  "
        f"p95={np.percentile(result.point_error,95)*1000:.1f}mm"
    )
    print(
        f"[stage2] 法线误差 mean={np.degrees(result.normal_error.mean()):.1f}deg  "
        f"接触点 mean={result.contact_count.mean():.1f}  "
        f"QP 失败 {result.solve_failures}"
    )
    print(
        f"[stage2] 吞吐 {result.timings['fps']:.1f} FPS  "
        f"(warmup {result.timings['warmup']:.2f}s, retarget {result.timings['retarget']:.1f}s)"
    )

    np.savez(
        args.out,
        qpos=result.qpos,
        frame_indices=result.frame_indices,
        fps=result.fps,
        point_error=result.point_error,
        normal_error=result.normal_error,
        contact_count=result.contact_count,
        floor_rows=result.floor_rows,
        solve_failures=result.solve_failures,
        selected=retargeter.selected,
        timings=np.array(list(result.timings.items()), dtype=object),
        robot_xml=str(bodies["robot_xml"]),
        human_xml=str(bodies["human_xml"]),
        bvh_path=str(bodies["bvh_path"]),
        scale=float(bodies["scale"]),
        ground_offset=float(bodies["ground_offset"]),
    )
    print(f"[save] {args.out}")

    if not args.no_pkl:
        pkl_path = args.pkl or str(Path(args.out).with_suffix(".pkl"))
        save_motion_pkl(
            pkl_path, robot.model, result.qpos, tgt_fps,
            extra={
                "source_file": str(bodies["bvh_path"]),
                "robot_xml": str(bodies["robot_xml"]),
                "human_xml": str(bodies["human_xml"]),
                "frame_indices": result.frame_indices,
                "source_fps": src_fps,
                "scale": float(bodies["scale"]),
                "ground_offset": float(bodies["ground_offset"]),
                "point_error": result.point_error,
                "normal_error": result.normal_error,
                "contact_count": result.contact_count,
                "quality_metrics": {
                    "point_error_mm_mean": float(result.point_error.mean() * 1000),
                    "point_error_mm_median": float(np.median(result.point_error) * 1000),
                    "normal_error_deg_mean": float(np.degrees(result.normal_error.mean())),
                    "solve_failures": int(result.solve_failures),
                    "retarget_fps": float(result.timings["fps"]),
                },
            },
        )
        print(f"[save] {pkl_path}  (root_trans/root_rot(xyzw)/dof/dof_full/fps，兼容 agmr)")


if __name__ == "__main__":
    main()
