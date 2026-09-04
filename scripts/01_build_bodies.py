#!/usr/bin/env python
"""Stage 0：建立人机两侧的 MuJoCo 身体，并采样 canonical T-pose 外表面点云。

- 由 BVH 层级生成人体 MJCF（free root + ball joints + 每段胶囊/椭球/盒体）
- 给机器人 MJCF 注入 T_pose keyframe
- 把人体按机器人身高归一化（论文 Fig.2 的 "Normalized Human Point Cloud"）
- 用统一采样器在 T-pose 下采样两侧点云 X^h（有序）与 X^r（无序）

输出: $UMR_OUTPUT_DIR/{bodies.npz, human_<clip>.xml}（默认 outputs/）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umr.bodies import bvh as bvh_mod
from umr.bodies.human_mjcf import (
    ALL_SEGMENTS,
    HumanBody,
    compute_ground_offset,
    write_human_mjcf,
)
from umr.bodies.robot import RobotBody, RobotSpec, prepare_robot_xml
from umr.bodies.surface import sample_model_surface
from umr.config import load_config
from umr.paths import OUTPUT_DIR, PROJECT_ROOT


def robot_surface_geoms(rb: RobotBody) -> list[int]:
    """机器人参与表面采样的 geom。

    取 visual 组（group==1，代表真实外形）加上所有基元 geom（脚底的 box/球在
    collision 组里，但它们才是真正的足底表面），并排除标记点 body。
    """
    m = rb.model
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b == 0 or rb.spec.is_marker(rb.body_names[b]):
            continue
        is_visual = int(m.geom_group[g]) == 1
        is_primitive = int(m.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH
        if is_visual or is_primitive:
            out.append(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--bvh", default=None)
    ap.add_argument("--n_points", type=int, default=None)
    ap.add_argument("--out", default=str(OUTPUT_DIR / "bodies.npz"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_points = args.n_points or int(cfg.sampling["n_points"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 机器人
    # ------------------------------------------------------------------
    robot_xml = PROJECT_ROOT / cfg.robot["xml"]
    spec = RobotSpec.from_config(cfg.robot)
    prepare_robot_xml(robot_xml, spec)
    rb = RobotBody(robot_xml, spec)
    robot_height = rb.height()
    robot_foot_h = rb.foot_height()
    print(f"[robot] {rb.spec.name}  nq={rb.model.nq} nv={rb.nv}")
    print(f"[robot] root={rb.spec.root_body}  feet={list(rb.spec.foot_bodies)}")
    print(f"[robot] height={robot_height:.4f} m  foot_height={robot_foot_h:.4f} m")

    # ------------------------------------------------------------------
    # 人体
    # ------------------------------------------------------------------
    bvh_path = Path(args.bvh) if args.bvh else PROJECT_ROOT / cfg.source["bvh"]
    t0 = time.perf_counter()
    motion = bvh_mod.load_bvh(
        bvh_path,
        scale=float(cfg.source["length_scale"]),
        auto_face_x=bool(cfg.source["auto_face_x"]),
    )
    timings["bvh_parse"] = time.perf_counter() - t0
    h_actor = bvh_mod.actor_height(motion)
    scale = robot_height / h_actor
    print(
        f"[human] {bvh_path.name}  frames={motion.num_frames} joints={motion.num_joints} "
        f"fps={motion.fps:.2f}"
    )
    print(f"[human] actor_height={h_actor:.4f} m  -> scale={scale:.4f} (归一化到机器人身高)")

    human_xml = OUTPUT_DIR / f"human_{bvh_path.stem}.xml"
    info = write_human_mjcf(motion, human_xml, scale=scale)
    hb = HumanBody(human_xml, motion, scale=scale)
    print(f"[human] MJCF -> {human_xml}  nq={hb.model.nq} nbody={hb.model.nbody}")

    # 把整段动作贴到 z=0 地面（跳过开头的合成静止帧）
    f0 = bvh_mod.first_motion_frame(motion)
    probe = np.linspace(f0, motion.num_frames - 1, min(400, motion.num_frames - f0)).astype(int)
    ground = compute_ground_offset(hb, probe, info.geom_segment)
    hb.root_offset = np.array([0.0, 0.0, ground])
    print(f"[human] ground_offset = {ground:+.4f} m")

    # ------------------------------------------------------------------
    # T-pose 表面采样
    # ------------------------------------------------------------------
    hb.set_tpose()
    t0 = time.perf_counter()
    human_pc = sample_model_surface(
        hb.model, hb.data, n_points,
        geom_segment=info.geom_segment,
        segment_names=ALL_SEGMENTS,
        oversample=float(cfg.sampling["oversample"]),
        cull_margin=float(cfg.sampling["cull_margin"]),
        seed=int(cfg.sampling["seed"]),
    )
    t_h = time.perf_counter() - t0

    rb.set_tpose()
    t0 = time.perf_counter()
    robot_pc = sample_model_surface(
        rb.model, rb.data, n_points,
        geom_ids=robot_surface_geoms(rb),
        segment_names=ALL_SEGMENTS,
        oversample=float(cfg.sampling["oversample"]),
        cull_margin=float(cfg.sampling["cull_margin"]),
        seed=int(cfg.sampling["seed"]),
    )
    t_r = time.perf_counter() - t0
    timings["point_sampling"] = t_h + t_r

    print(f"[sample] human N={len(human_pc)} ({t_h:.2f}s)  robot N={len(robot_pc)} ({t_r:.2f}s)")
    print(f"[sample] human bbox z=[{human_pc.points[:,2].min():.3f}, {human_pc.points[:,2].max():.3f}]"
          f"  robot bbox z=[{robot_pc.points[:,2].min():.3f}, {robot_pc.points[:,2].max():.3f}]")
    covered = sorted({human_pc.segment_names[s] for s in np.unique(human_pc.segment) if s >= 0})
    missing = [s for s in ALL_SEGMENTS if s not in covered]
    print(f"[sample] 人体分段覆盖 {len(covered)}/{len(ALL_SEGMENTS)}" + (f"  缺失={missing}" if missing else ""))

    # ------------------------------------------------------------------
    out = {
        "robot_xml": str(robot_xml),
        "robot_spec": rb.spec.to_json(),  # 下游脚本据此还原机器人语义，无需再读配置
        "human_xml": str(human_xml),
        "bvh_path": str(bvh_path),
        "scale": scale,
        "ground_offset": ground,
        "actor_height": h_actor,
        "robot_height": robot_height,
        "robot_foot_height": robot_foot_h,
        "fps": motion.fps,
        "num_frames": motion.num_frames,
        "timings": np.array(list(timings.items()), dtype=object),
    }
    out.update(human_pc.to_dict("human_"))
    out.update(robot_pc.to_dict("robot_"))
    np.savez(args.out, **out)
    print(f"[save] {args.out}")
    print("[time] " + "  ".join(f"{k}={v:.2f}s" for k, v in timings.items()))


if __name__ == "__main__":
    main()
