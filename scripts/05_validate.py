#!/usr/bin/env python
"""用 MuJoCo 动力学校验重定向结果，并汇总全流程指标。

- ``mj_inverse`` 反求关节力矩、自由基座残余力
- 质心 / ZMP 与支撑多边形
- 足部穿透与离地、关节限位、时间平滑度
- 可选：PD 控制在 ``mj_step`` 里回放（``--replay``）
- 论文 Table I 形式的耗时/吞吐汇总

输入: $UMR_OUTPUT_DIR/{bodies,correspondence,motion}.npz
输出: $UMR_OUTPUT_DIR/{report.md, metrics.npz}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco

from umr.bodies import bvh as bvh_mod
from umr.bodies.human_mjcf import HumanBody
from umr.bodies.robot import RobotBody, geom_lowest_z
from umr.metrics import (
    foot_ground_metrics,
    foot_tracking,
    format_table,
    joint_limit_violation,
    smoothness,
    summarize,
)
from umr.paths import OUTPUT_DIR
from umr.sim.validate import replay_with_pd, validate_motion


def side_foot_heights(
    model: mujoco.MjModel, data: mujoco.MjData, kind: str, keys: tuple[str, ...] = ("Left", "Right")
) -> list[float]:
    """当前姿态下左右脚各自的最低点高度。``kind`` 为 'h'(人体) 或 'r'(机器人)。"""
    attr = mujoco.mjtObj.mjOBJ_GEOM if kind == "h" else mujoco.mjtObj.mjOBJ_BODY
    out = []
    for key in keys:
        geoms = []
        for g in range(model.ngeom):
            if kind == "h":
                name = mujoco.mj_id2name(model, attr, g) or ""
                hit = name.startswith(f"hg_{key}") and ("Ankle" in name or "Toe" in name)
            else:
                b = int(model.geom_bodyid[g])
                name = mujoco.mj_id2name(model, attr, b) or ""
                hit = key in name
            if hit:
                geoms.append(g)
        out.append(min(geom_lowest_z(model, data, g) for g in geoms))
    return out


def kv(d: dict, fmt: str = "{:.3f}") -> str:
    return "  ".join(f"{k}={fmt.format(v)}" for k, v in d.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bodies", default=str(OUTPUT_DIR / "bodies.npz"))
    ap.add_argument("--correspondence", default=str(OUTPUT_DIR / "correspondence.npz"))
    ap.add_argument("--motion", default=str(OUTPUT_DIR / "motion.npz"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "report.md"))
    ap.add_argument("--replay", action="store_true", help="额外做 PD 回放仿真")
    ap.add_argument("--replay_seconds", type=float, default=5.0)
    ap.add_argument("--max_frames", type=int, default=None, help="限制校验帧数以加快速度")
    args = ap.parse_args()

    bodies = np.load(args.bodies, allow_pickle=True)
    corr = np.load(args.correspondence, allow_pickle=True)
    motion = np.load(args.motion, allow_pickle=True)

    robot = RobotBody.from_bodies(bodies)
    qpos = motion["qpos"]
    fps = float(motion["fps"])
    if args.max_frames:
        qpos = qpos[: args.max_frames]

    print(f"[validate] {qpos.shape[0]} 帧 @ {fps:.1f} FPS")
    val = validate_motion(robot.model, robot.spec, qpos, fps)

    # 逐脚的离地高度跟踪：把源动作与重定向结果的左右脚底高度对起来比
    bvh = bvh_mod.load_bvh(str(bodies["bvh_path"]))
    human = HumanBody(str(bodies["human_xml"]), bvh, scale=float(bodies["scale"]))
    human.root_offset = np.array([0.0, 0.0, float(bodies["ground_offset"])])
    h_foot = np.array(
        [side_foot_heights(human.model, human.data, "h") for f in motion["frame_indices"][: len(qpos)]
         if (human.set_frame(int(f)) or True)]
    )
    r_foot = np.array(
        [side_foot_heights(robot.model, robot.data, "r", robot.spec.foot_bodies)
         for k in range(len(qpos)) if (robot.set_qpos(qpos[k]) or True)]
    )
    m_track = foot_tracking(h_foot, r_foot)
    print(f"[track ] 逐脚离地高度 {kv(m_track)}")

    lower, upper = robot.joint_limits()
    m_point = summarize(motion["point_error"], 1000.0)
    m_normal = summarize(np.degrees(motion["normal_error"]))
    m_foot = foot_ground_metrics(val.foot_height)
    m_limit = joint_limit_violation(qpos, lower, upper)
    m_smooth = smoothness(qpos, fps)
    m_torque = summarize(np.abs(val.torque).max(axis=1))
    stage1 = {k: float(v) for k, v in corr["stage1_metrics"]}
    stage1_t = {k: float(v) for k, v in corr["stage1_timings"]}
    stage0_t = {k: float(v) for k, v in bodies["timings"]}
    stage2_t = {k: float(v) for k, v in motion["timings"]}

    print(f"[point ] {kv(m_point)}   (mm)")
    print(f"[normal] {kv(m_normal)}   (deg)")
    print(f"[foot  ] {kv(m_foot)}")
    print(f"[limit ] {kv(m_limit, '{:.5f}')}")
    print(f"[smooth] {kv(m_smooth)}")
    print(f"[torque] {kv(m_torque)}   (N·m, 逐帧最大关节力矩)")
    finite_margin = val.zmp_margin[np.isfinite(val.zmp_margin)]
    m_zmp = summarize(finite_margin, 1000.0) if len(finite_margin) else {}
    print(
        f"[ZMP   ] 落在支撑多边形内 {val.support_ok.mean()*100:.1f}%，"
        f"有符号裕度(负=内部) median={m_zmp.get('median', float('nan')):.0f}mm "
        f"p95={m_zmp.get('p95', float('nan')):.0f}mm；"
        f"支撑点 mean={val.support_size.mean():.1f}，"
        f"腾空帧 {(val.support_size == 0).mean()*100:.1f}%"
    )

    replay = None
    if args.replay:
        print(f"[replay] PD 回放 {args.replay_seconds:.1f}s ...")
        replay = replay_with_pd(robot.model, qpos, fps, max_seconds=args.replay_seconds)
        print(f"[replay] {kv(replay)}")

    # ------------------------------------------------------------------
    # 论文 Table I 形式的耗时汇总
    # ------------------------------------------------------------------
    setup_total = stage0_t.get("point_sampling", 0) + sum(stage1_t.values())
    table = format_table(
        [
            ("Stage I", "Point cloud sampling", f"{stage0_t.get('point_sampling', 0):.2f} s"),
            ("", "Geodesic precomputation", f"{stage1_t.get('geodesic', 0):.2f} s"),
            ("", "Correspondence training", f"{stage1_t.get('training', 0):.2f} s"),
            ("", "Link binding", f"{stage1_t.get('binding', 0):.2f} s"),
            ("", "Total setup time", f"{setup_total:.2f} s"),
            ("Stage II", "BVH parse + FK", f"{stage0_t.get('bvh_parse', 0):.2f} s"),
            ("", "Motion retargeting", f"{stage2_t.get('fps', 0):.2f} FPS"),
            ("", "Frames retargeted", f"{qpos.shape[0]}"),
        ],
        title="UMR computational cost (paper Table I style)",
    )
    print("\n" + table)

    # ------------------------------------------------------------------
    report = [
        "# UMR 复现结果报告",
        "",
        f"- 源动作: `{Path(str(bodies['bvh_path'])).name}`",
        f"- 机器人: `{robot.spec.name}`  (nq={robot.model.nq}, nv={robot.nv}, 身高 {float(bodies['robot_height']):.3f} m)",
        f"- 演员身高 {float(bodies['actor_height']):.3f} m，归一化尺度 {float(bodies['scale']):.4f}",
        f"- 重定向 {qpos.shape[0]} 帧 @ {fps:.1f} FPS",
        "",
        "## Stage I 点云对应学习",
        "",
        f"- Chamfer (recon->target) **{stage1['chamfer_recon_to_target_mm']:.2f} mm**, "
        f"(target->recon) {stage1['chamfer_target_to_recon_mm']:.2f} mm",
        f"- 2cm 覆盖率 {stage1['coverage_2cm']*100:.1f}%",
        f"- **解剖学一致性 {stage1['anatomical_consistency']*100:.1f}%**（无任何人工骨骼映射）",
        "",
        "## Stage II 重定向质量",
        "",
        "| 指标 | min | median | mean | p95 | max |",
        "|---|---|---|---|---|---|",
        "| 点匹配误差 (mm) | "
        + " | ".join(f"{m_point[k]:.1f}" for k in ["min", "median", "mean", "p95", "max"]) + " |",
        "| 法线误差 (deg) | "
        + " | ".join(f"{m_normal[k]:.1f}" for k in ["min", "median", "mean", "p95", "max"]) + " |",
        "| 最大关节力矩 (N·m) | "
        + " | ".join(f"{m_torque[k]:.1f}" for k in ["min", "median", "mean", "p95", "max"]) + " |",
        "",
        f"- 足部穿透 mean {m_foot['penetration_mean_mm']:.2f} mm / max {m_foot['penetration_max_mm']:.2f} mm",
        f"- 逐脚离地高度跟踪：左 corr {m_track['left_corr']:.3f} / RMSE {m_track['left_rmse_mm']:.1f} mm，"
        f"右 corr {m_track['right_corr']:.3f} / RMSE {m_track['right_rmse_mm']:.1f} mm "
        f"（摆动幅度 {m_track['swing_range_mm']:.0f} mm）",
        f"- 关节限位违反比例 {m_limit['violation_ratio']*100:.4f}%（最大 {m_limit['max_violation_rad']:.2e} rad）",
        f"- 关节速度 mean {m_smooth['joint_vel_mean']:.3f} rad/s，加速度 mean {m_smooth['joint_acc_mean']:.2f} rad/s²",
        f"- ZMP 落在支撑多边形内 {val.support_ok.mean()*100:.1f}%，有符号裕度中位数 "
        f"{m_zmp.get('median', float('nan')):.0f} mm（负值表示在支撑区内部）；"
        f"腾空帧占 {(val.support_size == 0).mean()*100:.1f}%",
        "  行走本身就是受控失衡，单支撑相 ZMP 越过脚缘属正常现象，此处只作参考。",
        f"- QP 求解失败 {int(motion['solve_failures'])} 次",
        "",
    ]
    if replay:
        report += [
            "## MuJoCo PD 开环回放",
            "",
            "无平衡控制器的欠驱动仿真，运动学参考在此条件下跌倒是预期行为；",
            "该指标只作可行性探针，不是稳定性结论。",
            "",
            f"- 跌倒前存活 {replay['survived_seconds']:.2f} s "
            f"({replay['frames_simulated']}/{replay['frames_total']} 帧)",
            f"- 跌倒前关节跟踪误差 {replay['joint_tracking_error_rad']:.4f} rad",
            f"- 结束时基座高度 {replay['final_base_height']:.3f} m",
            "",
        ]
    report += ["## 计算开销", "", "```", table, "```", ""]

    Path(args.out).write_text("\n".join(report), encoding="utf-8")
    print(f"\n[save] {args.out}")

    # 指标与报告放在一起，这样 --out 指到别处时不会覆盖主目录的结果
    np.savez(
        Path(args.out).with_name("metrics.npz"),
        torque=val.torque, com=val.com, zmp=val.zmp,
        foot_height=val.foot_height, support_ok=val.support_ok,
        base_residual=val.base_residual,
        summary=json.dumps(
            {
                "point_mm": m_point, "normal_deg": m_normal, "foot": m_foot,
                "foot_tracking": m_track,
                "limit": m_limit, "smooth": m_smooth, "torque": m_torque,
                "stage1": stage1, "replay": replay,
                "zmp_in_support": float(val.support_ok.mean()),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
