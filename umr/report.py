"""用 MuJoCo 动力学校验重定向结果，并汇总成一份指标报告。

- ``mj_inverse`` 反求关节力矩、自由基座残余力
- 质心 / ZMP 与支撑多边形
- 足部穿透与逐脚离地高度跟踪、关节限位、时间平滑度
- 可选：PD 控制在 ``mj_step`` 里开环回放
- 论文 Table I 形式的耗时/吞吐汇总
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from umr.bodies.human_mjcf import HumanBody
from umr.bodies.robot import RobotBody
from umr.metrics import (
    foot_ground_metrics,
    foot_tracking,
    format_table,
    joint_limit_violation,
    smoothness,
    summarize,
)
from umr.paths import ClipLayout, SetupLayout
from umr.sim.validate import (
    ValidationResult,
    foot_geom_sides,
    replay_with_pd,
    side_foot_heights,
    validate_motion,
)
from umr.sim.views import clip_from_meta

Log = Callable[[str], None]

QUANTILES = ("min", "median", "mean", "p95", "max")


def _kv(d: dict, fmt: str = "{:.3f}") -> str:
    return "  ".join(f"{k}={fmt.format(v)}" for k, v in d.items())


def _row(name: str, m: dict[str, float]) -> str:
    return f"| {name} | " + " | ".join(f"{m[k]:.1f}" for k in QUANTILES) + " |"


@dataclass
class RunEvaluation:
    """一次运行的全部校验结果。"""

    layout: ClipLayout
    header: dict[str, Any]
    validation: ValidationResult
    point: dict[str, float]
    normal: dict[str, float]
    torque: dict[str, float]
    foot: dict[str, float]
    tracking: dict[str, float]
    limit: dict[str, float]
    smooth: dict[str, float]
    zmp: dict[str, float]
    stage1: dict[str, float]
    cost_table: str
    solve_failures: int
    replay: dict[str, float] | None = None
    # PD 回放的逐帧仿真 qpos。只在内存里传给播放器，不进 report.md / metrics.npz。
    replay_qpos: np.ndarray | None = None
    figures: list[Path] = field(default_factory=list)

    def to_markdown(self) -> str:
        h = self.header
        val = self.validation
        lines = [
            "# UMR 复现结果报告",
            "",
            f"- 源动作: `{h['bvh_name']}`  ({h['human_skeleton']} 骨架)",
            (
                f"- 机器人: `{h['robot_name']}`  (nq={h['nq']}, nv={h['nv']}, "
                f"身高 {h['robot_height']:.3f} m)"
            ),
            f"- 演员身高 {h['actor_height']:.3f} m，归一化尺度 {h['scale']:.4f}",
            (
                f"- 重定向 {h['frames']} 帧 @ {h['fps']:.1f} FPS"
                f"（源 {h['source_fps']:.1f} FPS，{h['interpolation']} 插值）"
            ),
            "",
            "## Stage I 点云对应学习",
            "",
            (
                f"- Chamfer (recon->target) "
                f"**{self.stage1['chamfer_recon_to_target_mm']:.2f} mm**, "
                f"(target->recon) {self.stage1['chamfer_target_to_recon_mm']:.2f} mm"
            ),
            f"- 2cm 覆盖率 {self.stage1['coverage_2cm'] * 100:.1f}%",
            (
                f"- **解剖学一致性 {self.stage1['anatomical_consistency'] * 100:.1f}%**"
                "（无任何人工骨骼映射）"
            ),
            "",
            "## Stage II 重定向质量",
            "",
            "| 指标 | min | median | mean | p95 | max |",
            "|---|---|---|---|---|---|",
            _row("点匹配误差 (mm)", self.point),
            _row("法线误差 (deg)", self.normal),
            _row("最大关节力矩 (N·m)", self.torque),
            "",
            (
                f"- 足部穿透 mean {self.foot['penetration_mean_mm']:.2f} mm / "
                f"max {self.foot['penetration_max_mm']:.2f} mm"
            ),
            (
                f"- 逐脚离地高度跟踪：左 corr {self.tracking['left_corr']:.3f} / "
                f"RMSE {self.tracking['left_rmse_mm']:.1f} mm，"
                f"右 corr {self.tracking['right_corr']:.3f} / "
                f"RMSE {self.tracking['right_rmse_mm']:.1f} mm "
                f"（摆动幅度 {self.tracking['swing_range_mm']:.0f} mm）"
            ),
            (
                f"- 关节限位违反比例 {self.limit['violation_ratio'] * 100:.4f}%"
                f"（最大 {self.limit['max_violation_rad']:.2e} rad）"
            ),
            (
                f"- 关节速度 mean {self.smooth['joint_vel_mean']:.3f} rad/s，"
                f"加速度 mean {self.smooth['joint_acc_mean']:.2f} rad/s²"
            ),
            (
                f"- ZMP 落在支撑多边形内 {val.support_ok.mean() * 100:.1f}%，"
                f"有符号裕度中位数 {self.zmp.get('median', float('nan')):.0f} mm"
                "（负值表示在支撑区内部）；"
                f"腾空帧占 {(val.support_size == 0).mean() * 100:.1f}%"
            ),
            "  行走本身就是受控失衡，单支撑相 ZMP 越过脚缘属正常现象，此处只作参考。",
            f"- QP 求解失败 {self.solve_failures} 次",
            "",
        ]
        if self.replay:
            r = self.replay
            lines += [
                "## MuJoCo PD 开环回放",
                "",
                "无平衡控制器的欠驱动仿真，运动学参考在此条件下跌倒是预期行为；",
                "该指标只作可行性探针，不是稳定性结论。",
                "",
                (
                    f"- 跌倒前存活 {r['survived_seconds']:.2f} s "
                    f"({r['frames_simulated']}/{r['frames_total']} 帧)"
                ),
                f"- 跌倒前关节跟踪误差 {r['joint_tracking_error_rad']:.4f} rad",
                f"- 结束时基座高度 {r['final_base_height']:.3f} m",
                "",
            ]
        lines += ["## 计算开销", "", "```", self.cost_table, "```", ""]
        if self.figures:
            lines += ["## 附带产物", ""]
            lines += [f"- `{p.name}`" for p in self.figures]
            lines += [""]
        return "\n".join(lines)

    def save(self) -> None:
        """写出 ``report.md`` 与 ``metrics.npz``。"""
        val = self.validation
        self.layout.report.write_text(self.to_markdown(), encoding="utf-8")
        np.savez(
            self.layout.metrics,
            torque=val.torque, com=val.com, zmp=val.zmp,
            foot_height=val.foot_height, support_ok=val.support_ok,
            base_residual=val.base_residual,
            summary=json.dumps(
                {
                    "point_mm": self.point, "normal_deg": self.normal, "foot": self.foot,
                    "foot_tracking": self.tracking, "limit": self.limit,
                    "smooth": self.smooth, "torque": self.torque,
                    "stage1": self.stage1, "replay": self.replay,
                    "zmp_in_support": float(val.support_ok.mean()),
                },
                ensure_ascii=False,
            ),
        )


def evaluate_run(
    layout: ClipLayout,
    setup: SetupLayout | None = None,
    *,
    replay_seconds: float | None = None,
    max_frames: int | None = None,
    log: Log = print,
) -> RunEvaluation:
    """对一段重定向结果做完整校验。"""
    if not layout.motion.exists():
        raise SystemExit(f"找不到 {layout.motion}，请先跑 scripts/retarget.py")
    motion = np.load(layout.motion, allow_pickle=True)

    setup = setup or SetupLayout(Path(str(motion["setup_dir"])))
    for path in (setup.bodies, setup.correspondence):
        if not path.exists():
            raise SystemExit(f"找不到 {path}，请先跑 scripts/retarget.py")
    bodies = np.load(setup.bodies, allow_pickle=True)
    corr = np.load(setup.correspondence, allow_pickle=True)

    robot = RobotBody(str(motion["robot_xml"]))
    qpos = motion["qpos"]
    fps = float(motion["fps"])
    if max_frames:
        qpos = qpos[:max_frames]

    log(f"[validate] {qpos.shape[0]} 帧 @ {fps:.1f} FPS")
    val = validate_motion(robot.model, qpos, fps)

    # 逐脚的离地高度跟踪：把源动作与重定向结果的左右脚底高度对起来比
    human = HumanBody(
        str(motion["human_xml"]), clip_from_meta(motion), scale=float(motion["scale"])
    )
    human.root_offset = np.array([0.0, 0.0, float(motion["ground_offset"])])
    human_sides = foot_geom_sides(human.model, "human")
    robot_sides = foot_geom_sides(robot.model, "robot")

    h_foot = np.empty((len(qpos), 2))
    r_foot = np.empty((len(qpos), 2))
    for k, f in enumerate(motion["frame_indices"][: len(qpos)]):
        human.set_frame(int(f))
        h_foot[k] = side_foot_heights(human.model, human.data, human_sides)
        robot.set_qpos(qpos[k])
        r_foot[k] = side_foot_heights(robot.model, robot.data, robot_sides)
    m_track = foot_tracking(h_foot, r_foot)
    log(f"[track ] 逐脚离地高度 {_kv(m_track)}")

    lower, upper = robot.joint_limits()
    m_point = summarize(motion["point_error"], 1000.0)
    m_normal = summarize(np.degrees(motion["normal_error"]))
    m_foot = foot_ground_metrics(val.foot_height)
    m_limit = joint_limit_violation(qpos, lower, upper)
    m_smooth = smoothness(qpos, fps)
    m_torque = summarize(np.abs(val.torque).max(axis=1))
    finite_margin = val.zmp_margin[np.isfinite(val.zmp_margin)]
    m_zmp = summarize(finite_margin, 1000.0) if len(finite_margin) else {}

    log(f"[point ] {_kv(m_point)}   (mm)")
    log(f"[normal] {_kv(m_normal)}   (deg)")
    log(f"[foot  ] {_kv(m_foot)}")
    log(f"[limit ] {_kv(m_limit, '{:.5f}')}")
    log(f"[smooth] {_kv(m_smooth)}")
    log(f"[torque] {_kv(m_torque)}   (N·m, 逐帧最大关节力矩)")
    log(f"[ZMP   ] 落在支撑多边形内 {val.support_ok.mean() * 100:.1f}%，"
        f"有符号裕度(负=内部) median={m_zmp.get('median', float('nan')):.0f}mm "
        f"p95={m_zmp.get('p95', float('nan')):.0f}mm；"
        f"支撑点 mean={val.support_size.mean():.1f}，"
        f"腾空帧 {(val.support_size == 0).mean() * 100:.1f}%")

    replay = replay_qpos = None
    if replay_seconds:
        log(f"[replay] PD 回放 {replay_seconds:.1f}s ...")
        replay, replay_qpos = replay_with_pd(robot.model, qpos, fps, max_seconds=replay_seconds)
        log(f"[replay] {_kv(replay)}")

    stage0_t = {k: float(v) for k, v in bodies["timings"]}
    stage1_t = {k: float(v) for k, v in corr["stage1_timings"]}
    stage2_t = {k: float(v) for k, v in motion["timings"]}
    setup_total = stage0_t.get("point_sampling", 0) + sum(stage1_t.values())
    cost_table = format_table(
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
    log("\n" + cost_table)

    return RunEvaluation(
        layout=layout,
        header={
            "bvh_name": Path(str(motion["bvh_path"])).name,
            "human_skeleton": str(motion["human_skeleton"]),
            "robot_name": str(bodies["robot_name"]),
            "nq": int(robot.model.nq),
            "nv": robot.nv,
            "robot_height": float(bodies["robot_height"]),
            "actor_height": float(bodies["actor_height"]),
            "scale": float(motion["scale"]),
            "frames": int(qpos.shape[0]),
            "fps": fps,
            "source_fps": float(motion["source_fps"]),
            "interpolation": str(motion["interpolation"]),
        },
        validation=val,
        point=m_point, normal=m_normal, torque=m_torque, foot=m_foot,
        tracking=m_track, limit=m_limit, smooth=m_smooth, zmp=m_zmp,
        stage1={k: float(v) for k, v in corr["stage1_metrics"]},
        cost_table=cost_table,
        solve_failures=int(motion["solve_failures"]),
        replay=replay,
        replay_qpos=replay_qpos,
    )
