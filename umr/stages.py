"""UMR 三个阶段的编排：建人机身体 -> 学点云对应 -> 逐帧重定向。

每个阶段把结果写成一个 npz，并在里面留一枚 ``stamp``：输入内容与相关配置段的
指纹。指纹不变就直接复用已有结果。三个阶段依次链式盖章（Stage I 的指纹含 Stage 0
的指纹，Stage II 又含 Stage I 的），所以上游一变，下游必然跟着重算。

Stage 0 / I 只依赖「机器人 + 源骨架 + 演员骨架尺寸」，与具体是哪一段动作无关，
因此它们的产物按骨架签名放在共享目录里（见 :mod:`umr.paths`）。批量处理同一个
演员的几十段导出时，十来分钟的对应学习只跑一次。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from umr.bodies import bvh as bvh_mod
from umr.bodies.human_mjcf import (
    ALL_SEGMENTS,
    HumanBody,
    compute_ground_offset,
    write_human_mjcf,
)
from umr.bodies.robot import RobotBody, RobotSpec, prepare_robot_xml
from umr.bodies.skeletons import Skeleton
from umr.bodies.surface import SurfacePointCloud, sample_model_surface
from umr.config import Config
from umr.correspondence.evaluate import anatomical_consistency
from umr.correspondence.geodesic import build_geodesic_graph, graph_stats
from umr.correspondence.train import (
    evaluate_correspondence,
    make_normalization,
    resolve_device,
    train_correspondence,
)
from umr.paths import PROJECT_ROOT, ClipLayout, SetupLayout
from umr.retarget.binding import LinkBinding, bind_points_to_links
from umr.retarget.export import save_motion_pkl
from umr.retarget.pipeline import UMRRetargeter

Log = Callable[[str], None]

DEFAULT_INTERPOLATION = "slerp"


# ----------------------------------------------------------------------
# 结果缓存
# ----------------------------------------------------------------------
def _fingerprint(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _digest(path: str | Path) -> str:
    """文件内容摘要。

    这里刻意不用 mtime：``prepare_robot_xml`` 每次都会重写机器人 MJCF，按 mtime
    盖章的话缓存永远命中不了。
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _reusable(path: Path, stamp: str, force: bool) -> bool:
    if force or not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return str(z["stamp"]) == stamp
    except (OSError, KeyError, ValueError):
        return False


def _stamp_of(path: Path) -> str:
    with np.load(path, allow_pickle=True) as z:
        return str(z["stamp"])


# ----------------------------------------------------------------------
# 源动作加载
# ----------------------------------------------------------------------
def load_source(cfg: Config, bvh_path: str | Path, skeleton: Skeleton) -> bvh_mod.BvhData:
    """加载整段 BVH（未重采样、未截取）。

    源帧率一律从 BVH 头部的 ``Frame Time`` 读出（见
    :func:`~umr.bodies.bvh.read_bvh_raw`），所以没有也不需要 ``--ori_fps`` 这种入参。
    """
    return bvh_mod.load_bvh(
        bvh_path,
        scale=float(cfg.source["length_scale"]),
        auto_face_x=bool(cfg.source["auto_face_x"]),
        human=skeleton,
    )


SHARE_MODES = ("dir", "file")


def setup_signature(
    cfg: Config, skeleton: Skeleton, bvh_path: str | Path, share: str = "dir"
) -> str:
    """Stage 0 / I 产物的寻址键：同键的片段共用一套对应关系。

    ``share="dir"``
        同一个目录下的片段共用一套，代表帧取该目录里排序第一个文件。动捕的一次
        采集通常就是「一个演员 + 一次标定 + 一整个目录的 take」，而 FZMotion 这类
        系统会给每条 take 单独解算骨架，逐文件比对的话 1 cm 级的 offset 抖动就会
        让每条 take 都重训一次对应——32 段就是 25 分钟起步（CPU 上按小时算）。
    ``share="file"``
        逐文件一套。目录里混了多个体型差异明显的演员时用这个。

    两种模式下 ``bodies.npz`` 自己的 stamp 都仍然含代表帧的骨架签名，所以换了内容
    照样会就地重算，共享只影响"存在哪"，不影响"对不对"。
    """
    if share == "file":
        key = bvh_mod.skeleton_signature(bvh_path, strip_prefix=skeleton.strip_prefix)
    elif share == "dir":
        key = str(Path(bvh_path).resolve().parent)
    else:
        raise ValueError(f"未知的 --share_setup: {share}（可用 {' / '.join(SHARE_MODES)}）")
    return _fingerprint(
        "setup/2", _digest(PROJECT_ROOT / cfg.robot["xml"]), skeleton.name, share, key,
        cfg.robot, cfg.source, cfg.sampling, cfg.correspondence,
    )


def setup_representative(clips: list[Path]) -> Path:
    """一组共用 setup 的片段里，用哪一段来建人体模型：排序最靠前的那个。"""
    return sorted(clips)[0]


# ----------------------------------------------------------------------
# Stage 0：人机 MuJoCo 身体 + canonical T-pose 表面点云
# ----------------------------------------------------------------------
def build_bodies(
    cfg: Config,
    bvh_path: str | Path,
    setup: SetupLayout,
    *,
    skeleton: Skeleton,
    n_points: int | None = None,
    force: bool = False,
    log: Log = print,
) -> Path:
    """由 BVH 层级生成人体 MJCF、给机器人注入 T_pose，并采样两侧表面点云。

    人体先按机器人身高归一化（论文 Fig.2 的 Normalized Human Point Cloud），再用
    同一个采样器在 T-pose 下取出有序的 X^h 与无序的 X^r。

    这一步只用到 BVH 的层级与标定帧，不碰动作内容，所以产物按骨架签名共享。
    """
    bvh_path = Path(bvh_path)
    robot_xml = PROJECT_ROOT / cfg.robot["xml"]
    n_points = int(n_points if n_points is not None else cfg.sampling["n_points"])

    def stamp() -> str:
        return _fingerprint(
            "bodies/2", _digest(robot_xml), skeleton.name,
            bvh_mod.skeleton_signature(bvh_path, strip_prefix=skeleton.strip_prefix),
            cfg.robot, cfg.source, cfg.sampling, n_points,
        )

    if _reusable(setup.bodies, stamp(), force):
        log(f"[stage0] 复用 {setup.bodies}")
        return setup.bodies

    setup.ensure()
    timings: dict[str, float] = {}

    # 注入 T_pose keyframe 会改写 MJCF，指纹必须在改写之后再取。
    spec = RobotSpec.from_config(cfg.robot)
    prepare_robot_xml(robot_xml, spec=spec)
    rb = RobotBody(robot_xml, spec)
    robot_height = rb.height()
    robot_foot_h = rb.foot_height()
    log(f"[stage0] {cfg.robot['name']}  nq={rb.model.nq} nv={rb.nv}")
    log(f"[stage0] 身高 {robot_height:.4f} m  脚踝离地 {robot_foot_h:.4f} m")

    t0 = time.perf_counter()
    motion = load_source(cfg, bvh_path, skeleton)
    timings["bvh_parse"] = time.perf_counter() - t0
    h_actor = bvh_mod.actor_height(motion)
    scale = robot_height / h_actor
    log(f"[stage0] {bvh_path.name}  骨架={skeleton.name} joints={motion.num_joints} "
        f"frames={motion.num_frames} fps={motion.fps:.2f}")
    log(f"[stage0] 演员身高 {h_actor:.4f} m -> 归一化尺度 {scale:.4f}")

    info = write_human_mjcf(motion, setup.human_xml, scale=scale, human=skeleton)
    hb = HumanBody(setup.human_xml, motion, scale=scale)
    log(f"[stage0] 人体 MJCF -> {setup.human_xml}  nq={hb.model.nq} nbody={hb.model.nbody}")

    sampling = dict(
        oversample=float(cfg.sampling["oversample"]),
        cull_margin=float(cfg.sampling["cull_margin"]),
        seed=int(cfg.sampling["seed"]),
        segment_names=ALL_SEGMENTS,
    )

    hb.set_tpose()
    t0 = time.perf_counter()
    human_pc = sample_model_surface(
        hb.model, hb.data, n_points, geom_segment=info.geom_segment, **sampling
    )
    t_h = time.perf_counter() - t0

    rb.set_tpose()
    t0 = time.perf_counter()
    robot_pc = sample_model_surface(
        rb.model, rb.data, n_points, geom_ids=rb.surface_geoms(), **sampling
    )
    timings["point_sampling"] = t_h + (time.perf_counter() - t0)

    log(f"[stage0] 采样 human N={len(human_pc)}  robot N={len(robot_pc)}  "
        f"({timings['point_sampling']:.2f}s)")
    covered = sorted({human_pc.segment_names[s] for s in np.unique(human_pc.segment) if s >= 0})
    missing = [s for s in ALL_SEGMENTS if s not in covered]
    log(f"[stage0] 人体分段覆盖 {len(covered)}/{len(ALL_SEGMENTS)}"
        + (f"  缺失={missing}" if missing else ""))

    out = {
        "stamp": stamp(),
        "robot_name": str(cfg.robot["name"]),
        "robot_xml": str(robot_xml),
        "human_xml": str(setup.human_xml),
        "human": skeleton.name,
        "geom_segment": np.array(list(info.geom_segment.items()), dtype=object),
        # 建这套 setup 时用的片段，仅供追溯：同签名的其它片段共用同一份产物。
        "source_bvh": str(bvh_path),
        "scale": scale,
        "actor_height": h_actor,
        "robot_height": robot_height,
        "robot_foot_height": robot_foot_h,
        "timings": np.array(list(timings.items()), dtype=object),
    }
    out.update(human_pc.to_dict("human_"))
    out.update(robot_pc.to_dict("robot_"))
    np.savez(setup.bodies, **out)
    log(f"[stage0] -> {setup.bodies}")
    return setup.bodies


# ----------------------------------------------------------------------
# Stage I：点云对应学习（论文 III-B，式 1-5）
# ----------------------------------------------------------------------
def learn_correspondence(
    cfg: Config,
    setup: SetupLayout,
    *,
    epochs: int | None = None,
    device: str | None = None,
    force: bool = False,
    log: Log = print,
) -> Path:
    """一次性从对齐的 T-pose 学出有序的人机点对，并把机器人一侧吸附绑定到 link。

    这套对应关系在整段动作、乃至同一演员的所有片段中复用，对应论文 Table I 的
    "Reusable Point Cloud Correspondence Setup"。
    """
    ccfg = cfg.correspondence
    epochs = int(epochs if epochs is not None else ccfg["epochs"])
    stamp = _fingerprint("correspondence/2", _stamp_of(setup.bodies), ccfg, epochs)

    if _reusable(setup.correspondence, stamp, force):
        log(f"[stage1] 复用 {setup.correspondence}")
        return setup.correspondence

    bodies = np.load(setup.bodies, allow_pickle=True)
    human_pc = SurfacePointCloud.from_dict(bodies, "human_")
    robot_pc = SurfacePointCloud.from_dict(bodies, "robot_")
    log(f"[stage1] X^h N={len(human_pc)}  X^r N={len(robot_pc)}")

    # 归一化对齐（论文 Fig.2 的 Normalized Human Point Cloud）
    norm = make_normalization(human_pc.points, robot_pc.points)
    xh = norm.normalize_human(human_pc.points)
    xr = norm.normalize_robot(robot_pc.points)
    log(f"[stage1] 归一化尺度 {norm.scale:.4f} m")

    # 测地图（式 5 的边集 E）
    t0 = time.perf_counter()
    edges = build_geodesic_graph(
        human_pc.points, human_pc.segment, human_pc.segment_names,
        k=int(ccfg["geodesic_k"]),
    )
    t_geo = time.perf_counter() - t0
    st = graph_stats(edges, len(human_pc))
    log(f"[stage1] 测地图 edges={st['num_edges']} mean_degree={st['mean_degree']:.2f} "
        f"isolated={st['isolated']}  ({t_geo:.2f}s)")

    # 训练（式 1-5）
    requested = device or str(ccfg["device"])
    dev = resolve_device(requested)
    note = ""
    if requested == "cuda" and dev == "cpu":
        note = "  （请求了 cuda 但没有可用 GPU，已回退）"
    if dev == "cpu":
        note += "  CPU 上 2500 epochs 约需 11 分钟"
    log(f"[stage1] 训练 {epochs} epochs，设备 = {dev}{note}")
    xr_hat_n, deform_n, hist = train_correspondence(
        xh, xr, edges,
        epochs=epochs,
        lr=float(ccfg["lr"]),
        weight_decay=float(ccfg["weight_decay"]),
        latent_dim=int(ccfg["latent_dim"]),
        encoder_channels=list(ccfg["encoder_channels"]),
        decoder_hidden=list(ccfg["decoder_hidden"]),
        lambda_chamfer=float(ccfg["lambda_chamfer"]),
        lambda_repulsion=float(ccfg["lambda_repulsion"]),
        lambda_edge=float(ccfg["lambda_edge"]),
        repulsion_k=int(ccfg["repulsion_k"]),
        repulsion_radius=float(ccfg["repulsion_radius"]) / norm.scale,
        device=dev,
        seed=int(ccfg["seed"]),
    )
    log(f"[stage1] 训练耗时 {hist['train_time']:.2f}s  device={hist['device']}")

    metrics = evaluate_correspondence(xr_hat_n, xr, norm.scale)
    log(f"[stage1] Chamfer recon->target {metrics['chamfer_recon_to_target_mm']:.2f} mm  "
        f"target->recon {metrics['chamfer_target_to_recon_mm']:.2f} mm  "
        f"coverage@2cm {metrics['coverage_2cm']*100:.1f}%")

    xr_hat = norm.denormalize_robot(xr_hat_n)

    # 绑定到机器人 link
    rb = RobotBody(str(bodies["robot_xml"]), RobotSpec.from_config(cfg.robot))
    rb.set_tpose()
    t0 = time.perf_counter()
    binding = bind_points_to_links(rb.model, rb.data, xr_hat, rb.surface_geoms())
    t_bind = time.perf_counter() - t0
    log(f"[stage1] 绑定 {len(binding.body_ids)} 点到 {len(np.unique(binding.body_ids))} 个 link "
        f"({t_bind:.2f}s)  吸附距离 mean={binding.snap_distance.mean()*1000:.2f}mm "
        f"p95={np.percentile(binding.snap_distance, 95)*1000:.2f}mm")

    # 机器人点通过共享下标继承人体分段标签 —— 论文的核心主张之一
    inherited_segment = human_pc.segment.copy()
    overall, per_seg, top_links = anatomical_consistency(
        inherited_segment, human_pc.segment_names, binding.body_ids, rb.body_names
    )
    metrics["anatomical_consistency"] = overall
    log(f"[stage1] 解剖学一致性 {overall*100:.1f}%  (无需任何人工骨骼映射)")
    for seg in sorted(top_links):
        acc = per_seg.get(seg)
        acc_s = f"{acc*100:5.1f}%" if acc is not None else "   n/a"
        links = ", ".join(f"{n}:{c}" for n, c in top_links[seg])
        log(f"           {seg:<12} {acc_s}  -> {links}")

    out = {
        "stamp": stamp,
        "xr_hat": xr_hat,
        "xr_hat_normalized": xr_hat_n,
        "deform_normalized": deform_n,
        "edges": edges,
        "inherited_segment": inherited_segment,
        "segment_names": np.array(human_pc.segment_names, dtype=object),
        "stage1_metrics": np.array(list(metrics.items()), dtype=object),
        "stage1_timings": np.array(
            [("geodesic", t_geo), ("training", hist["train_time"]), ("binding", t_bind)],
            dtype=object,
        ),
        "loss_history": np.array(
            [[h["epoch"], h["loss"], h["chamfer"], h["repulsion"], h["edge"]]
             for h in hist["history"]]
        ),
    }
    out.update(norm.to_dict())
    out.update(binding.to_dict("bind_"))
    np.savez(setup.correspondence, **out)
    log(f"[stage1] -> {setup.correspondence}")
    return setup.correspondence


# ----------------------------------------------------------------------
# Stage II：对应引导的重定向（论文 III-C，式 6-14）
# ----------------------------------------------------------------------
def retarget_motion(
    cfg: Config,
    bvh_path: str | Path,
    setup: SetupLayout,
    clip: ClipLayout,
    *,
    skeleton: Skeleton,
    tgt_fps: float | None = None,
    interpolation: str = DEFAULT_INTERPOLATION,
    start: float = 0.0,
    duration: float | None = None,
    n_selected: int | None = None,
    iterations: int | None = None,
    trust_region: str | None = None,
    solver: str | None = None,
    point_selection: str | None = None,
    tpose_offset: float | None = None,
    lock_ankle_roll: bool = False,
    export_pkl: bool = True,
    force: bool = False,
    log: Log = print,
) -> Path:
    """复用 Stage I 的对应点，在 mink 的约束 Gauss-Newton QP 里逐帧求机器人关节角。"""
    rcfg = cfg.retarget
    overrides = {
        "tgt_fps": tgt_fps, "interpolation": interpolation,
        "start": start, "duration": duration, "n_selected": n_selected,
        "iterations": iterations, "trust_region": trust_region, "solver": solver,
        "point_selection": point_selection, "tpose_offset": tpose_offset,
        "lock_ankle_roll": lock_ankle_roll,
    }
    stamp = _fingerprint(
        "motion/2", _stamp_of(setup.correspondence), rcfg,
        _digest(bvh_path), skeleton.name, overrides,
    )

    if _reusable(clip.motion, stamp, force):
        log(f"[stage2] 复用 {clip.motion}")
        return clip.motion

    clip.ensure()
    bodies = np.load(setup.bodies, allow_pickle=True)
    corr = np.load(setup.correspondence, allow_pickle=True)
    human_pc = SurfacePointCloud.from_dict(bodies, "human_")
    binding = LinkBinding.from_dict(corr, "bind_")
    geom_segment = {str(k): str(v) for k, v in bodies["geom_segment"]}

    robot = RobotBody(str(bodies["robot_xml"]), RobotSpec.from_config(cfg.robot))
    source = load_source(cfg, bvh_path, skeleton)
    src_fps = source.fps
    # tgt_fps 为 None 就保持源帧率——源帧率从 BVH 头里自动读，不用命令行给。
    motion = bvh_mod.prepare_clip(
        source, tgt_fps=tgt_fps, start=start, duration=duration,
        interpolation=interpolation, log=lambda m: log(f"[stage2] {m}"),
    )

    scale = float(bodies["scale"])
    human = HumanBody(str(bodies["human_xml"]), motion, scale=scale)

    # prepare_clip 已经把标定帧和不要的时间段切掉了，这里逐帧求解即可。
    fps = motion.fps
    frame_indices = np.arange(motion.num_frames)
    log(f"[stage2] 求解 {fps:.1f} FPS x {len(frame_indices)} 帧 "
        f"({len(frame_indices) / fps:.1f}s)")

    # 贴地偏移逐片段计算：不同片段的最低点不同，不能跟着 Stage 0 一起共享。
    probe = np.linspace(frame_indices[0], frame_indices[-1], min(400, len(frame_indices)))
    ground = compute_ground_offset(human, probe.astype(int), geom_segment)
    human.root_offset = np.array([0.0, 0.0, ground])
    log(f"[stage2] 地面偏移 {ground:+.4f} m")

    retargeter = UMRRetargeter(
        robot, human,
        human_body_ids=human_pc.body_ids,
        human_local_pos=human_pc.local_pos,
        human_local_normal=human_pc.local_normal,
        robot_body_ids=binding.body_ids,
        robot_local_pos=binding.local_pos,
        robot_local_normal=binding.local_normal,
        segment=corr["inherited_segment"],
        segment_names=[str(s) for s in corr["segment_names"]],
        n_selected=int(n_selected if n_selected is not None else rcfg["n_selected"]),
        point_selection=str(
            point_selection if point_selection is not None else rcfg["point_selection"]
        ),
        tpose_offset=float(
            tpose_offset if tpose_offset is not None else rcfg["tpose_offset"]
        ),
        iterations=int(iterations if iterations is not None else rcfg["iterations"]),
        dt=float(rcfg["dt"]),
        damping=float(rcfg["damping"]),
        solver=str(solver or rcfg["solver"]),
        trust_region=str(trust_region or rcfg["trust_region"]),
        trust_region_radius=float(rcfg["trust_region_radius"]),
        floor_height=float(rcfg["floor_height"]),
        floor_band=float(rcfg["floor_band"]),
        floor_margin=float(rcfg["floor_margin"]),
        contact_threshold=float(rcfg["contact_threshold"]),
        contact_weight=float(rcfg["contact_weight"]),
        posture_cost=float(rcfg["posture_cost"]),
        self_collision=bool(rcfg["self_collision"]),
    )
    log(f"[stage2] |I|={len(retargeter.selected)}  "
        f"地面约束候选点={len(retargeter.floor_cache.local_pos)}  solver={retargeter.solver}  "
        f"trust_region={retargeter.trust_region}  iters/frame={retargeter.iterations}")
    # T-pose 基线是式 (7) 原式消不掉的那部分残差，报出来才好判断逐帧误差是大是小。
    log(f"[stage2] T-pose 基线 位置 {retargeter.tpose_baseline_mm:.1f}mm "
        f"法线 {retargeter.tpose_baseline_deg:.1f}deg  "
        f"偏置补偿 alpha={retargeter.tpose_offset:.2f}")

    result = retargeter.run(frame_indices, fps)

    if lock_ankle_roll:
        # 站定那一段是唯一能确信"脚平放在地上"的时刻，其余帧的踝 roll 一律锁到它。
        n_static = bvh_mod.static_head_frames(motion)
        if n_static < 2:
            n_static = min(int(round(0.5 * fps)), motion.num_frames)
            log(f"[stage2] 片头没找到静止段，回退到前 {n_static} 帧取踝 roll 锁定值")
        locked = retargeter.lock_ankle_roll(result, n_static)
        if locked:
            detail = "  ".join(f"{n}={np.degrees(v):+.1f}deg" for n, v in locked.items())
            log(f"[stage2] 踝 roll 锁定（片头静止 {n_static} 帧 / {n_static/fps:.2f}s）  {detail}")
        else:
            log("[stage2] 踝 roll 锁定已开启，但模型里找不到 *ankle_roll* 关节，已跳过")

    log(f"[stage2] 点匹配误差 mean={result.point_error.mean()*1000:.1f}mm  "
        f"median={np.median(result.point_error)*1000:.1f}mm  "
        f"p95={np.percentile(result.point_error, 95)*1000:.1f}mm")
    log(f"[stage2] 法线误差 mean={np.degrees(result.normal_error.mean()):.1f}deg  "
        f"接触点 mean={result.contact_count.mean():.1f}  QP 失败 {result.solve_failures}")
    log(f"[stage2] 吞吐 {result.timings['fps']:.1f} FPS  "
        f"(warmup {result.timings['warmup']:.2f}s, retarget {result.timings['retarget']:.1f}s)")

    np.savez(
        clip.motion,
        stamp=stamp,
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
        setup_dir=str(setup.root),
        robot_xml=str(bodies["robot_xml"]),
        human_xml=str(bodies["human_xml"]),
        scale=scale,
        ground_offset=ground,
        # 复原这一段人体动作所需的全部入参，供可视化与报告脚本重建（见
        # umr.sim.views.clip_from_meta）。
        bvh_path=str(bvh_path),
        human_skeleton=skeleton.name,
        length_scale=float(cfg.source["length_scale"]),
        auto_face_x=bool(cfg.source["auto_face_x"]),
        source_fps=src_fps,
        start=float(start),
        duration=-1.0 if duration is None else float(duration),
        interpolation=interpolation,
    )
    log(f"[stage2] -> {clip.motion}")

    if export_pkl:
        save_motion_pkl(
            clip.motion_pkl, robot.model, result.qpos, fps,
            extra={
                "source_file": str(bvh_path),
                "human": skeleton.name,
                "source_fps": src_fps,
                "robot_xml": str(bodies["robot_xml"]),
                "human_xml": str(bodies["human_xml"]),
                "frame_indices": result.frame_indices,
                "scale": scale,
                "ground_offset": ground,
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
        log(f"[stage2] -> {clip.motion_pkl}  (root_trans/root_rot(xyzw)/dof/dof_full/fps，兼容 GMR)")

    return clip.motion


def run_pipeline(
    cfg: Config,
    bvh_path: str | Path,
    setup: SetupLayout,
    clip: ClipLayout,
    *,
    skeleton: Skeleton,
    force: bool = False,
    log: Log = print,
    stage1: dict[str, Any] | None = None,
    stage2: dict[str, Any] | None = None,
) -> Path:
    """跑完 Stage 0 / I / II，返回 ``motion.npz`` 路径。"""
    build_bodies(cfg, bvh_path, setup, skeleton=skeleton, force=force, log=log)
    learn_correspondence(cfg, setup, force=force, log=log, **(stage1 or {}))
    return retarget_motion(
        cfg, bvh_path, setup, clip, skeleton=skeleton, force=force, log=log, **(stage2 or {})
    )
