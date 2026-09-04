#!/usr/bin/env python
"""Stage I：点云对应学习（论文 III-B，式 1-5）。

一次性从对齐的 canonical T-pose 学出一组有序的人机表面点对，并把机器人一侧的
点吸附绑定到具体 link。这套对应关系在整段动作中复用，对应论文 Table I 的
"Reusable Point Cloud Correspondence Setup"。

输入: $UMR_OUTPUT_DIR/bodies.npz
输出: $UMR_OUTPUT_DIR/correspondence.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from umr.bodies.robot import RobotBody
from umr.bodies.surface import SurfacePointCloud
from umr.config import load_config
from umr.correspondence.evaluate import anatomical_consistency
from umr.correspondence.geodesic import build_geodesic_graph, graph_stats
from umr.correspondence.train import (
    evaluate_correspondence,
    make_normalization,
    resolve_device,
    train_correspondence,
)
from umr.paths import OUTPUT_DIR
from umr.retarget.binding import bind_points_to_links

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

robot_surface_geoms = import_module("01_build_bodies").robot_surface_geoms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--bodies", default=str(OUTPUT_DIR / "bodies.npz"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "correspondence.npz"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"],
                    help="Stage I 训练设备；默认取配置里的 correspondence.device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg.correspondence
    bodies = np.load(args.bodies, allow_pickle=True)

    human_pc = SurfacePointCloud.from_dict(bodies, "human_")
    robot_pc = SurfacePointCloud.from_dict(bodies, "robot_")
    print(f"[stage1] X^h N={len(human_pc)}  X^r N={len(robot_pc)}")

    # ------------------------------------------------------------------
    # 归一化对齐（论文 Fig.2 的 Normalized Human Point Cloud）
    # ------------------------------------------------------------------
    norm = make_normalization(human_pc.points, robot_pc.points)
    xh = norm.normalize_human(human_pc.points)
    xr = norm.normalize_robot(robot_pc.points)
    print(f"[stage1] 归一化尺度 = {norm.scale:.4f} m")

    # ------------------------------------------------------------------
    # 测地图（式 5 的边集 E）
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    edges = build_geodesic_graph(
        human_pc.points, human_pc.segment, human_pc.segment_names,
        k=int(ccfg["geodesic_k"]),
    )
    t_geo = time.perf_counter() - t0
    st = graph_stats(edges, len(human_pc))
    print(
        f"[stage1] 测地图 edges={st['num_edges']} mean_degree={st['mean_degree']:.2f} "
        f"isolated={st['isolated']}  ({t_geo:.2f}s)"
    )

    # ------------------------------------------------------------------
    # 训练（式 1-5）
    # ------------------------------------------------------------------
    epochs = args.epochs or int(ccfg["epochs"])
    requested = args.device or str(ccfg["device"])
    device = resolve_device(requested)
    note = ""
    if requested == "cuda" and device == "cpu":
        note = "  （请求了 cuda 但没有可用 GPU，已回退）"
    if device == "cpu":
        note += "  CPU 上 2500 epochs 约需 11 分钟"
    print(f"[stage1] 训练 {epochs} epochs，设备 = {device}{note}")
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
        device=device,
        seed=int(ccfg["seed"]),
    )
    print(f"[stage1] 训练耗时 {hist['train_time']:.2f}s  device={hist['device']}")

    metrics = evaluate_correspondence(xr_hat_n, xr, norm.scale)
    print(
        f"[stage1] Chamfer recon->target = {metrics['chamfer_recon_to_target_mm']:.2f} mm  "
        f"target->recon = {metrics['chamfer_target_to_recon_mm']:.2f} mm  "
        f"coverage@2cm = {metrics['coverage_2cm']*100:.1f}%"
    )

    xr_hat = norm.denormalize_robot(xr_hat_n)

    # ------------------------------------------------------------------
    # 绑定到机器人 link
    # ------------------------------------------------------------------
    rb = RobotBody.from_bodies(bodies)
    rb.set_tpose()
    t0 = time.perf_counter()
    binding = bind_points_to_links(rb.model, rb.data, xr_hat, robot_surface_geoms(rb))
    t_bind = time.perf_counter() - t0
    print(
        f"[stage1] 绑定 {len(binding.body_ids)} 点到 {len(np.unique(binding.body_ids))} 个 link "
        f"({t_bind:.2f}s)  吸附距离 mean={binding.snap_distance.mean()*1000:.2f}mm "
        f"p95={np.percentile(binding.snap_distance,95)*1000:.2f}mm"
    )

    # 机器人点通过共享下标继承人体分段标签 —— 论文的核心主张之一
    inherited_segment = human_pc.segment.copy()

    overall, per_seg, top_links = anatomical_consistency(
        inherited_segment, human_pc.segment_names, binding.body_ids, rb.body_names
    )
    metrics["anatomical_consistency"] = overall
    print(f"[stage1] 解剖学一致性 = {overall*100:.1f}%  (无需任何人工骨骼映射)")
    for seg in sorted(top_links):
        acc = per_seg.get(seg)
        acc_s = f"{acc*100:5.1f}%" if acc is not None else "   n/a"
        links = ", ".join(f"{n}:{c}" for n, c in top_links[seg])
        print(f"           {seg:<12} {acc_s}  -> {links}")

    out = {
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
            [[h["epoch"], h["loss"], h["chamfer"], h["repulsion"], h["edge"]] for h in hist["history"]]
        ),
    }
    out.update(norm.to_dict())
    out.update(binding.to_dict("bind_"))
    np.savez(args.out, **out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
