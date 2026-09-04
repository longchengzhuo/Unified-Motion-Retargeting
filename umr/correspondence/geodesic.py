"""人体模板上的固定测地图（论文式 5 中的边集 E）。

论文用"人体模板的测地图"来约束形变向量的局部平滑。直接在 3D 空间做 kNN 会在
身体贴近处产生错误连边（例如 T-pose 下两条大腿内侧、手臂与躯干），使平滑项把
本应分开的区域粘在一起。

这里默认用**分段感知 kNN 图**：只有当两点属于同一分段、或属于运动学上相邻的
两个分段时才连边。这等价于在身体表面的连通分量上做近邻搜索，是测地邻域的一个
廉价而稳健的近似。若安装了 ``potpourri3d``，可用 ``method="heat"`` 在真实模板
网格上跑热法测地线。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from umr.bodies.human_mjcf import SEGMENT_ADJACENCY


def build_segment_adjacency_matrix(segment_names: list[str]) -> np.ndarray:
    """(S, S) 布尔矩阵：分段之间是否允许连边（含自身）。"""
    s = len(segment_names)
    idx = {n: i for i, n in enumerate(segment_names)}
    adj = np.eye(s, dtype=bool)
    for a, b in SEGMENT_ADJACENCY:
        if a in idx and b in idx:
            adj[idx[a], idx[b]] = True
            adj[idx[b], idx[a]] = True
    return adj


def build_geodesic_graph(
    points: np.ndarray,
    segment: np.ndarray,
    segment_names: list[str],
    k: int = 8,
    max_radius: float | None = None,
) -> np.ndarray:
    """构造分段感知的 kNN 图。

    Args:
        points: (N, 3) 人体模板 T-pose 点云。
        segment: (N,) 每点的分段 id。
        segment_names: 分段名列表。
        k: 每个点保留的近邻数。
        max_radius: 边长上限（米）；None 时取平均最近邻距离的 6 倍。

    Returns:
        (E, 2) 无向边集（i < l，已去重）。
    """
    n = points.shape[0]
    tree = cKDTree(points)
    # 多取一些候选，因为跨分段的边会被过滤掉
    kq = min(n, k * 4 + 1)
    dist, idx = tree.query(points, k=kq)

    if max_radius is None:
        max_radius = float(np.median(dist[:, 1]) * 6.0)

    adj = build_segment_adjacency_matrix(segment_names)
    seg_ok = np.zeros_like(idx, dtype=bool)
    seg_i = segment[:, None]
    seg_j = segment[idx]
    valid_seg = (seg_i >= 0) & (seg_j >= 0)
    seg_ok[valid_seg] = adj[seg_i.repeat(kq, axis=1)[valid_seg], seg_j[valid_seg]]
    # 未标注的点退化为普通 kNN
    seg_ok |= ~valid_seg

    keep = seg_ok & (dist <= max_radius)
    keep[:, 0] = False  # 自身

    edges = set()
    for i in range(n):
        cand = idx[i][keep[i]][:k]
        for j in cand:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a != b:
                edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def graph_stats(edges: np.ndarray, n: int) -> dict[str, float]:
    deg = np.bincount(edges.reshape(-1), minlength=n)
    return {
        "num_edges": int(len(edges)),
        "mean_degree": float(deg.mean()),
        "isolated": int((deg == 0).sum()),
    }
