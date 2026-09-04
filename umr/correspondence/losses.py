"""Stage I 的训练目标（论文式 2-5）。

.. math::

    L_{corr} = \\lambda_c L_c + \\lambda_r L_r + \\lambda_e L_e
"""

from __future__ import annotations

import torch


def _pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) 与 (B, M, 3) 之间的平方距离矩阵 (B, N, M)。"""
    return torch.cdist(a, b, p=2.0).pow(2)


def chamfer_loss(xr_hat: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
    """式 (3)：对称的最近邻平方距离。"""
    d = _pairwise_sq_dist(xr_hat, xr)
    return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()


def repulsion_loss(xr_hat: torch.Tensor, k: int = 8, radius: float = 0.03) -> torch.Tensor:
    """式 (4)：阻止多个重建点塌缩到同一局部区域。

    .. math::  L_r = \\frac{1}{N K_r}\\sum_i \\sum_{\\ell \\in N_r(i)}
               \\exp(-\\|\\hat{x}^r_i - \\hat{x}^r_\\ell\\|_2^2 / r^2)
    """
    d = _pairwise_sq_dist(xr_hat, xr_hat)
    # 排除自身
    n = d.shape[1]
    eye = torch.eye(n, device=d.device, dtype=torch.bool).unsqueeze(0)
    d = d.masked_fill(eye, float("inf"))
    knn = d.topk(k, dim=2, largest=False).values
    return torch.exp(-knn / (radius**2)).mean()


def edge_smoothness_loss(deform: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """式 (5)：在人体模板的固定测地图上约束形变向量平滑。

    Args:
        deform: (B, N, 3) 形变向量。
        edges: (E, 2) 边集 ``E``。
    """
    di = deform[:, edges[:, 0]]
    dl = deform[:, edges[:, 1]]
    return (di - dl).pow(2).sum(dim=-1).mean()


def correspondence_loss(
    xr_hat: torch.Tensor,
    deform: torch.Tensor,
    xr: torch.Tensor,
    edges: torch.Tensor,
    lambda_chamfer: float = 1.0,
    lambda_repulsion: float = 0.3,
    lambda_edge: float = 3.0,
    repulsion_k: int = 8,
    repulsion_radius: float = 0.03,
) -> tuple[torch.Tensor, dict[str, float]]:
    """式 (2) 的总损失，返回 ``(loss, 各项标量)``。"""
    lc = chamfer_loss(xr_hat, xr)
    lr = repulsion_loss(xr_hat, k=repulsion_k, radius=repulsion_radius)
    le = edge_smoothness_loss(deform, edges)
    loss = lambda_chamfer * lc + lambda_repulsion * lr + lambda_edge * le
    return loss, {
        "loss": loss.detach().item(),
        "chamfer": lc.detach().item(),
        "repulsion": lr.detach().item(),
        "edge": le.detach().item(),
    }


@torch.no_grad()
def chamfer_distance_mm(xr_hat: torch.Tensor, xr: torch.Tensor, scale: float) -> tuple[float, float]:
    """评估用：以毫米表示的单向平均最近邻距离 (recon->target, target->recon)。"""
    d = _pairwise_sq_dist(xr_hat, xr).sqrt() * scale * 1000.0
    return float(d.min(dim=2).values.mean()), float(d.min(dim=1).values.mean())
