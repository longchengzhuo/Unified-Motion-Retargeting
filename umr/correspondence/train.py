"""Stage I 的训练循环（论文 III-B）。

只学一次：给定人机两侧的 canonical T-pose 点云，学到一组可复用的有序对应点。
论文 Table I 把这一步归为 "Reusable Point Cloud Correspondence Setup"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from umr.correspondence.losses import chamfer_distance_mm, correspondence_loss
from umr.correspondence.model import CorrespondenceNet


def resolve_device(device: str = "auto") -> str:
    """把 ``auto`` / ``cuda`` / ``cpu`` 解析成实际可用的设备。

    GPU 只在 Stage I 的对应学习里用到（Stage II 全程 numpy + MuJoCo，不碰 torch）。
    请求 ``cuda`` 但机器上没有可用 GPU 时自动退回 ``cpu``，只是慢一些，结果不受影响。
    """
    device = str(device).lower()
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device 只能是 auto / cuda / cpu，收到 {device!r}")
    if device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Normalization:
    """把两侧点云对齐到同一归一化空间（论文 Fig.2 的 Normalized Human Point Cloud）。

    人体在 Stage 0 已按机器人身高缩放，因此这里只需各自去中心、共享一个尺度。
    """

    human_center: np.ndarray
    robot_center: np.ndarray
    scale: float

    def normalize_human(self, x: np.ndarray) -> np.ndarray:
        return (x - self.human_center) / self.scale

    def normalize_robot(self, x: np.ndarray) -> np.ndarray:
        return (x - self.robot_center) / self.scale

    def denormalize_robot(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.robot_center

    def to_dict(self) -> dict:
        return {
            "norm_human_center": self.human_center,
            "norm_robot_center": self.robot_center,
            "norm_scale": np.float64(self.scale),
        }

    @staticmethod
    def from_dict(d) -> "Normalization":
        return Normalization(
            human_center=d["norm_human_center"],
            robot_center=d["norm_robot_center"],
            scale=float(d["norm_scale"]),
        )


def make_normalization(human_pts: np.ndarray, robot_pts: np.ndarray) -> Normalization:
    ch = human_pts.mean(axis=0)
    cr = robot_pts.mean(axis=0)
    sh = np.linalg.norm(human_pts - ch, axis=1).max()
    sr = np.linalg.norm(robot_pts - cr, axis=1).max()
    return Normalization(human_center=ch, robot_center=cr, scale=float(max(sh, sr)))


def train_correspondence(
    human_pts: np.ndarray,
    robot_pts: np.ndarray,
    edges: np.ndarray,
    *,
    epochs: int = 1500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    latent_dim: int = 512,
    encoder_channels: list[int] | None = None,
    decoder_hidden: list[int] | None = None,
    lambda_chamfer: float = 1.0,
    lambda_repulsion: float = 0.3,
    lambda_edge: float = 3.0,
    repulsion_k: int = 8,
    repulsion_radius: float = 0.03,
    device: str = "auto",
    seed: int = 0,
    log_every: int = 100,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """学习人机点云之间的有序对应。

    Args:
        human_pts: (N, 3) 归一化后的人体 T-pose 点云 ``X^h``（有序）。
        robot_pts: (M, 3) 归一化后的机器人 T-pose 点云 ``X^r``（无序）。
        edges: (E, 2) 人体模板的测地图边集。
        其余为超参，含义见论文式 (2)-(5)。

    Returns:
        ``(xr_hat, deform, history)``，前两者形状 (N, 3)，均在归一化空间中。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 让 GPU 上的结果尽量可复现；部分 reduce 仍有原子加带来的微小非确定性。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device(resolve_device(device))

    xh = torch.as_tensor(human_pts, dtype=torch.float32, device=dev).unsqueeze(0)
    xr = torch.as_tensor(robot_pts, dtype=torch.float32, device=dev).unsqueeze(0)
    ed = torch.as_tensor(edges, dtype=torch.long, device=dev)

    net = CorrespondenceNet(
        encoder_channels=encoder_channels or [3, 64, 128, 256, latent_dim],
        decoder_hidden=decoder_hidden or [512, 512, 256],
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    history: list[dict] = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        opt.zero_grad(set_to_none=True)
        xr_hat, deform = net(xh, xr)
        loss, parts = correspondence_loss(
            xr_hat, deform, xr, ed,
            lambda_chamfer=lambda_chamfer,
            lambda_repulsion=lambda_repulsion,
            lambda_edge=lambda_edge,
            repulsion_k=repulsion_k,
            repulsion_radius=repulsion_radius,
        )
        loss.backward()
        opt.step()
        sched.step()
        parts["epoch"] = ep
        history.append(parts)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(
                f"  ep {ep:5d}  loss={parts['loss']:.6f}  "
                f"chamfer={parts['chamfer']:.6f}  repulsion={parts['repulsion']:.6f}  "
                f"edge={parts['edge']:.6f}"
            )
    train_time = time.perf_counter() - t0

    net.eval()
    with torch.no_grad():
        xr_hat, deform = net(xh, xr)
    return (
        xr_hat.squeeze(0).cpu().numpy().astype(np.float64),
        deform.squeeze(0).cpu().numpy().astype(np.float64),
        {"history": history, "train_time": train_time, "device": str(dev)},
    )


def evaluate_correspondence(
    xr_hat: np.ndarray, robot_pts: np.ndarray, scale: float, device: str = "cpu"
) -> dict[str, float]:
    """Stage I 的质量指标：双向 Chamfer 距离（毫米）与覆盖率。"""
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    a = torch.as_tensor(xr_hat, dtype=torch.float32, device=dev).unsqueeze(0)
    b = torch.as_tensor(robot_pts, dtype=torch.float32, device=dev).unsqueeze(0)
    fwd, bwd = chamfer_distance_mm(a, b, scale)

    # 覆盖率：有多少目标点被至少一个重建点在阈值内命中
    from scipy.spatial import cKDTree

    tree = cKDTree(xr_hat)
    d, _ = tree.query(robot_pts, k=1)
    thresh = 0.02 / scale  # 归一化空间中的 2cm
    return {
        "chamfer_recon_to_target_mm": fwd,
        "chamfer_target_to_recon_mm": bwd,
        "coverage_2cm": float((d < thresh).mean()),
    }
