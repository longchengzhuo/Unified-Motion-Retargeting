"""Stage I 的网络：PointNet 编码器 + MLP 解码器（论文式 1）。

.. math::

    \\hat{X}^r = X^h + D_\\theta(E_\\theta(X^r))

编码器 :math:`E_\\theta` 把**无序**的机器人点云汇总成一个全局隐向量；解码器
:math:`D_\\theta` 对每个**有序**的人体模板点预测一个形变向量，于是重建点的下标
天然继承自人体点云，这正是论文所说的 "indexed correspondence"。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """PointNet 风格编码器：逐点共享 MLP + 全局 max-pool。"""

    def __init__(self, channels: list[int] = [3, 64, 128, 256, 512]):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], 1),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.mlp = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, N, 3) -> (B, latent)。"""
        h = self.mlp(x.transpose(1, 2))
        return h.max(dim=2).values


class DeformationDecoder(nn.Module):
    """逐点 MLP 解码器：由 (人体点坐标, 全局隐向量) 预测形变向量。"""

    def __init__(self, latent_dim: int, hidden: list[int] = [512, 512, 256]):
        super().__init__()
        dims = [3 + latent_dim] + list(hidden)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Conv1d(dims[i], dims[i + 1], 1), nn.ReLU(inplace=True)]
        layers += [nn.Conv1d(dims[-1], 3, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, xh: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Args: xh (B, N, 3), latent (B, L) -> (B, N, 3) 形变向量。"""
        B, N, _ = xh.shape
        z = latent.unsqueeze(2).expand(B, latent.shape[1], N)
        inp = torch.cat([xh.transpose(1, 2), z], dim=1)
        return self.mlp(inp).transpose(1, 2)


class CorrespondenceNet(nn.Module):
    """完整的对应关系网络，实现论文式 (1)。"""

    def __init__(
        self,
        encoder_channels: list[int] = [3, 64, 128, 256, 512],
        decoder_hidden: list[int] = [512, 512, 256],
    ):
        super().__init__()
        self.encoder = PointNetEncoder(encoder_channels)
        self.decoder = DeformationDecoder(self.encoder.out_dim, decoder_hidden)

    def forward(self, xh: torch.Tensor, xr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: xh (B, N, 3) 有序人体点云, xr (B, M, 3) 无序机器人点云。

        Returns:
            ``(xr_hat, deform)``，均为 (B, N, 3)。
        """
        latent = self.encoder(xr)
        deform = self.decoder(xh, latent)
        return xh + deform, deform
