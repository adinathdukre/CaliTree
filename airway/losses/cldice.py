from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SkeletonPrecisionLoss(nn.Module):

    def __init__(self, iters: int = 10, smooth: float = 1.0,
                 apply_nonlin: bool = True):
        super().__init__()
        self.iters = iters
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, net_output: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(net_output) if self.apply_nonlin else net_output
        p = p.float()
        skel = soft_skel(p, self.iters)
        g = target.float()
        inside = (skel * g).sum()
        total = skel.sum()
        return 1.0 - (inside + self.smooth) / (total + self.smooth)
