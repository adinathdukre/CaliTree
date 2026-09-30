from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from airway.losses.cldice import SkeletonPrecisionLoss


class GeneralUnionLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, r: float = 0.7, smooth: float = 1e-5,
                 apply_nonlin: bool = True):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.r = r
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(net_output) if self.apply_nonlin else net_output
        p = p.float()
        g = target.float()
        num = (p.clamp_min(1e-6).pow(self.r) * g).sum()
        den = (self.alpha * p + self.beta * g).sum()
        return 1.0 - (num + self.smooth) / (den + self.smooth)


class SkeletonRecallLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, apply_nonlin: bool = True):
        super().__init__()
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, net_output: torch.Tensor, skeleton: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(net_output) if self.apply_nonlin else net_output
        s = skeleton.float()
        return 1.0 - ((p.float() * s).sum() + self.smooth) / (s.sum() + self.smooth)


class AirwayRegionLoss(nn.Module):
    def __init__(self, batch_dice: bool = False, ddp: bool = True,
                 smooth: float = 1e-5,
                 airway_channel: int = 0, centerline_channel: int = 3,
                 gul_weight: float = 1.0, gul_alpha: float = 0.2, gul_r: float = 0.7,
                 skel_weight: float = 1.0,
                 skelprec_weight: float = 0.0, skelprec_iters: int = 10,
                 weight_bce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=torch.sigmoid, batch_dice=batch_dice, do_bg=True,
            smooth=smooth, ddp=ddp)
        self.airway_channel = airway_channel
        self.centerline_channel = centerline_channel
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.gul_weight = gul_weight
        self.skel_weight = skel_weight
        self.skelprec_weight = skelprec_weight
        self.gul = GeneralUnionLoss(alpha=gul_alpha, r=gul_r) if gul_weight > 0 else None
        self.skel = SkeletonRecallLoss() if skel_weight > 0 else None
        self.skelprec = SkeletonPrecisionLoss(iters=skelprec_iters) \
            if skelprec_weight > 0 else None

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        n = net_output.shape[1]

        bce_map = self.bce(net_output, target)
        bce = bce_map.mean(dim=[0] + list(range(2, bce_map.ndim))).mean()

        dice = net_output.new_zeros(())
        for c in range(n):
            dice = dice + self.dc(net_output[:, c:c + 1], target[:, c:c + 1])
        dice = dice / n

        loss = self.weight_bce * bce + self.weight_dice * dice

        a = self.airway_channel
        if self.gul is not None:
            loss = loss + self.gul_weight * self.gul(
                net_output[:, a:a + 1], target[:, a:a + 1])

        if self.skel is not None:
            if n <= self.centerline_channel:
                raise ValueError(
                    "Skeleton Recall needs a centerline region; the network predicts "
                    f"only {n} regions")
            c = self.centerline_channel
            loss = loss + self.skel_weight * self.skel(
                net_output[:, a:a + 1], target[:, c:c + 1])

        if self.skelprec is not None:
            loss = loss + self.skelprec_weight * self.skelprec(
                net_output[:, a:a + 1], target[:, a:a + 1])
        return loss


class MixedDeepSupervisionWrapper(nn.Module):
    def __init__(self, loss_full: nn.Module, loss_rest: nn.Module, weight_factors):
        super().__init__()
        assert any(x != 0 for x in weight_factors), "at least one weight must be non-zero"
        self.weight_factors = tuple(weight_factors)
        self.loss_full = loss_full
        self.loss_rest = loss_rest

    def forward(self, *args):
        total = None
        for i, inputs in enumerate(zip(*args)):
            wi = self.weight_factors[i]
            if wi == 0.0:
                continue
            l = (self.loss_full if i == 0 else self.loss_rest)(*inputs)
            total = wi * l if total is None else total + wi * l
        return total


def deep_supervision_weights(n_scales: int, is_ddp: bool, compiled: bool):
    weights = np.array([1 / (2 ** i) for i in range(n_scales)], dtype=float)
    weights[-1] = 1e-6 if (is_ddp and not compiled) else 0.0
    return weights / weights.sum()
