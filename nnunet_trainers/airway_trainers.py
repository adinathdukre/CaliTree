from __future__ import annotations

import os
import sys

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from airway.losses.airway_losses import (
    AirwayRegionLoss, MixedDeepSupervisionWrapper, deep_supervision_weights)


class nnUNetTrainerAirwayCB(nnUNetTrainer):
    gul_weight = 1.0
    gul_alpha = 0.2
    gul_r = 0.7
    skel_weight = 1.0
    skelprec_weight = 0.0
    skelprec_iters = 10
    oversample_fg = 0.5

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = self.oversample_fg

    def _make_loss(self, full_res: bool):
        n = len(self.label_manager.foreground_regions)
        common = dict(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
            smooth=1e-5,
            airway_channel=0,
            centerline_channel=3,
        )
        if not full_res:
            return AirwayRegionLoss(gul_weight=0.0, skel_weight=0.0, **common)
        return AirwayRegionLoss(
            gul_weight=self.gul_weight, gul_alpha=self.gul_alpha, gul_r=self.gul_r,
            skel_weight=self.skel_weight if n > 3 else 0.0,
            skelprec_weight=self.skelprec_weight,
            skelprec_iters=self.skelprec_iters, **common)

    def _build_loss(self):
        if not self.label_manager.has_regions:
            raise RuntimeError(
                f"{type(self).__name__} expects a region-based dataset.json")
        if self.label_manager.ignore_label is not None:
            raise NotImplementedError("ignore_label is not supported by this trainer")

        loss_full = self._make_loss(full_res=True)
        loss_rest = self._make_loss(full_res=False)
        self.print_to_log_file(
            f"loss: AirwayRegionLoss over {len(self.label_manager.foreground_regions)} "
            f"regions; GUL(w={self.gul_weight}, alpha={self.gul_alpha}, r={self.gul_r}) + "
            f"SkeletonRecall(w={self.skel_weight}) + "
            f"SkeletonPrecision(w={self.skelprec_weight}, iters={self.skelprec_iters})")
        if not self.enable_deep_supervision:
            return loss_full
        n_scales = len(self._get_deep_supervision_scales())
        weights = deep_supervision_weights(n_scales, self.is_ddp, self._do_i_compile())
        return MixedDeepSupervisionWrapper(loss_full, loss_rest, weights)


class nnUNetTrainerAirwayCB_clprec025(nnUNetTrainerAirwayCB):
    gul_alpha = 0.1
    skelprec_weight = 0.25
