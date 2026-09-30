from __future__ import annotations

import os
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)

try:
    from torch._dynamo import OptimizedModule
except ImportError:
    OptimizedModule = ()


def _unwrap(net):
    mod = net.module if hasattr(net, "module") else net
    if OptimizedModule and isinstance(mod, OptimizedModule):
        mod = mod._orig_mod
    return mod


class nnUNetTrainerT2Warm(nnUNetTrainerNoMirroring):
    NUM_EPOCHS = 1000
    WARM_LR = 3e-3
    CKPT_NAME = "checkpoint_final.pth"

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.NUM_EPOCHS
        self.initial_lr = self.WARM_LR

    def initialize(self):
        already = self.was_initialized
        super().initialize()
        if not already:
            self._warm_start()

    def _warm_start(self) -> None:
        init_dir = os.environ.get("ATM_T2_INIT")
        if not init_dir:
            raise RuntimeError(
                "set ATM_T2_INIT to the folder holding the ATM'26 Track-2 baseline "
                "nnUNet_ckpts (fold_*/checkpoint_final.pth)")
        ckpt = Path(init_dir) / f"fold_{self.fold}" / self.CKPT_NAME
        if not ckpt.is_file():
            raise FileNotFoundError(f"warm start weights not found: {ckpt}")

        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        src = saved["network_weights"]
        mod = _unwrap(self.network)
        dst = mod.state_dict()

        missing, mismatched = [], []
        for key, tensor in dst.items():
            if key not in src:
                missing.append(key)
            elif tuple(src[key].shape) != tuple(tensor.shape):
                mismatched.append(key)
            else:
                dst[key] = src[key]
        if missing or mismatched:
            raise RuntimeError(
                f"{ckpt} does not match the plans: {len(missing)} missing, "
                f"{len(mismatched)} mismatched, e.g. {(missing + mismatched)[:5]}")
        mod.load_state_dict(dst)
        self.print_to_log_file(f"warm start: loaded {len(dst)} tensors from {ckpt}")
