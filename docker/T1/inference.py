from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_REPO = Path(__file__).resolve().parents[2]
if (_REPO / "nnunet_trainers").is_dir():
    os.environ.setdefault("nnUNet_extTrainer", str(_REPO / "nnunet_trainers"))
    sys.path.insert(0, str(_REPO))

RESOURCES = Path(os.environ.get("ATM_RESOURCES", "/opt/app/resources"))
OUTPUT_SLUG = "binary-airway-segmentation"


def _log(msg: str) -> None:
    print(f"[atm26-t1] {msg}", flush=True)


MAX_FORWARD_PASSES = 2700

MIRROR_AXES = (2,)
TTA_PASSES = 2 ** len(MIRROR_AXES)

DEFAULT_TILE_STEP = 0.75


def plan_tile_step(image, predictor, folds, base_step: float):
    try:
        from nnunetv2.inference.sliding_window_prediction import (
            compute_steps_for_sliding_window)
        cm = predictor.configuration_manager
        patch = list(cm.patch_size)
        target = list(cm.spacing)
        sp = list(image.GetSpacing())[::-1]
        sh = list(image.GetSize())[::-1]
        res = [max(int(round(sh[i] * sp[i] / target[i])), patch[i]) for i in range(3)]
        for step in (base_step, 0.85, 1.0):
            n = 1
            for s in compute_steps_for_sliding_window(res, patch, step):
                n *= len(s)
            total = n * max(len(folds), 1) * TTA_PASSES
            if total <= MAX_FORWARD_PASSES or step >= 1.0:
                _log(f"resampled {res}, {n} tiles/fold x {len(folds)} folds "
                     f"x {TTA_PASSES} TTA = {total} forward passes "
                     f"at tile_step {step}")
                return step
    except Exception as exc:
        _log(f"tile-step planning failed ({exc}); using {base_step}")
    return base_step


def _load_postproc_config():
    from airway.inference.predict import PostprocConfig
    from airway.inference.reconnect import ReconnectConfig

    cfg_path = RESOURCES / "postproc.json"
    cfg = PostprocConfig()
    if cfg_path.is_file():
        raw = json.loads(cfg_path.read_text())
        if "config" in raw and isinstance(raw["config"], dict):
            raw = raw["config"]
        rc = raw.pop("reconnect_cfg", None)
        applied = [k for k, v in raw.items() if hasattr(cfg, k)]
        for k in applied:
            setattr(cfg, k, raw[k])
        if rc:
            cfg.reconnect_cfg = ReconnectConfig(**rc)
        if not applied:
            raise RuntimeError(
                f"{cfg_path.name} set no PostprocConfig field (keys: "
                f"{sorted(raw)}); refusing to run with silent defaults")
        _log(f"loaded tuned post-processing from {cfg_path.name}: {applied}")
    return cfg


def main() -> int:
    t_start = time.time()

    from airway.inference import batch as batch_io
    from airway.inference.predict import build_predictor, postprocess

    cfg = _load_postproc_config()

    model_dir = RESOURCES / "nnUNet_ckpts"
    folds = tuple(int(p.name.split("_")[-1]) for p in sorted(model_dir.glob("fold_*"))
                  if p.is_dir())
    if not folds:
        raise RuntimeError(f"no fold_* directories under {model_dir}")

    device = os.environ.get("ATM_DEVICE", "cuda")
    if os.environ.get("ATM_FOLDS"):
        folds = tuple(int(f) for f in os.environ["ATM_FOLDS"].split(","))
    tile_step = float(os.environ.get("ATM_TILE_STEP", str(DEFAULT_TILE_STEP)))
    if (device != "cuda" or os.environ.get("ATM_FOLDS")
            or tile_step != DEFAULT_TILE_STEP):
        _log(f"overrides device={device} folds={folds} tile_step={tile_step}")

    t = time.time()
    predictor = build_predictor(str(model_dir), folds,
                                checkpoint="checkpoint_best.pth",
                                tile_step_size=tile_step,
                                device=device, use_mirroring=True)
    allowed = predictor.allowed_mirroring_axes
    if allowed is None or not set(MIRROR_AXES) <= set(allowed):
        raise RuntimeError(
            f"checkpoint allows mirroring axes {allowed}; "
            f"{MIRROR_AXES} is not a subset, refusing to run flip TTA")
    predictor.allowed_mirroring_axes = MIRROR_AXES
    _log(f"predictor ready in {time.time() - t:.0f}s, folds={folds}, "
         f"flip TTA mirror axes {MIRROR_AXES} ({TTA_PASSES} passes/tile)")

    def predict_one(image, path):
        predictor.tile_step_size = tile_step
        planned = plan_tile_step(image, predictor, folds, tile_step)
        if planned != tile_step:
            _log(f"  size guard: tile_step {tile_step} -> {planned}")
            predictor.tile_step_size = planned

        npy = sitk.GetArrayFromImage(image)[None].astype(np.float32)
        props = {"spacing": list(image.GetSpacing())[::-1]}
        t0 = time.time()
        _seg, probs = predictor.predict_single_npy_array(
            npy, props, None, None, True)
        del npy, _seg
        probs = np.asarray(probs, dtype=np.float32)
        t_inf = time.time() - t0

        spacing_zyx = np.array(image.GetSpacing()[::-1], float)
        mask, info = postprocess(probs, spacing_zyx, cfg)
        del probs
        _log(f"  inference {t_inf:.0f}s, "
             f"{json.dumps({k: v for k, v in info.items() if k != 'reconnect'})}")
        return mask

    rc = batch_io.run(predict_one, OUTPUT_SLUG, lambda m: _log(m), max_label=1)
    _log(f"total {time.time() - t_start:.0f}s")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
