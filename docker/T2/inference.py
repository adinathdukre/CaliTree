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
OUTPUT_SLUG = "multi-class-airway-segmentation"


def _log(msg: str) -> None:
    print(f"[atm26-t2] {msg}", flush=True)


MAX_PATCHES = 1600


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
            total = n * max(len(folds), 1)
            if total <= MAX_PATCHES or step >= 1.0:
                _log(f"resampled {res}, {n} tiles/fold x {len(folds)} folds "
                     f"= {total} patches at tile_step {step}")
                return step
    except Exception as exc:
        _log(f"tile-step planning failed ({exc}); using {base_step}")
    return base_step


def _load_seg_config():
    from airway.inference.predict import PostprocConfig
    from airway.inference.reconnect import ReconnectConfig

    seg_cfg = PostprocConfig()
    cfg_path = RESOURCES / "postproc_t2.json"
    if cfg_path.is_file():
        raw = json.loads(cfg_path.read_text())
        if "config" in raw and isinstance(raw["config"], dict):
            raw = raw["config"]
        rc = raw.pop("reconnect_cfg", None)
        applied = [k for k in raw if hasattr(seg_cfg, k)]
        for k in applied:
            setattr(seg_cfg, k, raw[k])
        if not applied:
            raise RuntimeError(
                f"{cfg_path.name} set no PostprocConfig field (keys: "
                f"{sorted(raw)}); refusing to run with silent defaults")
        if rc:
            seg_cfg.reconnect_cfg = ReconnectConfig(**rc)
        _log("loaded tuned Track-2 segmentation thresholds")
    return seg_cfg


def main() -> int:
    t_start = time.time()

    import torch
    from airway.inference import batch as batch_io
    from airway.inference.predict import build_predictor, postprocess
    from airway.graph.anatomy import MAX_CLASS_LABEL
    from airway.inference.lowmem_export import peak_rss_gib
    from airway.inference.track2_decode import TreeDecodeConfig
    from airway.inference.track2_warm import build_voxel_predictor, label_case

    seg_cfg = _load_seg_config()

    seg_dir = RESOURCES / "nnUNet_ckpts"
    folds = tuple(int(p.name.split("_")[-1]) for p in sorted(seg_dir.glob("fold_*"))
                  if p.is_dir())
    if not folds:
        raise RuntimeError(f"no fold_* under {seg_dir}")

    device = os.environ.get("ATM_DEVICE", "cuda")
    if os.environ.get("ATM_FOLDS"):
        folds = tuple(int(f) for f in os.environ["ATM_FOLDS"].split(","))
    tile_step = float(os.environ.get("ATM_TILE_STEP", "0.75"))
    if device != "cuda" or os.environ.get("ATM_FOLDS") or tile_step != 0.75:
        _log(f"overrides device={device} folds={folds} tile_step={tile_step}")

    t = time.time()
    predictor = build_predictor(str(seg_dir), folds,
                                checkpoint="checkpoint_best.pth",
                                tile_step_size=tile_step,
                                device=device, use_mirroring=False)

    warm_dir = RESOURCES / "nnUNet_t2warm"
    if not warm_dir.is_dir():
        raise RuntimeError(
            f"{warm_dir} missing: the 21-class labelling model is required")
    warm_folds = tuple(int(p.name.split("_")[-1])
                       for p in sorted(warm_dir.glob("fold_*")) if p.is_dir())
    if os.environ.get("ATM_WARM_FOLDS"):
        warm_folds = tuple(int(f) for f in os.environ["ATM_WARM_FOLDS"].split(","))
    else:
        warm_folds = warm_folds[:1]
    if len(warm_folds) > 1:
        _log(f"WARNING: {len(warm_folds)} voxel folds doubles the 21-channel "
             f"logit accumulator; 1 fold is what was measured and what fits")
    voxel_predictor = build_voxel_predictor(
        str(warm_dir), warm_folds, "checkpoint_final.pth", device)
    decode_cfg = TreeDecodeConfig()
    _log(f"models ready in {time.time() - t:.0f}s: seg folds={folds}, "
         f"voxel folds={warm_folds}, decode={decode_cfg}")

    def predict_one(image, path):
        predictor.tile_step_size = tile_step
        planned = plan_tile_step(image, predictor, folds, tile_step)
        if planned != tile_step:
            _log(f"  size guard: tile_step {tile_step} -> {planned}")
            predictor.tile_step_size = planned

        npy = sitk.GetArrayFromImage(image)[None].astype(np.float32)
        props = {"spacing": list(image.GetSpacing())[::-1]}
        t0 = time.time()
        _seg, probs = predictor.predict_single_npy_array(npy, props, None, None, True)
        del npy, _seg
        probs = np.asarray(probs, dtype=np.float32)
        t_seg = time.time() - t0

        spacing_zyx = np.array(image.GetSpacing()[::-1], float)
        mask, seg_info = postprocess(probs, spacing_zyx, seg_cfg)
        del probs
        _log(f"  seg {t_seg:.0f}s -> {seg_info['voxels_final']:,} voxels, "
             f"{seg_info['components_final']} component(s)")

        try:
            vcm = voxel_predictor.configuration_manager
            sp = np.array(image.GetSpacing()[::-1], float)
            sh = np.array(image.GetSize()[::-1], float)
            nvox = float(np.prod(sh * sp) / np.prod(np.array(vcm.spacing, float)))
            gib = 21 * nvox * 2 / 2 ** 30
            _log(f"  21-ch accumulator {gib:.1f} GiB on a {nvox/1e6:.0f}M-voxel "
                 f"model grid (32 GiB cap; streamed export adds ~1 channel)")
        except Exception:
            pass

        t1 = time.time()
        labels, info = label_case(image, mask, voxel_predictor, decode_cfg)
        del mask
        _log(f"  labelling {time.time() - t1:.0f}s: {json.dumps(info)}; "
             f"peak RSS {peak_rss_gib():.1f} GiB")
        return labels

    rc = batch_io.run(predict_one, OUTPUT_SLUG, lambda m: _log(m),
                      max_label=MAX_CLASS_LABEL)
    _log(f"total {time.time() - t_start:.0f}s, peak RSS {peak_rss_gib():.1f} GiB "
         f"(platform cap 32 GiB)")
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
