from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from airway.inference.reconnect import ReconnectConfig, reconnect
from airway.eval.metrics import keep_largest_component, num_components

CH_AIRWAY, CH_MID_DISTAL, CH_DISTAL, CH_CENTERLINE = 0, 1, 2, 3


@dataclass
class PostprocConfig:

    thr_airway: float = 0.5
    thr_mid_distal: float = 0.5
    thr_distal: float = 0.5
    use_union: bool = True

    band_mode: str = "adaptive"
    thr_airway_thin: float = 0.25
    thin_seed: float = 0.5
    thin_dilate_vox: float = 2

    layout: str = "regions"

    reconnect: bool = True
    reconnect_cfg: ReconnectConfig = field(default_factory=ReconnectConfig)

    unbridged: str = "drop"
    keep_min_voxels: int = 200

    min_component_voxels: int = 0


def airway_probability(prob: np.ndarray, cfg: PostprocConfig) -> np.ndarray:
    if cfg.layout == "binary_softmax":
        if prob.shape[0] < 2:
            raise ValueError("binary_softmax layout needs 2 channels, got "
                             f"{prob.shape[0]}")
        return prob[1]
    return prob[CH_AIRWAY]


def fuse_bands(prob: np.ndarray, cfg: PostprocConfig) -> np.ndarray:
    if prob.ndim != 4:
        raise ValueError(f"expected (C, Z, Y, X) probabilities, got {prob.shape}")
    p_air = airway_probability(prob, cfg)
    mask = p_air >= cfg.thr_airway
    if cfg.layout != "regions":
        return mask

    if cfg.band_mode == "adaptive" and prob.shape[0] > CH_DISTAL:
        near = _thin_airway_prior(prob, cfg)
        if near is not None:
            mask |= near & (p_air >= cfg.thr_airway_thin)
        return mask

    if cfg.use_union and prob.shape[0] > CH_MID_DISTAL:
        mask |= prob[CH_MID_DISTAL] >= cfg.thr_mid_distal
    if cfg.use_union and prob.shape[0] > CH_DISTAL:
        mask |= prob[CH_DISTAL] >= cfg.thr_distal
    return mask


def _thin_airway_prior(prob: np.ndarray, cfg: PostprocConfig) -> np.ndarray | None:
    from scipy import ndimage

    seed = prob[CH_DISTAL] >= cfg.thin_seed
    if prob.shape[0] > CH_MID_DISTAL:
        seed = seed | (prob[CH_MID_DISTAL] >= cfg.thin_seed)
    if cfg.layout == "regions" and prob.shape[0] > CH_CENTERLINE:
        seed = seed | (prob[CH_CENTERLINE] >= cfg.thin_seed)
    if not seed.any():
        return None
    r = int(max(1, round(cfg.thin_dilate_vox)))
    return ndimage.binary_dilation(seed, iterations=r)


def tubularity_field(prob: np.ndarray, cfg: PostprocConfig) -> np.ndarray:
    p = airway_probability(prob, cfg)
    if cfg.layout == "regions" and prob.shape[0] > CH_CENTERLINE:
        p = np.maximum(p, prob[CH_CENTERLINE])
    return p


def postprocess(prob: np.ndarray, spacing_zyx, cfg: PostprocConfig | None = None
                ) -> tuple[np.ndarray, dict]:
    cfg = cfg or PostprocConfig()
    mask = fuse_bands(prob, cfg)
    info: dict = {"voxels_thresholded": int(mask.sum()),
                  "components_thresholded": num_components(mask)}

    if not mask.any():
        p = airway_probability(prob, cfg)
        k = min(64, p.size)
        flat = np.argpartition(p.reshape(-1), -k)[-k:]
        mask = np.zeros(p.shape, bool)
        mask.reshape(-1)[flat] = True
        info["empty_fallback"] = True
        info["empty_fallback_voxels"] = int(mask.sum())

    if cfg.min_component_voxels > 0:
        mask = _drop_small(mask, cfg.min_component_voxels)

    thresholded = mask.copy() if cfg.unbridged == "keep" else None

    if cfg.reconnect:
        field = tubularity_field(prob, cfg)
        rc = ReconnectConfig(**{**cfg.reconnect_cfg.__dict__})
        out, report = reconnect(field, spacing_zyx, rc, seed_mask=mask)
        info["reconnect"] = report.__dict__
        mask = out.astype(bool)
    elif cfg.unbridged == "drop":
        mask = keep_largest_component(mask).astype(bool)

    if cfg.unbridged == "keep":
        survivors = _drop_small(thresholded, cfg.keep_min_voxels)
        mask = (mask | survivors) if cfg.reconnect else survivors

    info["voxels_final"] = int(mask.sum())
    info["components_final"] = num_components(mask)
    return mask.astype(np.uint8), info


def _drop_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(lab.reshape(-1))
    counts[0] = 0
    keep = np.nonzero(counts >= min_voxels)[0]
    return np.isin(lab, keep)


def write_submission(mask: np.ndarray, reference: sitk.Image, out_path: Path) -> None:
    arr = np.asarray(mask)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    arr = (arr > 0).astype(np.uint8)
    if not arr.any():
        raise ValueError("refusing to write an empty segmentation")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr, isVector=False)
    img.CopyInformation(reference)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def validate_submission(path: Path, reference: sitk.Image) -> None:
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"{path}: must be 3D, got {img.GetDimension()}D")
    if img.GetPixelID() != sitk.sitkUInt8:
        raise ValueError(f"{path}: must be uint8, got {img.GetPixelIDTypeAsString()}")
    arr = sitk.GetArrayFromImage(img)
    vals = set(np.unique(arr).tolist())
    if not vals <= {0, 1}:
        raise ValueError(f"{path}: not binary, values={sorted(vals)}")
    if not arr.any():
        raise ValueError(f"{path}: empty segmentation")
    if img.GetSize() != reference.GetSize():
        raise ValueError(f"{path}: size {img.GetSize()} != {reference.GetSize()}")
    for attr in ("Spacing", "Origin", "Direction"):
        a = getattr(img, f"Get{attr}")()
        b = getattr(reference, f"Get{attr}")()
        if not np.allclose(a, b, rtol=1e-5, atol=1e-6):
            raise ValueError(f"{path}: {attr} {a} != {b}")


def build_predictor(model_folder: str, folds, checkpoint: str = "checkpoint_best.pth",
                    tile_step_size: float = 0.5, use_mirroring: bool = False,
                    device: str = "cuda", verbose: bool = False):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=(device == "cuda"),
        device=torch.device(device),
        verbose=verbose,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=folds, checkpoint_name=checkpoint)
    return predictor
