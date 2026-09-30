from __future__ import annotations

import numpy as np

from airway.inference.lowmem_export import predict_npy_lowmem
from airway.inference.track2_decode import TreeDecodeConfig, decode_case

VOXEL_TILE_STEP = 0.5
VOXEL_MIRRORING = False


def build_voxel_predictor(model_dir: str, folds, checkpoint: str, device: str):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=VOXEL_TILE_STEP,
        use_gaussian=True,
        use_mirroring=VOXEL_MIRRORING,
        perform_everything_on_device=False,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir, use_folds=tuple(folds), checkpoint_name=checkpoint)
    return predictor


def label_case(image, mask: np.ndarray, voxel_predictor,
               cfg: TreeDecodeConfig | None = None) -> tuple[np.ndarray, dict]:
    import SimpleITK as sitk

    props = {"spacing": [abs(v) for v in image.GetSpacing()[::-1]]}
    voxel = np.asarray(predict_npy_lowmem(
        voxel_predictor,
        sitk.GetArrayFromImage(image)[None].astype(np.float32), props),
        dtype=np.uint8)

    spacing_zyx = np.abs(np.array(image.GetSpacing()[::-1], float))
    try:
        labels, info = decode_case(mask, voxel, spacing_zyx, cfg)
    except Exception as exc:
        labels = np.where(np.asarray(mask) > 0, voxel, 0).astype(np.uint8)
        if not labels.any():
            labels = voxel
        info = {"decode_failed": type(exc).__name__ + ": " + str(exc)[:200],
                "fallback": "voxel argmax masked by stage 1",
                "n_branches": 0, "classes": int(len(np.unique(labels)) - 1)}
    info["voxel_classes"] = int(len(np.unique(voxel)) - 1)
    del voxel
    return labels, info
