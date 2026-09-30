from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)

MIN_BRANCH_VOXELS = 5
BD_THRESHOLD = 0.8
SMOOTH = 1e-5


def skeletonize_3d(mask: np.ndarray) -> np.ndarray:
    return (skeletonize(mask > 0) > 0).astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    components, count = ndimage.label(mask, structure=CONNECTIVITY_26)
    if count <= 1:
        return mask
    counts = np.bincount(components.ravel())
    counts[0] = 0
    return (components == counts.argmax()).astype(np.uint8)


def num_components(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask > 0, structure=CONNECTIVITY_26)
    return int(count)


def betti0_error(prediction: np.ndarray) -> int:
    return abs(num_components(prediction) - 1)


def dice_coefficient(prediction: np.ndarray, label: np.ndarray) -> float:
    prediction = prediction > 0
    label = label > 0
    inter = np.count_nonzero(prediction & label)
    return float((2.0 * inter + SMOOTH) /
                 (np.count_nonzero(prediction) + np.count_nonzero(label) + SMOOTH))


def tree_length_detected(prediction: np.ndarray, gt_skeleton: np.ndarray) -> float:
    prediction = prediction > 0
    skel = gt_skeleton > 0
    return float((np.count_nonzero(prediction & skel) + SMOOTH) /
                 (np.count_nonzero(skel) + SMOOTH))


def branch_detected(prediction: np.ndarray, gt_parsing: np.ndarray,
                    gt_skeleton: np.ndarray,
                    threshold: float = BD_THRESHOLD) -> tuple[int, int, float]:
    skeleton_mask = gt_skeleton > 0
    branch_ids = gt_parsing[skeleton_mask].astype(np.int64, copy=False)
    gt_counts = np.bincount(branch_ids)[1:]
    total = int(gt_counts.shape[0])
    covered = branch_ids[(prediction > 0)[skeleton_mask]]
    pred_counts = np.bincount(covered)[1:]
    if pred_counts.shape[0] < total:
        pred_counts = np.pad(pred_counts, (0, total - pred_counts.shape[0]))
    coverage = pred_counts / np.maximum(gt_counts, 1)
    detected = int(np.count_nonzero(coverage >= threshold))
    return total, detected, float(detected / total if total else 0.0)


def _centerline_overlap(volume: np.ndarray, skeleton: np.ndarray) -> float:
    volume = volume > 0
    skeleton = skeleton > 0
    n = np.count_nonzero(skeleton)
    if n == 0:
        return 0.0
    return float(np.count_nonzero(volume & skeleton) / n)


def cl_dice(prediction: np.ndarray, label: np.ndarray,
            gt_skeleton: np.ndarray) -> float:
    pred_skel = skeletonize_3d(prediction)
    t_sens = _centerline_overlap(prediction, gt_skeleton)
    t_prec = _centerline_overlap(label, pred_skel)
    denom = t_prec + t_sens
    if denom == 0:
        return 0.0
    return float(2.0 * t_prec * t_sens / denom)


def parse_skeleton_branches(skeleton: np.ndarray) -> tuple[np.ndarray, int]:
    skel = (skeleton > 0).astype(np.uint8)
    filtered = ndimage.convolve(skel, CONNECTIVITY_26, mode="constant") * skel
    parsed = skel.copy()
    parsed[filtered > 3] = 0

    cd, num = ndimage.label(parsed, structure=CONNECTIVITY_26)
    if num == 0:
        return cd.astype(np.int32), 0
    counts = np.bincount(cd.ravel(), minlength=num + 1)
    small = np.nonzero(counts[1:] < MIN_BRANCH_VOXELS)[0] + 1
    if small.size:
        parsed[np.isin(cd, small)] = 0
        cd, num = ndimage.label(parsed, structure=CONNECTIVITY_26)
    return cd.astype(np.int32), int(num)


def tree_parsing(branch_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    out = np.zeros(mask.shape, np.int32)
    src = np.argwhere(branch_ids > 0)
    if src.size == 0:
        return out
    fg = np.argwhere(mask > 0)
    if fg.size == 0:
        return out
    ids = branch_ids[src[:, 0], src[:, 1], src[:, 2]]
    _, idx = cKDTree(src.astype(np.float64)).query(fg.astype(np.float64), k=1,
                                                   workers=-1)
    out[fg[:, 0], fg[:, 1], fg[:, 2]] = ids[idx]
    return out


def build_reference_artifacts(label: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    full = keep_largest_component(label)
    if not full.any():
        z = np.zeros(label.shape, np.int32)
        return z.astype(np.uint8), z, 0
    idx = np.nonzero(full)
    margin = 4
    sl = tuple(slice(max(int(i.min()) - margin, 0), min(int(i.max()) + margin + 1, s))
               for i, s in zip(idx, full.shape))
    lab = ndimage.binary_fill_holes(np.ascontiguousarray(full[sl])).astype(np.uint8)

    skel_c = skeletonize_3d(lab)
    branch_ids, n = parse_skeleton_branches(skel_c)
    skel = np.zeros(label.shape, np.uint8)
    skel[sl] = skel_c
    if n == 0:
        return skel, np.zeros(label.shape, np.int32), 0
    parsing_c = tree_parsing(branch_ids, lab)
    parsing = np.zeros(label.shape, np.int32)
    parsing[sl] = parsing_c

    present = np.unique(parsing[parsing > 0])
    remap = np.zeros(int(parsing.max()) + 1, np.int32)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    parsing = remap[parsing]
    assert not np.any((skel > 0) & (parsing <= 0)), "skeleton voxel without branch id"
    return skel, parsing, int(len(present))


@dataclass
class Track1Metrics:
    DSC: float
    clDice: float
    TLD: float
    BD: float
    Betti0Error: int
    branch_detected: int
    branch_total: int
    n_components: int
    precision: float
    recall: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_track1(prediction: np.ndarray, label: np.ndarray,
                    gt_skeleton: np.ndarray | None = None,
                    gt_parsing: np.ndarray | None = None) -> Track1Metrics:
    prediction_raw = (np.asarray(prediction) > 0).astype(np.uint8)
    label = (np.asarray(label) > 0).astype(np.uint8)
    if prediction_raw.shape != label.shape:
        raise ValueError(f"shape mismatch {prediction_raw.shape} vs {label.shape}")

    if gt_skeleton is None or gt_parsing is None:
        gt_skeleton, gt_parsing, _ = build_reference_artifacts(label)
    gt_skeleton = (gt_skeleton > 0).astype(np.uint8)

    prediction_lcc = keep_largest_component(prediction_raw)
    total, detected, bd = branch_detected(prediction_lcc, gt_parsing, gt_skeleton)

    tp = float(np.count_nonzero(prediction_raw & label))
    n_pred = float(np.count_nonzero(prediction_raw))
    n_gt = float(np.count_nonzero(label))
    return Track1Metrics(
        DSC=dice_coefficient(prediction_raw, label),
        clDice=cl_dice(prediction_raw, label, gt_skeleton),
        TLD=tree_length_detected(prediction_lcc, gt_skeleton),
        BD=bd,
        Betti0Error=betti0_error(prediction_raw),
        branch_detected=detected,
        branch_total=total,
        n_components=num_components(prediction_raw),
        precision=tp / n_pred if n_pred else 0.0,
        recall=tp / n_gt if n_gt else 0.0,
    )
