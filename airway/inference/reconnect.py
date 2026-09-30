from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass
class ReconnectConfig:
    tau_seed: float = 0.5
    tau_corridor: float = 0.05
    max_gap_mm: float = 12.0
    max_mean_cost: float = 2.5
    min_component_voxels: int = 15
    max_components: int = 400
    paint_radius_mm: float = 0.0
    eps: float = 1e-6


@dataclass
class ReconnectReport:
    n_components_before: int
    n_components_after: int
    n_candidates: int
    n_bridged: int
    n_dropped: int
    voxels_added: int
    bridged_voxels_recovered: int


def _component_sizes(labels: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(labels.reshape(-1), minlength=n + 1)


def reconnect(prob: np.ndarray, spacing_zyx, cfg: ReconnectConfig | None = None,
              seed_mask: np.ndarray | None = None
              ) -> tuple[np.ndarray, ReconnectReport]:
    cfg = cfg or ReconnectConfig()
    spacing = np.asarray(spacing_zyx, float)
    prob = np.asarray(prob, np.float32)

    mask = (prob >= cfg.tau_seed) if seed_mask is None else np.asarray(seed_mask, bool)
    mask = mask.astype(bool)
    if not mask.any():
        return mask.astype(np.uint8), ReconnectReport(0, 0, 0, 0, 0, 0, 0)

    labels, n = ndimage.label(mask, structure=CONNECTIVITY_26)
    n_before = int(n)
    if n <= 1:
        return mask.astype(np.uint8), ReconnectReport(n_before, n_before, 0, 0, 0, 0, 0)

    sizes = _component_sizes(labels, n)
    sizes[0] = 0
    main_id = int(sizes.argmax())

    candidates = [i for i in range(1, n + 1)
                  if i != main_id and sizes[i] >= cfg.min_component_voxels]
    candidates.sort(key=lambda i: -sizes[i])
    n_dropped = (n - 1) - len(candidates)
    candidates = candidates[: cfg.max_components]

    if not candidates:
        out = (labels == main_id)
        return out.astype(np.uint8), ReconnectReport(
            n_before, 1, 0, 0, n_dropped, 0, 0)

    bridge = np.zeros_like(mask, dtype=bool)
    radius = ndimage.distance_transform_edt(mask, sampling=spacing).astype(np.float32)
    r_typical = float(np.median(radius[mask])) if mask.any() else float(spacing.min())

    n_bridged = 0
    recovered = 0
    pad_vox = np.ceil((cfg.max_gap_mm + 4.0) / spacing).astype(int)
    for cid in candidates:
        res = _bridge_one(prob, mask, labels, cid, main_id, spacing, pad_vox,
                          radius, r_typical, cfg)
        if res is None:
            continue
        path_global, r = res
        _paint_path(bridge, path_global, r, spacing)
        n_bridged += 1
        recovered += int(sizes[cid])

    out = mask | bridge
    out_labels, out_n = ndimage.label(out, structure=CONNECTIVITY_26)
    if out_n > 1:
        out_sizes = _component_sizes(out_labels, out_n)
        out_sizes[0] = 0
        out = out_labels == int(out_sizes.argmax())

    added = int(out.sum()) - int((labels == main_id).sum())
    return out.astype(np.uint8), ReconnectReport(
        n_components_before=n_before,
        n_components_after=int(ndimage.label(out, structure=CONNECTIVITY_26)[1]),
        n_candidates=len(candidates),
        n_bridged=n_bridged,
        n_dropped=n_dropped + (len(candidates) - n_bridged),
        voxels_added=added,
        bridged_voxels_recovered=recovered,
    )


def _bridge_one(prob, mask, labels, cid, main_id, spacing, pad_vox,
                radius, r_typical, cfg):
    idx = np.argwhere(labels == cid)
    lo = np.maximum(idx.min(0) - pad_vox, 0)
    hi = np.minimum(idx.max(0) + pad_vox + 1, np.array(mask.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    sub_labels = labels[sl]
    sub_main = sub_labels == main_id
    if not sub_main.any():
        return None

    sub_prob = prob[sl]
    sub_mask = mask[sl]
    sub_comp = sub_labels == cid

    costs = -np.log(np.maximum(sub_prob, cfg.eps)).astype(np.float64)
    costs += 0.5
    costs[sub_prob < cfg.tau_corridor] = np.inf
    costs[sub_mask] = 0.01
    if not np.isfinite(costs[sub_comp]).any():
        return None

    mcp = MCP_Geometric(costs, sampling=tuple(float(s) for s in spacing),
                        fully_connected=True)
    starts = [tuple(p) for p in np.argwhere(sub_main)]
    cum, _ = mcp.find_costs(starts=starts)

    c_costs = np.where(sub_comp, cum, np.inf)
    best = float(np.min(c_costs))
    if not np.isfinite(best):
        return None
    target = np.unravel_index(int(np.argmin(c_costs)), c_costs.shape)
    try:
        path = np.array(mcp.traceback(target))
    except ValueError:
        return None
    if len(path) < 2:
        return None

    outside = ~sub_mask[tuple(path.T)]
    seg_len = np.linalg.norm(np.diff(path.astype(float) * spacing, axis=0), axis=1)
    step_outside = outside[1:] | outside[:-1]
    gap_mm = float(seg_len[step_outside].sum())
    if gap_mm > cfg.max_gap_mm:
        return None
    if best / max(gap_mm, 1e-3) > cfg.max_mean_cost:
        return None

    r_end = float(radius[sl][target])
    r = cfg.paint_radius_mm or max(min(r_end if r_end > 0 else r_typical, r_typical),
                                   float(spacing.min()))
    return path + lo, r


def _paint_path(out: np.ndarray, path: np.ndarray, radius_mm: float,
                spacing: np.ndarray) -> None:
    rad_vox = np.maximum(np.ceil(radius_mm / spacing).astype(int), 0)
    shape = out.shape
    rz, ry, rx = (int(v) for v in rad_vox)
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    d2 = ((zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2)
    stamp = d2 <= radius_mm ** 2
    offs = np.argwhere(stamp) - np.array([rz, ry, rx])

    pts = (path[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    ok = np.all((pts >= 0) & (pts < np.array(shape)), axis=1)
    pts = pts[ok]
    out[pts[:, 0], pts[:, 1], pts[:, 2]] = True
