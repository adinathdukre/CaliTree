from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airway.eval.metrics import keep_largest_component
from airway.graph.anatomy import BACKGROUND, TRACHEA
from airway.graph.teasar import TeasarTree

N_CLASS = 21
EPS = 1e-6


def branch_log_posteriors(labels: np.ndarray, vol: np.ndarray, n: int,
                          prior: float = 0.5, dist: np.ndarray | None = None,
                          sigma: float = 0.0) -> np.ndarray:
    counts = np.full((n, N_CLASS), prior, np.float64)
    fg = vol > 0
    w = (np.exp(-dist[fg] / sigma).astype(np.float64)
         if (sigma > 0 and dist is not None) else 1.0)
    np.add.at(counts, (vol[fg].astype(np.int64) - 1,
                       labels[fg].astype(np.int64)), w)
    counts[:, BACKGROUND] *= 0.25
    return np.log(counts / counts.sum(1, keepdims=True) + EPS)


def vote_decode(tree, log_fine: np.ndarray, counts_fg: np.ndarray,
                min_evidence: float) -> tuple[np.ndarray, int]:
    lf = log_fine.copy()
    lf[:, BACKGROUND] = -np.inf
    labels = lf.argmax(1)

    weak = counts_fg < min_evidence
    order = np.argsort(tree.generation)
    n_inherited = 0
    for i in order:
        if not weak[i]:
            continue
        par = int(tree.parent[i])
        labels[i] = TRACHEA if par < 0 else labels[par]
        n_inherited += 1
    return labels, n_inherited


@dataclass
class TreeDecodeConfig:
    min_evidence: float = 0.40
    prior: float = 0.5
    sigma: float = 0.0
    fill: bool = False


def decode_case(mask: np.ndarray, voxel_labels: np.ndarray,
                spacing_zyx: np.ndarray,
                cfg: TreeDecodeConfig | None = None) -> tuple[np.ndarray, dict]:
    cfg = cfg or TreeDecodeConfig()
    mask = keep_largest_component(np.asarray(mask).astype(np.uint8)).astype(bool)
    if not mask.any():
        raise RuntimeError("empty airway mask")

    labels = np.asarray(voxel_labels).astype(np.uint8)
    tree = TeasarTree(mask, np.asarray(spacing_zyx, float))
    vol = tree.vol
    if tree.n < 3:
        raise RuntimeError(f"tree construction failed (n={tree.n})")

    log_fine = branch_log_posteriors(labels, vol, tree.n, cfg.prior,
                                     getattr(tree, "dist", None), cfg.sigma)

    tot = np.bincount(vol[vol > 0].astype(np.int64) - 1, minlength=tree.n)
    hit = np.bincount(vol[(vol > 0) & (labels > 0)].astype(np.int64) - 1,
                      minlength=tree.n)
    counts_fg = np.divide(hit, np.maximum(tot, 1), dtype=np.float64)

    branch_labels, n_inherited = vote_decode(tree, log_fine, counts_fg,
                                             cfg.min_evidence)

    out = np.zeros(mask.shape, np.uint8)
    fg = vol > 0
    lut = np.concatenate([[BACKGROUND], branch_labels]).astype(np.uint8)
    out[fg] = lut[vol[fg]]

    if cfg.fill:
        keep = fg & (labels > 0)
        out[keep] = labels[keep]

    info = {"n_branches": int(tree.n),
            "classes": int(len(np.unique(out)) - 1),
            "inherited": int(n_inherited),
            "weak_frac": round(float((counts_fg < cfg.min_evidence).mean()), 4)}
    return out, info
