from __future__ import annotations

import numpy as np

BACKGROUND = 0
TRACHEA = 19
MAX_CLASS_LABEL = 20

CLASS_NAMES = {
    0: "background",
    1: "LB1+2", 2: "LB3", 3: "LB4", 4: "LB5", 5: "LB6", 6: "LB8", 7: "LB9",
    8: "LB10", 9: "RB1", 10: "RB2", 11: "RB3", 12: "RB4", 13: "RB5",
    14: "RB6", 15: "RB7", 16: "RB8", 17: "RB9", 18: "RB10",
    19: "trachea", 20: "LB7",
}
NAME_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

SEG_GROUP: dict[int, tuple[str, str]] = {
    1: ("L", "LUL"), 2: ("L", "LUL"), 3: ("L", "LUL"), 4: ("L", "LUL"),
    5: ("L", "LLL"), 6: ("L", "LLL"), 7: ("L", "LLL"), 8: ("L", "LLL"),
    20: ("L", "LLL"),
    9: ("R", "RUL"), 10: ("R", "RUL"), 11: ("R", "RUL"),
    12: ("R", "RML"), 13: ("R", "RML"),
    14: ("R", "RLL"), 15: ("R", "RLL"), 16: ("R", "RLL"), 17: ("R", "RLL"),
    18: ("R", "RLL"),
}

LOBES = ["LUL", "LLL", "RUL", "RML", "RLL"]
LOBE_SIDE = {"LUL": "L", "LLL": "L", "RUL": "R", "RML": "R", "RLL": "R"}
LOBE_SEGMENTS: dict[str, list[int]] = {
    lobe: sorted(c for c, (_, lb) in SEG_GROUP.items() if lb == lobe)
    for lobe in LOBES
}
SIDE_SEGMENTS: dict[str, list[int]] = {
    side: sorted(c for c, (s, _) in SEG_GROUP.items() if s == side)
    for side in ("L", "R")
}


def hierarchy_path(c: int) -> tuple | None:
    if c == TRACHEA:
        return ("trachea",)
    if c in SEG_GROUP:
        side, lobe = SEG_GROUP[c]
        return ("trachea", side, lobe, c)
    return None


def _hier_distance(a: int, b: int) -> float | None:
    if a == b:
        return 0.0
    pa, pb = hierarchy_path(a), hierarchy_path(b)
    if pa is None or pb is None:
        return None
    common = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        common += 1
    return float((len(pa) - common) + (len(pb) - common))


def build_class_distance_matrix(max_class: int = MAX_CLASS_LABEL
                                ) -> tuple[np.ndarray, float]:
    n = max_class + 1
    hier = [c for c in range(n) if hierarchy_path(c) is not None]
    d_max = max((_hier_distance(a, b) for a in hier for b in hier), default=1.0)
    D = np.full((n, n), float(d_max), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            d = _hier_distance(a, b)
            if d is not None:
                D[a, b] = d
    return D, float(d_max)


def lobe_of(c: int) -> str | None:
    return SEG_GROUP[c][1] if c in SEG_GROUP else None


def side_of(c: int) -> str | None:
    return SEG_GROUP[c][0] if c in SEG_GROUP else None


def subtree_consistency(pred: np.ndarray, mask_top: np.ndarray,
                        trachea_label: int = TRACHEA) -> float:
    n = pred.shape[0]
    cal = top = 0
    for i in range(n):
        if pred[i] == trachea_label:
            continue
        members = pred[mask_top[i] == 1]
        if members.shape[0] <= 1:
            continue
        cal += 1
        if np.all(members == pred[i]):
            top += 1
    return float(top) / cal if cal else 0.0


def topological_distance(spd: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                         valid_mask: np.ndarray, td_fill: float = 50.0) -> float:
    idx = np.where(valid_mask)[0]
    if idx.size == 0:
        return 0.0
    d = np.full(idx.shape[0], td_fill, dtype=np.float64)
    for k, i in enumerate(idx):
        targets = np.where(gt == pred[i])[0]
        if targets.size:
            dmin = float(np.min(spd[i, targets]))
            if np.isfinite(dmin):
                d[k] = dmin
    return float(np.mean(d))


def tree_accuracy(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray,
                  dist_matrix: np.ndarray | None = None,
                  d_max: float | None = None) -> float:
    if dist_matrix is None or d_max is None:
        dist_matrix, d_max = build_class_distance_matrix()
    g, p = gt[valid_mask], pred[valid_mask]
    if g.size == 0:
        return 0.0
    dvals = dist_matrix[p, g]
    return float(np.mean(1.0 - dvals / d_max))
