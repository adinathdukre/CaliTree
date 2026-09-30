from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def teasar_params(spacing_zyx: np.ndarray) -> dict:
    m = float(np.max(spacing_zyx))
    return {"scale": 5 * m, "const": 3 * m, "pdrf_scale": 100000, "pdrf_exponent": 2,
            "soma_acceptance_threshold": 10000, "soma_detection_threshold": 10000,
            "soma_invalidation_const": 300, "soma_invalidation_scale": 2,
            "max_paths": 10000}


def teasar_branches(mask: np.ndarray, spacing_zyx: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, int, list[set[int]]]:
    import kimimaro

    skels = kimimaro.skeletonize(
        mask.astype(np.uint8), teasar_params=teasar_params(spacing_zyx),
        anisotropy=tuple(float(s) for s in spacing_zyx), dust_threshold=0,
        fix_branching=True, fix_borders=True, fill_holes=True,
        fix_avocados=False, progress=False, parallel=1, parallel_chunk_size=100)
    if not skels:
        raise RuntimeError("kimimaro returned no skeleton")
    skel = max(skels.values(), key=lambda s: len(s.vertices))

    vox = np.rint(skel.vertices / np.asarray(spacing_zyx, float)).astype(np.int64)
    for a in range(3):
        np.clip(vox[:, a], 0, mask.shape[a] - 1, out=vox[:, a])
    edges = np.asarray(skel.edges, np.int64)
    nv = vox.shape[0]

    deg = np.bincount(edges.ravel(), minlength=nv)
    is_bif = deg >= 3

    keep = ~is_bif[edges[:, 0]] & ~is_bif[edges[:, 1]]
    e = edges[keep]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    n_comp, comp = connected_components(g, directed=False)

    branch_of = np.zeros(nv, np.int64)
    order, nb = {}, 0
    for v in range(nv):
        if is_bif[v]:
            continue
        c = comp[v]
        if c not in order:
            nb += 1
            order[c] = nb
        branch_of[v] = order[c]

    adj: list[set[int]] = [set() for _ in range(nb + 1)]
    nbrs: list[list[int]] = [[] for _ in range(nv)]
    for a, b in edges:
        nbrs[a].append(b); nbrs[b].append(a)
    bif_ids = np.where(is_bif)[0]
    seen_bif = set()
    for v in bif_ids:
        if v in seen_bif:
            continue
        stack, cluster = [v], []
        seen_bif.add(v)
        while stack:
            u = stack.pop(); cluster.append(u)
            for w in nbrs[u]:
                if is_bif[w] and w not in seen_bif:
                    seen_bif.add(w); stack.append(w)
        touching = {branch_of[w] for u in cluster for w in nbrs[u]
                    if not is_bif[w] and branch_of[w] > 0}
        for x in touching:
            adj[x] |= (touching - {x})

    sel = branch_of > 0
    return vox[sel], branch_of[sel].astype(np.int32), nb, adj


class TeasarTree:

    def __init__(self, mask: np.ndarray, spacing_zyx: np.ndarray):
        sk, sk_ids, n, adj = teasar_branches(mask, spacing_zyx)
        from scipy.spatial import cKDTree
        self.vol = np.zeros(mask.shape, np.int32)
        self.dist = np.zeros(mask.shape, np.float32)
        fgv = np.argwhere(mask)
        if fgv.size and sk.size:
            d, idx = cKDTree(sk.astype(np.float64) * spacing_zyx).query(
                fgv.astype(np.float64) * spacing_zyx, k=1, workers=-1)
            self.vol[fgv[:, 0], fgv[:, 1], fgv[:, 2]] = sk_ids[idx]
            self.dist[fgv[:, 0], fgv[:, 1], fgv[:, 2]] = d.astype(np.float32)
        counts = np.bincount(self.vol[self.vol > 0].astype(np.int64), minlength=n + 1)
        root1 = int(counts[1:].argmax()) + 1
        self.n = n
        self.root = root1 - 1
        self.parent = np.full(n, -1, np.int64)
        self.generation = np.zeros(n, np.int64)
        seen = np.zeros(n + 1, bool); seen[root1] = True
        q = [root1]
        while q:
            v = q.pop(0)
            for w in sorted(adj[v]):
                if not seen[w]:
                    seen[w] = True
                    self.parent[w - 1] = v - 1
                    self.generation[w - 1] = self.generation[v - 1] + 1
                    q.append(w)
        for i in range(n):
            if i != self.root and self.parent[i] < 0:
                self.parent[i] = self.root
                self.generation[i] = 1
N_CLASS = 21
EPS = 1e-6
