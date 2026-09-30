from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize


R1_MM = 2.5
R2_MM = 1.0

HIST_EDGES = np.array([0, .25, .5, .75, 1., 1.5, 2., 2.5, 3., 4., 5., 7.5, 10., 20.])


def airway_bbox(mask: np.ndarray, margin: int = 4) -> tuple[slice, ...]:
    idx = np.nonzero(mask)
    return tuple(
        slice(max(int(i.min()) - margin, 0), min(int(i.max()) + margin + 1, s))
        for i, s in zip(idx, mask.shape)
    )


def band_label(seg: np.ndarray, spacing_zyx) -> tuple[np.ndarray, np.ndarray]:
    m = seg > 0
    if not m.any():
        seg[:] = 0
        return seg, np.zeros(len(HIST_EDGES) - 1, np.int64)

    sl = airway_bbox(m)
    sub = np.ascontiguousarray(m[sl])
    del m

    try:
        import edt as _edt
        r = _edt.edt(sub, anisotropy=tuple(float(v) for v in spacing_zyx),
                     black_border=False, parallel=1)
    except Exception:
        r = ndimage.distance_transform_edt(sub, sampling=spacing_zyx).astype(np.float32)

    skel = skeletonize(sub) > 0
    skel_coords = np.argwhere(skel).astype(np.int32)
    if skel_coords.shape[0] == 0:
        seg[:] = 0
        seg[sl] = np.where(sub, 3, 0).astype(np.uint8)
        return seg, np.zeros(len(HIST_EDGES) - 1, np.int64)

    r_at_skel = r[tuple(skel_coords.T)]
    hist, _ = np.histogram(r_at_skel, bins=HIST_EDGES)
    del r, skel

    fg_coords = np.argwhere(sub).astype(np.int32)
    sp = np.asarray(spacing_zyx, np.float64)

    from scipy.spatial import cKDTree
    tree_mm = cKDTree(skel_coords * sp)
    _, idx = tree_mm.query(fg_coords * sp, k=1, workers=1)
    calibre = r_at_skel[idx]

    band = np.full(fg_coords.shape[0], 2, np.uint8)
    band[calibre >= R1_MM] = 1
    band[calibre < R2_MM] = 3

    tree_vox = cKDTree(skel_coords.astype(np.float64))
    d_vox, _ = tree_vox.query(fg_coords.astype(np.float64), k=1, p=1, workers=1)
    band[d_vox <= 2.0] += 3

    seg[:] = 0
    out_sub = np.zeros(sub.shape, np.uint8)
    out_sub[tuple(fg_coords.T)] = band
    seg[sl] = out_sub
    return seg, hist


def read_header(path: Path) -> sitk.ImageFileReader:
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return r


def process_case(cid: str, img_dir: Path, lab_dir: Path, out_img: Path, out_lab: Path,
                 link_images: bool, skip_existing: bool = False) -> dict:
    if skip_existing and (out_lab / f"{cid}.mha").exists() and \
            (out_img / f"{cid}_0000.mha").exists():
        return {"case": cid, "skipped": True}
    img_src = img_dir / f"{cid}_0000.mha"
    lab_src = lab_dir / f"{cid}.mha"
    hdr = read_header(img_src)
    lab = sitk.ReadImage(str(lab_src))

    img_size = tuple(hdr.GetSize())
    if lab.GetSize() != img_size:
        return {"case": cid, "error": f"size mismatch img{img_size} lab{lab.GetSize()}"}
    img_spacing = tuple(hdr.GetSpacing())
    if not np.allclose(lab.GetSpacing(), img_spacing, rtol=1e-4, atol=1e-6):
        return {"case": cid,
                "error": f"spacing mismatch img{img_spacing} lab{lab.GetSpacing()}"}

    seg = (sitk.GetArrayFromImage(lab) > 0).astype(np.uint8)
    spacing_zyx = np.array(img_spacing[::-1], float)
    bands, hist = band_label(seg, spacing_zyx)
    assert bands.dtype == np.uint8, f"band map must be uint8, got {bands.dtype}"

    band_img = sitk.GetImageFromArray(bands)
    band_img.CopyInformation(lab)
    sitk.WriteImage(band_img, str(out_lab / f"{cid}.mha"), useCompression=True)

    dst_img = out_img / f"{cid}_0000.mha"
    if link_images:
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        try:
            os.link(img_src, dst_img)
        except OSError:
            shutil.copy2(img_src, dst_img)
    else:
        shutil.copy2(img_src, dst_img)

    counts = np.bincount(bands.reshape(-1), minlength=7)
    return {
        "case": cid,
        "spacing_xyz": [float(v) for v in img_spacing],
        "shape_zyx": [int(v) for v in seg.shape],
        "fg_voxels": int((seg > 0).sum()),
        "band_voxels": [int(c) for c in counts],
        "radius_hist": [int(c) for c in hist],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path,
                    help="Track-1 imagesTr folder with <case>_0000.mha")
    ap.add_argument("--labels", required=True, type=Path,
                    help="Track-1 labelsTr folder with <case>.mha")
    ap.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--dataset-name", default="ATM26Track1Bands")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--copy-images", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if args.nnunet_raw is None:
        raise SystemExit("pass --nnunet-raw or set nnUNet_raw")

    root = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    out_img, out_lab = root / "imagesTr", root / "labelsTr"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name[: -len("_0000.mha")] for p in args.images.glob("*_0000.mha"))
    cases = [c for c in cases if (args.labels / f"{c}.mha").exists()]
    print(f"preparing {len(cases)} cases -> {root}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_case, c, args.images, args.labels, out_img, out_lab,
                          not args.copy_images, args.skip_existing): c
                for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"case": futs[f], "error": repr(e)})
            if i % 25 == 0:
                print(f"  {i}/{len(cases)}", flush=True)

    errs = [r for r in results if "error" in r]
    for r in errs:
        print("  ERROR", r["case"], r["error"])

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "airway": [1, 2, 3, 4, 5, 6],
            "mid_distal": [2, 3, 5, 6],
            "distal": [3, 6],
            "centerline": [4, 5, 6],
        },
        "regions_class_order": [1, 2, 3, 4],
        "numTraining": len(results) - len(errs),
        "file_ending": ".mha",
        "band_thresholds_mm": {"R1": R1_MM, "R2": R2_MM},
    }
    with open(root / "dataset.json", "w") as fh:
        json.dump(dataset_json, fh, indent=2)
    print(f"wrote {root}/dataset.json")


if __name__ == "__main__":
    main()
