from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

T2_LABELS = {
    "background": 0,
    "LB1+2": 1, "LB3": 2, "LB4": 3, "LB5": 4, "LB6": 5, "LB8": 6, "LB9": 7,
    "LB10": 8, "RB1": 9, "RB2": 10, "RB3": 11, "RB4": 12, "RB5": 13, "RB6": 14,
    "RB7": 15, "RB8": 16, "RB9": 17, "RB10": 18, "trachea": 19, "LB7": 20,
}


def process_case(cid: str, img_dir: Path, lab_dir: Path,
                 out_img: Path, out_lab: Path, link: bool) -> str:
    lab = sitk.ReadImage(str(lab_dir / f"{cid}.mha"))
    seg = sitk.GetArrayFromImage(lab)
    if seg.min() < 0 or seg.max() > 20:
        raise ValueError(f"{cid}: label values outside [0, 20]")
    out = sitk.GetImageFromArray(seg.astype(np.uint8))
    out.CopyInformation(lab)
    sitk.WriteImage(out, str(out_lab / f"{cid}.mha"), useCompression=True)

    src = img_dir / f"{cid}_0000.mha"
    dst = out_img / f"{cid}_0000.mha"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    return cid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path,
                    help="Track-1 imagesTr folder with <case>_0000.mha")
    ap.add_argument("--labels", required=True, type=Path,
                    help="Track-2 labelsTr folder with <case>.mha")
    ap.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--dataset-id", type=int, default=502)
    ap.add_argument("--dataset-name", default="ATM26Track2Seg")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args()
    if args.nnunet_raw is None:
        raise SystemExit("pass --nnunet-raw or set nnUNet_raw")

    root = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    out_img, out_lab = root / "imagesTr", root / "labelsTr"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.stem for p in args.labels.glob("*.mha"))
    missing = [c for c in cases if not (args.images / f"{c}_0000.mha").exists()]
    if missing:
        raise SystemExit(f"Track-2 cases without a CT image: {missing}")
    print(f"preparing {len(cases)} cases -> {root}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_case, c, args.images, args.labels, out_img, out_lab,
                          not args.copy_images) for c in cases]
        for f in as_completed(futs):
            f.result()
            done += 1

    with open(root / "dataset.json", "w") as fh:
        json.dump({
            "channel_names": {"0": "CT"},
            "labels": T2_LABELS,
            "numTraining": done,
            "file_ending": ".mha",
        }, fh, indent=2)
    print(f"wrote {root}/dataset.json")


if __name__ == "__main__":
    main()
