from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DATASET = "Dataset502_ATM26Track2Seg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-plans", required=True, type=Path,
                    help="plans.json shipped with the ATM'26 Track-2 baseline weights")
    ap.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--preprocessed", type=Path,
                    default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--plans-name", default="nnUNetPlansOrg")
    a = ap.parse_args()
    if a.nnunet_raw is None or a.preprocessed is None:
        raise SystemExit("set nnUNet_raw and nnUNet_preprocessed (source scripts/env.sh)")

    out_dir = a.preprocessed / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = json.loads(a.baseline_plans.read_text())
    plans["dataset_name"] = DATASET
    plans["plans_name"] = a.plans_name

    dst = out_dir / f"{a.plans_name}.json"
    dst.write_text(json.dumps(plans, indent=4, sort_keys=False))
    shutil.copy(a.nnunet_raw / DATASET / "dataset.json", out_dir / "dataset.json")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
