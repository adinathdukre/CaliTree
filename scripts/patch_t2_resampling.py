from __future__ import annotations

import argparse
import json
from pathlib import Path

TORCH_FN = "resample_torch_fornnunet"
TORCH_KWARGS = {
    "is_seg": False,
    "force_separate_z": False,
    "memefficient_seg_resampling": False,
}


def patch(path: Path, dry: bool) -> dict:
    plans = json.loads(path.read_text())
    changed = []
    for name, cfg in plans.get("configurations", {}).items():
        if "resampling_fn_probabilities" not in cfg:
            continue
        if cfg["resampling_fn_probabilities"] == TORCH_FN:
            continue
        changed.append((name, cfg["resampling_fn_probabilities"]))
        cfg["resampling_fn_probabilities"] = TORCH_FN
        cfg["resampling_fn_probabilities_kwargs"] = dict(TORCH_KWARGS)
    if changed and not dry:
        path.write_text(json.dumps(plans, indent=4, sort_keys=False))
    return {"path": str(path), "changed": changed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="plans.json files to patch")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for p in a.paths:
        path = Path(p)
        if not path.is_file():
            print(f"  SKIP (missing) {path}")
            continue
        r = patch(path, a.dry_run)
        if r["changed"]:
            for name, old in r["changed"]:
                print(f"  {'would patch' if a.dry_run else 'patched'} "
                      f"{path.name}::{name}  {old} -> {TORCH_FN}")
        else:
            print(f"  already torch (or no probabilities fn): {path}")


if __name__ == "__main__":
    main()
