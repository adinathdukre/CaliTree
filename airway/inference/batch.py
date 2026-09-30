from __future__ import annotations

import gc
import os
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import SimpleITK as sitk

INPUT_DIR = Path(os.environ.get("ATM_INPUT", "/input"))
OUTPUT_DIR = Path(os.environ.get("ATM_OUTPUT", "/output"))
INPUT_SLUG = "lung-ct"

INPUT_SUFFIXES = (".mha", ".tif", ".tiff")


def find_input_images(input_dir: Path = INPUT_DIR) -> list[Path]:
    d = input_dir / "images" / INPUT_SLUG
    if not d.is_dir():
        d = input_dir
    if not d.is_dir():
        raise RuntimeError(f"input directory {d} does not exist")
    found = sorted(p for p in d.iterdir()
                   if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES)
    if not found:
        raise RuntimeError(
            f"no {'/'.join(INPUT_SUFFIXES)} inputs under {d}; "
            f"directory holds {[p.name for p in d.iterdir()][:10]}")
    return found


def case_id_of(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_0000")] if stem.endswith("_0000") else stem


def output_name(path: Path, batch: bool) -> str:
    return f"{case_id_of(path)}.mha" if batch else "output.mha"


def write_mask(array: np.ndarray, reference: sitk.Image, path: Path,
               max_label: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if max_label == 1:
        arr = (arr > 0).astype(np.uint8, copy=False)
    else:
        if arr.size and (arr.min() < 0 or arr.max() > max_label):
            raise ValueError(f"labels must lie in [0, {max_label}], got "
                             f"[{arr.min()}, {arr.max()}]")
        arr = arr.astype(np.uint8, copy=False)
    if not arr.any():
        raise RuntimeError("refusing to write an empty segmentation")
    img = sitk.GetImageFromArray(arr, isVector=False)
    img.CopyInformation(reference)
    sitk.WriteImage(img, str(path), useCompression=True)
    return path


def _release(*objs) -> None:
    for o in objs:
        del o
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def run(predict_one: Callable[[sitk.Image, Path], np.ndarray],
        output_slug: str,
        log: Callable[[str], None],
        max_label: int = 1,
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR) -> int:
    try:
        import torch
        ctx = torch.inference_mode()
    except Exception:
        import contextlib
        ctx = contextlib.nullcontext()

    inputs = find_input_images(input_dir)
    batch = len(inputs) > 1
    log(f"{len(inputs)} case(s) at {input_dir}, "
        f"{'BATCH' if batch else 'single-case'} mode")

    out_dir = output_dir / "images" / output_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    failures: list[str] = []
    with ctx:
        for i, path in enumerate(inputs, 1):
            t0 = time.time()
            cid = case_id_of(path)
            try:
                image = sitk.ReadImage(str(path))
                prediction = predict_one(image, path)
                written = write_mask(prediction, image,
                                     out_dir / output_name(path, batch),
                                     max_label=max_label)
                log(f"[{i}/{len(inputs)}] {cid} -> {written.name} "
                    f"({time.time() - t0:.0f}s)")
                _release(prediction, image)
            except Exception:
                failures.append(cid)
                log(f"[{i}/{len(inputs)}] {cid} FAILED after "
                    f"{time.time() - t0:.0f}s, continuing so the remaining "
                    f"cases still produce output")
                traceback.print_exc()
                _release()

    n_ok = len(inputs) - len(failures)
    log(f"{n_ok}/{len(inputs)} case(s) written in {time.time() - started:.0f}s")
    if failures:
        log(f"FAILED cases ({len(failures)}): {', '.join(failures[:20])}")
        return 0 if n_ok else 1
    return 0
