from __future__ import annotations

import numpy as np


def streamed_logits_to_segmentation(predicted_logits, predictor, properties):
    import torch
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    from nnunetv2.configuration import default_num_processes

    plans_manager = predictor.plans_manager
    configuration_manager = predictor.configuration_manager
    label_manager = predictor.label_manager

    if label_manager.has_regions:
        raise NotImplementedError(
            "streamed_logits_to_segmentation is argmax-only; this model is "
            "region-based and must go through nnU-Net's own export")
    if len(label_manager.foreground_labels) >= 255:
        raise NotImplementedError("more than 254 foreground labels needs uint16")

    old_threads = torch.get_num_threads()
    torch.set_num_threads(default_num_processes)
    try:
        spacing_transposed = [properties['spacing'][i] for i in plans_manager.transpose_forward]
        current_spacing = configuration_manager.spacing if \
            len(configuration_manager.spacing) == \
            len(properties['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *configuration_manager.spacing]
        new_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        target_shape = tuple(int(v) for v in
                             properties['shape_after_cropping_and_before_resampling'])
        n_ch = int(predicted_logits.shape[0])

        best_val = torch.empty(target_shape, dtype=torch.float32)
        best_idx = torch.zeros(target_shape, dtype=torch.uint8)

        for c in range(n_ch):
            chan = configuration_manager.resampling_fn_probabilities(
                predicted_logits[c:c + 1], target_shape, current_spacing, new_spacing)
            if isinstance(chan, np.ndarray):
                chan = torch.from_numpy(chan)
            chan = chan[0]

            if c == 0:
                best_val.copy_(chan)
            else:
                upd = chan > best_val
                best_idx.masked_fill_(upd, c)
                torch.maximum(best_val, chan, out=best_val)
                del upd
            del chan

        del best_val
        segmentation = best_idx.numpy()

        out = np.zeros(properties['shape_before_cropping'], dtype=np.uint8)
        out = insert_crop_into_image(out, segmentation, properties['bbox_used_for_cropping'])
        del segmentation, best_idx
        return out.transpose(plans_manager.transpose_backward)
    finally:
        torch.set_num_threads(old_threads)


def predict_npy_lowmem(predictor, npy, props):
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

    ppa = PreprocessAdapterFromNpy([npy], [None], [props], [None],
                                   predictor.plans_manager, predictor.dataset_json,
                                   predictor.configuration_manager,
                                   num_threads_in_multithreaded=1,
                                   verbose=predictor.verbose)
    dct = next(ppa)
    del ppa, npy
    data, data_properties = dct['data'], dct['data_properties']
    del dct

    logits = predictor.predict_logits_from_preprocessed_data(data).cpu()
    del data
    seg = streamed_logits_to_segmentation(logits, predictor, data_properties)
    del logits
    return seg


def peak_rss_gib() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return float("nan")
