#!/usr/bin/env bash
export ATM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ATM_DATA="${ATM_DATA:-$ATM_ROOT/data}"

export nnUNet_raw="${nnUNet_raw:-$ATM_DATA/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$ATM_DATA/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$ATM_DATA/nnUNet_results}"
export nnUNet_extTrainer="$ATM_ROOT/nnunet_trainers"
export PYTHONPATH="$ATM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
