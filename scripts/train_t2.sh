#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

FOLD=${1:?usage: train_t2.sh <fold>}
shift
: "${ATM_T2_INIT:?set ATM_T2_INIT to the Track-2 baseline nnUNet_ckpts folder}"

PREP="$nnUNet_preprocessed/Dataset502_ATM26Track2Seg"
[ -f "$PREP/nnUNetPlansOrg.json" ] || { echo "missing $PREP/nnUNetPlansOrg.json (run scripts/prepare_t2_plans.py)" >&2; exit 1; }

nnUNetv2_train 502 3d_fullres_SAS "$FOLD" -tr nnUNetTrainerT2Warm -p nnUNetPlansOrg "$@"
