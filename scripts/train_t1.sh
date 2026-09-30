#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

FOLD=${1:?usage: train_t1.sh <fold> [trainer]}
TRAINER=${2:-nnUNetTrainerAirwayCB_clprec025}
shift $(( $# < 2 ? $# : 2 ))

nnUNetv2_train 501 3d_fullres "$FOLD" -tr "$TRAINER" -p nnUNetResEncUNetMPlans "$@"
