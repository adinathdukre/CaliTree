#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

DATASET_ID=${DATASET_ID:-501}
NPROC=${NPROC:-8}

nnUNetv2_extract_fingerprint -d "$DATASET_ID" -np "$NPROC" --verify_dataset_integrity
nnUNetv2_plan_experiment -d "$DATASET_ID" -pl nnUNetPlannerResEncM
nnUNetv2_preprocess -d "$DATASET_ID" -plans_name nnUNetResEncUNetMPlans -c 3d_fullres -np "$NPROC"
