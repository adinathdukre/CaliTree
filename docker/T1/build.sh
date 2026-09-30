#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RES=${1:?usage: build.sh <weights_dir> [tag]}
TAG=${2:-atm26-track1}
REPO=../..

[ -d "$RES/nnUNet_ckpts" ] || { echo "no nnUNet_ckpts under $RES" >&2; exit 1; }

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/${TAG}_ctx.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
cp Dockerfile requirements.txt constraints.txt inference.py "$STAGE"/
cp -r "$REPO/airway" "$STAGE"/airway
mkdir -p "$STAGE"/nnunet_trainers
for f in airway_trainers.py; do cp "$REPO/nnunet_trainers/$f" "$STAGE"/nnunet_trainers/; done
cp -r "$RES" "$STAGE"/resources

docker build --platform=linux/amd64 -t "$TAG" "$STAGE"
echo "built $TAG"

if [ "${ATM_EXPORT:-0}" = 1 ]; then
    docker save "$TAG" | gzip -c > "${TAG}.tar.gz"
    echo "exported ${TAG}.tar.gz"
fi
