#!/usr/bin/env bash
# Train the dav3_4view_with_multiview_supervision branch. See README.md for what this branch is.
#
# Extra arguments are forwarded to train.py, so config entries can be
# overridden without editing stage.yaml, e.g.
#   ./scripts/train_dav3_4view_with_multiview_supervision.sh --opts dataset.local_data_root /data/preprocessed
set -e
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train.py --config ./config/dav3_4view_with_multiview_supervision "$@"
