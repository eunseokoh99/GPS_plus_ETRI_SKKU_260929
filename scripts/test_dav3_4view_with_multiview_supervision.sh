#!/usr/bin/env bash
# Render the dav3_4view_with_multiview_supervision branch's val set from a trained checkpoint.
#   CKPT=<path> ./scripts/test_dav3_4view_with_multiview_supervision.sh
set -e
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
CKPT=${CKPT:?set CKPT=<path to .pth>}
PHASE=${PHASE:-val}          # 'val' / 'train', or a custom '<seq>_process' set
SHOW_PATH=${SHOW_PATH:-}     # default: experiments/<exp_name>/test_show_<phase>

CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    --config ./config/dav3_4view_with_multiview_supervision \
    --ckpt "${CKPT}" \
    --phase "${PHASE}" \
    ${VIEW:+--view ${VIEW}} \
    ${SHOW_PATH:+--show_path "${SHOW_PATH}"}
