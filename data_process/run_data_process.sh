#!/usr/bin/env bash
# Build one of the two preprocessed datasets from the raw THumanMV capture.
#
#   ./run_data_process.sh --rect    {RAW_DATA_PATH} {PREPROCESSED_PATH}
#   ./run_data_process.sh --no-rect {RAW_DATA_PATH} {PREPROCESSED_WO_RECT_PATH}
#
# See data_process.md for the output layout and the camera -> view mapping.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ne 3 ]]; then
    echo "usage: $0 (--rect|--no-rect) {RAW_DATA_PATH} {PREPROCESSED_PATH}" >&2
    exit 1
fi
RECT="$1"; RAW="$2"; OUT="$3"
if [[ "$RECT" != "--rect" && "$RECT" != "--no-rect" ]]; then
    echo "first argument must be --rect or --no-rect (got '$RECT')" >&2
    exit 1
fi

TRAIN_SEQS=(s1a1 s1a2 s1a3 s2a1 s2a2 s2a3 s3a1 s3a2 s3a3)
VAL_SEQS=(s1a6 s2a4 s3a5)
JOBS="${JOBS:-16}"

for seq in "${TRAIN_SEQS[@]}"; do
    python step_0.py -i "$seq" -t train "$RECT" --data-root "$RAW" --processed-root "$OUT" -j "$JOBS"
    python step_1.py -i "$seq" -t train         --data-root "$RAW" --processed-root "$OUT" -j "$JOBS"
done

for seq in "${VAL_SEQS[@]}"; do
    python step_0.py -i "$seq" -t val "$RECT" --data-root "$RAW" --processed-root "$OUT" -j "$JOBS"
    python step_1.py -i "$seq" -t val         --data-root "$RAW" --processed-root "$OUT" -j "$JOBS"
done

echo "done -> $OUT   (expect 8100 train / 270 val scene folders)"
