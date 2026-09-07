#!/bin/bash
# Evaluate an explicit local model directory with Lecture-6-era OLMES.
set -Eeuo pipefail

: "${OLMES_ROOT:?set OLMES_ROOT to pinned lecture-era checkout}"
: "${MODEL_DIR:?set MODEL_DIR to the exact Base/SFT/DPO/RLVR model directory}"

OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
PROFILE="${PROFILE:-smoke}"
OLMES_BIN="${OLMES_BIN:-$OLMES_ROOT/.venv/bin/olmes}"

[[ -x "$OLMES_BIN" ]] || { echo "OLMES_NOT_FOUND: $OLMES_BIN" >&2; exit 40; }
[[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] || {
  echo "MODEL_DIR_INVALID: $MODEL_DIR" >&2; exit 40;
}

ARGS=(
  --model "$MODEL_DIR"
  --model-type vllm
  --task mmlu::olmes gsm8k::olmes
  --output-dir "$OUTPUT_DIR"
  --gpus 1
)

case "$PROFILE" in
  smoke)
    # OLMES --limit is an authoritative CLI option. 16 instances/task keeps this
    # as an integration check while still exercising the requested Qwen model.
    ARGS+=(--limit 16 --random-subsample-seed 42)
    ;;
  tutorial_initial)
    # Full MMLU + GSM8K as used by the Lecture-6 example.
    ;;
  dry_run)
    ARGS+=(--limit 16 --random-subsample-seed 42 --dry-run)
    ;;
  *)
    echo "UNKNOWN_EVAL_PROFILE=$PROFILE" >&2
    exit 40
    ;;
esac

"$OLMES_BIN" "${ARGS[@]}"
