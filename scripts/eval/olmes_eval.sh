#!/bin/bash
# Generic local-model OLMES adapter for Base/SFT/DPO/RLVR lineage checkpoints.
set -Eeuo pipefail

: "${OLMES_ROOT:?set OLMES_ROOT to pinned lecture-era OLMES checkout}"
: "${MODEL_DIR:?set MODEL_DIR to a local HF-format model/checkpoint}"

MODEL_LABEL="${MODEL_LABEL:-Qwen/Qwen2.5-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
PROFILE="${PROFILE:-smoke}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
OLMES_BIN="${OLMES_BIN:-$OLMES_ROOT/.venv/bin/olmes}"

[[ -x "$OLMES_BIN" ]] || { echo "OLMES_NOT_FOUND: $OLMES_BIN" >&2; exit 40; }
[[ -d "$MODEL_DIR" ]] || { echo "MODEL_DIR_NOT_FOUND: $MODEL_DIR" >&2; exit 40; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "MODEL_CONFIG_MISSING: $MODEL_DIR/config.json" >&2; exit 40; }
mkdir -p "$OUTPUT_DIR"

MODEL_ARGS="{\"model_path\":\"$MODEL_DIR\",\"max_length\":$MAX_LENGTH,\"trust_remote_code\":true}"
TASKS=(mmlu::olmes gsm8k::olmes)
EXTRA_ARGS=()

case "$PROFILE" in
  smoke)
    EXTRA_ARGS+=(--limit 32 --random-subsample-seed 42)
    ;;
  tutorial)
    ;;
  tulu3_full)
    TASKS=(tulu_3_dev tulu_3_unseen)
    ;;
  *)
    echo "UNKNOWN_EVAL_PROFILE=$PROFILE" >&2
    exit 40
    ;;
esac

"$OLMES_BIN" \
  --model "$MODEL_LABEL" \
  --model-type vllm \
  --model-args "$MODEL_ARGS" \
  --task "${TASKS[@]}" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
