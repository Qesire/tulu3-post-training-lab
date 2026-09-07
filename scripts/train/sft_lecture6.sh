#!/bin/bash
# Lecture-6 Qwen2.5-3B SFT adapter for pinned Open-Instruct.
set -Eeuo pipefail

: "${OPEN_INSTRUCT_ROOT:?set OPEN_INSTRUCT_ROOT to pinned lecture-era checkout}"
: "${DATASET_REF:?set DATASET_REF to frozen local SFT .parquet file}"

MODEL_DIR="${MODEL_DIR:-/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5_3b_sft}"
PROFILE="${PROFILE:-reference}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
WITH_TRACKING="${WITH_TRACKING:-0}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/accelerate}"

[[ -x "$ACCELERATE_BIN" ]] || { echo "ACCELERATE_NOT_FOUND: $ACCELERATE_BIN" >&2; exit 40; }
[[ -d "$MODEL_DIR" ]] || { echo "MODEL_DIR_NOT_FOUND: $MODEL_DIR" >&2; exit 40; }
[[ -f "$DATASET_REF" && "$DATASET_REF" == *.parquet ]] || {
  echo "DATASET_REF_INVALID: expected local .parquet: $DATASET_REF" >&2; exit 40;
}

MAX_SEQ_LENGTH=4096
PREPROC_WORKERS=16
GRAD_ACCUM="${GRAD_ACCUM:-$((32 / NUM_PROCESSES))}"
EXTRA_ARGS=()

case "$PROFILE" in
  reference)
    ;;
  smoke)
    MAX_SEQ_LENGTH=1024
    PREPROC_WORKERS=4
    GRAD_ACCUM=1
    EXTRA_ARGS+=(--max_train_samples 32 --max_train_steps 2)
    ;;
  *)
    echo "UNKNOWN_PROFILE=$PROFILE" >&2
    exit 40
    ;;
esac

TRACKING_ARGS=()
if [[ "$WITH_TRACKING" == "1" ]]; then
  TRACKING_ARGS+=(--with_tracking --report_to wandb --wandb_project_name tulu3-qwen2.5-3b)
fi

cd "$OPEN_INSTRUCT_ROOT"
"$ACCELERATE_BIN" launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes "$NUM_PROCESSES" \
  open_instruct/finetune.py \
  --do_not_randomize_output_dir \
  --model_name_or_path "$MODEL_DIR" \
  --tokenizer_name "$MODEL_DIR" \
  --use_flash_attn \
  --dataset_mixer_list "$DATASET_REF" 1.0 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --preprocessing_num_workers "$PREPROC_WORKERS" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --num_train_epochs 2 \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps 10 \
  --checkpointing_steps epoch \
  --keep_last_n_checkpoints 3 \
  --gradient_checkpointing \
  --seed 42 \
  --no_push_to_hub \
  --no_try_launch_beaker_eval_jobs \
  --no_try_auto_save_to_beaker \
  "${TRACKING_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
