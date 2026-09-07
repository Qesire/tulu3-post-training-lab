#!/bin/bash
# Lecture-era Tulu3 DPO integration adapter for the project's SFT checkpoint.
# This launcher is smoke-only until Qwen2.5-3B formal DPO is pilot-frozen.
set -Eeuo pipefail

: "${OPEN_INSTRUCT_ROOT:?set OPEN_INSTRUCT_ROOT to pinned lecture-era checkout}"
: "${SFT_CHECKPOINT_DIR:?set SFT_CHECKPOINT_DIR to an explicit SFT output checkpoint}"
: "${DATASET_REF:?set DATASET_REF to frozen local DPO .parquet file}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5_3b_dpo_smoke}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/accelerate}"

[[ -x "$ACCELERATE_BIN" ]] || { echo "ACCELERATE_NOT_FOUND: $ACCELERATE_BIN" >&2; exit 40; }
[[ -d "$SFT_CHECKPOINT_DIR" ]] || { echo "SFT_CHECKPOINT_MISSING: $SFT_CHECKPOINT_DIR" >&2; exit 40; }
[[ -f "$SFT_CHECKPOINT_DIR/config.json" ]] || {
  echo "SFT_CHECKPOINT_INCOMPLETE: missing config.json under $SFT_CHECKPOINT_DIR" >&2; exit 40;
}
[[ -f "$DATASET_REF" && "$DATASET_REF" == *.parquet ]] || {
  echo "DPO_DATASET_REF_INVALID: expected local .parquet: $DATASET_REF" >&2; exit 40;
}

cd "$OPEN_INSTRUCT_ROOT"
"$ACCELERATE_BIN" launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes 1 \
  open_instruct/dpo_tune_cache.py \
  --do_not_randomize_output_dir \
  --model_name_or_path "$SFT_CHECKPOINT_DIR" \
  --tokenizer_name "$SFT_CHECKPOINT_DIR" \
  --use_flash_attn \
  --dataset_mixer_list "$DATASET_REF" 1.0 \
  --max_seq_length 1024 \
  --preprocessing_num_workers 4 \
  --max_train_samples 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-7 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --weight_decay 0.0 \
  --num_train_epochs 1 \
  --max_train_steps 2 \
  --dpo_loss_type dpo_norm \
  --dpo_beta 5 \
  --gradient_checkpointing \
  --no_concatenated_forward \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps 1 \
  --seed 8 \
  --no_push_to_hub \
  --no_try_launch_beaker_eval_jobs \
  --no_try_auto_save_to_beaker
