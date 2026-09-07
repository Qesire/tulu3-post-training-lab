#!/bin/bash
# Lecture-era Tulu3 RLVR/GRPO integration adapter for this project's DPO checkpoint.
# This launcher is smoke-only until the Qwen2.5-3B resource/method pilot is frozen.
set -Eeuo pipefail

: "${OPEN_INSTRUCT_ROOT:?set OPEN_INSTRUCT_ROOT to pinned lecture-era checkout}"
: "${DPO_CHECKPOINT_DIR:?set DPO_CHECKPOINT_DIR to an explicit DPO output checkpoint}"
: "${DATASET_REF:?set DATASET_REF to frozen local RLVR .parquet file}"

PYTHON_BIN="${PYTHON_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5_3b_rlvr_smoke}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${TMPDIR:-/tmp}/tulu3_rlvr_cache_${USER:-user}_${SLURM_JOB_ID:-local}}"

[[ -x "$PYTHON_BIN" ]] || { echo "POSTTRAIN_PYTHON_MISSING: $PYTHON_BIN" >&2; exit 40; }
[[ -d "$DPO_CHECKPOINT_DIR" ]] || { echo "DPO_CHECKPOINT_MISSING: $DPO_CHECKPOINT_DIR" >&2; exit 40; }
[[ -f "$DPO_CHECKPOINT_DIR/config.json" ]] || {
  echo "DPO_CHECKPOINT_INCOMPLETE: missing config.json under $DPO_CHECKPOINT_DIR" >&2; exit 40;
}
[[ -f "$DATASET_REF" && "$DATASET_REF" == *.parquet ]] || {
  echo "RLVR_DATASET_REF_INVALID: expected local .parquet: $DATASET_REF" >&2; exit 40;
}
[[ -f "$OPEN_INSTRUCT_ROOT/open_instruct/grpo_fast.py" ]] || {
  echo "GRPO_FAST_ENTRYPOINT_MISSING: $OPEN_INSTRUCT_ROOT/open_instruct/grpo_fast.py" >&2; exit 40;
}

mkdir -p "$OUTPUT_DIR" "$DATASET_CACHE_DIR"
cd "$OPEN_INSTRUCT_ROOT"

# Required by the pinned single-GPU debug path. The smoke deliberately uses
# synchronous rollout and CPU offload to reduce the memory risk on a 32-GB 5090.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

"$PYTHON_BIN" open_instruct/grpo_fast.py \
  --exp_name qwen2.5_3b_rlvr_5090_smoke \
  --model_name_or_path "$DPO_CHECKPOINT_DIR" \
  --tokenizer_name_or_path "$DPO_CHECKPOINT_DIR" \
  --attn_implementation flash_attention_2 \
  --dataset_mixer_list "$DATASET_REF" 8 \
  --dataset_mixer_list_splits train \
  --dataset_mixer_eval_list "$DATASET_REF" 4 \
  --dataset_mixer_eval_list_splits train \
  --dataset_cache_mode local \
  --dataset_local_cache_dir "$DATASET_CACHE_DIR" \
  --max_prompt_token_length 256 \
  --response_length 256 \
  --pack_length 512 \
  --per_device_train_batch_size 1 \
  --num_unique_prompts_rollout 2 \
  --num_samples_per_prompt_rollout 2 \
  --total_episodes 8 \
  --num_epochs 1 \
  --num_mini_batches 1 \
  --learning_rate 5e-7 \
  --lr_scheduler_type constant \
  --beta 0.01 \
  --kl_estimator kl3 \
  --temperature 1.0 \
  --apply_verifiable_reward true \
  --non_stop_penalty true \
  --non_stop_penalty_value 0.0 \
  --deepspeed_stage 2 \
  --deepspeed_offload_param true \
  --deepspeed_offload_optimizer true \
  --num_learners_per_node 1 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_sync_backend gloo \
  --vllm_gpu_memory_utilization 0.25 \
  --vllm_enforce_eager \
  --single_gpu_mode \
  --async_steps 0 \
  --local_eval_every -1 \
  --gather_whole_model false \
  --gradient_checkpointing \
  --seed 1 \
  --save_freq 1000000 \
  --output_dir "$OUTPUT_DIR" \
  --push_to_hub false
