#!/bin/bash
#SBATCH --job-name=tulu3-p0-runtime
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
RUN_BASE="${RUN_BASE:-$REPO_ROOT/runs}"
RUN_ROOT="$RUN_BASE/runtime_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"/{audit,logs,results,scheduler}

exec > >(tee "$RUN_ROOT/logs/runtime.out") 2> >(tee "$RUN_ROOT/logs/runtime.err" >&2)

printf 'JOB_ID=%s\nNODE=%s\nSTART=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export RUNTIME_PROBE_OUT="$RUN_ROOT/audit/runtime.json"

"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/runtime_probe.py"

MODEL_DIR="${MODEL_DIR:-/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B}"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_model_asset.py" \
  "$MODEL_DIR" --output "$RUN_ROOT/audit/model_manifest.json"

cp "$REPO_ROOT/manifests/upstream.yaml" "$RUN_ROOT/audit/upstream.yaml"
printf 'STATUS=PASS_RUNTIME_PREFLIGHT\nEND=%s\nRUN_ROOT=%s\n' "$(date -Is)" "$RUN_ROOT"
