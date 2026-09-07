#!/bin/bash
#SBATCH --job-name=tulu3-e0-eval
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OLMES_ROOT="${OLMES_ROOT:-/home/scc/pb23061276/projects/tulu3-upstream/olmes-lecture6}"
: "${MODEL_DIR:?set MODEL_DIR to the Base/SFT/DPO/RLVR local model to evaluate}"
MODEL_LABEL="${MODEL_LABEL:-Qwen/Qwen2.5-3B}"
PYTHON_BIN="${PYTHON_BIN:-$OLMES_ROOT/.venv/bin/python}"
RUN_BASE="${RUN_BASE:-$REPO_ROOT/runs}"
RUN_ROOT="$RUN_BASE/eval_smoke_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"/{audit,config,logs,results,scheduler}

exec > >(tee "$RUN_ROOT/logs/eval_smoke.out") 2> >(tee "$RUN_ROOT/logs/eval_smoke.err" >&2)

[[ -x "$PYTHON_BIN" ]] || { echo "OLMES_PYTHON_MISSING=$PYTHON_BIN" >&2; exit 40; }
EXPECTED_COMMIT="f122a7bad99551ffb3f5feec14169af794926f6b"
ACTUAL_COMMIT="$(git -C "$OLMES_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || {
  echo "OLMES_COMMIT_MISMATCH expected=$EXPECTED_COMMIT actual=$ACTUAL_COMMIT" >&2; exit 40;
}

export RUNTIME_PROBE_OUT="$RUN_ROOT/audit/runtime.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/runtime_probe.py"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_model_asset.py" \
  "$MODEL_DIR" --output "$RUN_ROOT/audit/model_manifest.json"
cp "$REPO_ROOT/configs/eval/smoke.yaml" "$RUN_ROOT/config/resolved_config.yaml"
cp "$REPO_ROOT/manifests/upstream.yaml" "$RUN_ROOT/audit/upstream.yaml"

PROFILE=smoke \
OLMES_ROOT="$OLMES_ROOT" \
MODEL_DIR="$MODEL_DIR" \
MODEL_LABEL="$MODEL_LABEL" \
OUTPUT_DIR="$RUN_ROOT/results/olmes" \
bash "$REPO_ROOT/scripts/eval/olmes_eval.sh"

echo "STATUS=PASS_EVAL_SMOKE"
echo "RUN_ROOT=$RUN_ROOT"
