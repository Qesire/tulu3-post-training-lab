#!/bin/bash
#SBATCH --job-name=tulu3-e0-eval
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OLMES_ROOT="${OLMES_ROOT:-/home/scc/pb23061276/projects/tulu3-upstream/olmes-lecture6}"
: "${MODEL_DIR:?set MODEL_DIR to an explicit Base/SFT/DPO/RLVR model directory}"
MODEL_STAGE="${MODEL_STAGE:-unknown}"
PYTHON_BIN="${PYTHON_BIN:-$OLMES_ROOT/.venv/bin/python}"
RUN_BASE="${RUN_BASE:-$REPO_ROOT/runs}"
RUN_ROOT="$RUN_BASE/eval_${MODEL_STAGE}_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"/{audit,config,logs,results,scheduler}

exec > >(tee "$RUN_ROOT/logs/eval.out") 2> >(tee "$RUN_ROOT/logs/eval.err" >&2)

on_exit() {
  rc=$?
  python3 - "$RUN_ROOT" "$rc" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root, rc = Path(sys.argv[1]), int(sys.argv[2])
(root / "results" / "exit.json").write_text(json.dumps({
    "schema_version": 1,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_code": rc,
    "status": "PASS" if rc == 0 else "FAILED",
}, indent=2) + "\n")
PY
}
trap on_exit EXIT

[[ -x "$PYTHON_BIN" ]] || { echo "OLMES_PYTHON_MISSING=$PYTHON_BIN" >&2; exit 40; }
EXPECTED_COMMIT="f122a7bad99551ffb3f5feec14169af794926f6b"
ACTUAL_COMMIT="$(git -C "$OLMES_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || {
  echo "OLMES_COMMIT_MISMATCH expected=$EXPECTED_COMMIT actual=$ACTUAL_COMMIT" >&2; exit 40;
}

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/tulu3_olmes_triton_${USER}_${SLURM_JOB_ID}}"
mkdir -p "$TRITON_CACHE_DIR"
trap 'rc=$?; rm -rf "$TRITON_CACHE_DIR"; on_exit; exit $rc' EXIT

export RUNTIME_PROBE_OUT="$RUN_ROOT/audit/runtime.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/runtime_probe.py"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_model_asset.py" \
  "$MODEL_DIR" --output "$RUN_ROOT/audit/model_manifest.json"
cp "$REPO_ROOT/configs/eval/smoke.yaml" "$RUN_ROOT/config/resolved_config.yaml"
cp "$REPO_ROOT/manifests/upstream.yaml" "$RUN_ROOT/audit/upstream.yaml"
printf '%s\n' "$MODEL_STAGE" > "$RUN_ROOT/audit/model_stage.txt"

# First prove that OLMES can resolve the exact model/tasks into a runnable command.
OLMES_ROOT="$OLMES_ROOT" MODEL_DIR="$MODEL_DIR" OUTPUT_DIR="$RUN_ROOT/results/dry_run" PROFILE=dry_run \
  bash "$REPO_ROOT/scripts/eval/olmes_lecture6.sh"

# Then run a bounded 16-instance/task evaluation using the requested model.
printf 'EVAL_SMOKE_START=%s\nMODEL_STAGE=%s\nMODEL_DIR=%s\n' "$(date -Is)" "$MODEL_STAGE" "$MODEL_DIR"
OLMES_ROOT="$OLMES_ROOT" MODEL_DIR="$MODEL_DIR" OUTPUT_DIR="$RUN_ROOT/results/olmes" PROFILE=smoke \
  bash "$REPO_ROOT/scripts/eval/olmes_lecture6.sh"

echo "STATUS=PASS_EVAL_SMOKE"
echo "RUN_ROOT=$RUN_ROOT"
