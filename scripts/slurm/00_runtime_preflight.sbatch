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
OPEN_INSTRUCT_ROOT="${OPEN_INSTRUCT_ROOT:-/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6}"
PYTHON_BIN="${PYTHON_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/python}"
RUN_BASE="${RUN_BASE:-$REPO_ROOT/runs}"
RUN_ROOT="$RUN_BASE/runtime_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"/{audit,logs,results,scheduler}

exec > >(tee "$RUN_ROOT/logs/runtime.out") 2> >(tee "$RUN_ROOT/logs/runtime.err" >&2)

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

printf 'JOB_ID=%s\nNODE=%s\nSTART=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"

[[ -x "$PYTHON_BIN" ]] || { echo "POSTTRAIN_PYTHON_MISSING=$PYTHON_BIN" >&2; exit 40; }
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_open_instruct_checkout.py" \
  "$OPEN_INSTRUCT_ROOT" \
  --expected-commit 8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377 \
  --output "$RUN_ROOT/audit/open_instruct_checkout.json"

export RUNTIME_PROBE_OUT="$RUN_ROOT/audit/runtime.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/runtime_probe.py"

MODEL_DIR="${MODEL_DIR:-/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B}"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_model_asset.py" \
  "$MODEL_DIR" --output "$RUN_ROOT/audit/model_manifest.json"

cp "$REPO_ROOT/manifests/upstream.yaml" "$RUN_ROOT/audit/upstream.yaml"
printf 'STATUS=PASS_RUNTIME_PREFLIGHT\nEND=%s\nRUN_ROOT=%s\n' "$(date -Is)" "$RUN_ROOT"
