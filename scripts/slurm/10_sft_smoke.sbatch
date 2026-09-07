#!/bin/bash
#SBATCH --job-name=tulu3-s0-sft
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OPEN_INSTRUCT_ROOT="${OPEN_INSTRUCT_ROOT:-/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6}"
MODEL_DIR="${MODEL_DIR:-/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B}"
: "${SFT_DATASET_FILE:?set SFT_DATASET_FILE to frozen local sft.parquet}"
DATASET_MANIFEST="${DATASET_MANIFEST:-$(dirname "$SFT_DATASET_FILE")/dataset_manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/python}"
RUN_BASE="${RUN_BASE:-$REPO_ROOT/runs}"
RUN_ROOT="$RUN_BASE/sft_smoke_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"/{audit,config,logs,results,scheduler,checkpoints}

exec > >(tee "$RUN_ROOT/logs/sft_smoke.out") 2> >(tee "$RUN_ROOT/logs/sft_smoke.err" >&2)

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

[[ -x "$PYTHON_BIN" ]] || { echo "POSTTRAIN_PYTHON_MISSING=$PYTHON_BIN" >&2; exit 40; }
[[ -f "$DATASET_MANIFEST" ]] || { echo "DATASET_MANIFEST_MISSING=$DATASET_MANIFEST" >&2; exit 40; }

"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_open_instruct_checkout.py" \
  "$OPEN_INSTRUCT_ROOT" \
  --expected-commit 8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377 \
  --output "$RUN_ROOT/audit/open_instruct_checkout.json"

export RUNTIME_PROBE_OUT="$RUN_ROOT/audit/runtime.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/runtime_probe.py"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_model_asset.py" \
  "$MODEL_DIR" --output "$RUN_ROOT/audit/model_manifest.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_dataset_asset.py" \
  "$SFT_DATASET_FILE" \
  --max-rows 32 \
  --binding-manifest "$DATASET_MANIFEST" \
  --binding-stage sft \
  --output "$RUN_ROOT/audit/sft_dataset.json"
cp "$DATASET_MANIFEST" "$RUN_ROOT/audit/dataset_manifest.json"
cp "$REPO_ROOT/configs/sft/smoke_5090.yaml" "$RUN_ROOT/config/resolved_config.yaml"
cp "$REPO_ROOT/manifests/upstream.yaml" "$RUN_ROOT/audit/upstream.yaml"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/audit/project_commit.txt"

printf 'SFT_SMOKE_START=%s\nDATASET_MANIFEST=%s\n' "$(date -Is)" "$DATASET_MANIFEST"
PROFILE=smoke \
OPEN_INSTRUCT_ROOT="$OPEN_INSTRUCT_ROOT" \
MODEL_DIR="$MODEL_DIR" \
DATASET_REF="$SFT_DATASET_FILE" \
OUTPUT_DIR="$RUN_ROOT/checkpoints/smoke" \
NUM_PROCESSES=1 \
WITH_TRACKING=0 \
bash "$REPO_ROOT/scripts/train/sft_lecture6.sh"

echo "STATUS=PASS_SFT_SMOKE"
echo "RUN_ROOT=$RUN_ROOT"
