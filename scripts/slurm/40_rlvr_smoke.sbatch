#!/bin/bash
#SBATCH --job-name=tulu3-r0-rlvr
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OPEN_INSTRUCT_ROOT="${OPEN_INSTRUCT_ROOT:-/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6}"
: "${DPO_CHECKPOINT_DIR:?set DPO_CHECKPOINT_DIR to a concrete DPO checkpoint/output}"
: "${RLVR_DATASET_FILE:?set RLVR_DATASET_FILE to frozen local rlvr.parquet}"
DATASET_MANIFEST="${DATASET_MANIFEST:-$(dirname "$RLVR_DATASET_FILE")/dataset_manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/python}"
RUN_BASE="${RUN_BASE:-$REPO_ROOT/runs}"
RUN_ROOT="$RUN_BASE/rlvr_smoke_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"/{audit,config,logs,results,scheduler,checkpoints,tmp,dataset_cache}

exec > >(tee "$RUN_ROOT/logs/rlvr_smoke.out") 2> >(tee "$RUN_ROOT/logs/rlvr_smoke.err" >&2)

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

export TMPDIR="$RUN_ROOT/tmp"
export TRITON_CACHE_DIR="$RUN_ROOT/tmp/triton"
export RAY_TMPDIR="$RUN_ROOT/tmp/ray"
export RUNTIME_PROBE_OUT="$RUN_ROOT/audit/runtime.json"
mkdir -p "$TRITON_CACHE_DIR" "$RAY_TMPDIR"

"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/runtime_probe.py"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_model_asset.py" \
  "$DPO_CHECKPOINT_DIR" --output "$RUN_ROOT/audit/dpo_input_model.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_dataset_asset.py" \
  "$RLVR_DATASET_FILE" \
  --max-rows 8 \
  --require-columns messages ground_truth dataset constraint_type constraint \
  --binding-manifest "$DATASET_MANIFEST" \
  --binding-stage rlvr \
  --output "$RUN_ROOT/audit/rlvr_dataset.json"
cp "$DATASET_MANIFEST" "$RUN_ROOT/audit/dataset_manifest.json"
cp "$REPO_ROOT/configs/rlvr/smoke_5090.yaml" "$RUN_ROOT/config/resolved_config.yaml"
cp "$REPO_ROOT/configs/rlvr/reference_tulu3_8b_grpo.yaml" "$RUN_ROOT/config/reference_tulu3_8b_grpo.yaml"
cp "$REPO_ROOT/manifests/upstream.yaml" "$RUN_ROOT/audit/upstream.yaml"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/audit/project_commit.txt"

printf 'RLVR_SMOKE_START=%s\nDPO_CHECKPOINT_DIR=%s\nRLVR_DATASET_FILE=%s\nDATASET_MANIFEST=%s\n' \
  "$(date -Is)" "$DPO_CHECKPOINT_DIR" "$RLVR_DATASET_FILE" "$DATASET_MANIFEST"

OPEN_INSTRUCT_ROOT="$OPEN_INSTRUCT_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
DPO_CHECKPOINT_DIR="$DPO_CHECKPOINT_DIR" \
DATASET_REF="$RLVR_DATASET_FILE" \
DATASET_CACHE_DIR="$RUN_ROOT/dataset_cache" \
OUTPUT_DIR="$RUN_ROOT/checkpoints/smoke" \
bash "$REPO_ROOT/scripts/train/rlvr_lecture6.sh"

echo "STATUS=PASS_RLVR_SMOKE"
echo "RUN_ROOT=$RUN_ROOT"
