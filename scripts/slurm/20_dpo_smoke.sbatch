#!/bin/bash
#SBATCH --job-name=tulu3-d0-dpo
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OPEN_INSTRUCT_ROOT="${OPEN_INSTRUCT_ROOT:-/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6}"
: "${SFT_CHECKPOINT_DIR:?set SFT_CHECKPOINT_DIR to a concrete SFT checkpoint/output}"
: "${DPO_DATASET_FILE:?set DPO_DATASET_FILE to frozen local dpo.parquet}"
DATASET_MANIFEST="${DATASET_MANIFEST:-$(dirname "$DPO_DATASET_FILE")/dataset_manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/python}"
RUN_BASE="${RUN_BASE:-$REPO_ROOT/runs}"
RUN_ROOT="$RUN_BASE/dpo_smoke_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"/{audit,config,logs,results,scheduler,checkpoints}

exec > >(tee "$RUN_ROOT/logs/dpo_smoke.out") 2> >(tee "$RUN_ROOT/logs/dpo_smoke.err" >&2)

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
  "$SFT_CHECKPOINT_DIR" --output "$RUN_ROOT/audit/sft_input_model.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/preflight/validate_dataset_asset.py" \
  "$DPO_DATASET_FILE" \
  --max-rows 16 \
  --binding-manifest "$DATASET_MANIFEST" \
  --binding-stage dpo \
  --output "$RUN_ROOT/audit/dpo_dataset.json"
cp "$DATASET_MANIFEST" "$RUN_ROOT/audit/dataset_manifest.json"
cp "$REPO_ROOT/configs/dpo/smoke_5090.yaml" "$RUN_ROOT/config/resolved_config.yaml"
cp "$REPO_ROOT/configs/dpo/reference_tulu3_8b.yaml" "$RUN_ROOT/config/reference_tulu3_8b.yaml"
cp "$REPO_ROOT/manifests/upstream.yaml" "$RUN_ROOT/audit/upstream.yaml"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/audit/project_commit.txt"

printf 'DPO_SMOKE_START=%s\nSFT_CHECKPOINT_DIR=%s\nDATASET_MANIFEST=%s\n' \
  "$(date -Is)" "$SFT_CHECKPOINT_DIR" "$DATASET_MANIFEST"
OPEN_INSTRUCT_ROOT="$OPEN_INSTRUCT_ROOT" \
SFT_CHECKPOINT_DIR="$SFT_CHECKPOINT_DIR" \
DATASET_REF="$DPO_DATASET_FILE" \
OUTPUT_DIR="$RUN_ROOT/checkpoints/smoke" \
bash "$REPO_ROOT/scripts/train/dpo_lecture6.sh"

echo "STATUS=PASS_DPO_SMOKE"
echo "RUN_ROOT=$RUN_ROOT"
