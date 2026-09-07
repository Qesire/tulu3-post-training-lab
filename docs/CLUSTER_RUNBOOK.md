# Cluster runbook: P107 RTX5090

This runbook is the operational handoff from GitHub design to Slurm execution.

## 0. Fixed bindings

```text
project repo:       Qesire/tulu3-post-training-lab
partition:          P107-RTX5090
account:            competition
qos:                qos_p107-rtx5090
base model:         /home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B
Open-Instruct:      /home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
Open-Instruct SHA:  8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377
OLMES:              /home/scc/pb23061276/projects/tulu3-upstream/olmes-lecture6
OLMES SHA:           f122a7bad99551ffb3f5feec14169af794926f6b
dataset root:       /home/scc/pb23061276/projects/tulu3-data
```

Do not mutate the historical `hif4` environment for this project. The training and evaluation environments are separate dedicated Python 3.12 virtual environments.

## 1. Clone/update the orchestration repository

Choose a stable cluster path, for example:

```bash
cd /home/scc/pb23061276/projects
git clone https://github.com/Qesire/tulu3-post-training-lab.git
cd tulu3-post-training-lab
git checkout main
git pull --ff-only
export REPO_ROOT="$PWD"
```

If the compute environment cannot access GitHub, transfer the repository directory from a network-enabled machine instead. Record the resulting project commit with:

```bash
git rev-parse HEAD
```

## 2. Prepare pinned Open-Instruct

Run once on a network-enabled preparation node:

```bash
cd "$REPO_ROOT"
bash scripts/setup/bootstrap_open_instruct.sh
```

Expected checkout:

```text
/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
```

The bootstrap intentionally applies the recorded Lecture6 W&B compatibility patch to `open_instruct/finetune.py`. Therefore the upstream checkout is not generically clean. Correctness is proved by `validate_open_instruct_checkout.py`, which requires the exact pinned HEAD plus the exact deterministic patch and pristine backup, and rejects every other source deviation.

Manual attestation:

```bash
OPEN_INSTRUCT_ROOT=/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
$OPEN_INSTRUCT_ROOT/.venv/bin/python \
  scripts/preflight/validate_open_instruct_checkout.py \
  "$OPEN_INSTRUCT_ROOT" \
  --output /tmp/open_instruct_checkout.json
cat /tmp/open_instruct_checkout.json
```

## 3. Prepare OLMES

```bash
cd "$REPO_ROOT"
bash scripts/setup/bootstrap_olmes.sh
```

Training and evaluation environments remain separate.

## 4. Materialize the three datasets

Preferred path on a network-enabled preparation node:

```bash
OPEN_INSTRUCT_ROOT=/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
DATA_ROOT=/home/scc/pb23061276/projects/tulu3-data
mkdir -p "$DATA_ROOT"

$OPEN_INSTRUCT_ROOT/.venv/bin/python \
  scripts/data/download_tulu3_datasets.py \
  --root "$DATA_ROOT" \
  --stage all
```

Required output:

```text
tulu3-data/
  sft.parquet
  dpo.parquet
  rlvr.parquet
  dataset_manifest.json
```

`dataset_manifest.json` is authoritative. It records the resolved Hugging Face revision, row count, columns, file size and SHA256 for each materialized stage. Every training smoke verifies its Parquet file against this manifest before GPU-heavy work.

If the cluster cannot reach Hugging Face, run the same materializer on a compatible network-enabled machine, then upload all four files together without renaming them. Do not upload only the Parquet files without the manifest.

## 5. P0: runtime preflight

```bash
cd "$REPO_ROOT"
sbatch scripts/slurm/00_runtime_preflight.sbatch
```

The job must produce a run root similar to:

```text
runs/runtime_<jobid>/
  audit/runtime.json
  audit/model_manifest.json
  audit/open_instruct_checkout.json
  audit/upstream.yaml
```

Do not start training if P0 fails.

## 6. E0: Base evaluation smoke

After the OLMES environment is ready:

```bash
MODEL_DIR=/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B \
MODEL_STAGE=base \
  sbatch scripts/slurm/30_eval_smoke.sbatch
```

This is a bounded MMLU/GSM8K integration evaluation, not the final benchmark.

## 7. S0: SFT smoke

```bash
DATA_ROOT=/home/scc/pb23061276/projects/tulu3-data
SFT_DATASET_FILE="$DATA_ROOT/sft.parquet" \
DATASET_MANIFEST="$DATA_ROOT/dataset_manifest.json" \
  sbatch scripts/slurm/10_sft_smoke.sbatch
```

Expected claim boundary: two optimizer steps on a bounded subset prove only the SFT runtime path.

## 8. E1: evaluate SFT output

After locating a concrete SFT output directory containing `config.json` and weights:

```bash
MODEL_DIR=/absolute/path/to/sft/output \
MODEL_STAGE=sft \
  sbatch scripts/slurm/30_eval_smoke.sbatch
```

## 9. D0: DPO smoke

```bash
DATA_ROOT=/home/scc/pb23061276/projects/tulu3-data
SFT_CHECKPOINT_DIR=/absolute/path/to/sft/output \
DPO_DATASET_FILE="$DATA_ROOT/dpo.parquet" \
DATASET_MANIFEST="$DATA_ROOT/dataset_manifest.json" \
  sbatch scripts/slurm/20_dpo_smoke.sbatch
```

Do not substitute the base model for `SFT_CHECKPOINT_DIR`; the project lineage is Base → SFT → DPO.

## 10. E2: evaluate DPO output

```bash
MODEL_DIR=/absolute/path/to/dpo/output \
MODEL_STAGE=dpo \
  sbatch scripts/slurm/30_eval_smoke.sbatch
```

## 11. R0: RLVR/GRPO smoke

```bash
DATA_ROOT=/home/scc/pb23061276/projects/tulu3-data
DPO_CHECKPOINT_DIR=/absolute/path/to/dpo/output \
RLVR_DATASET_FILE="$DATA_ROOT/rlvr.parquet" \
DATASET_MANIFEST="$DATA_ROOT/dataset_manifest.json" \
  sbatch scripts/slurm/40_rlvr_smoke.sbatch
```

The one-5090 RLVR job is intentionally a bounded resource pilot. It is not the formal upstream 8B recipe. It must pass checkout attestation, DPO checkpoint validation, RLVR schema validation and manifest SHA256 binding before entering the GRPO path.

## 12. E3: evaluate RLVR output

```bash
MODEL_DIR=/absolute/path/to/rlvr/output \
MODEL_STAGE=rlvr \
  sbatch scripts/slurm/30_eval_smoke.sbatch
```

## 13. Promotion rule

Promotion is sequential:

```text
P0 runtime PASS
  -> E0 Base smoke
  -> S0 SFT smoke PASS
  -> SFT resource pilot
  -> E1
  -> D0 DPO smoke PASS
  -> DPO resource/method pilot
  -> E2
  -> R0 RLVR smoke PASS
  -> RLVR resource/method pilot
  -> E3
```

A smoke failure is an implementation/resource failure until diagnosed. A smoke pass does not authorize formal scientific claims or automatically freeze the full-training hyperparameters.
