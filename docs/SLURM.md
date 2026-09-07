# Slurm deployment notes

Target defaults:

```text
partition = P107-RTX5090
account   = competition
qos       = qos_p107-rtx5090
```

Existing base-model asset:

```text
/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B
```

## Runtime separation

Do not mutate the historical `hif4` environment. It is evidence that Qwen2.5-3B + FlashAttention 2.8.3 work on RTX 5090, but the Lecture-6-era Open-Instruct pin requires Python 3.12 and Torch 2.8.x.

Prepare the pinned environment with:

```bash
bash scripts/setup/bootstrap_open_instruct.sh
```

Default upstream checkout:

```text
/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
```

## Dataset closure

Formal GPU jobs must not rely on Hugging Face network access. Materialize datasets first as local Parquet files and freeze resolved HF revisions + SHA256:

```bash
OPEN_INSTRUCT_ROOT=/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
$OPEN_INSTRUCT_ROOT/.venv/bin/python scripts/data/download_tulu3_datasets.py \
  --root /home/scc/pb23061276/projects/tulu3-data --stage all
```

This produces:

```text
/home/scc/pb23061276/projects/tulu3-data/
  sft.parquet
  dpo.parquet
  rlvr.parquet
  dataset_manifest.json
```

The local Parquet format is intentional because the pinned Open-Instruct local dataset loader directly recognizes `.parquet` and `.jsonl` files.

## Execution order

```bash
sbatch scripts/slurm/00_runtime_preflight.sbatch

SFT_DATASET_FILE=/home/scc/pb23061276/projects/tulu3-data/sft.parquet \
  sbatch scripts/slurm/10_sft_smoke.sbatch
```

The smoke runs 32 examples for 2 optimizer steps, sequence length 1024, and does not constitute a scientific result.
