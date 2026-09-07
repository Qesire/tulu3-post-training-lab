# AI 大模型编程培训及兴趣小组 Lecture6：大模型 SFT 案例及 Tulu3 框架教程

> GitHub 工程化复现：Qwen2.5-3B Base → SFT → DPO → RLVR；训练目标环境为 USTC Slurm `P107-RTX5090`。

本仓库以课程参考作业 **“AI 大模型编程培训及兴趣小组 Lecture6：大模型 SFT 案例及智星云安装部署 Tulu3 框架教程”** 为题目与方法主线，保留教程中的 Tulu3 / Open-Instruct / OLMES 结构，同时将运行环境改造成可审计、可复现的 Slurm 实验工程。

## 1. 实验主线

```text
Qwen2.5-3B Base
      │
      ├── E0: Base benchmark
      ▼
SFT (Tulu-3 SFT mixture)
      │
      ├── E1: SFT benchmark
      ▼
DPO (Tulu-3 preference mixture)
      │
      ├── E2: DPO benchmark
      ▼
RLVR (RLVR GSM/MATH/IF mixed constraints)
      │
      └── E3: RLVR benchmark
```

当前目标不是立即跑满训练，而是先在 GitHub 冻结：

1. upstream 版本；
2. 模型与数据 binding；
3. SFT/DPO/RLVR 配置；
4. Slurm 5090 launcher；
5. smoke → pilot → formal 的实验层级；
6. 训练与评测产物契约。

## 2. 教程基线

### Base model

- Hugging Face ID: `Qwen/Qwen2.5-3B`
- 集群已有本地资产：`/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B`

### Datasets

| Stage | Dataset | Tutorial size |
|---|---|---:|
| SFT | `allenai/tulu-3-sft-mixture` | ~939k |
| DPO | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | ~273k |
| RLVR | `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` | ~30k |

### Lecture-6 SFT hyperparameters

| Parameter | Value |
|---|---:|
| precision | bf16 |
| max sequence length | 4096 |
| per-device batch size | 1 |
| gradient accumulation | 8 |
| learning rate | 2e-5 |
| scheduler | linear |
| warmup ratio | 0.03 |
| weight decay | 0 |
| epochs | 2 |
| gradient checkpointing | enabled |
| seed | 42 |

The tutorial runs 4 processes, so its nominal effective batch size is `4 × 1 × 8 = 32`.

## 3. Slurm target

Known cluster target:

```text
partition: P107-RTX5090
account:   competition
qos:       qos_p107-rtx5090
GPU:       NVIDIA GeForce RTX 5090 (~32 GB)
```

The existing cluster already contains Qwen2.5-3B and has previously run FlashAttention on RTX 5090. This repository therefore treats the model asset as reusable and performs runtime attestation before training instead of redownloading model weights inside GPU jobs.

## 4. Upstream policy

This repository is an **experiment/orchestration layer**, not a fork of AllenAI Open-Instruct.

Pinned upstream revisions are recorded in `manifests/upstream.yaml`. To follow the tutorial rather than silently track 2026 `main`, Open-Instruct is pinned to commit `8fcf9c6b...`, the latest upstream commit on the tutorial's stated 2025-11-14 reference date. That revision requires Python 3.12, Torch 2.8.x, Transformers >=4.57 and FlashAttention >=2.8.3.

The tutorial's three-call W&B compatibility fix is implemented as an explicit, fail-closed patch under `scripts/setup/apply_lecture6_open_instruct_patch.py`; upstream source is never silently edited. The already-validated `hif4` environment remains untouched.

## 5. Repository structure

```text
configs/        frozen experiment configuration
manifests/      model/dataset/upstream/runtime bindings
scripts/        preflight, data, train, eval, Slurm launchers
patches/        explicit upstream patches only
tests/          CPU/static correctness checks
docs/           method and experiment contracts
```

## 6. Experiment states

Every stage uses:

```text
smoke  -> proves the code path works
pilot  -> proves resource/memory/runtime feasibility
formal -> executes the frozen scientific configuration
```

Smoke results must not be interpreted as scientific results.

## 7. Status

- [x] Project skeleton
- [x] Tutorial SFT reference config
- [x] Qwen2.5-3B cluster binding
- [x] Tulu3 dataset identifiers
- [x] P107 RTX5090 Slurm defaults
- [x] Runtime probe
- [x] Dataset download/materialization utility
- [x] Static CI
- [ ] Freeze local dataset revisions/digests
- [x] Lecture-era Open-Instruct/OLMES revision policy
- [x] Dedicated Open-Instruct bootstrap + explicit Lecture-6 W&B patch
- [x] Offline-compatible local Parquet dataset binding
- [x] Executable 2-step SFT smoke launcher
- [ ] Materialize/freeze dataset revisions and SHA256 on cluster
- [ ] Validate dedicated Python 3.12/Open-Instruct environment on 5090
- [ ] Run SFT smoke
- [ ] Adapt/freeze DPO for Qwen2.5-3B lineage
- [ ] Freeze canonical RLVR algorithm/config
- [ ] Run formal stage evaluations

## 8. First cluster command

After cloning the repository on the cluster, the first GPU action should be runtime preflight, not full SFT:

```bash
sbatch scripts/slurm/00_runtime_preflight.sbatch
```

Then inspect the generated `runtime.json` before enabling training.

## 9. Preparation sequence

On a network-enabled preparation/login node:

```bash
bash scripts/setup/bootstrap_open_instruct.sh
OPEN_INSTRUCT_ROOT=/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
$OPEN_INSTRUCT_ROOT/.venv/bin/python scripts/data/download_tulu3_datasets.py \
  --root /home/scc/pb23061276/projects/tulu3-data --stage all
```

The materializer writes `sft.parquet`, `dpo.parquet`, `rlvr.parquet` plus a manifest containing the resolved Hugging Face revision and SHA256. The use of Parquet is intentional: the pinned Lecture-6 Open-Instruct directly recognizes local `.parquet`/`.jsonl` dataset paths.

Then submit runtime preflight and, only after it passes, the SFT smoke:

```bash
sbatch scripts/slurm/00_runtime_preflight.sbatch
SFT_DATASET_FILE=/home/scc/pb23061276/projects/tulu3-data/sft.parquet \
  sbatch scripts/slurm/10_sft_smoke.sbatch
```
