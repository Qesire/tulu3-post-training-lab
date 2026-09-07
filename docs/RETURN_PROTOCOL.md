# Return protocol

The cluster phase should return compact evidence packets rather than raw run directories or model checkpoints.

## A0 asset closure

Before the first GPU smoke, run the CPU-only asset closure from the dedicated Open-Instruct environment:

```bash
cd /home/scc/pb23061276/projects/tulu3-post-training-lab
OPEN_INSTRUCT_ROOT=/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
MODEL_DIR=/home/scc/pb23061276/projects/wan-quant/models/LLM/Qwen2.5-3B
DATA_ROOT=/home/scc/pb23061276/projects/tulu3-data
OUT=/home/scc/pb23061276/projects/tulu3-asset-closure

$OPEN_INSTRUCT_ROOT/.venv/bin/python \
  scripts/preflight/asset_closure.py \
  --open-instruct-root "$OPEN_INSTRUCT_ROOT" \
  --model-dir "$MODEL_DIR" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT"
```

Required success marker:

```text
STATUS=PASS_ASSET_CLOSURE
```

The output directory contains:

```text
ASSET_CLOSURE.json
open_instruct_checkout.json
model_manifest.json
dataset_manifest.json
dataset_sft.json
dataset_dpo.json
dataset_rlvr.json
```

Do not submit SFT/DPO/RLVR training smoke if A0 fails.

## Packing one run

After any Slurm job finishes, package its run root:

```bash
OPEN_INSTRUCT_ROOT=/home/scc/pb23061276/projects/tulu3-upstream/open-instruct-lecture6
RUN_ROOT=/absolute/path/to/runs/<run_name>

$OPEN_INSTRUCT_ROOT/.venv/bin/python \
  scripts/collect/pack_run.py "$RUN_ROOT"
```

Default output:

```text
<run_name>_RETURN.tar.gz
```

The default packet includes only:

```text
audit/
config/
logs/
results/
scheduler/
RETURN_MANIFEST.json
```

`checkpoints/` is intentionally excluded. This keeps return packets small and prevents multi-GB model weights from being uploaded when the debugging task only requires execution evidence.

Every `RETURN_MANIFEST.json` records relative path, size and SHA256 for every included file. The packer also prints the SHA256 of the final tarball.

## When checkpoints are required

Only include model checkpoints for a task that explicitly needs weight inspection or transfer:

```bash
python scripts/collect/pack_run.py "$RUN_ROOT" --include-checkpoints
```

This may create a very large archive and should not be the default debugging workflow.

## Recommended return order

For the first cluster execution, return evidence in this order:

```text
1. A0 asset-closure directory or its JSON files
2. P0 runtime RETURN packet
3. E0 base-eval RETURN packet
4. S0 SFT-smoke RETURN packet
```

Only after S0 passes should the workflow progress to an SFT pilot/checkpoint, then DPO and RLVR.
