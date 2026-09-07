# Experiment protocol

## DAG

1. `P0` runtime preflight
2. `A0` asset closure
3. `E0` base evaluation
4. `S0` SFT smoke
5. `S1` SFT pilot
6. `S2` SFT formal
7. `E1` SFT evaluation
8. `D0` DPO smoke
9. `D1` DPO pilot/formal
10. `E2` DPO evaluation
11. `R0` RLVR smoke
12. `R1` RLVR pilot/formal
13. `E3` RLVR evaluation

## Run-root contract

Every Slurm run should create a unique run root with at least:

```text
RUN_ROOT/
  audit/
  config/
  logs/
  results/
  scheduler/
  checkpoints/
```

Recommended authority files:

- `runtime.json`
- `resolved_config.yaml`
- `model_manifest.json`
- `dataset_manifest.json`
- `train_summary.json`
- `metrics.jsonl`

## Fail-closed policy

A formal job must fail before GPU-heavy work if any required model path, dataset path, upstream revision, or method parameter is unresolved.
