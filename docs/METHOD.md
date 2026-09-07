# Method contract

## Scope

The project follows the Lecture-6 Tulu3 post-training sequence:

`Qwen2.5-3B Base -> SFT -> DPO -> RLVR`.

## Evidence authority

Three levels are distinguished:

1. **Tutorial-defined**: explicitly specified by the supplied Lecture-6 document.
2. **Lecture-era upstream-reference**: extracted from pinned AllenAI revisions corresponding to the tutorial period.
3. **Qwen2.5-3B formal**: experimentally frozen for this project after compatibility/resource pilots.

Do not silently promote an upstream 8B parameter into a Qwen2.5-3B formal parameter.

## Upstream time binding

The tutorial explicitly contains a 2025-11-14 observation date and a 2025-12-08 OLMES update note. Accordingly this project pins a lecture-era Open-Instruct commit rather than 2026 `main`:

```text
open-instruct 8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377
olmes         f122a7bad99551ffb3f5feec14169af794926f6b
```

The pinned Open-Instruct source contains the three `wandb_tracker.run.get_url()` call sites described by the tutorial. The tutorial W&B fix is therefore represented explicitly and reproducibly by `scripts/setup/apply_lecture6_open_instruct_patch.py`.

## Current method status

### SFT

Tutorial-defined. The reference configuration is `configs/sft/reference_lecture6.yaml`. The 5090 smoke deliberately reduces data/steps/context only to test the integration path.

### DPO

Dataset identity is tutorial-defined. A canonical Tulu3 8B reference configuration is recorded, but Qwen2.5-3B adaptation remains open.

### RLVR

Dataset identity and stage purpose are tutorial-defined. The supplied tutorial does not specify the exact RL algorithm or training hyperparameters; these remain open until frozen from an authoritative implementation/source.
