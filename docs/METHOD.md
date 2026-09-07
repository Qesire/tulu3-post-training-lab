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

Dataset identity is tutorial-defined. The exact lecture-era Tulu3 8B DPO script is frozen as upstream-reference authority. The project changes the lineage input to this project's SFT checkpoint. Qwen2.5-3B formal resource/hyperparameter choices remain pilot-gated.

### RLVR

The supplied tutorial identifies the RLVR stage and `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, but it does not contain a complete RL trainer command or exact algorithm choice.

That tutorial gap is now closed by the pinned lecture-era Open-Instruct source rather than by inference. At commit `8fcf9c6b...`, `scripts/train/tulu3/grpo_8b.sh` uses the exact tutorial RLVR dataset, starts from the Tulu3 DPO model, runs GRPO, enables verifiable reward, and fixes the reference values recorded in `configs/rlvr/reference_tulu3_8b_grpo.yaml`.

The canonical upstream reference includes, among other values:

```text
algorithm                GRPO
input stage              DPO
beta                     0.01
KL estimator             kl3
learning rate            5e-7
samples / prompt         16
temperature              1.0
max prompt length        2048
response length          2048
total episodes           2,000,000
DeepSpeed stage          2
seed                     1
verifiable reward        enabled
reward-model multiplier  0.0
```

This establishes the **algorithmic authority**, but not the final 1×RTX5090 training shape. The historical Tulu3 run is multi-node/multi-GPU and uses Llama-3.1-Tulu-3-8B-DPO.

For integration testing only, `configs/rlvr/smoke_5090.yaml` and `scripts/train/rlvr_lecture6.sh` adapt the same method to a bounded Qwen2.5-3B DPO checkpoint using the pinned `grpo_fast.py` single-GPU debug path. The smoke preserves GRPO/verifiable-reward/beta/KL/LR/temperature/seed semantics while shrinking rollout sizes and context, collocating actor and vLLM, and enabling CPU offload. It proves only execution feasibility. Formal Qwen2.5-3B rollout/update batching is frozen only after the resource pilot.

## Promotion rule

A stage may move from `smoke` to `formal` only when both are true:

1. its algorithmic/method authority is closed; and
2. target-hardware resource parameters have been measured and frozen without using the formal result to tune the acceptance rule.

For RLVR, condition 1 is now closed by pinned upstream authority; condition 2 remains open until the RTX5090 pilot completes.
