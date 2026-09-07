from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_sft_reference_matches_lecture6_core_values():
    cfg = load("configs/sft/reference_lecture6.yaml")
    t = cfg["training"]
    assert cfg["dataset"]["hf_id"] == "allenai/tulu-3-sft-mixture"
    assert t["max_seq_length"] == 4096
    assert t["per_device_train_batch_size"] == 1
    assert t["gradient_accumulation_steps"] == 8
    assert t["learning_rate"] == 2e-5
    assert t["warmup_ratio"] == 0.03
    assert t["num_train_epochs"] == 2
    assert t["seed"] == 42


def test_dataset_ids_are_frozen():
    cfg = load("manifests/datasets.yaml")["datasets"]
    assert cfg["sft"]["hf_id"] == "allenai/tulu-3-sft-mixture"
    assert cfg["dpo"]["hf_id"] == "allenai/llama-3.1-tulu-3-8b-preference-mixture"
    assert cfg["rlvr"]["hf_id"] == "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"


def test_dpo_reference_matches_pinned_tulu3_script_but_is_not_formal_qwen():
    cfg = load("configs/dpo/reference_tulu3_8b.yaml")
    t = cfg["reference_training"]
    assert cfg["status"] == "reference_only_requires_qwen3b_formal_pilot"
    assert cfg["source"]["commit"] == "8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377"
    assert t["max_seq_length"] == 2048
    assert t["gradient_accumulation_steps"] == 16
    assert t["learning_rate"] == 5e-7
    assert t["dpo_loss_type"] == "dpo_norm"
    assert t["dpo_beta"] == 5
    assert t["seed"] == 8


def test_dpo_smoke_is_integration_only_and_lineage_bound():
    cfg = load("configs/dpo/smoke_5090.yaml")
    assert cfg["lineage"]["input_stage"] == "sft"
    assert cfg["claim_limit"] == "runtime_path_only"
    assert cfg["training"]["max_steps"] == 2
    assert cfg["training"]["concatenated_forward"] is False


def test_rlvr_method_is_fail_closed():
    cfg = load("configs/rlvr/reference_tutorial.yaml")
    assert cfg["status"] == "method_open"
    assert "algorithm" in cfg["method_open_fields"]
