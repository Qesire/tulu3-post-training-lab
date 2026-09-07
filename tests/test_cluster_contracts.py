from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text()


def test_training_smokes_attest_pinned_open_instruct_checkout():
    for path in [
        "scripts/slurm/10_sft_smoke.sbatch",
        "scripts/slurm/20_dpo_smoke.sbatch",
        "scripts/slurm/40_rlvr_smoke.sbatch",
    ]:
        body = text(path)
        assert "validate_open_instruct_checkout.py" in body
        assert "8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377" in body


def test_training_smokes_bind_datasets_to_materialization_manifest():
    expected = {
        "scripts/slurm/10_sft_smoke.sbatch": "sft",
        "scripts/slurm/20_dpo_smoke.sbatch": "dpo",
        "scripts/slurm/40_rlvr_smoke.sbatch": "rlvr",
    }
    for path, stage in expected.items():
        body = text(path)
        assert "DATASET_MANIFEST=" in body
        assert '--binding-manifest "$DATASET_MANIFEST"' in body
        assert f"--binding-stage {stage}" in body
        assert 'cp "$DATASET_MANIFEST" "$RUN_ROOT/audit/dataset_manifest.json"' in body


def test_runtime_preflight_uses_dedicated_posttrain_python():
    body = text("scripts/slurm/00_runtime_preflight.sbatch")
    assert 'PYTHON_BIN="${PYTHON_BIN:-$OPEN_INSTRUCT_ROOT/.venv/bin/python}"' in body
    assert "validate_open_instruct_checkout.py" in body


def test_cluster_runbook_preserves_model_lineage():
    body = text("docs/CLUSTER_RUNBOOK.md")
    assert "Base → SFT → DPO" in body
    assert "DPO_CHECKPOINT_DIR" in body
    assert "RLVR_DATASET_FILE" in body
    assert "dataset_manifest.json" in body
