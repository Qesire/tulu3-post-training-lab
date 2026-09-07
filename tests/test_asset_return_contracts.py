from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text()


def test_asset_closure_covers_model_all_datasets_and_upstream():
    body = text("scripts/preflight/asset_closure.py")
    assert "validate_open_instruct_checkout.py" in body
    assert "validate_model_asset.py" in body
    assert '"sft": data_root / "sft.parquet"' in body
    assert '"dpo": data_root / "dpo.parquet"' in body
    assert '"rlvr": data_root / "rlvr.parquet"' in body
    assert '"status": "PASS_ASSET_CLOSURE"' in body


def test_return_packet_excludes_checkpoints_by_default():
    body = text("scripts/collect/pack_run.py")
    assert 'DEFAULT_DIRS = ("audit", "config", "logs", "results", "scheduler")' in body
    assert 'if args.include_checkpoints:' in body
    assert "RETURN_MANIFEST.json" in body
    assert "RETURN_PACKET_SHA256" in body
