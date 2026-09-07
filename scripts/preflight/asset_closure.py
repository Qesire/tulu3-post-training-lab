#!/usr/bin/env python3
"""Build a fail-closed A0 asset-closure packet before any training smoke.

This is intentionally CPU-only orchestration. It validates the pinned
Open-Instruct checkout, the base model asset, and all three materialized
Parquet datasets against the authoritative dataset_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    cp = subprocess.run(cmd, check=False)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)


def git_head(root: Path) -> str | None:
    cp = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip() if cp.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-instruct-root", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--expected-open-instruct-commit",
        default="8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377",
    )
    args = ap.parse_args()

    py = Path(sys.executable).resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.resolve()
    manifest = data_root / "dataset_manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"DATASET_MANIFEST_MISSING: {manifest}")

    checkout_json = out / "open_instruct_checkout.json"
    model_json = out / "model_manifest.json"
    run(
        [
            str(py),
            str(PROJECT_ROOT / "scripts/preflight/validate_open_instruct_checkout.py"),
            str(args.open_instruct_root.resolve()),
            "--expected-commit",
            args.expected_open_instruct_commit,
            "--output",
            str(checkout_json),
        ]
    )
    run(
        [
            str(py),
            str(PROJECT_ROOT / "scripts/preflight/validate_model_asset.py"),
            str(args.model_dir.resolve()),
            "--output",
            str(model_json),
        ]
    )

    stage_files = {
        "sft": data_root / "sft.parquet",
        "dpo": data_root / "dpo.parquet",
        "rlvr": data_root / "rlvr.parquet",
    }
    dataset_outputs: dict[str, str] = {}
    for stage, path in stage_files.items():
        stage_out = out / f"dataset_{stage}.json"
        cmd = [
            str(py),
            str(PROJECT_ROOT / "scripts/preflight/validate_dataset_asset.py"),
            str(path),
            "--binding-manifest",
            str(manifest),
            "--binding-stage",
            stage,
            "--output",
            str(stage_out),
        ]
        if stage == "rlvr":
            cmd.extend(
                [
                    "--require-columns",
                    "messages",
                    "ground_truth",
                    "dataset",
                    "constraint_type",
                    "constraint",
                ]
            )
        run(cmd)
        dataset_outputs[stage] = str(stage_out)

    manifest_copy = out / "dataset_manifest.json"
    manifest_copy.write_bytes(manifest.read_bytes())

    payload = {
        "schema_version": 1,
        "status": "PASS_ASSET_CLOSURE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "project_commit": git_head(PROJECT_ROOT),
        "python": str(py),
        "open_instruct_root": str(args.open_instruct_root.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "data_root": str(data_root),
        "dataset_manifest": {
            "path": str(manifest),
            "sha256": sha256(manifest),
        },
        "authority_files": {
            "open_instruct_checkout": str(checkout_json),
            "model_manifest": str(model_json),
            "datasets": dataset_outputs,
        },
    }
    closure = out / "ASSET_CLOSURE.json"
    closure.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"STATUS=PASS_ASSET_CLOSURE\nASSET_CLOSURE={closure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
